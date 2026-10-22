"""Behavioral integration checks for bounded reference search and suitability."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

from mathagent.agent import Agent
from mathagent.literature import LiteratureTools
from mathagent.refcheck import CheckRunner
from mathagent.tools import Workspace
from tests.test_research import FakeClient, response


RBM = {'identifier': '10.1214/aop/1176990437',
       'title': 'Some Potential Theory for Reflecting Brownian Motion in Holder and Lipschitz Domains',
       'authors': ['Richard F. Bass', 'Pei Hsu'], 'year': 1991, 'type': 'journal-article',
       'abstract': 'Constructs normally reflecting Brownian motion and boundary local time in Lipschitz domains.'}
SMOOTH = {'identifier': '10.0000/fixture-smooth',
          'title': 'Synchronous couplings of reflected Brownian motions in smooth domains',
          'authors': ['Krzysztof Burdzy', 'Zhen-Qing Chen', 'Peter Jones'], 'year': 2006,
          'type': 'journal-article', 'abstract': 'Studies synchronous coupling on bounded planar domains with smooth boundary.'}
LIP = {'identifier': '10.0000/fixture-lip',
       'title': 'Uniqueness for reflecting Brownian motion in lip domains',
       'authors': ['Richard F. Bass', 'Krzysztof Burdzy', 'Zhen-Qing Chen'], 'year': 2005,
       'type': 'journal-article', 'abstract': 'Strong existence and pathwise uniqueness in planar lip domains with Lipschitz constant strictly below one.'}
SOBOLEV = {'identifier': '10.0000/fixture-sobolev', 'title': 'Sobolev Spaces',
           'authors': ['Robert A. Adams', 'John J. F. Fournier'], 'year': 2003, 'type': 'book',
           'abstract': 'Sobolev embeddings and compact embeddings on bounded Lipschitz domains. Does not treat the heat semigroup or compact resolvents.'}
HEAT = {'identifier': '10.0000/fixture-heat', 'title': 'Heat Kernels and Spectral Theory',
        'authors': ['E. B. Davies'], 'year': 1989, 'type': 'book',
        'abstract': 'Heat semigroups and compact resolvents on bounded domains. Complements a separate Sobolev embeddings text.'}


def recalled(record, **changes):
    result = {'authors': [name.split()[-1] for name in record['authors']], 'title': record['title'],
              'year': str(record.get('year') or ''), 'locator': '', 'why': 'Plausible source to look up.'}
    result.update(changes)
    return result


def recall(candidates=(), statement='Normally reflecting Brownian motion on bounded Lipschitz domains in every dimension.',
           keywords=('reflecting Brownian motion', 'Lipschitz domains')):
    return {'statement': statement, 'keywords': list(keywords), 'candidates': list(candidates)}


def selection(ids, coverage='supported', unresolved=''):
    return {'selections': [{'id': source_id, 'reason': 'The indexed description addresses the requested scope.',
                            'coverage': coverage, 'limitations': 'The precise theorem and its full hypotheses were not read.'}
                           for source_id in ids], 'unresolved': unresolved}


class ExplicitSelectorClient(FakeClient):
    def __init__(self, recall_value, selector):
        super().__init__([response(json.dumps(recall_value))])
        self.selector = selector
        self.selection_prompts = []
        self.pools = []

    def stream(self, payload):
        prompt = '\n'.join(message.get('content', '') for message in payload['messages'])
        marker = 'ACTUAL BIBLIOGRAPHIC RECORDS (select only R identifiers):\n'
        if marker not in prompt:
            yield from super().stream(payload)
            return
        self.requests.append(copy.deepcopy(payload))
        pool = json.loads(prompt.split(marker, 1)[1].split('\n', 1)[0])
        self.pools.append(pool)
        self.selection_prompts.append(prompt)
        result = self.selector(pool, prompt, len(self.pools))
        if isinstance(result, BaseException):
            raise result
        yield from response(json.dumps(result))


class ReferencePipelineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        sleeper = patch('mathagent.literature.time.sleep')
        sleeper.start()
        self.addCleanup(sleeper.stop)
        self.http_calls = []

    def runner(self, client):
        literature = LiteratureTools(self.root, online=True)
        return CheckRunner(Agent(client, Workspace(self.root, literature=literature), ctx=32768, predict=4096))

    def http(self, handler):
        def request(url, headers, max_bytes):
            parsed = urlsplit(url)
            params = parse_qs(parsed.query)
            provider = ('zbmath' if 'zbmath' in parsed.netloc else 'crossref' if 'crossref' in parsed.netloc
                        else 'openlibrary' if 'openlibrary' in parsed.netloc else 'arxiv')
            query = (params.get('search_string') or params.get('query.bibliographic') or params.get('q')
                     or params.get('search_query') or [''])[0]
            self.http_calls.append((provider, query))
            value = handler(provider, query)
            if isinstance(value, int):
                return value, {}, b'provider unavailable'
            records = value or []
            if provider == 'crossref':
                body = {'message': {'items': [
                    {'DOI': record['identifier'], 'title': [record['title']],
                     'author': [{'given': name.rsplit(' ', 1)[0], 'family': name.rsplit(' ', 1)[-1]}
                                for name in record['authors']],
                     'issued': {'date-parts': [[record['year']]]} if record.get('year') else {},
                     'type': record['type'], 'abstract': record.get('abstract', '')} for record in records]}}
            elif provider == 'zbmath':
                body = {'result': [
                    {'identifier': record['identifier'], 'title': {'title': record['title']},
                     'contributors': {'authors': [{'name': name} for name in record['authors']]},
                     'year': record.get('year'), 'document_type': {'description': record['type']},
                     'zbmath_url': 'https://zbmath.org/fixture-' + str(i),
                     'editorial_contributions': [{'text': record.get('abstract', '')}]} for i, record in enumerate(records)]}
            elif provider == 'openlibrary':
                body = {'docs': [
                    {'key': '/works/OL' + str(i + 1000) + 'W', 'title': record['title'],
                     'author_name': record['authors'], 'first_publish_year': record.get('year')}
                    for i, record in enumerate(records)]}
            else:
                entries = []
                for i, record in enumerate(records):
                    entries.append('<entry><id>https://arxiv.org/abs/2412.' + str(10000 + i) + 'v1</id>'
                        '<title>' + escape(record['title']) + '</title>'
                        '<summary>' + escape(record.get('abstract', '')) + '</summary>' +
                        ''.join('<author><name>' + escape(name) + '</name></author>' for name in record['authors']) +
                        '<published>' + str(record.get('year') or 2024) + '-01-01T00:00:00Z</published></entry>')
                body = '<feed xmlns="http://www.w3.org/2005/Atom">' + ''.join(entries) + '</feed>'
                return 200, {'content-type': 'application/atom+xml'}, body.encode()
            return 200, {'content-type': 'application/json'}, json.dumps(body).encode()
        return request

    def run_search(self, question, recall_value, selector, handler, reference_context=None):
        client = ExplicitSelectorClient(recall_value, selector)
        runner = self.runner(client)
        with patch.object(runner.literature, '_request_once', side_effect=self.http(handler)):
            result = runner.start(question, lookup='standard', reference_context=reference_context)
        return runner, result, client

    def references(self, runner):
        return json.loads(runner.result_for_agent())['references']

    def test_provider_outage_fails_over_and_returns_actual_article(self):
        runner, result, _ = self.run_search(
            'Give me an article for normally reflected Brownian motion on Lipschitz domains.',
            recall([recalled(RBM)]), lambda pool, *_: selection([pool[0]['id']]),
            lambda provider, query: 502 if provider == 'zbmath' else [RBM] if provider == 'crossref' else [])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual([provider for provider, _ in self.http_calls], ['zbmath', 'crossref'])
        refs = self.references(runner)
        self.assertEqual([item['record'] for item in refs], ['doi:' + RBM['identifier']])
        self.assertTrue(refs[0]['metadata_confirmed'])
        self.assertFalse(refs[0]['exact_statement_confirmed'])
        self.assertIn('zbmath', runner.state['standard_provider_errors'])
        self.assertNotIn('Wikipedia', runner.state['answer'])

    def test_fake_record_id_cannot_create_a_reference_or_inject_bibliographic_metadata(self):
        def selector(pool, *_):
            return {'selections': [{'id': 'R404', 'reason': 'Phantom title 2099', 'title': 'Phantom title', 'year': '2099'},
                                   {'id': pool[0]['id'], 'coverage': 'supported', 'reason': 'Relevant original paper.'}],
                    'unresolved': ''}
        runner, result, _ = self.run_search('Give me an article for reflecting Brownian motion on Lipschitz domains.',
                                           recall([recalled(RBM)]), selector, lambda *_: [RBM])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(self.references(runner)), 1)
        self.assertNotIn('Phantom', runner.state['answer'])
        self.assertNotIn('2099', runner.state['answer'])

    def test_malformed_selection_ids_are_rejected_without_crashing_or_inventing_sources(self):
        def selector(pool, *_):
            return {'selections': [{'id': ['R1'], 'reason': 'malformed identifier'},
                                   {'id': pool[0]['id'], 'coverage': 'supported'}], 'unresolved': ''}
        runner, result, _ = self.run_search('Give me an article for reflecting Brownian motion on Lipschitz domains.',
                                           recall([recalled(RBM)]), selector, lambda *_: [RBM])
        self.assertEqual(result['status'], 'answered', runner.state.get('stop_reason'))
        self.assertEqual(len(self.references(runner)), 1)
        self.assertEqual(self.references(runner)[0]['record'], 'zbl:' + RBM['identifier'])

    def test_topic_discovery_recovers_when_recalled_authors_are_wrong(self):
        guess = recalled(RBM, authors=['Burdzy', 'Whitmart'], title='Reflected Brownian motion in Lipschitz domains', year='1997')
        def provider(provider, query):
            return [RBM] if provider == 'crossref' and 'Burdzy' not in query and 'Whitmart' not in query else []
        runner, result, client = self.run_search(
            'Find an article on normally reflecting Brownian motion on Lipschitz domains.', recall([guess]),
            lambda pool, *_: selection([item['id'] for item in pool if 'Hsu' in ' '.join(item['record']['authors'])]), provider)
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(any(provider == 'crossref' and 'Whitmart' not in query and 'Burdzy' not in query
                            for provider, query in self.http_calls))
        self.assertEqual(self.references(runner)[0]['record'], 'doi:' + RBM['identifier'])
        self.assertEqual(len(client.pools), 1)
        self.assertNotIn('Whitmart', runner.state['answer'])
        self.assertNotIn('1997', runner.state['answer'])

    def test_rejected_smooth_or_planar_lip_reference_triggers_independent_topic_discovery(self):
        for narrow in (SMOOTH, LIP):
            with self.subTest(title=narrow['title']):
                self.http_calls.clear()
                def selector(pool, *_):
                    suitable = [item['id'] for item in pool if item['record']['title'] == RBM['title']]
                    return selection(suitable, unresolved='' if suitable else 'Only restricted smooth or planar lip scope was found.')
                runner, result, client = self.run_search(
                    'Find an article for normal reflecting Brownian motion in arbitrary bounded Lipschitz domains in every dimension.',
                    recall([recalled(narrow)]), selector,
                    lambda provider, query: [narrow] if provider == 'zbmath' else [RBM] if provider == 'crossref' else [])
                self.assertEqual(result['status'], 'answered')
                self.assertGreaterEqual(len(client.pools), 2)
                self.assertIn(narrow['title'], [item['record']['title'] for item in client.pools[0]])
                self.assertEqual(len(self.references(runner)), 1)
                self.assertEqual(self.references(runner)[0]['record'], 'doi:' + RBM['identifier'])
                self.assertIn(RBM['title'], runner.state['answer'])
                self.assertNotIn(narrow['title'], runner.state['answer'])

    def test_selector_rejection_continues_to_remaining_provider_with_relevant_article(self):
        def selector(pool, *_):
            suitable = [item['id'] for item in pool if item['record']['title'] == RBM['title']]
            return selection(suitable, unresolved='' if suitable else
                             'The records only treat smooth or restricted planar lip domains.')

        def provider(provider, query):
            return [SMOOTH] if provider == 'crossref' else [LIP] if provider == 'arxiv' else [RBM] if provider == 'zbmath' else []

        runner, result, client = self.run_search(
            'Find an article for normal reflecting Brownian motion in bounded Lipschitz domains in arbitrary dimension.',
            recall(), selector, provider)
        self.assertEqual(result['status'], 'answered', runner.state.get('stop_reason'))
        self.assertEqual([provider for provider, _ in self.http_calls], ['crossref', 'crossref', 'arxiv', 'zbmath'])
        self.assertNotEqual(self.http_calls[0][1], self.http_calls[1][1],
                            'The second Crossref request must use a distinct broader retrieval query.')
        self.assertEqual(len(client.pools), 2, 'One bounded retry should use the remaining provider after relevance rejection.')
        self.assertEqual([item['record'] for item in self.references(runner)], ['zbl:' + RBM['identifier']])
        self.assertNotIn(SMOOTH['title'], runner.state['answer'])
        self.assertNotIn(LIP['title'], runner.state['answer'])

    def test_model_terminology_alternative_is_searched_without_weakening_selector_scope(self):
        original = ('Find an article on normally reflected Brownian motion on bounded Lipschitz domains '
                    'in arbitrary dimension, without smoothness or a small Lipschitz constant.')
        first_query = 'reflected Brownian motion Lipschitz domain'
        alternate_query = 'reflecting Brownian motion Lipschitz domains'
        recalled_value = recall([], statement='Smooth planar lip domains with a small Lipschitz constant.',
                                keywords=('Brownian motion', 'domains'))
        recalled_value['search_queries'] = [first_query, alternate_query]

        def selector(pool, prompt, call):
            self.assertIn(original, prompt, 'Query relaxation must not replace the exact human scope.')
            return selection([item['id'] for item in pool if item['record']['title'] == RBM['title']])

        runner, result, client = self.run_search(
            original, recalled_value, selector,
            lambda provider, query: [RBM] if provider == 'crossref' and query == alternate_query else [])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(self.http_calls[0], ('crossref', first_query))
        self.assertIn(('crossref', alternate_query), self.http_calls)
        self.assertEqual(runner.state['goal'], original)
        self.assertIn(original, client.selection_prompts[-1])
        self.assertEqual([item['record'] for item in self.references(runner)], ['doi:' + RBM['identifier']])
        self.assertLessEqual(len(self.http_calls), runner.state['settings']['max_requests'])

    def test_pruned_full_pool_does_not_refill_with_rejected_evidence_before_late_query(self):
        original = ('Find an article for normally reflected Brownian motion on bounded Lipschitz domains '
                    'in arbitrary dimension, without assuming smooth boundary.')
        first_query = 'reflected Brownian motion Lipschitz domain'
        alternate_query = 'reflected Brownian motion normal Lipschitz domain'
        relaxed_query = 'Brownian motion Lipschitz domains'
        nearby = [{**SMOOTH, 'identifier': '10.0000/fixture-nearby-' + str(i),
                   'title': 'Nearby smooth reflected Brownian motion problem ' + str(i),
                   'abstract': 'Treats a restricted smooth planar setting only.'} for i in range(12)]
        recalled_value = recall()
        recalled_value['search_queries'] = [first_query, alternate_query]
        rejected_ids = set()

        def selector(pool, prompt, call):
            self.assertIn(original, prompt)
            if call == 1:
                self.assertEqual(len(pool), 12)
                rejected_ids.update(item['id'] for item in pool[1:])
                return selection([pool[0]['id']], coverage='partial',
                                 unresolved='The Lipschitz domain setting remains uncovered.')
            self.assertTrue(rejected_ids.isdisjoint({item['id'] for item in pool}),
                            'Previously rejected records must not refill the pruned bounded pool.')
            suitable = [item['id'] for item in pool if item['record']['title'] == RBM['title']]
            self.assertEqual(len(suitable), 1, 'The later relevant article must have room in the pool.')
            return selection(suitable)

        def provider(provider, query):
            if provider != 'crossref':
                return []
            return nearby[:8] if query == first_query else nearby[8:] if query == alternate_query else [RBM] if query == relaxed_query else []

        runner, result, client = self.run_search(original, recalled_value, selector, provider)
        self.assertEqual(result['status'], 'answered', runner.state.get('stop_reason'))
        self.assertIn(('crossref', relaxed_query), self.http_calls)
        self.assertEqual(len(client.pools), 2)
        self.assertEqual([item['record'] for item in self.references(runner)], ['doi:' + RBM['identifier']])
        self.assertEqual(runner.state['goal'], original)
        self.assertLessEqual(len(self.http_calls), runner.state['settings']['max_requests'])

    def test_two_complementary_verified_sources_cover_a_compound_reference_request(self):
        def selector(pool, *_):
            ids = [item['id'] for item in pool if item['record']['title'] in (SOBOLEV['title'], HEAT['title'])]
            if len(ids) < 2:
                return selection([], unresolved='A complementary heat semigroup and compact resolvent source is still needed.')
            return selection(ids)
        runner, result, _ = self.run_search(
            'Give me references for Sobolev embeddings on bounded Lipschitz domains and applications to heat semigroups and compact resolvents.',
            recall([], statement='Sobolev embeddings and heat semigroup/compact resolvent theory on bounded Lipschitz domains.',
                   keywords=('Sobolev embeddings', 'heat semigroups compact resolvents')),
            selector, lambda provider, query: [SOBOLEV, HEAT] if provider == 'crossref' else [])
        self.assertEqual(result['status'], 'answered')
        refs = self.references(runner)
        self.assertEqual({item['record'] for item in refs}, {'doi:' + SOBOLEV['identifier'], 'doi:' + HEAT['identifier']})
        self.assertIn(SOBOLEV['title'], runner.state['answer'])
        self.assertIn(HEAT['title'], runner.state['answer'])
        self.assertTrue(all(item['metadata_confirmed'] and not item['exact_statement_confirmed'] for item in refs))

    def test_partial_first_candidate_does_not_end_a_compound_request_with_missing_coverage(self):
        def selector(pool, *_):
            ids = [item['id'] for item in pool]
            covers_heat = any(item['record']['title'] == HEAT['title'] for item in pool)
            return selection(ids[:2], coverage='supported' if covers_heat else 'partial',
                             unresolved='' if covers_heat else 'Heat semigroup and compact resolvent coverage remains missing.')
        runner, result, _ = self.run_search(
            'Give me references covering Sobolev embeddings on bounded Lipschitz domains and heat semigroups with compact resolvents.',
            recall([recalled(SOBOLEV), recalled(HEAT)], keywords=('Sobolev embeddings', 'heat semigroups compact resolvents')),
            selector,
            lambda provider, query: [SOBOLEV] if provider == 'zbmath' and 'Adams' in query else
                                    [HEAT] if provider == 'zbmath' and 'Davies' in query else
                                    [SOBOLEV, HEAT] if provider == 'crossref' else [])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(self.references(runner)), 2,
                         'A single explicitly partial source must not stop retrieval while a complementary candidate remains.')
        self.assertIn(HEAT['title'], runner.state['answer'])

    def test_recalled_book_type_rejects_identically_titled_journal_review(self):
        book = {'identifier': '10.0000/fixture-durrett-book', 'title': 'Probability: Theory and Examples',
                'authors': ['Richard Durrett'], 'year': 1991, 'type': 'book',
                'abstract': 'Probability textbook with stochastic processes.'}
        review = {**book, 'identifier': '10.0000/fixture-durrett-review', 'type': 'journal-article',
                  'authors': ['John Reviewer', 'Richard Durrett'],
                  'abstract': 'A journal review of the book, not the textbook itself.'}
        runner, result, client = self.run_search(
            'Give me a classical reference on stochastic processes.',
            recall([recalled(book, source_type='book')], keywords=('stochastic processes', 'probability')),
            lambda pool, *_: selection([pool[0]['id']]),
            lambda provider, query: [review] if provider == 'crossref' else [book] if provider == 'openlibrary' else [])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual([provider for provider, _ in self.http_calls], ['zbmath', 'crossref', 'openlibrary'])
        self.assertEqual(self.references(runner)[0]['record'], 'openlibrary:OL1000W')
        self.assertNotIn('Reviewer', runner.state['answer'])
        self.assertTrue(all(item['record']['type'] == 'book' for item in client.pools[0]))

    def test_book_request_rejects_matching_book_section_record(self):
        section = {**SOBOLEV, 'identifier': '10.0000/fixture-sobolev-section', 'type': 'book-section'}
        runner, result, client = self.run_search(
            'Give me a book for Sobolev embeddings on bounded Lipschitz domains.',
            recall([recalled(SOBOLEV)], keywords=('Sobolev embeddings', 'bounded Lipschitz domains')),
            lambda pool, *_: selection([pool[0]['id']]),
            lambda provider, query: [section] if provider == 'zbmath' else [SOBOLEV] if provider == 'crossref' else [])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(self.references(runner)[0]['record'], 'doi:' + SOBOLEV['identifier'])
        self.assertTrue(all(item['record']['type'] == 'book' for item in client.pools[0]))

    def test_missing_record_year_is_not_filled_by_recall_or_selector(self):
        no_year = {**RBM, 'year': None}
        def selector(pool, *_):
            out = selection([pool[0]['id']])
            out['selections'][0].update(year='2099', title='A hallucinated title')
            return out
        runner, result, _ = self.run_search('Give me an article on reflecting Brownian motion on Lipschitz domains.',
                                           recall([recalled(RBM, year='1997')]), selector, lambda *_: [no_year])
        self.assertEqual(result['status'], 'answered')
        self.assertNotIn('1997', runner.state['answer'])
        self.assertNotIn('2099', runner.state['answer'])
        self.assertNotIn('hallucinated title', runner.state['answer'])
        self.assertFalse(any(year in self.references(runner)[0]['reference'] for year in ('1997', '2099')))

    def test_exact_reference_context_hypotheses_reach_relevance_selector(self):
        exact = 'Let Ω be a bounded Lipschitz domain in R^d, d arbitrary; reflection is normal, not oblique. No smoothness or small Lipschitz constant is assumed.'
        messages = [{'role': 'user', 'content': exact}, {'role': 'user', 'content': 'Give me a suitable article.'}]
        context = {'query': 'Give me a suitable article.', 'files': [], 'history': messages[:1], 'messages': messages}
        runner, result, client = self.run_search('Give me a suitable article.', recall([recalled(RBM)]),
                                               lambda pool, *_: selection([pool[0]['id']]), lambda *_: [RBM],
                                               reference_context=context)
        self.assertEqual(result['status'], 'answered')
        self.assertIn(exact, client.selection_prompts[0])
        self.assertEqual(runner.state['reference_context'], context)
        self.assertTrue(all(exact not in query for _, query in self.http_calls),
                        'Private source hypotheses are selector context, not a verbatim public search query.')

    def test_resume_after_selection_pause_preserves_evidence_without_repeating_http(self):
        client = ExplicitSelectorClient(recall([recalled(RBM)]), lambda *_: KeyboardInterrupt())
        runner = self.runner(client)
        with patch.object(runner.literature, '_request_once', side_effect=self.http(lambda *_: [RBM])):
            paused = runner.start('Give me an article for normal reflecting Brownian motion on Lipschitz domains.', lookup='standard')
        self.assertEqual(paused['status'], 'paused')
        self.assertEqual(len(self.http_calls), 1)
        evidence = copy.deepcopy(runner.state['evidence'])
        spent = runner.literature.stats['requests']
        resumed_client = ExplicitSelectorClient(recall(), lambda pool, *_: selection([pool[0]['id']]))
        resumed = self.runner(resumed_client)
        with patch.object(resumed.literature, '_request_once') as request:
            result = resumed.resume(paused['id'])
            request.assert_not_called()
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(resumed.state['evidence'], evidence)
        self.assertEqual(resumed.literature.stats['requests'], spent)
        self.assertEqual(len(self.references(resumed)), 1)


if __name__ == '__main__':
    unittest.main()
