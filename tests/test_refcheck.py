"""Literature check: recall, per-guess verification and the controller's evidence levels."""
import copy
import json
from pathlib import Path
import sys
from urllib.parse import unquote
import tempfile
import unittest
from unittest.mock import patch

from mathagent.agent import Agent
from mathagent.literature import LiteratureError, LiteratureTools, _digest, bibliography_keys, cites_place
from mathagent import refcheck
from mathagent.refcheck import CheckRunner
from mathagent.tools import Workspace
from tests.test_research import FakeClient, response, tool

ARXIV = '2412.14786v1'  # the citing paper, read through its arXiv HTML
PAPER = '\n'.join([
    'Admissibility theory in abstract Sobolev scales',
    'When the domain is bounded and smooth, the domain of the Neumann Laplacian is exactly the space of all w in H^2',
    'such that the normal derivative vanishes, with equivalent norms; see, e.g., [Bre11, Theorem 9.26].',
    'References',
    '[Bre11] Haim Brezis. Functional analysis, Sobolev spaces and partial differential equations. Springer, 2011.',
]).encode()
ZBMATH = {'result': [{'identifier': '1220.46002', 'title': {'title': 'Functional analysis, Sobolev spaces and partial differential equations'},
                      'contributors': {'authors': [{'name': 'Brezis, Haim'}]}, 'year': '2011',
                      'document_type': {'description': 'book / book article'},
                      'source': {'source': 'Universitext. New York, NY: Springer', 'book': [{'publisher': 'New York, NY: Springer'}]},
                      'zbmath_url': 'https://zbmath.org/5633610',
                      'editorial_contributions': [{'text': 'IX. Sobolev spaces and variational formulation of boundary value problems in dimension N.'}]}]}
BOOK = {'DOI': '10.1007/978-0-387-70914-7', 'title': ['Functional Analysis, Sobolev Spaces and Partial Differential Equations'],
        'author': [{'given': 'Haim', 'family': 'Brezis'}], 'issued': {'date-parts': [[2011]]}, 'type': 'book'}
CITING = {'DOI': '10.1000/ADMISS', 'title': ['Admissibility theory in abstract Sobolev scales'], 'issued': {'date-parts': [[2024]]},
          'author': [{'family': 'Author'}], 'reference': [{'unstructured': 'H. Brezis, Functional analysis, Springer, 2011'}]}
ATOM = (b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2412.14786v1</id>'
        b'<title>Admissibility theory in abstract Sobolev scales</title><summary>Neumann Laplacian.</summary>'
        b'<author><name>A. Author</name></author><published>2024-12-01</published></entry></feed>')


def page(paper=PAPER):
    return b'<html><body><article class="ltx_document">' + b''.join(b'<p>' + line + b'</p>' for line in paper.split(b'\n')) + b'</article></body></html>'


RECALL = {'statement': 'H^2 regularity for the Neumann problem on a smooth bounded domain.',
          'keywords': ['Neumann problem', 'H2 regularity'],
          'candidates': [{'authors': ['Brezis'], 'title': 'Functional analysis, Sobolev spaces and partial differential equations',
                          'year': '2011', 'locator': 'Theorem 9.26', 'why': 'Chapter 9 treats elliptic regularity.'}]}


def fetch(url, headers=None, max_bytes=None, paper=PAPER):
    """The citation chain's sources: Crossref (the work, then papers on the topic), OpenCitations, arXiv."""
    if 'zbmath' in url:
        return json.dumps(ZBMATH).encode(), 'application/json', url
    if 'api.crossref.org' in url:
        items = [BOOK] if 'Brezis' in unquote(url) and 'Functional' in unquote(url) else [CITING]
        return json.dumps({'message': {'items': items}}).encode(), 'application/json', url
    if 'api.opencitations.net' in url:
        return json.dumps([{'citing': 'omid:br/1 doi:10.1000/admiss'}]).encode(), 'application/json', url
    if 'export.arxiv.org' in url:
        return ATOM, 'application/atom+xml', url
    if url == f'https://arxiv.org/html/{ARXIV}':
        return page(paper), 'text/html', url
    raise AssertionError('unexpected fetch ' + url)


DOC = 'doc-' + _digest('arxiv:' + ARXIV)[:24]


def text(value):
    return response(json.dumps(value))


class CheckTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patcher = patch('mathagent.literature.time.sleep')
        patcher.start()
        self.addCleanup(patcher.stop)
        # A compact profile keeps scripted replies short: two guesses, no open-paper search.
        compact = patch.dict(refcheck.EFFORTS, {'low': {**refcheck.EFFORTS['low'], 'candidates': 2, 'discover': False}})
        compact.start()
        self.addCleanup(compact.stop)

    def run_check(self, replies, effort='medium', guesses=(), online=True, fetcher=None, lookup=None):
        lit = LiteratureTools(self.root, online=online)
        self.client = FakeClient(replies)
        runner = CheckRunner(Agent(self.client, Workspace(self.root, literature=lit), ctx=16384, predict=4096))
        with patch.object(lit, '_fetch', side_effect=fetcher or fetch):
            options = {'lookup': lookup} if lookup is not None else {}
            result = runner.start('Find a reference on the regularity of elliptic problems with Neumann boundary conditions.',
                                  effort=effort, guesses=guesses, **options)
        return runner, result

    def searches(self):
        return [response('Search.', [tool('search_papers', query='au:Brezis ti:Functional analysis', provider='zbmath'),
                                     tool('find_quotes', author='Brezis', locator='Theorem 9.26', topic='Neumann regularity',
                                          title='Functional analysis')]),
                response('Open the hit.', [tool('open_paper', identifier_or_url=ARXIV)]),
                response('Find the number.', [tool('search_paper', document_id=DOC, query='9.26')])]

    def verdict(self, **fields):
        value = {'level': 'cited', 'locator': 'Theorem 9.26', 'document_id': DOC, 'lines': 'L3-L3',
                 'record': 'zbl:1220.46002', 'statement_found': 'The Neumann Laplacian has domain H^2 with zero normal derivative.', 'note': ''}
        value.update(fields)
        return text(value)

    def test_standard_lookup_finishes_with_one_book_without_a_locator_hunt(self):
        alternatives = [RECALL['candidates'][0],
                        {'authors': ['Evans'], 'title': 'Partial differential equations',
                         'year': '1998', 'locator': 'Section 6.3', 'why': 'Alternative source.'},
                        {'authors': ['Gilbarg', 'Trudinger'], 'title': 'Elliptic partial differential equations',
                         'year': '2001', 'locator': 'Chapter 6', 'why': 'Alternative source.'}]
        fetched = []

        def metadata_only(url, headers=None, max_bytes=None):
            fetched.append(url)
            self.assertIn('zbmath', url)
            return json.dumps(ZBMATH).encode(), 'application/json', url

        # Only a recall reply is supplied: another verifier/model round would fail.
        runner, result = self.run_check([text({**RECALL, 'candidates': alternatives})],
                                        lookup='standard', fetcher=metadata_only)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(len(fetched), 1)
        self.assertEqual(len(runner.state['candidates']), 1)
        self.assertEqual([e['tool'] for e in runner.state['evidence']], ['search_papers'])
        self.assertTrue(all(e['arguments']['provider'] == 'zbmath' for e in runner.state['evidence']))
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual(verdict['level'], 'located')
        self.assertEqual(verdict['record']['identifier'], 'zbl:1220.46002')
        self.assertFalse(verdict.get('quote'))
        compact = json.loads(runner.result_for_agent())
        self.assertEqual(compact['lookup'], 'standard')
        self.assertEqual(compact['disposition'], 'answer_with_qualification')
        self.assertFalse(compact['precise_place_confirmed'])
        self.assertEqual(compact['references'][0]['level'], 'located')
        self.assertIn('Brezis', runner.state['answer'])
        self.assertIn('unconfirmed', runner.state['answer'].lower())
        self.assertNotIn('What it states', runner.state['answer'])

    def test_human_precision_intent_ignores_explicitly_declined_locators(self):
        from mathagent.refcheck import precise_reference_request
        self.assertFalse(precise_reference_request('Find a classical reference.'))
        self.assertFalse(precise_reference_request('Find a book without an exact theorem number.'))
        self.assertFalse(precise_reference_request("I don't need the precise reference, just a standard book."))
        self.assertTrue(precise_reference_request('Find the exact theorem number in this book.'))
        self.assertTrue(precise_reference_request('Which chapter contains it?'))
        self.assertTrue(precise_reference_request('Verify this citation.'))
        self.assertTrue(precise_reference_request('No need for an exact page; verify this citation.'))
        self.assertTrue(precise_reference_request('No need for an exact page but verify this citation.'))
        self.assertTrue(precise_reference_request('Verify the reference Hirsch, Differential Topology.'))
        self.assertTrue(precise_reference_request('Check whether this reference proves the assertion.'))
        self.assertTrue(precise_reference_request('Give the page on which it appears.'))

    def test_standard_lookup_with_no_record_keeps_only_an_honest_memory_suggestion(self):
        fetched = []

        def no_record(url, headers=None, max_bytes=None):
            fetched.append(url)
            self.assertIn('zbmath', url)
            return b'{"result": []}', 'application/json', url

        runner, result = self.run_check([text(RECALL)], lookup='standard', fetcher=no_record)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(len(fetched), 1)
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'not_found')
        answer = runner.state['answer']
        self.assertIn('Brezis', answer)
        self.assertRegex(answer.lower(), r'memory|model recall')
        self.assertIn('unconfirmed', answer.lower())
        self.assertNotIn('Work record:', answer)
        self.assertNotIn('@book', answer)
        compact = json.loads(runner.result_for_agent())
        self.assertEqual(compact['references'][0]['level'], 'not_found')
        self.assertEqual(compact['disposition'], 'answer_with_qualification')
        self.assertFalse(compact['precise_place_confirmed'])
        self.assertEqual([e['tool'] for e in runner.state['evidence']], ['search_papers'])

    def test_standard_lookup_offline_finishes_without_network_or_fake_confirmation(self):
        def forbidden_fetch(*args, **kwargs):
            self.fail('Offline standard lookup must not fetch a source.')

        runner, result = self.run_check([text(RECALL)], lookup='standard', online=False,
                                        fetcher=forbidden_fetch)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(len(runner.state['candidates']), 1)
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'not_found')
        self.assertIn('Brezis', runner.state['answer'])
        self.assertRegex(runner.state['answer'].lower(), r'memory|model recall')
        compact = json.loads(runner.result_for_agent())
        self.assertEqual(compact['references'][0]['level'], 'not_found')
        self.assertFalse(compact['precise_place_confirmed'])
        self.assertFalse(any(e['tool'] in ('find_quotes', 'open_paper', 'read_paper', 'search_paper')
                             for e in runner.state['evidence']))

    def test_standard_book_record_never_confirms_a_guessed_chapter(self):
        guess = {**RECALL['candidates'][0], 'locator': 'Chapter 9'}
        runner, _ = self.run_check([text({**RECALL, 'candidates': [guess]})], lookup='standard')
        compact = json.loads(runner.result_for_agent())
        self.assertEqual(compact['references'][0]['level'], 'located')
        self.assertFalse(compact['precise_place_confirmed'])
        self.assertIn('Chapter 9', runner.state['answer'])
        self.assertIn('unconfirmed', runner.state['answer'].lower())
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual([e['tool'] for e in runner.state['evidence']], ['search_papers'])

    def test_invalid_lookup_is_rejected_before_model_work(self):
        with self.assertRaises(ValueError):
            self.run_check([], lookup='guess-until-success')
        self.assertEqual(self.client.requests, [])

    def test_standard_answer_preserves_record_metadata_and_flags_edition_mismatch(self):
        guess = {**RECALL['candidates'][0], 'year': '2000', 'locator': 'Chapter 9'}
        runner, _ = self.run_check([text({**RECALL, 'candidates': [guess]})], lookup='standard')
        answer = runner.state['answer']
        self.assertIn('(2011)', answer)
        self.assertIn('Publisher in the bibliographic record: New York, NY: Springer.', answer)
        self.assertIn('record is dated 2011', answer)
        self.assertIn('model recall suggested 2000', answer)
        self.assertIn('may refer to a different edition', answer)
        self.assertIn('unconfirmed', answer)
        compact = json.loads(runner.result_for_agent())
        self.assertIn('(2011)', compact['references'][0]['reference'])
        self.assertNotIn('(2000)', compact['references'][0]['reference'])
        self.assertFalse(compact['precise_place_confirmed'])

    def test_standard_lookup_does_not_replace_a_book_with_an_unrelated_article(self):
        guess = {'authors': ['Hirsch'], 'title': 'Differential Topology', 'year': '1976',
                 'locator': 'Chapter 5', 'why': 'Recalled standard book.'}
        unrelated = copy.deepcopy(ZBMATH)
        unrelated['result'][0].update(
            title={'title': 'On immersions of manifolds in differential topology'},
            contributors={'authors': [{'name': 'Hirsch, Morris W.'}]}, year='1961',
            document_type={'description': 'article'})
        def metadata(url, headers=None, max_bytes=None):
            return json.dumps(unrelated).encode(), 'application/json', url
        runner, _ = self.run_check([text({**RECALL, 'candidates': [guess]})],
                                  lookup='standard', fetcher=metadata)
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'not_found')
        self.assertIn('*Differential Topology* (1976)', runner.state['answer'])
        self.assertIn('bibliographic lookup did not confirm it', runner.state['answer'])
        self.assertNotIn('immersions', runner.state['answer'])
        self.assertNotIn('1961', runner.state['answer'])

    def test_cited_reference_with_record_quote_and_bibtex(self):
        runner, result = self.run_check([text(RECALL), *self.searches(), response('Enough evidence.'), self.verdict()])
        self.assertEqual(result['status'], 'answered')
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual(verdict['level'], 'cited')
        self.assertIn('Theorem 9.26', verdict['quote'])
        self.assertEqual(verdict['record']['identifier'], 'zbl:1220.46002')
        answer = runner.state['answer']
        self.assertIn('Cited by another paper', answer)
        self.assertIn(f'[{DOC}:L3-L3]', answer)
        self.assertIn('zbl = {1220.46002}', answer)
        self.assertIn('@book{brezis2011', answer)
        self.assertIn('Cited there as: Haim Brezis. Functional analysis', answer)
        self.assertNotIn('Edition:', answer)
        compact = json.loads(runner.result_for_agent())
        self.assertEqual(compact['references'][0]['level'], 'cited')
        self.assertTrue((self.root / '.mathagent' / 'checks' / result['id'] / 'report.md').exists())
        self.assertFalse((self.root / '.mathagent' / 'research').exists())  # not listed as a literature report
        # The verifier never gets a tool to start another check, nor workspace file tools.
        names = {t['function']['name'] for r in self.client.requests for t in r.get('tools', [])}
        self.assertEqual(names, {'search_papers', 'find_quotes', 'open_paper', 'read_paper', 'search_paper'})
        # The controller looked up the record and the quotes before the model's first round.
        self.assertEqual([e['tool'] for e in runner.state['evidence'][:2]], ['search_papers', 'find_quotes'])

    def test_find_quotes_reads_the_citing_lines_and_the_bibliography(self):
        lit = LiteratureTools(self.root, online=True)
        with patch.object(lit, '_fetch', side_effect=fetch):
            value = json.loads(lit.execute('find_quotes', {'author': 'Haim Brezis', 'locator': 'Thm. 9.26', 'topic': 'Neumann',
                                                           'title': 'Functional analysis, Sobolev spaces and partial differential equations'}))
            self.assertIn('error', json.loads(lit.execute('find_quotes', {'author': 'Brezis'})))
            self.assertIn('error', json.loads(lit.execute('find_quotes', {'author': '9.26; drop', 'locator': 'Theorem 9.26'})))
        self.assertEqual((value['number'], value['citation_chain']['on_arxiv']), ('9.26', 1))
        quote, bib = value['quotes']
        self.assertEqual((quote['citation'], quote['start_line'], quote['end_line']), (f'{DOC}:L2-L4', 2, 4))
        self.assertIn('[Bre11, Theorem 9.26]', quote['passage'])
        self.assertIn('Functional analysis, Sobolev spaces', bib['bibliography_entry'])

    def test_find_quotes_falls_back_to_arxiv_topic_search_when_the_chain_finds_nothing(self):
        lit = LiteratureTools(self.root, online=True)

        def request(url, headers, max_bytes):
            if 'api.crossref.org' in url:  # the citation chain finds no citing paper here
                return 200, {'content-type': 'application/json'}, b'{"message": {"items": []}}'
            if 'export.arxiv.org' in url:
                return 200, {'content-type': 'application/atom+xml'}, ATOM
            if url == f'https://arxiv.org/html/{ARXIV}':
                return 200, {'content-type': 'text/html'}, page()
            raise AssertionError('unexpected request ' + url)
        with patch.object(lit, '_request_once', side_effect=request):
            value = json.loads(lit.execute('find_quotes', {'author': 'Brezis', 'locator': 'Theorem 9.26', 'topic': 'Neumann Laplacian'}))
        self.assertEqual(value['searched_with'], 'arXiv topic search')
        self.assertEqual(value['citation_chain']['topic_citing'], 0)
        self.assertIn('[Bre11, Theorem 9.26]', value['quotes'][0]['passage'])

    def test_citation_chain_finds_a_citing_paper_on_the_topic_through_its_arxiv_version(self):
        lit = LiteratureTools(self.root, online=True)
        other = {'DOI': '10.1000/OTHER', 'title': ['Neumann problems without Brezis'], 'issued': {'date-parts': [[2020]]},
                 'reference': [{'unstructured': 'H. Brezis, P. Mironescu, Gagliardo-Nirenberg inequalities, 2018'}]}
        def request(url, headers, max_bytes):
            if 'api.crossref.org' in url:
                items = [BOOK] if 'Brezis' in unquote(url) and 'Functional' in unquote(url) else [CITING, other]
                return 200, {'content-type': 'application/json'}, json.dumps({'message': {'items': items}}).encode()
            if 'api.opencitations.net' in url:
                return 200, {'content-type': 'application/json'}, json.dumps([{'citing': 'omid:br/1 doi:10.1000/elsewhere'}]).encode()
            if 'export.arxiv.org' in url:
                self.assertIn('ti:Admissibility', unquote(url))  # only the citing paper is looked up on arXiv
                return 200, {'content-type': 'application/atom+xml'}, ATOM
            if url == f'https://arxiv.org/html/{ARXIV}':
                return 200, {'content-type': 'text/html'}, page()
            raise AssertionError('unexpected request ' + url)
        with patch.object(lit, '_request_once', side_effect=request):
            value = json.loads(lit.execute('find_quotes', {'author': 'Brezis', 'locator': 'Theorem 9.26', 'topic': 'Neumann Laplacian',
                                                           'title': 'Functional analysis, Sobolev spaces and partial differential equations'}))
        self.assertTrue(value['searched_with'].startswith('citation chain'), value)
        self.assertEqual(value['citation_chain'], {'work_dois': ['10.1007/978-0-387-70914-7'], 'citing_known': 1,
                                                   'topic_papers': 2, 'topic_citing': 1, 'on_arxiv': 1})
        self.assertIn('[Bre11, Theorem 9.26]', value['quotes'][0]['passage'])

    def test_a_citation_of_the_place_is_not_an_equation_number(self):
        keys = bibliography_keys(['[5]', 'Gilbarg D. and Trudinger N., Elliptic equations, 2001.',
                                  '[GT] D. Gilbarg, N. Trudinger, Elliptic PDE, Springer, 1983.'], 'Gilbarg')
        self.assertEqual(keys, ['5', 'GT'])
        # With the title, only the entry of that work counts, not the author's other works or a neighbour's words.
        lines = ['[6]', 'H. Brezis, Operateurs maximaux monotones, North-Holland, 1973.', '[7]', 'A. Other, Sobolev spaces, 2001.',
                 '[8] H. Brezis, Functional analysis, Sobolev spaces and PDE, Springer, 2011.']
        self.assertEqual(bibliography_keys(lines, 'Brezis', 'Functional analysis, Sobolev spaces and partial differential equations'), ['8'])
        for text, number, expected in [('see, e.g., [GT, Theorem 9.19]. In', '9.19', True), ('[5, Thm. 9.19; 4]', '9.19', True),
                                       ('by Theorem 9.19 in [5] we get', '9.19', True), ('by Theorem 9.19 in [7] we get', '9.19', False),
                                       ('Gilbarg and Trudinger, Theorem 9.19, give', '9.19', True),
                                       ('(9.19) the identity holds', '9.19', False), ('Equations (9.5) and (9.19) show', '9.19', False),
                                       ('[7, Theorem 9.19]', '9.19', False), ('by [Gil83, Thm. 9.19]', '9.19', True),
                                       ('by [Tru83, Thm. 9.19]', '9.19', False)]:
            self.assertEqual(cites_place(text, number, 'Gilbarg', keys), expected, text)

    def test_wrong_number_is_lowered_to_located(self):
        # The verifier claims the guessed number, but the line it read quotes 9.26.
        runner, result = self.run_check([text({**RECALL, 'candidates': [{**RECALL['candidates'][0], 'locator': 'Theorem 5.3'}]}),
                                         *self.searches(), self.verdict(locator='Theorem 5.3')], effort='low')
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual(verdict['level'], 'located')
        self.assertTrue(any('5.3' in reason for reason in verdict['reasons']))
        self.assertEqual(result['status'], 'partial')

    def test_corrected_number_found_in_read_lines_is_kept(self):
        runner, _ = self.run_check([text({**RECALL, 'candidates': [{**RECALL['candidates'][0], 'locator': 'Theorem 9.25'}]}),
                                    *self.searches(), self.verdict(lines='')], effort='low')
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual((verdict['level'], verdict['locator']), ('cited', 'Theorem 9.26'))
        self.assertIn(verdict['lines'], ('L3-L3', 'L2-L4'))

    def test_a_contradicted_guess_keeps_the_corrected_place_when_a_read_passage_cites_it(self):
        guess = {**RECALL, 'candidates': [{**RECALL['candidates'][0], 'locator': 'Theorem 5.3'}]}
        runner, result = self.run_check([text(guess), response('Read.', [tool('find_quotes', author='Brezis', locator='Theorem 9.26', topic='Neumann')]),
                                         response('Done.'), self.verdict(level='contradicted', locator='Theorem 9.26', lines='')], effort='low')
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual((verdict['level'], verdict['locator'], verdict['corrected_from']), ('cited', 'Theorem 9.26', 'Theorem 5.3'))
        self.assertIn('corrected from the guess Theorem 5.3', runner.state['answer'])
        self.assertEqual(result['status'], 'answered')

    def test_an_older_cited_edition_is_flagged(self):
        guess = {**RECALL, 'candidates': [{**RECALL['candidates'][0], 'year': '2011'}]}
        old = PAPER.replace(b'Springer, 2011.', b'Masson, Paris, 1983.')
        def older(url, headers=None, max_bytes=None):
            if 'zbmath' in url:
                return json.dumps({'result': []}).encode(), 'application/json', url
            return fetch(url, paper=old)
        runner, _ = self.run_check([text(guess), response('Done.'), self.verdict(record='')], effort='low', fetcher=older)
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual((verdict['level'], verdict['cited_edition']), ('cited', '1983'))
        self.assertIn('uses the 1983 edition', runner.state['answer'])

    def test_claim_without_any_read_passage_or_record_is_not_found(self):
        def nothing(url, headers=None, max_bytes=None):
            return json.dumps({'result': [], 'results': []}).encode(), 'application/json', url
        runner, result = self.run_check([text(RECALL), response('I know this one.'),
                                         self.verdict(level='read', document_id='doc-' + 'b' * 24, record='')],
                                        effort='low', fetcher=nothing)
        verdict = runner.state['candidates'][0]['verdict']
        self.assertEqual(verdict['level'], 'not_found')
        self.assertIn('No reference could be confirmed', runner.state['answer'])
        self.assertEqual(result['status'], 'partial')

    def test_caller_guesses_come_first_and_duplicates_merge(self):
        guess = {'authors': ['Brezis'], 'title': 'Functional analysis', 'year': '2011', 'locator': 'Theorem 9.26', 'why': 'asked'}
        runner, _ = self.run_check([text(RECALL), *self.searches(), response('Done.'), self.verdict()], guesses=[guess])
        self.assertEqual([c['source'] for c in runner.state['candidates']], ['caller'])
        self.assertIn('already suggested', self.client.requests[0]['messages'][1]['content'])

    def test_whitespace_capped_recall_recovers_without_the_same_json_grammar(self):
        broken = response('{"statement":"H2 regularity","candidates":[{"year":' + ' ' * 1000,
                          complete=False)
        runner, result = self.run_check([broken, text(RECALL), *self.searches(),
                                        response('Enough evidence.'), self.verdict()])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'cited')
        self.assertIn('format', self.client.requests[0])
        self.assertNotIn('format', self.client.requests[1])
        self.assertFalse(self.client.requests[1]['think'])
        self.assertTrue(any('constrained decoding' in w for w in runner.state['warnings']))

    def test_discovery_adds_the_reference_an_open_paper_cites(self):
        empty = {**RECALL, 'candidates': []}
        discovered = {'found': True, 'candidate': RECALL['candidates'][0]}
        runner, result = self.run_check([
            text(empty),
            response('Search.', [tool('search_papers', query='Neumann H2 regularity', provider='arxiv')]),
            response('Read it.', [tool('open_paper', identifier_or_url=ARXIV), tool('search_paper', document_id=DOC, query='Theorem')]),
            text(discovered),
            response('Record.', [tool('search_papers', query='au:Brezis ti:Functional analysis', provider='zbmath')]),
            self.verdict(lines='L3'),
        ])
        self.assertEqual(runner.state['candidates'][0]['source'], 'discovery')
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'cited')
        self.assertEqual(result['status'], 'answered')

    def test_plain_recall_recovery_rejects_malformed_candidate_types(self):
        runner, result = self.run_check([response('{', complete=False),
            text({'statement': 'Result sought', 'keywords': 17, 'candidates': [{'authors': 23, 'title': 'Work'}]})],
            effort='low')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(runner.state['candidates'], [])
        self.assertEqual(runner.state['keywords'], [])

    def test_repeated_calls_are_skipped_and_reused_across_guesses(self):
        evans = {'authors': ['Evans'], 'title': 'Partial differential equations', 'year': '1998', 'locator': 'Theorem 6.3.4', 'why': 'w'}
        search = tool('search_papers', query='au:Brezis ti:Functional analysis', provider='zbmath', limit=8)
        lit_calls = []
        original = fetch
        with patch.object(sys.modules[__name__], 'fetch', side_effect=lambda *a, **k: lit_calls.append(a[0]) or original(*a, **k)):
            runner, _ = self.run_check([text({**RECALL, 'candidates': [RECALL['candidates'][0], evans]}),
                                        response('Search twice.', [search, search]), response('Again.', [search]),
                                        self.verdict(level='located', document_id='', lines=''),
                                        response('Same search.', [search]), response('Nothing.'),
                                        text({'level': 'not_found', 'locator': '', 'document_id': '', 'lines': '', 'record': '',
                                              'statement_found': '', 'note': ''})], effort='low')
        mine = [u for u in lit_calls if 'zbmath' in u and 'Functional+analysis&' in u]
        self.assertEqual(len(mine), 1)
        self.assertIn('results_per_page=5', mine[0])
        repeated = [e['candidate'] for e in runner.state['evidence'] if e['arguments'].get('query') == 'au:Brezis ti:Functional analysis']
        self.assertEqual(repeated, [0, 1])
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'located')

    def test_offline_check_reports_without_network(self):
        runner, result = self.run_check([text(RECALL), *self.searches(), self.verdict()], effort='low', online=False)
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'not_found')
        self.assertEqual(result['status'], 'partial')
        self.assertTrue(any('Offline' in e['result'] for e in runner.state['evidence']))


if __name__ == '__main__':
    unittest.main()
