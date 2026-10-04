"""Literature review: the reading-list pipeline, with scripted model answers and canned sources."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote

from mathagent.agent import Agent
from mathagent.literature import LiteratureError, LiteratureTools
from mathagent.litreview import ReviewRunner, breadth
from mathagent.tools import Workspace
from tests.test_research import FakeClient, response

ATOM = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/1001.00001v2</id><title>Sharp observability estimates for heat equations</title>
<summary>Observability of the heat equation with explicit constants.</summary><author><name>Sylvain Ervedoza</name></author>
<published>2010-01-01</published><updated>2010-02-01</updated></entry></feed>'''


def work(n, title, year, refs=(), cited=10, kind='journal-article'):
    """A Crossref item; references are DOIs, as in Crossref's open reference lists."""
    return {'DOI': f'10.1000/W{n}', 'title': [title], 'author': [{'given': 'Ann', 'family': f'Surname{n}'}],
            'issued': {'date-parts': [[year]]}, 'type': kind, 'container-title': ['Journal'], 'is-referenced-by-count': cited,
            'reference': [{'DOI': f'10.1000/w{r}'} for r in refs] + [{'key': 'unstructured, no DOI'}]}


SEARCH = [work(1, 'Sharp observability estimates for heat equations', 2011, refs=[9, 10], cited=67),
          work(2, 'Controllability of the heat equation: a survey', 2005, refs=[9, 10], cited=300),
          work(3, 'Null controllability of parabolic equations', 2016, refs=[9], cited=20),
          work(4, 'A cooking recipe for observability', 2019, refs=[], cited=5000)]
BACKWARD = [work(9, 'Controlabilite exacte de la chaleur', 1995, cited=900), work(10, 'Carleman estimates for parabolic equations', 1996, cited=800)]
HIDDEN = [{'DOI': '10.1080/03605309508821097', 'title': ['Contr\u00f4le exact de l\u2019\u00e9quation de la chaleur'],
           'author': [{'given': 'G.', 'family': 'Lebeau'}, {'given': 'L.', 'family': 'Robbiano'}],
           'issued': {'date-parts': [[1995]]}, 'type': 'journal-article', 'container-title': ['Comm. PDE &amp; Appl.']}]
S2_SEARCH = {'data': [{'paperId': 'p1', 'title': 'Sharp observability estimates for heat equations', 'year': 2011,
                       'authors': [{'name': 'Ann Surname1'}], 'externalIds': {'DOI': '10.1000/W1', 'ArXiv': '1001.00001'},
                       'citationCount': 67, 'abstract': 'Sharp constants for the observability of heat equations.'}]}
S2_CITING = {'data': [{'citingPaper': {'paperId': 'p11', 'title': 'Observability of the heat equation on manifolds', 'year': 2022,
                                       'authors': [{'name': 'Cy Surname11'}], 'externalIds': {'DOI': '10.1000/W11'}, 'citationCount': 8}}]}
ZBMATH = {'result': [{'identifier': '1234.93001', 'title': {'title': 'Control and nonlinearity'}, 'year': '2007',
                      'contributors': {'authors': [{'name': 'Coron, Jean-Michel'}]}, 'document_type': {'description': 'book / book article'},
                      'source': {'book': [{'publisher': 'Providence: AMS'}]}, 'zbmath_url': 'https://zbmath.org/1',
                      'msc': [{'code': '93B05'}, {'code': '93C20'}],
                      'editorial_contributions': [{'text': 'A textbook on the control of nonlinear systems, with a chapter on the heat equation.'}]},
                     {'identifier': '0819.35071', 'title': {'title': 'zbMATH Open Web Interface contents unavailable due to conflicting licenses.'},
                      'contributors': {'authors': [{'name': 'zbMATH Open Web Interface contents unavailable due to conflicting licenses.'}]},
                      'links': [{'type': 'doi', 'identifier': '10.1080/03605309508821097'}], 'year': '1995'}]}


def fetch(url, headers=None, max_bytes=None):
    if 'api.crossref.org' in url:
        if 'filter=' in url:
            wanted = unquote(url).lower()
            items = HIDDEN if '10.1080' in wanted else [w for w in BACKWARD if w['DOI'].lower() in wanted]
        else:
            items = [] if 'Coron' in url or 'Imaginary' in url else SEARCH
        return json.dumps({'message': {'items': items}}).encode(), 'application/json', url
    if 'api.semanticscholar.org' in url:
        body = S2_CITING if '/citations' in url else S2_SEARCH
        return json.dumps(body).encode(), 'application/json', url
    if 'export.arxiv.org' in url:
        return ATOM, 'application/atom+xml', url
    if 'zbmath' in url:
        if 'Imaginary' in url:
            return json.dumps({'result': []}).encode(), 'application/json', url
        return json.dumps(ZBMATH).encode(), 'application/json', url
    raise AssertionError('unexpected fetch ' + url)


SCOPE = {'topic': 'Observability and control of the heat equation', 'subfield': 'Control of PDE', 'msc': ['93B07', 'bad'],
         'intent': 'both', 'assumptions': 'A graduate reader; classical and recent results.',
         'queries': ['observability heat equation', 'null controllability parabolic'],
         'seeds': [{'authors': ['Coron'], 'title': 'Control and nonlinearity', 'year': '2007'},
                   {'authors': ['Nobody'], 'title': 'Imaginary treatise on heat', 'year': '1999'}]}


def text(value):
    return response(json.dumps(value))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        sleeper = patch('mathagent.literature.time.sleep')
        sleeper.start()
        self.addCleanup(sleeper.stop)

    def run_review(self, replies, rounds=1, online=True, **budgets):
        self.lit = LiteratureTools(self.root, online=online)
        self.client = FakeClient(replies)
        runner = ReviewRunner(Agent(self.client, Workspace(self.root, literature=self.lit), ctx=32768, predict=4096))
        settings = dict(max_rounds=rounds, max_tokens=40000, max_input_tokens=300000, max_seconds=600, max_requests=60, max_chars=30000)
        settings.update(budgets)
        with patch.object(self.lit, '_fetch', side_effect=fetch):
            result = runner.start('What should a graduate student read on observability of the heat equation?', **settings)
        return runner, result

    def pool(self, runner):
        return {r['title']: r for r in runner.state['pool']}

    def numbers(self, runner, *titles):
        pool = self.pool(runner)
        return [pool[t]['n'] for t in titles]

    def full_run(self):
        # The numbers are known only after the sweep, so the scripted answers are built lazily.
        runner_box = {}

        class Lazy(FakeClient):
            def stream(inner, payload):
                inner.requests.append(payload)
                role = len(inner.requests)
                runner = runner_box['runner']
                pool = self.pool(runner) if runner.state else {}
                n = {t: r['n'] for t, r in pool.items()}
                if role == 1:
                    reply = text(SCOPE)
                elif role == 2:
                    items = [{'n': r['n'], 'relevance': 0 if 'cooking' in r['title'] else 3 if r['cited_by'] and r['cited_by'] > 50 else 2,
                              'role': 'exclude' if 'cooking' in r['title'] else 'entry' if 'survey' in r['title'] or 'nonlinearity' in r['title'] else 'core',
                              'note': ''} for r in pool.values()]
                    reply = text({'items': items})
                elif role == 3:
                    reply = text({'orientation': 'Observability of parabolic equations, from Carleman estimates to sharp constants.',
                                  'entry': [n['Controllability of the heat equation: a survey'], n['Control and nonlinearity'], 999],
                                  'themes': [{'name': 'Carleman estimates', 'core': [n['Carleman estimates for parabolic equations'], n['Controlabilite exacte de la chaleur']],
                                              'deeper': [n['Null controllability of parabolic equations']]},
                                             {'name': 'Sharp constants', 'core': [n['Sharp observability estimates for heat equations'],
                                                                                  n['Controllability of the heat equation: a survey']],
                                              'deeper': [n['A cooking recipe for observability']]}],
                                  'reading_order': f'Start with #{n["Controllability of the heat equation: a survey"]}, then the Carleman papers.',
                                  'gaps': 'Few works on manifolds.'})
                else:
                    listed = [int(x) for x in __import__('re').findall(r'^#(\d+)', payload['messages'][1]['content'], __import__('re').M)]
                    reply = text({'items': [{'n': m, 'why': f'Explains result {m}.'} for m in listed]})
                yield from reply

        client = Lazy([])
        lit = LiteratureTools(self.root, online=True)
        runner = ReviewRunner(Agent(client, Workspace(self.root, literature=lit), ctx=32768, predict=4096))
        runner_box['runner'] = runner
        self.client = client
        with patch.object(lit, '_fetch', side_effect=fetch):
            result = runner.start('What should a graduate student read on observability of the heat equation?', max_rounds=1,
                                  max_tokens=40000, max_input_tokens=300000, max_seconds=600, max_requests=60, max_chars=30000)
        return runner, result

    def test_reading_list_from_sources_graph_and_model_choices(self):
        runner, result = self.full_run()
        self.assertEqual(result['status'], 'listed', runner.state.get('citation_issues'))
        pool = self.pool(runner)
        # The arXiv hit, the Crossref and the Semantic Scholar records of the same paper are one candidate.
        sharp = pool['Sharp observability estimates for heat equations']
        self.assertEqual((sharp['arxiv'], sharp['doi']), ('1001.00001', '10.1000/w1'))
        self.assertEqual(set(sharp['sources']), {'semantic_scholar', 'crossref', 'arxiv'})
        # Backward citations: works cited by several key papers join the pool and count as hubs.
        # Backward: works the pool keeps citing (Crossref references) join the pool and rank as hubs.
        self.assertEqual(pool['Carleman estimates for parabolic equations']['pool_cites'], 2)
        self.assertEqual(pool['Controlabilite exacte de la chaleur']['pool_cites'], 3)
        self.assertIn('backward', pool['Carleman estimates for parabolic equations']['found_by'])
        self.assertIn('forward', pool['Observability of the heat equation on manifolds']['found_by'])
        # A remembered book is found on zbMATH; an imaginary one is reported, never listed.
        self.assertIn('seed 1', pool['Control and nonlinearity']['found_by'])
        self.assertEqual([u['title'] for u in runner.state['unverified']], ['Imaginary treatise on heat'])
        draft = runner.state['draft']
        self.assertIn('# Reading list: Observability and control of the heat equation', draft)
        self.assertIn('## Start here (entry points)', draft)
        self.assertIn('[doi:10.1000/w2](https://doi.org/10.1000/w2)', draft)
        self.assertIn('[arXiv:1001.00001](https://arxiv.org/abs/1001.00001)', draft)
        self.assertIn('[Zbl 1234.93001]', draft)
        self.assertIn('*Why:* Explains result', draft)
        self.assertIn('Start with Surname2 (2005), then the Carleman papers.', draft)
        self.assertIn('Imaginary treatise on heat', draft.split('## Notes & gaps')[1])
        self.assertIn('MSC: 93B07', draft)  # invalid codes from the model are dropped
        self.assertIn('93B05', draft)  # codes seen in zbMATH records
        # Each work is listed once, an excluded or unknown number never.
        self.assertEqual(draft.count('*Controllability of the heat equation: a survey*'), 1)
        self.assertNotIn('cooking', draft)
        # The model saw numbered titles and abstracts, never an identifier to copy.
        for request in self.client.requests[1:]:
            self.assertNotIn('doi.org', request['messages'][1]['content'])
            self.assertNotIn('1001.00001', request['messages'][1]['content'])
        self.assertTrue((runner.directory / 'report.md').exists())
        self.assertEqual([n['title'] for n in runner.state['notes']], ['Keyword sweep', 'Citation graph and ranking', 'Screening'])
        # A zbMATH record shown without metadata is completed from Crossref by its DOI, never listed blank.
        lebeau = pool['Contrôle exact de l’équation de la chaleur']
        self.assertEqual((lebeau['doi'], lebeau['venue']), ('10.1080/03605309508821097', 'Comm. PDE & Appl'))
        self.assertFalse(any('contents unavailable' in r['title'] + ' '.join(r['authors']) for r in runner.state['pool']))

    def test_offline_review_reports_unreached_sources_and_lists_nothing_invented(self):
        runner, result = self.run_review([text(SCOPE)], online=False)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(runner.state['pool'], [])
        self.assertIn('No candidate was screened as relevant', runner.state['draft'])
        self.assertIn('crossref: 0/', runner.state['draft'])
        self.assertIn('Offline', runner.state['draft'])

    def test_budget_exhaustion_still_lists_the_best_verified_candidates(self):
        runner, result = self.run_review([response(json.dumps(SCOPE), count=3900)], max_tokens=4100)
        self.assertIn(result['status'], ('budget_exhausted', 'partial'))
        report = (runner.directory / 'report.md').read_text()
        self.assertIn('## Candidates found (not organised)', report)
        self.assertIn('doi.org/10.1000/w', report)

    def test_a_source_out_of_quota_is_not_asked_again(self):
        calls = []

        def quota(url, headers=None, max_bytes=None):
            calls.append(url)
            if 'api.semanticscholar.org' in url:
                raise LiteratureError('Provider rate limited this request; try later or use a different provider')
            return fetch(url)
        self.lit = LiteratureTools(self.root, online=True)
        runner = ReviewRunner(Agent(FakeClient([text(SCOPE)]), Workspace(self.root, literature=self.lit), ctx=32768, predict=4096))
        with patch.object(self.lit, '_fetch', side_effect=quota):
            runner.start('Heat observability reading list', max_rounds=1, max_tokens=40000, max_input_tokens=300000,
                         max_seconds=600, max_requests=60, max_chars=30000)
        self.assertEqual(len([u for u in calls if 'semanticscholar' in u]), 1)
        self.assertIn('semantic_scholar', runner.state['down'])
        self.assertIn('Carleman estimates for parabolic equations', [r['title'] for r in runner.state['pool']])
        self.assertIn('come from Semantic Scholar, which was unavailable', (runner.directory / 'report.md').read_text())

    def test_breadth_grows_with_effort(self):
        self.assertEqual(breadth(1), {'queries': 3, 'hubs': 2, 'screen': 45})
        self.assertEqual(breadth(3), {'queries': 5, 'hubs': 6, 'screen': 75})
        self.assertEqual(breadth(100), {'queries': 12, 'hubs': 12, 'screen': 160})


if __name__ == '__main__':
    unittest.main()
