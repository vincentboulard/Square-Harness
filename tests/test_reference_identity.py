"""Bibliographic identity recovery, source-gated recommendations and natural routing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mathagent.agent import Agent
from mathagent.literature import LiteratureTools
from mathagent.mode_calls import ModeOrchestrator, automatic_reference_action
from mathagent.refcheck import CheckRunner
from mathagent.tools import Workspace
from tests.test_orchestrator import Client, final
from tests.test_research import response
from tests.test_refcheck import StandardSelectionClient, empty_metadata

WRONG = {'statement': 'A classical reference on stochastic processes.',
         'keywords': ['stochastic processes', 'Brownian motion'],
         'candidates': [{'authors': ['Karatzas', 'Shreve'],
                         'title': 'Brownian Motion and Stochastic Processes',
                         'year': '1989', 'locator': '', 'why': 'Recalled graduate text.'}]}


def book(title='Brownian motion and stochastic calculus', year='1988',
         authors=('Karatzas, Ioannis', 'Shreve, Steven E.'), kind='book / book article',
         identifier='0638.60065'):
    return {'identifier': identifier, 'title': {'title': title}, 'year': year,
            'contributors': {'authors': [{'name': name} for name in authors]},
            'document_type': {'description': kind},
            'source': {'source': 'Graduate Texts in Mathematics 113. Springer-Verlag.',
                       'book': [{'publisher': 'Springer-Verlag'}]},
            'zbmath_url': 'https://zbmath.org/192908'}


def provider_metadata(url, records):
    """Expose the same fixture works through each provider's actual wire format."""
    if 'zbmath' in url:
        return json.dumps({'result': records}).encode(), 'application/json', url
    if 'crossref' in url:
        items = []
        for record in records:
            authors = []
            for author in record['contributors']['authors']:
                family, _, given = author['name'].partition(', ')
                authors.append({'family': family, 'given': given})
            title = record['title']['title']
            items.append({'DOI': '10.1000/' + record['identifier'] + '.' + str(record['year'] or 'undated'),
                          'title': [title], 'author': authors,
                          'issued': {'date-parts': [[int(record['year'])]]} if record['year'] else {},
                          'type': 'book' if 'book' in record['document_type']['description'] else 'journal-article',
                          'publisher': 'Springer-Verlag'})
        return json.dumps({'message': {'items': items}}).encode(), 'application/json', url
    if 'openlibrary' in url:
        docs = [{'key': '/works/OL' + str(index + 1) + 'W', 'title': record['title']['title'],
                 'author_name': [a['name'] for a in record['contributors']['authors']],
                 **({'first_publish_year': int(record['year'])} if record['year'] else {})}
                for index, record in enumerate(records) if 'book' in record['document_type']['description']]
        return json.dumps({'docs': docs}).encode(), 'application/json', url
    return empty_metadata(url)


class BibliographicRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run_number = 0

    def run_lookup(self, records, *, recall=WRONG, online=True, emit=None, selector=None, **limits):
        client = StandardSelectionClient([response(json.dumps(recall))], selector=selector)
        self.run_number += 1
        folder = self.root/str(self.run_number)
        folder.mkdir()
        lit = LiteratureTools(folder, online=online)
        runner = CheckRunner(Agent(client, Workspace(folder, literature=lit), ctx=16384), emit or (lambda *args: None))
        fetched = []
        def fetch(url, *args, **kwargs):
            if not online:
                self.fail('Offline lookup contacted the network.')
            fetched.append(url)
            batch = records[min(len(fetched)-1, len(records)-1)]
            data, mime, _ = provider_metadata(url, batch)
            return 200, {'content-type': mime}, data
        with patch.object(lit, '_request_once', side_effect=fetch):
            result = runner.start('Give me a classical reference on stochastic processes.',
                                  lookup='standard', **limits)
        return runner, result, client, fetched

    def isolated_selector(self):
        runner = CheckRunner.__new__(CheckRunner)
        runner.state = {'goal': 'Give me a classical reference on stochastic processes.',
                        'check': {'lookup': 'standard'}, 'settings': {'online': True},
                        'candidates': [], 'warnings': []}
        return runner

    def test_same_work_from_distinct_providers_uses_one_recommendation_slot(self):
        for shared_doi in (True, False):
            with self.subTest(shared_doi=shared_doi):
                runner = self.isolated_selector()
                record = {'title': 'Brownian Motion and Stochastic Calculus',
                          'authors': ['Karatzas, Ioannis', 'Shreve, Steven E.'],
                          'year': '1988', 'type': 'book', 'abstract': 'Continuous-time processes.'}
                first = {**record, 'identifier': 'doi:10.1000/brownian', 'url': 'https://doi.org/10.1000/brownian'}
                second = {**record, 'identifier': 'zbl:0638.60065', 'title': 'Brownian motion and stochastic calculus',
                          'authors': ['Steven E. Shreve', 'Ioannis Karatzas'], 'url': 'https://zbmath.org/192908'}
                if shared_doi:
                    first['doi'] = second['doi'] = '10.1000/brownian'
                runner._standard_pool_add(first)
                runner._standard_pool_add(second)
                self.assertEqual(len(runner.state['standard_pool']), 1)
                selections = [{'id': source['id'], 'reason': 'Continuous-time processes.',
                               'coverage': 'supported', 'limitations': ''}
                              for source in runner.state['standard_pool']]
                with patch.object(runner, '_call', return_value={'text': json.dumps({'selections': selections, 'unresolved': ''})}):
                    runner._standard_select()
                self.assertEqual(len(runner.chosen()), 1)
                self.assertEqual(runner._compose_standard().count('Suggested standard reference:'), 1)

    def test_long_two_reference_agent_result_preserves_valid_json_metadata_and_scope(self):
        runner = self.isolated_selector()
        runner.state.update(status='answered', id='two-long-scope-references')
        records = [
            {'identifier': 'doi:10.1000/sobolev', 'title': 'Sobolev Spaces', 'authors': ['Adams, Robert A.'],
             'year': '1975', 'type': 'book', 'url': 'https://doi.org/10.1000/sobolev',
             'abstract': 'Sobolev spaces and embedding theory.'},
            {'identifier': 'doi:10.1000/heat', 'title': 'Analysis of Heat Equations on Domains',
             'authors': ['Ouhabaz, El Maati'], 'year': '2005', 'type': 'book',
             'url': 'https://doi.org/10.1000/heat', 'abstract': 'Heat equations and operator semigroups.'}]
        for record in records:
            runner._standard_pool_add(record)
        reason = 'Indexed subject relevance. ' * 30
        limitations = 'Full-text assumptions remain unchecked. ' * 20
        response_value = {'selections': [{'id': source['id'], 'reason': reason, 'coverage': 'supported',
                                          'limitations': limitations} for source in runner.state['standard_pool']],
                          'unresolved': ''}
        with patch.object(runner, '_call', return_value={'text': json.dumps(response_value)}):
            runner._standard_select()
        serialized = runner.result_for_agent()
        self.assertGreater(len(serialized), 3000)
        result = json.loads(serialized)
        self.assertEqual(len(result['references']), 2)
        self.assertFalse(result['precise_place_confirmed'])
        for record, reference in zip(records, result['references']):
            self.assertEqual(reference['record'], record['identifier'])
            self.assertEqual(reference['url'], record['url'])
            self.assertIn(record['title'], reference['reference'])
            self.assertIn(record['year'], reference['reference'])
            self.assertTrue(reference['metadata_confirmed'])
            self.assertFalse(reference['exact_statement_confirmed'])
            self.assertEqual(reference['suitability']['reason'], reason[:800])
            self.assertEqual(reference['suitability']['limitations'], limitations[:800])

    def test_duplicate_description_is_retained_with_its_source_only_for_the_same_dated_edition(self):
        for later_year in ('1988', '1991'):
            with self.subTest(later_year=later_year):
                runner = self.isolated_selector()
                canonical_url = 'https://doi.org/10.1000/brownian'
                description_url = 'https://zbmath.org/192908'
                original = {'identifier': 'doi:10.1000/brownian', 'title': 'Brownian Motion and Stochastic Calculus',
                            'authors': ['Karatzas, Ioannis', 'Shreve, Steven E.'], 'year': '1988',
                            'type': 'book', 'url': canonical_url}
                review = 'The book develops Brownian motion, martingales and stochastic calculus.'
                runner._standard_pool_add(original)
                runner._standard_pool_add({**original, 'identifier': 'zbl:0638.60065', 'year': later_year,
                                           'review': review, 'url': description_url})
                self.assertEqual(len(runner.state['standard_pool']), 1)
                record = runner.state['standard_pool'][0]['record']
                self.assertEqual(record['url'], canonical_url)
                self.assertEqual(record['year'], '1988')
                if later_year == '1988':
                    self.assertEqual(record['review'], review)
                    self.assertEqual(record['description_source_url'], description_url)
                else:
                    self.assertNotIn('review', record)
                    self.assertNotIn('description_source_url', record)

    def test_repeated_selection_of_a_discovered_record_reuses_its_saved_candidate(self):
        runner = self.isolated_selector()
        runner._standard_pool_add({'identifier': 'doi:10.1000/processes', 'title': 'Stochastic Processes',
                                   'authors': ['Doob, Joseph L.'], 'year': '1953', 'type': 'book',
                                   'abstract': 'General stochastic processes.'})
        source_id = runner.state['standard_pool'][0]['id']
        answer = {'selections': [{'id': source_id, 'reason': 'A useful partial lead.', 'coverage': 'partial',
                                  'limitations': 'Further scope still needs checking.'}],
                  'unresolved': 'Further scope needs checking.'}
        with patch.object(runner, '_call', return_value={'text': json.dumps(answer)}):
            runner._standard_select()
            runner._standard_select()
        self.assertEqual(len(runner.state['candidates']), 1)
        self.assertEqual(len(runner.chosen()), 1)
        self.assertEqual(runner.state['standard_pool'][0]['candidate_index'], 0)

    def test_source_id_is_not_reassigned_after_a_rejected_pool_is_cleared(self):
        runner = self.isolated_selector()
        first = {'identifier': 'doi:10.1000/old', 'title': 'An unrelated lead',
                 'authors': ['Author, First'], 'year': '2000', 'type': 'book'}
        runner._standard_pool_add(first)
        old_id = runner.state['standard_pool'][0]['id']
        runner.state['standard_pool'] = []
        runner._standard_pool_add({**first, 'identifier': 'doi:10.1000/new', 'title': 'A different actual work'})
        self.assertNotEqual(runner.state['standard_pool'][0]['id'], old_id)

    def test_wrong_title_and_year_recover_across_providers_and_relevance_selection(self):
        runner, result, client, fetched = self.run_lookup([[], [book(), book(year='1991')]])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(fetched), 2)
        self.assertEqual([e['arguments']['query'] for e in runner.state['evidence']],
                         ['au:Karatzas ti:Brownian Motion and Stochastic Processes',
                          'Karatzas Brownian Motion and Stochastic Processes'])
        self.assertEqual([e['arguments']['provider'] for e in runner.state['evidence']], ['zbmath', 'crossref'])
        self.assertIn('stochastic calculus', runner.state['answer'].lower())
        self.assertIn('(1988)', runner.state['answer'])
        self.assertNotIn('Stochastic Processes', runner.state['answer'])
        self.assertNotIn('1989', runner.state['answer'])
        self.assertIn('differs from model recall', runner.state['answer'])
        compact = json.loads(runner.result_for_agent())
        self.assertEqual(compact['references'][0]['record'], runner.state['standard_pool'][0]['record']['identifier'])
        self.assertTrue(runner.state['standard_pool'][0]['record']['identifier'].startswith('doi:'))
        self.assertEqual(compact['references'][0]['level'], 'located')
        self.assertFalse(compact['precise_place_confirmed'])

    def test_exact_record_stops_after_one_query(self):
        correct = copy.deepcopy(WRONG)
        correct['candidates'][0].update(title='Brownian motion and stochastic calculus', year='1988')
        runner, result, client, fetched = self.run_lookup([[book()]], recall=correct)
        self.assertEqual(len(fetched), 1)
        self.assertEqual(result['status'], 'answered')
        self.assertFalse(runner.state['candidates'][0].get('bibliographic_recovery', False))

    def test_available_recalled_edition_is_preferred_to_the_first_record(self):
        correct = copy.deepcopy(WRONG)
        correct['candidates'][0].update(title='Brownian motion and stochastic calculus', year='1991')
        runner, result, client, fetched = self.run_lookup([[book(), book(year='1991')]], recall=correct)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(runner.chosen()[0]['verdict']['record']['year'], '1991')
        self.assertIn('(1991)', runner.state['answer'])
        self.assertNotIn('(1988)', runner.state['answer'])

    def test_near_title_with_other_authors_or_unrelated_title_is_not_silently_relabeled(self):
        for record in (book(authors=('Karatzas, Ioannis', 'Another, Author')),
                       book(title='Methods of Mathematical Finance')):
            with self.subTest(record=record):
                runner, result, client, fetched = self.run_lookup([[], [record]])
                self.assertEqual(result['status'], 'partial')
                self.assertNotIn('Suggested standard reference', runner.state['answer'])
                self.assertNotIn('1989', runner.state['answer'])
                self.assertEqual(json.loads(runner.result_for_agent())['references'], [])

    def test_article_title_recovery_is_as_eligible_as_book_recovery(self):
        runner, result, client, fetched = self.run_lookup([[book(kind='journal article')]])
        self.assertEqual(result['status'], 'answered')
        chosen = json.loads(runner.result_for_agent())['references']
        self.assertEqual(chosen[0]['level'], 'located')
        self.assertEqual(runner.state['standard_pool'][0]['record']['type'], 'journal article')
        self.assertIn('stochastic calculus', runner.state['answer'].lower())

    def test_two_distinct_nearby_titles_are_not_relabeled_as_the_recalled_work(self):
        runner, result, client, fetched = self.run_lookup(
            [[], [book(), book(title='Brownian motion and stochastic integration', identifier='0638.60066')]])
        self.assertEqual(result['status'], 'partial')
        self.assertGreater(len(fetched), 2)
        self.assertLessEqual(len(fetched), runner.state['settings']['max_requests'])
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'not_found')
        self.assertNotIn('stochastic calculus', runner.state['answer'].lower())
        self.assertNotIn('stochastic integration', runner.state['answer'].lower())

    def test_ambiguous_work_can_be_independently_discovered_and_selected_by_its_source_id(self):
        runner, result, client, fetched = self.run_lookup(
            [[], [book(), book(title='Brownian motion and stochastic integration', identifier='0638.60066')]],
            selector=lambda pool: [next(item['id'] for item in pool
                                       if item['record']['title'] == 'Brownian motion and stochastic integration')])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(runner.state['candidates'][0]['verdict']['level'], 'not_found')
        self.assertEqual(runner.chosen()[0]['source'], 'bibliographic_discovery')
        self.assertIn('stochastic integration', runner.state['answer'])
        self.assertNotIn('differs from model recall', runner.state['answer'])
        selected_record = next(item['record']['identifier'] for item in runner.state['standard_pool']
                               if item['record']['title'] == 'Brownian motion and stochastic integration')
        self.assertEqual(json.loads(runner.result_for_agent())['references'][0]['record'], selected_record)

    def test_online_failure_withholds_false_metadata_and_fabricated_author_leads(self):
        runner, result, client, fetched = self.run_lookup([[], []])
        self.assertGreater(len(fetched), 2)
        self.assertLessEqual(len(fetched), runner.state['settings']['max_requests'])
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result['status'], 'partial')
        self.assertNotIn('Karatzas', runner.state['answer'])
        self.assertNotIn('Brownian Motion and Stochastic Processes', runner.state['answer'])
        self.assertNotIn('1989', runner.state['answer'])
        self.assertEqual(json.loads(runner.result_for_agent())['references'], [])

    def test_missing_record_year_never_uses_the_guessed_year(self):
        runner, result, client, fetched = self.run_lookup([[], [book(year='')]])
        self.assertEqual(result['status'], 'answered')
        self.assertNotIn('1989', runner.state['answer'])
        self.assertNotIn('1989', json.loads(runner.result_for_agent())['references'][0]['reference'])

    def test_offline_recollection_remains_explicitly_unverified_and_no_network(self):
        runner, result, client, fetched = self.run_lookup([[]], online=False)
        self.assertEqual(fetched, [])
        self.assertEqual(result['status'], 'answered')
        self.assertIn('Unverified', runner.state['answer'])
        self.assertIn('model recall', runner.state['answer'])
        self.assertNotIn('Bibliographic identity was checked', runner.state['answer'])
        self.assertIn('did not confirm', runner.state['answer'])
        self.assertFalse(runner.state['candidates'][0].get('bibliographic_recovery'))

    def test_explicit_request_limit_is_preserved(self):
        runner, result, client, fetched = self.run_lookup([[], []], max_requests=1)
        self.assertEqual(len(fetched), 1)
        self.assertEqual(runner.state['settings']['max_requests'], 1)
        self.assertNotIn('Suggested standard reference', runner.state['answer'])

    def test_pause_before_fallback_preserves_pending_query_and_does_not_repeat_recall(self):
        paused = []
        def emit(kind, value):
            if kind == 'notice' and 'with crossref.' in value:
                paused.append(True)
                raise KeyboardInterrupt()
        runner, result, client, fetched = self.run_lookup([[]], emit=emit)
        self.assertTrue(paused)
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(runner.state['candidates'][0]['standard_search_stage'], 2)
        self.assertIsNotNone(runner.state['pending_tools'])
        runner.emit = lambda *args: None
        with patch.object(runner.literature, '_fetch',
                          side_effect=lambda url, **kwargs: provider_metadata(url, [book()])):
            resumed = runner.resume(result['id'])
        self.assertEqual(resumed['status'], 'answered')
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(runner.state['evidence']), 2)


class NaturalReferenceRoutingTests(unittest.TestCase):
    def test_standalone_reference_is_checked_without_parent_recall(self):
        query = 'Give me a classical reference on stochastic processes.'
        started = []
        def worker(action, child, save):
            started.append(action)
            return {'status': 'answered', 'answer': 'Record-backed reference.',
                    'disposition': 'answer_with_qualification', 'job_id': 'one_lookup'}
        client = Client([])
        runner = ModeOrchestrator(Agent(client, Workspace('/tmp'), ctx=32768), worker)
        result = runner.run(query)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], 'Record-backed reference.')
        self.assertEqual(result['main_calls'], 0)
        self.assertEqual([action['mode'] for action in started], ['check'])
        self.assertEqual(started[0]['request'], query)
        self.assertEqual(client.requests, [])

    def test_additional_tasks_opt_outs_quotes_files_and_precision_keep_normal_routing(self):
        for query in (
                'Give me a reference and prove the result.',
                'Give me a classical reference from memory only.',
                "Give me a classical reference, don't check online.",
                'Give me a classical reference without searching.',
                'Give me the exact reference for this theorem.',
                '"Give me a classical reference" is an example.',
                '> Give me a classical reference',
                'Give me a reference from https://example.com/'):
            with self.subTest(query=query):
                self.assertIsNone(automatic_reference_action(query))
        self.assertIsNone(automatic_reference_action('Give me a classical reference.', ['source.tex']))

    def test_explicit_mode_still_takes_precedence_over_automatic_check(self):
        client = Client([final('Survey complete.')])
        started = []
        def worker(action, child, save):
            started.append(action['mode'])
            return {'status': 'complete', 'answer': 'Survey.'}
        runner = ModeOrchestrator(Agent(client, Workspace('/tmp'), ctx=32768), worker, required_mode='literature')
        runner.run('Give me a classical reference on stochastic processes.')
        self.assertEqual(started, ['literature'])
