"""Reference requests must reach one scholarly check without parent inventions."""
import copy
import json
import unittest

from mathagent.agent import Agent
from mathagent.mode_calls import ModeOrchestrator, automatic_reference_action
from mathagent.orchestrator import Orchestrator
from mathagent.tools import Workspace
from tests.test_orchestrator import Client, action, call, calls, final


class ReferenceRoutingTests(unittest.TestCase):
    def runner(self, responses=(), worker_result=None, **kwargs):
        started = []
        value = worker_result or {'status': 'answered', 'answer': 'Verified scholarly reference.',
                                  'disposition': 'answer_with_qualification', 'job_id': 'lookup'}
        def worker(operation, child, save):
            started.append(copy.deepcopy(child))
            return copy.deepcopy(value)
        client = Client(responses)
        runner = ModeOrchestrator(Agent(client, Workspace('/tmp'), ctx=32768), worker, **kwargs)
        return runner, client, started

    def test_articles_books_and_conjunctive_topics_get_one_exact_request(self):
        queries = [
            'Give me an article on reflected Brownian motion in Lipschitz domains.',
            'a classical book on Maliavin calculus',
            'Give me a good book for Sobolev injections on bounded domain, and application on heat equation on such domain (compact resolvent, etc)',
            'What is a good reference on stochastic processes?',
            'Could you recommend a paper about observability and control?',
            'Ok give me a classical reference about PDE please',
            'Recommend a reference for Brownian Motion and Stochastic Calculus.',
            'Donne-moi une bonne reference sur les espaces de Sobolev et les equations paraboliques.',
        ]
        for query in queries:
            with self.subTest(query=query):
                runner, client, started = self.runner()
                result = runner.run(query)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['answer'], 'Verified scholarly reference.')
                self.assertEqual(client.requests, [])
                self.assertEqual(len(started), 1)
                self.assertEqual(started[0]['action']['mode'], 'check')
                self.assertEqual(started[0]['action']['request'], query)
                self.assertEqual(started[0]['context']['query'], query)

    def test_reference_followup_retains_exact_original_hypotheses_and_history(self):
        history = [{'role': 'user', 'content': 'Let Omega be a bounded Lipschitz domain. Is reflected Brownian motion well defined?'},
                   {'role': 'assistant', 'content': 'The distinction between weak and strong solutions matters.'}]
        query = 'Can you give me an article for that?'
        runner, client, started = self.runner()
        runner.run(query, history=history)
        self.assertEqual(client.requests, [])
        self.assertEqual(started[0]['action']['request'], query)
        self.assertEqual(started[0]['context']['history'], history)
        self.assertEqual(started[0]['context']['query'], query)
        self.assertNotIn('smooth', started[0]['action']['request'])

    def test_topic_only_followup_continues_a_reference_request_without_guessing_context(self):
        history = [{'role': 'user', 'content': 'Give me an article on reflected Brownian motion in smooth domains.'},
                   {'role': 'assistant', 'content': 'A verified source for smooth domains.'}]
        query = 'And for Lipschitz domains?'
        runner, client, started = self.runner()
        result = runner.run(query, history=history)
        self.assertEqual(result['answer'], 'Verified scholarly reference.')
        self.assertEqual(client.requests, [])
        self.assertEqual(started[0]['action']['request'], query)
        self.assertEqual(started[0]['context']['history'], history)
        self.assertIsNone(automatic_reference_action(query))
        self.assertIsNone(automatic_reference_action(query, history=[{'role': 'user', 'content': 'Prove existence for smooth domains.'}]))
        self.assertIsNone(automatic_reference_action('And for Lipschitz domains, prove existence?', history=history))
        self.assertIsNone(automatic_reference_action('What about the proof?', history=history))
        self.assertIsNone(automatic_reference_action('What about a counterexample?', history=history))

    def test_independent_work_opt_outs_and_pinned_sources_keep_normal_routing(self):
        queries = [
            'Give me a reference and prove the result.',
            'Find a reference and a complete proof.',
            'Give me a reference and an explanation.',
            'Give me an article. Explain the proof.',
            'Find a standard reference for this theorem. Include a complete proof.',
            'Prove the theorem and give me a reference.',
            'Explain this book.',
            'This book is difficult.',
            'My manuscript has twenty references.',
            'Find a reference and then prove the result.',
            'Find a reference and also explain the proof.',
            'Give me a book. Check the hypotheses.',
            'Give me a reference from memory only.',
            "Give me an article, don't search online.",
            'Recommend a book without browsing.',
            'Give me the exact reference and page for this theorem.',
            'Give me a reference for Lemma 2 in source.tex.',
            'Give me a reference from https://example.com/paper.pdf.',
            '> Give me an article on reflected Brownian motion.',
            '"Give me a reference" is an example sentence.',
        ]
        for query in queries:
            with self.subTest(query=query):
                self.assertIsNone(automatic_reference_action(query))
        self.assertIsNone(automatic_reference_action('Give me an article for this.', ['source.tex']))

    def test_no_search_preference_persists_until_human_changes_it(self):
        history = [{'role': 'user', 'content': 'For this conversation, do not search online.'},
                   {'role': 'assistant', 'content': 'I will work from recall.'}]
        query = 'Give me an article for reflected Brownian motion in Lipschitz domains.'
        self.assertIsNone(automatic_reference_action(query, history=history))
        runner, client, started = self.runner([final('An explicitly unverified memory suggestion.')])
        result = runner.run(query, history=history)
        self.assertEqual(started, [])
        self.assertIn('unverified', result['answer'])
        allowed = automatic_reference_action(query + ' You can search online.', history=history)
        self.assertIsNotNone(allowed)
        self.assertEqual(allowed['request'], query + ' You can search online.')

    def test_compound_book_worker_answer_cannot_acquire_unchecked_showalter_supplement(self):
        query = 'Give me a good book for Sobolev injections on bounded domain, and application on heat equation on such domain (compact resolvent, etc)'
        answer = 'The verified book covers Sobolev spaces; coverage of the requested heat equation applications remains unconfirmed.'
        runner, client, started = self.runner([final('Also use Showalter, Equations of Evolution, Springer, 1984.')],
            {'status': 'answered', 'answer': answer, 'disposition': 'answer_with_qualification'})
        result = runner.run(query)
        self.assertEqual(result['answer'], answer)
        self.assertNotIn('Showalter', result['answer'])
        self.assertEqual(client.requests, [])
        self.assertEqual(started[0]['action']['request'], query)

    def test_explicit_literature_mode_is_not_replaced_by_check(self):
        runner, client, started = self.runner([final('Survey from the worker.')], required_mode='literature')
        runner.run('Give me an article about heat kernels and spectral gaps.')
        self.assertEqual([c['action']['mode'] for c in started], ['literature'])
        self.assertEqual(len(client.requests), 1)

    def test_partial_lookup_answer_cannot_be_replaced_by_unchecked_parent_reference(self):
        answer = 'No suitable article was confirmed; the available source has stronger boundary hypotheses.'
        runner, client, started = self.runner([final('Unchecked invented reference.')],
            {'status': 'partial', 'answer': answer, 'disposition': 'answer_with_qualification'})
        result = runner.run('Find an article on reflected Brownian motion in Lipschitz domains.')
        self.assertEqual(result['answer'], answer)
        self.assertEqual(client.requests, [])
        self.assertEqual(len(started), 1)

    def test_provider_failure_without_answer_has_no_parent_wikipedia_escape(self):
        runner, client, started = self.runner([final('Unchecked invented reference.')],
            {'status': 'failed', 'error': {'message': 'The scholarly provider timed out.', 'retryable': True}})
        result = runner.run('Give me a book about stochastic processes.')
        self.assertIn('did not produce a verified reference', result['answer'])
        self.assertIn('provider timed out', result['answer'])
        self.assertNotIn('Wikipedia', result['answer'])
        self.assertEqual(client.requests, [])
        self.assertEqual(len(started), 1)

    def test_parallel_parent_cannot_add_a_literature_worker_to_bypass_one_reference_check(self):
        started = []
        def worker(operation, child, save):
            started.append(operation['mode'])
            return {'status': 'partial', 'answer': 'No suitable source confirmed.',
                    'disposition': 'answer_with_qualification'}
        client = Client([calls(call(args=action(mode='check')), call(args=action(mode='literature'), call_id='bypass'))])
        runner = Orchestrator(Agent(client, Workspace('/tmp'), ctx=32768), worker)
        result = runner.run('Give me an article on reflected Brownian motion.')
        self.assertEqual(result['answer'], 'No suitable source confirmed.')
        self.assertEqual(started, ['check'])
        self.assertEqual(len(runner.state['children']), 1)

    def test_saved_parent_batch_cannot_open_wikipedia_after_finished_check(self):
        started, opened = [], []
        def worker(operation, child, save):
            started.append(operation)
            return {'status': 'partial', 'answer': 'No suitable source confirmed.',
                    'disposition': 'answer_with_qualification'}
        client = Client([calls(call(args=action(mode='check')), call('open_url',
            {'url': 'https://en.wikipedia.org/wiki/Brownian_motion'}, 'fallback'))])
        workspace = Workspace('/tmp')
        workspace.execute = lambda *args, **kwargs: opened.append(args) or json.dumps({'title': 'Wikipedia'})
        runner = Orchestrator(Agent(client, workspace, ctx=32768), worker)
        result = runner.run('Give me an article on reflected Brownian motion.')
        self.assertEqual(result['answer'], 'No suitable source confirmed.')
        self.assertEqual(opened, [])
        self.assertEqual(len(started), 1)
        outcomes = [json.loads(m['content']) for m in runner.state['messages'] if m['role'] == 'tool']
        self.assertEqual(outcomes[-1]['status'], 'invalid_action')
        self.assertIn('saved evidence', outcomes[-1]['error'])


if __name__ == '__main__':
    unittest.main()
