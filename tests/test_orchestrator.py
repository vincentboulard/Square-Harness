import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from mathagent.agent import Agent, AgentError
from mathagent.backends import _messages
from mathagent.literature import LiteratureTools
from mathagent.orchestrator import Orchestrator
from mathagent.tools import Workspace


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        yield from response


def final(text='Final answer', reason='stop', **stats):
    return [{'message': {'content': text}, 'done': True, 'done_reason': reason,
             'eval_count': stats.get('eval_count', 30),
             'prompt_eval_count': stats.get('prompt_eval_count', 100)}]


def action(mode='critic', request='Check the exact stated lemma.', **extra):
    return dict({'mode': mode, 'request': request, 'files': [], 'effort': 'low',
                 'reason': 'An independent check addresses the mathematical uncertainty.'}, **extra)


def call(name='delegate', args=None, call_id='tool_1'):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': action() if args is None else args}}


def calls(*items):
    return [{'message': {'tool_calls': list(items)}, 'done': True,
             'done_reason': 'tool_calls', 'eval_count': 60, 'prompt_eval_count': 100}]


class OrchestratorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.workspace = Workspace(tmp.name, allow_python=True)
        self.root = Path(tmp.name)
        self.children = []

    def worker(self, action, child, checkpoint):
        self.children.append(copy.deepcopy(child))
        child['job_id'] = 'job_' + str(len(self.children))
        checkpoint()
        return {'status': 'reviewed', 'answer': 'Use the exact hypothesis H.',
                'job_id': child['job_id'], 'warnings': ['Model review is not formal verification.']}

    def runner(self, responses, worker=None, **kwargs):
        client = Client(responses)
        agent = Agent(client, self.workspace, ctx=32768, predict=30000, think=True)
        return Orchestrator(agent, worker or self.worker, **kwargs), client

    def test_elementary_proof_is_one_direct_nonthinking_call(self):
        answer = 'If a >= 0, a*a >= 0. If a < 0, (-a)*(-a) >= 0 and equals a*a.'
        runner, client = self.runner([final(answer)])
        result = runner.run('Prove a^2 >= 0 for real a.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], answer)
        self.assertEqual(self.children, [])
        self.assertEqual(len(client.requests), 1)
        self.assertFalse(client.requests[0]['think'])
        self.assertEqual(client.requests[0]['options']['num_predict'], 8192)  # room for a full direct answer
        self.assertEqual(runner.agent.history[-1]['content'], answer)

    def test_parent_only_exposes_delegation_and_allowed_read_only_tools(self):
        runner, client = self.runner([final()])
        runner.run('Hello')
        names = {item['function']['name'] for item in client.requests[0]['tools']}
        self.assertEqual(names, {'delegate', 'read_result', 'list_files', 'read_file', 'search_text'})
        self.assertNotIn('write_file', names)
        self.assertNotIn('run_python', names)

    def test_worker_result_and_native_call_id_are_fed_to_parent(self):
        runner, client = self.runner([calls(call()), final()])
        result = runner.run('Check the lemma.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.children), 1)
        message = client.requests[1]['messages'][-1]
        self.assertEqual(message['tool_call_id'], 'tool_1')
        self.assertEqual(json.loads(message['content'])['result']['status'], 'reviewed')
        self.assertIn('Use the exact hypothesis H.', message['content'])
        self.assertEqual(_messages(runner.state['messages'])[-2]['tool_call_id'], 'tool_1')

    def test_adapts_dependent_next_objective_to_previous_result(self):
        runner, client = self.runner([
            calls(call('delegate', action('explore', 'Find the obstruction.'), 'search')),
            calls(call('delegate', action('critic', 'Check whether the obstruction needs hypothesis H.'), 'review')),
            final('The missing hypothesis H is essential.')])
        result = runner.run('Investigate this conjecture.')
        self.assertEqual(len(result['children']), 2)
        self.assertEqual(result['actions'], 2)
        self.assertIn('Use the exact hypothesis H.', client.requests[2]['messages'][-3]['content'])
        self.assertIn('Use the exact hypothesis H.', json.dumps(self.children[1]['context']['messages']))

    def test_original_hypotheses_history_and_attachments_never_replaced_by_objective(self):
        query = 'For every smooth u satisfying H1, H2 and boundary condition B, prove estimate E.'
        history = [{'role': 'user', 'content': 'H2 means the uniform condition, not pointwise.'},
                   {'role': 'assistant', 'content': 'The parameter ranges over the closed set K.'}]
        (self.root / 'lemma.tex').write_text('Full source')
        runner, client = self.runner([calls(call(args=action(request='Prove E.'))), final()])
        runner.run(query, files=['lemma.tex'], history=history)
        context = self.children[0]['context']
        self.assertEqual(context['query'], query)
        self.assertEqual(context['history'], history)
        self.assertEqual(context['files'], ['lemma.tex'])
        self.assertEqual(self.children[0]['action']['files'], ['lemma.tex'])
        self.assertEqual(client.requests[0]['messages'][1:3], history)
        self.assertIn(query, client.requests[0]['messages'][3]['content'])

    def test_invalid_action_can_be_corrected_without_starting_invalid_worker(self):
        runner, client = self.runner([
            calls(call(args=action(mode='unknown'))),
            calls(call(args=action(), call_id='corrected')), final()])
        result = runner.run('Check it.')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(result['actions'], 2)
        self.assertEqual(json.loads(client.requests[1]['messages'][-1]['content'])['status'], 'invalid_action')

    def test_long_request_is_rejected_explicitly_without_slicing_hypotheses(self):
        runner, client = self.runner([calls(call(args=action(request='H' * 8001))), final()])
        runner.run('A proof request')
        self.assertEqual(self.children, [])
        self.assertIn('no text was truncated', client.requests[1]['messages'][-1]['content'])
        self.assertEqual(len(runner.state['messages'][1]['tool_calls'][0]['function']['arguments']['request']), 8001)

    def test_forbidden_tools_are_not_executed(self):
        runner, client = self.runner([calls(call('write_file', {'path': 'evil.txt', 'content': 'changed'})), final()])
        runner.run('Hello')
        self.assertFalse((self.root / 'evil.txt').exists())
        self.assertIn('forbidden', client.requests[1]['messages'][-1]['content'])

    def test_read_source_excerpt_before_answer(self):
        (self.root / 'lemma.tex').write_text('Let H be the uniform condition.\n')
        runner, client = self.runner([
            calls(call('read_file', {'path': 'lemma.tex', 'start_line': 1, 'end_line': 1})), final()])
        runner.run('What is H in lemma.tex?', files=['lemma.tex'])
        self.assertIn('Let H be the uniform condition.', client.requests[1]['messages'][-1]['content'])
        self.assertEqual(self.children, [])

    def test_length_limited_tool_call_is_not_executed_or_completed(self):
        events = calls(call())
        events[0]['message']['content'] = 'Unfinished statement'
        events[0]['done_reason'] = 'length'
        runner, client = self.runner([events])
        result = runner.run('Do research')
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('output limit', result['warnings'][0])
        self.assertEqual(self.children, [])
        self.assertEqual(runner.agent.history, [])
        self.assertEqual(runner.state['messages'][-1]['role'], 'user')

    def test_missing_usage_and_incomplete_stream_are_unfinished(self):
        for response in ([{'message': {'content': 'Candidate'}, 'done': True, 'done_reason': 'stop'}],
                         [{'message': {'content': 'Partial'}}],
                         final('Candidate', eval_count=None),
                         final('Candidate', prompt_eval_count=None)):
            with self.subTest(response=response):
                runner, _ = self.runner([response])
                result = runner.run('Prove it')
                self.assertEqual(result['status'], 'incomplete')
                self.assertTrue(result['warnings'])

    def test_overreported_output_cap_is_unfinished(self):
        runner, _ = self.runner([final('Candidate', eval_count=8193)])
        result = runner.run('Prove it')
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('output allowance', result['warnings'][0])

    def test_input_overflow_preserves_exact_context_and_does_not_infer(self):
        runner, client = self.runner([])
        runner.agent.ctx = 8192
        history = [{'role': 'user', 'content': 'uniform ' * 3000}]
        query = 'At the critical time T, retain all the preceding hypotheses.'
        with self.assertRaisesRegex(AgentError, 'No mathematical context was truncated'):
            runner.run(query, history=history)
        self.assertEqual(client.requests, [])
        self.assertEqual(runner.state['history'], history)
        self.assertEqual(runner.state['query'], query)
        self.assertEqual(runner.state['messages'][0]['content'], history[0]['content'])

    def test_exact_tokenizer_allows_large_unicode_input_without_byte_pruning(self):
        runner, client = self.runner([final()])
        runner.agent.ctx = 8192
        client.count_input_tokens = lambda payload: 1000
        history = [{'role': 'user', 'content': '∀α∈ℝ ' * 2000}]
        runner.run('Use all of these hypotheses.', history=history)
        self.assertEqual(client.requests[0]['messages'][1]['content'], history[0]['content'])

    def test_read_result_reuses_completed_worker(self):
        runner, client = self.runner([
            calls(call()), calls(call('read_result', {'job_id': 'job_1'}, 'retrieve')), final()])
        runner.run('Check it')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(json.loads(client.requests[2]['messages'][-1]['content'])['result']['status'], 'reviewed')

    def test_interrupt_and_resume_existing_child_without_duplicate_job_or_call(self):
        snapshots, executions = [], []

        def worker(action, child, checkpoint):
            executions.append(child['id'])
            if child['job_id'] is None:
                child['job_id'] = 'durable_job'
                checkpoint()
                raise KeyboardInterrupt()
            self.assertEqual(child['job_id'], 'durable_job')
            return {'status': 'reviewed', 'answer': 'A resumed result', 'job_id': child['job_id']}

        runner, client = self.runner([calls(call()), final()], worker, checkpoint=snapshots.append)
        with self.assertRaises(KeyboardInterrupt):
            runner.run('Check the exact statement')
        state = copy.deepcopy(snapshots[-1])
        self.assertEqual(state['status'], 'interrupted')
        self.assertEqual(state['children'][0]['job_id'], 'durable_job')
        resumed = Orchestrator(runner.agent, worker, checkpoint=snapshots.append)
        result = resumed.run('Check the exact statement', state=state)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(executions[0], executions[1])
        self.assertEqual(len(result['children']), 1)
        self.assertEqual(result['actions'], 1)
        self.assertEqual(client.requests[1]['messages'][-1]['tool_call_id'], 'tool_1')

    def test_completed_child_is_not_reexecuted_if_interrupted_before_tool_append(self):
        interrupted = [False]
        snapshots = []

        def save(state):
            snapshots.append(state)
            if not interrupted[0] and state['children'] and state['children'][0]['status'] == 'complete':
                interrupted[0] = True
                raise KeyboardInterrupt()

        runner, _ = self.runner([calls(call()), final()], checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            runner.run('Check it')
        self.assertEqual(len(self.children), 1)
        resumed = Orchestrator(runner.agent, self.worker)
        result = resumed.run('Check it', state=snapshots[-1])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.children), 1)

    def test_worker_operational_error_preserves_pending_child_for_resume(self):
        def worker(action, child, checkpoint):
            child['job_id'] = 'existing_job'
            checkpoint()
            raise OSError('Transport interrupted')

        runner, _ = self.runner([calls(call())], worker)
        with self.assertRaisesRegex(OSError, 'Transport interrupted'):
            runner.run('Check it')
        self.assertEqual(runner.state['children'][0]['job_id'], 'existing_job')
        self.assertEqual(runner.state['pending_calls'][0]['status'], 'running')
        self.assertEqual(runner.state['status'], 'incomplete')

    def test_saved_permanent_failure_is_reused_before_appending_native_result(self):
        interrupted, snapshots, launches = [False], [], []
        failure = {'code': 'context_overflow', 'message': 'Exact context cannot fit.',
                   'retryable': False, 'scope': 'context', 'details': {}}

        def worker(action, child, checkpoint):
            launches.append(child['id'])
            return {'status': 'failed', 'job_id': 'failed_job', 'answer': '', 'error': failure}

        def save(state):
            snapshots.append(state)
            if not interrupted[0] and state['blockers'] and state['children'][0]['status'] == 'complete':
                interrupted[0] = True
                raise KeyboardInterrupt()

        runner, client = self.runner([calls(call(args=action('literature'))), final('The exact context cannot fit.')],
                                     worker, checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            runner.run('Read these sources')
        resumed = Orchestrator(runner.agent, worker)
        result = resumed.run('Read these sources', state=snapshots[-1])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(launches), 1)
        native = next(m for m in resumed.state['messages'] if m['role'] == 'tool')
        value = json.loads(native['content'])
        self.assertEqual(value['job_id'], 'failed_job')
        self.assertEqual(value['result']['error'], failure)
        self.assertFalse(resumed.state['force_final'])

    def test_saved_web_failure_reuses_same_native_transaction_after_crash(self):
        library = LiteratureTools(self.root, online=False)
        self.workspace = Workspace(self.root, literature=library)
        interrupted, snapshots = [False], []

        def save(state):
            snapshots.append(state)
            if not interrupted[0] and state['blockers'] and state['pending_calls'][0].get('result'):
                interrupted[0] = True
                raise KeyboardInterrupt()

        runner, client = self.runner([calls(call('open_url', {'url': 'https://example.org/source'}, 'url_call')),
                                      final('The page is unavailable offline.')], checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            runner.run('Read this source')
        original = snapshots[-1]['pending_calls'][0]['result']
        resumed = Orchestrator(runner.agent, self.worker)
        result = resumed.run('Read this source', state=snapshots[-1])
        self.assertEqual(result['status'], 'complete')
        native = next(m for m in resumed.state['messages'] if m['role'] == 'tool')
        self.assertEqual(native['tool_call_id'], 'url_call')
        self.assertEqual(json.loads(native['content']), original)
        self.assertEqual(original['error']['code'], 'offline_uncached')
        self.assertFalse(resumed.state['force_final'])
        self.assertEqual(library.stats['requests'], 0)

    def test_saved_web_result_and_character_charge_resume_atomically(self):
        library = LiteratureTools(self.root, online=True)
        self.workspace = Workspace(self.root, literature=library)
        with patch.object(library, '_fetch', return_value=(b'<html><body><p>Exact cached source passage.</p></body></html>',
                                                         'text/html', 'https://example.org/source')):
            page = json.loads(self.workspace.execute('open_url', {'url': 'https://example.org/source'}))
        initial_chars = library.stats['returned_chars']
        interrupted, snapshots = [False], []

        def save(state):
            state['web_budget'] = library.snapshot()
            snapshots.append(state)
            if not interrupted[0] and library.stats['returned_chars'] > initial_chars:
                self.assertIn('result', state['pending_calls'][0])
                interrupted[0] = True
                raise KeyboardInterrupt()

        runner, _ = self.runner([calls(call('read_page', {'page_id': page['page_id']}, 'read_call')),
                                 final('The cached passage is preserved.')], checkpoint=save)
        library.on_budget_change = runner._save
        with self.assertRaises(KeyboardInterrupt):
            runner.run('Read the cached page')
        charged = library.stats['returned_chars']
        self.assertGreater(charged, initial_chars)
        library.restore(snapshots[-1]['web_budget'])
        resumed = Orchestrator(runner.agent, self.worker)
        library.on_budget_change = resumed._save
        result = resumed.run('Read the cached page', state=snapshots[-1])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(library.stats['returned_chars'], charged)
        native = next(m for m in resumed.state['messages'] if m['role'] == 'tool')
        self.assertEqual(native['tool_call_id'], 'read_call')
        self.assertIn('Exact cached source passage.', json.loads(native['content'])['passage'])

    def test_independent_group_runs_concurrently_with_ordered_results(self):
        first_started = threading.Event()
        second_started = threading.Event()
        active, maximum = [0], [0]
        lock = threading.Lock()
        snapshots = []

        def worker(action, child, checkpoint):
            with lock:
                active[0] += 1
                maximum[0] = max(maximum[0], active[0])
            # Both child identifiers must exist in the durable checkpoint
            # before either callback starts execution.
            self.assertEqual(len(snapshots[-1]['children']), 2)
            self.assertTrue(all(item['child_id'] for item in snapshots[-1]['pending_calls']))
            if action['request'] == 'Independent A':
                first_started.set()
                self.assertTrue(second_started.wait(2))
                time.sleep(0.03)
            else:
                second_started.set()
                self.assertTrue(first_started.wait(2))
            with lock:
                active[0] -= 1
            return {'status': 'complete', 'answer': action['request'], 'job_id': child['id']}

        runner, client = self.runner([
            calls(call(args=action(request='Independent A'), call_id='A'),
                  call(args=action(request='Independent B'), call_id='B')), final()],
            worker, checkpoint=snapshots.append, concurrency=2)
        result = runner.run('Check two independent questions')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(maximum[0], 2)
        tool_messages = [item for item in client.requests[1]['messages'] if item['role'] == 'tool']
        self.assertEqual([item['tool_call_id'] for item in tool_messages], ['A', 'B'])
        self.assertEqual([json.loads(item['content'])['result']['answer'] for item in tool_messages],
                         ['Independent A', 'Independent B'])

    def test_parallel_group_bound_and_completed_child_resume(self):
        barrier = threading.Barrier(2)
        seen = []
        active, maximum = [0], [0]
        lock = threading.Lock()
        interrupted = [False]

        def worker(action, child, checkpoint):
            with lock:
                seen.append(action['request'])
                active[0] += 1
                maximum[0] = max(maximum[0], active[0])
            if action['request'] in {'A', 'B'} and not interrupted[0]:
                barrier.wait(2)
            if action['request'] == 'B' and child['job_id'] is None:
                child['job_id'] = 'resume_B'
                checkpoint()
                interrupted[0] = True
                with lock:
                    active[0] -= 1
                raise KeyboardInterrupt()
            time.sleep(0.02)
            with lock:
                active[0] -= 1
            return {'status': 'complete', 'answer': action['request'], 'job_id': child['job_id'] or child['id']}

        runner, client = self.runner([
            calls(*(call(args=action(request=item), call_id=item) for item in 'ABCD')), final()],
            worker, concurrency=2)
        with self.assertRaises(KeyboardInterrupt):
            runner.run('Four independent objectives')
        self.assertEqual(maximum[0], 2)
        self.assertEqual(len(runner.state['children']), 4)
        completed = [item['action']['request'] for item in runner.state['children'] if item['status'] == 'complete']
        self.assertEqual(set(completed), {'A', 'C', 'D'})
        resumed = Orchestrator(runner.agent, worker, concurrency=2)
        result = resumed.run('Four independent objectives', state=runner.state)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(seen.count('A'), 1)
        self.assertEqual(seen.count('B'), 2)
        self.assertEqual(seen.count('C'), 1)
        self.assertEqual(seen.count('D'), 1)
        self.assertEqual([item['tool_call_id'] for item in client.requests[-1]['messages'] if item['role'] == 'tool'],
                         list('ABCD'))

    def test_action_budget_provides_final_synthesis_without_executing_extra_calls(self):
        runner, client = self.runner([
            calls(call(call_id='A'), call(call_id='B')), final()], max_actions=1)
        result = runner.run('Check it')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(result['actions'], 1)
        self.assertNotIn('tools', client.requests[1])
        self.assertEqual(client.requests[1]['options']['num_predict'], 8192)
        self.assertIn('budget_exhausted', client.requests[1]['messages'][-1]['content'])

    def test_worker_incomplete_status_is_preserved_as_evidence(self):
        def worker(action, child, checkpoint):
            return {'status': 'incomplete', 'answer': 'An unproved reduction',
                    'warnings': ['The key estimate is unproved.']}

        runner, client = self.runner([calls(call()), final('The key estimate remains unproved.')], worker)
        runner.run('Prove it')
        result = json.loads(client.requests[1]['messages'][-1]['content'])['result']
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('unproved', result['warnings'][0])

    def test_standard_reference_disposition_closes_lookup_without_another_tool(self):
        worker_answer = ('Morris W. Hirsch, Differential Topology. Publisher and edition were not verified. '
                         'Chapter 4 is an unconfirmed guess.')
        def lookup(action, child, checkpoint):
            self.children.append(copy.deepcopy(child))
            child['job_id'] = 'reference_1'
            checkpoint()
            return {'status': 'answered', 'answer': worker_answer,
                    'disposition': 'answer_with_qualification', 'job_id': 'reference_1'}
        # No synthesis reply is supplied: the qualified worker answer is final.
        runner, client = self.runner([calls(call(args=action(mode='check')))], lookup)
        result = runner.run('Find a classical reference for the tubular neighborhood theorem.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], worker_answer)
        self.assertEqual(len(self.children), 1)
        self.assertEqual(result['actions'], 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(runner.state['main_calls'], 1)
        self.assertEqual(runner.state['answer_source'], {'kind': 'worker_result', 'job_id': 'reference_1'})
        self.assertEqual(runner.agent.history[-1]['content'], worker_answer)
        self.assertNotIn('Springer', result['answer'])
        self.assertNotIn('1976', result['answer'])
        self.assertIn('Publisher and edition were not verified.', result['answer'])

    def test_saved_qualified_reference_returns_without_model_or_worker_reexecution(self):
        answer = 'Brezis is a suggested standard reference; the precise place is unconfirmed.'
        snapshots, executions, interrupted = [], [], [False]
        query = 'Find a standard reference for Neumann regularity.'

        def lookup(action, child, checkpoint):
            executions.append(child['id'])
            return {'status': 'answered', 'answer': answer, 'job_id': 'saved_reference',
                    'disposition': 'answer_with_qualification'}

        def save(state):
            snapshots.append(state)
            if (not interrupted[0] and state['children'] and state['children'][0]['status'] == 'complete'
                    and any(m['role'] == 'tool' for m in state['messages'])):
                interrupted[0] = True
                raise KeyboardInterrupt()

        runner, client = self.runner([calls(call(args=action(mode='check'), call_id='lookup_call'))],
                                     lookup, checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            runner.run(query)
        saved = copy.deepcopy(snapshots[-1])
        [native] = [m for m in saved['messages'] if m['role'] == 'tool']
        self.assertEqual(native['tool_call_id'], 'lookup_call')
        resumed = Orchestrator(runner.agent, lookup)
        result = resumed.run(query, state=saved)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], answer)
        self.assertEqual(len(executions), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(result['main_calls'], 1)
        self.assertEqual([m for m in resumed.state['messages'] if m['role'] == 'tool'], [native])
        self.assertEqual(resumed.state['answer_source'], {'kind': 'worker_result', 'job_id': 'saved_reference'})
        completed = copy.deepcopy(resumed.state)
        again = Orchestrator(runner.agent, lookup)
        self.assertEqual(again.run(query, state=completed), result)
        self.assertEqual(again.state['messages'], completed['messages'])
        self.assertEqual(len(executions), 1)
        self.assertEqual(len(client.requests), 1)

    def test_interrupted_qualified_answer_emission_resumes_without_duplicate_native_answer(self):
        answer = 'Hirsch is a suggested standard reference; its chapter is unconfirmed.'
        query = 'Find a standard reference for the tubular neighborhood theorem.'
        executions, interrupted = [], [False]

        def lookup(action, child, checkpoint):
            executions.append(child['id'])
            return {'status': 'answered', 'answer': answer, 'job_id': 'emission_reference',
                    'disposition': 'answer_with_qualification'}

        def emit(kind, value):
            if kind == 'text' and value == answer and not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt()

        runner, client = self.runner([calls(call(args=action(mode='check'), call_id='lookup_call'))],
                                     lookup, emit=emit)
        with self.assertRaises(KeyboardInterrupt):
            runner.run(query)
        saved = copy.deepcopy(runner.state)
        [native] = [m for m in saved['messages'] if m['role'] == 'tool']
        self.assertEqual(saved['messages'][-1]['content'], answer)
        resumed = Orchestrator(runner.agent, lookup)
        result = resumed.run(query, state=saved)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], answer)
        self.assertEqual(len(executions), 1)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual([m for m in resumed.state['messages'] if m['role'] == 'tool'], [native])
        self.assertEqual([m['content'] for m in resumed.state['messages']
                          if m['role'] == 'assistant' and m.get('content') == answer], [answer])
        self.assertEqual(resumed.state['answer_source'], {'kind': 'worker_result', 'job_id': 'emission_reference'})
        self.assertEqual(resumed.agent.history[-1]['content'], answer)
        self.assertEqual(result['main_calls'], 1)

    def test_standard_reference_cannot_launch_two_workers_for_one_lookup(self):
        runner, client = self.runner([calls(call(args=action(mode='check'), call_id='first'),
            call(args=action(mode='check', request='Now find the exact theorem number.'), call_id='retry')), final()])
        result = runner.run('Find a standard reference for this theorem.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.children), 1)
        errors = [json.loads(m['content']) for m in runner.state['messages'] if m['role'] == 'tool']
        self.assertTrue(any(r.get('status') == 'invalid_action' for r in errors))

    def test_standard_reference_does_not_stop_independent_requested_proof(self):
        def worker(action, child, checkpoint):
            value = self.worker(action, child, checkpoint)
            if action['mode'] == 'check':
                value['disposition'] = 'answer_with_qualification'
            return value
        runner, client = self.runner([calls(call(args=action(mode='check'))),
            calls(call(args=action(mode='prove'), call_id='proof')), final()], worker)
        result = runner.run('Find a reference and prove the lemma.')
        self.assertEqual(result['status'], 'complete')
        self.assertIn('tools', client.requests[1])
        self.assertEqual([c['action']['mode'] for c in self.children], ['check', 'prove'])

    def test_standard_reference_does_not_skip_an_additional_request_in_another_sentence(self):
        requests = [
            'Find a standard reference for this theorem. Include a complete proof.',
            'Find a standard reference for this theorem. Explain why it applies.',
            'Find a standard reference for this theorem. Check the hypotheses.',
        ]
        for query in requests:
            with self.subTest(query=query):
                worker_answer = 'A standard reference is suggested; the precise place remains unconfirmed.'
                parent_answer = 'Here is the requested proof or application review, with the reference qualified.'

                def lookup(action, child, checkpoint):
                    return {'status': 'answered', 'answer': worker_answer, 'job_id': 'qualified_reference',
                            'disposition': 'answer_with_qualification'}

                runner, client = self.runner([calls(call(args=action(mode='check'))), final(parent_answer)], lookup)
                result = runner.run(query)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['answer'], parent_answer)
                self.assertEqual(len(client.requests), 2)
                self.assertEqual(result['main_calls'], 2)
                self.assertNotIn('answer_source', runner.state)
                self.assertIn('tools', client.requests[1])
                [native] = [m for m in client.requests[1]['messages'] if m['role'] == 'tool']
                self.assertEqual(json.loads(native['content'])['result']['answer'], worker_answer)

    def test_lookup_at_action_limit_replaces_continuation_promise_with_answer(self):
        promise = "The check worker couldn't confirm the exact section number. Let me try to find it directly from an open-access source."
        runner, client = self.runner([
            calls(call(args=action(mode='check'), call_id='A'), call(args=action(mode='check'), call_id='B')),
            calls(*(call('list_files', {}, str(i)) for i in range(4))),
            final(promise), final('Hirsch is located; the theorem number remains unconfirmed.')], max_actions=6)
        result = runner.run('Find references for result A and result B.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['actions'], 6)
        self.assertEqual(len(self.children), 2)
        self.assertNotIn('tools', client.requests[-1])
        self.assertIn('FINAL ANSWER REQUIRED NOW', client.requests[-1]['messages'][0]['content'])
        self.assertEqual(client.requests[-1]['messages'][-1]['role'], 'user')
        self.assertIn('CONTROLLER RECOVERY REQUEST', client.requests[-1]['messages'][-1]['content'])
        self.assertIn('without read/cited evidence', client.requests[-1]['messages'][-1]['content'])
        self.assertFalse(any('CONTROLLER RECOVERY REQUEST' in m.get('content', '') for m in runner.state['messages']))
        self.assertTrue(runner.state['continuation_final_retried'])
        self.assertIn('unconfirmed', result['answer'])

    def test_continuation_with_actions_available_carries_out_next_step(self):
        runner, client = self.runner([calls(call()), final('Let me verify the exact reference.'),
                                     calls(call('read_result', {'job_id': 'job_1'}, 'read')), final('The reference is unconfirmed.')])
        result = runner.run('Verify this reference.')
        self.assertEqual(result['status'], 'complete')
        self.assertIn('tools', client.requests[2])
        self.assertEqual(len(self.children), 1)

    def test_repeated_continuation_remains_incomplete_across_resume(self):
        runner, client = self.runner([calls(call()), final('Je vais chercher la référence.'),
                                     final('Let me search for the source.'), final('I will verify the source.')])
        result = runner.run('Find a reference.')
        self.assertEqual(result['status'], 'incomplete')
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Find a reference.', state=runner.state)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(len(self.children), 1)

    def test_opening_plan_followed_by_answer_is_not_retried(self):
        answer = 'Let me verify the reference.\n\nIt is located, but the theorem number is unconfirmed.'
        runner, client = self.runner([calls(call()), final(answer)])
        self.assertEqual(runner.run('Find a reference.')['status'], 'complete')
        self.assertEqual(len(client.requests), 2)

    def test_new_turn_allows_lookup_beyond_six_actions(self):
        runner, client = self.runner([calls(*(call('list_files', {}, str(i)) for i in range(6))),
                                     calls(call('list_files', {}, 'seventh')), final()])
        result = runner.run('Inspect sources.')
        self.assertEqual(result['actions'], 7)
        self.assertEqual(result['status'], 'complete')
        self.assertIn('tools', client.requests[1])

    def test_completed_run_is_idempotent_and_resume_rejects_changed_context(self):
        runner, client = self.runner([final()])
        result = runner.run('Original question')
        resumed = Orchestrator(runner.agent, self.worker)
        self.assertEqual(resumed.run('Original question', state=runner.state), result)
        self.assertEqual(len(client.requests), 1)
        with self.assertRaisesRegex(ValueError, 'original query'):
            resumed.run('Changed question', state=runner.state)

    def test_ollama_generated_tool_ids_are_distinct_across_conversation_turns(self):
        runner, client = self.runner([calls(call(call_id=None)), final(), calls(call(call_id=None)), final()])
        runner.run('First question')
        first_id = runner.state['messages'][1]['tool_calls'][0]['id']
        history = copy.deepcopy(runner.agent.history)
        next_turn = Orchestrator(runner.agent, self.worker)
        result = next_turn.run('Second question', history=history)
        self.assertEqual(result['status'], 'complete')
        current_call = next(message for message in next_turn.state['messages'][len(history):]
                            if message.get('tool_calls'))
        second_id = current_call['tool_calls'][0]['id']
        self.assertNotEqual(first_id, second_id)
        _messages(next_turn.state['messages'])

    def test_worker_adapter_cannot_omit_result_status(self):
        runner, _ = self.runner([calls(call())], lambda action, child, checkpoint: {'answer': 'Candidate'})
        with self.assertRaisesRegex(AgentError, 'explicit result status'):
            runner.run('Check it')
        self.assertEqual(runner.state['children'][0]['status'], 'running')

    def _capped_proof(self, limit=8192, following=()):
        def worker(action, child, checkpoint):
            self.children.append(copy.deepcopy(child))
            return {'status': 'candidate_complete', 'answer': 'The saved proof uses all hypotheses H1 and H2.',
                    'job_id': 'saved_proof', 'artifact': 'proof-artifact.md'}

        runner, client = self.runner([
            calls(call(args=action(mode='prove', request='Prove E under H1 and H2.'))),
            final('UNREVIEWED CUT-OFF FRAGMENT', reason='length', eval_count=limit),
            *following], worker)
        result = runner.run('Prove E under H1 and H2.')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(runner.state['children'][0]['result']['status'], 'candidate_complete')
        self.assertEqual(runner.state['children'][0]['job_id'], 'saved_proof')
        return runner, client

    def test_empty_post_worker_answer_gets_one_synthesis_retry_without_repeating_worker(self):
        runner, client = self.runner([calls(call(args=action(mode='prove'))), final(''), final('Saved result with its unresolved obligations.')])
        result = runner.run('Prove the original statement.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(len(client.requests), 3)
        self.assertNotIn('tools', client.requests[-1])
        self.assertTrue(runner.state['empty_final_retried'])

    def test_repeated_empty_synthesis_is_bounded_across_resume(self):
        runner, client = self.runner([calls(call(args=action(mode='prove'))), final(''), final(''), final('')])
        result = runner.run('Prove the original statement.')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(client.requests), 3)
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove the original statement.', state=runner.state)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(len(self.children), 1)

    def test_post_worker_full_answer_has_room_and_retains_adaptive_tools(self):
        runner, client = self.runner([
            calls(call(args=action(mode='prove'), call_id='proof')),
            calls(call(args=action(mode='critic', request='Check the assembly of the saved proof.'), call_id='assembly')),
            final('The reviewed argument is complete.')])
        result = runner.run('Give a complete proof.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(client.requests[0]['options']['num_predict'], 8192)
        self.assertEqual(client.requests[1]['options']['num_predict'], 8192)
        self.assertIn('delegate', {item['function']['name'] for item in client.requests[1]['tools']})
        self.assertEqual(client.requests[2]['options']['num_predict'], 8192)
        self.assertEqual(len(self.children), 2)

    def test_legacy_length_resume_regenerates_final_without_rerunning_saved_proof(self):
        runner, client = self._capped_proof(limit=1024, following=[final('Complete replacement proof.')])
        old = copy.deepcopy(runner.state)
        # Reproduce the deployed checkpoint format from before output_limit
        # and final_recovery were recorded. No user data is touched.
        old['last_call'] = {'status': 'incomplete', 'partial': {'role': 'assistant',
            'content': 'UNREVIEWED CUT-OFF FRAGMENT'},
            'stats': {'done_reason': 'length', 'eval_count': 1024, 'prompt_eval_count': 100}}
        client.remaining_tokens = 17551
        original_messages = copy.deepcopy(old['messages'])
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove E under H1 and H2.', state=old)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], 'Complete replacement proof.')
        self.assertEqual(len(self.children), 1)
        request = client.requests[-1]
        self.assertNotIn('tools', request)
        self.assertEqual(request['options']['num_predict'], 8192)
        self.assertEqual(request['messages'][1:], original_messages)
        self.assertNotIn('UNREVIEWED CUT-OFF FRAGMENT', json.dumps(request['messages']))
        self.assertEqual(resumed.state['children'][0]['job_id'], 'saved_proof')
        self.assertEqual(result['actions'], 1)
        _messages(resumed.state['messages'])

    def test_capped_final_resume_increases_allowance_without_stitching_partial(self):
        runner, client = self._capped_proof(following=[final('Fresh complete answer.')])
        original_messages = copy.deepcopy(runner.state['messages'])
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove E under H1 and H2.', state=runner.state)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(client.requests[-1]['options']['num_predict'], 16384)
        self.assertNotIn('tools', client.requests[-1])
        self.assertEqual(client.requests[-1]['messages'][1:], original_messages)
        self.assertEqual(resumed.state['last_call']['output_limit'], 16384)
        self.assertEqual(len(self.children), 1)

    def test_synthesis_recovery_preserves_expanded_allowance_after_interruption(self):
        runner, client = self._capped_proof(following=[KeyboardInterrupt(), final('Completed after pause.')])
        recovering = Orchestrator(runner.agent, runner.delegate)
        with self.assertRaises(KeyboardInterrupt):
            recovering.run('Prove E under H1 and H2.', state=runner.state)
        self.assertEqual(client.requests[-1]['options']['num_predict'], 16384)
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove E under H1 and H2.', state=recovering.state)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(client.requests[-1]['options']['num_predict'], 16384)
        self.assertNotIn('tools', client.requests[-1])
        self.assertEqual(len(self.children), 1)

    def test_synthesis_recovery_cannot_launch_unexpected_tool_calls(self):
        runner, client = self._capped_proof(following=[calls(call(args=action(), call_id='forbidden_retry'))])
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove E under H1 and H2.', state=runner.state)
        self.assertNotIn('tools', client.requests[-1])
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(result['actions'], 1)

    def test_resume_cannot_dispatch_forbidden_recovery_action_from_crash_window(self):
        runner, client = self._capped_proof(following=[
            calls(call(args=action(), call_id='forbidden_retry')), final('Recovered replacement proof.')])
        interrupted = [False]

        def save(state):
            # The native action has been saved but its disabled-tools result
            # has not yet been appended. A crash here must not grant tools on
            # the following Resume.
            if (not interrupted[0] and state['last_call'].get('final_recovery')
                    and any(p['call']['id'] == 'forbidden_retry' and p['status'] == 'pending'
                            for p in state['pending_calls'])):
                interrupted[0] = True
                raise KeyboardInterrupt()

        recovering = Orchestrator(runner.agent, runner.delegate, checkpoint=save)
        with self.assertRaises(KeyboardInterrupt):
            recovering.run('Prove E under H1 and H2.', state=runner.state)
        self.assertEqual(len(self.children), 1)
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove E under H1 and H2.', state=recovering.state)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['actions'], 1)
        self.assertEqual(len(self.children), 1)
        self.assertEqual(resumed.state['children'][0]['job_id'], 'saved_proof')
        forbidden_result = next(message for message in client.requests[-1]['messages']
                                if message.get('tool_call_id') == 'forbidden_retry')
        self.assertEqual(json.loads(forbidden_result['content'])['status'], 'budget_exhausted')

    def test_recovery_fits_output_to_context_without_clipping_inputs(self):
        runner, client = self._capped_proof(following=[final('Complete proof in the available room.')])
        client.count_input_tokens = lambda payload: runner.agent.ctx - 512 - 12000
        original = copy.deepcopy(runner.state['messages'])
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Prove E under H1 and H2.', state=runner.state)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(client.requests[-1]['options']['num_predict'], 12000)
        self.assertEqual(client.requests[-1]['messages'][1:], original)

    def test_no_identical_capped_retry_when_context_or_remaining_budget_cannot_grow(self):
        for restricted in ('context', 'tokens'):
            with self.subTest(restricted=restricted):
                runner, client = self._capped_proof()
                if restricted == 'context':
                    client.count_input_tokens = lambda payload: runner.agent.ctx - 512 - 8192
                else:
                    client.remaining_tokens = 8192
                attempted_before = len(client.requests)
                resumed = Orchestrator(runner.agent, runner.delegate)
                with self.assertRaises(AgentError):
                    resumed.run('Prove E under H1 and H2.', state=runner.state)
                self.assertEqual(len(client.requests), attempted_before)
                self.assertEqual(resumed.state['children'][0]['job_id'], 'saved_proof')

    def test_recovery_does_not_repeat_maximum_capped_allowance(self):
        runner, client = self._capped_proof(following=[final('Still cut off', reason='length', eval_count=16384)])
        first_retry = Orchestrator(runner.agent, runner.delegate)
        first = first_retry.run('Prove E under H1 and H2.', state=runner.state)
        self.assertEqual(first['status'], 'incomplete')
        self.assertEqual(client.requests[-1]['options']['num_predict'], 16384)
        attempted_before = len(client.requests)
        second_retry = Orchestrator(runner.agent, runner.delegate)
        with self.assertRaises(AgentError):
            second_retry.run('Prove E under H1 and H2.', state=first_retry.state)
        self.assertEqual(len(client.requests), attempted_before)
        self.assertEqual(len(self.children), 1)

    def test_capped_action_is_not_mistaken_for_a_final_answer_recovery(self):
        event = calls(call(call_id='capped_action'))
        event[0]['message']['content'] = 'I will delegate a check.'
        event[0]['done_reason'] = 'length'
        event[0]['eval_count'] = 1024
        runner, client = self.runner([event, calls(call(call_id='valid_action')), final()])
        result = runner.run('Check this uncertain claim.')
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(self.children), 0)
        resumed = Orchestrator(runner.agent, runner.delegate)
        result = resumed.run('Check this uncertain claim.', state=runner.state)
        self.assertEqual(result['status'], 'complete')
        self.assertIn('tools', client.requests[1])
        self.assertEqual(len(self.children), 1)


if __name__ == '__main__':
    unittest.main()
