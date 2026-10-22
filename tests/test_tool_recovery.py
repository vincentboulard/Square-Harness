"""Invalid completed tool batches recover once without execution or lost usage."""
import copy
import io
import json
import unittest
from unittest.mock import patch

from mathagent.agent import Agent, AgentError
from mathagent.assistant_budget import BudgetClient
from mathagent.backends import OpenAICompatible, ToolCallError, _messages
from mathagent.gui import store
from mathagent.orchestrator import Orchestrator
from tests import test_assistant_gui as assistant_fixture
from tests.test_assistant_gui import delegation
from tests.test_backends import delta, finish, sse, usage
from tests.test_gui import text, GuiCase
from tests.test_gui_features import RECALL
from tests.test_orchestrator import action


def malformed(arguments='{"mode":"check","request":"Evans, Evans', output=37):
    return sse(delta(tool_calls=[{'index': 0, 'id': 'bad', 'type': 'function',
                                'function': {'name': 'delegate', 'arguments': arguments}}]),
               finish('tool_calls'), usage(output=output, prompt=1000))


class ToolRecoveryTests(unittest.TestCase):
    def runner(self, replies, worker=None, checkpoint=None):
        client = OpenAICompatible()
        client.count_input_tokens = lambda payload: 1000
        requests = []

        def request(endpoint, data):
            requests.append(copy.deepcopy(data))
            reply = replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return io.BytesIO(reply)

        client.request = request
        budget = BudgetClient(client)
        agent = Agent(budget, None, model='test', ctx=32768)
        started = []

        def delegate(objective, child, save):
            started.append(objective)
            return {'status': 'complete', 'answer': 'Evans, Partial Differential Equations.',
                    'disposition': 'answer_with_qualification', 'job_id': 'one_check'}

        runner = Orchestrator(agent, worker or delegate, checkpoint=checkpoint)
        return runner, requests, started, budget

    def corrected(self):
        return sse(delta(tool_calls=[{'index': 0, 'id': 'corrected', 'type': 'function',
                                     'function': {'name': 'delegate', 'arguments': json.dumps(
                                         action('check', 'Find one classical PDE reference.'))}}]),
                   finish('tool_calls'), usage(output=60, prompt=1000))

    def test_malformed_capped_call_retries_then_runs_one_check_and_answers(self):
        runner, requests, started, budget = self.runner([malformed(output=8192), self.corrected()])
        query = 'Please give me a classical book or reference for PDEs'
        result = runner.run(query)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer'], 'Evans, Partial Differential Equations.')
        self.assertEqual([a['mode'] for a in started], ['check'])
        self.assertEqual((result['main_calls'], result['actions']), (2, 1))
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]['temperature'], 0)
        self.assertEqual(requests[1]['temperature'], 0.6)
        self.assertIn('CONTROLLER TOOL-CALL RECOVERY', requests[1]['messages'][-1]['content'])
        self.assertEqual(requests[1]['messages'][1]['content'], query)
        self.assertNotIn('Evans, Evans', json.dumps(requests[1]))
        self.assertEqual(budget.snapshot()['tokens'], 8192 + 60)
        self.assertEqual(budget.snapshot()['reserved_tokens'], 0)
        [rejected] = runner.state['rejected_tool_calls']
        self.assertEqual(rejected['stats']['eval_count'], 8192)
        self.assertEqual(rejected['calls'][0]['function']['arguments'], '{"mode":"check","request":"Evans, Evans')
        self.assertFalse(runner.state['tool_call_recovery_pending'])
        _messages(runner.conversation_history())

    def test_rejected_call_refunds_only_measured_unused_reservation(self):
        runner, requests, started, budget = self.runner([malformed(output=37), self.corrected()])
        runner.run('Give me a classical reference for PDEs.')
        self.assertEqual(budget.snapshot()['tokens'], 37 + 60)
        self.assertEqual(budget.snapshot()['input_tokens'], 2000)
        self.assertEqual(budget.snapshot()['reserved_tokens'], 0)

    def test_repeated_bad_json_is_terminal_and_never_starts_a_worker(self):
        runner, requests, started, budget = self.runner([malformed(), malformed('[]')])
        result = runner.run('Give me a classical reference for PDEs.')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('again after one correction', result['warnings'][-1])
        self.assertEqual(started, [])
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(runner.state['rejected_tool_calls']), 2)
        self.assertEqual(budget.snapshot()['tokens'], 74)
        runner.run(runner.state['query'], state=runner.state)
        self.assertEqual(len(requests), 2)

    def test_pause_during_correction_preserves_the_retry_bound_and_feedback(self):
        runner, requests, started, budget = self.runner([malformed(), KeyboardInterrupt(), self.corrected()])
        query = 'Give me a classical reference for PDEs.'
        with self.assertRaises(KeyboardInterrupt):
            runner.run(query)
        saved = copy.deepcopy(runner.state)
        self.assertTrue(saved['tool_call_retried'])
        self.assertTrue(saved['tool_call_recovery_pending'])
        resumed = Orchestrator(runner.agent, runner.delegate)
        self.assertEqual(resumed.run(query, state=saved)['status'], 'complete')
        self.assertEqual(len(started), 1)
        self.assertIn('CONTROLLER TOOL-CALL RECOVERY', requests[-1]['messages'][-1]['content'])
        self.assertEqual(budget.snapshot()['tokens'], 37 + 8192 + 60)

    def test_old_malformed_json_checkpoint_gets_correction_without_resetting_calls(self):
        runner, requests, started, budget = self.runner([self.corrected()])
        query = 'Give me a classical reference for PDEs.'
        runner._new_state(query, [], [])
        saved = copy.deepcopy(runner.state)
        for key in ('tool_call_retried', 'tool_call_recovery_pending', 'rejected_tool_calls'):
            saved.pop(key)
        saved.update(status='interrupted', main_calls=1, warnings=['Malformed JSON tool arguments'],
                     last_call={'status': 'running', 'tools_enabled': True, 'stats': {}, 'partial': {'content': ''}})
        result = runner.run(query, state=saved)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['main_calls'], 2)
        self.assertEqual(len(started), 1)
        self.assertIn('CONTROLLER TOOL-CALL RECOVERY', requests[0]['messages'][-1]['content'])

    def test_valid_call_in_a_rejected_batch_never_escapes_and_raw_arguments_survive(self):
        good = json.dumps({'path': 'source.tex'})
        bad = '{"path":'
        body = sse(delta(tool_calls=[
            {'index': 0, 'id': 'good', 'function': {'name': 'read_file', 'arguments': good}},
            {'index': 1, 'id': 'bad', 'function': {'name': 'write_file', 'arguments': bad}}]),
            finish('tool_calls'), usage())
        client = OpenAICompatible()
        payload = {'model': 'test', 'messages': [], 'options': {'num_predict': 128}}
        with patch.object(client, 'request', return_value=io.BytesIO(body)):
            with self.assertRaises(ToolCallError) as caught:
                list(client.stream(payload))
        self.assertEqual([c['function']['arguments'] for c in caught.exception.calls], [good, bad])
        self.assertEqual(caught.exception.stats['eval_count'], 7)

    def test_recovery_does_not_rerun_a_completed_worker(self):
        written = sse(delta('The saved finding answers the request.'), finish(), usage(output=30, prompt=1000))
        runner, requests, started, budget = self.runner([self.corrected(), malformed(), written])
        result = runner.run('Find a reference and explain the result.')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(started), 1)
        self.assertEqual(result['actions'], 1)
        self.assertIn('Evans, Partial Differential Equations.', json.dumps(requests[-1]['messages']))
        self.assertEqual(budget.snapshot()['tokens'], 60 + 37 + 30)

    def test_bad_error_usage_keeps_reservation_charged(self):
        from mathagent.assistant_budget import AssistantBudget
        runner, requests, started, budget = self.runner([malformed(output=1)])
        def invalid_stream(payload):
            raise ToolCallError('Malformed JSON tool arguments', [], {'eval_count': 8193})
            yield
        budget.client.stream = invalid_stream
        with self.assertRaises(AssistantBudget):
            runner.run('Give me a classical reference for PDEs.')
        self.assertEqual(budget.snapshot()['tokens'], 8192)
        self.assertEqual(budget.snapshot()['reserved_tokens'], 0)
        self.assertFalse(runner.state['tool_call_retried'])

    def test_completed_qualified_lookup_clears_stale_progress_but_keeps_other_work(self):
        progress = {'phase': 'verify', 'completed': 0, 'total': 2, 'unit': 'research rounds'}
        state = {'status': 'complete', 'progress': progress,
                 'answer_source': {'kind': 'worker_result', 'job_id': 'check_1'},
                 'children': [{'job_id': 'check_1', 'status': 'complete', 'action': {'mode': 'check'},
                               'result': {'disposition': 'answer_with_qualification'}}]}
        self.assertIsNone(store.assistant_view({'assistant_state': state})['assistant_progress'])
        self.assertEqual(state['progress'], progress)
        state['children'][0]['action']['mode'] = 'explain'
        self.assertEqual(store.assistant_view({'assistant_state': state})['assistant_progress'], progress)
        state['children'][0]['action']['mode'] = 'check'
        state['status'] = 'incomplete'
        self.assertEqual(store.assistant_view({'assistant_state': state})['assistant_progress'], progress)

    def test_transport_failure_is_not_reclassified_or_retried(self):
        runner, requests, started, budget = self.runner([AgentError('Cannot reach model server')])
        with self.assertRaisesRegex(AgentError, 'Cannot reach'):
            runner.run('Give me a classical reference for PDEs.')
        self.assertEqual(len(requests), 1)
        self.assertFalse(runner.state['tool_call_retried'])


class ToolRecoveryGuiTests(GuiCase):
    extra_args = ('--ctx', '40960')
    openai_fixture = assistant_fixture.AssistantGuiTests.openai_fixture

    def test_actual_sse_failure_recovers_to_one_saved_literature_check_and_answer(self):
        self.openai_fixture()
        chat = self.ok('POST', '/api/chats', {'mode': 'free', 'online': False})['id']
        bad = delegation('check', 'Find a classical PDE book.')
        bad[0]['message']['tool_calls'][0]['function']['arguments'] = '{"mode":"check","request":"Evans, Evans'
        bad[0]['eval_count'] = 37
        answer = 'A classical source is suggested; its precise coverage remains unverified. The requested explanation is elementary.'
        self.fake.replies = [bad, delegation('check', 'Find a classical PDE book.'),
                             text(json.dumps(RECALL)), text(answer)]
        self.ok('POST', f'/api/chats/{chat}/route',
                {'content': 'Explain PDE regularity and give me a classical reference.', 'files': []})
        task = self.wait_task(timeout=30)
        self.assertEqual(task['state'], 'done', task)
        saved = store.load_chat(self.root, chat)
        state = saved['assistant_state']
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['main_calls'], 3)
        self.assertEqual(len(state['children']), 1)
        self.assertEqual(state['children'][0]['action']['mode'], 'check')
        self.assertEqual(state['answer'], answer)
        self.assertIn('unverified', state['answer'])
        self.assertEqual(len([i for i in saved['transcript'] if i['role'] == 'route']), 1)
        self.assertEqual(state['budget']['tokens'], 37 + 8 + 20 + 20)
        self.assertEqual(state['budget']['reserved_tokens'], 0)
        _messages(saved['history'])


if __name__ == '__main__':
    unittest.main()
