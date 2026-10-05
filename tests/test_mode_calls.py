"""Mode selection forces an existing worker, with unchanged ordinary routing."""
import copy
import json
import unittest

from mathagent.mode_calls import ModeOrchestrator, validated_mode_call
from mathagent.agent import AgentError
from tests.test_orchestrator import Client, final
from mathagent.agent import Agent
from mathagent.tools import Workspace
from tests.test_gui import GuiCase, text, PROOF, APPROVAL


class ModeCallValidationTests(unittest.TestCase):
    def test_only_selected_token_is_removed_and_other_text_is_preserved(self):
        message = '🙂 @prove and @prove show this'
        self.assertEqual(validated_mode_call(message, {'mode': 'prove', 'start': 13}),
                         ('🙂 @prove and  show this', 'prove'))
        self.assertEqual(validated_mode_call('@unknown @prove show this', None),
                         ('@unknown @prove show this', None))

    def test_stale_malformed_email_and_empty_calls_are_rejected(self):
        for content, selected in [
            ('@proven Show this', {'mode': 'prove', 'start': 0}),
            ('mail@prove.com', {'mode': 'prove', 'start': 4}),
            ('@prove', {'mode': 'prove', 'start': 0}),
            ('@unknown Show this', {'mode': 'unknown', 'start': 0}),
            ('@prove Show this', {'mode': 'prove', 'start': True}),
            ('@prove Show this', {'mode': 'prove', 'start': -1}),
            ('@prove Show this', {'mode': ['prove'], 'start': 0}),
            ('@prove Show this', {}),
            ('@prove Show this', 'prove'),
        ]:
            with self.subTest(content=content, selected=selected), self.assertRaises(ValueError):
                validated_mode_call(content, selected)


class ModeCallControllerTests(unittest.TestCase):
    def test_every_mode_executes_before_synthesis_without_a_routing_inference(self):
        for mode in ('prove', 'critic', 'explore', 'check', 'literature', 'referee', 'writeup'):
            calls = []
            def worker(action, child, save):
                calls.append(copy.deepcopy(action))
                return {'status': 'complete', 'answer': 'Worker findings'}
            client = Client([final('Synthesis')])
            agent = Agent(client, Workspace('/tmp'), ctx=32768, predict=8192)
            runner = ModeOrchestrator(agent, worker, required_mode=mode)
            result = runner.run('Use the exact hypothesis H.')
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([call['mode'] for call in calls], [mode])
            self.assertEqual(calls[0]['request'], 'Use the exact hypothesis H.')
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(result['actions'], 1)

    def test_resume_keeps_the_same_forced_child_and_does_not_repeat_it(self):
        children = []
        def worker(action, child, save):
            children.append(child['id'])
            return {'status': 'complete', 'answer': 'Worker findings'}
        runner = ModeOrchestrator(Agent(Client([AgentError('Disconnected')]), Workspace('/tmp'), ctx=32768, predict=8192),
                                  worker, required_mode='prove')
        with self.assertRaises(AgentError):
            runner.run('Use hypothesis H.')
        saved = copy.deepcopy(runner.state)
        resumed = ModeOrchestrator(Agent(Client([final()]), Workspace('/tmp'), ctx=32768, predict=8192), worker)
        self.assertEqual(resumed.run('Use hypothesis H.', state=saved)['status'], 'complete')
        self.assertEqual(len(children), 1)
        self.assertEqual(resumed.state['required_mode'], 'prove')


class ModeCallGuiTests(GuiCase):
    extra_args = ('--ctx', '40960')

    def test_selected_prove_bypasses_classifier_runs_proof_then_synthesizes(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [text(PROOF), text(json.dumps(APPROVAL)), text('Reviewed proof returned.')]
        self.ok('POST', f'/api/chats/{chat}/route',
                {'content': 'For real x, @prove show x=x.', 'mode_call': {'mode': 'prove', 'start': 12}})
        task = self.wait_task()
        self.assertEqual(task['state'], 'done', task)
        saved = self.ok('GET', f'/api/chats/{chat}')
        cards = [item for item in saved['transcript'] if item['role'] == 'route']
        self.assertEqual([item['mode'] for item in cards], ['prove'])
        self.assertEqual(cards[0]['status'], 'done')
        self.assertEqual([item['content'] for item in saved['transcript'] if item['role'] == 'assistant'][-1],
                         'Reviewed proof returned.')
        self.assertEqual(len(self.fake.requests), 3)
        self.assertNotIn('tools', self.fake.requests[0])
        self.assertIn('For real x,  show x=x.', json.dumps(self.fake.requests[0]))

    def test_unconfirmed_text_uses_normal_assistant_and_stale_selection_saves_nothing(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [text('Literal text received.')]
        self.ok('POST', f'/api/chats/{chat}/route', {'content': '@prove is just text'})
        self.wait_task()
        self.assertEqual(len(self.fake.requests), 1)
        before = self.ok('GET', f'/api/chats/{chat}')['transcript']
        status, _, _ = self.request('POST', f'/api/chats/{chat}/route',
                                   {'content': '@proven is text', 'mode_call': {'mode': 'prove', 'start': 0}})
        self.assertEqual(status, 400)
        self.assertEqual(self.ok('GET', f'/api/chats/{chat}')['transcript'], before)
