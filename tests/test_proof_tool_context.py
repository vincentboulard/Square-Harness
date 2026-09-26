"""Regression checks for retained tool evidence and current-candidate recording."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mathagent.agent import Agent
from mathagent.ledger import ProofStore
from mathagent.proof import ProofContext, ProofRunner
from mathagent.proof_policy import REVIEW_BATCH_SCHEMA
from mathagent.tools import Workspace
from test_proof import FakeClient, batch, critic, response, review


def tool_response(name, arguments):
    return [{'message': {'content': 'Inspect the needed evidence.',
                         'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]},
             'done': True, 'done_reason': 'stop', 'eval_count': 10}]


class ProofToolContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, replies, *, ctx=8192):
        self.client = FakeClient(replies)
        runner = ProofRunner(Agent(self.client, Workspace(self.root), ctx=ctx, predict=2048))
        runner.store = ProofStore.create(self.root, 'Prove x=x for every real x.',
            {'max_rounds': 5, 'max_tokens': 60000, 'max_seconds': 60,
             'max_predict': 8192})
        runner.state['rounds_started'] = 1
        runner.state['pending'] = {'index': 1, 'phase': 'solve', 'fresh': False,
                                   'task': 'Write an exact argument for arbitrary real x.'}
        return runner

    @staticmethod
    def record(index, *, argument='Reflexivity applies to real x.'):
        return {'id': f'C{index}', 'round': index, 'statement': 'For every real x, x=x.',
                'status': 'reviewed', 'assumptions': ['x is real'], 'dependencies': [],
                'argument': argument, 'objection': '', 'evidence': 'Exact supporting evidence.',
                'resolves': [], 'resolution': '', 'next_task': 'OLD_TASK_MARKER',
                'review_artifact': 'OLD_REVIEW_ARTIFACT', 'candidate_artifact': 'OLD_CANDIDATE_ARTIFACT'}

    def test_tool_continuation_reselects_optional_records_and_preserves_returned_evidence(self):
        runner = self.runner([tool_response('read_proof_claim', {'claim_id': 'C1'}),
                              response('For arbitrary real x, equality is reflexive.')])
        runner.state['claims'] = [self.record(n, argument='An exact derivation: ' + 'a' * 2800)
                                  for n in range(1, 21)]
        original = copy.deepcopy(runner.state['claims'])
        with patch.object(runner, '_tool', wraps=runner._tool) as execute:
            result = runner._solve(runner.state['pending'])
        self.assertTrue(result['complete'])
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(self.client.requests), 2)
        first, continuation = self.client.requests
        tool_messages = [m for m in continuation['messages'] if m['role'] == 'tool']
        self.assertEqual(len(tool_messages), 1)
        returned = tool_messages[0]['content']
        self.assertEqual(json.loads(json.loads(returned)['content']), original[0])
        self.assertEqual(runner.state['claims'], original)
        self.assertIn(runner.state['goal'], continuation['messages'][1]['content'])
        self.assertIn(runner.state['pending']['task'], continuation['messages'][1]['content'])
        self.assertLess(continuation['messages'][1]['content'].count('An exact derivation:'),
                        first['messages'][1]['content'].count('An exact derivation:'))
        limit = (runner.agent.ctx - 2048 - 256) * 3
        self.assertLessEqual(len(json.dumps(continuation, ensure_ascii=False).encode()), limit)
        old_behavior = copy.deepcopy(first)
        old_behavior['messages'].extend(continuation['messages'][2:])
        self.assertGreater(len(json.dumps(old_behavior, ensure_ascii=False).encode()), limit,
                           'The fixture must exercise a real envelope overflow before rebuilding')
        self.assertEqual([p['options']['num_predict'] for p in self.client.requests], [2048, 2048])
        artifacts = runner.state['pending']['tools']
        self.assertEqual(len(artifacts), 1)
        saved = json.loads(runner.store.read_artifact(artifacts[0]))
        self.assertEqual(saved['result'], returned)
        self.assertEqual(ProofStore.load(self.root, runner.state['id']).state['pending']['tools'], artifacts)

    def test_oversized_mandatory_tool_evidence_stops_without_replaying_or_dropping_it(self):
        runner = self.runner([])
        artifact = runner.store.write_artifact('large-evidence', '\\' * 12000)
        self.client.replies = iter([tool_response('read_proof_artifact', {'filename': artifact})])
        with patch.object(runner, '_tool', wraps=runner._tool) as execute:
            with self.assertRaises(ProofContext):
                runner._solve(runner.state['pending'])
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(len(runner.state['pending']['tools']), 1)
        saved = json.loads(runner.store.read_artifact(runner.state['pending']['tools'][0]))
        self.assertEqual(json.loads(saved['result'])['content'], '\\' * 6000)
        self.assertEqual(json.loads(saved['result'])['next_offset'], 6000)

    def test_recorder_reads_current_material_last_without_mutating_past_evidence(self):
        runner = self.runner([response(batch())], ctx=16384)
        runner.state['claims'] = [self.record(1)]
        original = copy.deepcopy(runner.state['claims'])
        runner._json_call('recorder', 'Record the current argument.',
                         'CURRENT_CANDIDATE_MARKER\nCURRENT_CRITIC_MARKER', REVIEW_BATCH_SCHEMA)
        prompt = self.client.requests[0]['messages'][1]['content']
        self.assertLess(prompt.index('Exact supporting evidence.'), prompt.index('CURRENT_CANDIDATE_MARKER'))
        self.assertLess(prompt.index('CURRENT_CANDIDATE_MARKER'), prompt.index('CURRENT_CRITIC_MARKER'))
        self.assertNotIn('OLD_TASK_MARKER', prompt)
        self.assertNotIn('OLD_REVIEW_ARTIFACT', prompt)
        self.assertNotIn('OLD_CANDIDATE_ARTIFACT', prompt)
        self.assertEqual(runner.state['claims'], original)

    def test_repeated_recorder_task_yields_to_new_critic_missing_work(self):
        runner = self.runner([])
        runner.state['next_task'] = 'Repeat the old task.'
        pending = {'index': 1, 'draft': 'candidate.md',
                   'critique': critic(complete_candidate=False,
                                     missing_work='Write the missing quantifier argument.')}
        runner._record_batch(batch(review(), next_task='Repeat the old task.', complete_candidate=False),
                             pending, {'complete': True})
        self.assertEqual(runner.state['next_task'], 'Write the missing quantifier argument.')

    def test_claimed_resolution_cannot_hide_historical_objection_from_final_audit(self):
        runner = self.runner([])
        gap = self.record(1)
        gap.update(status='gap', objection='The original hypotheses also allow x=0.',
                   evidence='The attempted argument assumes x>0 without justification.')
        alleged_repair = self.record(2)
        alleged_repair.update(resolves=['C1'], resolution='The new conclusion is asserted, so C1 is resolved.')
        runner.state['claims'] = [gap, alleged_repair]
        original = copy.deepcopy(runner.state['claims'])
        messages = runner._context('Audit every step independently.', 'Check the full candidate.',
                                   complete_ledger=True, retrieval=False)
        self.assertIn(gap['objection'], messages[1]['content'])
        self.assertIn(gap['evidence'], messages[1]['content'])
        self.assertIn('C1', runner._visible_claims)
        self.assertEqual(runner.state['claims'], original)


if __name__ == '__main__':
    unittest.main()
