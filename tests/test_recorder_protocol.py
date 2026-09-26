"""Replay recorder failures without a GPU or weakening proof review checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mathagent.agent import Agent
from mathagent.ledger import LedgerError, ProofStore
from mathagent.proof import ProofContext, ProofRunner
from mathagent.tools import Workspace
from test_proof import FakeClient, audit, batch, critic, response, review


class RecorderProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runner(self, replies):
        self.client = FakeClient(replies)
        return ProofRunner(Agent(self.client, Workspace(self.root), ctx=16384, predict=2048))

    @staticmethod
    def invented_batch():
        return batch(review(critical_claim='For real x, x+0=x.'),
                     review(dependencies=['C1']))

    def test_same_batch_id_is_corrected_before_any_ledger_commit(self):
        original = 'For each real x, reflexivity gives x=x.'
        runner = self.runner([response(original), response(critic(complete_candidate=False)),
            response(self.invented_batch()), response(batch())])
        self.assertEqual(runner.start('Prove x=x.', max_rounds=1)['status'], 'budget_exhausted')
        self.assertEqual([c['role'] for c in runner.state['calls']],
                         ['solver', 'critic', 'recorder', 'recorder'])
        self.assertEqual(runner.state['tokens_charged'], 40)
        self.assertEqual(runner.state['rounds_started'], 1)
        self.assertEqual([c['status'] for c in runner.state['claims']], ['reviewed'])
        saved_round = runner.state['rounds'][0]
        self.assertEqual(saved_round['recorder_attempts'], 2)
        self.assertIn('not an existing record ID', saved_round['recorder_rejections'][0]['errors'][0])
        repair_request = self.client.requests[3]['messages'][1]['content']
        self.assertIn('RECORDER PROTOCOL CORRECTION', repair_request)
        self.assertIn('same candidate and critique', repair_request)
        self.assertIn(original, repair_request)
        draft = json.loads(runner.store.read_artifact(saved_round['draft']))
        self.assertEqual(draft['text'], original)
        self.assertIn(original, self.client.requests[-1]['messages'][1]['content'])

    def test_two_malformed_batches_stop_without_fabricating_mathematical_gap(self):
        runner = self.runner([response('x=x by reflexivity.'), response(critic(complete_candidate=False)),
            response(self.invented_batch()), response(self.invented_batch())])
        result = runner.start('Prove x=x.', max_rounds=10)
        self.assertEqual(result['status'], 'stalled')
        self.assertEqual(runner.state['claims'], [])
        self.assertEqual(runner.state['rounds_started'], 1)
        self.assertIsNone(runner.state['final_audit'])
        self.assertEqual(runner.state['protocol_error']['stage'], 'recorder')
        self.assertIn('not a mathematical verdict', runner.state['stop_reason'])
        before = copy.deepcopy(runner.state)
        runner.resume(result['id'])
        self.assertEqual(runner.state, before)
        self.assertEqual(len(self.client.requests), 4)

    def test_empty_complete_batch_is_corrected_before_solver_can_repeat(self):
        empty = batch()
        empty['claims'] = []
        # A saved job already in review retains its recorder protocol even
        # though a new clean candidate would now take the direct audit path.
        runner = self.runner([response(empty), response(batch()), response(audit())])
        store = ProofStore.create(self.root, 'Prove x=x.',
            {'max_rounds': 1, 'max_tokens': 60000, 'max_seconds': 60,
             'host': self.client.host, 'model': runner.agent.model, 'ctx': runner.agent.ctx,
             'predict': runner.agent.predict, 'think': runner.agent.think})
        draft = store.write_artifact('candidate', json.dumps(
            {'text': 'x=x by reflexivity.', 'thinking': '', 'complete': True}))
        store.state.update(status='paused', rounds_started=1,
            pending={'index': 1, 'phase': 'review', 'fresh': False, 'think': False,
                     'task': 'Audit the complete proof.', 'draft': draft,
                     'solver_truncated': False, 'critique': critic()})
        store.save()
        self.assertEqual(runner.resume(store.state['id'])['status'], 'candidate_complete')
        self.assertEqual(runner.state['rounds_started'], 1)
        self.assertEqual([c['role'] for c in runner.state['calls']],
                         ['recorder', 'recorder', 'auditor'])
        self.assertIn('claims is empty',
                      runner.state['rounds'][0]['recorder_rejections'][0]['errors'][0])

    def test_omitted_resolution_is_repaired_and_old_objection_reaches_auditor(self):
        objection = 'The argument assumes x>0 and omits x=0.'
        previous = review(disposition='gap', objection=objection)
        fixed = review(resolves=['C1'], resolution='Reflexivity holds also at x=0, without positivity.')
        runner = self.runner([response('An incomplete argument.'),
            response(critic(complete_candidate=False, missing_work=objection)),
            response(batch(previous, complete_candidate=False)),
            response('For all real x, including zero, reflexivity gives x=x.'),
            response(critic(complete_candidate=False)), response(batch()), response(batch(fixed)),
            response('For arbitrary real x, reflexivity gives x=x.'), response(critic()), response(audit())])
        self.assertEqual(runner.start('Prove x=x.', max_rounds=3)['status'], 'candidate_complete')
        self.assertEqual([c['status'] for c in runner.state['claims']], ['gap', 'reviewed', 'reviewed'])
        self.assertEqual(runner.state['claims'][1]['resolves'], ['C1'])
        self.assertEqual(runner.state['rounds_started'], 3)
        self.assertEqual(runner.state['tokens_charged'], 100)
        self.assertIn(objection, self.client.requests[-1]['messages'][1]['content'])
        self.assertIn('same claim has unresolved recorded objections (C1)',
                      self.client.requests[6]['messages'][1]['content'])

    def test_correction_can_retain_a_gap_instead_of_inventing_a_resolution(self):
        gap = review(disposition='gap', objection='The positivity assumption is unjustified.')
        runner = self.runner([response('Incomplete.'), response(critic(complete_candidate=False)),
            response(batch(gap, complete_candidate=False)), response('Still incomplete.'),
            response(critic(complete_candidate=False)), response(batch()),
            response(batch(gap, complete_candidate=False))])
        self.assertEqual(runner.start('Prove x=x.', max_rounds=2)['status'], 'budget_exhausted')
        self.assertTrue(all(c['status'] == 'gap' for c in runner.state['claims']))
        self.assertNotIn('protocol_error', runner.state)
        self.assertIsNone(runner.state['final_audit'])

    def test_fresh_critic_objection_survives_recorder_correction(self):
        objection = critic(first_invalid_step='Every real x is positive.',
            reason='Zero is allowed.', missing_work='Handle nonpositive values.', complete_candidate=False)
        runner = self.runner([response('Assume positivity.'), response(objection),
            response(self.invented_batch()), response(batch())])
        runner.start('Prove x=x.', max_rounds=1)
        self.assertIsNone(runner.state['final_audit'])
        record = next(c for c in runner.state['claims'] if c['statement'] == objection['first_invalid_step'])
        self.assertEqual(record['status'], 'gap')
        self.assertEqual(record['objection'], objection['reason'])
        self.assertEqual([c['role'] for c in runner.state['calls']],
                         ['solver', 'critic', 'recorder', 'recorder'])

    def test_interrupted_correction_cannot_be_retried_by_resume(self):
        runner = self.runner([response('x=x by reflexivity.'), response(critic(complete_candidate=False)),
            response(self.invented_batch()), [KeyboardInterrupt()]])
        with self.assertRaises(KeyboardInterrupt):
            runner.start('Prove x=x.')
        self.assertEqual(runner.state['pending']['recorder_attempts'], 2)
        charged = runner.state['tokens_charged']
        self.assertEqual(charged, 30 + runner.state['calls'][-1]['reserved_tokens'])
        result = runner.resume(runner.state['id'])
        self.assertEqual(result['status'], 'stalled')
        runner.resume(result['id'])
        self.assertEqual(runner.state['tokens_charged'], charged)
        self.assertEqual(len(self.client.requests), 4)
        self.assertEqual(runner.state['claims'], [])

    def test_interrupted_initial_recorder_gets_only_one_new_dispatch(self):
        runner = self.runner([response('x=x by reflexivity.'), response(critic(complete_candidate=False)),
            [KeyboardInterrupt()], response(batch())])
        with self.assertRaises(KeyboardInterrupt):
            runner.start('Prove x=x.', max_rounds=1)
        charged = runner.state['tokens_charged']
        result = runner.resume(runner.state['id'])
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(runner.state['tokens_charged'], charged + 10)
        self.assertEqual(runner.state['rounds'][0]['recorder_attempts'], 2)
        self.assertEqual(len(self.client.requests), 4)

    def test_recorder_recovery_does_not_bypass_generated_token_budget(self):
        runner = self.runner([response('x=x by reflexivity.', count=128),
            response(critic(complete_candidate=False), count=128), response(self.invented_batch(), count=200)])
        result = runner.start('Prove x=x.', max_tokens=512)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(runner.state['tokens_charged'], 456)
        self.assertEqual(len(self.client.requests), 3)
        self.assertEqual(runner.state['claims'], [])

    def test_known_unresolved_dependency_is_a_mathematical_gap_without_protocol_retry(self):
        first = review(critical_claim='Some necessary lemma.', disposition='gap',
                       objection='The lemma remains unproved.')
        dependent = review(dependencies=['C1'])
        runner = self.runner([response('A partial argument.'), response(critic(complete_candidate=False)),
            response(batch(first, complete_candidate=False)), response('Assume the unproved lemma.'),
            response(critic(complete_candidate=False)), response(batch(dependent, complete_candidate=False))])
        runner.start('Prove x=x.', max_rounds=2)
        self.assertEqual([c['role'] for c in runner.state['calls']],
                         ['solver', 'critic', 'recorder', 'solver', 'critic', 'recorder'])
        self.assertEqual(runner.state['claims'][-1]['status'], 'gap')
        self.assertIn('Unresolved or unknown dependencies', runner.state['claims'][-1]['objection'])

    def test_storage_error_is_not_reinterpreted_as_bad_model_json(self):
        runner = self.runner([])
        runner.store = ProofStore.create(self.root, 'Prove x=x.',
            {'max_rounds': 1, 'max_tokens': 60000, 'max_seconds': 60})
        with mock.patch.object(runner, '_json_call', side_effect=LedgerError('Storage failed')):
            with self.assertRaisesRegex(LedgerError, 'Storage failed'):
                runner._review({'text': 'x=x.', 'thinking': '', 'complete': True}, critic())

    def test_pre_dispatch_context_failure_does_not_consume_attempt(self):
        runner = self.runner([response('x=x by reflexivity.'), response(critic(complete_candidate=False))])
        with mock.patch.object(runner, '_review', side_effect=ProofContext('Too little context')):
            result = runner.start('Prove x=x.')
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(runner.state['pending']['recorder_attempts'], 0)
        self.assertEqual(len(self.client.requests), 2)


if __name__ == '__main__':
    unittest.main()
