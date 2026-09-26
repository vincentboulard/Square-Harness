"""Whole-candidate auditing skips bookkeeping without losing objections or recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mathagent.agent import Agent
from mathagent.ledger import ProofStore
from mathagent.proof import ProofRunner
from mathagent.tools import Workspace
from test_proof import FakeClient, audit, batch, critic, response


class ProofFastPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runner(self, replies):
        self.client = FakeClient(replies)
        return ProofRunner(Agent(self.client, Workspace(self.root), ctx=16384, predict=2048))

    def test_success_copies_exact_evidence_without_inventing_lemma_dependencies(self):
        goal = 'Prove x=x for every real x.'
        text = '  For arbitrary real x, reflexivity gives x=x.\n'
        verdict = audit(explanation='The arbitrary real x was handled by reflexivity.')
        runner = self.runner([response(text), response(critic()), response(verdict)])
        result = runner.start(goal, max_rounds=1)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([c['role'] for c in runner.state['calls']], ['solver', 'critic', 'auditor'])
        self.assertEqual(runner.state['tokens_charged'], 30)
        self.assertTrue(runner.state['rounds'][0]['fast_path'])
        self.assertNotIn('recorder_attempts', runner.state['rounds'][0])
        claim, = runner.state['claims']
        self.assertEqual(claim['record_kind'], 'whole_candidate_audit')
        self.assertEqual(claim['statement'], goal)
        self.assertEqual(claim['argument'], text)
        self.assertEqual(claim['evidence'], verdict['explanation'])
        self.assertEqual(claim['status'], 'reviewed')
        self.assertEqual(claim['dependencies'], [])
        self.assertEqual(claim['resolves'], [])
        self.assertEqual(Path(result['proof_path']).read_text(), text)
        self.assertIn('Model reviews are fallible', result['report'])

    def test_failed_direct_audit_reaches_next_solver_and_next_whole_proof_auditor(self):
        objection = 'The endpoint x=0 was omitted from the derivation.'
        rejected = audit(verdict='gap', explanation='Only positive x were handled.',
                         objection=objection, next_task='Repair the missing endpoint.')
        runner = self.runner([response('For positive x, x=x.'), response(critic()), response(rejected),
                              response('For every real x, reflexivity gives x=x.'),
                              response(critic()), response(audit())])
        result = runner.start('Prove x=x for every real x.', max_rounds=2)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([c['role'] for c in runner.state['calls']],
                         ['solver', 'critic', 'auditor', 'solver', 'critic', 'auditor'])
        old, new = runner.state['claims']
        self.assertEqual(old['status'], 'gap')
        self.assertEqual(old['whole_proof_objection'], objection)
        self.assertEqual(new['status'], 'reviewed')
        self.assertEqual(new['resolves'], [], 'A final audit must not invent local resolution evidence')
        self.assertIn(objection, self.client.requests[3]['messages'][1]['content'])
        self.assertNotIn(objection, json.dumps(self.client.requests[4]['messages']))
        self.assertIn(objection, self.client.requests[5]['messages'][1]['content'])
        self.assertEqual(runner.state['rounds'][0]['audit']['objection'], objection)

    def test_complete_audit_with_outstanding_next_task_cannot_approve_or_lose_task(self):
        task = 'Prove the missing boundary case.'
        runner = self.runner([response('A proposed proof.'), response(critic()),
                              response(audit(next_task=task))])
        result = runner.start('Prove the theorem including its boundary.', max_rounds=1)
        self.assertNotEqual(result['status'], 'candidate_complete')
        self.assertIsNone(result['proof_path'])
        self.assertEqual(runner.state['final_audit']['verdict'], 'uncertain')
        self.assertIn(task, runner._context('Solver', 'Continue.', fresh=True)[1]['content'])

    def test_checkpoint_is_still_partial_even_when_critic_claims_complete(self):
        runner = self.runner([response(thinking='An unfinished derivation.', complete=False),
                              response('A salvaged candidate that claims success.'),
                              response(critic()), response(batch())])
        result = runner.start('Prove x=x.', max_rounds=1)
        self.assertNotEqual(result['status'], 'candidate_complete')
        self.assertEqual([c['role'] for c in runner.state['calls']],
                         ['solver', 'checkpoint', 'critic', 'recorder'])
        self.assertFalse(runner.state['rounds'][0]['fast_path'])
        self.assertIsNone(runner.state['final_audit'])
        self.assertIsNone(result['proof_path'])

    def test_cached_audit_resumes_local_recording_without_new_tokens_or_duplicate_claim(self):
        runner = self.runner([response('For all real x, reflexivity gives x=x.'),
                              response(critic()), response(audit())])
        record = runner._record_audited_candidate

        def interrupt_after_mutation(*args, **kwargs):
            record(*args, **kwargs)
            raise KeyboardInterrupt()

        with mock.patch.object(runner, '_record_audited_candidate', side_effect=interrupt_after_mutation):
            with self.assertRaises(KeyboardInterrupt):
                runner.start('Prove x=x.', max_rounds=1)
        proof_id = runner.state['id']
        paused = ProofStore.load(self.root, proof_id)
        self.assertEqual(paused.state['pending']['phase'], 'audit')
        self.assertEqual(paused.state['pending']['audit']['verdict'], 'complete')
        self.assertEqual(paused.state['claims'], [], 'An interrupted local commit must roll back')
        charged = paused.state['tokens_charged']
        # Finishing already-paid local bookkeeping does not need more tokens.
        paused.state['settings']['max_tokens'] = charged
        paused.save()
        offline = self.runner([])
        result = offline.resume(proof_id)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(self.client.requests, [])
        self.assertEqual(offline.state['tokens_charged'], charged)
        self.assertEqual(len(offline.state['claims']), 1)
        self.assertEqual(len(offline.state['rounds']), 1)
        before = copy.deepcopy(offline.state)
        offline.resume(proof_id)
        self.assertEqual(offline.state, before)
        self.assertEqual(self.client.requests, [])


if __name__ == '__main__':
    unittest.main()
