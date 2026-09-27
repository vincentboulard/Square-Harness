"""Behavioural regressions for v0.5's candidate-preserving proof controller."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mathagent.agent import Agent
from mathagent.backends import TokenBudgetError
from mathagent.ledger import LedgerError, ProofStore
from mathagent.proof import ProofRunner, check_context, validate_review
from mathagent.proof_policy import initial_request
from mathagent.tools import Workspace


def event(text='', *, count=10, thinking='', reason='stop'):
    if not isinstance(text, str):
        text = json.dumps(text)
    return {'message': {'content': text, 'thinking': thinking}, 'done': True,
            'done_reason': reason, 'eval_count': count, 'prompt_eval_count': 40}


def approval(explanation='No materially invalid inference was found; the proof checks out.'):
    return {'explanation': explanation, 'issues': [], 'verdict': 'no_issue_found'}


def objection():
    return {'explanation': 'A coefficient needs examination.', 'issues': [{
        'location': 'The last displayed inequality', 'kind': 'invalid_inference',
        'evidence': 'For a=b=1, the asserted inequality 2ab <= a² is false.'}], 'verdict': 'issues_found'}


class Client:
    backend = 'openai'
    host = 'http://fake.invalid'
    timeout = 60
    thinking_budget_policy = {'ordinary': .75, 'structured': .75}

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def count_input_tokens(self, payload):
        return 40

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        for item in value:
            if isinstance(item, BaseException):
                raise item
            yield item


class ProofV05Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, replies):
        client = Client(replies)
        agent = Agent(client, Workspace(self.root), ctx=40960, predict=77,
                      seed=17, temperature=.6, top_p=.95, think=True)
        return ProofRunner(agent)

    def run_proof(self, replies, **kwargs):
        runner = self.runner(replies)
        settings = dict(max_predict=512, verify_tokens=256, max_tokens=5000, max_rounds=3)
        settings.update(kwargs)
        return runner, runner.start('Prove x=x for every real x.', **settings)

    def test_first_solve_matches_direct_and_positive_explanation_stops(self):
        text = 'For every real x, equality is reflexive, hence x=x.\n'
        runner, result = self.run_proof([[event(text)], [event(approval())]])
        request = runner.agent.client.requests[0]
        self.assertEqual(request['messages'], initial_request('Prove x=x for every real x.'))
        self.assertNotIn('tools', request)
        self.assertEqual(request['options']['num_predict'], 512)
        self.assertEqual(request['options']['seed'], 17)
        self.assertEqual(result['answer'], text)
        self.assertEqual(Path(result['proof_path']).read_text(), text)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([c['role'] for c in runner.state['calls']], ['solver', 'verifier'])
        self.assertTrue(all(r['think'] for r in runner.agent.client.requests))
        self.assertEqual(result['tokens_charged'], 20)
        self.assertEqual(result['initial_candidate'], result['selected_candidate'])

    def test_contradictory_positive_verdict_rechecks_same_proof_not_solver(self):
        malformed = objection(); malformed['verdict'] = 'no_issue_found'
        runner, result = self.run_proof([[event('Original proof')], [event(malformed)], [event(approval())]])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(result['answer'], 'Original proof')
        self.assertEqual([c['role'] for c in runner.state['calls']], ['solver', 'verifier', 'verifier'])
        self.assertIn('SAME candidate', runner.agent.client.requests[2]['messages'][0]['content'])
        self.assertEqual(len(runner.state['candidates']), 1)

    def test_malformed_reviews_stop_with_original_after_one_retry(self):
        runner, result = self.run_proof([[event('Original proof')], [event('{bad')], [event('still bad')]])
        self.assertEqual(result['status'], 'review_unavailable')
        self.assertEqual(result['answer'], 'Original proof')
        self.assertEqual(len(runner.agent.client.requests), 3)

    def test_uncertainty_is_not_a_rewrite_instruction(self):
        review = {'verdict': 'uncertain', 'explanation': 'I cannot settle the boundary case.', 'issues': []}
        runner, result = self.run_proof([[event('Original proof')], [event(review)]])
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['answer'], 'Original proof')
        self.assertEqual(len(runner.state['calls']), 2)

    def test_targeted_repair_can_rebut_false_criticism(self):
        corrected = 'Complete proof: the alleged inequality was not asserted. Exact calculation proves x=x.'
        runner, result = self.run_proof([[event('Original proof')], [event(objection())],
                                      [event(corrected)], [event(approval())]])
        repair = runner.agent.client.requests[2]['messages'][0]['content']
        self.assertIn('fallible allegations', repair)
        self.assertIn('rebut an invalid objection', repair)
        self.assertIn('Original proof', repair)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(result['answer'], corrected)
        self.assertEqual(result['initial_candidate']['id'], 'P1')
        self.assertEqual(result['selected_candidate']['id'], 'P2')
        self.assertEqual(runner._candidate_text(result['initial_candidate']), 'Original proof')
        verifier = runner.agent.client.requests[3]['messages'][0]['content']
        self.assertIn(corrected, verifier)
        self.assertNotIn('A coefficient needs examination.', verifier)
        self.assertIn('A coefficient needs examination.', result['report'])

    def test_unapproved_revision_never_replaces_baseline(self):
        uncertain = dict(approval(), verdict='uncertain')
        runner, result = self.run_proof([[event('Original proof')], [event(objection())],
                                      [event('Possibly worse revision')], [event(uncertain)]])
        self.assertEqual(result['answer'], 'Original proof')
        self.assertEqual(len(runner.state['candidates']), 2)
        self.assertEqual(result['selected_candidate']['review_status'], 'issues_found')

    def test_truncated_revision_never_replaces_original(self):
        runner, result = self.run_proof([[event('Original proof')], [event(objection())],
                                      [event('A fragment', reason='length')]], max_rounds=2)
        self.assertEqual(result['status'], 'attempts_exhausted')
        self.assertEqual(result['answer'], 'Original proof')
        self.assertFalse(runner.state['candidates'][1]['transport_complete'])
        self.assertEqual(len(runner.agent.client.requests), 3)

    def test_complete_continuation_replaces_fragment_even_when_review_uncertain(self):
        uncertain = dict(approval(), verdict='uncertain')
        runner, result = self.run_proof([[event('Fragment', reason='length')],
                                       [event('Finished written candidate')], [event(uncertain)]])
        self.assertEqual(result['status'], 'uncertain')
        self.assertEqual(result['answer'], 'Finished written candidate')
        self.assertEqual(result['initial_candidate']['id'], 'P1')
        self.assertEqual(result['selected_candidate']['id'], 'P2')
        self.assertEqual(runner.state['selection_history'][1]['reason'], 'finished_response_replaces_fragment')
        self.assertIn('finished_response_replaces_fragment', result['report'])

    def test_complete_continuation_retained_when_review_unavailable(self):
        runner, result = self.run_proof([[event('Fragment', reason='length')],
                                       [event('Finished written candidate')], [event('bad')], [event('bad')]])
        self.assertEqual(result['status'], 'review_unavailable')
        self.assertEqual(result['answer'], 'Finished written candidate')
        self.assertEqual(runner._candidate_text(result['initial_candidate']), 'Fragment')

    def test_continuation_uses_exact_written_and_thinking_not_summary(self):
        runner, result = self.run_proof([[event('Partial equation: x=', thinking='Exact note: equality is reflexive', reason='length')],
                                      [event('For every x, x=x.')], [event(approval())]])
        request = runner.agent.client.requests[1]
        self.assertIn('Partial equation: x=', request['messages'][0]['content'])
        self.assertIn('Exact note: equality is reflexive', request['messages'][0]['content'])
        self.assertIn('not a resumed', request['messages'][0]['content'])
        self.assertTrue(request['think'])
        self.assertEqual(result['answer'], 'For every x, x=x.')

    def test_attempt_not_started_when_solve_and_review_do_not_fit(self):
        runner, result = self.run_proof([], max_tokens=767)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['tokens_charged'], 0)
        self.assertEqual(runner.agent.client.requests, [])

    def test_late_repair_not_started_and_caps_not_shrunk(self):
        runner, result = self.run_proof([[event('Original proof', count=500)], [event(objection(), count=250)]], max_tokens=900)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['answer'], 'Original proof')
        self.assertEqual([r['options']['num_predict'] for r in runner.agent.client.requests], [512, 256])

    def test_review_protocol_retry_does_not_starve_later_call(self):
        runner, result = self.run_proof([[event('Original proof', count=512)], [event('invalid', count=256)]], max_tokens=768)
        self.assertEqual(result['status'], 'review_unavailable')
        self.assertEqual(len(runner.agent.client.requests), 2)
        self.assertEqual(result['answer'], 'Original proof')

    def test_final_completion_at_deadline_is_saved_and_actual_tokens_charged(self):
        runner = self.runner([])
        expired = [False]
        def stream(payload):
            runner.agent.client.requests.append(copy.deepcopy(payload))
            expired[0] = True
            yield event('Finished at deadline', count=12)
        runner.agent.client.stream = stream
        runner._elapsed = lambda: 2 if expired[0] else 0
        result = runner.start('Prove x=x.', max_seconds=1, max_predict=512, verify_tokens=256, max_tokens=5000)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(result['answer'], 'Finished at deadline')
        self.assertEqual(result['tokens_charged'], 12)
        self.assertEqual(len(runner.agent.client.requests), 1)

    def test_interruption_exports_exact_partial_without_new_inference(self):
        partial = {'message': {'content': 'Useful partial result'}, 'done': False}
        runner, result = self.run_proof([[partial, KeyboardInterrupt()]])
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(result['answer'], 'Useful partial result')
        self.assertFalse(result['selected_candidate']['transport_complete'])
        self.assertEqual(result['tokens_charged'], 512)
        self.assertEqual(len(runner.agent.client.requests), 1)

    def test_over_cap_usage_is_charged_and_stops(self):
        runner, result = self.run_proof([[event('Original proof', count=600)]])
        self.assertEqual(result['status'], 'budget_violation')
        self.assertEqual(result['tokens_charged'], 600)
        self.assertEqual(len(runner.agent.client.requests), 1)

    def test_missing_usage_charged_at_reserved_cap(self):
        no_count = event('Original proof'); del no_count['eval_count']
        runner, result = self.run_proof([[no_count], [event(approval())]])
        self.assertEqual(result['tokens_charged'], 522)

    def test_completed_call_before_phase_commit_is_not_dispatched_twice(self):
        runner = self.runner([[event('Original proof')], [event(approval())]])
        original = runner._candidate
        interrupted = [False]
        def interrupt_once(*args):
            if not interrupted[0]:
                interrupted[0] = True
                raise KeyboardInterrupt
            return original(*args)
        with patch.object(runner, '_candidate', side_effect=interrupt_once):
            paused = runner.start('Prove x=x.', max_predict=512, verify_tokens=256, max_tokens=5000)
        self.assertEqual(paused['status'], 'paused')
        self.assertEqual(len(runner.agent.client.requests), 1)
        runner._candidate = original
        finished = runner.resume(paused['id'])
        self.assertEqual(finished['status'], 'candidate_complete')
        self.assertEqual(len(runner.agent.client.requests), 2)
        self.assertEqual(finished['tokens_charged'], 20)

    def test_completed_verifier_before_phase_commit_is_reused_even_low_remaining(self):
        runner = self.runner([[event('Original proof', count=512)], [event(approval(), count=256)]])
        real = validate_review
        with patch('mathagent.proof.validate_review', side_effect=KeyboardInterrupt):
            paused = runner.start('Prove x=x.', max_predict=512, verify_tokens=256, max_tokens=768)
        self.assertEqual(paused['status'], 'paused')
        self.assertEqual(paused['tokens_charged'], 768)
        finished = runner.resume(paused['id'])
        self.assertEqual(finished['status'], 'candidate_complete')
        self.assertEqual(len(runner.agent.client.requests), 2)

    def test_interrupted_solver_reserves_full_cap_then_continues_saved_material(self):
        partial = {'message': {'content': 'Saved fragment', 'thinking': 'Saved working note'}, 'done': False}
        runner = self.runner([[partial, KeyboardInterrupt()], [event('Complete candidate')], [event(approval())]])
        paused = runner.start('Prove x=x.', max_predict=512, verify_tokens=256, max_tokens=5000)
        self.assertEqual(paused['status'], 'paused')
        self.assertEqual(paused['tokens_charged'], 512)
        final = runner.resume(paused['id'])
        self.assertEqual(final['status'], 'candidate_complete')
        self.assertEqual(final['tokens_charged'], 532)
        self.assertEqual(len(runner.agent.client.requests), 3)
        self.assertIn('Saved fragment', runner.agent.client.requests[1]['messages'][0]['content'])

    def test_recovered_final_stream_event_refunds_once(self):
        runner = self.runner([[event('Original proof')], [event(approval())]])
        original = runner._charge_completed
        with patch.object(runner, '_charge_completed', side_effect=KeyboardInterrupt):
            paused = runner.start('Prove x=x.', max_predict=512, verify_tokens=256, max_tokens=5000)
        self.assertEqual(paused['tokens_charged'], 512)
        runner._charge_completed = original
        final = runner.resume(paused['id'])
        self.assertEqual(final['tokens_charged'], 20)
        self.assertEqual(len(runner.agent.client.requests), 2)
        self.assertEqual(runner.resume(final['id'])['tokens_charged'], 20)

    def test_resume_rejects_endpoint_change_before_new_dispatch(self):
        runner, result = self.run_proof([KeyboardInterrupt()])
        runner.agent.client.host = 'http://other.invalid'
        with self.assertRaisesRegex(ValueError, 'host'):
            runner.resume(result['id'])

    def test_resume_restores_saved_sampling_model_and_thinking_budget(self):
        runner, result = self.run_proof([KeyboardInterrupt(), [event('proof')], [event(approval())]])
        runner.agent.temperature = .9
        runner.agent.model = 'changed-model'
        runner.agent.ctx = 1024
        runner.agent.client.thinking_budget_policy = {'ordinary': .2}
        finished = runner.resume(result['id'])
        self.assertEqual(finished['status'], 'candidate_complete')
        request = runner.agent.client.requests[1]
        self.assertEqual(request['options']['temperature'], .6)
        self.assertEqual(request['model'], 'qwen3.8:27b')
        self.assertEqual(request['options']['num_ctx'], 40960)
        self.assertEqual(runner.agent.client.thinking_budget_policy, {'ordinary': .75, 'structured': .75})

    def test_completed_run_is_viewable_offline_with_other_cli_defaults(self):
        runner, result = self.run_proof([[event('proof')], [event(approval())]])
        runner.agent.client.host = 'http://other.invalid'
        runner.agent.model = 'different'
        self.assertEqual(runner.resume(result['id'])['answer'], 'proof')
        self.assertEqual(len(runner.agent.client.requests), 2)

    def test_proof_thinking_is_independent_of_ordinary_chat_setting(self):
        runner = self.runner([[event('proof')], [event(approval())]])
        runner.agent.think = False
        result = runner.start('Prove x=x.', max_predict=512, verify_tokens=256, max_tokens=5000)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertTrue(all(r['think'] for r in runner.agent.client.requests))

    def test_changed_transport_source_blocks_continuation(self):
        runner, result = self.run_proof([KeyboardInterrupt()])
        real = runner._provenance
        def changed():
            value = real()
            value['transport_sha256']['backends.py'] = 'changed'
            return value
        with patch.object(runner, '_provenance', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'transport_sha256'):
                runner.resume(result['id'])

    def test_full_sources_are_pinned_and_reviewed(self):
        (self.root / 'lemma.md').write_text('Exact hypothesis: x is real.')
        runner = self.runner([[event('proof')], [event(approval())]])
        result = runner.start('Prove x=x.', source_files=['lemma.md'], max_predict=512, verify_tokens=256, max_tokens=5000)
        for request in runner.agent.client.requests:
            self.assertIn('Exact hypothesis: x is real.', request['messages'][0]['content'])
        (self.root / 'lemma.md').write_text('Changed later.')
        before = len(runner.agent.client.requests)
        runner.resume(result['id'])
        self.assertEqual(len(runner.agent.client.requests), before)

    def test_context_guard_retains_candidate_and_sends_no_clipped_review(self):
        runner = self.runner([[event('Entire proof')]])
        counts = iter([40, 40960])
        runner.agent.client.count_input_tokens = lambda payload: next(counts)
        result = runner.start('Prove x=x.', max_predict=512, verify_tokens=256, max_tokens=5000)
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(result['answer'], 'Entire proof')
        self.assertEqual(len(runner.agent.client.requests), 1)
        self.assertEqual(result['tokens_charged'], 10)

    def test_tampered_candidate_is_not_exported(self):
        runner, result = self.run_proof([[event('proof')], [event(approval())]])
        artifact = runner.store.directory / 'artifacts' / result['selected_candidate']['artifact']
        artifact.write_text('different proof')
        with self.assertRaisesRegex(LedgerError, 'digest'):
            runner.resume(result['id'])

    def test_legacy_storage_read_only_and_not_resumed(self):
        runner, result = self.run_proof([[event('proof')], [event(approval())]])
        store = runner.store
        state = copy.deepcopy(store.state)
        state.update(version=1, rounds=[], claims=[], next_task='', stagnant_rounds=0, final_audit=None)
        (store.directory / 'state.json').write_text(json.dumps(state))
        loaded = ProofStore.load(self.root, result['id'])
        self.assertEqual(loaded.state['version'], 1)
        with self.assertRaisesRegex(LedgerError, 'read-only'):
            loaded.write_artifact('new', 'not allowed')
        with self.assertRaisesRegex(LedgerError, 'read-only'):
            loaded.save()
        with self.assertRaisesRegex(ValueError, 'read-only'):
            runner.resume(result['id'])

    def test_corrupt_state_backup_never_dispatches_inference(self):
        runner, result = self.run_proof([[event('proof')], [event(approval())]])
        (runner.store.directory / 'state.json').write_text('{broken')
        with self.assertWarns(RuntimeWarning):
            recovered = runner.resume(result['id'])
        self.assertEqual(recovered['status'], 'needs_recovery')
        self.assertEqual(len(runner.agent.client.requests), 2)

    def test_conservative_context_fallback_and_exact_counter(self):
        client = Client([])
        payload = {'messages': initial_request('x=x'), 'options': {'num_ctx': 1000, 'num_predict': 128}}
        self.assertEqual(check_context(client, payload)['method'], 'server token count')
        client.count_input_tokens = lambda payload: None
        self.assertIn('byte estimate', check_context(client, payload)['method'])


if __name__ == '__main__':
    unittest.main()
