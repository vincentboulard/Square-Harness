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
from types import SimpleNamespace

from mathagent import benchmark
from mathagent.proof import ProofBudget, ProofRunner, check_context, count_input, validate_review
from mathagent.proof_policy import COMMON_INSTRUCTION, initial_request
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


class ExactClient(Client):
    """Server-like counter: about 3.5 bytes per token, as measured on the A10 runs."""
    def count_input_tokens(self, payload):
        return int(len(json.dumps(payload['messages'], ensure_ascii=False).encode()) / 3.47) + 1


class ByteFallbackClient(Client):
    count_input_tokens = None  # e.g. Ollama: conservative 1 byte = 1 token estimate


REAL = dict(max_predict=32768, verify_tokens=16384, min_solve_tokens=16384, max_tokens=120000, max_rounds=2)
PROOF_7KB = 'Proof. ' + ('We use $\\sum_{k=0}^{n}\\binom{n}{k}x^k$ and check each case. ' * 110)
LONG_OBJECTION = {'explanation': 'The argument is mostly sound. ' * 100, 'verdict': 'issues_found', 'issues': [{
    'location': 'Step 3', 'kind': 'invalid_inference', 'evidence': 'For n=2 the displayed bound fails: 5 > 4. ' * 30}]}


class ProofV051RealSizeTests(unittest.TestCase):
    """v0.5.1: realistic 32k solves in a 40,960-token context."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def start(self, client, **kwargs):
        agent = Agent(client, Workspace(self.root), ctx=40960, predict=77, seed=17, temperature=.6, top_p=.95, think=True)
        runner = ProofRunner(agent)
        settings = dict(REAL); settings.update(kwargs)
        return runner, runner.start('Prove the identity for every integer n >= 1.', **settings)

    def repair_scenario(self, client_class):
        client = client_class([[event(PROOF_7KB, count=20000)], [event(LONG_OBJECTION, count=9000)],
                               [event('Repaired complete proof.', count=21000)], [event(approval(), count=8000)]])
        runner, result = self.start(client)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(result['answer'], 'Repaired complete proof.')
        initial, repair = client.requests[0], client.requests[2]
        self.assertEqual(initial['options']['num_predict'], 32768)  # the direct-call allowance is never fitted
        self.assertEqual(initial['messages'], initial_request('Prove the identity for every integer n >= 1.'))
        count = count_input(client, repair)[0]
        self.assertEqual(repair['options']['num_predict'], min(32768, 40960 - count - 1))
        self.assertEqual(runner.state['calls'][2]['reserved_tokens'], repair['options']['num_predict'])
        return count, repair['options']['num_predict']

    def test_repair_fits_with_exact_counter(self):
        count, cap = self.repair_scenario(ExactClient)
        self.assertEqual(cap, 32768)

    def test_repair_fits_with_byte_fallback_by_fitting_its_cap(self):
        count, cap = self.repair_scenario(ByteFallbackClient)
        self.assertGreater(count + 32768 + 1, 40960)  # v0.5.0 stopped here with needs_context
        self.assertTrue(16384 <= cap < 32768)

    def test_repair_below_minimum_allowance_is_context_failure_not_clipping(self):
        huge = 'x' * 21000  # the review fits; the repair would leave < 16384 output tokens
        client = ByteFallbackClient([[event(huge, count=20000)], [event(LONG_OBJECTION, count=9000)]])
        runner, result = self.start(client)
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(result['answer'], huge)
        self.assertEqual(len(client.requests), 2)

    def test_long_notes_continue_as_marked_tail_excerpt(self):
        notes = 'BEGIN-NOTES ' + ('checking the parity case carefully; ' * 2500) + ' END-NOTES'
        partial = 'Partial written derivation: S_n = 2J - n. ' * 40
        client = ExactClient([[event(partial, thinking=notes, reason='length', count=32768)],
                              [event('Finished continuation proof.', count=20000)], [event(approval(), count=8000)]])
        runner, result = self.start(client)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(result['answer'], 'Finished continuation proof.')
        request = client.requests[1]
        content = request['messages'][0]['content']
        self.assertIn(partial, content)
        self.assertIn('final excerpt, earlier notes omitted', content)
        self.assertIn('END-NOTES', content)
        self.assertNotIn('BEGIN-NOTES', content)
        cap = request['options']['num_predict']
        self.assertGreaterEqual(cap, 16384)
        self.assertLessEqual(count_input(client, request)[0] + cap + 1, 40960)
        self.assertEqual(runner.state['candidates'][1]['kind'], 'continue')

    def test_unfittable_continuation_becomes_fresh_solve(self):
        huge_fragment = 'fragment text ' * 9000
        client = ExactClient([[event(huge_fragment, thinking='notes', reason='length', count=32768)],
                              [event('Fresh complete proof.', count=20000)], [event(approval(), count=8000)]])
        runner, result = self.start(client)
        self.assertEqual(client.requests[1]['messages'], initial_request('Prove the identity for every integer n >= 1.'))
        self.assertEqual(client.requests[1]['options']['num_predict'], 32768)
        self.assertEqual(runner.state['candidates'][1]['kind'], 'retry')
        self.assertEqual(result['answer'], 'Fresh complete proof.')
        self.assertEqual(result['status'], 'candidate_complete')

    def test_verifier_is_sampled_and_framed_as_review(self):
        client = ExactClient([[event('A complete proof.', count=20000)], [event(approval(), count=8000)]])
        runner, result = self.start(client, verify_temperature=.3)
        solver, verifier = client.requests
        self.assertEqual(solver['options']['temperature'], .6)
        self.assertEqual(verifier['options']['temperature'], .3)
        self.assertEqual(runner.state['settings']['verify_temperature'], .3)
        content = verifier['messages'][0]['content']
        self.assertFalse(content.startswith(COMMON_INSTRUCTION))
        self.assertIn('TASK GIVEN TO THE SOLVER', content)
        self.assertIn('Prove the identity for every integer n >= 1.', content)
        self.assertIn('A complete proof.', content)
        self.assertEqual(len(client.requests), 2)  # one approval ends the job

    def test_verifier_temperature_defaults_to_solver_temperature(self):
        client = ExactClient([[event('A complete proof.')], [event(approval())]])
        runner, _ = self.start(client)
        self.assertEqual(client.requests[1]['options']['temperature'], .6)

    def test_verifier_output_is_bounded(self):
        with self.assertRaises(ValueError):
            validate_review(json.dumps(approval('x' * 4001)))
        many = dict(objection(), issues=objection()['issues'] * 6)
        with self.assertRaises(ValueError):
            validate_review(json.dumps(many))

    def test_repair_not_started_without_time_for_a_measured_cycle(self):
        client = ExactClient([[event('A proof.')], [event(objection())]])
        runner, result = self.start(client, max_rounds=1, max_seconds=1800)
        self.assertEqual(result['status'], 'attempts_exhausted')
        runner.state['calls'][0]['seconds'], runner.state['calls'][1]['seconds'] = 600, 400
        runner._clock = None
        runner._seconds_before = 1500
        with self.assertRaisesRegex(ProofBudget, 'first solve and review took 1000'):
            runner._reserve_attempt()
        runner._seconds_before = 500
        runner._reserve_attempt()  # 1300 s left covers a 1000 s cycle

    def test_benchmark_records_and_forwards_new_settings(self):
        data = benchmark.init_smoke(self.root / 'data')
        args = benchmark.parser().parse_args(['--manifest', str(data), '--output', str(self.root / 'out'), '--arms', 'proof'])
        _, plan = benchmark.preflight(args)
        self.assertEqual(plan['settings']['min_solve_tokens'], 16384)
        self.assertEqual(plan['settings']['verify_temperature'], .6)
        self.assertEqual(plan['settings']['request_timeout'], 1800)
        job = self.root / 'job'
        job.mkdir()
        with patch('mathagent.benchmark.ProofRunner') as runner:
            runner.return_value.start.return_value = {'answer': 'proof', 'status': 'candidate_complete'}
            benchmark._proof(SimpleNamespace(workspace=SimpleNamespace(root=job)), 'goal', job, args)
        kwargs = runner.return_value.start.call_args.kwargs
        self.assertEqual((kwargs['min_solve_tokens'], kwargs['verify_temperature']), (16384, .6))


    def test_continuation_fits_remaining_budget_and_reserves_full_review(self):
        client = ExactClient([[event('Partial derivation.', count=32768, reason='length')],
                              [event('Completed written proof.', count=20000)], [event(approval(), count=10000)]])
        runner, result = self.start(client, max_tokens=72000)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(runner.state['rounds_started'], 2)
        self.assertEqual([r['options']['num_predict'] for r in client.requests],
                         [32768, 72000 - 32768 - 16384, 16384])
        self.assertLessEqual(runner.state['tokens_charged'], 72000)

    def test_continuation_still_requires_minimum_solve_and_full_review(self):
        client = ExactClient([[event('Partial derivation.', count=32768, reason='length')]])
        runner, result = self.start(client, max_tokens=54930)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(runner.state['rounds_started'], 1)
        self.assertEqual(len(client.requests), 1)

    def test_repair_allowance_is_separate_and_gates_the_second_attempt(self):
        client = ExactClient([[event(PROOF_7KB, count=20000)], [event(LONG_OBJECTION, count=9000)],
                              [event('Repaired complete proof.', count=12000)], [event(approval(), count=8000)]])
        runner, result = self.start(client, max_tokens=80000, repair_tokens=14000, min_solve_tokens=14000)
        self.assertEqual([r['options']['num_predict'] for r in client.requests], [32768, 16384, 14000, 16384])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(result['answer'], 'Repaired complete proof.')
        # A first cycle that used its full ceilings still leaves room for one bounded repair.
        runner.state['tokens_charged'] = 32768 + 16384
        runner._reserve_attempt('repair')
        runner.state['tokens_charged'] = 80000 - 14000 - 16384 + 1
        with self.assertRaisesRegex(ProofBudget, '30384 tokens must remain'):
            runner._reserve_attempt('repair')

    def test_minimum_allowance_cannot_exceed_repair_allowance(self):
        with self.assertRaisesRegex(ValueError, 'min_solve_tokens'):
            self.start(ExactClient([]), repair_tokens=14000, min_solve_tokens=16384)


if __name__ == '__main__':
    unittest.main()
