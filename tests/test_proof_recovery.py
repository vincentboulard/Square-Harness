"""Regression coverage for proof resumption across interruption and lost state."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

from mathagent.agent import Agent
from mathagent.ledger import ProofStore
from mathagent.proof import ProofRunner
from mathagent.tools import Workspace


def done(text, count=20):
    return {'message': {'content': text}, 'done': True,
            'done_reason': 'stop', 'eval_count': count}


def review(*, complete=False, two_claims=False):
    claim = {
        'critical_claim': 'Every real x satisfies x=x.' if complete else 'The attempted positivity assertion fails at zero.',
        'assumptions': [], 'dependencies': [],
        'argument': 'Equality is reflexive for each real x.' if complete else 'Zero is a real number and does not satisfy 0>0.',
        'disposition': 'supported' if complete else 'refuted',
        'objection': '' if complete else 'The attempted strict positivity assertion is false.',
        'evidence': 'Reflexivity of equality.' if complete else 'The admissible value x=0 is a counterexample.',
        'next_task': 'Audit the complete proof.' if complete else 'Give a derivation that also covers zero.',
        'new_progress': True, 'complete_candidate': False, 'resolves': [], 'resolution': '',
    }
    claims = [claim]
    if two_claims:
        claims.append(dict(claim, critical_claim='In particular, zero equals zero.',
                           argument='Apply reflexivity of equality to the real number zero.'))
    return {'claims': claims, 'complete_candidate': complete,
            'next_task': claim['next_task'],
            'strategy_summary': 'Direct use of equality.' if complete else 'A proposed positivity argument failed at zero.'}


def critic(*, complete=False):
    return {'valid_steps': ['Reflexivity covers every real x.'] if complete else [],
            'first_invalid_step': '' if complete else 'Every real x is strictly positive.',
            'reason': '' if complete else 'Zero is allowed and is not strictly positive.',
            'missing_work': '' if complete else 'A derivation covering every real x is still needed.',
            'complete_candidate': complete}


CHECKPOINT = 'PARTIAL. The attempted assertion that every real x is positive fails at zero. No complete proof was obtained.'
PLAN = {'approach': 'Inspect the defining relation.', 'task': 'Use the definition of equality for arbitrary real x.',
        'difference': 'Do not impose strict positivity.', 'deliverable': 'Write a derivation covering zero and every other real value.'}


AUDIT = {'verdict': 'complete', 'explanation': 'Reflexivity applies to every real x; all stated quantifiers are covered.',
         'objection': '', 'next_task': ''}


class ScriptedClient:
    host = 'http://local-test-model.invalid'
    timeout = 600

    def __init__(self, scripts=()):
        self.scripts = iter(scripts)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        try:
            script = next(self.scripts)
        except StopIteration:
            raise AssertionError('Unexpected inference call') from None
        for event in script:
            if isinstance(event, BaseException):
                raise event
            yield event


class ProofRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = Workspace(self.root)
        self.settings = {'max_rounds': 1, 'max_tokens': 4096, 'max_seconds': 60,
                         'model': 'test-model', 'ctx': 16384, 'predict': 512,
                         'think': True, 'host': ScriptedClient.host, 'max_predict': 2048}

    def runner(self, client, emit=lambda kind, text: None):
        return ProofRunner(Agent(client, self.workspace, model='test-model', ctx=16384,
                                 predict=512), emit)

    def seeded_solver(self, *, call_status='running', complete=False, status='paused',
                      seconds=0, checkpoint=100, max_seconds=60):
        settings = dict(self.settings, max_seconds=max_seconds)
        store = ProofStore.create(self.root, 'Prove that every real x satisfies x=x.', settings)
        stream = store.start_stream('solver')
        text = 'Let x be an arbitrary real number. By reflexivity, x=x. This proves the statement.'
        event = done(text) if complete else {'message': {'thinking': 'The positivity assertion needs a counterexample.'}, 'done': False}
        store.append_stream(stream, event)
        charged = 20 if complete and call_status == 'complete' else 512
        call = {'role': 'solver', 'round': 1, 'status': call_status,
                'reserved_tokens': 512, 'stream': stream}
        if call_status == 'complete':
            call['charged_tokens'] = charged
        store.state.update(status=status, rounds_started=1, tokens_charged=charged,
                           seconds_used=seconds, active_checkpoint_wall=checkpoint,
                           pending={'index': 1, 'phase': 'solve', 'fresh': False,
                                    'task': 'Establish one intermediate claim.'}, calls=[call])
        store.save()
        return store, stream, text

    def test_interrupted_solver_resumes_review_of_partial_in_same_round(self):
        scratch = 'At the admissible value x=0, the attempted positivity assertion would say 0>0.'
        client = ScriptedClient([[{'message': {'thinking': scratch}, 'done': False}, KeyboardInterrupt()]])
        runner = self.runner(client)
        with self.assertRaises(KeyboardInterrupt):
            runner.start('Prove that every real x satisfies x=x.', max_rounds=1,
                         max_tokens=4096, max_seconds=60)
        proof_id = runner.state['id']
        paused = ProofStore.load(self.root, proof_id)
        self.assertEqual(paused.state['status'], 'paused')
        self.assertEqual(paused.state['rounds_started'], 1)
        self.assertEqual(paused.state['tokens_charged'], 512)
        self.assertEqual(paused.state['calls'][0]['status'], 'interrupted')
        stream = paused.state['pending']['partial']
        self.assertIn(scratch, paused.read_artifact(stream))

        resumed_client = ScriptedClient([[done(CHECKPOINT, 25)], [done(json.dumps(critic()), 15)],
                                         [done(json.dumps(review()), count=30)]])
        result = self.runner(resumed_client).resume(proof_id)
        final = ProofStore.load(self.root, proof_id)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(final.state['rounds_started'], 1)
        self.assertEqual(final.state['tokens_charged'], 582)
        self.assertEqual([c['role'] for c in final.state['calls']], ['solver', 'checkpoint', 'critic', 'recorder'])
        self.assertEqual(len(resumed_client.requests), 3)
        self.assertIn('format', resumed_client.requests[1])
        self.assertIn(scratch, resumed_client.requests[0]['messages'][-1]['content'])
        self.assertIn('refuted', result['report'])
        round_data = final.state['rounds'][0]
        raw = json.loads(final.read_artifact(round_data['raw_draft']))
        checkpoint = json.loads(final.read_artifact(round_data['draft']))
        self.assertFalse(raw['complete'])
        self.assertEqual(raw['thinking'], scratch)
        self.assertFalse(checkpoint['complete'])
        self.assertIn('PARTIAL', checkpoint['text'])

    def test_completed_solver_before_draft_commit_is_recovered_without_rerunning(self):
        store, stream, text = self.seeded_solver(call_status='complete', complete=True)
        client = ScriptedClient([[done(json.dumps(critic(complete=True)), 15)],
                                 [done(json.dumps(review(complete=True)), 30)],
                                 [done(json.dumps(AUDIT), 25)]])
        result = self.runner(client).resume(store.state['id'])
        final = ProofStore.load(self.root, store.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(final.state['rounds_started'], 1)
        self.assertEqual(final.state['tokens_charged'], 90)
        self.assertEqual([c['role'] for c in final.state['calls']], ['solver', 'critic', 'recorder', 'auditor'])
        self.assertEqual(len(client.requests), 3)
        self.assertTrue(all('format' in request for request in client.requests))
        self.assertIn(text, client.requests[0]['messages'][-1]['content'])
        self.assertIn(text, result['report'])
        candidate = json.loads(final.read_artifact(final.state['rounds'][0]['draft']))
        self.assertEqual(candidate['stream'], stream)
        self.assertTrue(candidate['complete'])

    def test_running_request_reservation_and_unclean_elapsed_time_survive_resume(self):
        store, stream, _ = self.seeded_solver(status='running', seconds=5, checkpoint=100,
                                            max_seconds=20)
        client = ScriptedClient()
        with mock.patch('mathagent.proof.time.time', return_value=120), \
             mock.patch('mathagent.proof.time.monotonic', return_value=200):
            result = self.runner(client).resume(store.state['id'])
        final = ProofStore.load(self.root, store.state['id'])
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(client.requests, [])
        self.assertEqual(final.state['seconds_used'], 25)
        self.assertEqual(final.state['tokens_charged'], 512)
        self.assertEqual(final.state['rounds_started'], 1)
        self.assertEqual(final.state['calls'][0]['status'], 'interrupted')
        self.assertEqual(final.state['pending']['partial'], stream)

    def test_clean_pause_does_not_charge_offline_time_or_reset_token_reservation(self):
        store, _, _ = self.seeded_solver(call_status='interrupted', status='paused', seconds=5,
                                        checkpoint=100, max_seconds=10)
        client = ScriptedClient([[done(CHECKPOINT, 25)], [done(json.dumps(critic()), 15)],
                                 [done(json.dumps(review()), 30)]])
        with mock.patch('mathagent.proof.time.time', return_value=10000), \
             mock.patch('mathagent.proof.time.monotonic', return_value=200):
            self.runner(client).resume(store.state['id'])
        final = ProofStore.load(self.root, store.state['id'])
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(final.state['seconds_used'], 5)
        self.assertEqual(final.state['tokens_charged'], 582)
        self.assertEqual(final.state['rounds_started'], 1)

    def test_backup_recovery_disables_inference_even_when_backup_lost_reservation(self):
        store = ProofStore.create(self.root, 'Prove the original statement.', self.settings)
        store.state['tokens_charged'] = 512
        store.save()  # The backup still has the original zero charge.
        (store.directory / 'state.json').write_text('{"interrupted":')
        client = ScriptedClient()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = self.runner(client).resume(store.state['id'])
        self.assertTrue(any('Recovered' in str(warning.message) for warning in caught))
        self.assertEqual(result['status'], 'needs_recovery')
        self.assertEqual(client.requests, [])
        self.assertIn('Automatic inference is disabled', result['report'])
        second_client = ScriptedClient()
        again = self.runner(second_client).resume(store.state['id'])
        self.assertEqual(again['status'], 'needs_recovery')
        self.assertEqual(second_client.requests, [])

    def test_exhausted_token_reservation_cannot_reset_by_resuming(self):
        store, stream, _ = self.seeded_solver(call_status='interrupted')
        store.state['tokens_charged'] = self.settings['max_tokens']
        store.save()
        client = ScriptedClient()
        result = self.runner(client).resume(store.state['id'])
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(client.requests, [])
        final = ProofStore.load(self.root, store.state['id'])
        self.assertEqual(final.state['tokens_charged'], self.settings['max_tokens'])
        self.assertEqual(final.state['rounds_started'], 1)
        draft = json.loads(final.read_artifact(final.state['pending']['draft']))
        self.assertFalse(draft['complete'])
        self.assertEqual(draft['stream'], stream)
        self.assertIn('needs a counterexample', draft['thinking'])
        repeated = self.runner(ScriptedClient()).resume(store.state['id'])
        self.assertEqual(repeated['status'], 'budget_exhausted')

    def test_source_aliases_read_pinned_snapshot_after_file_changes(self):
        original = 'Original statement: every real x satisfies x=x.\n'
        path = self.root / 'lemma.tex'
        path.write_text(original)
        source = {'path': 'lemma.tex', 'content': original,
                  'sha256': hashlib.sha256(original.encode()).hexdigest()}
        store = ProofStore.create(self.root, 'Prove the pinned statement.', self.settings, [source])
        path.write_text('CHANGED STATEMENT: every real x satisfies x=0.\n')
        (self.root / 'alias.tex').symlink_to(path)
        runner = self.runner(ScriptedClient())
        runner.store = ProofStore.load(self.root, store.state['id'])
        for alias in ('lemma.tex', './lemma.tex', 'alias.tex'):
            with self.subTest(alias=alias):
                result = runner._tool('read_file', {'path': alias})
                self.assertIn('PINNED ORIGINAL SOURCE', result)
                self.assertIn(original.strip(), result)
                self.assertNotIn('CHANGED STATEMENT', result)
        self.assertIn(original.strip(), runner._base())

    def test_interrupt_after_batch_notice_does_not_duplicate_claims_on_resume(self):
        client = ScriptedClient([
            [done('For every real x, equality is reflexive, so x=x.')],
            [done(json.dumps(critic(complete=True)))],
            [done(json.dumps(review(complete=True, two_claims=True)))],
        ])

        def interrupt_after_commit(kind, text):
            if kind == 'notice' and text.startswith('C1:'):
                raise KeyboardInterrupt()

        runner = self.runner(client, interrupt_after_commit)
        with self.assertRaises(KeyboardInterrupt):
            runner.start('Prove x=x for every real x.', max_rounds=1,
                         max_tokens=16000, max_seconds=60)
        paused = ProofStore.load(self.root, runner.state['id'])
        self.assertEqual(paused.state['pending']['phase'], 'audit')
        before = copy.deepcopy(paused.state['claims'])
        self.assertEqual([claim['id'] for claim in before], ['C1', 'C2'])
        resumed = ScriptedClient([[done(json.dumps(AUDIT), 25)]])
        result = self.runner(resumed).resume(paused.state['id'])
        final = ProofStore.load(self.root, paused.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(final.state['claims'], before)
        self.assertEqual(len(final.state['rounds']), 1)
        self.assertEqual(final.state['rounds_started'], 1)
        self.assertEqual(len(resumed.requests), 1)
        self.assertEqual(final.state['calls'][-1]['role'], 'auditor')

    def test_batch_interruption_rolls_back_all_records_before_retry(self):
        client = ScriptedClient([
            [done('For every real x, equality is reflexive, so x=x.')],
            [done(json.dumps(critic(complete=True)))],
            [done(json.dumps(review(complete=True, two_claims=True)))],
        ])
        runner = self.runner(client)
        original = runner._record

        def interrupted_record(*args, **kwargs):
            record = original(*args, **kwargs)
            if record['id'] == 'C2':
                raise KeyboardInterrupt()
            return record

        with mock.patch.object(runner, '_record', side_effect=interrupted_record):
            with self.assertRaises(KeyboardInterrupt):
                runner.start('Prove x=x for every real x.', max_rounds=1,
                             max_tokens=16000, max_seconds=60)
        paused = ProofStore.load(self.root, runner.state['id'])
        self.assertEqual(paused.state['pending']['phase'], 'review')
        self.assertEqual(paused.state['claims'], [])
        self.assertEqual(paused.state['stagnant_rounds'], 0)
        # The valid recorder response was committed before ledger mutation;
        # retry only the atomic local batch, without paying for another call.
        resumed = ScriptedClient([[done(json.dumps(AUDIT))]])
        result = self.runner(resumed).resume(paused.state['id'])
        final = ProofStore.load(self.root, paused.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([claim['id'] for claim in final.state['claims']], ['C1', 'C2'])
        self.assertEqual([call['role'] for call in final.state['calls']],
                         ['solver', 'critic', 'recorder', 'auditor'])
        self.assertEqual(final.state['rounds_started'], 1)

    def seeded_phase(self, phase, role, *, done_before_commit=False):
        store = ProofStore.create(self.root, 'Prove x=x for every real x.',
                                  dict(self.settings, max_tokens=16000))
        raw = {'text': 'For every real x, equality is reflexive, so x=x.',
               'thinking': '', 'calls': [], 'complete': phase != 'checkpoint', 'stats': {}}
        if phase == 'checkpoint':
            raw.update(text='', thinking='Equality is reflexive, but the argument was interrupted.')
        candidate = store.write_artifact('candidate', json.dumps(raw))
        pending = {'index': 1, 'phase': phase, 'fresh': False, 'think': True,
                   'task': 'Use the defining relation.', 'draft': candidate,
                   'solver_truncated': phase == 'checkpoint'}
        if phase in {'review', 'audit'}:
            pending['critique'] = critic(complete=True)
        stream = store.start_stream(role)
        event = done('UNCOMMITTED RESPONSE', 23) if done_before_commit else {
            'message': {'content': '{"incomplete":'}, 'done': False}
        store.append_stream(stream, event)
        reserved = 1024
        charged = 23 if done_before_commit else reserved
        call = {'role': role, 'round': 1, 'status': 'complete' if done_before_commit else 'running',
                'reserved_tokens': reserved, 'stream': stream}
        if done_before_commit:
            call['charged_tokens'] = charged
        store.state.update(status='paused', rounds_started=1, tokens_charged=charged,
                           calls=[call], pending=pending)
        store.save()
        return store, charged

    def test_all_interrupted_non_solver_phases_retry_with_original_charge(self):
        cases = [('plan', 'planner'), ('checkpoint', 'checkpoint'), ('critic', 'critic'),
                 ('review', 'recorder'), ('audit', 'auditor')]
        for phase, role in cases:
            with self.subTest(phase=phase):
                store, charged = self.seeded_phase(phase, role)
                tail = [[done(json.dumps(critic(complete=True)))],
                        [done(json.dumps(review(complete=True)))], [done(json.dumps(AUDIT))]]
                if phase == 'plan':
                    scripts = [[done(json.dumps(PLAN))], [done('Every real x equals itself.')]] + tail
                elif phase == 'checkpoint':
                    scripts = [[done(CHECKPOINT)], [done(json.dumps(critic()))],
                               [done(json.dumps(review()))]]
                elif phase == 'critic':
                    scripts = tail
                elif phase == 'review':
                    scripts = tail[1:]
                else:
                    scripts = tail[2:]
                client = ScriptedClient(scripts)
                result = self.runner(client).resume(store.state['id'])
                final = ProofStore.load(self.root, store.state['id'])
                self.assertEqual(final.state['calls'][0]['status'], 'interrupted')
                self.assertEqual(final.state['calls'][1]['role'], role)
                self.assertEqual(final.state['tokens_charged'], charged + 20 * len(scripts))
                self.assertEqual(final.state['rounds_started'], 1)
                self.assertEqual(len(client.requests), len(scripts))
                self.assertEqual(result['status'], 'budget_exhausted' if phase == 'checkpoint' else 'candidate_complete')

    def test_completed_uncommitted_critic_is_not_treated_as_a_committed_verdict(self):
        store, charged = self.seeded_phase('critic', 'critic', done_before_commit=True)
        client = ScriptedClient([[done(json.dumps(critic(complete=True)))],
                                 [done(json.dumps(review(complete=True)))],
                                 [done(json.dumps(AUDIT))]])
        result = self.runner(client).resume(store.state['id'])
        final = ProofStore.load(self.root, store.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([call['role'] for call in final.state['calls']],
                         ['critic', 'critic', 'recorder', 'auditor'])
        self.assertEqual(final.state['tokens_charged'], charged + 60)
        self.assertNotIn('UNCOMMITTED RESPONSE', result['report'])

    def test_terminal_audit_result_does_not_restart_inference(self):
        store, _ = self.seeded_phase('finish', 'auditor', done_before_commit=True)
        store.state['status'] = 'candidate_complete'
        store.state['final_audit'] = {'round': 1, 'candidate': store.state['pending']['draft'], **AUDIT}
        store.save()
        before = copy.deepcopy(store.state)
        client = ScriptedClient()
        result = self.runner(client).resume(store.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(client.requests, [])
        self.assertEqual(ProofStore.load(self.root, store.state['id']).state, before)

    def test_resume_keeps_saved_adaptive_cap_and_model_settings(self):
        store, _ = self.seeded_phase('plan', 'planner')
        store.state['settings']['max_predict'] = 1024
        store.state['truncation_streak'] = 3
        store.save()
        client = ScriptedClient([[done(json.dumps(PLAN))], [done('Every real x equals itself.')],
                                 [done(json.dumps(critic(complete=True)))],
                                 [done(json.dumps(review(complete=True)))],
                                 [done(json.dumps(AUDIT))]])
        runner = self.runner(client)
        runner.agent.predict = 4096
        result = runner.resume(store.state['id'])
        final = ProofStore.load(self.root, store.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(runner.agent.predict, 512)
        self.assertEqual(final.state['settings']['max_predict'], 1024)
        self.assertEqual(client.requests[1]['options']['num_predict'], 1024)


if __name__ == '__main__':
    unittest.main()
