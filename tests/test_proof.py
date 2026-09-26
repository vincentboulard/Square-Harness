"""Controller tests exercise untrusted verdicts, task selection and compaction."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from mathagent.agent import Agent
from mathagent.ledger import ProofStore
from mathagent.proof import ProofRunner, ProofContext, _normal
from mathagent.tools import Workspace


def response(text='', *, thinking='', complete=True, count=10):
    if not isinstance(text, str):
        text = json.dumps(text)
    return [{'message': {'content': text, 'thinking': thinking}, 'done': True,
             'done_reason': 'stop' if complete else 'length', 'eval_count': count}]


def review(**changes):
    value = {'critical_claim': 'For every real x, x=x.', 'assumptions': ['x is real'],
             'dependencies': [], 'argument': 'Equality is reflexive for every real x.',
             'disposition': 'supported', 'objection': '', 'evidence': '',
             'next_task': 'Audit the complete proof.', 'new_progress': True,
             'complete_candidate': False, 'resolves': [], 'resolution': ''}
    value.update(changes)
    return value


def batch(*claims, **changes):
    value = {'claims': list(claims) or [review()], 'complete_candidate': True,
             'next_task': 'Audit the complete proof.',
             'strategy_summary': 'Use reflexivity for arbitrary real x.'}
    value.update(changes)
    return value


def critic(**changes):
    value = {'valid_steps': ['For arbitrary real x, reflexivity gives x=x.'],
             'first_invalid_step': '', 'reason': '', 'missing_work': '',
             'complete_candidate': True}
    value.update(changes)
    return value


def plan(index=1):
    return {'approach': f'Route {index}: inspect the exact obligation.',
            'task': f'Establish a precise independently justified claim in route {index}.',
            'difference': 'Investigate the unresolved inference rather than repeat it.',
            'deliverable': f'The exact claim for route {index} and its supporting argument.'}


def audit(**changes):
    value = {'verdict': 'complete', 'explanation': 'Reflexivity applies to arbitrary real x.',
             'objection': '', 'next_task': ''}
    value.update(changes)
    return value


class FakeClient:
    host = 'http://fake.invalid'
    timeout = 60

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        try:
            events = next(self.replies)
        except StopIteration:
            raise AssertionError('Unexpected inference call')
        for event in events:
            if isinstance(event, BaseException):
                raise event
            yield event


class ProofControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def runner(self, replies, **options):
        self.client = FakeClient(replies)
        return ProofRunner(Agent(self.client, Workspace(self.root), ctx=options.get('ctx', 16384),
                                 predict=options.get('predict', 2048)))

    def test_success_requires_distinct_full_audit_and_offline_resume(self):
        runner = self.runner([response('For arbitrary real x, reflexivity gives x=x.'),
                              response(critic()), response(audit())])
        result = runner.start('Prove x=x for every real x.')
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual([c['role'] for c in runner.state['calls']], ['solver', 'critic', 'auditor'])
        self.assertEqual(runner.state['tokens_charged'], 30)
        self.assertEqual(runner.state['rounds_started'], 1)
        self.assertFalse(self.client.requests[1]['think'])
        self.assertNotIn('tools', self.client.requests[2])
        before = copy.deepcopy(runner.state)
        runner.resume(result['id'])
        self.assertEqual(runner.state, before)
        self.assertEqual(len(self.client.requests), 3)

    def test_truncated_thinking_is_reviewed_and_objection_survives_next_round(self):
        objection = 'The claimed bound fails at the admissible value x=0.'
        bad = review(critical_claim='Every real x has x>0.', disposition='refuted',
                     objection=objection, evidence='x=0 gives 0>0, which is false.',
                     complete_candidate=True, next_task='Use the definition of equality.')
        runner = self.runner([response(thinking='I keep trying x>0.', complete=False),
                              response('PARTIAL: The proposed positivity argument remains unsupported.'),
                              response(critic(missing_work='A valid argument is still needed.', complete_candidate=False)),
                              response(batch(bad, complete_candidate=False)), response('For arbitrary x, equality is reflexive.'),
                              response(critic()), response(audit())])
        result = runner.start('Prove x=x for every real x.', max_rounds=2)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(runner.state['claims'][0]['status'], 'refuted')
        self.assertIn('I keep trying x>0.', self.client.requests[1]['messages'][1]['content'])
        self.assertEqual(runner.state['calls'][1]['role'], 'checkpoint')
        self.assertIn(objection, self.client.requests[4]['messages'][1]['content'])
        self.assertEqual(runner.state['rounds_started'], 2)
        self.assertEqual(len([c for c in runner.state['calls'] if c['role'] == 'auditor']), 1)

    def test_repeated_rejected_claim_cannot_be_promoted_without_resolution(self):
        bad = review(disposition='refuted', objection='This derivation uses an extra assumption.',
                     evidence='It assumes x>0 although x=0 is allowed.', complete_candidate=False)
        runner = self.runner([response('An attempted proof.'), response(critic(complete_candidate=False)), response(batch(bad)),
                              response('I now assert the same conclusion.'), response(critic(complete_candidate=False)),
                              response(batch()), response(batch())])
        runner.start('Prove x=x.', max_rounds=2)
        self.assertEqual([c['status'] for c in runner.state['claims']], ['refuted'])
        self.assertEqual(runner.state['status'], 'stalled')
        self.assertIn('unresolved recorded objections', runner.state['stop_reason'])
        self.assertIsNone(runner.state['final_audit'])
        self.assertEqual(len(self.client.requests), 7)

    def test_explicit_repair_is_new_record_old_objection_is_preserved(self):
        bad = review(disposition='gap', objection='The argument assumes positivity.',
                     evidence='x=0 is admissible.', complete_candidate=False)
        fixed = review(resolves=['C1'], resolution='Reflexivity applies also when x=0; positivity is unused.')
        runner = self.runner([response('A flawed attempt.'), response(critic(complete_candidate=False)), response(batch(bad)),
                              response('For every real x, reflexivity gives x=x.'),
                              response(critic(complete_candidate=False)), response(batch(fixed, complete_candidate=False)),
                              response('For arbitrary real x, reflexivity gives x=x.'),
                              response(critic()), response(audit())])
        self.assertEqual(runner.start('Prove x=x.', max_rounds=3)['status'], 'candidate_complete')
        self.assertEqual(runner.state['claims'][0]['status'], 'gap')
        self.assertEqual(runner.state['claims'][1]['resolves'], ['C1'])

    def test_stagnation_invokes_concrete_planner_and_continues_after_four_rounds(self):
        stuck = review(disposition='uncertain', critical_claim='', argument='',
                       new_progress=False, objection='No derivation.')
        replies = []
        for n in range(1, 6):
            if n >= 3:
                replies.append(response(plan(n)))
            replies.extend([response('Still stuck.'),
                            response(critic(missing_work='No derivation.', complete_candidate=False)),
                            response(batch(stuck, complete_candidate=False))])
        runner = self.runner(replies)
        result = runner.start('Prove x=x.', max_rounds=5)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual(runner.state['rounds_started'], 5)
        self.assertEqual(len(runner.state['strategies']), 3)
        self.assertEqual([s['round'] for s in runner.state['strategies']], [3, 4, 5])
        self.assertIn(plan(3)['task'], runner.state['rounds'][2]['task'])
        self.assertEqual(len(runner.state['claims']), 1, 'Identical records must be reused, not duplicated')
        self.assertEqual(runner.state['rounds'][4]['claim_ids'], ['C1'])

    def test_bad_review_json_and_truncated_verdict_never_mean_success(self):
        for verdict in (response('{incomplete'), response(batch(), complete=False),
                        response(batch(complete_candidate='true'))):
            with self.subTest(verdict=verdict):
                runner = self.runner([response('Some proof.'), response(critic(complete_candidate=False)), verdict, verdict])
                result = runner.start('Prove x=x.', max_rounds=1)
                self.assertEqual(result['status'], 'stalled')
                self.assertEqual(runner.state['claims'], [])
                self.assertIn('Invalid recorder JSON', runner.state['stop_reason'])
                self.assertIsNone(runner.state['final_audit'])

    def test_empty_or_contradictory_audit_cannot_complete(self):
        for verdict in (audit(explanation=''), audit(objection='Missing the case x=0.')):
            with self.subTest(verdict=verdict):
                runner = self.runner([response('x=x by reflexivity.'), response(critic()), response(verdict)])
                runner.start('Prove x=x.', max_rounds=1)
                self.assertEqual(runner.state['final_audit']['verdict'], 'uncertain')
                self.assertNotEqual(runner.state['status'], 'candidate_complete')

    def test_verdict_checks_dependencies_evidence_and_remaining_objections(self):
        invalid = [review(dependencies=['C999']), review(resolves=['C999'], resolution='Fixed.'),
                   review(objection='An essential step is missing.'),
                   review(disposition='refuted', objection='Maybe false.', evidence='')]
        for value in invalid:
            with self.subTest(value=value):
                invalid_references = bool(value['dependencies'] or value['resolves'])
                replies = [response('Attempt.'), response(critic(complete_candidate=False)), response(batch(value))]
                if invalid_references:
                    replies.append(response(batch(value)))
                runner = self.runner(replies)
                runner.start('Prove x=x.', max_rounds=1)
                if invalid_references:
                    self.assertEqual(runner.state['status'], 'stalled')
                    self.assertEqual(runner.state['claims'], [])
                else:
                    self.assertEqual(runner.state['claims'][0]['status'], 'gap')
                self.assertIsNone(runner.state['final_audit'])

    def test_separate_atomic_results_survive_with_the_remaining_obligation(self):
        first = review(critical_claim='For real x, x+0=x.', argument='Zero is the additive identity.')
        second = review(critical_claim='For real x, x*1=x.', argument='One is the multiplicative identity.')
        unresolved = review(critical_claim='The two identities imply the original goal.',
                            disposition='gap', argument='', objection='No connecting argument was supplied.',
                            new_progress=False)
        next_task = 'Write the missing connecting argument explicitly.'
        runner = self.runner([response('Two independently useful identities, but no complete derivation.'),
                              response(critic(missing_work=next_task, complete_candidate=False)),
                              response(batch(first, second, unresolved, complete_candidate=False,
                                             next_task=next_task))])
        result = runner.start('Prove x=x for real x.', max_rounds=1)
        self.assertEqual(result['status'], 'budget_exhausted')
        self.assertEqual([c['status'] for c in runner.state['claims']], ['reviewed', 'reviewed', 'gap'])
        self.assertEqual(runner.state['rounds'][0]['claim_ids'], ['C1', 'C2', 'C3'])
        self.assertEqual(runner.state['claims'][0]['argument'], first['argument'])
        self.assertEqual(runner.state['claims'][1]['argument'], second['argument'])
        self.assertEqual(runner.state['next_task'], next_task)
        self.assertIsNone(runner.state['final_audit'])

    def test_fresh_critic_objection_cannot_be_discarded_by_friendly_recorder(self):
        bad_step = 'Every admissible x is positive.'
        reason = 'The original hypotheses allow x=0.'
        runner = self.runner([response('Assume x>0 and conclude x=x.'),
                              response(critic(first_invalid_step=bad_step, reason=reason,
                                              missing_work='Handle the remaining admissible values.')),
                              response(batch())])
        runner.start('Prove x=x for every real x.', max_rounds=1)
        self.assertIsNone(runner.state['final_audit'])
        objection = next(c for c in runner.state['claims'] if c['statement'] == bad_step)
        self.assertEqual(objection['status'], 'gap')
        self.assertEqual(objection['objection'], reason)
        self.assertEqual([c['role'] for c in runner.state['calls']], ['solver', 'critic', 'recorder'])

    def test_critic_is_blind_to_prior_ledger_while_recorder_can_use_it(self):
        prior = review(argument='PREVIOUS_REVIEW_MARKER: reflexivity was considered.')
        runner = self.runner([response('A partial observation.'),
                              response(critic(missing_work='Write the full derivation.', complete_candidate=False)),
                              response(batch(prior, complete_candidate=False)),
                              response('For arbitrary x, x=x by reflexivity.'),
                              response(critic(complete_candidate=False)),
                              response(batch(complete_candidate=False))])
        runner.start('Prove x=x for every real x.', max_rounds=2)
        self.assertIn('PREVIOUS_REVIEW_MARKER', self.client.requests[3]['messages'][1]['content'])
        self.assertNotIn('PREVIOUS_REVIEW_MARKER', json.dumps(self.client.requests[4]['messages']))
        self.assertIn('PREVIOUS_REVIEW_MARKER', self.client.requests[5]['messages'][1]['content'])

    def test_invalid_independent_critique_blocks_completion(self):
        for verdict in (response('{broken'), response(critic(), complete=False),
                        response(critic(complete_candidate='true'))):
            with self.subTest(verdict=verdict):
                runner = self.runner([response('For arbitrary x, x=x.'), verdict, response(batch())])
                runner.start('Prove x=x.', max_rounds=1)
                self.assertIsNone(runner.state['final_audit'])
                self.assertFalse(runner.state['rounds'][0]['critique']['complete_candidate'])

    def test_repeated_truncation_expands_budget_then_requests_direct_written_work(self):
        stuck = review(disposition='uncertain', critical_claim='', argument='',
                       objection='No usable proof yet.', new_progress=False)
        replies = []
        for _ in range(2):
            replies.extend([response(thinking='An unfinished calculation.', complete=False),
                            response('PARTIAL: The calculation is not justified.'),
                            response(critic(missing_work='Justify the calculation.', complete_candidate=False)),
                            response(batch(stuck, complete_candidate=False))])
        replies.extend([response(plan(3)), response('For arbitrary x, x=x by reflexivity.'),
                        response(critic()), response(audit())])
        runner = self.runner(replies, predict=2048)
        self.assertEqual(runner.start('Prove x=x.', max_rounds=3)['status'], 'candidate_complete')
        solver_payloads = [request for request, call in zip(self.client.requests, runner.state['calls'])
                           if call['role'] == 'solver']
        self.assertEqual([p['options']['num_predict'] for p in solver_payloads], [2048, 4096, 8192])
        self.assertEqual([p['think'] for p in solver_payloads], [True, True, False])
        self.assertEqual(len([c for c in runner.state['calls'] if c['role'] == 'checkpoint']), 2)

    def test_compaction_preserves_exact_records_and_guard_for_unseen_references(self):
        runner = self.runner([])
        runner.store = ProofStore.create(self.root, 'Prove x=x.',
            {'max_rounds': 10, 'max_tokens': 60000, 'max_seconds': 1800})
        for n in range(1, 12):
            runner.state['claims'].append({'id': f'C{n}', 'round': n, 'status': 'reviewed',
                'statement': f'Claim {n}', 'argument': 'Long supporting argument. ' * 200,
                'objection': '', 'evidence': '', 'resolves': []})
        before = copy.deepcopy(runner.state['claims'])
        messages = runner._context('Solver', 'Continue.', output=2048)
        self.assertIn('Records omitted', messages[1]['content'])
        self.assertEqual(runner.state['claims'], before)
        record = json.loads(runner._tool('read_proof_claim', {'claim_id': 'C1'}))
        self.assertEqual(json.loads(record['content']), before[0])
        value = review(dependencies=['C1'], _visible_claims=['C11'])
        claim = runner._record(value, {'index': 12, 'draft': 'candidate.md'}, {'complete': True})
        self.assertEqual(claim['status'], 'gap')
        self.assertIn('could not inspect', claim['objection'])

    def test_original_theorem_cannot_be_truncated_to_fit(self):
        (self.root / 'large.tex').write_text('A large theorem statement. ' * 3000)
        runner = self.runner([], ctx=8192)
        result = runner.start('Prove large.tex.')
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(self.client.requests, [])
        self.assertEqual(runner.state['sources'][0]['content'], (self.root / 'large.tex').read_text())

    def test_exact_duplicate_detection_respects_mathematical_case(self):
        self.assertNotEqual(_normal('A = b'), _normal('a = B'))
        self.assertNotEqual(_normal('a b'), _normal('ab'))
        self.assertEqual(_normal('A  =\n b'), _normal('A = b'))

    def test_explicit_source_with_spaces_disables_partial_filename_inference(self):
        (self.root / 'my lemma.tex').write_text('Prove x=x for real x.')
        runner = self.runner([response('x=x by reflexivity.'), response(critic()), response(audit())])
        runner.start('Prove my lemma.tex.', source_files=['my lemma.tex'])
        self.assertEqual([s['path'] for s in runner.state['sources']], ['my lemma.tex'])

    def test_fresh_context_retains_whole_proof_objection(self):
        runner = self.runner([response('x=x by reflexivity.'), response(critic()),
                              response(audit(verdict='gap', objection='The global limiting step is missing.'))])
        runner.start('Prove x=x.', max_rounds=1)
        messages = runner._context('Solver', 'New route.', fresh=True)
        self.assertIn('The global limiting step is missing.', messages[1]['content'])

    def test_success_after_pause_clears_obsolete_stop_reason(self):
        runner = self.runner([response('x=x by reflexivity.'), [KeyboardInterrupt()],
                              response(critic()), response(audit())])
        with self.assertRaises(KeyboardInterrupt):
            runner.start('Prove x=x.')
        result = runner.resume(runner.state['id'])
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertNotIn('stop_reason', runner.state)
        self.assertIn('passed a whole-proof model audit', result['report'])
        self.assertIn('## Mathematical progress', result['report'])
        self.assertIn('x=x by reflexivity.', (runner.store.directory / 'ledger.md').read_text())


if __name__ == '__main__':
    unittest.main()
