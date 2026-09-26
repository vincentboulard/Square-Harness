"""Adversarial contracts and accounting for coordinated proof search."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mathagent.agent import Agent
from mathagent import cooperative
from mathagent.portfolio import _GatedClient
from mathagent.tools import Workspace


GOAL = 'For every real x and y, prove (x+y)^2=x^2+2*x*y+y^2.'
PROOF = ('For arbitrary real x and y, distributivity gives '
         '(x+y)^2=x*x+x*y+y*x+y*y. Since x*y=y*x, this is x^2+2*x*y+y^2.')


def contract(task_id='T1', **changes):
    value = {'id': task_id, 'statement': 'For real x and y, x*y=y*x.',
             'hypotheses': ['x and y are real'], 'setting': 'The real field.',
             'quantifiers': 'For every x and every y.',
             'constant_dependencies': 'No constants occur.', 'dependencies': [],
             'deliverable': 'A self-contained proof of the exact statement.'}
    value.update(changes)
    return value


def plan(route='decompose', **changes):
    value = {'route': route, 'rationale': 'Establish two independent algebraic identities.',
             'composition': 'Expand the square and then substitute commutativity into the cross term.',
             'tasks': [] if route == 'direct' else [
                 contract(), contract('T2', statement='For real x and y, (x+y)^2=x*x+x*y+y*x+y*y.')]}
    value.update(changes)
    return value


def critic(**changes):
    value = {'valid_steps': ['Distributivity and commutativity hold for all stated real numbers.'],
             'first_invalid_step': '', 'reason': '', 'missing_work': '', 'complete_candidate': True}
    value.update(changes)
    return value


def audit(**changes):
    value = {'verdict': 'complete', 'explanation': 'The expansion proves the exact identity for arbitrary real x and y.',
             'objection': '', 'next_task': ''}
    value.update(changes)
    return value


def event(value, *, count=10, reason='stop', done=True):
    return {'message': {'content': value if isinstance(value, str) else json.dumps(value)},
            'done': done, 'done_reason': reason, 'eval_count': count, 'prompt_eval_count': 5}


class Client:
    host = 'http://fake.invalid'
    timeout = 60

    def __init__(self, replies, on_request=None):
        self.replies, self.requests, self.on_request = iter(replies), [], on_request

    def stream(self, payload):
        self.requests.append(copy.deepcopy(payload))
        if self.on_request:
            self.on_request(payload)
        try:
            reply = next(self.replies)
        except StopIteration:
            raise AssertionError('Unexpected additional cooperative inference')
        if isinstance(reply, BaseException):
            raise reply
        yield from reply


class PlanContractTests(unittest.TestCase):
    def test_direct_and_acyclic_contracts_retain_exact_mathematical_data(self):
        value = plan()
        value['tasks'].append(contract('T3', statement=GOAL, dependencies=['T1', 'T2']))
        original = copy.deepcopy(value)
        cooperative.validate_plan(value)
        self.assertEqual(value, original)
        cooperative.validate_plan(plan('direct'))

    def test_malformed_or_unschedulable_plans_are_rejected(self):
        values = []
        values.append(plan(route='unsupported'))
        values.append(plan('direct', tasks=[contract()]))
        values.append(plan(tasks=[]))
        values.append(plan(tasks=[contract()]))
        values.append(plan(rationale='   '))
        values.append(plan(composition=''))
        values.append(plan(tasks=[contract(), contract('T1', statement='Another claim.')]))
        values.append(plan(tasks=[contract(), contract('T2')]))
        values.append(plan(tasks=[contract(dependencies=['T1']), contract('T2', statement='Another claim.')]))
        values.append(plan(tasks=[contract(dependencies=['T3']), contract('T2', statement='Another claim.')]))
        values.append(plan(tasks=[contract(dependencies=['T2']), contract('T2', statement='Another claim.', dependencies=['T1'])]))
        values.append(plan(tasks=[contract(), contract('T2', statement='Another claim.'),
                                  contract('T3', statement='A third claim.'), contract('T4', statement='A fourth claim.')]))
        values.append(plan(tasks=[contract(statement=''), contract('T2', statement='Another claim.')]))
        value = plan()
        del value['tasks'][0]['constant_dependencies']
        values.append(value)
        value = plan()
        value['tasks'][0]['hypotheses'] = 'x is real'
        values.append(value)
        for index, value in enumerate(values):
            with self.subTest(index=index, plan=value), self.assertRaises(ValueError):
                cooperative.validate_plan(value)

    def test_budget_is_one_finite_partition_including_all_overhead(self):
        for total in (8192, 8193, 30000, 60000):
            budget = cooperative.cooperative_budget(total)
            self.assertEqual(sum(budget.values()), total)
            self.assertTrue(all(type(value) is int and value > 0 for value in budget.values()))
        for total in (0, 8191, True, 12000.5):
            with self.subTest(total=total), self.assertRaises(ValueError):
                cooperative.cooperative_budget(total)

    def test_repair_cannot_promote_failed_or_unknown_prerequisites(self):
        results = [{'id': 'T1', 'dependency_ready': True, 'contract': contract(),
                    'candidate': {'text': 'Multiplication is commutative in the real field.', 'complete': True}},
                   {'id': 'T2', 'dependency_ready': False,
                    'contract': contract('T2', statement='Another claim.'), 'candidate': None}]
        cooperative.validate_repair({'needed': False, 'reason': 'No targeted repair is available.', 'task': None}, results)
        valid = {'needed': True, 'reason': 'Check the missing case using the established scoped identity.',
                 'task': contract('R1', statement='Establish the missing case.', dependencies=['T1'])}
        cooperative.validate_repair(valid, results)
        invalid = [
            {'needed': True, 'reason': 'Repair is needed.', 'task': None},
            {'needed': False, 'reason': 'Repair is unnecessary.', 'task': contract('R1')},
            {**valid, 'task': contract('T1')},
            {**valid, 'task': contract('R1', dependencies=['T2'])},
            {**valid, 'task': contract('R1', dependencies=['T999'])},
            {**valid, 'task': contract('R1', dependencies=['R1'])},
        ]
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                cooperative.validate_repair(value, results)


class CooperativeControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()

    def run_proof(self, replies, **options):
        self.client = Client(replies, on_request=options.pop('on_request', None))
        agent = Agent(self.client, Workspace(self.workspace), model='test',
                      ctx=options.pop('ctx', 32768), predict=256)
        return cooperative.run_cooperative_proof(agent, GOAL,
            output_dir=options.pop('output_dir', self.root / 'cooperative'),
            max_tokens=options.pop('max_tokens', 30000), max_seconds=30,
            max_rounds=1, max_predict=512, **options)

    def fake_graph(self, agent, tasks, **options):
        results = []
        for task in tasks:
            results.append({'id': task['id'], 'contract': copy.deepcopy(task),
                'status': 'finished', 'proof_status': 'candidate_complete', 'dependency_ready': True,
                'candidate': {'id': task['id'], 'text': 'EXACT_WORKER_EVIDENCE_' + task['id'], 'complete': True},
                'partial_artifacts': [], 'obligations': [], 'final_audit': audit(),
                'tokens_charged': 30, 'measured_completion_tokens': 30,
                'prompt_tokens': 15, 'reserved_unmeasured_tokens': 0, 'request_count': 3})
        return {'status': 'completed', 'results': results, 'jobs': [],
                'execution_errors': [], 'tokens_charged': 30 * len(tasks),
                'measured_completion_tokens': 30 * len(tasks), 'prompt_tokens': 15 * len(tasks),
                'reserved_unmeasured_tokens': 0, 'request_count': 3 * len(tasks)}

    def test_direct_route_charges_advisor_and_preserves_exact_goal_for_final_audit(self):
        def reserved_before_dispatch(payload):
            if len(self.client.requests) == 1:
                state = json.loads((self.root / 'cooperative' / 'state.json').read_text())
                self.assertGreaterEqual(state['tokens_charged'], payload['options']['num_predict'])

        with mock.patch.object(cooperative, 'run_task_graph') as workers:
            result = self.run_proof([[event(plan('direct'))], [event(PROOF)],
                                     [event(critic())], [event(audit())]],
                                    on_request=reserved_before_dispatch)
        workers.assert_not_called()
        self.assertEqual(len(self.client.requests), 4)
        self.assertEqual(result['tokens_charged'], 40)
        self.assertEqual(Path(result['proof_path']).read_text(), PROOF)
        audit_request = self.client.requests[-1]
        self.assertIn(GOAL, json.dumps(audit_request))
        self.assertIn(PROOF, json.dumps(audit_request))
        self.assertNotIn('FRESH INDEPENDENT CRITIQUE', json.dumps(audit_request))
        before = len(self.client.requests)
        agent = Agent(self.client, Workspace(self.workspace), model='test', ctx=32768, predict=256)
        with self.assertRaises(FileExistsError):
            cooperative.run_cooperative_proof(agent, GOAL, output_dir=self.root / 'cooperative')
        self.assertEqual(len(self.client.requests), before)

    def test_shared_gate_covers_advisor_and_all_proof_calls_without_double_acquiring(self):
        class Gate:
            def __init__(self):
                self.active, self.acquired, self.released = False, 0, 0

            def acquire(self, *, timeout):
                if self.active:
                    raise AssertionError('The same non-reentrant request gate was acquired twice')
                self.active = True
                self.acquired += 1
                return True

            def release(self):
                if not self.active:
                    raise AssertionError('A request gate was released without acquisition')
                self.active = False
                self.released += 1

        for wrapped in (False, True):
            with self.subTest(already_wrapped=wrapped):
                gate = Gate()
                client = Client([[event(plan('direct'))], [event(PROOF)],
                                 [event(critic())], [event(audit())]],
                                on_request=lambda payload: self.assertTrue(gate.active))
                supplied_client = _GatedClient(client, gate) if wrapped else client
                agent = Agent(supplied_client, Workspace(self.workspace), model='test', ctx=32768, predict=256)
                result = cooperative.run_cooperative_proof(agent, GOAL,
                    output_dir=self.root / f'gate-{wrapped}', max_tokens=30000,
                    max_seconds=30, max_rounds=1, max_predict=512, request_gate=gate)
                self.assertEqual(result['status'], 'candidate_complete')
                self.assertEqual((gate.acquired, gate.released), (4, 4))
                self.assertFalse(gate.active)
                self.assertIs(agent.client, supplied_client)

    def test_overreported_advisor_tokens_stop_before_workers_or_assembly(self):
        cap = cooperative.cooperative_budget(30000)['advisor']
        with mock.patch.object(cooperative, 'run_task_graph') as workers:
            result = self.run_proof([[event(plan(), count=cap + 1)]])
        workers.assert_not_called()
        self.assertEqual(result['status'], 'budget_violation')
        self.assertEqual(result['tokens_charged'], cap + 1)
        self.assertEqual(len(self.client.requests), 1)
        self.assertIsNone(result['proof_path'])

    def test_invalid_or_truncated_plan_cannot_dispatch_hidden_fallback_work(self):
        for index, reply in enumerate([
                event(plan(tasks=[contract(dependencies=['T2']),
                                  contract('T2', statement='Another claim.', dependencies=['T1'])])),
                event('not JSON'), event(plan(), reason='length')]):
            with self.subTest(index=index), mock.patch.object(cooperative, 'run_task_graph') as workers:
                result = self.run_proof([[reply]], output_dir=self.root / f'invalid-{index}')
            workers.assert_not_called()
            self.assertEqual(result['status'], 'invalid_plan')
            self.assertEqual(len(self.client.requests), 1)
            self.assertIsNone(result['proof_path'])

    def test_original_sources_are_exact_and_unrelated_workspace_files_are_excluded(self):
        source = 'ORIGINAL_QUANTIFIER: for every real x and y, with no positivity assumption.\n'
        (self.workspace / 'problem.txt').write_text(source)
        (self.workspace / 'solution.txt').write_text('PRIVATE_REFERENCE_NOT_FOR_THE_MODEL')
        result = self.run_proof([[event(plan('direct'))], [event(PROOF)],
                                 [event(critic())], [event(audit())]], source_files=['problem.txt'])
        self.assertIsNotNone(result['proof_path'])
        for payload in self.client.requests:
            self.assertIn('ORIGINAL_QUANTIFIER', json.dumps(payload))
            self.assertIn('with no positivity assumption', json.dumps(payload))
            self.assertNotIn('PRIVATE_REFERENCE_NOT_FOR_THE_MODEL', json.dumps(payload))
        self.assertEqual((self.workspace / 'problem.txt').read_text(), source)

    def test_interrupted_advisor_retains_reservation_without_implicit_replay(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_proof([KeyboardInterrupt()])
        state = json.loads((self.root / 'cooperative' / 'state.json').read_text())
        self.assertEqual(state['status'], 'interrupted')
        self.assertEqual(state['tokens_charged'], cooperative.cooperative_budget(30000)['advisor'])
        self.assertIsNone(state['proof_path'])

    def test_one_targeted_repair_wave_retains_prior_objection_and_exact_contract_scope(self):
        proposed = plan()
        proposed['tasks'][0]['hypotheses'].append('EXTRA_HYPOTHESIS: x is positive; removal remains open.')
        repair_task = contract('R1', statement='Prove the missing boundary case.', dependencies=['T1'])
        repair = {'needed': True, 'reason': 'The whole-proof audit found a missing boundary case.', 'task': repair_task}
        failure = audit(verdict='gap', explanation='The stated universal quantifier is not covered.',
                        objection='MISSING_BOUNDARY_CASE: x=0 is not justified.', next_task='Prove the x=0 case.')
        with mock.patch.object(cooperative, 'run_task_graph', side_effect=self.fake_graph) as graphs:
            result = self.run_proof([[event(proposed)], [event(PROOF)], [event(critic())], [event(failure)],
                                    [event(repair)], [event(PROOF)], [event(critic())], [event(failure)]])
        self.assertEqual(graphs.call_count, 2)
        self.assertEqual(graphs.call_args_list[1].args[1], [repair_task])
        self.assertEqual(set(graphs.call_args_list[1].kwargs['external_results']), {'T1', 'T2'})
        self.assertEqual(len(result['stages']), 2)
        self.assertEqual(len(self.client.requests), 8)
        self.assertEqual(result['tokens_charged'], 170)
        self.assertEqual(result['request_count'], 17)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['proof_path'])
        self.assertEqual(Path(result['answer_path']).read_text(), PROOF)
        final_request = json.dumps(self.client.requests[-1])
        self.assertIn('MISSING_BOUNDARY_CASE', final_request)
        self.assertIn(repair_task['statement'], final_request)
        self.assertNotIn('EXACT_WORKER_EVIDENCE_R1', final_request)
        self.assertIn('EXACT_WORKER_EVIDENCE_R1', json.dumps(self.client.requests[-3]))
        self.assertIn('EXTRA_HYPOTHESIS', final_request)
        for stage in result['stages']:
            path = next((Path(stage['workspace']) / '.mathagent' / 'proofs').glob('*/state.json'))
            ledger = json.loads(path.read_text())
            self.assertEqual(ledger['goal'], GOAL)
            self.assertEqual(ledger['sources'], [])
        self.assertNotIn('EXACT_WORKER_EVIDENCE', json.dumps(self.client.requests[-2]))

    def test_worker_overrun_stops_before_any_assembly_or_repair(self):
        def overrun(agent, tasks, **options):
            result = self.fake_graph(agent, tasks, **options)
            result.update(status='budget_violation', tokens_charged=options['token_budget'] + 1)
            return result
        with mock.patch.object(cooperative, 'run_task_graph', side_effect=overrun) as graphs:
            result = self.run_proof([[event(plan())]])
        self.assertEqual(graphs.call_count, 1)
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(result['status'], 'budget_violation')
        self.assertEqual(result['stages'], [])
        self.assertIsNone(result['proof_path'])

    def test_interrupted_graph_uses_settled_usage_or_retains_unknown_full_reservation(self):
        allocation = cooperative.cooperative_budget(30000)['workers']
        for settled in (False, True):
            output_dir = self.root / f'interrupted-graph-{settled}'

            def interrupt(agent, tasks, **options):
                if settled:
                    state = self.fake_graph(agent, tasks, **options)
                    state.update(status='interrupted', token_budget=options['token_budget'])
                    directory = Path(options['output_dir'])
                    directory.mkdir(parents=True)
                    (directory / 'state.json').write_text(json.dumps(state))
                raise KeyboardInterrupt

            with self.subTest(settled_journal=settled), \
                    mock.patch.object(cooperative, 'run_task_graph', side_effect=interrupt) as graphs:
                with self.assertRaises(KeyboardInterrupt):
                    self.run_proof([[event(plan())]], output_dir=output_dir)
                state = json.loads((output_dir / 'state.json').read_text())
                self.assertEqual(state['status'], 'interrupted')
                self.assertEqual(state['tokens_charged'], 10 + (60 if settled else allocation))
                self.assertEqual(state['measured_completion_tokens'], 70 if settled else 10)
                self.assertEqual(state['reserved_unmeasured_tokens'], 0 if settled else allocation)
                self.assertEqual(state['request_count'], 7 if settled else 1)
                self.assertEqual(state['stages'], [])
                self.assertIsNone(state['proof_path'])
                self.assertEqual(len(self.client.requests), 1)
                self.assertEqual(graphs.call_count, 1)

    def test_oversized_advisor_input_abstains_without_clipping_or_dispatch(self):
        result = self.run_proof([], ctx=1024)
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(self.client.requests, [])
        self.assertEqual(result['tokens_charged'], 0)
        self.assertIsNone(result['proof_path'])

    def test_assembly_cannot_approve_when_real_historical_objection_exceeds_audit_context(self):
        def oversized(agent, tasks, **options):
            result = self.fake_graph(agent, tasks, **options)
            result['results'][0]['obligations'] = [{
                'statement': 'Check the exact missing boundary argument.', 'status': 'gap',
                'objection': 'ACTUAL_HISTORICAL_OBJECTION_MUST_NOT_BE_CLIPPED ' + 'x' * 80000}]
            return result
        with mock.patch.object(cooperative, 'run_task_graph', side_effect=oversized):
            result = self.run_proof([[event(plan())], [event(PROOF)], [event(critic())]], ctx=8192)
        self.assertEqual(result['status'], 'needs_context')
        self.assertEqual(len(self.client.requests), 3)
        self.assertIsNone(result['proof_path'])
        path = next((Path(result['stages'][0]['workspace']) / '.mathagent' / 'proofs').glob('*/state.json'))
        ledger = json.loads(path.read_text())
        self.assertIn('x' * 80000, ledger['claims'][0]['argument'])

    def test_large_successful_helper_body_does_not_replace_or_overflow_global_audit(self):
        body = 'LARGE_HELPER_PROOF_BODY ' + 'x' * 80000

        def oversized_helper(agent, tasks, **options):
            result = self.fake_graph(agent, tasks, **options)
            result['results'][0]['candidate']['text'] = body
            return result

        with mock.patch.object(cooperative, 'run_task_graph', side_effect=oversized_helper):
            result = self.run_proof([[event(plan())], [event(PROOF)], [event(critic())], [event(audit())]],
                                    ctx=8192)
        self.assertEqual(result['status'], 'candidate_complete')
        self.assertEqual(Path(result['proof_path']).read_text(), PROOF)
        final_request = json.dumps(self.client.requests[-1])
        self.assertIn(GOAL, final_request)
        self.assertIn(PROOF, final_request)
        self.assertIn(contract()['statement'], final_request)
        self.assertIn(contract()['constant_dependencies'], final_request)
        self.assertNotIn('LARGE_HELPER_PROOF_BODY', final_request)
        self.assertIn('not supplied as proof here', final_request.lower())
        path = next((Path(result['stages'][0]['workspace']) / '.mathagent' / 'proofs').glob('*/state.json'))
        ledger = json.loads(path.read_text())
        self.assertIn(body, ledger['claims'][0]['argument'])


if __name__ == '__main__':
    unittest.main()
