"""Dependency evidence, one-shot dispatch and conservative shared accounting."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mathagent.agent import Agent, Ollama
from mathagent.cooperative_workers import run_task_graph, _task_result
from mathagent.ledger import ProofStore
from mathagent.tools import Workspace


def contract(task_id, dependencies=()):
    return {'id': task_id, 'statement': f'Exact statement {task_id}: for every real x, x=x.',
            'hypotheses': ['x is real'], 'setting': 'Real numbers',
            'quantifiers': 'For every real x', 'constant_dependencies': 'No constants',
            'dependencies': list(dependencies), 'deliverable': 'A complete written proof'}


def finish(job, *, status='candidate_complete', tokens=20, text=None, complete=True):
    store = ProofStore.create(job['workspace'], job['goal'],
        {'max_rounds': 1, 'max_tokens': job['token_budget'], 'max_seconds': 60})
    text = text or 'WRITTEN_PROOF_' + job['id'] + ': Equality is reflexive.'
    draft = store.write_artifact('candidate', json.dumps({
        'text': text, 'thinking': 'HIDDEN_THINKING_NOT_EXPORTED', 'complete': complete, 'calls': []}))
    store.state.update(status=status, tokens_charged=tokens,
        calls=[{'reserved_tokens': 20, 'charged_tokens': tokens,
                'stats': {'eval_count': tokens, 'prompt_eval_count': 7}}],
        rounds=[{'index': 1, 'draft': draft, 'critique': {'complete_candidate': status == 'candidate_complete'}}],
        final_audit={'candidate': draft, 'verdict': 'complete' if status == 'candidate_complete' else 'gap',
                     'explanation': 'Reflexivity applies.', 'objection': '' if status == 'candidate_complete' else 'Missing justification.',
                     'next_task': ''})
    if status != 'candidate_complete':
        store.state['claims'] = [{
            'id': 'C1', 'round': 1, 'statement': job['contract']['statement'], 'status': 'gap',
            'assumptions': [], 'dependencies': [], 'argument': text, 'objection': 'Missing justification.',
            'evidence': '', 'resolves': [], 'resolution': '', 'candidate_artifact': draft,
            'review_artifact': None, 'next_task': 'Prove the missing step.'}]
    store.save()
    (Path(job['directory']) / 'worker-result.json').write_text(json.dumps({'status': 'finished'}))
    return store


class Process:
    def __init__(self, context, job):
        self.context, self.job, self.pid = context, job, None

    def start(self):
        self.pid = 123
        self.context.started.append(self.job['id'])
        self.context.on_start(self.job)

    def is_alive(self):
        return False

    def join(self, *args):
        pass


class Context:
    def __init__(self, on_start=finish):
        self.started, self.on_start = [], on_start

    def Process(self, *, target, args, name):
        return Process(self, args[0])


class CooperativeWorkerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.agent = Agent(Ollama(), Workspace(self.root), model='test', ctx=32768, predict=256)
        self.context = Context()
        patch = mock.patch('mathagent.cooperative_workers.multiprocessing.get_context', return_value=self.context)
        patch.start()
        self.addCleanup(patch.stop)

    def run_graph(self, tasks, **options):
        return run_task_graph(self.agent, tasks, goal='GLOBAL_CONCLUSION_NOT_AN_ASSUMPTION',
            sources=options.pop('sources', []), output_dir=options.pop('output_dir', self.root / 'graph'),
            token_budget=options.pop('token_budget', 6000), max_seconds=options.pop('max_seconds', 60),
            max_rounds=1, max_predict=512, **options)

    def test_ready_tasks_are_dispatched_before_join_and_contracts_are_distinct(self):
        seen = []

        def on_start(job):
            state = json.loads((self.root / 'graph' / 'state.json').read_text())
            seen.append(copy.deepcopy(state))
            self.assertEqual(sum(item['token_budget'] for item in state['jobs']), 6001)
            self.assertTrue(next(item for item in state['jobs'] if item['id'] == job['id'])['dispatched'])
            self.assertGreaterEqual(state['tokens_charged'], job['token_budget'])
            self.assertIn(job['contract']['statement'], job['goal'])
            self.assertNotIn('GLOBAL_CONCLUSION_NOT_AN_ASSUMPTION', job['goal'])
            finish(job)

        self.context.on_start = on_start
        result = self.run_graph([contract('A'), contract('B'), contract('C', ['A', 'B'])], token_budget=6001)
        self.assertEqual(self.context.started, ['A', 'B', 'C'])
        self.assertEqual([job['token_budget'] for job in result['jobs']], [2001, 2000, 2000])
        self.assertEqual(len(seen[1]['results']), 0)
        self.assertEqual([item['id'] for item in seen[2]['results']], ['A', 'B'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['request_count'], 3)
        self.assertEqual(result['tokens_charged'], 60)
        self.assertEqual(result['prompt_tokens'], 21)
        self.assertTrue(all(item['dependency_ready'] for item in result['results']))
        self.assertEqual(result['jobs'][2]['prerequisites'], ['A', 'B'])

    def test_transitive_exact_proofs_and_sources_are_pinned_under_safe_filenames(self):
        source = {'path': '../../do-not-write-here.md', 'content': 'PINNED_TEXT_ONLY',
                  'sha256': hashlib.sha256(b'PINNED_TEXT_ONLY').hexdigest()}
        (self.root / 'reference.txt').write_text('PRIVATE_ANSWER_MUST_NOT_BE_COPIED')
        result = self.run_graph([contract('../../A'), contract('B', ['../../A']), contract('C', ['B'])], sources=[source])
        last = result['jobs'][2]
        self.assertEqual(last['prerequisites'], ['../../A', 'B'])
        workspace = Path(last['workspace'])
        evidence = '\n'.join((workspace / name).read_text() for name in last['source_files'])
        self.assertIn('WRITTEN_PROOF_../../A', evidence)
        self.assertIn('WRITTEN_PROOF_B', evidence)
        self.assertIn('Fallible model-reviewed proof evidence', evidence)
        self.assertIn('CONTEXT ONLY, NOT AN ASSUMPTION', evidence)
        self.assertIn('PINNED_TEXT_ONLY', evidence)
        self.assertNotIn('PRIVATE_ANSWER_MUST_NOT_BE_COPIED', evidence)
        self.assertFalse((workspace / 'reference.txt').exists())
        self.assertFalse((self.root / 'do-not-write-here.md').exists())
        self.assertTrue(all(Path(name).name == name for name in last['source_files']))

    def test_partial_prerequisite_blocks_descendants_but_preserves_written_evidence(self):
        self.context.on_start = lambda job: finish(job, status='budget_exhausted')
        result = self.run_graph([contract('C', ['B']), contract('B', ['A']), contract('A')])
        self.assertEqual(self.context.started, ['A'])
        self.assertEqual(result['status'], 'partial')
        results = {item['id']: item for item in result['results']}
        self.assertEqual(results['B']['status'], 'blocked')
        self.assertEqual(results['C']['status'], 'blocked')
        self.assertEqual(results['B']['tokens_charged'], 0)
        self.assertFalse(results['A']['dependency_ready'])
        self.assertIn('WRITTEN_PROOF_A', results['A']['candidate']['text'])
        self.assertEqual(results['A']['partial_artifacts'], [])
        self.assertNotIn('HIDDEN_THINKING_NOT_EXPORTED', json.dumps(results['A']['partial_artifacts']))
        self.assertEqual(results['A']['obligations'][0]['objection'], 'Missing justification.')
        self.assertEqual(result['tokens_charged'], 20)
        self.assertEqual(result['execution_errors'], [])

    def test_external_repair_prerequisites_keep_transitive_evidence_without_double_charging(self):
        initial = self.run_graph([contract('A'), contract('B', ['A'])])
        external = {item['id']: item for item in initial['results']}
        repair = self.run_graph([contract('R1', ['B'])], external_results=external,
                                output_dir=self.root / 'repair', token_budget=1000)
        self.assertEqual(repair['status'], 'completed')
        self.assertEqual(repair['jobs'][0]['prerequisites'], ['A', 'B'])
        self.assertEqual(repair['tokens_charged'], 20)
        self.assertEqual(repair['request_count'], 1)
        self.assertEqual(len(repair['external_prerequisites']), 2)
        external['B']['candidate']['artifact'] = '/fake/not-the-audited-proof.md'
        rejected = self.run_graph([contract('R2', ['B'])], external_results=external,
                                  output_dir=self.root / 'rejected')
        self.assertEqual(rejected['status'], 'partial')
        self.assertFalse(rejected['jobs'][0]['dispatched'])
        self.assertEqual(rejected['tokens_charged'], 0)

    def test_truncated_written_argument_is_retained_but_never_admitted_as_prerequisite(self):
        self.context.on_start = lambda job: finish(job, status='budget_exhausted', complete=False)
        result = self.run_graph([contract('A'), contract('B', ['A'])])
        first = result['results'][0]
        self.assertIsNone(first['candidate'])
        self.assertFalse(first['dependency_ready'])
        self.assertFalse(first['partial_artifacts'][0]['complete'])
        self.assertIn('WRITTEN_PROOF_A', first['partial_artifacts'][0]['text'])
        self.assertNotIn('HIDDEN_THINKING_NOT_EXPORTED', json.dumps(first['partial_artifacts']))
        self.assertEqual(result['results'][1]['status'], 'blocked')

    def test_cycle_unknown_duplicate_and_bad_source_fail_before_creating_directory(self):
        invalid = [[contract('A', ['missing'])], [contract('A', ['A'])],
                   [contract('A', ['B']), contract('B', ['A'])], [contract('A'), contract('A')]]
        for tasks in invalid:
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                self.run_graph(tasks)
            self.assertFalse((self.root / 'graph').exists())
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.run_graph([contract('A')], sources=[{'path': 'x', 'content': 'data', 'sha256': 'wrong'}])
        self.assertFalse((self.root / 'graph').exists())
        self.assertEqual(self.context.started, [])

    def test_output_budget_violation_stops_queued_work_and_preserves_observed_usage(self):
        self.context.on_start = lambda job: finish(job, status='budget_violation', tokens=257)
        result = self.run_graph([contract('A'), contract('B')], concurrency=1)
        self.assertEqual(self.context.started, ['A'])
        self.assertEqual(result['status'], 'budget_violation')
        self.assertEqual(result['tokens_charged'], 257)
        self.assertEqual(result['results'][1]['status'], 'not_dispatched')
        self.assertTrue(result['execution_errors'])

    def test_missing_child_ledger_charges_full_allocation_and_blocks_dependents(self):
        self.context.on_start = lambda job: None
        result = self.run_graph([contract('A'), contract('B', ['A'])], token_budget=2000)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['tokens_charged'], 1000)
        self.assertEqual(result['reserved_unmeasured_tokens'], 1000)
        self.assertEqual(self.context.started, ['A'])

    def test_returned_worker_execution_failure_is_not_hidden_by_finished_process(self):
        for status in ('paused', 'interrupted', 'error', 'needs_recovery'):
            with self.subTest(status=status):
                def stopped(job):
                    store = finish(job, status=status)
                    store.state['stop_reason'] = 'Model connection closed during the critic request.'
                    store.save()

                self.context.started.clear()
                self.context.on_start = stopped
                result = self.run_graph([contract('A'), contract('B', ['A'])],
                                        output_dir=self.root / status)
                first = result['results'][0]
                self.assertEqual(self.context.started, ['A'])
                self.assertEqual(result['status'], 'error')
                self.assertEqual(first['status'], 'failed')
                self.assertEqual(first['proof_status'], status)
                self.assertEqual(first['error'], 'Model connection closed during the critic request.')
                self.assertEqual(first['stop_reason'], first['error'])
                self.assertEqual(result['execution_errors'], ['A: ' + first['error']])
                self.assertFalse(first['dependency_ready'])
                self.assertIn('WRITTEN_PROOF_A', first['candidate']['text'])
                self.assertEqual(result['results'][1]['status'], 'blocked')
                self.assertEqual(result['tokens_charged'], 20)

    def test_context_limit_is_retained_without_becoming_a_transport_failure(self):
        def context_limited(job):
            store = finish(job, status='needs_context')
            store.state['stop_reason'] = 'The exact prerequisite proof does not fit the context window.'
            store.save()

        self.context.on_start = context_limited
        result = self.run_graph([contract('A'), contract('B', ['A'])])
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['execution_errors'], [])
        self.assertEqual(result['results'][0]['proof_status'], 'needs_context')
        self.assertIn('does not fit', result['results'][0]['stop_reason'])
        self.assertFalse(result['results'][0]['dependency_ready'])
        self.assertEqual(result['results'][1]['status'], 'blocked')

    def test_interrupt_preserves_full_dispatched_allocation_and_one_shot_directory(self):
        self.context.on_start = lambda job: None

        def interrupt(kind, value):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_graph([contract('A'), contract('B')], emit=interrupt)
        saved = json.loads((self.root / 'graph' / 'state.json').read_text())
        self.assertEqual(saved['status'], 'interrupted')
        self.assertEqual(saved['tokens_charged'], 3000)
        self.assertEqual(saved['reserved_unmeasured_tokens'], 3000)
        self.assertEqual(self.context.started, ['A'])
        with self.assertRaises(FileExistsError):
            self.run_graph([contract('A')])

    def test_deadline_before_dispatch_has_no_phantom_token_charge(self):
        with mock.patch('mathagent.cooperative_workers.time.monotonic', side_effect=[0, 0, 2, 2, 2]):
            result = self.run_graph([contract('A')], max_seconds=1)
        self.assertEqual(result['status'], 'time_exhausted')
        self.assertEqual(result['tokens_charged'], 0)
        self.assertEqual(self.context.started, [])


if __name__ == '__main__':
    unittest.main()
