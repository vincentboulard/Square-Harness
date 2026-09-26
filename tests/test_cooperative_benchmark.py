"""Frozen cooperative benchmark settings, resource gates and blinded export."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mathagent import benchmark


class CooperativeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifest = benchmark.init_smoke(self.root / 'dataset')
        self.args = benchmark.parser().parse_args(['--manifest', str(self.manifest),
            '--output', str(self.root / 'runs'), '--arms', 'cooperative'])

    def test_defaults_preserve_the_original_three_arm_experiment(self):
        self.assertEqual(benchmark.parser().parse_args([]).arms, ['raw-best', 'sequential', 'parallel'])

    def test_dry_run_freezes_allocation_without_network_or_writes(self):
        with patch.object(benchmark, 'create_client', side_effect=AssertionError('No network')), \
                patch('mathagent.cooperative.run_cooperative_proof', side_effect=AssertionError('No inference')), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = benchmark.main(['--manifest', str(self.manifest), '--output', str(self.args.output),
                '--arms', 'cooperative', '--dry-run'])
        self.assertEqual(code, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['jobs'], 2)
        self.assertEqual(plan['generated_token_ceiling'], 120000)
        self.assertEqual(plan['settings']['arms'], ['cooperative'])
        self.assertEqual(plan['cooperative_concurrency'], 3)
        self.assertEqual(plan['cooperative_strategy_version'], 1)
        self.assertEqual(sum(plan['cooperative_budget'].values()), 60000)
        self.assertFalse(self.args.output.exists())

    def test_global_gate_uses_largest_selected_algorithm_width(self):
        self.args.arms = ['parallel', 'cooperative']
        self.args.branch_concurrency = 1
        self.args.workers = 2
        self.args.max_in_flight = 5
        with self.assertRaisesRegex(ValueError, 'branch width'):
            benchmark.preflight(self.args)
        self.args.max_in_flight = 6
        benchmark.preflight(self.args)
        self.args.cooperative_concurrency = 1
        self.args.max_in_flight = 2
        benchmark.preflight(self.args)
        self.args.branch_concurrency = 3
        with self.assertRaisesRegex(ValueError, 'branch width'):
            benchmark.preflight(self.args)

    def test_cooperative_floor_and_concurrency_are_validated_before_dispatch(self):
        self.args.tokens = 8191
        with self.assertRaisesRegex(ValueError, '8192'):
            benchmark.preflight(self.args)
        self.args.tokens = 8192
        self.args.selection_tokens = 999999  # Not a selection-based arm.
        benchmark.preflight(self.args)
        for concurrency in (0, 4):
            self.args.cooperative_concurrency = concurrency
            with self.assertRaisesRegex(ValueError, 'Cooperative concurrency'):
                benchmark.preflight(self.args)

    def run_mock_job(self, changes=None):
        data, _ = benchmark.preflight(self.args)
        self.args.output.mkdir()
        candidate = self.root / 'candidate.md'
        candidate.write_text('A partial mathematical argument; the final estimate remains unproved.', encoding='utf-8')
        result = {'status': 'incomplete', 'answer_path': str(candidate), 'proof_path': None,
                  'directory': str(self.root / 'cooperative'), 'report_path': str(self.root / 'report.md'),
                  'tokens_charged': 300, 'measured_completion_tokens': 200, 'prompt_tokens': 50,
                  'reserved_unmeasured_tokens': 100, 'request_count': 4, 'execution_errors': [],
                  'workers': [], 'stages': [], 'plan': {'strategy': 'decompose'}}
        result.update(changes or {})
        client = SimpleNamespace(host='http://127.0.0.1:8000', backend='openai', timeout=600)
        gate = object()
        with patch.object(benchmark, 'create_client', return_value=client), \
                patch('mathagent.cooperative.run_cooperative_proof', return_value=result) as run:
            record = benchmark.run_job((data['problems'][0], 'cooperative', 0, 29), self.args.output, self.args, gate)
        return data, record, run, gate

    def test_dispatch_forwards_limits_and_exports_exact_unverified_answer_blindly(self):
        data, record, run, gate = self.run_mock_job()
        self.assertEqual(run.call_args.kwargs['max_tokens'], 60000)
        self.assertEqual(run.call_args.kwargs['concurrency'], 3)
        self.assertEqual(run.call_args.kwargs['seed'], 29)
        self.assertEqual(run.call_args.kwargs['source_files'], ('statement.txt',))
        self.assertIs(run.call_args.kwargs['request_gate'], gate)
        self.assertIn(benchmark.COMMON_INSTRUCTION, run.call_args.args[1])
        self.assertEqual(record['status'], 'incomplete')
        self.assertEqual(record['tokens_charged'], 300)
        self.assertEqual(record['request_count'], 4)
        self.assertIsNone(record['proof_path'])
        text = Path(record['answer_path']).read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), record['answer_sha256'])
        benchmark.export_grading(self.args.output, [record], data['problems'], 3)
        self.assertEqual((self.args.output / 'grading/submission-0001/answer.md').read_text(), text)
        self.assertNotIn('cooperative', (self.args.output / 'grading/scores.csv').read_text())
        self.assertEqual(json.loads((self.args.output / 'grading-key.private.json').read_text())['submission-0001']['arm'], 'cooperative')

    def test_worker_execution_error_overrides_model_completion_and_keeps_usage(self):
        _, record, _, _ = self.run_mock_job({'status': 'candidate_complete',
            'execution_errors': ['Worker A exceeded its output allowance.']})
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['workflow_status'], 'candidate_complete')
        self.assertEqual(record['tokens_charged'], 300)
        self.assertEqual(record['execution_errors'], ['Worker A exceeded its output allowance.'])

    def test_invalid_plan_is_an_execution_error_without_an_approved_answer(self):
        _, record, _, _ = self.run_mock_job({'status': 'invalid_plan', 'answer_path': None})
        self.assertEqual(record['status'], 'error')
        self.assertEqual(record['workflow_status'], 'invalid_plan')
        self.assertEqual(Path(record['answer_path']).read_text(), '')
        self.assertIsNone(record['proof_path'])

    def test_no_assembled_candidate_exports_empty_submission_without_inventing_proof(self):
        _, record, _, _ = self.run_mock_job({'answer_path': None})
        self.assertEqual(record['status'], 'incomplete')
        self.assertEqual(Path(record['answer_path']).read_text(), '')


if __name__ == '__main__':
    unittest.main()
