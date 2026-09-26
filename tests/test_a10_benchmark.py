"""A10 trials retain the scientific allocation while explicitly serializing branches."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mathagent import benchmark

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_a10', ROOT / 'scripts/run-a10-benchmark.py')
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class A10BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = benchmark.init_smoke(self.root / 'statements')
        self.launch = self.root / 'launch.json'
        lock = json.loads((ROOT / 'deployment/llama-a10.lock.json').read_text())
        self.record = {'schema_version': 1, 'weights_sha256': lock['weights_sha256'],
            'template_sha256': lock['template_sha256'], 'model_alias': lock['model_alias'],
            'model_path': '/models/' + lock['model_filename'], 'llama_commit': lock['llama_commit'],
            'image_id': 'sha256:' + 'c' * 64, 'context_per_slot': 32768, 'parallel': 1,
            'host': 'http://127.0.0.1:8000', 'quantization': 'Q4_K_M',
            'cache_type_k': 'f16', 'cache_type_v': 'f16', 'context_shift': False}
        self.launch.write_text(json.dumps(self.record))
        self.acceptance = self.root / 'acceptance.json'
        self.acceptance.write_text(json.dumps({'ok': True, 'accepted_for_benchmark': True,
            'backend': 'llamacpp', 'host': self.record['host'], 'model': self.record['model_alias'],
            'launch_record': {'sha256': hashlib.sha256(self.launch.read_bytes()).hexdigest()},
            'settings': {'context_per_slot': 32768, 'parallel': 1}}))
        self.argv = ['--manifest', str(self.manifest), '--output', str(self.root / 'results'),
                     '--launch-record', str(self.launch)]

    def test_offline_preflight_preserves_three_attempts_and_budget_with_one_active_slot(self):
        with patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(run.main(self.argv + ['--dry-run']), 0)
        plan = json.loads(out.getvalue())
        settings = plan['settings']
        self.assertEqual(settings['tokens'], 60000)
        self.assertEqual(settings['branches'], 3)
        self.assertEqual(settings['branch_concurrency'], 1)
        self.assertEqual(settings['max_in_flight'], 1)
        self.assertEqual(settings['selection_tokens'], 6144)
        self.assertEqual(settings['dtype'], 'q4_k_m')
        self.assertEqual(settings['ctx'], 32768)
        self.assertEqual(settings['seconds'], 43200)
        self.assertEqual(settings['request_timeout'], 7200)
        self.assertEqual(settings['selection_seconds'], 3600)
        self.assertEqual(settings['arms'], ['raw-best', 'sequential', 'parallel'])
        self.assertEqual(plan['generated_token_ceiling'], 360000)
        self.assertEqual(plan['serving']['profile'], 'a10-q4-serial-branches')
        self.assertFalse((self.root / 'results').exists())

    def test_live_run_requires_acceptance_before_connecting_or_creating_outputs(self):
        with patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(self.argv), 1)
        self.assertFalse((self.root / 'results').exists())

    def test_v100_and_other_model_records_cannot_be_used_for_a10(self):
        for key, wrong in (('parallel', 3), ('quantization', 'Q8_0'), ('weights_sha256', 'a' * 64),
                           ('model_alias', 'square-qwen'), ('template_sha256', 'b' * 64)):
            self.launch.write_text(json.dumps({**self.record, key: wrong}))
            with self.subTest(key=key), patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run.main(self.argv + ['--dry-run']), 1)

    def test_old_acceptance_is_rejected_after_server_rebuild(self):
        self.record['image_id'] = 'sha256:' + 'd' * 64
        self.launch.write_text(json.dumps(self.record))
        with patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(self.argv + ['--acceptance', str(self.acceptance), '--dry-run']), 1)

    def test_scoring_embeds_the_a10_condition_and_retains_failed_trials(self):
        with patch.object(run, 'verify_current_server') as verify, patch.object(benchmark, 'execute', return_value=[{'status': 'error'}]) as execute:
            self.assertEqual(run.main(self.argv + ['--acceptance', str(self.acceptance)]), 1)
        verify.assert_called_once_with(self.record)
        parsed, data, plan = execute.call_args.args
        self.assertEqual(parsed.branch_concurrency, 1)
        self.assertEqual(plan['serving']['launch_record'], self.record)
        self.assertTrue(plan['serving']['acceptance']['accepted_for_benchmark'])
        self.assertIn('not a concurrent-inference speed test', plan['serving']['experimental_condition'])


if __name__ == '__main__':
    unittest.main()
