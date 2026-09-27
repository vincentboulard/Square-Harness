"""The V100 launch path binds the scientific run to verified serving settings."""
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

SPEC = importlib.util.spec_from_file_location('run_v100', Path(__file__).resolve().parents[1] / 'scripts/run-v100-benchmark.py')
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class V100BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = benchmark.init_smoke(self.root / 'statements')
        self.launch = self.root / 'launch.json'
        self.template = 'A verified chat template'
        self.record = {'schema_version': 1, 'weights_sha256': 'a' * 64,
            'template_sha256': hashlib.sha256(self.template.encode()).hexdigest(),
            'model_alias': 'square-qwen', 'model_path': '/models/qwen.gguf',
            'llama_commit': 'b' * 40, 'image_id': 'sha256:' + 'c' * 64,
            'context_per_slot': 32768, 'parallel': 3, 'host': 'http://127.0.0.1:8000',
            'quantization': 'Q8_0', 'cache_type_k': 'f16', 'cache_type_v': 'f16', 'context_shift': False}
        self.launch.write_text(json.dumps(self.record))
        self.acceptance = self.root / 'acceptance.json'
        self.acceptance.write_text(json.dumps({'ok': True, 'accepted_for_benchmark': True,
            'backend': 'llamacpp', 'host': self.record['host'], 'model': 'square-qwen',
            'launch_record': {'sha256': hashlib.sha256(self.launch.read_bytes()).hexdigest()},
            'settings': {'context_per_slot': 32768, 'parallel': 3}}))
        self.argv = ['--manifest', str(self.manifest), '--output', str(self.root / 'results'),
                     '--launch-record', str(self.launch)]

    def test_offline_preflight_keeps_protocol_and_never_creates_results(self):
        with patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(run.main(self.argv + ['--dry-run']), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['settings']['tokens'], 90000)
        self.assertEqual(plan['settings']['predict'], 16384)
        self.assertEqual(plan['settings']['verify_tokens'], 8192)
        self.assertEqual(plan['settings']['dtype'], 'q8_0')
        self.assertEqual(plan['settings']['request_timeout'], 7200)
        self.assertEqual(plan['settings']['arms'], ['raw-single', 'proof'])
        self.assertEqual(plan['generated_token_ceiling'], 2 * (16384 + 90000))
        self.assertIsNone(plan['serving']['acceptance'])
        self.assertFalse((self.root / 'results').exists())

    def test_live_run_requires_acceptance_before_any_model_connection(self):
        with patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(self.argv), 1)
        self.assertFalse((self.root / 'results').exists())

    def test_changed_launch_record_invalidates_previous_acceptance(self):
        self.record['image_id'] = 'sha256:' + 'd' * 64
        self.launch.write_text(json.dumps(self.record))
        with patch.object(run, 'create_client', side_effect=AssertionError('No network')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run.main(self.argv + ['--acceptance', str(self.acceptance), '--dry-run']), 1)

    def test_failed_jobs_return_nonzero_and_evidence_is_saved_in_plan(self):
        with patch.object(run, 'verify_current_server') as verify, patch.object(benchmark, 'execute', return_value=[{'status': 'error'}]) as execute:
            self.assertEqual(run.main(self.argv + ['--acceptance', str(self.acceptance)]), 1)
        verify.assert_called_once_with(self.record)
        plan = execute.call_args.args[2]
        self.assertTrue(plan['serving']['acceptance']['accepted_for_benchmark'])
        self.assertEqual(plan['serving']['launch_record'], self.record)

    def test_current_server_reconfiguration_is_rejected_before_scoring(self):
        props = {'model_path': self.record['model_path'], 'total_slots': 3,
                 'default_generation_settings': {'n_ctx': 8192},
                 'chat_template': self.template, 'build_info': 'bbbbbbb'}
        response = io.BytesIO(json.dumps(props).encode())
        with patch.object(run, 'create_client') as factory:
            factory.return_value.opener.open.return_value = response
            with self.assertRaisesRegex(ValueError, 'differs'):
                run.verify_current_server(self.record)


if __name__ == '__main__':
    unittest.main()
