"""Command-line wiring for experiments and proof-mode experiment flags."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mathagent import cli


class ExperimentCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ui, self.client = Mock(), Mock()
        self.client.models.return_value = ['qwen3.8:27b']
        self.result = {'id': 'job', 'status': 'consistent', 'report': 'Experiment report.',
                       'directory': str(self.root / '.mathagent/research/job')}

    def run_cli(self, arguments, **patches):
        with patch('sys.argv', ['square-harness', '--workspace', str(self.root), *arguments]), \
                patch.object(cli, 'UI', return_value=self.ui), patch.object(cli, 'Ollama', return_value=self.client), \
                contextlib.ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(patch.object(cli, name, value))
            return cli.main()

    def test_experiment_mode_passes_permission_and_limits(self):
        runner = Mock()
        runner.start.return_value = self.result
        code = self.run_cli(['--mode', 'experiment', '--experiments', 'auto', '--experiment-seconds', '30',
                             '--experiment-memory', '1024', '--experiment-runs', '2', '--prompt', 'Is 2 > 1?'],
                            ExperimentRunner=Mock(return_value=runner))
        self.assertEqual(code, 0)
        runner.start.assert_called_once_with('Is 2 > 1?', permission='auto', run_seconds=30, memory_mb=1024,
                                             source_files=[], max_rounds=2, max_tokens=30000, max_seconds=1800)

    def test_proof_flags_enable_experiments(self):
        runner = Mock()
        runner.start.return_value = {'id': 'p', 'status': 'refuted', 'report': 'r', 'answer': '',
                                     'proof_path': str(self.root / 'p.md'), 'directory': str(self.root)}
        self.run_cli(['--proof-refute-first', '--experiments', 'off', '--prompt', 'Prove it.'],
                     ProofRunner=Mock(return_value=runner))
        experiments = runner.start.call_args.kwargs['experiments']
        self.assertEqual(experiments['permission'], 'off')
        self.assertTrue(experiments['refute_first'])
        self.assertFalse(experiments['test_objections'])

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.run_cli(['--experiment-seconds', '1', '--prompt', 'x'])


if __name__ == '__main__':
    unittest.main()
