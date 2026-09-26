"""CLI boundaries: durable jobs are available offline and keep their budgets."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mathagent import cli


class ProofCliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ui = Mock()
        self.client = Mock()
        self.client.models.return_value = ['qwen3.8:27b']
        self.runner = Mock()
        self.result = {'id': 'proof-example', 'status': 'budget_exhausted',
                       'report': 'Unresolved: the shifted-weight estimate still has a gap.',
                       'directory': str(self.root / '.mathagent' / 'proofs' / 'proof-example')}
        self.runner.start.return_value = self.result
        self.runner.resume.return_value = self.result

    def run_cli(self, arguments):
        with patch('sys.argv', ['mathagent', '--workspace', str(self.root), *arguments]), \
                patch.object(cli, 'UI', return_value=self.ui), \
                patch.object(cli, 'Ollama', return_value=self.client), \
                patch.object(cli, 'ProofRunner', return_value=self.runner):
            return cli.main()

    def output(self):
        return '\n'.join(str(call.args[0]) for call in self.ui.say.call_args_list)

    def test_llamacpp_client_uses_explicit_backend_and_long_request_timeout(self):
        self.client.models.return_value = ['square-qwen']
        with patch.object(cli, 'create_client', return_value=self.client) as factory:
            code = self.run_cli(['--backend', 'llamacpp', '--request-timeout', '7200',
                                 '--prompt', 'Prove the supplied statement.'])
        self.assertEqual(code, 0)
        factory.assert_called_once_with('llamacpp', 'http://localhost:8000', timeout=7200.0)
        self.runner.start.assert_called_once()

    def test_default_prove_dispatches_bounded_job_and_prints_debrief(self):
        code = self.run_cli(['--proof-rounds', '3', '--proof-tokens', '8000',
                             '--proof-seconds', '90', '--proof-file', 'statement.tex',
                             '--prompt', 'Read statement.tex and prove its statement.'])
        self.assertEqual(code, 0)
        self.runner.start.assert_called_once_with('Read statement.tex and prove its statement.',
            max_rounds=3, max_tokens=8000, max_seconds=90.0, max_predict=8192,
            source_files=['statement.tex'])
        self.client.models.assert_called_once()
        self.assertIn(self.result['report'], self.output())
        self.assertIn('/ledger proof-example', self.output())
        self.assertNotIn('/resume proof-example', self.output())
        self.assertIn('report.md', self.output())

    def test_adaptive_output_ceiling_is_forwarded_for_a_new_proof(self):
        self.assertEqual(self.run_cli(['--ctx', '32768', '--predict', '6144',
                                      '--proof-max-predict', '12288', '--prompt', 'Prove it.']), 0)
        self.assertEqual(self.runner.start.call_args.kwargs['max_predict'], 12288)

    def test_too_small_adaptive_ceiling_stops_before_contacting_model(self):
        self.assertEqual(self.run_cli(['--proof-max-predict', '2048', '--prompt', 'Prove it.']), 1)
        self.client.models.assert_not_called()
        self.runner.start.assert_not_called()
        self.assertIn('--proof-max-predict must be at least --predict', self.output())

    def test_adaptive_ceiling_does_not_limit_ordinary_chat(self):
        with patch.object(cli.Agent, 'run', return_value='Checked.') as run:
            self.assertEqual(self.run_cli(['--ctx', '32768', '--predict', '12288',
                                          '--mode', 'critic', '--prompt', 'Audit it.']), 0)
        run.assert_called_once()
        self.runner.start.assert_not_called()

    def test_only_resumable_results_advertise_resume(self):
        self.runner.start.return_value = {**self.result, 'status': 'paused'}
        self.assertEqual(self.run_cli(['--prompt', 'Prove it.']), 0)
        self.assertIn('/resume proof-example', self.output())

    def test_explicit_prove_uses_persistent_runner_even_from_other_mode(self):
        self.run_cli(['--mode', 'critic', '--prompt', '/prove Prove the supplied statement.'])
        self.assertEqual(self.runner.start.call_args.args[0], 'Prove the supplied statement.')

    def test_ordinary_critic_remains_an_agent_query(self):
        with patch.object(cli.Agent, 'run', return_value='This comparison fails.') as run:
            self.assertEqual(self.run_cli(['--mode', 'critic', '--prompt', 'Audit this inequality.']), 0)
        run.assert_called_once_with('Audit this inequality.', self.ui.emit)
        self.runner.start.assert_not_called()

    def test_proofs_and_ledger_are_available_without_ollama(self):
        directory = self.root / 'saved-proof'
        directory.mkdir()
        (directory / 'report.md').write_text('Counterexample retained; proof unfinished.', encoding='utf-8')
        store = SimpleNamespace(directory=directory,
                                state={'id': 'saved-proof', 'status': 'paused', 'goal': 'Prove the statement'})
        self.client.models.side_effect = AssertionError('Offline inspection must not contact Ollama')
        with patch.object(cli.ProofStore, 'list', return_value=[{
                **store.state, 'rounds_started': 2, 'updated_at': '2026-09-20T12:00:00'}]), \
                patch.object(cli.ProofStore, 'load', return_value=store):
            self.assertEqual(self.run_cli(['--prompt', '/proofs']), 0)
            self.assertEqual(self.run_cli(['--prompt', '/ledger saved-proof']), 0)
        self.client.models.assert_not_called()
        self.client.stream.assert_not_called()
        self.assertIn('rounds 2', self.output())
        self.assertIn('Counterexample retained', self.output())

    def test_resume_does_not_replace_saved_budgets_or_require_default_model(self):
        store = SimpleNamespace(state={'id': 'saved-proof', 'status': 'paused', 'goal': 'Prove the statement'})
        self.client.models.side_effect = AssertionError('Resume uses its saved model')
        with patch.object(cli.ProofStore, 'load', return_value=store):
            self.assertEqual(self.run_cli(['--resume', 'saved-proof', '--proof-rounds', '99',
                                          '--proof-tokens', '999999', '--proof-max-predict', '128']), 0)
        self.runner.resume.assert_called_once_with('saved-proof')
        self.runner.start.assert_not_called()
        self.client.models.assert_not_called()

    def test_implicit_resume_prefers_active_job_over_newer_finished_job(self):
        jobs = [
            {'id': 'finished', 'status': 'budget_exhausted', 'updated_at': '2026-09-20T12:00:00'},
            {'id': 'paused', 'status': 'paused', 'updated_at': '2026-09-20T11:00:00'}]
        with patch.object(cli.ProofStore, 'list', return_value=jobs), \
                patch.object(cli.ProofStore, 'load') as load:
            cli.select_proof(self.root)
        load.assert_called_once_with(self.root, 'paused')

    def test_interrupted_proof_reports_saved_job_for_resume(self):
        self.runner.start.side_effect = KeyboardInterrupt
        store = SimpleNamespace(directory=self.root / '.mathagent' / 'proofs' / 'paused',
                                state={'id': 'paused', 'status': 'paused'})
        with patch.object(cli, 'select_proof', return_value=store):
            self.assertEqual(self.run_cli(['--prompt', 'Prove it.']), 130)
        self.assertIn('/resume paused', self.output())
        self.assertIn('original budget', self.output())

    def test_invalid_proof_budgets_fail_before_any_inference(self):
        for flag in ('--proof-rounds=101', '--proof-rounds=0', '--proof-tokens=0', '--proof-tokens=511',
                     '--proof-seconds=nan', '--proof-seconds=inf', '--proof-seconds=-1',
                     '--proof-max-predict=0', '--proof-max-predict=127'):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli([flag, '--prompt', 'Prove it.'])
                self.assertEqual(error.exception.code, 2)
        self.client.models.assert_not_called()
        self.runner.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
