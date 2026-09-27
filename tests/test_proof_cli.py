"""Public v0.5 commands, offline inspection, and explicit proof budgets."""
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
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ui, self.client, self.runner = Mock(), Mock(), Mock()
        self.client.models.return_value = ['qwen3.8:27b']
        self.result = {'id': 'proof-example', 'status': 'budget_exhausted',
                       'report': 'The selected original has an unresolved objection.',
                       'answer': 'Original retained answer.',
                       'proof_path': str(self.root / 'proof.md'),
                       'directory': str(self.root / '.mathagent/proofs/proof-example')}
        self.runner.start.return_value = self.runner.resume.return_value = self.result

    def run_cli(self, arguments):
        with patch('sys.argv', ['square-harness', '--workspace', str(self.root), *arguments]), \
                patch.object(cli, 'UI', return_value=self.ui), \
                patch.object(cli, 'Ollama', return_value=self.client), \
                patch.object(cli, 'ProofRunner', return_value=self.runner):
            return cli.main()

    def output(self):
        return '\n'.join(str(call.args[0]) for call in self.ui.say.call_args_list)

    def test_defaults_offer_direct_sized_solve_and_separate_verification(self):
        self.assertEqual(self.run_cli(['--prompt', 'Prove x=x.']), 0)
        self.runner.start.assert_called_once_with('Prove x=x.', max_rounds=3,
            max_tokens=120000, max_seconds=1800, max_predict=32768,
            verify_tokens=16384, min_solve_tokens=None, repair_tokens=None, verify_temperature=None, source_files=[])
        self.assertIn(self.result['answer'], self.output())
        self.assertIn(self.result['report'], self.output())
        self.assertIn('/proof-report proof-example', self.output())
        self.assertNotIn('/resume proof-example', self.output())

    def test_explicit_limits_and_pinned_file_are_forwarded(self):
        self.assertEqual(self.run_cli(['--proof-rounds', '2', '--proof-tokens', '8000',
            '--proof-solve-tokens', '2048', '--proof-verify-tokens', '1024',
            '--proof-file', 'statement.tex', '--proof-seconds', '90',
            '--prompt', 'Prove the statement in statement.tex.']), 0)
        options = self.runner.start.call_args.kwargs
        self.assertEqual((options['max_predict'], options['verify_tokens']), (2048, 1024))
        self.assertEqual(options['source_files'], ['statement.tex'])
        self.assertEqual(options['max_seconds'], 90)

    def test_invalid_or_starved_new_job_stops_before_model_contact(self):
        for flags in (['--proof-tokens', '8000'], ['--ctx', '8192'],
                      ['--proof-solve-tokens', '40960']):
            with self.subTest(flags=flags):
                self.assertEqual(self.run_cli([*flags, '--prompt', 'Prove it.']), 1)
        self.client.models.assert_not_called()
        self.runner.start.assert_not_called()

    def test_removed_options_cannot_silently_activate_an_old_algorithm(self):
        for flag in ('--proof-strategy=cooperative', '--proof-workers=3', '--proof-literature',
                     '--proof-max-predict=8192', '--proof-rounds=0', '--proof-verify-tokens=0'):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli([flag, '--prompt', 'Prove it.'])
        self.client.models.assert_not_called()

    def test_ordinary_chat_keeps_its_own_limits_and_mode(self):
        with patch.object(cli.Agent, 'run', return_value='Checked.') as run:
            self.assertEqual(self.run_cli(['--ctx', '8192', '--predict', '1024',
                '--mode', 'critic', '--no-think', '--prompt', 'Check this step.']), 0)
        run.assert_called_once_with('Check this step.', self.ui.emit)
        self.runner.start.assert_not_called()

    def test_proofs_and_reports_work_without_model_server(self):
        directory = self.root / 'saved-proof'
        directory.mkdir()
        (directory / 'report.md').write_text('Old work remains readable.', encoding='utf-8')
        store = SimpleNamespace(directory=directory,
            state={'id': 'saved-proof', 'status': 'paused', 'goal': 'Prove the statement'})
        self.client.models.side_effect = AssertionError('Must stay offline')
        with patch.object(cli.ProofStore, 'list', return_value=[{
                **store.state, 'rounds_started': 2, 'updated_at': '2026-09-27T12:00:00'}]), \
                patch.object(cli.ProofStore, 'load', return_value=store):
            for command in ('/proofs', '/proof-report saved-proof', '/ledger saved-proof'):
                self.assertEqual(self.run_cli(['--prompt', command]), 0)
        self.assertIn('Old work remains readable.', self.output())
        self.client.models.assert_not_called()
        self.client.stream.assert_not_called()

    def test_resume_does_not_inject_new_default_budgets_or_model(self):
        store = SimpleNamespace(state={'id': 'saved-proof', 'status': 'paused', 'goal': 'Prove it.'})
        self.client.models.side_effect = AssertionError('Saved model governs resume')
        with patch.object(cli.ProofStore, 'load', return_value=store):
            self.assertEqual(self.run_cli(['--resume', 'saved-proof', '--proof-tokens', '999999']), 0)
        self.runner.resume.assert_called_once_with('saved-proof')
        self.runner.start.assert_not_called()

    def test_interruption_retains_inspection_and_resume_path(self):
        self.runner.start.side_effect = KeyboardInterrupt
        store = SimpleNamespace(directory=self.root / 'saved', state={'id': 'paused', 'status': 'paused'})
        with patch.object(cli, 'select_proof', return_value=store):
            self.assertEqual(self.run_cli(['--prompt', 'Prove it.']), 130)
        self.assertIn('/resume paused', self.output())
        self.assertIn('original budget', self.output())

    def test_version_is_available_without_workspace_or_server(self):
        with patch('sys.argv', ['square-harness', '--version']), \
                contextlib.redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as caught:
            cli.main()
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('0.5.1', output.getvalue())


if __name__ == '__main__':
    unittest.main()
