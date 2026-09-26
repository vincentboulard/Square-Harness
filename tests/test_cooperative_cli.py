"""Opt-in cooperative routing without contacting a model server."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mathagent import cli


class CooperativeCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ui, self.client, self.runner = Mock(), Mock(), Mock()
        self.client.models.return_value = ['qwen3.8:27b']
        self.report = self.root / 'report.md'
        self.report.write_text('The original theorem remains unresolved.', encoding='utf-8')
        self.proof = self.root / 'proof.md'
        self.proof.write_text('Exact complete proof text.', encoding='utf-8')
        self.result = {'status': 'incomplete', 'answer_path': None, 'proof_path': None,
                       'directory': str(self.root), 'report_path': str(self.report),
                       'execution_errors': []}

    def run_cli(self, args):
        with patch('sys.argv', ['square-harness', '--workspace', str(self.root), *args]), \
                patch.object(cli, 'UI', return_value=self.ui), \
                patch.object(cli, 'Ollama', return_value=self.client), \
                patch.object(cli, 'ProofRunner', return_value=self.runner):
            return cli.main()

    def output(self):
        return '\n'.join(str(call.args[0]) for call in self.ui.say.call_args_list)

    def test_strategy_is_opt_in_and_does_not_change_independent_defaults(self):
        args = cli.parser().parse_args([])
        self.assertEqual(args.proof_strategy, 'independent')
        self.assertEqual(args.proof_workers, 1)
        self.assertEqual(args.cooperative_concurrency, 3)

    def test_cooperative_dispatch_forwards_global_limits_and_prints_saved_report(self):
        with patch('mathagent.cooperative.run_cooperative_proof', return_value=self.result) as run:
            code = self.run_cli(['--proof-strategy', 'cooperative', '--cooperative-concurrency', '2',
                '--proof-rounds', '4', '--proof-tokens', '20000', '--proof-seconds', '120',
                '--proof-file', 'theorem.tex', '--seed', '19', '--prompt', 'Prove the original theorem.'])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[1], 'Prove the original theorem.')
        options = run.call_args.kwargs
        self.assertEqual(options['max_tokens'], 20000)
        self.assertEqual(options['max_seconds'], 120)
        self.assertEqual(options['max_rounds'], 4)
        self.assertEqual(options['max_predict'], 8192)
        self.assertEqual(options['source_files'], ['theorem.tex'])
        self.assertEqual(options['concurrency'], 2)
        self.assertEqual(options['seed'], 19)
        self.assertEqual(options['output_dir'].parent, self.root.resolve() / '.mathagent' / 'cooperative')
        self.runner.start.assert_not_called()
        self.assertIn(self.report.read_text(), self.output())
        self.assertIn('one-shot jobs and cannot be resumed', self.output())
        self.assertNotIn('/resume ', self.output())

    def test_completed_proof_prints_exact_text_with_model_review_caveat(self):
        result = {**self.result, 'status': 'candidate_complete', 'proof_path': str(self.proof),
                  'answer_path': str(self.proof)}
        with patch('mathagent.cooperative.run_cooperative_proof', return_value=result):
            self.assertEqual(self.run_cli(['--proof-strategy', 'cooperative', '--prompt', '/prove Goal']), 0)
        self.assertIn(self.proof.read_text(), self.output())
        self.assertIn('independent mathematical checking', self.output())
        self.assertNotIn(self.report.read_text(), self.output())

    def test_incomplete_candidate_is_not_presented_as_audited_proof(self):
        result = {**self.result, 'answer_path': str(self.proof),
                  'execution_errors': ['Worker transport failed.']}
        with patch('mathagent.cooperative.run_cooperative_proof', return_value=result):
            self.assertEqual(self.run_cli(['--proof-strategy', 'cooperative', '--prompt', 'Goal']), 0)
        self.assertIn('Unverified assembled candidate:', self.output())
        self.assertIn('Worker transport failed.', self.output())
        self.assertNotIn(self.proof.read_text(), self.output())

    def test_conflicting_and_underfunded_modes_fail_before_model_access(self):
        for flags in (['--proof-workers', '3'], ['--proof-branch-concurrency', '1'],
                      ['--proof-literature'], ['--resume', 'saved'], ['--proof-tokens', '8191'],
                      ['--cooperative-concurrency', '0'], ['--cooperative-concurrency', '4']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(['--proof-strategy', 'cooperative', *flags])
                self.assertEqual(error.exception.code, 2)
        self.client.models.assert_not_called()

    def test_parent_slash_resume_cannot_accidentally_select_a_child(self):
        with patch.object(cli, 'select_proof', side_effect=AssertionError('Must not choose unrelated proof')):
            self.assertEqual(self.run_cli(['--proof-strategy', 'cooperative', '--prompt', '/resume']), 1)
        self.runner.resume.assert_not_called()
        self.client.models.assert_not_called()
        self.assertIn('Cooperative parent resume is not supported', self.output())

    def test_interruption_retains_parent_directory_without_advertising_resume(self):
        with patch('mathagent.cooperative.run_cooperative_proof', side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_cli(['--proof-strategy', 'cooperative', '--prompt', 'Goal']), 130)
        self.assertIn('Cooperative proof interrupted', self.output())
        self.assertIn('.mathagent/cooperative/', self.output())
        self.assertIn('cannot be resumed', self.output())
        self.assertNotIn('discarded', self.output())
        self.assertNotIn('/resume ', self.output())


if __name__ == '__main__':
    unittest.main()
