"""Offline-default routing, report export and research budget CLI boundaries."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mathagent import cli


class ResearchCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ui = Mock()
        self.ui.approve.return_value = False
        self.client = Mock()
        self.client.models.return_value = ['qwen3.8:27b']
        self.runner = Mock()
        self.result = {'id': 'research-example', 'status': 'reviewed',
                       'report': '# Literature report\n\nEvidence remains limited.\n',
                       'directory': str(self.root / '.mathagent/research/research-example')}
        self.runner.start.return_value = self.result
        self.runner.resume.return_value = self.result

    def run_cli(self, args):
        with patch('sys.argv', ['mathagent', '--workspace', str(self.root), *args]), \
                patch.object(cli, 'UI', return_value=self.ui), \
                patch.object(cli, 'Ollama', return_value=self.client), \
                patch.object(cli, 'ResearchRunner', return_value=self.runner) as runner_class, \
                patch.object(cli, 'ReviewRunner', return_value=self.runner):
            runner_class.list.return_value = [dict(self.result, kind='literature', goal='A topic')]
            runner_class.inspect.return_value = self.result
            return cli.main()

    def test_literature_mode_receives_all_saved_budgets(self):
        with patch.object(cli, 'LiteratureTools', wraps=cli.LiteratureTools) as library:
            self.assertEqual(self.run_cli(['--mode', 'literature', '--research-rounds', '3',
                '--research-tokens', '5000', '--research-input-tokens', '18000',
                '--research-seconds', '120', '--research-requests', '4', '--research-chars', '8000',
                '--prompt', 'Compare local estimates.']), 0)
            self.assertFalse(library.call_args.kwargs['online'])
        self.runner.start.assert_called_once_with('Compare local estimates.', kind='literature',
            source_files=[], max_rounds=3, max_tokens=5000, max_input_tokens=18000,
            max_seconds=120.0, max_requests=4, max_chars=8000)

    def test_referee_slash_command_pins_source_and_exports_markdown(self):
        self.assertEqual(self.run_cli(['--research-file', 'paper.tex', '--output', 'reports/review.md',
                                      '--prompt', '/referee Check the argument and compare prior work.']), 0)
        self.assertEqual(self.runner.start.call_args.kwargs['kind'], 'referee')
        self.assertEqual(self.runner.start.call_args.kwargs['source_files'], ['paper.tex'])
        self.assertEqual((self.root / 'reports/review.md').read_text(), self.result['report'])

    def test_report_export_never_silently_overwrites(self):
        target = self.root / 'report.md'
        target.write_text('Original manuscript notes')
        self.assertEqual(self.run_cli(['--mode', 'literature', '--output', 'report.md', '--prompt', 'Topic']), 0)
        self.assertEqual(target.read_text(), 'Original manuscript notes')
        self.ui.approve.assert_called_once()

    def test_invalid_export_stops_before_model(self):
        for dest in ['../outside.md', 'bad.txt']:
            with self.subTest(dest=dest):
                self.assertEqual(self.run_cli(['--mode', 'literature', '--output', dest, '--prompt', 'Topic']), 1)
        self.client.models.assert_not_called()
        self.runner.start.assert_not_called()

    def test_saved_report_and_listing_need_no_model(self):
        self.assertEqual(self.run_cli(['--prompt', '/researches']), 0)
        self.assertEqual(self.run_cli(['--prompt', '/research-report research-example']), 0)
        self.assertEqual(self.run_cli(['--prompt', '/skills']), 0)
        self.client.models.assert_not_called()
        self.client.stream.assert_not_called()

    def test_research_resume_does_not_supply_new_budget(self):
        self.assertEqual(self.run_cli(['--research-resume', 'research-example', '--research-tokens', '999999']), 0)
        self.runner.resume.assert_called_once_with('research-example')
        self.client.models.assert_not_called()

    def test_external_model_host_and_python_require_explicit_online(self):
        for options in [['--host', 'http://example.org:11434'], ['--allow-python']]:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli([*options, '--prompt', '/status'])
        self.client.models.assert_not_called()
        self.assertEqual(self.run_cli(['--online', '--host', 'http://example.org:11434', '--prompt', '/status']), 0)

    def test_explicit_offline_and_online_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli(['--online', '--offline', '--prompt', '/status'])

    def test_invalid_research_budget_stops_before_model(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli(['--research-requests', '-1', '--prompt', '/status'])
        self.client.models.assert_not_called()

    def test_proof_does_not_acquire_literature_tools_from_online_chat(self):
        proof = Mock()
        proof.start.return_value = dict(self.result, status='paused')
        with patch.object(cli, 'ProofRunner', return_value=proof):
            self.assertEqual(self.run_cli(['--online', '--prompt', 'Prove x=x.']), 0)
            self.assertNotIn('allow_literature', proof.start.call_args.kwargs)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_cli(['--online', '--proof-literature', '--prompt', 'Prove x=x.'])



class WriteupCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ui, self.client, self.runner = Mock(), Mock(), Mock()
        self.client.models.return_value = ['qwen3.8:27b']
        self.runner.start.return_value = {'id': 'writeup-example', 'status': 'partial', 'report': '# Write-up\n',
                                          'directory': str(self.root / '.mathagent/research/writeup-example')}

    def run_cli(self, args):
        with patch('sys.argv', ['mathagent', '--workspace', str(self.root), *args]), \
                patch.object(cli, 'UI', return_value=self.ui), \
                patch.object(cli, 'Ollama', return_value=self.client), \
                patch.object(cli, 'WriteupRunner', return_value=self.runner):
            return cli.main()

    def test_writeup_pins_notes_and_template_and_names_the_output(self):
        self.assertEqual(self.run_cli(['--research-file', 'notes.md', '--template-file', 'macros.sty',
                                      '--output', 'lemma.tex', '--prompt', '/writeup A short section.']), 0)
        call = self.runner.start.call_args
        self.assertEqual(call.args[0], 'A short section.')
        self.assertEqual((call.kwargs['source_files'], call.kwargs['template_files'], call.kwargs['output']),
                         (['notes.md'], ['macros.sty'], 'lemma.tex'))
        self.assertFalse((self.root / 'lemma.tex').exists())  # the runner writes it; the CLI exports no report there

    def test_writeup_output_must_be_tex(self):
        self.assertEqual(self.run_cli(['--mode', 'writeup', '--output', 'lemma.md', '--prompt', 'Write up.']), 1)
        self.runner.start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
