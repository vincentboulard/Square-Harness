"""Folders, file access, uploads, Default mode routing, the job queue and the Write-up skill."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    from .test_gui import FakeOllama, GuiCase, text, tool
except ImportError:  # unittest discover imports test modules as top-level names
    from test_gui import FakeOllama, GuiCase, text, tool

from mathagent import router
from mathagent.agent import Agent
from mathagent.gui import store
from mathagent.literature import LiteratureTools
from mathagent.tools import Workspace
from mathagent.writeup import WriteupRunner, compile_latex, template_vocabulary, untraced_paragraphs


class WorkspaceReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, content in (('lemma.tex', 'Lemma\n'), ('calc.py', 'print(1)\n'), ('notes.md', 'notes\n'),
                              ('macros.sty', '\\newcommand{\\R}{\\mathbb{R}}\n')):
            (self.root / name).write_text(content)
        (self.root / 'paper.pdf').write_bytes(b'%PDF sample')

    def test_default_reads_everything_including_pdf_text(self):
        library = LiteratureTools(self.root)
        workspace = Workspace(self.root, literature=library)
        listed = workspace.list_files().splitlines()
        self.assertEqual(sorted(listed), ['calc.py', 'lemma.tex', 'macros.sty', 'notes.md', 'paper.pdf'])
        with patch.object(LiteratureTools, '_pdf', return_value=('Theorem 1.\nProof.', 'mock')):
            self.assertIn('2: Proof.', workspace.read_file('paper.pdf'))
        self.assertIn('\\newcommand', workspace.read_file('macros.sty'))

    def test_unticked_types_are_invisible_and_unreadable(self):
        workspace = Workspace(self.root, literature=LiteratureTools(self.root), read_types=['tex'])
        self.assertEqual(sorted(workspace.list_files().splitlines()), ['lemma.tex', 'macros.sty'])
        self.assertIn('not allowed', workspace.execute('read_file', {'path': 'calc.py'}))
        self.assertIn('not allowed', workspace.execute('read_file', {'path': 'paper.pdf'}))
        self.assertNotIn('calc.py', workspace.search_text('print'))
        # Pinned sources remain available to the runners through path()/text().
        self.assertEqual(workspace.text(workspace.path('calc.py')), 'print(1)\n')

    def test_nothing_readable_removes_the_file_tools(self):
        names = [s['function']['name'] for s in Workspace(self.root, read_types=[]).schemas()]
        self.assertNotIn('read_file', names)
        self.assertNotIn('list_files', names)
        self.assertIn('write_file', names)
        with self.assertRaises(ValueError):
            Workspace(self.root, read_types=['exe'])


class FolderTests(GuiCase):
    def setUp(self):
        super().setUp()
        self.server.watcher = self.watcher
        (self.root / 'papers' / 'draft').mkdir(parents=True)
        (self.root / 'papers' / 'draft' / 'a.tex').write_text('x')
        (self.root / '.secret').mkdir()
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside)
        (self.root / 'escape').symlink_to(outside)

    def test_browsing_stays_below_the_root(self):
        listing = self.ok('GET', '/api/folders')
        self.assertEqual([f['name'] for f in listing['folders']], ['papers'])
        inner = self.ok('GET', '/api/folders?path=papers')
        self.assertEqual(inner['folders'][0], {'name': 'draft', 'path': 'papers/draft', 'files': {'tex': 1, 'pdf': 0, 'py': 0}, 'jobs': False})
        for bad in ('..', '.secret', 'escape', '/etc', 'papers/../..'):
            status, _, _ = self.request('GET', '/api/folders?path=' + bad)
            self.assertIn(status, (400, 404), bad)
        created = self.ok('POST', '/api/folders', {'path': 'papers', 'name': 'new paper'})
        self.assertEqual(created['path'], 'papers/new paper')
        self.assertEqual(self.request('POST', '/api/folders', {'path': '', 'name': '.hidden'})[0], 400)

    def test_switching_folders_changes_jobs_settings_and_is_refused_while_busy(self):
        self.ok('POST', '/api/workspace', {'path': 'papers/draft'})
        status = self.ok('GET', '/api/status')
        self.assertEqual(status['workspace_path'], 'papers/draft')
        self.assertEqual(status['recent'], ['papers/draft'])
        self.assertEqual(self.ok('GET', '/api/files')['files'][0]['path'], 'a.tex')
        saved = self.ok('POST', '/api/settings', {'read': {'pdf': False}})
        self.assertEqual(saved['read'], {'tex': True, 'pdf': False, 'py': True, 'text': True})
        self.assertFalse((self.root / '.mathagent' / 'gui-settings.json').exists())
        self.assertTrue((self.root / 'papers' / 'draft' / '.mathagent' / 'gui-settings.json').exists())
        self.assertEqual(self.request('POST', '/api/settings', {'read': {'pdf': 'no'}})[0], 400)
        self.assertEqual(self.request('POST', '/api/workspace', {'path': 'escape'})[0], 400)
        slow = [{'message': {'content': 'word '}, 'done': False}] * 100 + text('end')
        self.fake.replies = [(slow, 0.02)]
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Go.'})
        self.wait_for(lambda: self.ok('GET', '/api/task')['task']['state'] == 'running', message='running')
        status, value, _ = self.request('POST', '/api/workspace', {'path': ''})
        self.assertEqual(status, 409)
        self.ok('POST', '/api/task/pause')
        self.wait_task()
        self.ok('POST', '/api/workspace', {'path': ''})
        self.assertEqual(self.ok('GET', '/api/status')['workspace_path'], '')


class UploadTests(GuiCase):
    def upload(self, name, data, headers=None):
        return self.request('POST', '/api/files/upload?name=' + name, raw=data,
                            headers={'Content-Type': 'application/octet-stream', **(headers or {})})

    def test_dropped_files_are_saved_without_replacing_anything(self):
        status, value, _ = self.upload('lemma.tex', b'\\begin{lemma}x\\end{lemma}\n')
        self.assertEqual((status, value['path']), (200, 'lemma.tex'))
        status, value, _ = self.upload('lemma.tex', b'second\n')
        self.assertEqual(value['path'], 'lemma-2.tex')
        self.assertEqual((self.root / 'lemma.tex').read_bytes(), b'\\begin{lemma}x\\end{lemma}\n')
        status, value, _ = self.upload('..%2F..%2Fevil.tex', b'x')
        self.assertEqual((status, value['path']), (200, 'evil.tex'))
        self.assertTrue((self.root / 'evil.tex').exists())
        for name, data in (('run.sh', b'x'), ('.bashrc.tex', b'x'), ('bin.tex', b'\x00\x01')):
            self.assertEqual(self.upload(name, data)[0], 400, name)
        self.assertEqual(self.upload('x.tex', b'x', {'Origin': 'http://attacker.example'})[0], 403)
        status, _, _ = self.request('POST', '/api/files/upload?name=y.tex', raw=b'x', auth=False,
                                    headers={'Content-Type': 'application/octet-stream'})
        self.assertEqual(status, 401)
        self.assertFalse((self.root / 'y.tex').exists())


def task(mode, request, files=(), effort='medium'):
    return {'mode': mode, 'request': request, 'files': list(files), 'effort': effort, 'reason': 'Because.'}


def route_reply(mode, request, files=(), question='', effort='medium'):
    return text(json.dumps({'tasks': [task(mode, request, files, effort)], 'question': question}))


class RouterTests(unittest.TestCase):
    def test_malformed_or_unknown_answers_become_a_question(self):
        self.assertEqual(router.parse('not json', 'hi')[0]['mode'], 'clarify')
        self.assertEqual(router.parse(json.dumps({'tasks': [{'mode': 'hack'}], 'question': ''}), 'hi')[0]['mode'], 'clarify')
        [route] = router.parse(json.dumps({'tasks': [{'mode': 'prove', 'request': '', 'files': ['a.tex', 3],
                                                      'effort': 'extreme', 'reason': 'r'}], 'question': ''}), 'Prove it.', ['b.tex'])
        self.assertEqual((route['mode'], route['request'], route['files'], route['effort']),
                         ('prove', 'Prove it.', ['b.tex', 'a.tex'], 'medium'))

    def test_one_message_can_ask_for_several_jobs_with_their_effort(self):
        routes = router.parse(json.dumps({'tasks': [task('prove', 'Prove A.', effort='high'),
                                                    task('literature', 'Survey A.', effort='low')] * 3,
                                          'question': ''}), 'Prove A and survey it.')
        self.assertEqual(len(routes), router.MAX_TASKS)
        self.assertEqual([(r['mode'], r['effort']) for r in routes[:2]], [('prove', 'high'), ('literature', 'low')])
        [question] = router.parse(json.dumps({'tasks': [], 'question': 'Which statement?'}), 'Do the thing.')
        self.assertEqual((question['mode'], question['question']), ('clarify', 'Which statement?'))
        # The earlier single-task answer shape still reads as one suggestion.
        [check] = router.parse(json.dumps({'tasks': [task('check', 'Find a reference for Rellich.', effort='low')], 'question': ''}), 'Ref?')
        self.assertEqual((check['mode'], check['effort']), ('check', 'low'))
        [old] = router.parse(json.dumps({'mode': 'referee', 'request': 'Review paper.tex.', 'files': [],
                                         'reason': 'r', 'question': ''}), 'Review it.')
        self.assertEqual((old['mode'], old['effort']), ('referee', 'medium'))
        schema = router.ROUTE_SCHEMA['properties']['tasks']
        self.assertEqual(schema['maxItems'], router.MAX_TASKS)
        self.assertEqual(schema['items']['properties']['effort']['enum'], list(router.EFFORTS))

    def test_tex_commands_survive_single_backslashes_in_json(self):
        # In JSON, a lone backslash before f, b, t, r or n is a control character, not TeX.
        answer = ('{"tasks": [{"mode": "prove", "request": "Show \\frac{1}{2} < \\beta, \\nabla u \\neq 0, '
                  '\\theta and \\rho.\\nu_n stays on its line.", "files": [], "effort": "low", "reason": "r"}], "question": ""}')
        [route] = router.parse(answer, 'Prove it.')
        self.assertEqual(route['request'], 'Show \\frac{1}{2} < \\beta, \\nabla u \\neq 0, \\theta and \\rho.\nu_n stays on its line.')


# A verifier verdict that stops a proof job at once, with its candidate kept.
UNCERTAIN = {'explanation': 'The review cannot decide whether the argument is complete.', 'issues': [], 'verdict': 'uncertain'}


class LegacyRouteTests(GuiCase):
    """Old saved suggestion cards remain usable after the Assistant upgrade."""

    def test_suggestions_saved_before_auto_start_can_still_start_with_their_effort(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        index = store.append_items(self.root, chat, [{'role': 'route', 'status': 'proposed', 'mode': 'prove',
                                                      'request': 'Prove 1=1.', 'files': [], 'effort': 'medium',
                                                      'reason': '', 'question': '', 'time': store._now()}])
        self.assertEqual(self.request('POST', f'/api/chats/{chat}/routes/{index}/start',
                                      {'mode': 'prove', 'request': 'x', 'files': [], 'limits': {'rounds': 0}})[0], 400)
        self.fake.replies = [text('Attempt.'), text(json.dumps(UNCERTAIN))]
        self.ok('POST', f'/api/chats/{chat}/routes/{index}/start', {'mode': 'prove', 'request': 'Prove 1=1.', 'files': [],
                                                                  'limits': {'rounds': 1, 'tokens': 7000, 'seconds': 90}})
        self.wait_task(timeout=30)
        job = self.ok('GET', f'/api/chats/{chat}')['transcript'][index]['job_id']
        settings = self.ok('GET', f'/api/proofs/{job}')['settings']
        self.assertEqual((settings['max_rounds'], settings['max_tokens'], settings['max_seconds']), (1, 7000, 90.0))

    def test_forms_and_resumes_queue_behind_a_running_job(self):
        blocker = self.ok('POST', '/api/chats', {'mode': 'critic'})['id']
        self.fake.replies = [([{'message': {'content': 'w '}, 'done': False}] * 150 + text('end'), 0.02)]
        self.ok('POST', f'/api/chats/{blocker}/messages', {'content': 'Long.'})
        self.wait_for(lambda: self.ok('GET', '/api/task')['task']['state'] == 'running', message='running')
        status, value, _ = self.request('POST', '/api/proofs', {'goal': 'Prove 1=1.'})
        self.assertEqual(status, 409)  # without the queue flag, a busy model still refuses
        proof = self.ok('POST', '/api/proofs', {'goal': 'Prove 1=1.', 'rounds': 1, 'tokens': 7000, 'queue': True})
        report = self.ok('POST', '/api/research', {'kind': 'referee', 'goal': 'Review paper.tex.', 'queue': True})
        (self.root / 'notes.md').write_text('u bounded\n')
        writeup = self.ok('POST', '/api/writeup', {'goal': 'Write up.', 'source_files': ['notes.md'], 'queue': True})
        self.assertEqual([t['task']['state'] for t in (proof, report, writeup)], ['queued'] * 3)
        self.assertEqual([t['id'] for t in (proof, report, writeup)], [None] * 3)
        self.assertEqual([t['label'] for t in self.ok('GET', '/api/task')['queue']], ['Proof', 'Review', 'Write-up'])
        for item in (proof, report, writeup):
            self.ok('POST', f'/api/queue/{item["task"]["id"]}/cancel')
        self.ok('POST', '/api/task/pause')
        self.wait_task()
        self.assertEqual(self.ok('GET', '/api/task')['queue'], [])




OUTLINE = {'title': 'Compactness notes', 'sections': [
    {'title': 'Setting', 'sources': ['M1:L1-L2'], 'goal': 'State the setting.'},
    {'title': 'Main result', 'sources': ['M1:L3-L4'], 'goal': 'State and prove the embedding.'}]}
SECTION_ONE = '\\section{Setting}\n% src: [M1:L1-L2]\nLet $u_n$ be bounded in $H^1(0,1)$ with values in $\\R$.'
SECTION_TWO = ('\\section{Main result}\n% src: [M1:L3-L4]\n\\begin{lemma}The embedding is compact.\\end{lemma}\n'
               '% TODO: the notes do not say which theorem gives compactness.')
NOTES = 'u_n bounded in H^1(0,1)\nvalues real\nembedding H^1 -> L^2 compact\nproof: ??\n'


class WriteupTests(GuiCase):
    def setUp(self):
        super().setUp()
        (self.root / 'notes.md').write_text(NOTES)
        (self.root / 'mymacros.sty').write_text('\\newcommand{\\R}{\\mathbb{R}}\n\\newtheorem{lemma}{Lemma}\n')

    def replies(self):
        return [text(json.dumps(OUTLINE)), text(SECTION_ONE), text(SECTION_TWO), text('Faithful; one TODO remains.')]

    def test_write_up_without_latex_is_exported_and_traced(self):
        self.fake.replies = self.replies()
        with patch('mathagent.writeup.shutil.which', return_value=None):
            started = self.ok('POST', '/api/writeup', {'goal': 'Write up my notes as a short article.',
                                                       'source_files': ['notes.md'], 'template_files': ['mymacros.sty'],
                                                       'output': 'notes-clean.tex', 'tokens': 12000, 'seconds': 60})
            self.assertEqual(started['task']['label'], 'Write-up')
            self.assertEqual(self.wait_task(timeout=30)['state'], 'done')
        detail = self.ok('GET', f'/api/research/{started["id"]}')
        self.assertEqual(detail['kind'], 'writeup')
        self.assertFalse(detail['settings']['online'])  # a write-up never searches, whatever the default
        self.assertEqual(detail['status'], 'partial')  # not compiled: never reported complete
        self.assertEqual(detail['outputs'], {'tex': 'notes-clean.tex'})
        document = (self.root / 'notes-clean.tex').read_text()
        self.assertIn('\\usepackage{mymacros}', document)
        self.assertIn('\\begin{lemma}', document)
        self.assertNotIn('\\newtheorem{theorem}', document)  # the template defines its own environments
        self.assertEqual(detail['citation_issues'], [])
        self.assertIn('R', detail['macros'])
        self.assertFalse(detail['compile']['available'])
        plan_request = self.fake.requests[0]
        self.assertEqual(plan_request['format']['required'], ['title', 'sections'])
        self.assertIn('\\newcommand{\\R}{\\mathbb{R}}', plan_request['messages'][1]['content'])
        self.assertIn('M1 L3: embedding H^1 -> L^2 compact', self.fake.requests[2]['messages'][1]['content'])
        self.assertEqual(self.request('GET', f'/api/research/{started["id"]}/pdf')[0], 404)
        review = self.fake.requests[3]['messages'][1]['content']
        self.assertIn('WRITE-UP TO REVIEW:', review)
        self.assertTrue(review.rstrip().endswith('Do not reproduce the document.'))
        self.assertIn('\\begin{lemma}', review)
        self.assertNotIn('review_context_issues', {k for k, v in detail.items() if v})

    def test_existing_output_is_never_replaced_and_inputs_are_checked(self):
        (self.root / 'notes-clean.tex').write_text('mine')
        self.fake.replies = self.replies()
        with patch('mathagent.writeup.shutil.which', return_value=None):
            started = self.ok('POST', '/api/writeup', {'goal': 'Write up.', 'source_files': ['notes.md'],
                                                       'output': 'notes-clean.tex', 'tokens': 12000})
            self.wait_task(timeout=30)
        self.assertEqual((self.root / 'notes-clean.tex').read_text(), 'mine')
        self.assertEqual(self.ok('GET', f'/api/research/{started["id"]}')['outputs']['tex'], 'notes-clean-2.tex')
        for body in ({'goal': 'Write up.'}, {'goal': 'x', 'source_files': ['notes.md'], 'template_files': ['notes.md']},
                     {'goal': 'x', 'source_files': ['notes.md'], 'output': 'out.md'}):
            status, value, _ = self.request('POST', '/api/writeup', body)
            self.assertEqual(status, 400, (body, value))

    def test_saved_template_is_reused_and_typed_notes_are_a_source(self):
        self.fake.replies = [text(json.dumps({'title': 'T', 'sections': [{'title': 'A', 'sources': ['M0:L1-L2'], 'goal': 'g'}]})),
                             text('\\section{A}\n% src: [M0:L1-L2]\n$\\R$'), text('Fine.')]
        with patch('mathagent.writeup.shutil.which', return_value=None):
            self.ok('POST', '/api/writeup', {'goal': 'Clean up.', 'notes': 'line one\nline two\n', 'template_files': ['mymacros.sty'],
                                             'save_template': 'My style', 'tokens': 12000})
            self.wait_task(timeout=30)
        self.assertEqual(self.ok('GET', '/api/templates')['templates'], [{'name': 'My style', 'files': ['mymacros.sty']}])
        self.fake.replies = [text(json.dumps({'title': 'T', 'sections': [{'title': 'A', 'sources': ['M0:L1-L1'], 'goal': 'g'}]})),
                             text('\\section{A}\n% src: [M0:L1-L1]\nText.'), text('Fine.')]
        with patch('mathagent.writeup.shutil.which', return_value=None):
            started = self.ok('POST', '/api/writeup', {'goal': 'Again.', 'notes': 'only line\n', 'template': 'My style', 'tokens': 12000})
            self.wait_task(timeout=30)
        detail = self.ok('GET', f'/api/research/{started["id"]}')
        self.assertEqual([s['path'] for s in detail['sources']], ['typed-notes.md', 'saved-template/mymacros.sty'])
        self.assertIn('\\usepackage{mymacros}', detail['document'])

    def test_a_review_that_copies_the_document_does_not_count(self):
        self.fake.replies = self.replies()[:-1] + [text('\\documentclass{amsart}\n\\begin{document}copied\\end{document}')]
        with patch('mathagent.writeup.shutil.which', return_value=None):
            started = self.ok('POST', '/api/writeup', {'goal': 'Write up.', 'source_files': ['notes.md'], 'tokens': 12000})
            self.wait_task(timeout=30)
        detail = self.ok('GET', f'/api/research/{started["id"]}')
        self.assertEqual((detail['review'], detail['review_complete']), ('', False))
        self.assertIn('The reviewer returned LaTeX instead of a review; the write-up is unreviewed.', detail['warnings'])

    @unittest.skipIf(shutil.which('latexmk') is None, 'latexmk is not installed')
    def test_compiles_with_the_template_macros_and_repairs_errors(self):
        broken = SECTION_TWO.replace('\\end{lemma}', '\\end{lemma}\n\\undefinedmacro')
        self.fake.replies = [text(json.dumps(OUTLINE)), text(SECTION_ONE), text(broken),
                             text(SECTION_TWO), text('Faithful.')]
        started = self.ok('POST', '/api/writeup', {'goal': 'Write up.', 'source_files': ['notes.md'],
                                                   'template_files': ['mymacros.sty'], 'tokens': 16000, 'seconds': 240})
        self.assertEqual(self.wait_task(timeout=240)['state'], 'done')
        detail = self.ok('GET', f'/api/research/{started["id"]}')
        self.assertTrue(detail['compile']['ok'], detail['compile'])
        self.assertIn('Fix the LaTeX compile errors', self.fake.requests[3]['messages'][1]['content'])
        self.assertEqual(detail['outputs'], {'tex': 'notes-writeup.tex', 'pdf': 'notes-writeup.pdf'})
        self.assertTrue((self.root / 'notes-writeup.pdf').read_bytes().startswith(b'%PDF'))
        status, body, headers = self.request('GET', f'/api/research/{started["id"]}/pdf')
        self.assertEqual((status, headers['Content-Type']), (200, 'application/pdf'))
        self.assertNotIn('Content-Security-Policy', headers)


class CompileSafetyTests(unittest.TestCase):
    @unittest.skipIf(shutil.which('latexmk') is None, 'latexmk is not installed')
    def test_no_shell_escape_and_no_absolute_reads(self):
        marker = Path(tempfile.gettempdir()) / f'square-escape-{os.getpid()}'
        result = compile_latex('\\documentclass{article}\\begin{document}\\immediate\\write18{touch '
                               + str(marker) + '}x\\end{document}')
        self.assertFalse(marker.exists())
        result = compile_latex('\\documentclass{article}\\begin{document}\\input{/etc/hostname}\\end{document}')
        self.assertFalse(result['ok'])

    def test_untraced_paragraphs_are_counted(self):
        latex = ('\\section{A}\nNo source here.\n\n% src: [M1:L1-L2]\nTraced.\n\n% TODO: only a note\n\n'
                 '% src: [M1:L3-L3]\n\\begin{lemma}x\\end{lemma}')
        self.assertEqual(untraced_paragraphs(latex), 1)

    def test_template_vocabulary(self):
        macros, theorems, definitions = template_vocabulary([{'content': '\\newcommand{\\R}{\\mathbb{R}}\n'
                                                              '\\def\\eps{\\varepsilon}\n\\newtheorem{lemma}{Lemma}\n\\newcommand{\\@x}{y}'}])
        self.assertEqual((macros, theorems), (['R', 'eps'], ['lemma']))
        self.assertEqual(definitions[0], '\\newcommand{\\R}{\\mathbb{R}}')


if __name__ == '__main__':
    unittest.main()


class EffortTableTests(unittest.TestCase):
    def test_saved_maximum_effort_routes_keep_their_budgets(self):
        from types import SimpleNamespace
        from mathagent.gui import effort
        args = SimpleNamespace(proof_solve_tokens=32768, proof_verify_tokens=16384,
                               proof_repair_tokens=None, proof_min_solve_tokens=None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chat = store.create_chat(root, 'free')
            chat['history'] = [{'role': 'user', 'content': 'Keep the original conversation.'}]
            chat['transcript'] = [{'role': 'user', 'content': 'Original request.', 'effort': 'brezis'}]
            for status, mode in (('proposed', 'prove'), ('failed', 'referee'), ('cancelled', 'writeup')):
                chat['transcript'].append({'role': 'route', 'status': status, 'mode': mode,
                                           'request': 'Work on the attached notes.', 'files': ['notes.tex'],
                                           'effort': 'brezis', 'reason': 'Maximum effort requested.',
                                           'time': chat['created_at'], 'error': 'Keep this detail.'})
            path = root / '.mathagent' / 'chats' / (chat['id'] + '.json')
            raw = json.dumps(chat, ensure_ascii=False).encode('utf-8')
            path.write_bytes(raw)

            loaded = store.load_chat(root, chat['id'])
            expected = json.loads(raw)
            for item in expected['transcript'][1:]:
                item['effort'] = 'poincare'
            self.assertEqual(loaded, expected)
            self.assertEqual(path.read_bytes(), raw)  # Loading remains read-only.
            for item in loaded['transcript'][1:]:
                with self.subTest(status=item['status'], mode=item['mode']):
                    self.assertEqual(effort.LABELS[item['effort']], 'Poincaré')
                    limits = effort.limits(item['mode'], item['effort'], args)
                    self.assertEqual((limits['seconds'], limits['tokens']), (7200, 200000))
                    if item['mode'] != 'writeup':
                        self.assertEqual(limits['rounds'], 10)
                    if item['mode'] == 'prove':
                        self.assertEqual((limits['solve_tokens'], limits['verify_tokens']), (32768, 16384))
                    else:
                        self.assertEqual(limits['input_tokens'], 800000)
                    if item['mode'] == 'referee':
                        self.assertEqual((limits['requests'], limits['chars']), (160, 150000))

    def test_server_and_interface_use_the_same_budgets(self):
        import re
        from mathagent.gui import effort
        source = Path(__file__).resolve().parents[1] / 'frontend' / 'src' / 'effort.ts'
        if not source.exists():
            self.skipTest('interface sources are not part of this distribution')
        rows = re.findall(r"id: '(\w+)', label: '([^']+)'.*?tries: (\d+), minutes: (\d+), tokens: ([\d_]+), "
                          r"input: ([\d_]+), requests: (\d+), chars: ([\d_]+)", source.read_text())
        table = {row[0]: dict(zip(('tries', 'minutes', 'tokens', 'input', 'requests', 'chars'),
                                  (int(v.replace('_', '')) for v in row[2:]))) for row in rows}
        self.assertEqual(table, effort.EFFORTS)
        self.assertEqual({row[0]: row[1] for row in rows}, effort.LABELS)

    def test_a_small_proof_budget_still_fits_one_solve_and_review(self):
        from types import SimpleNamespace
        from mathagent.gui import effort
        args = SimpleNamespace(proof_solve_tokens=32768, proof_verify_tokens=16384,
                               proof_repair_tokens=None, proof_min_solve_tokens=None)
        low = effort.limits('prove', 'low', args)
        self.assertEqual((low['rounds'], low['seconds'], low['tokens']), (1, 60, 30000))
        self.assertLessEqual(low['solve_tokens'] + low['verify_tokens'], low['tokens'])
        self.assertEqual(effort.limits('prove', 'medium', args)['solve_tokens'], 32768)
        self.assertEqual(effort.limits('referee', 'poincare', args)['seconds'], 7200)
        self.assertEqual(effort.limits('critic', 'high', args), {})


class OnlineAndEffortTests(GuiCase):
    def test_online_search_is_on_by_default_and_a_per_conversation_choice(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        self.assertTrue(chat['settings']['online'])
        self.fake.replies = [text('Online answer.')]
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Search please.'})
        self.wait_task()
        tools = [t['function']['description'] for t in self.fake.requests[-1]['tools'] if t['function']['name'] == 'search_papers']
        self.assertNotIn('Offline', tools[0])
        self.assertFalse(self.ok('POST', f'/api/chats/{chat["id"]}/settings', {'online': False})['settings']['online'])
        self.fake.replies = [text('Offline answer.')]
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Search please.'})
        self.wait_task()
        tools = [t['function']['description'] for t in self.fake.requests[-1]['tools'] if t['function']['name'] == 'search_papers']
        self.assertIn('Offline: cached data only.', tools[0])
        self.assertFalse(self.ok('POST', '/api/chats', {'mode': 'free', 'online': False})['settings']['online'])
        status = self.ok('GET', '/api/status')
        self.assertEqual((status['online'], status['online_locked']), (True, False))
        self.assertIsNone(status['defaults']['proof_repair_tokens'])

    def test_reports_record_the_online_choice(self):
        (self.root / 'paper.tex').write_text('Claim.\n')
        self.fake.replies = [text('Plan.'), text('Notes.'), text('# Report\n'), text('Review.'), text('# Report\n')]
        # No web request is allowed: the job records the online choice without reaching the network.
        started = self.ok('POST', '/api/research', {'kind': 'literature', 'goal': 'Survey.', 'online': True, 'requests': 0,
                                                    'rounds': 1, 'tokens': 12000, 'input_tokens': 60000})
        self.wait_task(timeout=30)
        self.assertTrue(self.ok('GET', f'/api/research/{started["id"]}')['settings']['online'])

    def test_assistant_records_the_worker_and_then_its_own_answer(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [tool('delegate', {'mode': 'explore', 'request': 'Explore these ideas.', 'files': [],
                                               'reason': 'Explore an alternative.', 'effort': 'low'}),
                             text('Some worker ideas.'), text('Some ideas.')]
        self.ok('POST', f'/api/chats/{chat}/route', {'content': 'Ideas?'})
        self.wait_for(lambda: any(item.get('role') == 'assistant' and item.get('content') == 'Some ideas.'
                                 for item in self.ok('GET', f'/api/chats/{chat}')['transcript']),
                      timeout=20, message='the answer')
        transcript = self.ok('GET', f'/api/chats/{chat}')['transcript']
        [worker] = [item for item in transcript if item['role'] == 'route']
        self.assertEqual((worker['mode'], worker['status'], worker['orchestrated']), ('explore', 'done', True))
        self.assertEqual(next(item['content'] for item in reversed(transcript) if item['role'] == 'assistant'), 'Some ideas.')
        self.assertIn('Explore approaches', self.fake.requests[1]['messages'][0]['content'])


RECALL = {'statement': 'H^2 regularity for the Neumann problem.', 'keywords': ['Neumann', 'regularity'],
          'candidates': [{'authors': ['Brezis'], 'title': 'Functional analysis', 'year': '2011', 'locator': 'Theorem 9.26', 'why': 'Chapter 9.'}]}
NOT_FOUND = {'level': 'not_found', 'locator': 'Theorem 9.26', 'document_id': '', 'lines': '', 'record': '',
             'statement_found': '', 'note': 'No source tool was used.'}


class LiteratureCheckTests(GuiCase):
    def setUp(self):
        super().setUp()
        from mathagent import refcheck
        compact = patch.dict(refcheck.EFFORTS, {'low': {**refcheck.EFFORTS['low'], 'candidates': 2, 'discover': False}})
        compact.start()
        self.addCleanup(compact.stop)

    def test_an_answer_can_call_check_reference_only_when_online(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})['id']
        guesses = [{'authors': ['Brezis'], 'title': 'Functional analysis', 'year': '2011', 'locator': 'Theorem 9.26'}]
        self.fake.replies = [tool('check_reference', {'statement': 'H^2 regularity for Neumann problems', 'guesses': guesses}),
                             text(json.dumps(RECALL)), text('Nothing to search.'), text(json.dumps(NOT_FOUND)),
                             text('Brezis is a likely source, but Theorem 9.26 is unverified.')]
        self.ok('POST', f'/api/chats/{chat}/messages', {'content': 'Which theorem gives Neumann regularity?'})
        self.wait_task(timeout=30)
        self.assertIn('check_reference', [t['function']['name'] for t in self.fake.requests[0]['tools']])
        nested = self.fake.requests[1]
        self.assertIn('already suggested', nested['messages'][1]['content'])
        self.assertNotIn('check_reference', [t['function']['name'] for r in self.fake.requests[1:4] for t in r.get('tools', [])])
        transcript = self.ok('GET', f'/api/chats/{chat}')['transcript']
        result = json.loads(next(i for i in transcript if i['role'] == 'tool')['content'])
        self.assertEqual(result['references'], [])
        self.assertEqual(result['not_confirmed'][0]['level'], 'not_found')
        self.assertEqual(transcript[-1]['content'], 'Brezis is a likely source, but Theorem 9.26 is unverified.')
        self.ok('POST', f'/api/chats/{chat}/settings', {'online': False})
        self.fake.replies = [text('Offline.')]
        self.ok('POST', f'/api/chats/{chat}/messages', {'content': 'Again?'})
        self.wait_task()
        self.assertNotIn('check_reference', [t['function']['name'] for t in self.fake.requests[-1]['tools']])


class OfflineLockTests(GuiCase):
    extra_args = ('--offline',)

    def test_explicit_offline_launch_locks_online_search_off(self):
        self.assertTrue(self.ok('GET', '/api/status')['online_locked'])
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        status, value, _ = self.request('POST', f'/api/chats/{chat["id"]}/settings', {'online': True})
        self.assertEqual(status, 400)
        self.assertIn('--offline', value['error'])
        self.assertEqual(self.request('POST', '/api/chats', {'mode': 'critic', 'online': True})[0], 400)
        self.assertEqual(self.request('POST', '/api/research', {'kind': 'literature', 'goal': 'x', 'online': True})[0], 400)
        self.assertEqual(self.fake.requests, [])
