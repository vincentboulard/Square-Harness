"""Folders, file access, uploads, Free mode routing and the Write-up skill."""
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


def route_reply(mode, request, files=(), question=''):
    return text(json.dumps({'mode': mode, 'request': request, 'files': list(files),
                            'reason': 'Because.', 'question': question}))


class RouterTests(unittest.TestCase):
    def test_malformed_or_unknown_answers_become_a_question(self):
        self.assertEqual(router.parse('not json', 'hi')['mode'], 'clarify')
        self.assertEqual(router.parse(json.dumps({'mode': 'hack'}), 'hi')['mode'], 'clarify')
        route = router.parse(json.dumps({'mode': 'prove', 'request': '', 'files': ['a.tex', 3],
                                         'reason': 'r', 'question': ''}), 'Prove it.', ['b.tex'])
        self.assertEqual((route['mode'], route['request'], route['files']), ('prove', 'Prove it.', ['b.tex', 'a.tex']))


# A verifier verdict that stops a proof job at once, with its candidate kept.
UNCERTAIN = {'explanation': 'The review cannot decide whether the argument is complete.', 'issues': [], 'verdict': 'uncertain'}


class FreeModeTests(GuiCase):
    def route(self, chat, content, files=()):
        self.ok('POST', f'/api/chats/{chat}/route', {'content': content, 'files': list(files)})
        return self.wait_for(lambda: [i for i in self.ok('GET', f'/api/chats/{chat}')['transcript']
                                      if i['role'] == 'route' and i['status'] == 'proposed'], message='a route')[-1]

    def test_one_conversation_mixes_a_proof_a_critique_and_a_referee_report(self):
        (self.root / 'paper.tex').write_text('Claim: x=x.\nProof: reflexivity.\n')
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [route_reply('prove', 'Prove that every real x satisfies x=x.')]
        suggestion = self.route(chat, 'Show that x equals itself')
        self.assertEqual(suggestion['mode'], 'prove')
        self.assertEqual(self.fake.requests[-1]['format']['properties']['mode']['enum'][-1], 'clarify')
        self.assertFalse(self.fake.requests[-1]['think'])
        transcript = self.ok('GET', f'/api/chats/{chat}')['transcript']
        index = transcript.index(suggestion)

        # Starting runs a real proof job (solve, then review); its ID is recorded on the card.
        self.fake.replies = [text('Unfinished attempt.'), text(json.dumps(UNCERTAIN))]
        started = self.ok('POST', f'/api/chats/{chat}/routes/{index}/start',
                          {'mode': 'prove', 'request': suggestion['request'], 'files': []})
        self.assertEqual(started['task']['kind'], 'proof')
        self.wait_task(timeout=30)
        card = self.ok('GET', f'/api/chats/{chat}')['transcript'][index]
        self.assertEqual(card['status'], 'started')
        proof = self.ok('GET', f'/api/proofs/{card["job_id"]}')
        self.assertTrue(proof['goal'].startswith('Prove that every real x'))
        self.assertEqual((proof['status'], proof['answer']), ('uncertain', 'Unfinished attempt.'))
        self.assertEqual(self.request('POST', f'/api/chats/{chat}/routes/{index}/start',
                                      {'mode': 'prove', 'request': 'again', 'files': []})[0], 400)

        # The router sees the earlier proof when reading the next message.
        self.fake.replies = [route_reply('critic', 'Is reflexivity enough here?')]
        critique = self.route(chat, 'Is that argument enough?')
        self.assertIn('Harness ran prove', self.fake.requests[-1]['messages'][1]['content'])
        self.assertIn('Outcome: uncertain', self.fake.requests[-1]['messages'][1]['content'])
        index = self.ok('GET', f'/api/chats/{chat}')['transcript'].index(critique)
        self.fake.replies = [text('Yes: reflexivity is an axiom of equality.')]
        self.ok('POST', f'/api/chats/{chat}/routes/{index}/start', {'mode': 'critic', 'request': critique['request'], 'files': []})
        self.wait_task()
        transcript = self.ok('GET', f'/api/chats/{chat}')['transcript']
        self.assertEqual(transcript[-1]['content'], 'Yes: reflexivity is an axiom of equality.')
        self.assertEqual([i['role'] for i in transcript].count('user'), 2)  # no duplicated message

        self.fake.replies = [route_reply('referee', 'Referee paper.tex.', ['paper.tex', 'missing.tex', 'other.tex'])]
        (self.root / 'other.tex').write_text('An unrelated statement.\n')
        referee = self.route(chat, 'Now referee paper.tex and missing.tex')
        # Files the user never named are not attached, even when they exist.
        self.assertEqual((referee['files'], referee['missing']), (['paper.tex'], ['missing.tex']))

    def test_jobs_started_while_busy_wait_in_the_queue(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [route_reply('prove', 'Prove 1=1.')]
        first = self.route(chat, 'Prove 1=1')
        blocker = self.ok('POST', '/api/chats', {'mode': 'critic'})['id']
        self.fake.replies = [([{'message': {'content': 'w '}, 'done': False}] * 150 + text('end'), 0.02)]
        self.ok('POST', f'/api/chats/{blocker}/messages', {'content': 'Long.'})
        self.wait_for(lambda: self.ok('GET', '/api/task')['task']['state'] == 'running', message='running')
        index = self.ok('GET', f'/api/chats/{chat}')['transcript'].index(first)
        queued = self.ok('POST', f'/api/chats/{chat}/routes/{index}/start', {'mode': 'prove', 'request': 'Prove 1=1.', 'files': []})
        self.assertEqual(queued['task']['state'], 'queued')
        snapshot = self.ok('GET', '/api/task')
        self.assertEqual([t['id'] for t in snapshot['queue']], [queued['task']['id']])
        self.assertEqual(self.ok('GET', f'/api/chats/{chat}')['transcript'][index]['status'], 'queued')
        self.ok('POST', f'/api/queue/{queued["task"]["id"]}/cancel')
        self.assertEqual(self.ok('GET', '/api/task')['queue'], [])
        self.assertEqual(self.ok('GET', f'/api/chats/{chat}')['transcript'][index]['status'], 'cancelled')
        self.ok('POST', '/api/task/pause')
        self.wait_task()

    def test_clarifying_questions_and_validation(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [text('garbage')]
        self.ok('POST', f'/api/chats/{chat}/route', {'content': 'Hmm'})
        card = self.wait_for(lambda: [i for i in self.ok('GET', f'/api/chats/{chat}')['transcript'] if i['role'] == 'route'],
                             message='a route')[0]
        self.assertEqual(card['mode'], 'clarify')
        index = self.ok('GET', f'/api/chats/{chat}')['transcript'].index(card)
        self.assertEqual(self.request('POST', f'/api/chats/{chat}/routes/{index}/start',
                                      {'mode': 'clarify', 'request': 'x', 'files': []})[0], 400)
        self.assertEqual(self.ok('POST', f'/api/chats/{chat}/routes/{index}/dismiss')['item']['status'], 'dismissed')
        critic = self.ok('POST', '/api/chats', {'mode': 'critic'})['id']
        self.assertEqual(self.request('POST', f'/api/chats/{critic}/route', {'content': 'x'})[0], 400)


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


class OnlineAndEffortTests(GuiCase):
    def test_online_search_is_a_per_conversation_choice(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'critic'})
        self.assertFalse(chat['settings']['online'])
        self.fake.replies = [text('Offline answer.')]
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Search please.'})
        self.wait_task()
        tools = [t['function']['description'] for t in self.fake.requests[-1]['tools'] if t['function']['name'] == 'search_papers']
        self.assertIn('Offline: cached data only.', tools[0])
        self.assertTrue(self.ok('POST', f'/api/chats/{chat["id"]}/settings', {'online': True})['settings']['online'])
        self.fake.replies = [text('Online answer.')]
        self.ok('POST', f'/api/chats/{chat["id"]}/messages', {'content': 'Search please.'})
        self.wait_task()
        tools = [t['function']['description'] for t in self.fake.requests[-1]['tools'] if t['function']['name'] == 'search_papers']
        self.assertNotIn('Offline', tools[0])
        self.assertFalse(self.ok('GET', '/api/status')['online_locked'])

    def test_reports_record_the_choice_and_card_effort_reaches_the_job(self):
        (self.root / 'paper.tex').write_text('Claim.\n')
        self.fake.replies = [text('Plan.'), text('Notes.'), text('# Report\n'), text('Review.'), text('# Report\n')]
        started = self.ok('POST', '/api/research', {'kind': 'literature', 'goal': 'Survey.', 'online': True,
                                                    'rounds': 1, 'tokens': 12000, 'input_tokens': 60000})
        self.wait_task(timeout=30)
        self.assertTrue(self.ok('GET', f'/api/research/{started["id"]}')['settings']['online'])
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [text(json.dumps({'mode': 'prove', 'request': 'Prove 1=1.', 'files': [], 'reason': 'r', 'question': ''}))]
        self.ok('POST', f'/api/chats/{chat}/route', {'content': 'Prove 1=1'})
        self.wait_for(lambda: any(i['role'] == 'route' for i in self.ok('GET', f'/api/chats/{chat}')['transcript']), message='a route')
        index = next(i for i, item in enumerate(self.ok('GET', f'/api/chats/{chat}')['transcript']) if item['role'] == 'route')
        self.fake.replies = [text('Attempt.'), text(json.dumps(UNCERTAIN))]
        self.ok('POST', f'/api/chats/{chat}/routes/{index}/start', {'mode': 'prove', 'request': 'Prove 1=1.', 'files': [],
                                                                  'limits': {'rounds': 1, 'tokens': 7000, 'seconds': 90}})
        self.wait_task(timeout=30)
        job = self.ok('GET', f'/api/chats/{chat}')['transcript'][index]['job_id']
        settings = self.ok('GET', f'/api/proofs/{job}')['settings']
        self.assertEqual((settings['max_rounds'], settings['max_tokens'], settings['max_seconds']), (1, 7000, 90.0))
        self.assertEqual(self.request('POST', f'/api/chats/{chat}/routes/{index}/start',
                                      {'mode': 'prove', 'request': 'x', 'files': [], 'limits': {'rounds': 0}})[0], 400)

    def test_free_mode_answers_record_the_workflow_that_answered(self):
        chat = self.ok('POST', '/api/chats', {'mode': 'free'})['id']
        self.fake.replies = [text(json.dumps({'mode': 'explore', 'request': 'Ideas?', 'files': [], 'reason': 'r', 'question': ''}))]
        self.ok('POST', f'/api/chats/{chat}/route', {'content': 'Ideas?'})
        self.wait_for(lambda: any(i['role'] == 'route' for i in self.ok('GET', f'/api/chats/{chat}')['transcript']), message='a route')
        index = next(i for i, item in enumerate(self.ok('GET', f'/api/chats/{chat}')['transcript']) if item['role'] == 'route')
        self.fake.replies = [text('Some ideas.')]
        self.ok('POST', f'/api/chats/{chat}/routes/{index}/start', {'mode': 'explore', 'request': 'Ideas?', 'files': []})
        self.wait_task()
        answer = self.ok('GET', f'/api/chats/{chat}')['transcript'][-1]
        self.assertEqual((answer['content'], answer['mode']), ('Some ideas.', 'explore'))
        self.assertIn('Explore approaches', self.fake.requests[-1]['messages'][0]['content'])


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
