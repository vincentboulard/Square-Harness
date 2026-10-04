"""Explicit tool registry; workspace-scoped text access and approved mutations."""
import difflib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile

TEXT = {'.tex', '.bib', '.sty', '.cls', '.md', '.txt', '.py', '.json', '.yaml', '.yml', '.toml', '.csv'}
# File groups the user can let the model discover and read. Pinned sources are
# explicit choices and stay readable whatever is enabled here.
READ_GROUPS = {'tex': {'.tex', '.bib', '.sty', '.cls'}, 'pdf': {'.pdf'}, 'py': {'.py'},
               'text': {'.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.csv'}}
SKIP = {'__pycache__', 'node_modules', 'venv', 'build', 'dist'}
MAX_BYTES = 1_000_000
MAX_RESULT = 6000


def schema(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': list(required), 'additionalProperties': False}}}


class Workspace:
    def __init__(self, root, approve=lambda preview: False, allow_python=False, literature=None, read_types=None):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('Workspace must be a directory')
        self.approve = approve
        self.allow_python = allow_python
        self.literature = literature
        # Set by the interface for answers: runs a literature check (a nested,
        # separately budgeted verifier) and returns its compact JSON result.
        self.checker = None
        groups = set(READ_GROUPS) if read_types is None else set(read_types)
        if groups - set(READ_GROUPS):
            raise ValueError('Unknown file group; use tex, pdf, py or text')
        self.readable = set().union(*(READ_GROUPS[g] for g in groups)) if groups else set()
        if self.literature is None:
            self.readable.discard('.pdf')  # PDF extraction uses the literature backend

    def path(self, filename, *, pdf=False):
        if not isinstance(filename, str) or not filename:
            raise ValueError('Provide a nonempty relative filename')
        rel = Path(filename)
        if rel.is_absolute() or '..' in rel.parts or any(p.startswith('.') for p in rel.parts):
            raise ValueError('Use a relative path; parent traversal and hidden paths are excluded')
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError('Path escapes workspace, possibly through a symlink')
        if any(part.startswith('.') for part in p.relative_to(self.root).parts):
            raise ValueError('Hidden paths are excluded, including symlink targets')
        if p.suffix.lower() not in TEXT and not (pdf and p.suffix.lower() == '.pdf'):
            raise ValueError('Unsupported extension; use text/LaTeX/Markdown/Python files')
        return p

    def _readable(self, p):
        if p.suffix.lower() not in self.readable:
            raise ValueError(f'The user has not allowed reading {p.suffix} files in this workspace; '
                             'ask them to pin the file or enable that file type')

    def text(self, p):
        st = p.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES:
            raise ValueError('Only regular text files up to 1 MB can be read')
        value = p.read_text(encoding='utf-8')
        if '\x00' in value:
            raise ValueError('Binary file excluded')
        return value

    def files(self, suffixes=None):
        allowed = TEXT if suffixes is None else suffixes
        count = 0
        for folder, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in SKIP
                             and not (Path(folder) / d).is_symlink())
            for name in sorted(files):
                p = Path(folder) / name
                if name.startswith('.') or p.is_symlink() or p.suffix.lower() not in allowed:
                    continue
                yield p
                count += 1
                if count >= 2000:
                    return

    def list_files(self):
        return '\n'.join(str(p.relative_to(self.root)) for p in self.files(self.readable)) or '(no readable files)'

    def read_file(self, path, start_line=1, end_line=100):
        if type(start_line) is not int or type(end_line) is not int or not 1 <= start_line <= end_line:
            raise ValueError('Line bounds must be positive integers with start <= end')
        p = self.path(path, pdf=True)
        self._readable(p)
        if p.suffix.lower() == '.pdf':
            # Local extraction, cached with the literature evidence; nothing is uploaded.
            summary = self.literature.import_local(str(p.relative_to(self.root)))
            lines = self.literature._document(summary['document_id'])[1]
        else:
            lines = self.text(p).splitlines()
        end = min(end_line, start_line + 199, len(lines))
        result = f'{path}: {len(lines)} total lines; requested excerpt {start_line}-{end}\n'
        return result + '\n'.join(f'{i+1}: {lines[i]}' for i in range(start_line-1, end))

    def search_text(self, pattern):
        if not isinstance(pattern, str) or not pattern:
            raise ValueError('Provide nonempty literal search text (not regex)')
        hits = []
        for p in self.files(self.readable - {'.pdf'}):
            try:
                lines = self.text(p).splitlines()
            except (OSError, ValueError, UnicodeError):
                continue
            for i, line in enumerate(lines, 1):
                if pattern.casefold() in line.casefold():
                    hits.append(f'{p.relative_to(self.root)}:{i}: {line[:500]}')
                    if len(hits) >= 50:
                        return '\n'.join(hits) + '\n[50-match limit]'
        return '\n'.join(hits) or '(no matches in supported files scanned)'

    def write_file(self, path, content):
        if not isinstance(content, str) or len(content.encode()) > 100_000:
            raise ValueError('Content must be text, at most 100 KB')
        p = self.path(path)
        old = self.text(p) if p.exists() else ''
        diff = ''.join(difflib.unified_diff(old.splitlines(True), content.splitlines(True),
                                           fromfile=path + ' (old)', tofile=path + ' (new)'))
        if not self.approve(f'WRITE {p}\n{diff or "(no textual changes)"}'):
            return 'User denied the write. No file changed.'
        p = self.path(path)  # Resolve again after the interactive approval.
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return f'Wrote {path} ({len(content.encode())} bytes)'

    def run_python(self, code):
        if not self.allow_python:
            return 'Python execution disabled; restart with --allow-python to enable approval prompts.'
        if not isinstance(code, str) or len(code) > 20000:
            raise ValueError('Provide Python source, at most 20000 characters')
        if not self.approve('RUN PYTHON (not sandboxed; has your account permissions)\n' + code):
            return 'User denied Python execution.'
        # A timeout and output-file limit are guardrails, NOT a security sandbox.
        wrapper = '''import resource
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
'''
        with tempfile.TemporaryFile() as out:
            p = subprocess.Popen([sys.executable, '-c', wrapper + code], cwd=self.root,
                                 stdout=out, stderr=out, start_new_session=True)
            timed_out = False
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also clean up ordinary descendants after an early parent exit.
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            out.seek(0)
            output = out.read(MAX_RESULT).decode('utf-8', errors='replace')
        return f'Exit {p.returncode}; timeout={timed_out}\n{output}\n[output capped at 6000 bytes]'

    def check_reference(self, statement, guesses=None):
        if not isinstance(statement, str) or not statement.strip() or len(statement) > 2000:
            raise ValueError('Describe the result to find a reference for, in at most 2000 characters')
        if guesses is not None and (not isinstance(guesses, list) or not all(isinstance(g, dict) for g in guesses)):
            raise ValueError('guesses must be a list of objects')
        return self.checker(statement.strip(), (guesses or [])[:3])

    def schemas(self):
        string = {'type': 'string'}
        integer = {'type': 'integer'}
        kinds = ', '.join(sorted(self.readable))
        result = [] if not self.readable else [
            schema('list_files', f'List nonhidden files the user allows you to read ({kinds}; scan cap 2000).', {}),
            schema('read_file', 'Read numbered lines of an allowed file' + (', including extracted PDF text' if '.pdf' in self.readable else '')
                   + '. At most 200 lines / 6000 characters.',
                   {'path': string, 'start_line': integer, 'end_line': integer}, ['path']),
            schema('search_text', 'Case-insensitive literal search in allowed text files (not PDFs); max 50 matches.',
                   {'pattern': string}, ['pattern']),
        ]
        result += [
            schema('write_file', 'Create/replace a text file ONLY when requested by the user; shows a diff for approval.',
                   {'path': string, 'content': string}, ['path', 'content']),
        ]
        if self.allow_python and (self.literature is None or self.literature.online):
            result.append(schema('run_python', 'Run Python after explicit approval. Not sandboxed. 30s timeout.',
                                 {'code': string}, ['code']))
        if self.literature is not None:
            result.extend(self.literature.schemas())
        if self.checker is not None and self.literature is not None and self.literature.online:
            guess = {'type': 'object', 'properties': {
                'authors': {'type': 'array', 'items': string}, 'title': string, 'year': string,
                'locator': {'type': 'string', 'description': 'Theorem, proposition or section, e.g. "Theorem 9.26"'}}}
            result.append(schema('check_reference',
                'Find and verify a precise reference (book or paper, and the theorem or section) for a known result. '
                'Give the result and your best guesses of where it is stated; a separate verifier checks them against '
                'zbMATH, Crossref, OpenCitations and open papers and returns how far each was confirmed (read, cited, located, not '
                'found). Takes a few minutes. Use it before giving a theorem number from memory.',
                {'statement': string, 'guesses': {'type': 'array', 'items': guess}}, ['statement']))
        return result

    def execute(self, name, arguments):
        allowed = {s['function']['name'] for s in self.schemas()}
        try:
            if name not in allowed:
                raise ValueError(f'Unknown or disabled tool: {name}')
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be an object')
            if self.literature is not None and name in {
                    item['function']['name'] for item in self.literature.schemas()}:
                # The literature layer returns bounded valid JSON and handles
                # its own offline gate. Do not split citation JSON mid-record.
                return self.literature.execute(name, arguments)
            value = str(getattr(self, name)(**arguments))
            return value if len(value) <= MAX_RESULT else value[:MAX_RESULT] + '\n[TRUNCATED: request a smaller excerpt]'
        except (OSError, ValueError, TypeError, UnicodeError) as e:
            return f'Tool error: {e}'
