"""Write-up: turn notes, drafts or PDFs into a LaTeX document in the user's style.

It reuses the research workflow: pinned snapshots, saved budgets, streams and
resume behave the same way. The controller reads the note lines each section
needs, so every `% src: [M1:L12-L30]` comment can be checked against lines that
were actually read. When latexmk is installed the result is compiled in a
temporary folder without shell escape and with TeX's paranoid file access.
The output is a draft: the mathematics still needs the author's check.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile

from .ledger import _atomic_write
from .research import ResearchRunner

TEMPLATE_SUFFIXES = ('.sty', '.cls', '.tex', '.bib')
REF = re.compile(r'(M\d+):L(\d+)(?:-L?(\d+))?')
MACRO = re.compile(r'\\(?:(?:re|provide)?newcommand\*?\s*\{?\s*\\([A-Za-z@]+)'
                   r'|def\s*\\([A-Za-z@]+)|DeclareMathOperator\*?\s*\{\s*\\([A-Za-z]+)\s*\})')
THEOREM = re.compile(r'\\(?:newtheorem|declaretheorem)\*?\s*(?:\[[^\]]*\]\s*)?\{([A-Za-z*]+)\}')
FILE_ERROR = re.compile(r'^(?:\./)?([^\s:/]+\.(?:tex|sty|cls)):(\d+): (.+)$', re.M)
DEFAULT_THEOREMS = ('theorem', 'lemma', 'proposition', 'corollary', 'definition', 'remark')

OUTLINE_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['title', 'sections'],
    'properties': {
        'title': {'type': 'string', 'maxLength': 300},
        'sections': {'type': 'array', 'maxItems': 12, 'items': {
            'type': 'object', 'additionalProperties': False, 'required': ['title', 'sources', 'goal'],
            'properties': {'title': {'type': 'string', 'maxLength': 200},
                           'sources': {'type': 'array', 'maxItems': 6, 'items': {'type': 'string', 'maxLength': 40}},
                           'goal': {'type': 'string', 'maxLength': 800}}}},
    },
}

REVIEW_POLICY = """Review the LaTeX write-up against the source lines shown. Check that
statements, hypotheses and constants match the notes, that nothing was added or
silently repaired, that unclear notes are marked with TODO comments, and that the
template's macros are used. Report concrete problems with source locators such as
[M1:L4-L9] and section names, as at most ten short Markdown bullet points. This is
a model review, not certification."""


def untraced_paragraphs(latex):
    """Paragraphs with text but no `% src:` comment: their origin cannot be checked."""
    count = 0
    for block in re.split(r'\n\s*\n', latex):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        body = [line for line in lines if not line.lstrip().startswith('%')
                and not re.match(r'\\(?:sub)*section\*?\{', line.strip())]
        if body and not any(re.match(r'\s*%\s*src:', line) for line in lines):
            count += 1
    return count


def template_vocabulary(sources):
    """Macros (with their defining lines) and theorem environments of the template."""
    macros, theorems, definitions = [], [], []
    for source in sources:
        for match in MACRO.finditer(source['content']):
            name = next(group for group in match.groups() if group)
            if '@' not in name and name not in macros:
                macros.append(name)
                start = source['content'].rfind('\n', 0, match.start()) + 1
                end = source['content'].find('\n', match.end())
                definitions.append(source['content'][start:end if end != -1 else None].strip()[:200])
        for name in THEOREM.findall(source['content']):
            if name not in theorems:
                theorems.append(name)
    return macros[:80], theorems[:30], definitions[:80]


def compile_latex(tex, files=(), timeout=60):
    """Compile with latexmk in a scratch folder: no shell escape, paranoid file access."""
    exe = shutil.which('latexmk')
    if exe is None:
        return {'available': False, 'ok': False, 'errors': [], 'log': '', 'pdf': None}
    with tempfile.TemporaryDirectory(prefix='square-writeup-') as folder:
        for name, content in files:
            base = Path(name).name
            if base and not base.startswith('.') and base != 'document.tex':
                Path(folder, base).write_text(content, encoding='utf-8')
        Path(folder, 'document.tex').write_text(tex, encoding='utf-8')
        env = dict(os.environ, openin_any='p', openout_any='p', shell_escape='f', max_print_line='1000')
        command = [exe, '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
                   '-no-shell-escape', 'document.tex']
        timed_out = False
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(command, cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                process.wait(timeout=max(1, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        log_path, pdf_path = Path(folder, 'document.log'), Path(folder, 'document.pdf')
        log = log_path.read_text(encoding='utf-8', errors='replace')[-60000:] if log_path.exists() else ''
        pdf = (pdf_path.read_bytes() if process.returncode == 0 and not timed_out and pdf_path.exists()
               and pdf_path.stat().st_size < 20_000_000 else None)
    errors = [{'file': name, 'line': int(line), 'message': message.strip()[:300]}
              for name, line, message in FILE_ERROR.findall(log)]
    if not errors:
        errors = [{'file': None, 'line': None, 'message': line[2:].strip()[:300]}
                  for line in log.splitlines() if line.startswith('! ')]
    if timed_out:
        errors.insert(0, {'file': None, 'line': None, 'message': f'Compilation stopped after {timeout} seconds'})
    if not errors and pdf is None:
        errors = [{'file': None, 'line': None, 'message': 'LaTeX produced no PDF; see the log'}]
    return {'available': True, 'ok': pdf is not None, 'errors': errors[:20], 'log': log, 'pdf': pdf}


def _clean_section(text, title):
    text = re.sub(r'^\s*```[a-zA-Z]*\s*\n|\n```\s*$', '', text.strip())
    if '\\begin{document}' in text:
        text = text.split('\\begin{document}', 1)[1]
    text = text.replace('\\end{document}', '').replace('\\maketitle', '').strip()
    if not re.match(r'\\section\*?\{', text):
        text = f'\\section{{{title}}}\n' + text
    return text


class WriteupRunner(ResearchRunner):
    KINDS = ('writeup',)
    TERMINAL = ('complete', 'partial', 'budget_exhausted', 'budget_violation')

    def start(self, goal, *, template_files=(), template_texts=(), notes='', output='', source_files=(), **budgets):
        if not isinstance(notes, str) or len(notes) > 200_000:
            raise ValueError('Typed notes are limited to 200000 characters; pin a file instead')
        texts = [(str(name), content) for name, content in template_texts]
        for name in [*map(str, template_files), *(name for name, _ in texts)]:
            if Path(name).suffix.lower() not in TEMPLATE_SUFFIXES:
                raise ValueError('Template files must be .sty, .cls, .tex or .bib')
        if any(not isinstance(content, str) or len(content) > 500_000 for _, content in texts):
            raise ValueError('Each template file must be text of at most 500 KB')
        if output:
            path = self.agent.workspace.path(output)
            if path.suffix.lower() != '.tex':
                raise ValueError('The output must be a workspace-relative .tex file')
        found = re.findall(r'(?<![\w/])[\w./-]+\.(?:tex|md|txt|pdf)\b', goal)
        if not source_files and not notes.strip() and not found:
            raise ValueError('Give the write-up something to work from: pin notes, a draft or a PDF, or paste rough notes')
        self._templates, self._template_texts = list(template_files), texts
        self._notes, self._output = notes, output
        return super().start(goal, kind='writeup', source_files=source_files, **budgets)

    def _extra_sources(self):
        sources = []
        if self._notes.strip():
            sources.append({'id': 'M0', 'path': 'typed-notes.md', 'content': self._notes,
                            'sha256': hashlib.sha256(self._notes.encode()).hexdigest()})
        pinned = self._pin(self._templates, 'T')
        # Saved templates arrive as contents: they live outside the workspace.
        for index, (name, content) in enumerate(self._template_texts, len(pinned) + 1):
            pinned.append({'id': f'T{index}', 'path': 'saved-template/' + Path(name).name, 'content': content,
                           'sha256': hashlib.sha256(content.encode()).hexdigest()})
        return sources + pinned

    def _extra_state(self):
        return {'output_name': self._output, 'outline': None, 'sections': [], 'macros': [], 'theorems': [], 'definitions': [],
                'document': '', 'section_lines': [], 'compile': None, 'repaired': False, 'outputs': {}}

    # -- helpers -----------------------------------------------------------------

    def _templates_in(self, suffixes=TEMPLATE_SUFFIXES):
        return [s for s in self.state['sources'] if s['id'].startswith('T') and Path(s['path']).suffix.lower() in suffixes]

    def _notes_in(self):
        return [s for s in self.state['sources'] if not s['id'].startswith('T')]

    def _room(self, cap):
        """Characters of source text that fit next to the fixed instructions."""
        return max(1500, (self.agent.ctx - cap - 256) * 3 - 9000)

    def _read_range(self, source_id, start, end):
        """Read through the recorded manuscript tool, so provenance checks see these lines."""
        lines, cursor = [], start
        while cursor <= end:
            value = json.loads(self._read_manuscript(source_id, cursor, end))
            if not value['lines']:
                break
            lines += value['lines']
            cursor = value['end_line'] + 1
        return lines

    def _excerpt(self, refs, room):
        parts, used, clipped = [], 0, False
        for ref in refs:
            match = REF.fullmatch(ref.strip('[] '))
            if not match or not any(s['id'] == match.group(1) for s in self.state['sources']):
                continue
            source_id, a, b = match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))
            b = min(b, a + 199)
            text = '\n'.join(f'{source_id} L{item["line"]}: {item["text"]}' for item in self._read_range(source_id, a, b))
            if used + len(text) > room:
                text, clipped = text[:max(0, room - used)], True
            parts.append(text)
            used += len(text)
            if clipped:
                break
        return '\n'.join(parts), clipped

    def _vocabulary_note(self):
        theorems, definitions = self.state['theorems'], self.state.get('definitions', [])
        shown = '\n'.join(definitions)[:2500]
        note = ('Template definitions; use these commands instead of writing their expansions:\n' + shown + '\n') if shown else 'No template macros were given.\n'
        note += ('Theorem environments: ' + ', '.join(theorems) + '.') if theorems else (
            'Theorem environments available: ' + ', '.join(DEFAULT_THEOREMS) + ', proof.')
        return note

    # -- workflow --------------------------------------------------------------------

    def _step(self, phase):
        if phase == 'plan':
            self._plan()
        elif phase == 'section':
            self._section()
        elif phase == 'assemble':
            self._assemble()
            self.state['phase'] = 'compile'
        elif phase == 'compile':
            self._compile()
        elif phase == 'repair':
            self._repair()
        elif phase == 'review':
            self._review()
        elif phase == 'export':
            self._export()
            self.state['phase'] = 'done'
        else:
            raise ValueError('Unknown write-up phase: ' + phase)

    def _plan(self):
        templates = self._templates_in()
        for source in templates:  # read in full: the model must follow them
            self._read_range(source['id'], 1, max(1, len(source['content'].splitlines())))
        self.state['macros'], self.state['theorems'], self.state['definitions'] = template_vocabulary(templates)
        cap = 1500
        room = self._room(cap)
        shown, used = [], 0
        for source in self._notes_in():
            total = len(source['content'].splitlines())
            end = total
            text = '\n'.join(f'{source["id"]} L{i + 1}: {line}' for i, line in enumerate(source['content'].splitlines()))
            if used + len(text) > room:
                end = max(1, sum(1 for _ in text[:max(0, room - used)].splitlines()))
            if end >= 1 and used < room:
                lines = self._read_range(source['id'], 1, end)
                shown.append('\n'.join(f'{source["id"]} L{item["line"]}: {item["text"]}' for item in lines))
                used += len(shown[-1])
        instruction = ('Plan the LaTeX write-up. Return JSON with a document title and the sections in reading order. '
                       'Each section has a short title, the source line ranges it draws on (such as "M1:L1-L40", only '
                       'from the lines shown), and a one-sentence goal. Cover all the notes shown and add no topics.\n\n'
                       'USER REQUEST:\n' + self.state['goal'] + '\n\n' + self._vocabulary_note() +
                       '\n\nNOTES (numbered lines):\n' + '\n'.join(shown))
        result = self._call('plan', instruction, cap=cap, format_schema=OUTLINE_SCHEMA)
        self.state['outline'] = self._outline(result['text'])
        self.state['plan'] = json.dumps(self.state['outline'], ensure_ascii=False, indent=2)
        self.state['sections'] = [{**section, 'latex': '', 'complete': False, 'truncated': False}
                                  for section in self.state['outline']['sections']]
        self.state['phase'] = 'section'

    def _outline(self, text):
        try:
            value = json.loads(text)
        except ValueError:
            value = None
        sections = []
        if isinstance(value, dict) and isinstance(value.get('sections'), list):
            for item in value['sections'][:12]:
                if isinstance(item, dict):
                    refs = [r.strip('[] ') for r in item.get('sources') or [] if isinstance(r, str) and REF.fullmatch(r.strip('[] '))]
                    sections.append({'title': str(item.get('title') or 'Section').strip()[:200] or 'Section',
                                     'sources': refs[:6], 'goal': str(item.get('goal') or '')[:800]})
        title = str(value.get('title') or '').strip()[:300] if isinstance(value, dict) else ''
        if not sections:
            self.state['warnings'].append('The planner gave no usable outline; the notes were split into consecutive parts.')
        # Every note line must belong to some section, whatever the planner chose.
        for source in self._notes_in():
            total = len(source['content'].splitlines())
            covered = set()
            for section in sections:
                for ref in section['sources']:
                    match = REF.fullmatch(ref)
                    if match and match.group(1) == source['id']:
                        covered.update(range(int(match.group(2)), int(match.group(3) or match.group(2)) + 1))
            missing = [n for n in range(1, total + 1) if n not in covered and source['content'].splitlines()[n - 1].strip()]
            if missing and sections:
                self.state['warnings'].append(f'The outline skipped {len(missing)} lines of {source["path"]}; they were added as further sections.')
            groups = []  # consecutive uncovered lines, at most 80 per section
            for n in missing:
                if groups and n == groups[-1][1] + 1 and n - groups[-1][0] < 80:
                    groups[-1][1] = n
                else:
                    groups.append([n, n])
            for a, b in groups:
                sections.append({'title': 'Further notes' if sections else 'Notes',
                                 'sources': [f'{source["id"]}:L{a}-L{b}'], 'goal': 'Write up these notes faithfully.'})
        return {'title': title or 'Notes', 'sections': sections[:24]}

    def _section(self):
        sections = self.state['sections']
        index = next(i for i, section in enumerate(sections) if not section['complete'])
        section = sections[index]
        cap = min(4096, self.agent.predict)
        excerpt, clipped = self._excerpt(section['sources'], self._room(cap) // 2)
        if clipped:
            self.state['warnings'].append(f'Section "{section["title"]}" saw only part of its source lines; split the notes or raise the context.')
        before = ', '.join(s['title'] for s in sections[:index]) or 'none'
        instruction = (f'Write section {index + 1} of {len(sections)}, titled "{section["title"]}". Goal: {section["goal"]}\n'
                       f'Sections before it: {before}. Output only this section\'s LaTeX, starting with \\section. '
                       'Put a comment such as % src: [M1:L12-L30] before each paragraph or environment, using only '
                       'the line numbers below. Mark unclear notes with % TODO comments; add no new mathematics.\n'
                       + self._vocabulary_note() + '\n\nSOURCE LINES:\n' + (excerpt or '(no readable lines)'))
        result = self._call('section', instruction, cap=cap)
        section['latex'] = _clean_section(result['text'], section['title'])
        section['truncated'] = not result['complete']
        section['complete'] = True
        if section['truncated']:
            self.state['warnings'].append(f'Section "{section["title"]}" hit the output limit and may be cut short.')
        if all(s['complete'] for s in sections):
            self.state['phase'] = 'assemble'

    def _assemble(self):
        template = next((s for s in self._templates_in(('.tex',)) if '\\begin{document}' in s['content']), None)
        title = self.state['outline']['title'] if self.state['outline'] else 'Notes'
        if template:
            preamble = template['content'].split('\\begin{document}', 1)[0].rstrip()
            if '\\title' not in preamble:
                preamble += f'\n\\title{{{title}}}'
            head = '\\begin{document}\n' + ('\\maketitle\n' if '\\maketitle' in template['content'] else '')
        else:
            classes = self._templates_in(('.cls',))
            styles = self._templates_in(('.sty',))
            lines = [f'\\documentclass{{{Path(classes[0]["path"]).stem if classes else "amsart"}}}',
                     '\\usepackage{amsmath,amssymb,amsthm}']
            lines += [f'\\usepackage{{{Path(s["path"]).stem}}}' for s in styles]
            if not self.state['theorems']:
                lines += ['\\newtheorem{theorem}{Theorem}[section]'] + [
                    f'\\newtheorem{{{name}}}[theorem]{{{name.capitalize()}}}' for name in ('lemma', 'proposition', 'corollary')]
                lines += ['\\theoremstyle{definition}', '\\newtheorem{definition}[theorem]{Definition}',
                          '\\theoremstyle{remark}', '\\newtheorem{remark}[theorem]{Remark}']
            lines.append(f'\\title{{{title}}}')
            preamble, head = '\n'.join(lines), '\\begin{document}\n\\maketitle\n'
        document = preamble + '\n\n' + head
        spans = []
        for section in self.state['sections']:
            start = document.count('\n') + 1
            document += '\n' + section['latex'] + '\n'
            spans.append([start, document.count('\n')])
        document += '\n\\end{document}\n'
        self.state['document'] = self.state['draft'] = document
        self.state['section_lines'] = spans
        # Only the provenance comments are citations; optional arguments like [Sobolev] are not.
        comments = '\n'.join(re.findall(r'%\s*src:\s*([^\n]*)', document))
        issues = self._citation_issues(comments) if comments else [
            'The write-up has no source comments; its content cannot be traced to the notes.']
        untraced = sum(untraced_paragraphs(section['latex']) for section in self.state['sections'])
        if untraced and comments:
            issues.append(f'{untraced} paragraph{"s have" if untraced > 1 else " has"} no source comment and cannot be traced to the notes.')
        self.state['citation_issues'] = issues

    def _compile(self):
        files = [(s['path'], s['content']) for s in self._templates_in()
                 if not (s['path'].lower().endswith('.tex') and '\\begin{document}' in s['content'])]
        remaining = self.state['settings']['max_seconds'] - self._elapsed()
        self.emit('notice', 'Compiling the document with latexmk')
        result = compile_latex(self.state['document'], files, timeout=int(min(60, max(1, remaining))))
        log_name = self._artifact('compile-log', result['log'] or 'No log.') if result['available'] else None
        if result['pdf']:
            _atomic_write(self.directory / 'artifacts' / 'compiled.pdf', result['pdf'])
        failing = sorted({i for error in result['errors'] if error['file'] == 'document.tex'
                          for i, (a, b) in enumerate(self.state['section_lines']) if a <= error['line'] <= b})
        self.state['compile'] = {'available': result['available'], 'ok': result['ok'], 'errors': result['errors'],
                                 'log': log_name, 'failing_sections': failing, 'pdf': bool(result['pdf'])}
        if not result['available']:
            self.state['warnings'].append('LaTeX (latexmk) is not installed here; the document was not compiled.')
            self.state['phase'] = 'review'
        elif result['ok'] or self.state['repaired'] or not failing:
            self.state['phase'] = 'review'
        else:
            self.state['phase'] = 'repair'

    def _repair(self):
        self.state['repaired'] = True
        compile_state = self.state['compile']
        for index in compile_state['failing_sections']:
            section = self.state['sections'][index]
            a, _ = self.state['section_lines'][index]
            errors = [f'line {e["line"] - a + 1} of this section: {e["message"]}' for e in compile_state['errors']
                      if e['file'] == 'document.tex' and self.state['section_lines'][index][0] <= e['line'] <= self.state['section_lines'][index][1]]
            instruction = ('Fix the LaTeX compile errors in this section. Change only what the errors require, keep the '
                           '% src comments, and output the whole corrected section.\n\n' + self._vocabulary_note() +
                           '\n\nERRORS:\n' + '\n'.join(errors) + '\n\nSECTION:\n' + section['latex'])
            result = self._call('repair', instruction, cap=min(4096, self.agent.predict))
            if result['text'].strip() and result['complete']:
                section['latex'] = _clean_section(result['text'], section['title'])
        self._assemble()
        self.state['phase'] = 'compile'

    def _review(self):
        settings = self.state['settings']
        if (settings['max_tokens'] - self.state['tokens_charged'] < 1200
                or settings['max_input_tokens'] - self.state['input_tokens_charged'] < 6000):
            self.state['warnings'].append('Not enough budget remained for an independent review.')
            self.state['phase'] = 'export'
            return
        refs = [ref for section in self.state['sections'] for ref in section['sources']]
        excerpt, _ = self._excerpt(refs, self._room(2000) // 3)
        compile_state = self.state['compile'] or {}
        status = ('compiled without errors' if compile_state.get('ok') else 'not compiled' if not compile_state.get('available')
                  else 'failed to compile: ' + '; '.join(e['message'] for e in compile_state.get('errors', [])[:5]))
        document = self.state['document']
        room = self._room(1500)
        if len(document) + len(excerpt) > room:
            document = document[: max(1000, room - len(excerpt))] + '\n% [document excerpt ends here]'
        instruction = ('WRITE-UP TO REVIEW:\n' + document + '\n\nSOURCE LINES:\n' + excerpt +
                       '\n\nCompilation: ' + status + '\nCitation checks: ' + json.dumps(self.state.get('citation_issues', [])))
        # Not the research 'review' role: that one checks literature evidence in context.
        # The request comes last, where small models follow it instead of copying the document.
        result = self._call('fidelity', instruction, cap=1500,
                            trailer=REVIEW_POLICY + '\nWrite the review now. Do not reproduce the document.')
        text = result['text'].strip()
        copied = text.startswith('\\documentclass') or text.count('\\section') > 2
        self.state['review'] = '' if copied else result['text']
        self.state['review_complete'] = result['complete'] and bool(text) and not copied
        if copied:
            self.state['warnings'].append('The reviewer returned LaTeX instead of a review; the write-up is unreviewed.')
        elif not result['complete']:
            self.state['warnings'].append('The review hit its output limit and may be incomplete.')
        self.state['phase'] = 'export'

    def _free_path(self, name):
        path = self.agent.workspace.path(name, pdf=True)
        stem, number = path.stem, 2
        while path.exists():
            path = path.with_name(f'{stem}-{number}{path.suffix}')
            number += 1
        return path

    def _export(self):
        workspace = self.agent.workspace
        name = self.state.get('output_name')
        if not name:
            notes = [s for s in self._notes_in() if s['id'] != 'M0']
            name = (Path(notes[0]['path']).stem + '-writeup.tex') if notes else 'writeup.tex'
        # The user named this new file when starting the job; nothing existing is replaced.
        tex = self._free_path(name)
        tex.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(tex, self.state['document'].encode('utf-8'))
        outputs = {'tex': str(tex.relative_to(workspace.root))}
        compiled = self.directory / 'artifacts' / 'compiled.pdf'
        if compiled.exists():
            pdf = self._free_path(str(Path(outputs['tex']).with_suffix('.pdf')))
            _atomic_write(pdf, compiled.read_bytes())
            outputs['pdf'] = str(pdf.relative_to(workspace.root))
        self.state['outputs'] = outputs
        self.emit('notice', 'Wrote ' + outputs['tex'] + (' and ' + outputs['pdf'] if 'pdf' in outputs else ''))

    def _finish(self):
        compile_state = self.state.get('compile') or {}
        truncated = any(section.get('truncated') for section in self.state['sections'])
        complete = (compile_state.get('ok') and not self.state.get('citation_issues') and not truncated
                    and self.state.get('review_complete'))
        self.state['status'] = 'complete' if complete else 'partial'
        self.state['stop_reason'] = 'Write-up finished; check the mathematics before using it.'

    def _report(self):
        s = self.state
        compile_state = s.get('compile') or {}
        todo = len(re.findall(r'%\s*TODO', s.get('document', '')))
        outputs = s.get('outputs') or {}
        compiled = ('compiled without errors' if compile_state.get('ok') else
                    'not compiled: LaTeX is not installed' if compile_state and not compile_state.get('available') else
                    f'failed to compile ({len(compile_state.get("errors", []))} errors)' if compile_state else 'not compiled yet')
        out = ['# Write-up', '', '**Model draft: check the mathematics and the compiled document before using it.**', '',
               f'Status: `{s["status"]}`. Phase: `{s["phase"]}`. Job ID: `{s["id"]}`.', '',
               s.get('stop_reason', 'The write-up is in progress.'), '', '## Output', '',
               f'- Document: `{outputs["tex"]}`' if outputs.get('tex') else '- Document: not exported yet',
               f'- PDF: `{outputs["pdf"]}`' if outputs.get('pdf') else '- PDF: none',
               f'- LaTeX: {compiled}', f'- TODO comments left for the author: {todo}', '',
               '## Request', '', s['goal']]
        warnings = list(dict.fromkeys(s.get('warnings', []) + s.get('citation_issues', [])))
        if warnings:
            out += ['', '## Controller checks', ''] + ['- ' + w for w in warnings]
        if compile_state.get('errors'):
            out += ['', '## Compile errors', ''] + [f'- {e.get("file") or ""}{":" + str(e["line"]) if e.get("line") else ""} {e["message"]}'
                                                   for e in compile_state['errors']]
        if s.get('review'):
            out += ['', '## Independent model review', '', s['review']]
        out += ['', '## Sources', '']
        for source in s['sources']:
            kind = 'template' if source['id'].startswith('T') else 'notes'
            out.append(f'- {source["id"]} ({kind}): `{source["path"]}`; read ranges: {s["manuscript_ranges"].get(source["id"], [])}.')
        if s.get('document'):
            out += ['', '## Document', '', '```latex', s['document'], '```']
        out += ['', f'Generated tokens charged: {s["tokens_charged"]}/{s["settings"]["max_tokens"]}; input tokens charged: '
                    f'{s["input_tokens_charged"]}/{s["settings"]["max_input_tokens"]}; elapsed seconds: '
                    f'{s["seconds_used"]:.1f}/{s["settings"]["max_seconds"]}.', '']
        _atomic_write(self.directory / 'report.md', '\n'.join(out).encode('utf-8'))
