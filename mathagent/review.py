"""Three review workflows on one page: a quick proof check, a journal referee report and an explanation.

The controller does the systematic work. It maps the manuscript (manuscript.py), picks
the proofs to check, gives each model call one proof with exactly what it needs, runs
independent passes through different lenses, has each alleged error re-checked, checks
that every quote is really at the lines it cites, and assembles the report itself. The
model gets small structured tasks (JSON). Calls that do not depend on each other run
concurrently when the backend allows it (`concurrency`); the result is the same either way.
A review is a model draft that needs human judgment, never a certificate.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
import re
import threading
import unicodedata
import zlib

from . import manuscript
from . import review_policy as P
from .agent import AgentError
from .backends import TokenBudgetError
from .ledger import _atomic_write, _bytes, _read
from .research import ResearchBudget, ResearchRunner, _estimate
from .router import repair_latex
from .worker_errors import WorkerInputError

# 'review' reads the paper (overview, typos); 'journal' is the detailed review that also checks the proofs.
VARIANTS = ('quick', 'review', 'journal', 'explain')
PIPELINES = {'quick': 'proof-check-v1', 'review': 'review-v1', 'journal': 'journal-v1', 'explain': 'explain-v1'}
SKILLS = {'quick': 'review-quick', 'review': 'review', 'journal': 'referee', 'explain': 'explain'}
TITLES = {'quick': 'Proof check', 'review': 'Referee report', 'journal': 'Detailed referee report', 'explain': 'Explanation'}
MANUSCRIPT_VARIANTS = ('review', 'journal')  # they need a pinned manuscript
LEVELS = ('low', 'medium', 'high', 'xhigh', 'poincare')
# Per effort: verifier passes (quick), how far the journal review goes, and the optional extras.
EFFORT = {
    'low': dict(passes=1, confirm=False, depth=0, appendix=False, main_passes=1, other_passes=1,
                cited=0, queries=0, presentation=0, explain_check=False, fills=1),
    'medium': dict(passes=2, confirm=True, depth=1, appendix=False, main_passes=1, other_passes=1,
                   cited=1, queries=2, presentation=0, explain_check=True, fills=1),
    'high': dict(passes=3, confirm=True, depth=None, appendix=False, main_passes=1, other_passes=1,
                 cited=2, queries=3, presentation=1, explain_check=True, fills=2),
    'xhigh': dict(passes=4, confirm=True, depth=None, appendix=True, main_passes=2, other_passes=1,
                  cited=3, queries=4, presentation=3, explain_check=True, fills=2),
    'poincare': dict(passes=5, confirm=True, depth=None, appendix=True, main_passes=2, other_passes=2,
                     cited=4, queries=6, presentation=5, explain_check=True, fills=2),
}
FINE_LINES = 60  # proofs up to this many lines are explained line by line
KIND_LABEL = {'invalid_inference': 'Invalid inference', 'missing_justification': 'Missing justification',
              'uncertainty': 'Uncertain step'}
RANK = {'invalid_inference': 2, 'missing_justification': 1, 'uncertainty': 0}


def level_for(rounds):
    """The effort level behind a number of tries (Low 1, Medium 3, High 5, Extra high 7, Poincaré 10)."""
    rounds = int(rounds)
    return 'low' if rounds <= 1 else 'medium' if rounds <= 3 else 'high' if rounds <= 5 else 'xhigh' if rounds <= 7 else 'poincare'


def runner_class(variant):
    return {'quick': QuickReviewRunner, 'review': SummaryReviewRunner, 'journal': JournalRunner, 'explain': ExplainRunner}[variant]


def variant_for_state(state):
    """quick, journal or explain for a saved review job; None for older referee jobs."""
    return next((variant for variant, pipeline in PIPELINES.items() if state.get('pipeline') == pipeline), None)


def runner_for_state(state):
    """The runner that resumes a saved referee job (older jobs keep the generic research loop)."""
    variant = variant_for_state(state)
    return runner_class(variant) if variant else ResearchRunner


_ESCAPE = re.compile(r'\\(?:(["\\/bfnrt])|(u[0-9a-fA-F]{4}))|\\')


def _repair(value):
    if isinstance(value, str):
        return repair_latex(value)
    if isinstance(value, list):
        return [_repair(v) for v in value]
    if isinstance(value, dict):
        return {k: _repair(v) for k, v in value.items()}
    return value


def loads(text):
    """JSON from a model that writes TeX in strings: a lone backslash (\\| or \\langle) is doubled,
    control characters are accepted, and TeX commands read as \\t, \\f, \\b, \\r are restored."""
    for attempt in (text, _ESCAPE.sub(lambda m: m.group(0) if m.group(1) or m.group(2) else '\\\\', text)):
        try:
            value = json.loads(attempt, strict=False)
        except ValueError:
            continue
        return _repair(value) if isinstance(value, dict) else None
    return None


_LOOP = re.compile(r'(.{3,40}?)\1{12,}', re.S)


def degenerate(text):
    """A decoding loop: one short piece repeated a dozen times in a row (e.g. \\tilde{\\tilde{\\tilde{…)."""
    return bool(_LOOP.search(text))


def _fold(text):
    text = unicodedata.normalize('NFKD', str(text)).lower()
    return re.sub(r'[^a-z0-9]', '', text)


def _words(text):
    return [w for w in re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKD', str(text)).lower()) if len(w) >= 3]


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ranges(numbers):
    """[3, 4, 5, 9] → [[3, 5], [9, 9]]."""
    out = []
    for n in sorted(set(numbers)):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def _item(prefix, block, indent):
    """A Markdown list item whose parts stay separate paragraphs (a quote must not swallow the next line)."""
    out = [prefix + block[0], '']
    for line in block[1:]:
        out += [indent + line, '']
    return out


_ARTEFACT = re.compile(r'(?i)extract|artifact|artefact|garbl|rendered|split across|line break|spacing|spaces|hyphen')


def _prose_comment(comment):
    """In a PDF extraction only comments on words survive: the quote is mostly words, the comment does not
    blame the extraction, and it is not about notation or formulas."""
    tokens = re.findall(r'\S+', comment['quote'])
    words = [t for t in tokens if re.fullmatch(r"[A-Za-z][a-z]+(?:[-'][a-z]+)*[,.;:)]?", t)]
    return (bool(tokens) and len(words) / len(tokens) >= .6 and comment.get('kind') not in ('notation', 'definition')
            and not _ARTEFACT.search(comment['comment'] + ' ' + comment.get('suggestion', '')))


def _span(ranges):
    return ', '.join(f'L{a}' if a == b else f'L{a}-L{b}' for a, b in ranges)


class ReviewBase(ResearchRunner):
    KINDS = ('referee',)
    TERMINAL = ('reviewed', 'partial', 'budget_exhausted', 'budget_violation')
    POLICY = P.POLICY
    VARIANT = None

    def __init__(self, agent, emit=lambda kind, value: None, concurrency=1):
        super().__init__(agent, emit)
        self.concurrency = max(1, min(4, int(concurrency)))
        self._mutex = threading.RLock()
        self._lines_cache = {}

    @property
    def emit(self):
        return self._emit

    @emit.setter
    def emit(self, value):
        def quiet(kind, text):
            if not (kind == 'notice' and str(text).startswith('Research phase: ')):
                value(kind, text)
        self._emit = quiet

    # -- start -----------------------------------------------------------------------

    def start(self, goal, *, kind='referee', source_files=(), target='', reference_context=None, **budgets):
        if kind != 'referee':
            raise ValueError('The review workflows are referee jobs')
        self._target = str(target or '').strip()[:200]
        if not source_files and self.VARIANT not in MANUSCRIPT_VARIANTS:
            # "Check Lemma 2 in notes.md": a named workspace file is the source, not the pasted text.
            named = re.findall(r'(?<![\w/])[\w./-]+\.(?:tex|md|txt|pdf)\b', goal)
            source_files = [n for n in dict.fromkeys(named) if (self.agent.workspace.root / n).is_file()][:1]
        self._pasted = not source_files and self.VARIANT not in MANUSCRIPT_VARIANTS
        return super().start(goal, kind='referee', source_files=source_files,
                             reference_context=reference_context, **budgets)

    def _skill_folder(self, kind):
        return SKILLS[self.VARIANT]

    def _pin(self, names, prefix):
        return [] if getattr(self, '_pasted', False) else super()._pin(names, prefix)

    def _extra_sources(self):
        if not getattr(self, '_pasted', False):
            return []
        return [{'id': 'P1', 'path': 'pasted-text.md', 'content': self._goal_text,
                 'sha256': hashlib.sha256(self._goal_text.encode()).hexdigest()}]

    def _extra_state(self):
        rounds = getattr(self, '_rounds', 3)
        return {'pipeline': self.PIPELINE, 'variant': self.VARIANT, 'target': getattr(self, '_target', ''),
                'level': level_for(rounds), 'map': None, 'unit': None, 'checks': {}, 'confirms': {},
                'verdict': '', 'concurrency': self.concurrency}

    # ResearchRunner.start pins sources before it knows the state; keep the raw goal for a pasted source.
    def _prepare(self, goal, budgets):
        self._goal_text = goal
        self._rounds = budgets.get('max_rounds', 6)

    # -- helpers ---------------------------------------------------------------------------

    @property
    def effort(self):
        return EFFORT[self.state.get('level') or 'medium']

    def _source(self, source_id):
        return next(s for s in self.state['sources'] if s['id'] == source_id)

    def _lines(self, source_id):
        if source_id not in self._lines_cache:
            self._lines_cache[source_id] = self._source(source_id)['content'].splitlines()
        return self._lines_cache[source_id]

    def _numbered(self, source_id, a, b, mark=True):
        lines = self._lines(source_id)
        a, b = max(1, a), min(len(lines), b)
        if b < a:
            return ''
        if mark:
            with self._mutex:
                self.state['manuscript_ranges'].setdefault(source_id, []).append([a, b])
        return '\n'.join(f'L{n}: {lines[n - 1].rstrip()}' for n in range(a, b + 1))

    def _cap(self, wanted):
        """The output ceiling a call really gets (the same rule as _ask)."""
        return min(wanted, self.agent.predict, max(128, self.agent.ctx // 2))

    def _room(self, cap, prefix=''):
        """Bytes of manuscript text a prompt may still carry for this call."""
        cap = self._cap(cap)
        overhead = len((self.POLICY + self.state['skill'] + prefix).encode()) + 2500
        return max(2000, (self.agent.ctx - cap - 700) * 3 - overhead)

    def _time_left(self, share):
        return self._elapsed() < self.state['settings']['max_seconds'] * share

    def _tokens_left(self, reserve=0):
        s = self.state
        return s['settings']['max_tokens'] - s['tokens_charged'] - reserve

    def _input_left(self, reserve=0):
        s = self.state
        return s['settings']['max_input_tokens'] - s['input_tokens_charged'] - reserve

    def _affordable(self, cap, input_tokens, reserve_out=0, reserve_in=0, share=.8):
        return (self._time_left(share) and self._tokens_left(reserve_out) >= min(cap, self.agent.predict)
                and self._input_left(reserve_in) >= input_tokens)

    @contextmanager
    def _timeout(self):
        client = self.agent.client
        old = getattr(client, 'timeout', None)
        if old is not None:
            client.timeout = max(.1, min(old, self.state['settings']['max_seconds'] - self._elapsed()))
        try:
            yield
        finally:
            if old is not None:
                client.timeout = old

    def _thinks(self, cap):
        """Checks reason before they answer (as Prove's verifier does) when the output ceiling leaves room."""
        return bool(getattr(self.agent, 'think', False)) and self._cap(cap) >= 4096

    def _judge(self, role, prompt, *, cap, schema, key, direct_cap=3000):
        """A call that reasons first when there is room, then once more without thinking (warmer) if unusable."""
        if self._thinks(cap):
            try:
                result = self._ask(role, prompt, cap=cap, schema=schema, key=key, think=True)
            except AgentError as exc:
                if isinstance(exc, ResearchBudget):
                    raise
                # Some servers fail on thinking plus structured output for a given input (vLLM answered 500).
                self._warn(f'The model server failed on a {role} call with thinking ({str(exc)[:120]}); it was asked again without thinking.')
                result = {'value': None}
            if result['value'] is not None:
                return result
            if not any('it was asked again without thinking' in w and role in w for w in self.state['warnings']):
                self._warn(f'A {role} answer was not usable after thinking; it was asked again without thinking.')
            return self._ask(role, prompt, cap=min(cap, direct_cap), schema=schema, key=key + ':direct', temperature=0.6)
        result = self._ask(role, prompt, cap=min(cap, direct_cap), schema=schema, key=key)
        if result['value'] is None and result['complete'] and self._affordable(min(cap, direct_cap), 2000):
            self._warn(f'A {role} answer was not usable; it was asked again.')
            result = self._ask(role, prompt, cap=min(cap, direct_cap), schema=schema, key=key + ':again', temperature=0.6)
        return result

    def _ask(self, role, prompt, *, cap, schema=None, key='', temperature=0.2, think=False):
        """One model call with a fixed prompt; thread-safe, budgeted, streamed to an artifact."""
        with self._mutex:
            self._check_time()
            settings = self.state['settings']
            cap = min(self._cap(cap), self._tokens_left())
            if cap < 128:
                raise ResearchBudget('Generated-token budget exhausted')
            payload = {'model': self.agent.model, 'stream': True, 'think': think,
                       'options': {'num_ctx': self.agent.ctx, 'num_predict': cap, 'temperature': temperature}}
            if self.agent.seed is not None:
                payload['options']['seed'] = (self.agent.seed + zlib.crc32(key.encode())) % (2 ** 31)
            if schema:
                payload['format'] = schema
            payload['messages'] = [{'role': 'system', 'content': self.POLICY + '\n' + self.state['skill']},
                                   {'role': 'user', 'content': prompt}]
            estimated = _estimate(payload)
            if estimated > self.agent.ctx - cap - 256:
                raise ResearchBudget(f'A {role} prompt does not fit the model context ({estimated} input tokens, context {self.agent.ctx}); nothing was clipped')
            if estimated > self._input_left():
                raise ResearchBudget('Input-token budget exhausted')
            request = self._artifact(role + '-request', json.dumps(payload, ensure_ascii=False, indent=2))
            stream_name = self._artifact(role + '-stream', '')
            call = {'role': role, 'status': 'running', 'request': request, 'stream': stream_name,
                    'reserved_tokens': cap, 'reserved_input_tokens': estimated, 'key': key}
            self.state['calls'].append(call)
            self.state['tokens_charged'] += cap
            self.state['input_tokens_charged'] += estimated
            self._save()
        text, done, stats, stream = '', False, {}, None
        try:
            fd = os.open(self.directory / 'artifacts' / stream_name, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                stream = self.agent.client.stream(payload)
                for event in stream:
                    output.write(json.dumps(event, ensure_ascii=False) + '\n')
                    output.flush()
                    text += event.get('message', {}).get('content', '')
                    if event.get('done'):
                        done = True
                        stats = {k: event[k] for k in ('eval_count', 'prompt_eval_count', 'done_reason') if k in event}
                    self._check_time()
            if not done:
                raise AgentError('Model stream ended without completion; partial output saved')
            if type(stats.get('eval_count')) is int and stats['eval_count'] > cap:
                raise TokenBudgetError(stats, cap)
            with self._mutex:
                for field, counter, reservation in [('eval_count', 'tokens_charged', cap), ('prompt_eval_count', 'input_tokens_charged', estimated)]:
                    count = stats.get(field)
                    if type(count) is int and count >= 0:
                        self.state[counter] += count - reservation
                call['stats'] = stats
                complete = stats.get('done_reason') != 'length'
                call['status'] = 'complete' if complete else 'truncated'
                self._save()
        except BaseException as exc:
            with self._mutex:
                call['status'] = 'interrupted'
                if isinstance(exc, TokenBudgetError):
                    call['stats'] = exc.stats
                    self.state['tokens_charged'] += exc.stats['eval_count'] - cap
                    call['status'] = 'budget_violation'
                self._save()
            raise
        finally:
            if stream is not None and hasattr(stream, 'close'):
                stream.close()
        value = loads(text) if schema and complete else None
        if value is not None and degenerate(text):
            self._warn(f'A {role} answer fell into a repetition loop and was discarded.')
            value = None
        return {'text': text, 'complete': complete, 'value': value}

    def _wave(self, jobs):
        """Run independent jobs [(key, fn)]; each fn stores its own result. Re-raise the first failure."""
        if not jobs:
            return
        errors = []
        with self._timeout():
            if self.concurrency <= 1 or len(jobs) == 1:
                for _, fn in jobs:
                    fn()
                self._report()
                return
            with ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix='mathagent-review') as pool:
                futures = [pool.submit(fn) for _, fn in jobs]
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append(exc)
        if errors:
            interrupts = [e for e in errors if isinstance(e, KeyboardInterrupt)]
            raise interrupts[0] if interrupts else errors[0]
        self._report()  # the page shows the findings so far while a long review continues

    def _warn(self, text):
        with self._mutex:
            if text not in self.state['warnings']:
                self.state['warnings'].append(text)

    # -- the unit to review and its packet ---------------------------------------------------

    def _map(self):
        source = self.state['sources'][0]
        mapping = manuscript.build(source['content'], source['path'])
        mapping.pop('labels', None)
        self.state['map'] = mapping
        for warning in mapping['warnings'][:6]:
            self._warn('Manuscript map: ' + warning)
        self.emit('notice', f'Mapped {source["path"]}: {len(mapping["units"])} statements, '
                            f'{len(manuscript.checkable(mapping))} with proofs')

    def _pick_unit(self):
        """Quick and explain: the one result to work on, from the target, the request or the text itself."""
        mapping, source = self.state['map'], self.state['sources'][0]
        target = self.state.get('target') or ''
        unit = manuscript.resolve(mapping, target) if target else None
        if unit is None and not self._is_pasted():
            unit = manuscript.resolve(mapping, self.state['goal'])
        if unit is None:
            candidates = manuscript.checkable(mapping) if self.VARIANT == 'quick' else mapping['units']
            if len(candidates) == 1:
                unit = candidates[0]
            elif self._is_pasted() and candidates:
                unit = max(candidates, key=manuscript.proof_length)
        if unit is None and not self._is_pasted() and mapping['units']:
            unit = self._ask_unit(mapping)
        if unit is not None:
            chosen = {k: unit[k] for k in ('id', 'name', 'statement', 'proofs', 'refs', 'statement_refs', 'equations', 'title')}
        elif self._is_pasted():
            chosen = self._split_pasted()
        else:
            names = ', '.join(u['name'] for u in mapping['units'][:12])
            raise WorkerInputError('Say which result to ' + ('check' if self.VARIANT == 'quick' else 'explain') +
                                   f', for example "Lemma 3.2"; {source["path"]} contains: {names or "no recognised statements"}.',
                                   code='target_not_found', details={'target': target})
        chosen['source'] = source['id']
        if self.VARIANT == 'quick' and not chosen['proofs']:
            raise WorkerInputError('No proof was found to check. Paste the statement and its proof, or use Prove to look for a proof.',
                                   code='no_proof')
        self.state['unit'] = chosen
        self.emit('notice', ('Checking ' if self.VARIANT == 'quick' else 'Explaining ') + chosen['name'])

    def _is_pasted(self):
        return self.state['sources'][0]['id'] == 'P1'

    def _ask_unit(self, mapping):
        schema = {'type': 'object', 'additionalProperties': False, 'required': ['unit'],
                  'properties': {'unit': {'type': 'string', 'maxLength': 8}}}
        prompt = ('Which statement of the manuscript does this request concern? Answer with its id (such as U3), '
                  'or an empty string if none.\n\nREQUEST:\n' + self.state['goal'] + '\n\nSTATEMENTS:\n' + manuscript.outline(mapping))
        result = self._ask('pick', prompt, cap=200, schema=schema, key='pick')
        uid = str((result['value'] or {}).get('unit') or '').strip()
        return manuscript.unit(mapping, uid)

    def _split_pasted(self):
        lines = self._lines('P1')
        proof_line = next((n for n, line in enumerate(lines, 1) if re.match(r'^\s*[*_#]*\s*(?:proof|preuve|démonstration)\b', line, re.I)), None)
        nonblank = [n for n, line in enumerate(lines, 1) if line.strip()]
        if not nonblank:
            raise WorkerInputError('Paste the statement and its proof.', code='invalid_worker_input')
        if proof_line:
            before = [n for n in nonblank if n < proof_line]
            statement = [before[0], before[-1]] if before else None
            proofs = [[proof_line, nonblank[-1]]]
        else:
            result = self._ask('extract', P.EXTRACT_POLICY + '\n\nTEXT:\n' + self._numbered('P1', 1, len(lines), mark=False),
                               cap=200, schema=P.EXTRACT_SCHEMA, key='extract')
            value = result['value'] or {}
            s0, s1 = _int(value.get('statement_start')), _int(value.get('statement_end'))
            p0, p1 = _int(value.get('proof_start')), _int(value.get('proof_end'))
            statement = [s0, s1] if 1 <= s0 <= s1 <= len(lines) else None
            proofs = [[p0, p1]] if 1 <= p0 <= p1 <= len(lines) else []
            if statement is None and not proofs:
                statement = [nonblank[0], nonblank[-1]]
                self._warn('The statement and the proof could not be told apart; the whole text is treated as the statement.')
        return {'id': 'P', 'name': 'the pasted result', 'statement': statement, 'proofs': proofs, 'refs': [],
                'statement_refs': [], 'equations': [], 'title': ''}

    def _proof_lines(self, unit):
        """Proof lines with content: blank lines and PDF extraction fragments ('(k)', 'e k') do not count."""
        lines = self._lines(unit['source'])
        return [n for a, b in unit['proofs'] for n in range(a, b + 1) if len(re.findall(r'[^\W\d_]', lines[n - 1])) >= 8]

    def _packet(self, unit, room, *, notation=(), segment=None, context=True):
        """Prompt text for one unit: shared notation first (stable prefix), context, then statement and proof.

        Returns (text, fits, ranges): the ranges are marked as read only when the prompt is sent (_send).
        """
        sid = unit['source']
        mapping = self.state['map']
        head = f'SOURCE {sid}. Lines are numbered; cite them as given.\n'
        if self._source(sid)['path'].lower().endswith('.pdf'):
            head += ('This text was extracted from a PDF: formulas may be garbled (accents, indices and fractions on separate '
                     'lines, symbols split). Read them as the authors meant, write formulas in clean LaTeX of your own '
                     'between dollar signs, and never copy garbled sequences.\n')
        ranges, parts = [], []
        for a, b in notation:
            parts.append('SETTING AND NOTATION OF THE MANUSCRIPT (context):\n' + self._numbered(sid, a, b, mark=False))
            ranges.append([a, b])
        core = ''
        if unit.get('statement'):
            a, b = unit['statement']
            core += f'\nSTATEMENT: {unit["name"]}' + (f' ({unit["title"]})' if unit.get('title') else '') + '\n' + self._numbered(sid, a, b, mark=False) + '\n'
            ranges.append([a, b])
        else:
            core += '\nSTATEMENT: not identified in the source; judge the argument on its own terms.\n'
        proofs = segment or unit['proofs']
        if proofs:
            label = 'PROOF' if segment is None else f'PROOF, PART L{segment[0][0]}-L{segment[-1][1]} (other parts are checked separately)'
            core += f'\n{label}:\n' + '\n'.join(self._numbered(sid, a, b, mark=False) for a, b in proofs) + '\n'
            ranges += [list(r) for r in proofs]
        budget = room - len(head.encode()) - len(core.encode()) - sum(len(p.encode()) for p in parts)
        fits = budget >= 0
        extra = []
        if context and mapping:
            for uid in unit.get('refs', []) + unit.get('statement_refs', []):
                other = manuscript.unit(mapping, uid)
                if not other or not other['statement'] or uid == unit.get('id'):
                    continue
                a, b = other['statement'][0], min(other['statement'][1], other['statement'][0] + 24)
                text = f'{other["name"]}' + (f' ({other["title"]})' if other['title'] else '') + ':\n' + self._numbered(sid, a, b, mark=False)
                extra.append(('statement', a, b, text))
            for a, b in unit.get('equations', []):
                extra.append(('equation', a, b, self._numbered(sid, a, b, mark=False)))
        chosen = []
        for kind, a, b, text in extra:
            if len(text.encode()) + 2 <= budget:
                chosen.append((kind, a, b, text))
                budget -= len(text.encode()) + 2
                ranges.append([a, b])
        if fits and len(chosen) < len(extra):
            self._warn(f'{unit["name"]}: some cited statements or equations did not fit the model context and were left out.')
        context_text = ''
        if any(k == 'statement' for k, *_ in chosen):
            context_text += '\nSTATEMENTS OF RESULTS USED (context, not to be checked):\n' + '\n\n'.join(t for k, _, _, t in chosen if k == 'statement') + '\n'
        if any(k == 'equation' for k, *_ in chosen):
            context_text += '\nEQUATIONS CITED (context):\n' + '\n'.join(t for k, _, _, t in chosen if k == 'equation') + '\n'
        return head + '\n\n'.join(parts) + context_text + core, fits, ranges

    def _mark(self, source_id, ranges):
        with self._mutex:
            self.state['manuscript_ranges'].setdefault(source_id, []).extend([list(r) for r in ranges])

    def _segments(self, unit, room, notation=()):
        """The proof as one packet, or split into parts that fit the context: [(segment, text, ranges)]."""
        packet, fits, ranges = self._packet(unit, room, notation=notation)
        if fits or not unit['proofs']:
            return [(None, packet, ranges)]
        numbers = [n for a, b in unit['proofs'] for n in range(a, b + 1)]
        lines = self._lines(unit['source'])
        size = max(10, len(numbers) // 2)
        while True:
            parts = [numbers[i:i + size] for i in range(0, len(numbers), size)]
            longest = max(len('\n'.join(lines[n - 1] for n in part).encode()) for part in parts)
            if longest + 4000 < room or size <= 10:
                break
            size = max(10, size * 2 // 3)
        self._warn(f'{unit["name"]}: the proof is longer than the model context; it was checked in {len(parts)} parts.')
        out = []
        for part in parts:
            text, _, part_ranges = self._packet(unit, room, notation=notation, segment=_ranges(part), context=False)
            out.append((_ranges(part), text, part_ranges))
        return out

    # -- checking (quick and journal) ---------------------------------------------------------

    CHECK_CAP = 8192  # with thinking; a direct answer gets at most 3000

    def _check_cap(self):
        return self._cap(self.CHECK_CAP if self._thinks(self.CHECK_CAP) else 3000)

    def _check_jobs(self, unit, lenses, journal=False, notation=()):
        """Jobs for the passes of one unit that have not run yet."""
        cap = self._check_cap()
        jobs = []
        room = self._room(cap, P.check_prompt('line', '', journal))
        segments = self._segments(unit, room, notation)
        for index, lens in enumerate(lenses):
            for part, (segment, packet, ranges) in enumerate(segments):
                key = f'{unit["id"]}#{index}' + (f'.{part}' if len(segments) > 1 else '')
                if key in self.state['checks']:
                    continue
                jobs.append((key, self._check_job(key, unit, lens, packet, ranges, segment, journal)))
        return jobs, max(len(text.encode()) for _, text, _ in segments) // 3 + 1500

    def _check_job(self, key, unit, lens, packet, ranges, segment, journal):
        def run():
            self.emit('notice', f'Checking {unit["name"]}: {lens} pass')
            self._mark(unit['source'], ranges)
            result = self._judge('check', P.check_prompt(lens, packet, journal), cap=self._check_cap(),
                                 schema=P.check_schema(journal), key=key)
            parsed = self._parse_check(result, unit, segment, journal)
            parsed.update(lens=lens, unit=unit['id'])
            with self._mutex:
                self.state['checks'][key] = parsed
                self._save()
        return run

    def _parse_check(self, result, unit, segment, journal):
        value = result['value']
        if value is None:
            return {'usable': False, 'error': 'The answer was not usable JSON' + ('' if result['complete'] else ' (cut at the output limit)'),
                    'issues': [], 'verdict': 'unavailable', 'explanation': ''}
        allowed = set(n for a, b in (segment or unit['proofs']) for n in range(a, b + 1))
        issues = []
        for item in value.get('issues') or []:
            if not isinstance(item, dict):
                continue
            kind = item.get('kind') if item.get('kind') in RANK else 'uncertainty'
            issue = {'start_line': _int(item.get('start_line')), 'end_line': _int(item.get('end_line')),
                     'quote': str(item.get('quote') or '')[:200], 'kind': kind,
                     'evidence': str(item.get('evidence') or '')[:1500], 'suggestion': str(item.get('suggestion') or '')[:800]}
            if journal:
                issue['severity'] = 'major' if item.get('severity') == 'major' else 'minor'
            if issue['end_line'] < issue['start_line']:
                issue['end_line'] = issue['start_line']
            issue['located'] = self._locate(issue, unit['source'], allowed)
            issues.append(issue)
        verdict = value.get('verdict') if value.get('verdict') in ('no_issue_found', 'issues_found', 'uncertain') else 'uncertain'
        concrete = [i for i in issues if i['kind'] != 'uncertainty']
        if verdict == 'no_issue_found' and issues:
            verdict = 'issues_found' if concrete else 'uncertain'
        elif verdict == 'issues_found' and not concrete:
            verdict = 'uncertain'
        parsed = {'usable': True, 'verdict': verdict, 'explanation': str(value.get('explanation') or '')[:3000], 'issues': issues}
        if journal:
            parsed['external'] = [{'key': str(e.get('key') or '')[:40], 'place': str(e.get('place') or '')[:80],
                                   'used_for': str(e.get('used_for') or '')[:300]}
                                  for e in value.get('external') or [] if isinstance(e, dict) and e.get('key')][:4]
        return parsed

    def _locate(self, issue, source_id, allowed):
        """'exact' or 'approximate' when the quote is at the cited lines (moved there if it is elsewhere in the proof), else 'no'."""
        lines = self._lines(source_id)
        quote = _fold(issue['quote'])
        words = _words(issue['quote'])

        def found(a, b, pad):
            window = '\n'.join(lines[max(0, a - 1 - pad):min(len(lines), b + pad)])
            if quote and len(quote) >= 4 and quote in _fold(window):
                return 'exact'
            if words:
                present = set(_words(window))
                if sum(w in present for w in words) >= max(2, .6 * len(words)):
                    return 'approximate'
            return None

        if issue['start_line'] in allowed or issue['end_line'] in allowed:
            hit = found(issue['start_line'], issue['end_line'], 1)
            if hit:
                return hit
        # Small models often miscount lines: move the issue to the proof line that holds its quote.
        for wanted in ('exact', 'approximate'):
            for n in sorted(allowed):
                if found(n, n, 0) == wanted:
                    issue['start_line'] = issue['end_line'] = n
                    return wanted
        return 'no'

    def _clusters(self, unit_id):
        """Merge the issues of all usable passes on a unit by overlapping lines."""
        items = []
        for key in sorted(self.state['checks']):
            check = self.state['checks'][key]
            if check.get('unit') != unit_id or not check.get('usable'):
                continue
            for issue in check['issues']:
                items.append(dict(issue, pass_key=key, lens=check.get('lens')))
        located = [i for i in items if i['located'] != 'no']
        clusters = []
        for issue in sorted(located, key=lambda i: (i['start_line'], i['end_line'])):
            for cluster in clusters:
                if issue['start_line'] <= cluster['end_line'] + 1 and issue['end_line'] >= cluster['start_line'] - 1 \
                        and cluster['end_line'] - cluster['start_line'] < 15:
                    cluster['items'].append(issue)
                    cluster['start_line'] = min(cluster['start_line'], issue['start_line'])
                    cluster['end_line'] = max(cluster['end_line'], issue['end_line'])
                    break
            else:
                clusters.append({'start_line': issue['start_line'], 'end_line': issue['end_line'], 'items': [issue]})
        for issue in items:
            if issue['located'] == 'no':
                clusters.append({'start_line': issue['start_line'], 'end_line': issue['end_line'], 'items': [issue], 'unlocated': True})
        out = []
        for cluster in clusters:
            best = max(cluster['items'], key=lambda i: (RANK[i['kind']], i['located'] == 'exact', i.get('severity') == 'major', len(i['evidence'])))
            kind = best['kind']
            entry = {'unit': unit_id, 'start_line': cluster['start_line'], 'end_line': cluster['end_line'],
                     'kind': kind, 'quote': best['quote'], 'evidence': best['evidence'], 'suggestion': best['suggestion'],
                     'passes': len({i['pass_key'] for i in cluster['items']}), 'lenses': sorted({i['lens'] for i in cluster['items'] if i['lens']}),
                     'located': best['located'], 'unlocated': bool(cluster.get('unlocated'))}
            if any('severity' in i for i in cluster['items']):
                entry['severity'] = 'major' if any(i.get('severity') == 'major' for i in cluster['items']) or kind == 'invalid_inference' else 'minor'
            entry['key'] = f'{unit_id}:L{entry["start_line"]}-L{entry["end_line"]}:{kind}' + (':u' if entry['unlocated'] else '')
            out.append(entry)
        return out

    def _confirm_job(self, cluster, unit, notation=()):
        key = cluster['key']
        cap = self._cap(4096 if self._thinks(4096) else 1500)

        def run():
            self.emit('notice', f'Re-checking an alleged issue in {unit["name"]} (lines {cluster["start_line"]}-{cluster["end_line"]})')
            room = self._room(cap, P.confirm_prompt(cluster, ''))
            packets = self._segments(unit, room, notation)
            # The part of a long proof that holds the alleged issue.
            _, text, ranges = next((p for p in packets if p[0] is None or any(a <= cluster['start_line'] <= b for a, b in p[0])), packets[0])
            self._mark(unit['source'], ranges)
            result = self._judge('confirm', P.confirm_prompt(cluster, text), cap=cap, schema=P.CONFIRM_SCHEMA, key=key)
            value = result['value'] or {}
            assessment = value.get('assessment') if value.get('assessment') in ('valid', 'rejected', 'undecided') else 'unavailable'
            entry = {'assessment': assessment, 'kind': value.get('kind') if value.get('kind') in RANK else cluster['kind'],
                     'argument': str(value.get('argument') or '')[:2000], 'suggestion': str(value.get('suggestion') or '')[:800]}
            with self._mutex:
                self.state['confirms'][key] = entry
                self._save()
        return key, run

    def _settle(self, cluster):
        """The status of a merged issue after its re-check (or without one)."""
        if cluster['kind'] == 'uncertainty':
            return 'uncertainty'
        confirm = self.state['confirms'].get(cluster['key'])
        if confirm is None:
            return 'unconfirmed'
        return {'valid': 'confirmed', 'rejected': 'rejected', 'undecided': 'undecided'}.get(confirm['assessment'], 'unconfirmed')

    def _unit_verdict(self, unit_id):
        checks = [c for c in self.state['checks'].values() if c.get('unit') == unit_id]
        usable = [c for c in checks if c.get('usable')]
        if not usable:
            return 'unavailable', []
        clusters = self._clusters(unit_id)
        for cluster in clusters:
            cluster['status'] = self._settle(cluster)
            confirm = self.state['confirms'].get(cluster['key'])
            if confirm and confirm['assessment'] == 'valid':
                cluster['kind'] = confirm['kind']
        statuses = {c['status'] for c in clusters if not (c['unlocated'] and c['status'] != 'confirmed')}
        if 'confirmed' in statuses:
            verdict = 'issues_found'
        elif 'unconfirmed' in statuses:
            verdict = 'issues_alleged'
        elif statuses & {'undecided', 'uncertainty'} or any(c['verdict'] == 'uncertain' for c in usable):
            verdict = 'uncertain'
        else:
            verdict = 'no_issue_found'
        return verdict, clusters

    # -- report helpers -----------------------------------------------------------------------

    def _loc(self, source_id, a, b):
        return f'[{source_id}:L{a}-L{b}]' if a != b else f'[{source_id}:L{a}]'

    def _issue_lines(self, cluster, source_id, show_passes=True):
        lines = []
        where = self._loc(source_id, cluster['start_line'], cluster['end_line'])
        status = {'confirmed': 'confirmed by an independent re-check', 'unconfirmed': 'not re-checked',
                  'undecided': 'the re-check could not decide', 'rejected': 'rejected by the re-check',
                  'uncertainty': 'a point the reviewer could not decide'}.get(cluster.get('status'), '')
        passes = f'; raised by {cluster["passes"]} of {self._passes_on(cluster["unit"])} passes' if show_passes else ''
        located = '' if cluster['located'] == 'exact' else ' (quote matched approximately)' if cluster['located'] == 'approximate' else ' (the quote was not found in the proof; location unsure)'
        lines.append(f'**{KIND_LABEL[cluster["kind"]]}** at {where} — {status}{passes}{located}.')
        if cluster['quote']:
            lines.append(f'> {cluster["quote"]}')
        lines.append(cluster['evidence'] or '(no explanation given)')
        confirm = self.state['confirms'].get(cluster['key'])
        if confirm and confirm['argument'] and confirm['assessment'] in ('valid', 'rejected', 'undecided'):
            lines.append(f'*Re-check:* {confirm["argument"]}')
        suggestion = (confirm or {}).get('suggestion') or cluster['suggestion']
        if suggestion and cluster.get('status') != 'rejected':
            lines.append(f'*Suggested repair:* {suggestion}')
        return lines

    def _passes_on(self, unit_id):
        return sum(1 for c in self.state['checks'].values() if c.get('unit') == unit_id and c.get('usable'))

    def _usage(self):
        s = self.state
        return (f'Generated tokens: {s["tokens_charged"]}/{s["settings"]["max_tokens"]}; input tokens: '
                f'{s["input_tokens_charged"]}/{s["settings"]["max_input_tokens"]}; time: {s["seconds_used"]:.0f}/'
                f'{s["settings"]["max_seconds"]:.0f} s; model calls: {len(s["calls"])}; concurrency: {s.get("concurrency", 1)}.')

    def _write_report(self, body, appendix):
        s = self.state
        head = [f'# {TITLES[self.VARIANT]}', '',
                '**Model draft — a fallible review by a language model, not a certificate.**', '',
                f'Status: `{s["status"]}`. Phase: `{s["phase"]}`. Effort: {s.get("level")}. Job ID: `{s["id"]}`.', '']
        if s.get('stop_reason') and s['status'] != 'reviewed':
            head += [s['stop_reason'], '']
        if getattr(self, '_headline', ''):
            head += [self._headline, '']
        s['draft'] = '\n'.join(body)
        notes = list(dict.fromkeys(s.get('warnings', []) + s.get('citation_issues', [])))
        tail = ['', '---', '']
        if notes:
            tail += ['## Controller notes', ''] + ['- ' + n for n in notes] + ['']
        tail += appendix + ['', '## Usage', '', self._usage(), '',
                            'state.json keeps the pinned source, the map, every check and re-check; artifacts/ keeps each request and streamed answer.']
        _atomic_write(self.directory / 'report.md', _bytes('\n'.join(head + body + tail)))

    def _report(self):
        try:
            body, appendix = self._render()
        except Exception as exc:  # a report must always be written, even after an error
            body, appendix = [f'*The report could not be assembled: {type(exc).__name__}: {exc}*'], []
        self._write_report(body, appendix)

    def _finish(self):
        body, _ = self._render()
        self.state['draft'] = '\n'.join(body)
        self.state['citation_issues'] = [i for i in self._citation_issues(self.state['draft'])
                                         if not i.startswith('Report does not cite')]
        complete = self._complete()
        self.state['status'] = 'reviewed' if complete else 'partial'
        self.state['stop_reason'] = ('Review finished; the output remains a model draft.' if complete else
                                     'Review finished with gaps (see the controller notes); the output remains a model draft.')


class QuickReviewRunner(ReviewBase):
    VARIANT = 'quick'
    PIPELINE = PIPELINES['quick']

    def start(self, goal, **kwargs):
        self._prepare(goal, kwargs)
        return super().start(goal, **kwargs)

    def _lenses(self):
        return list(P.LENS_ORDER[:self.effort['passes']])

    def _step(self, phase):
        if phase == 'plan':
            self._map()
            self.state['phase'] = 'extract'
        elif phase == 'extract':
            self._pick_unit()
            self.state['phase'] = 'verify'
        elif phase == 'verify':
            unit = self.state['unit']
            jobs, estimate = self._check_jobs(unit, self._lenses())
            affordable = []
            for key, fn in jobs:
                if not affordable or self._affordable(self._check_cap() * (len(affordable) + 1), estimate * (len(affordable) + 1), share=.75):
                    affordable.append((key, fn))
            if len(affordable) < len(jobs):
                self._warn(f'Only {len(affordable) + len(self.state["checks"])} of {len(self._lenses())} planned passes fit the budget.')
            self._wave(affordable)
            self.state['phase'] = 'confirm'
        elif phase == 'confirm':
            unit = self.state['unit']
            if self.effort['confirm']:
                clusters = [c for c in self._clusters(unit['id']) if c['kind'] != 'uncertainty']
                jobs = []
                for cluster in sorted(clusters, key=lambda c: (-RANK[c['kind']], c['unlocated'], -c['passes']))[:6]:
                    if cluster['key'] in self.state['confirms']:
                        continue
                    if not self._affordable(1500 * (len(jobs) + 1), 3000 * (len(jobs) + 1), share=.92):
                        self._warn('Some alleged issues were not re-checked: the budget ran out.')
                        break
                    jobs.append(self._confirm_job(cluster, unit))
                self._wave(jobs)
            self.state['phase'] = 'done'
        else:
            raise ValueError('Unknown review phase: ' + phase)

    def _complete(self):
        unit = self.state.get('unit')
        if not unit:
            return False
        usable = [c for c in self.state['checks'].values() if c.get('usable')]
        return len(usable) >= len(self._lenses())

    def _render(self):
        s = self.state
        unit = s.get('unit')
        body = []
        if not unit:
            return ['*No proof was selected yet.*'], []
        sid = unit['source']
        verdict, clusters = self._unit_verdict(unit['id'])
        s['verdict'] = verdict
        # In pass order, not completion order: concurrent passes finish in any order.
        usable = [s['checks'][key] for key in sorted(s['checks']) if s['checks'][key].get('usable')]
        lenses = ', '.join(dict.fromkeys(c['lens'] for c in usable if c.get('lens')))
        where = _span(unit['proofs'])
        headline = {
            'no_issue_found': f'∎ **No issue found** by {len(usable)} independent pass{"es" if len(usable) != 1 else ""} ({lenses}).',
            'issues_found': '■ **Issues found**: at least one step is wrong or unjustified (confirmed by an independent re-check).',
            'issues_alleged': '■ **Issues alleged**: the passes report problems that were not re-checked at this effort.',
            'uncertain': '◪ **Uncertain**: the passes could not decide every step.',
            'unavailable': '□ **No usable check yet.**',
        }[verdict]
        if verdict == 'no_issue_found' and any(c['status'] == 'rejected' for c in clusters):
            headline = ('∎ **No issue stands**: the objections the passes raised were rejected by an independent '
                        're-check (see below; judge them yourself).')
        self._headline = headline
        source = 'the pasted text' if sid == 'P1' else f'`{self._source(sid)["path"]}`'
        passes = f' by {len(usable)} independent pass{"es" if len(usable) != 1 else ""} ({lenses})' if usable else ''
        body += [f'## {unit["name"].capitalize() if unit["id"] == "P" else unit["name"]}', '',
                 f'Proof checked: {where} of {source}{passes}. The proof was not rewritten.']
        real = [c for c in clusters if c['status'] in ('confirmed', 'unconfirmed') and not (c['unlocated'] and c['status'] != 'confirmed')]
        open_ = [c for c in clusters if c['status'] in ('undecided', 'uncertainty')]
        rejected = [c for c in clusters if c['status'] == 'rejected']
        loose = [c for c in clusters if c['unlocated'] and c['status'] == 'unconfirmed']
        if real:
            body += ['', '## Issues', '']
            for n, cluster in enumerate(sorted(real, key=lambda c: (c['status'] != 'confirmed', -RANK[c['kind']], c['start_line'])), 1):
                body += _item(f'{n}. ', self._issue_lines(cluster, sid), '   ')
        if open_:
            body += ['', '## Points not settled', '']
            for cluster in open_:
                body += _item('- ', self._issue_lines(cluster, sid), '  ')
        if loose:
            body += ['', '## Objections that could not be located', '',
                     'These were raised without a quote found in the proof; read them with care.', '']
            for cluster in loose:
                body += ['- ' + cluster['evidence'][:600]]
        if rejected:
            body += ['', '## Objections raised and rejected', '']
            for cluster in rejected:
                confirm = s['confirms'].get(cluster['key'], {})
                body += [f'- {self._loc(sid, cluster["start_line"], cluster["end_line"])} {cluster["evidence"][:400]} — *rejected:* {confirm.get("argument", "")[:500]}']
        if usable:
            body += ['', '## What each pass saw', '']
            for key in sorted(s['checks']):
                check = s['checks'][key]
                if check.get('usable'):
                    body += [f'- **{check.get("lens")}** — {check["verdict"].replace("_", " ")}: {check["explanation"][:900]}']
        appendix = ['## The statement and proof as numbered', '', '```']
        lines = self._lines(sid)
        ranges = ([unit['statement']] if unit.get('statement') else []) + unit['proofs']
        for a, b in ranges:
            appendix += [f'L{n}: {lines[n - 1]}' for n in range(a, min(b, a + 400) + 1)]
        appendix += ['```']
        failed = [s['checks'][key] for key in sorted(s['checks']) if not s['checks'][key].get('usable')]
        if failed:
            appendix += ['', f'{len(failed)} pass(es) gave no usable answer: ' + '; '.join(c.get('error', '') for c in failed)]
        return body, appendix


class ExplainRunner(ReviewBase):
    VARIANT = 'explain'
    PIPELINE = PIPELINES['explain']

    def start(self, goal, **kwargs):
        self._prepare(goal, kwargs)
        return super().start(goal, **kwargs)

    def _extra_state(self):
        return {**super()._extra_state(), 'explanation': None, 'explain_problems': None, 'fills': 0}

    def _fine(self):
        unit = self.state['unit']
        return len(self._proof_lines(unit)) <= FINE_LINES if unit['proofs'] else True

    def _explain_cap(self):
        return self._cap(16384 if self._thinks(16384) else 8000)

    def _step(self, phase):
        if phase == 'plan':
            self._map()
            self.state['phase'] = 'extract'
        elif phase == 'extract':
            self._pick_unit()
            self.state['phase'] = 'explain'
        elif phase == 'explain':
            unit = self.state['unit']
            granularity = P.FINE if self._fine() else P.COARSE
            policy = P.EXPLAIN_POLICY.format(granularity=granularity if unit['proofs'] else 'There is no proof: give steps = [] and explain the statement.')
            packet, fits, ranges = self._packet(unit, self._room(self._explain_cap(), policy), notation=())
            if not fits:
                self._warn('The proof is longer than the model context allows for one explanation; parts may be missing.')
            self._mark(unit['source'], ranges)
            with self._timeout():
                result = self._judge('explain', policy + '\n\n' + packet, cap=self._explain_cap(), schema=P.EXPLAIN_SCHEMA,
                                     key='explain', direct_cap=8000)
            value = self._clean_explanation(result['value'])
            if value is None:
                raise AgentError('The explanation was not usable JSON' + ('' if result['complete'] else ' (cut at the output limit; give it more generated tokens)'))
            self.state['explanation'] = value
            self.state['phase'] = 'fill'
        elif phase == 'fill':
            missing = self._missing()
            if missing and self.state['fills'] < self.effort['fills'] and self._affordable(3000, 4000, share=.85):
                self._fill(missing)
                self.state['fills'] += 1
                return
            if missing:
                self._warn('Proof lines not explained one by one: ' + _span(missing) + '.')
            self.state['phase'] = 'check' if self.effort['explain_check'] else 'done'
        elif phase == 'check':
            if self._affordable(1500, 6000, share=.8):
                self._check_explanation()
            self.state['phase'] = 'fix' if self.state.get('explain_problems') else 'done'
        elif phase == 'fix':
            if self._affordable(self._explain_cap(), 8000, share=.95):
                self._fix()
            else:
                self._warn('The accuracy check found problems that could not be corrected within the budget; they are listed below.')
            self.state['phase'] = 'done'
        else:
            raise ValueError('Unknown review phase: ' + phase)

    def _clean_explanation(self, value):
        if not isinstance(value, dict) or not str(value.get('says') or '').strip():
            return None
        unit = self.state['unit']
        allowed = set(n for a, b in unit['proofs'] for n in range(a, b + 1))
        steps = []
        for item in value.get('steps') or []:
            if not isinstance(item, dict):
                continue
            a, b = _int(item.get('start_line')), _int(item.get('end_line'))
            if b < a:
                a, b = b, a
            if allowed and not (allowed & set(range(a, b + 1))):
                continue
            if allowed:
                inside = sorted(allowed & set(range(a, b + 1)))
                a, b = inside[0], inside[-1]
            steps.append({'start_line': a, 'end_line': b, 'what': str(item.get('what') or '')[:800],
                          'why': str(item.get('why') or '')[:1500], 'uses': str(item.get('uses') or '')[:400]})
        facts = [{'name': str(f.get('name') or '')[:200], 'statement': str(f.get('statement') or '')[:800]}
                 for f in value.get('facts') or [] if isinstance(f, dict) and f.get('name')]
        return {'says': str(value.get('says'))[:2500], 'idea': str(value.get('idea') or '')[:1200],
                'steps': sorted(steps, key=lambda s: (s['start_line'], s['end_line'])), 'facts': facts[:10],
                'hypotheses': str(value.get('hypotheses') or '')[:1500], 'gaps': str(value.get('gaps') or '')[:1000]}

    def _missing(self):
        unit, value = self.state['unit'], self.state['explanation']
        if not unit['proofs'] or not self._fine():
            return []
        covered = set(n for s in value['steps'] for n in range(s['start_line'], s['end_line'] + 1))
        return _ranges([n for n in self._proof_lines(unit) if n not in covered])

    def _fill(self, missing):
        unit = self.state['unit']
        packet, _, _ = self._packet(unit, self._room(3000, P.FILL_POLICY), context=False)
        prompt = (P.FILL_POLICY + '\nMISSING LINES: ' + _span(missing) + '\n\nWALKTHROUGH SO FAR:\n' +
                  json.dumps(self.state['explanation']['steps'], ensure_ascii=False) + '\n\n' + packet)
        with self._timeout():
            result = self._ask('fill', prompt, cap=3000, schema=P.STEPS_SCHEMA, key=f'fill{self.state["fills"]}')
        wanted = set(n for a, b in missing for n in range(a, b + 1))
        added = []
        for item in (result['value'] or {}).get('steps') or []:
            if isinstance(item, dict):
                a, b = _int(item.get('start_line')), _int(item.get('end_line'))
                if wanted & set(range(min(a, b), max(a, b) + 1)):
                    added.append(item)
        if added:
            merged = dict(self.state['explanation'], steps=self.state['explanation']['steps'] + added)
            self.state['explanation'] = self._clean_explanation(merged) or self.state['explanation']

    def _check_explanation(self):
        unit = self.state['unit']
        explanation = json.dumps(self.state['explanation'], ensure_ascii=False)
        packet, _, _ = self._packet(unit, self._room(1500, P.EXPLAIN_CHECK_POLICY) - len(explanation.encode()), context=False)
        prompt = P.EXPLAIN_CHECK_POLICY + '\n\nEXPLANATION:\n' + explanation + '\n\n' + packet
        with self._timeout():
            result = self._ask('explain-check', prompt, cap=1500, schema=P.EXPLAIN_CHECK_SCHEMA, key='check')
        problems = [{'where': str(p.get('where') or '')[:200], 'problem': str(p.get('problem') or '')[:800], 'fix': str(p.get('fix') or '')[:800]}
                    for p in (result['value'] or {}).get('problems') or [] if isinstance(p, dict) and p.get('problem')]
        self.state['explain_problems'] = problems
        self.state['explain_checked'] = result['value'] is not None

    def _fix(self):
        unit = self.state['unit']
        material = ('\n\nPROBLEMS:\n' + json.dumps(self.state['explain_problems'], ensure_ascii=False) +
                    '\n\nEXPLANATION:\n' + json.dumps(self.state['explanation'], ensure_ascii=False))
        packet, _, _ = self._packet(unit, self._room(self._explain_cap(), P.EXPLAIN_FIX_POLICY) - len(material.encode()), context=False)
        prompt = P.EXPLAIN_FIX_POLICY + material + '\n\n' + packet
        with self._timeout():
            result = self._ask('explain-fix', prompt, cap=self._explain_cap(), schema=P.EXPLAIN_SCHEMA, key='fix')
        fixed = self._clean_explanation(result['value'])
        if fixed is None:
            self._warn('The corrected explanation was not usable; the first version is kept and the problems are listed.')
            return
        before = len(self._missing())
        previous = self.state['explanation']
        self.state['explanation'] = fixed
        if len(self._missing()) > before:
            self.state['explanation'] = dict(fixed, steps=previous['steps'])
        self.state['explain_fixed'] = True

    def _complete(self):
        return bool(self.state.get('explanation')) and not (self.effort['explain_check'] and not self.state.get('explain_checked'))

    def _render(self):
        s, unit, value = self.state, self.state.get('unit'), self.state.get('explanation')
        if not unit:
            return ['*No result was selected yet.*'], []
        if not value:
            return [f'## {unit["name"]}', '', '*The explanation is not written yet.*'], []
        sid = unit['source']
        lines = self._lines(sid)
        show_source = self._source(sid)['path'].lower().endswith(('.tex', '.md', '.txt')) or sid == 'P1'
        body = [f'## {unit["name"].capitalize() if unit["id"] == "P" else unit["name"]}', '', '### What it says', '', value['says'],
                '', '### The idea', '', value['idea'] or '—']
        if value['steps']:
            body += ['', '### Step by step', '']
            for step in value['steps']:
                a, b = step['start_line'], step['end_line']
                body += [f'**{"Line" if a == b else "Lines"} {a if a == b else f"{a}–{b}"}** {self._loc(sid, a, b)}', '']
                if show_source:
                    excerpt = [lines[n - 1].strip() for n in range(a, min(b, a + 3) + 1) if lines[n - 1].strip()]
                    body += ['> ' + line for line in excerpt] + ([''] if excerpt else [])
                body += [step['what'], '']
                if step['why']:
                    body += [f'*Why:* {step["why"]}', '']
                if step['uses']:
                    body += [f'*Uses:* {step["uses"]}', '']
            missing = self._missing()
            body += [('Every line of the proof is explained.' if not missing else f'Lines not explained one by one: {_span(missing)}.')
                     if self._fine() else 'The proof is long: lines are grouped into steps.']
        if value['facts']:
            body += ['', '### Standard facts used', '']
            body += [f'- **{f["name"]}** — {f["statement"]}' for f in value['facts']]
        if value['hypotheses']:
            body += ['', '### Where the hypotheses matter', '', value['hypotheses']]
        if value['gaps']:
            body += ['', '### What the proof leaves implicit', '', value['gaps']]
        problems = s.get('explain_problems') or []
        appendix = []
        if problems:
            appendix += ['## Accuracy check', '', ('Problems found and corrected in the text above (corrections are themselves model output):'
                                                    if s.get('explain_fixed') else 'Problems found by the accuracy check (not corrected):'), '']
            appendix += [f'- {p["where"]}: {p["problem"]}' for p in problems]
        elif s.get('explain_checked'):
            appendix += ['## Accuracy check', '', 'An independent pass found no problem in the explanation.']
        return body, appendix


class JournalRunner(ReviewBase):
    """The detailed review: read the paper, then check its proofs."""
    VARIANT = 'journal'
    PIPELINE = PIPELINES['journal']
    CHECK_PROOFS = True

    def start(self, goal, **kwargs):
        self._prepare(goal, kwargs)
        return super().start(goal, **kwargs)

    def _extra_state(self):
        return {**super()._extra_state(), 'scope': None, 'plan_units': [], 'unchecked': [], 'cited_checks': [],
                'novelty': None, 'search_results': [], 'presentation': [], 'presentation_done': [], 'write': None,
                'reading': {}, 'read_parts': []}

    def _notation(self):
        return [tuple(r) for r in ((self.state.get('scope') or {}).get('notation') or [])]

    def _units(self):
        mapping = self.state['map']
        if mapping['units'] and manuscript.checkable(mapping):
            return mapping['units']
        # Nothing recognised: check the text in fixed-size excerpts.
        return [{'id': f'C{i}', 'type': 'excerpt', 'name': f'Lines {a}–{b}', 'number': '', 'label': '', 'title': '',
                 'statement': None, 'proofs': [[a, b]], 'refs': [], 'statement_refs': [], 'cites': [], 'equations': [],
                 'section': None, 'appendix': False, 'main': i == 1}
                for i, (a, b) in enumerate(manuscript.chunks(mapping['lines'], 120), 1)]

    def _unit(self, uid):
        unit = next((u for u in self._units() if u['id'] == uid), None)
        return dict(unit, source=self.state['sources'][0]['id']) if unit else None

    def _main_ids(self):
        scope = self.state.get('scope') or {}
        return scope.get('main') or [u['id'] for u in self._units() if u['main']]

    def _step(self, phase):
        if phase == 'plan':
            if not self.state['sources']:
                raise WorkerInputError('Pin the manuscript to review.', code='invalid_worker_input')
            if len(self.state['sources']) > 1:
                self._warn('Only the first pinned file is reviewed as the manuscript; the others were not read.')
            self._map()
            self.state['read_parts'] = self._read_parts()
            self.state['phase'] = 'read'
        elif phase == 'read':
            self._read()
            self.state['phase'] = 'scope'
        elif phase == 'scope':
            self._scope()
            if self.CHECK_PROOFS:
                self._plan_units()
            self.state['phase'] = 'check' if self.CHECK_PROOFS else 'literature'
        elif phase == 'check':
            self._checks()
            self.state['phase'] = 'confirm'
        elif phase == 'confirm':
            self._confirms()
            self.state['phase'] = 'literature'
        elif phase == 'literature':
            self._literature()
            self.state['phase'] = 'write'
        elif phase == 'presentation':  # jobs saved before reading came first
            self.state['phase'] = 'write'
        elif phase == 'write':
            self._write()
            self.state['phase'] = 'done'
        else:
            raise ValueError('Unknown review phase: ' + phase)

    # -- scope and plan -----------------------------------------------------------------------

    def _scope(self):
        mapping, sid = self.state['map'], self.state['sources'][0]['id']
        parts = ['REQUEST AND FOCUS OF THE REVIEW:\n' + self.state['goal']]
        lines = self._lines(sid)
        if mapping['abstract']:
            parts.append('ABSTRACT:\n' + self._numbered(sid, *mapping['abstract']))
        sections = mapping['sections']
        first = sections[0]['line'] if sections else 1
        end = sections[1]['line'] - 1 if len(sections) > 1 else min(len(lines), first + 150)
        intro, size = [], 0
        for n in range(first, min(end, first + 220) + 1):
            size += len(lines[n - 1].encode())
            if size > 14000:
                break
            intro.append(n)
        if intro:
            parts.append('INTRODUCTION (excerpt):\n' + self._numbered(sid, intro[0], intro[-1], mark=False))
        summaries = [f'Lines {a}-{b}: {self.state["reading"][f"L{a}"]["summary"]}' for a, b in self.state.get('read_parts') or []
                     if f'L{a}' in self.state.get('reading', {})]
        if summaries:
            parts.append('THE PAPER PART BY PART (your reading notes):\n' + '\n\n'.join(summaries))
        parts.append('STATEMENTS (id: name, section, lines):\n' + manuscript.outline(self._map_view()))
        bib = list(mapping['bibliography'].items())[:40]
        if bib:
            parts.append('BIBLIOGRAPHY (key: entry):\n' + '\n'.join(f'[{k}] {v[:140]}' for k, v in bib))
        cap = 8192 if self._thinks(8192) else 3000
        prompt = P.SCOPE_POLICY + '\n\n' + '\n\n'.join(parts)
        if len(prompt.encode()) > self._room(cap):
            prompt = prompt[:self._room(cap)]
        with self._timeout():
            result = self._judge('scope', prompt, cap=cap, schema=P.SCOPE_SCHEMA, key='scope')
        value = result['value'] or {}
        ids = {u['id'] for u in self._units()}
        main = [i for i in value.get('main') or [] if i in ids]
        notation = []
        for r in value.get('notation') or []:
            a, b = _int(r.get('start_line')), _int(r.get('end_line'))
            if 1 <= a <= b <= len(lines):
                notation.append([a, min(b, a + 59)])
        while sum(b - a + 1 for a, b in notation) > 120:
            notation.pop()
        queries = [re.sub(r'\s+', ' ', str(q)).strip()[:100] for q in value.get('queries') or [] if str(q).strip()]
        queries = [' '.join(q.split()[:10]) for q in queries][:6]
        # Search queries are public topic words: a query copied from the manuscript never leaves the machine.
        text = ' '.join(_words(self.state['sources'][0]['content']))
        copied = [q for q in queries if len(_words(q)) >= 4 and ' '.join(_words(q)) in text]
        if copied:
            self._warn(f'{len(copied)} search quer{"y was" if len(copied) == 1 else "ies were"} copied from the manuscript and dropped.')
        queries = [q for q in queries if q not in copied]
        if not value:
            self._warn('The scoping answer was not usable; the main results were chosen from the manuscript structure.')
        self.state['scope'] = {'overview': str(value.get('overview') or '')[:3500], 'field': str(value.get('field') or '')[:200], 'contribution': str(value.get('contribution') or '')[:1500],
                               'main': main, 'notation': notation, 'queries': queries,
                               'closest': [str(k)[:40] for k in value.get('closest') or []][:6]}

    def _map_view(self):
        mapping = dict(self.state['map'])
        mapping['units'] = self._units()
        return mapping

    def _plan_units(self):
        effort, view = self.effort, self._map_view()
        order = manuscript.priority(view, depth=effort['depth'], main_ids=self._main_ids(), appendix=effort['appendix'])
        if not order:
            order = [u['id'] for u in self._units() if u['proofs']][:3]
        self.state['plan_units'] = order
        main = set(self._main_ids())
        missing = [u['name'] for u in self._units() if u['id'] in main and not u['proofs']]
        if missing:
            self._warn('No proof was found in the manuscript map for: ' + ', '.join(missing) + '.')
        self.emit('notice', f'Plan: check {len(order)} of {len(manuscript.checkable(view))} proofs, main results first')

    # -- checks ---------------------------------------------------------------------------------

    def _lenses(self, uid):
        main = uid in set(self._main_ids())
        count = self.effort['main_passes'] if main else self.effort['other_passes']
        return ['line', 'adversarial'][:count]

    def _write_reserve(self):
        return min(5000, self.state['settings']['max_tokens'] // 5)

    def _checks(self):
        pending = []
        for uid in self.state['plan_units']:
            unit = self._unit(uid)
            if unit:
                jobs, estimate = self._check_jobs(unit, self._lenses(uid), journal=True, notation=self._notation())
                pending += [(key, fn, estimate) for key, fn in jobs]
        batch_size = max(1, self.concurrency * 2)
        index = 0
        while index < len(pending):
            batch = []
            for key, fn, estimate in pending[index:index + batch_size]:
                need_out = self._check_cap() * (len(batch) + 1)
                need_in = estimate * (len(batch) + 1)
                if not self._affordable(need_out, need_in, reserve_out=self._write_reserve(), reserve_in=12000, share=.65):
                    break
                batch.append((key, fn))
            if not batch:
                break
            self._wave(batch)
            index += len(batch)
            self._save()
        checked = {c['unit'] for c in self.state['checks'].values() if c.get('usable')}
        self.state['unchecked'] = [uid for uid in self.state['plan_units'] if uid not in checked]
        if self.state['unchecked']:
            self._warn(f'{len(self.state["unchecked"])} planned proof(s) were not checked: the budget ran out.')

    def _confirms(self):
        if not self.effort['confirm']:
            return
        main = set(self._main_ids())
        clusters = []
        for uid in self.state['plan_units']:  # plan order, so ties are settled the same way at any concurrency
            clusters += [c for c in self._clusters(uid) if c['kind'] != 'uncertainty']
        clusters.sort(key=lambda c: (c.get('severity') != 'major', c['unit'] not in main, -RANK[c['kind']], c['unlocated'], -c['passes']))
        jobs = []
        for cluster in clusters[:12 if self.state['level'] in ('low', 'medium') else 24]:
            if cluster['key'] in self.state['confirms']:
                continue
            if not self._affordable(1500 * (len(jobs) + 1), 4000 * (len(jobs) + 1), reserve_out=self._write_reserve(), reserve_in=12000, share=.78):
                self._warn('Some alleged issues were not re-checked: the budget ran out.')
                break
            jobs.append(self._confirm_job(cluster, self._unit(cluster['unit']), self._notation()))
        for start in range(0, len(jobs), max(1, self.concurrency * 2)):
            self._wave(jobs[start:start + max(1, self.concurrency * 2)])

    # -- literature -------------------------------------------------------------------------------

    def _literature(self):
        online = bool(self.literature and self.literature.online)
        if not online:
            if self.effort['queries'] or self.effort['cited']:
                self._warn('Online search was off: ' + ('no cited result was checked and ' if self.CHECK_PROOFS else '')
                           + 'novelty was not compared with the literature.')
            return
        self._cited_checks()
        self._novelty()

    def _cited_checks(self):
        wanted = self.effort['cited'] - len(self.state['cited_checks'])
        if wanted <= 0:
            return
        bibliography = self.state['map']['bibliography']
        main = set(self._main_ids())
        candidates, seen = [], {(c['key'], c['place']) for c in self.state['cited_checks']}
        for key in sorted(self.state['checks'], key=lambda k: (self.state['checks'][k].get('unit') not in main, k)):
            for item in self.state['checks'][key].get('external') or []:
                bib_key = item['key'].strip('[] ')
                if bib_key in bibliography and item['place'] and item['used_for'] and (bib_key, item['place']) not in seen:
                    seen.add((bib_key, item['place']))
                    candidates.append(dict(item, key=bib_key, entry=bibliography[bib_key]))
        from .refcheck import CheckRunner
        from .agent import Agent
        from .literature import LiteratureTools
        from .tools import Workspace
        for item in candidates[:wanted]:
            seconds = min(240, (self.state['settings']['max_seconds'] - self._elapsed()) * .2)
            tokens = min(15000, int(self._tokens_left(self._write_reserve()) * .3))
            if seconds < 30 or tokens < 4000 or not self._time_left(.8):
                self._warn('Some cited results were not checked: the budget ran out.')
                break
            question = (f'{item["entry"]} — at {item["place"]}: {item["used_for"]}. '
                        'Check that this place of this work states this result.')
            parent = self.literature
            literature = LiteratureTools(self.agent.workspace.root, online=True, timeout=getattr(parent, 'timeout', 15))
            inner = Agent(self.agent.client, Workspace(self.agent.workspace.root, literature=literature, read_types=()),
                          self.agent.model, self.agent.ctx, self.agent.predict, False, 'check', seed=self.agent.seed)
            self.emit('notice', f'Checking the citation [{item["key"]}, {item["place"]}]')
            checker = CheckRunner(inner, emit=lambda kind, value: self.emit('tool', '↳ ' + str(value)) if kind == 'tool' else None)
            with self._timeout():
                try:
                    checker.start(question, guesses=self._guess(item), effort='low', max_seconds=seconds, max_tokens=tokens)
                    result = json.loads(checker.result_for_agent())
                except (ValueError, OSError, AgentError) as exc:
                    result = {'status': 'error', 'references': [], 'error': str(exc)}
            used = checker.state or {}
            self.state['tokens_charged'] += used.get('tokens_charged', 0)
            self.state['input_tokens_charged'] += used.get('input_tokens_charged', 0)
            refs = result.get('references') or []
            best = refs[0] if refs else {}
            self.state['cited_checks'].append({'key': item['key'], 'place': item['place'], 'used_for': item['used_for'],
                                               'entry': item['entry'][:300], 'level': best.get('level', 'not_found'),
                                               'found_place': best.get('place'), 'statement_found': best.get('statement_found'),
                                               'check_id': result.get('check_id'), 'error': result.get('error')})
            self._save()

    @staticmethod
    def _guess(item):
        entry = re.sub(r'\s+', ' ', item['entry'])
        parts = [p.strip() for p in re.split(r'(?<=[A-Za-zÀ-ÿ\]\)]{2})\.\s+', entry) if p.strip()]
        if len(parts) < 2:
            return []
        authors = [re.sub(r'^(?:and\s+)', '', a).strip() for a in re.split(r',\s*|\s+and\s+', parts[0]) if a.strip()]
        year = re.search(r'\b(1[5-9]|20)\d\d\b', entry)
        return [{'authors': authors[:4], 'title': parts[1][:300], 'year': year.group() if year else '',
                 'locator': item['place'], 'why': 'cited by the manuscript at this place'}]

    def _novelty(self):
        scope = self.state['scope'] or {}
        if self.state.get('novelty') is not None or not scope.get('queries') or not self.effort['queries']:
            return
        results, seen = list(self.state['search_results']), {r['title'].lower() for r in self.state['search_results']}
        done = {e['arguments'].get('query') for e in self.state['evidence'] if e['tool'] == 'search_papers'}
        for query in scope['queries'][:self.effort['queries']]:
            if query in done:
                continue
            if not self._time_left(.82):
                break
            self.emit('tool', 'search_papers ' + json.dumps({'query': query}, ensure_ascii=False))
            raw = self.literature.execute('search_papers', {'query': query, 'limit': 5})
            evidence_id = f'E{len(self.state["evidence"]) + 1}'
            filename = self._artifact('evidence-' + evidence_id, str(raw))
            self.state['evidence'].append({'id': evidence_id, 'tool': 'search_papers', 'arguments': {'query': query},
                                           'artifact': filename, 'result': str(raw), 'round': 0})
            try:
                hits = json.loads(raw).get('results') or []
            except (ValueError, AttributeError):
                hits = []
            for hit in hits:
                title = str(hit.get('title') or '').strip()
                if title and title.lower() not in seen:
                    seen.add(title.lower())
                    results.append({'title': title[:300], 'year': str(hit.get('year') or hit.get('published') or '')[:10],
                                    'authors': ', '.join(map(str, (hit.get('authors') or [])[:4]))[:200],
                                    'abstract': str(hit.get('abstract') or hit.get('summary') or '')[:500],
                                    'url': str(hit.get('url') or hit.get('source_url') or '')[:300], 'query': query})
            self.state['search_results'] = results[:15]
            self._save()
        if not results or not self._affordable(1500, 6000, reserve_out=self._write_reserve(), share=.86):
            return
        listing = '\n'.join(f'{n}. {r["title"]} ({r["authors"]}, {r["year"]}). {r["abstract"][:350]}' for n, r in enumerate(results, 1))
        prompt = (P.NOVELTY_POLICY + '\n\nFIELD: ' + scope.get('field', '') + '\nCLAIMED CONTRIBUTION: ' + scope.get('contribution', '')
                  + '\n\nSEARCH RESULTS:\n' + listing)
        with self._timeout():
            result = self._ask('novelty', prompt, cap=1500, schema=P.NOVELTY_SCHEMA, key='novelty')
        value = result['value'] or {}
        related = [{'n': _int(r.get('n')), 'relation': str(r.get('relation') or '')[:300]} for r in value.get('related') or []
                   if isinstance(r, dict) and 1 <= _int(r.get('n')) <= len(results)]
        self.state['novelty'] = {'assessment': str(value.get('assessment') or '')[:2000], 'related': related}

    # -- presentation ------------------------------------------------------------------------------

    def _read_parts(self):
        """The paper up to its bibliography, in parts that follow its sections (at most 150 lines and 14 kB each)."""
        sid, mapping = self.state['sources'][0]['id'], self.state['map']
        lines = self._lines(sid)
        end = len(lines)
        for n, line in enumerate(lines, 1):
            if re.match(r'^\s*(?:#{1,6}\s*)?(?:\d+\.\s+)?(references|bibliography)\s*$', line, re.I) or \
                    re.search(r'\\begin\{thebibliography\}|\\bibliography\{|\\printbibliography', line):
                end = n - 1
                break
        starts = sorted({1} | {sec['line'] for sec in mapping['sections'] if 1 < sec['line'] <= end})
        parts = []
        for index, a in enumerate(starts):
            b = (starts[index + 1] - 1) if index + 1 < len(starts) else end
            while a <= b:
                stop, size = a, 0
                while stop <= b and stop - a < 150 and size + len(lines[stop - 1].encode()) <= 14000:
                    size += len(lines[stop - 1].encode())
                    stop += 1
                stop = max(stop, a + 1)
                parts.append([a, stop - 1])
                a = stop
        return [r for r in parts if any(lines[n - 1].strip() for n in range(r[0], r[1] + 1))]

    def _read(self):
        """Read the whole paper first: what each part does, and its typos and presentation problems."""
        sid = self.state['sources'][0]['id']
        pdf = self._source(sid)['path'].lower().endswith('.pdf')
        jobs = []
        for a, b in self.state['read_parts']:
            key = f'L{a}'
            if key in self.state['reading']:
                continue
            jobs.append((key, self._read_job(key, sid, a, b, pdf)))
        size = max(1, self.concurrency * 2)
        for start in range(0, len(jobs), size):
            batch = []
            for key, fn in jobs[start:start + size]:
                if not self._affordable(3000 * (len(batch) + 1), 6000 * (len(batch) + 1), reserve_out=self._write_reserve() + 6000,
                                        reserve_in=20000, share=.45):
                    break
                batch.append((key, fn))
            if not batch:
                break
            self._wave(batch)
        missing = [r for r in self.state['read_parts'] if f'L{r[0]}' not in self.state['reading']]
        if missing:
            self._warn(f'{len(missing)} part(s) of the paper were not read for understanding and presentation: the budget ran out ({_span(missing)}).')

    def _read_job(self, key, sid, a, b, pdf):
        def run():
            self.emit('notice', f'Reading lines {a}-{b}')
            prompt = P.READ_POLICY + ('\n' + P.PDF_READING if pdf else '') + '\n\nPART OF THE MANUSCRIPT:\n' + self._numbered(sid, a, b)
            result = self._ask('read', prompt, cap=3000, schema=P.READ_SCHEMA, key=key)
            value = result['value'] or {}
            found, dropped = [], 0
            for item in value.get('comments') or []:
                if not isinstance(item, dict):
                    continue
                comment = {'start_line': _int(item.get('start_line')), 'end_line': _int(item.get('end_line')),
                           'quote': str(item.get('quote') or '')[:200], 'kind': str(item.get('kind') or 'clarity')[:20],
                           'comment': str(item.get('comment') or '')[:400], 'suggestion': str(item.get('suggestion') or '')[:300]}
                comment['end_line'] = max(comment['end_line'], comment['start_line'])
                comment['located'] = self._locate(comment, sid, set(range(a, b + 1)))
                if comment['located'] == 'no' or not comment['comment'] or re.search(
                        r'(?i)\bno (?:change|correction|issue|error)s? (?:is )?(?:needed|required|found)', comment['comment'] + ' ' + comment['suggestion']):
                    continue  # a non-comment ("no change needed") is not a referee remark
                if pdf and not _prose_comment(comment):
                    dropped += 1
                    continue
                found.append(comment)
            with self._mutex:
                self.state['reading'][key] = {'start_line': a, 'end_line': b, 'summary': str(value.get('summary') or '')[:1500],
                                              'usable': result['value'] is not None, 'dropped': dropped}
                self.state['presentation'] += found
                self._save()
        return run

    # -- writing --------------------------------------------------------------------------------------

    def _comments(self):
        """Major and minor comments and open questions, from settled findings, in a fixed order."""
        main = set(self._main_ids())
        major, minor, questions, rejected, verdicts = [], [], [], [], {}
        for uid in self.state['plan_units']:
            if not any(c.get('unit') == uid for c in self.state['checks'].values()):
                continue
            verdict, clusters = self._unit_verdict(uid)
            verdicts[uid] = verdict
            for cluster in clusters:
                if cluster['unlocated'] and cluster['status'] != 'confirmed':
                    continue
                entry = dict(cluster, unit_name=self._unit(uid)['name'], main=uid in main)
                if cluster['status'] == 'rejected':
                    rejected.append(entry)
                elif cluster['status'] in ('undecided', 'uncertainty'):
                    questions.append(entry)
                elif cluster.get('severity') == 'major':
                    major.append(entry)
                else:
                    minor.append(entry)
        order = lambda c: (c['status'] != 'confirmed', not c['main'], -RANK[c['kind']], self.state['plan_units'].index(c['unit']), c['start_line'])
        return sorted(major, key=order), sorted(minor, key=order), sorted(questions, key=order), rejected, verdicts

    def _write(self):
        major, minor, questions, _, verdicts = self._comments()
        scope = self.state['scope'] or {}
        kinds = {}
        for c in self.state['presentation']:
            kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
        findings = {'overview': scope.get('overview'), 'field': scope.get('field'), 'claimed_contribution': scope.get('contribution'),
                    'proofs_checked': [{'name': self._unit(u)['name'], 'main': u in set(self._main_ids()),
                                        'verdict': v.replace('_', ' ')} for u, v in verdicts.items()],
                    'proofs_not_checked': [self._unit(u)['name'] for u in self.state['unchecked']],
                    'major': [{'n': n, 'where': c['unit_name'], 'kind': KIND_LABEL[c['kind']], 'status': c['status'],
                               'issue': c['evidence'][:400]} for n, c in enumerate(major, 1)],
                    'minor': [{'n': n, 'where': c['unit_name'], 'issue': c['evidence'][:200]} for n, c in enumerate(minor, 1)],
                    'typos_and_presentation_comments': kinds,
                    'parts_not_read': len([r for r in self.state.get('read_parts') or [] if f'L{r[0]}' not in self.state.get('reading', {})]),
                    'questions': len(questions),
                    'cited_results_checked': [{'reference': c['entry'][:120], 'place': c['place'], 'level': c['level']} for c in self.state['cited_checks']],
                    'novelty_search': (self.state.get('novelty') or {}).get('assessment') or ('not done (offline or low effort)')}
        if not self.CHECK_PROOFS:
            findings['proofs'] = 'not checked in this review: it covers understanding and presentation only'
            for key in ('proofs_checked', 'proofs_not_checked', 'major', 'minor', 'questions'):
                findings.pop(key, None)
        prompt = (P.WRITE_POLICY + ('' if self.CHECK_PROOFS else '\n' + P.NO_PROOFS_NOTE)
                  + '\n\nFINDINGS:\n' + json.dumps(findings, ensure_ascii=False, indent=1))
        if self._tokens_left() < 800 or not self._time_left(.98):
            self._warn('No budget was left to write the prose parts; the report lists the findings only.')
            return
        try:
            with self._timeout():
                result = self._ask('write', prompt, cap=4000, schema=P.WRITE_SCHEMA, key='write')
        except ResearchBudget as exc:
            self._warn(f'The prose parts were not written: {exc}')
            return
        value = result['value']
        if value is None:
            self._warn('The prose answer was not usable JSON; the report lists the findings only.')
            return
        self.state['write'] = {k: str(value.get(k) or '')[:3000] for k in P.WRITE_SCHEMA['properties']}
        if self.state['write']['recommendation'] not in P.RECOMMENDATIONS:
            self.state['write']['recommendation'] = 'no_recommendation'

    def _provisional(self):
        if not self.CHECK_PROOFS:
            return [], []
        main = set(self._main_ids())
        checked = {c['unit'] for c in self.state['checks'].values() if c.get('usable')}
        missing = [self._unit(u)['name'] for u in main if self._unit(u) and self._unit(u)['proofs'] and u not in checked]
        no_proof = [self._unit(u)['name'] for u in main if self._unit(u) and not self._unit(u)['proofs']]
        return missing, no_proof

    def _complete(self):
        if not self.CHECK_PROOFS:
            return bool(self.state.get('write')) and all(f'L{r[0]}' in self.state['reading'] for r in self.state['read_parts'])
        missing, _ = self._provisional()
        return bool(self.state.get('write')) and not self.state['unchecked'] and not missing

    def _render(self):
        s = self.state
        if not s.get('map'):
            return ['*The manuscript is not mapped yet.*'], []
        sid = s['sources'][0]['id']
        write = s.get('write') or {}
        major, minor, questions, rejected, verdicts = self._comments()
        missing, no_proof = self._provisional()
        scope = s.get('scope') or {}
        body = ['## Summary', '', write.get('summary') or scope.get('contribution') or '*Not written (see the controller notes).*', '']
        # First what the paper does, then its writing, then the mathematics of its proofs.
        if scope.get('overview') or s.get('reading'):
            overview = scope.get('overview') or '*No overview was written.*'
            if write.get('summary') and _fold(write['summary'])[:150] == _fold(overview)[:150]:
                overview = '*As in the summary above.*'  # the model sometimes repeats its overview as the summary
            body += ['## The paper in brief', '', overview, '']
            parts = [s['reading'][f'L{a}'] for a, b in s.get('read_parts') or [] if s['reading'].get(f'L{a}', {}).get('summary')]
            if parts:
                body += ['### Part by part', '']
                body += [f'- {self._loc(sid, r["start_line"], r["end_line"])} {r["summary"]}' for r in parts] + ['']
        typos = sorted(s['presentation'], key=lambda c: (c['start_line'], c['end_line']))
        body += ['## Typos and presentation', '']
        if typos:
            for k, c in enumerate(typos, 1):
                text = (f'{k}. {self._loc(sid, c["start_line"], c["end_line"])} *{c.get("kind", "clarity").replace("_", " ")}* — '
                        + (f'"{c["quote"]}": ' if c['quote'] else '') + c['comment'])
                body += [text + (f' *Suggestion:* {c["suggestion"]}' if c.get('suggestion') else ''), '']
        else:
            body += ['None found.' if s.get('reading') else 'The paper was not read for presentation.', '']
        dropped = sum(r.get('dropped', 0) for r in s.get('reading', {}).values())
        if self._source(sid)['path'].lower().endswith('.pdf'):
            body += [f'The manuscript was read from a PDF extraction, whose formulas are garbled: only comments on words and '
                     f'sentences are kept{f" ({dropped} on symbols or formulas were left out)" if dropped else ""}. '
                     'Review the TeX source for typos in formulas and notation.', '']
        if not self.CHECK_PROOFS:
            return self._render_summary(body, write, sid)
        body += ['## Assessment', '', '### Significance and novelty', '', write.get('significance') or '—', '',
                '### Correctness', '', write.get('correctness') or '—', '', '### Presentation', '', write.get('presentation') or '—']
        body += ['', '## Major comments', '']
        if major:
            for n, cluster in enumerate(major, 1):
                body += _item(f'{n}. *{cluster["unit_name"]}.* ', self._issue_lines(cluster, sid), '   ')
        else:
            body += ['None.' if verdicts else 'No proof was checked.', '']
        body += ['## Minor comments', '']
        n = 0
        for cluster in minor:
            n += 1
            body += _item(f'{n}. *{cluster["unit_name"]}.* ', self._issue_lines(cluster, sid), '   ')
        if not n:
            body += ['None in the proofs checked (typos and presentation are listed above).', '']
        if questions:
            body += ['## Questions to the authors', '']
            for k, cluster in enumerate(questions, 1):
                body += _item(f'{k}. *{cluster["unit_name"]}.* ', self._issue_lines(cluster, sid, show_passes=False), '   ')
        recommendation = P.RECOMMENDATIONS.get(write.get('recommendation'), 'No recommendation')
        body += ['## Recommendation', '', f'**{recommendation}**' + (' (provisional)' if (missing or s['unchecked']) and write else '')]
        if missing or s['unchecked']:
            body += ['', 'Provisional: ' + ('the proofs of ' + ', '.join(missing) + ' were not checked' if missing else
                                            f'{len(s["unchecked"])} planned proof(s) were not checked') + '; see the scope below.']
        body += ['', write.get('reasons') or '']
        body += ['', '## Confidential comments to the editor', '', write.get('confidential') or '—']
        # The harness appendix: what was and was not examined.
        appendix = ['## Scope of this review', '']
        units = {u['id']: u for u in self._units()}
        checkable = [u for u in self._units() if u['proofs']]
        checked = [u for u in checkable if u['id'] in verdicts]
        read = [r for r in s.get('read_parts') or [] if f'L{r[0]}' in s.get('reading', {})]
        appendix += [f'Read for understanding and presentation: {sum(b - a + 1 for a, b in read)} of '
                     f'{sum(b - a + 1 for a, b in s.get("read_parts") or [])} lines before the bibliography ({len(read)} parts).', '']
        appendix += [f'Proofs checked: {len(checked)} of {len(checkable)} found in the manuscript map '
                     f'({len(s["plan_units"])} planned at this effort).', '',
                     '| Result | Proof lines | Passes | Outcome |', '|---|---|---|---|']
        for uid in s['plan_units'] + [u['id'] for u in checkable if u['id'] not in s['plan_units']]:
            u = units.get(uid)
            if not u:
                continue
            passes = self._passes_on(uid)
            outcome = verdicts.get(uid, 'not checked').replace('_', ' ')
            appendix.append(f'| {u["name"]}{" (main)" if uid in set(self._main_ids()) else ""} | {_span(u["proofs"])} | {passes} | {outcome} |')
        if no_proof:
            appendix += ['', 'Main results without a proof in the map (possibly proved elsewhere or missed by the map): ' + ', '.join(no_proof) + '.']
        scope = s.get('scope') or {}
        if scope.get('notation'):
            appendix += ['', 'Setting and notation given to every check: ' + _span(scope['notation']) + '.']
        if rejected:
            appendix += ['', '### Objections raised and rejected by a re-check', '']
            for cluster in rejected:
                confirm = s['confirms'].get(cluster['key'], {})
                appendix += [f'- *{cluster["unit_name"]}* {self._loc(sid, cluster["start_line"], cluster["end_line"])}: {cluster["evidence"][:300]} — *rejected:* {confirm.get("argument", "")[:400]}']
        if s['cited_checks']:
            appendix += ['', '### Cited results checked', '']
            for c in s['cited_checks']:
                found = f' (found at {c["found_place"]})' if c.get('found_place') and c['found_place'] != c['place'] else ''
                appendix += [f'- [{c["key"]}] {c["entry"][:160]} — {c["place"]}: **{c["level"].replace("_", " ")}**{found}. Used for: {c["used_for"]}']
        if s.get('novelty') is not None or s['search_results']:
            appendix += ['', '### Literature searched (titles and abstracts only)', '']
            appendix += [f'Queries: ' + '; '.join(f'"{q}"' for q in dict.fromkeys(r['query'] for r in s['search_results']))]
            for n, r in enumerate(s['search_results'], 1):
                relation = next((x['relation'] for x in (s.get('novelty') or {}).get('related', []) if x['n'] == n), '')
                link = f'[{r["title"]}]({r["url"]})' if r['url'].startswith('http') else r['title']
                appendix += [f'{n}. {link} ({r["authors"]}, {r["year"]})' + (f' — {relation}' if relation else '')]
            if (s.get('novelty') or {}).get('assessment'):
                appendix += ['', s['novelty']['assessment']]
            appendix += ['', 'Search hits do not establish or refute novelty.']
        return body, appendix


class SummaryReviewRunner(JournalRunner):
    """The review: understand the whole paper and list its typos and presentation problems; no proof checks."""
    VARIANT = 'review'
    PIPELINE = PIPELINES['review']
    CHECK_PROOFS = False

    def _render_summary(self, body, write, sid):
        s = self.state
        body += ['## Assessment', '', '### Significance and novelty', '', write.get('significance') or '—', '',
                 '### Presentation', '', write.get('presentation') or '—', '',
                 '### Correctness', '', 'The proofs were not checked in this review; a detailed review checks them.', '']
        recommendation = P.RECOMMENDATIONS.get(write.get('recommendation'), 'No recommendation')
        body += ['## Recommendation', '', f'**{recommendation}** (the proofs were not checked)', '', write.get('reasons') or '',
                 '', '## Confidential comments to the editor', '', write.get('confidential') or '—']
        read = [r for r in s.get('read_parts') or [] if f'L{r[0]}' in s.get('reading', {})]
        appendix = ['## Scope of this review', '',
                    f'Read for understanding and presentation: {sum(b - a + 1 for a, b in read)} of '
                    f'{sum(b - a + 1 for a, b in s.get("read_parts") or [])} lines before the bibliography ({len(read)} parts). '
                    'Proofs were not checked.']
        if s.get('novelty') is not None or s['search_results']:
            appendix += ['', '### Literature searched (titles and abstracts only)', '']
            for n, r in enumerate(s['search_results'], 1):
                relation = next((x['relation'] for x in (s.get('novelty') or {}).get('related', []) if x['n'] == n), '')
                link = f'[{r["title"]}]({r["url"]})' if r['url'].startswith('http') else r['title']
                appendix += [f'{n}. {link} ({r["authors"]}, {r["year"]})' + (f' — {relation}' if relation else '')]
            if (s.get('novelty') or {}).get('assessment'):
                appendix += ['', s['novelty']['assessment']]
            appendix += ['', 'Search hits do not establish or refute novelty.']
        return body, appendix
