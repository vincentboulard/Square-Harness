"""Bounded natural-language proof search with durable, evidence-linked checkpoints.

The controller enforces bookkeeping, not mathematical truth. Reviewed claims
and complete candidates are model judgments, never formal certificates.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .agent import AgentError
from .ledger import ProofStore, LedgerError
from .prompts import SYSTEM
from .tools import schema


class ProofBudget(AgentError):
    pass


class ProofContext(AgentError):
    pass


from .proof_policy import (
    REVIEW_SCHEMA, REVIEW_BATCH_SCHEMA, CRITIC_SCHEMA, PLAN_SCHEMA, AUDIT_SCHEMA,
    SOLVER_POLICY, REVIEW_POLICY, CRITIC_POLICY, PLAN_POLICY, CHECKPOINT_POLICY, AUDIT_POLICY,
)


def _normal(text):
    # Mathematical symbols are case sensitive; do not conflate A with a.
    return re.sub(r'\s+', ' ', text).strip()


def _validate(value, spec, path='response'):
    """Small strict validator for our fixed schemas; never trust format alone."""
    kind = spec['type']
    if kind == 'object':
        if not isinstance(value, dict) or set(value) != set(spec['properties']):
            raise ValueError(f'{path}: incorrect JSON fields')
        for key, child in spec['properties'].items():
            _validate(value[key], child, path + '.' + key)
    elif kind == 'array':
        if not isinstance(value, list) or len(value) > spec['maxItems']:
            raise ValueError(f'{path}: invalid list')
        for item in value:
            _validate(item, spec['items'], path)
    elif kind == 'boolean':
        if type(value) is not bool:
            raise ValueError(f'{path}: expected boolean')
    elif kind == 'string':
        if not isinstance(value, str) or len(value) > spec.get('maxLength', 10000):
            raise ValueError(f'{path}: invalid string')
        if 'enum' in spec and value not in spec['enum']:
            raise ValueError(f'{path}: invalid value')


class ProofRunner:
    def __init__(self, agent, emit=lambda kind, value: None):
        self.agent, self.emit = agent, emit
        self.store = None
        self._clock = None
        self._seconds_before = 0

    @property
    def state(self):
        return self.store.state

    def start(self, goal, *, max_rounds=10, max_tokens=60000,
              max_seconds=1800, source_files=(), max_predict=8192,
              allow_literature=False):
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 60000:
            raise ValueError('Provide a nonempty proof goal, at most 60000 characters')
        if type(max_rounds) is not int or not 1 <= max_rounds <= 100:
            raise ValueError('Proof rounds must be between 1 and 100')
        if type(max_tokens) is not int or max_tokens < 512:
            raise ValueError('Proof token budget must be at least 512')
        if type(max_predict) is not int or max_predict < max(128, self.agent.predict):
            raise ValueError('proof-max-predict must be at least 128 and at least --predict')
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Proof time budget must be finite and positive')
        names = list(source_files)
        if not names:
            names = re.findall(r'(?<![\w/])[\w./-]+\.(?:tex|md|txt)\b', goal)
        sources = []
        for name in dict.fromkeys(str(n) for n in names):
            path = Path(name)
            if path.is_absolute():
                path = path.resolve().relative_to(self.agent.workspace.root)
            p = self.agent.workspace.path(str(path))
            content = self.agent.workspace.text(p)
            sources.append({'path': os.path.normpath(str(path)),
                            'resolved_path': str(p.relative_to(self.agent.workspace.root)), 'content': content,
                            'sha256': hashlib.sha256(content.encode()).hexdigest()})
        settings = {'max_rounds': max_rounds, 'max_tokens': max_tokens,
                    'max_seconds': max_seconds, 'model': self.agent.model,
                    'ctx': self.agent.ctx, 'predict': self.agent.predict,
                    'max_predict': max_predict, 'policy_version': 2,
                    'harness_version': '0.4.0',
                    'allow_literature': bool(allow_literature),
                    'policy_sha256': hashlib.sha256(Path(__file__).with_name('proof_policy.py').read_bytes()).hexdigest(),
                    'think': self.agent.think,
                    'host': getattr(self.agent.client, 'host', None)}
        self.store = ProofStore.create(self.agent.workspace.root, goal.strip(), settings, sources)
        library = getattr(self.agent.workspace, 'literature', None)
        if allow_literature and library is not None:
            self.state['literature'] = library.snapshot()
            self.store.save()
        self.emit('notice', f'Proof {self.state["id"]}: up to {max_rounds} work rounds; checkpoints in {self.store.directory}')
        return self._run()

    def resume(self, proof_id):
        self.store = ProofStore.load(self.agent.workspace.root, proof_id)
        settings = self.state['settings']
        if settings.get('host') != getattr(self.agent.client, 'host', None):
            raise ValueError('Resume with the original --host; saved proof data will not be sent to another server implicitly')
        for key in ('model', 'ctx', 'predict', 'think'):
            setattr(self.agent, key, settings[key])
        self.emit('notice', f'Resuming proof {proof_id}; saved budgets and model settings retained')
        for src in self.state['sources']:
            try:
                current = self.agent.workspace.text(self.agent.workspace.path(src['path']))
            except (OSError, ValueError):
                current = None
            if current != src['content']:
                self.emit('notice', f'{src["path"]} changed or is unavailable; this run keeps its original saved statement')
        return self._run()

    def _elapsed(self):
        return self._seconds_before + (time.monotonic() - self._clock if self._clock is not None else 0)

    def _save(self):
        if self._clock is not None:
            self.state['seconds_used'] = self._elapsed()
            self.state['active_checkpoint_wall'] = time.time()
        library = getattr(self.agent.workspace, 'literature', None)
        if self.state['settings'].get('allow_literature') and library is not None:
            self.state['literature'] = library.snapshot()
        self.store.save()

    def _run(self):
        with self.store.lock():
            # Refresh under the lock, so two simultaneous resumes cannot use
            # the same budget snapshot even if load occurred before locking.
            self.store = ProofStore.load(self.agent.workspace.root, self.state['id'])
            if self.state.get('recovery_notice'):
                self.state['status'] = 'needs_recovery'
                self.state['stop_reason'] = ('A backup was recovered; the latest budget reservation may be missing. '
                    'Automatic inference is disabled for this job. Inspect its retained artifacts and start a new explicitly budgeted goal.')
                self.store.save()
                return self._result()
            if self.state['status'] in {'candidate_complete', 'budget_exhausted', 'stalled', 'needs_recovery'}:
                return self._result()
            library = getattr(self.agent.workspace, 'literature', None)
            if self.state['settings'].get('allow_literature') and library is not None:
                if self.state.get('literature'):
                    saved = self.state['literature']
                    library.reset_budget(max_requests=saved['max_requests'], max_chars=saved['max_chars'])
                    library.restore(self.state['literature'])
                library.on_budget_change = self._save
            if self.state['status'] == 'running' and self.state.get('active_checkpoint_wall'):
                # An unclean exit has no trustworthy stop timestamp. Count the
                # interval until resume conservatively; normal paused jobs do
                # not charge offline time.
                self.state['seconds_used'] += max(0, time.time() - self.state['active_checkpoint_wall'])
            self._seconds_before = self.state['seconds_used']
            self._clock = time.monotonic()
            self.state['status'] = 'running'
            if self.state['settings'].get('allow_literature') and library is not None:
                library.deadline = time.monotonic() + max(0, self.state['settings']['max_seconds'] - self._elapsed())
            self.state['settings'].setdefault('max_predict', max(8192, self.agent.predict))
            self.state.setdefault('truncation_streak', 0)
            self.state.setdefault('strategies', [])
            if self.state.get('stop_reason'):
                self.state['previous_stop_reason'] = self.state.pop('stop_reason')
            try:
                self._recover_interrupted_call()
                while True:
                    if self._elapsed() >= self.state['settings']['max_seconds']:
                        raise ProofBudget('Elapsed-time budget exhausted')
                    if self.state['pending'] is None:
                        if self.state['rounds_started'] >= self.state['settings']['max_rounds']:
                            raise ProofBudget('Work-round budget exhausted')
                        self.state['rounds_started'] += 1
                        n = self.state['rounds_started']
                        assembling = self.state.pop('assemble_next', False)
                        fresh = not assembling and n > 1 and (self.state['stagnant_rounds'] >= 2 or n % 4 == 0)
                        direct = self.agent.think and (assembling or self.state['truncation_streak'] >= 2 or fresh and n % 2 == 0)
                        self.state['pending'] = {'index': n, 'phase': 'plan' if fresh else 'solve',
                                                 'fresh': fresh, 'assembling': assembling, 'think': self.agent.think and not direct,
                                                 'task': self.state['next_task']}
                        self._save()
                    p = self.state['pending']
                    self.emit('notice', f'Proof round {p["index"]}/{self.state["settings"]["max_rounds"]}: {p["phase"]}')
                    if p['phase'] == 'plan':
                        plan = self._plan()
                        p['task'] = plan['task'] + '\nRequired deliverable: ' + plan['deliverable']
                        p['plan'] = plan
                        self.state['strategies'].append({'round': p['index'], **plan})
                        p['phase'] = 'solve'
                        self._save()
                        self.emit('notice', 'New approach: ' + plan['approach'])
                    if p['phase'] == 'solve':
                        result = self._solve(p)
                        p['draft'] = self.store.write_artifact('candidate', json.dumps(result, ensure_ascii=False))
                        p['solver_truncated'] = not result['complete'] or not result['text'].strip()
                        p['phase'] = 'checkpoint' if p['solver_truncated'] else 'critic'
                        self._save()
                    if p['phase'] == 'checkpoint':
                        raw = json.loads(self.store.read_artifact(p['draft']))
                        checkpoint = self._checkpoint(raw)
                        p['raw_draft'] = p['draft']
                        p['draft'] = self.store.write_artifact('checkpoint', json.dumps(checkpoint, ensure_ascii=False))
                        p['phase'] = 'critic'
                        self._save()
                    if p['phase'] == 'critic':
                        draft = json.loads(self.store.read_artifact(p['draft']))
                        p['critique'] = self._critic(draft)
                        p['phase'] = 'review'
                        self._save()
                    if p['phase'] == 'review':
                        draft = json.loads(self.store.read_artifact(p['draft']))
                        # Legacy interrupted jobs may reach review without the
                        # new blind critique; acquire it before recording.
                        if 'critique' not in p:
                            p['critique'] = self._critic(draft)
                            self._save()
                        review = self._review(draft, p['critique'])
                        snapshot = copy.deepcopy(self.state)
                        try:
                            claims = self._record_batch(review, p, draft)
                            p['claim_ids'] = [c['id'] for c in claims]
                            p['review_artifact'] = review.get('_artifact')
                            p['strategy_summary'] = review['strategy_summary']
                            approved = claims and all(c['status'] == 'reviewed' for c in claims)
                            p['phase'] = 'audit' if (review['complete_candidate'] and approved
                                and p['critique']['complete_candidate'] and not p['critique']['first_invalid_step'].strip()
                                and not p['critique']['missing_work'].strip() and draft['complete']) else 'finish'
                        except BaseException:
                            self.store.state = snapshot
                            raise
                        self._save()
                        # Notices can be interrupted. Emit only AFTER the batch
                        # and the next phase are durably committed together.
                        for claim in claims:
                            self.emit('notice', f'{claim["id"]}: {claim["status"]}. {claim["objection"] or claim["statement"]}')
                    if p['phase'] == 'audit':
                        draft = json.loads(self.store.read_artifact(p['draft']))
                        audit = self._audit(draft)
                        self.state['final_audit'] = {'round': p['index'], 'candidate': p['draft'], **audit}
                        if audit['verdict'] == 'complete':
                            self.state['status'] = 'candidate_complete'
                        else:
                            self.state['next_task'] = audit['next_task'] or audit['objection'] or 'Repair the unresolved whole-proof audit'
                            for claim in self.state['claims']:
                                if claim['id'] in p.get('claim_ids', [p.get('claim_id')]):
                                    claim['whole_proof_objection'] = audit['objection'] or audit['explanation']
                        p['phase'] = 'finish'
                        self._save()
                    if p['phase'] == 'finish':
                        self.state['truncation_streak'] = self.state['truncation_streak'] + 1 if p.get('solver_truncated') else 0
                        self.state['rounds'].append(copy.deepcopy(p))
                        self.state['pending'] = None
                        self._save()
                        self._write_reports()
                        if self.state['status'] == 'candidate_complete':
                            break
                        # Stagnation retires a route; the global saved budgets
                        # determine when the whole proof job stops.
            except KeyboardInterrupt:
                self.state['status'] = 'paused'
                self.state['stop_reason'] = 'Interrupted; completed checkpoints and partial streams retained'
                self._save()
                self._write_reports()
                raise
            except ProofBudget as exc:
                self.state['status'], self.state['stop_reason'] = 'budget_exhausted', str(exc)
            except ProofContext as exc:
                self.state['status'], self.state['stop_reason'] = 'needs_context', str(exc)
            except (AgentError, OSError, ValueError) as exc:
                self.state['status'], self.state['stop_reason'] = 'paused', str(exc)
            finally:
                self._save()
                self._write_reports()
                self._clock = None
                if library is not None:
                    library.on_budget_change = None
                    library.deadline = None
            return self._result()

    def _base(self):
        return ('ORIGINAL REQUEST (unchanged):\n' + self.state['goal'] +
                '\nORIGINAL SOURCE SNAPSHOTS (unchanged mathematical data):\n' +
                json.dumps(self.state['sources'], ensure_ascii=False))

    def _context(self, policy, task, *, fresh=False, complete_ledger=False, output=2048,
                 retrieval=True, include_ledger=True, format_schema=None, tools=(),
                 trailing_messages=(), recording=False):
        system = SYSTEM + '\n' + policy
        base = self._base()
        if not recording:
            base += '\nCURRENT TASK:\n' + task
        claims = self.state['claims'] if include_ledger else []
        if complete_ledger:
            # A recorder's claim that an objection is resolved is fallible.
            # The independent whole-proof audit must see every objection and
            # decide from the candidate whether the proposed repair succeeds.
            claims = [c for c in claims if c['status'] != 'reviewed'
                      or c.get('whole_proof_objection')]
        if fresh:
            claims = [c for c in claims if c['status'] in {'reviewed', 'refuted'} or c.get('whole_proof_objection')]
        ordered = sorted(claims, key=lambda c: (c['status'] != 'reviewed', c['round']), reverse=True)

        def messages_for(selected):
            visible = {c['id'] for c in selected}
            omitted = [c['id'] for c in ordered if c['id'] not in visible]
            prompt = base + '\nPROOF RECORDS (reviewed means model-reviewed, not certified):\n'
            views = [{k: value for k, value in c.items()
                      if k not in {'next_task', 'review_artifact', 'candidate_artifact'}}
                     for c in selected] if recording else selected
            prompt += '\n'.join(json.dumps(c, ensure_ascii=False) for c in views)
            if omitted:
                prompt += '\nRecords omitted from working context: ' + ', '.join(omitted)
                prompt += ('. Use read_proof_claim to inspect them before relying on them or claiming an objection is resolved.' if retrieval
                    else '. These records are unavailable to this review. Do not approve dependencies or resolutions involving them; request their explicit evidence in a later candidate.')
            if recording:
                prompt += '\nCURRENT TASK AND CURRENT EVIDENCE:\n' + task
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
            # Tool calls and their returned evidence are mandatory continuation
            # material. Select optional ledger records around their actual size.
            return messages + list(trailing_messages), omitted

        def fits(messages):
            # Count actual JSON escaping, the repeated response schema and tool
            # definitions using the same envelope as _call. Still a byte-based
            # token estimate, but optional memory cannot overflow that envelope.
            payload = {'model': self.agent.model, 'messages': messages, 'stream': True,
                       'think': False, 'options': {'num_ctx': self.agent.ctx,
                       'num_predict': output, 'temperature': 0.6}}
            if format_schema:
                payload['format'] = format_schema
            if tools:
                payload['tools'] = list(tools)
            return len(json.dumps(payload, ensure_ascii=False).encode()) <= (self.agent.ctx - output - 256) * 3

        selected = []
        messages, omitted = messages_for(selected)
        if not fits(messages):
            raise ProofContext('Original statement and current proof material do not fit the context estimate. Nothing was silently truncated; use a larger context or a smaller standalone statement')
        for claim in ordered:
            candidate, _ = messages_for(selected + [claim])
            if fits(candidate):
                selected.append(claim)
        messages, omitted = messages_for(selected)
        if omitted and complete_ledger:
            raise ProofContext('Whole-proof audit cannot fit all unresolved objections and the complete candidate; no success verdict issued')
        if omitted:
            self.emit('notice', f'Context rebuilt: {len(selected)} exact proof records loaded, {len(omitted)} retained on disk')
        self._visible_claims = [c['id'] for c in selected]
        return messages

    def _tools(self):
        # Writing a manuscript is never necessary to advance a proof ledger.
        # The controller saves proof artifacts automatically; existing explicit
        # write_file approval remains available in the ordinary chat modes.
        tools = [s for s in self.agent.workspace.schemas() if s['function']['name'] != 'write_file']
        library = getattr(self.agent.workspace, 'literature', None)
        if library is not None and not self.state['settings'].get('allow_literature', False):
            research_names = {s['function']['name'] for s in library.schemas()}
            tools = [s for s in tools if s['function']['name'] not in research_names]
        tools.append(schema('read_proof_claim', 'Read an exact saved claim with argument, status, dependencies and objections.',
                            {'claim_id': {'type': 'string'}, 'offset': {'type': 'integer'}}, ['claim_id']))
        tools.append(schema('read_proof_artifact', 'Read up to 6000 characters of a prior artifact. Use next_offset for more.',
                            {'filename': {'type': 'string'}, 'offset': {'type': 'integer'}}, ['filename']))
        return tools

    @staticmethod
    def _excerpt(text, offset=0):
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer')
        end = min(offset + 6000, len(text))
        return json.dumps({'offset': offset, 'total_characters': len(text),
                           'next_offset': end if end < len(text) else None,
                           'content': text[offset:end]}, ensure_ascii=False)

    def _tool(self, name, args):
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be an object')
        if name == 'read_proof_claim':
            claim = next((c for c in self.state['claims'] if c['id'] == args.get('claim_id')), None)
            return self._excerpt(json.dumps(claim, ensure_ascii=False), args.get('offset', 0)) if claim else 'Unknown proof claim'
        if name == 'read_proof_artifact':
            return self._excerpt(self.store.read_artifact(args.get('filename', '')), args.get('offset', 0))
        if name not in {s['function']['name'] for s in self._tools()}:
            return 'Unknown or disabled proof tool'
        if name == 'read_file' and 'path' in args:
            requested = Path(args['path'])
            if requested.is_absolute():
                requested = requested.relative_to(self.agent.workspace.root)
            logical = os.path.normpath(str(requested))
            # Match the original spelling first, including ./ and ../ aliases.
            # This also keeps a pinned source readable after deletion.
            src = next((s for s in self.state['sources'] if s['path'] == logical), None)
            if src is None:
                resolved = str(self.agent.workspace.path(str(requested)).relative_to(self.agent.workspace.root))
                src = next((s for s in self.state['sources'] if s.get('resolved_path', s['path']) == resolved), None)
            if src:
                start, end = args.get('start_line', 1), args.get('end_line', 100)
                if type(start) is not int or type(end) is not int or not 1 <= start <= end:
                    raise ValueError('Invalid line range')
                lines = src['content'].splitlines()
                return 'PINNED ORIGINAL SOURCE\n' + '\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1, min(end, start+199, len(lines))))
        return self.agent.workspace.execute(name, args)

    def _solve(self, pending):
        if pending.get('partial'):
            recovered = self._read_stream(pending.pop('partial'))
            recovered['complete'] = recovered['complete'] and not recovered['calls']
            recovered['interruption'] = 'Recovered saved stream. Incomplete output is not a proof; requested tools are not replayed.'
            return recovered
        task = pending['task']
        if pending['fresh']:
            task = ('Choose a materially different approach from recent attempts. Preserve valid objections but do not inherit a favoured unproved strategy. Establish one new intermediate claim.\n' + task)
        streak = self.state.get('truncation_streak', 0)
        cap = min(self.agent.predict * (2 ** min(streak, 2)),
                  self.state['settings'].get('max_predict', max(8192, self.agent.predict)), self.agent.ctx // 2)
        pending['output_allowance'] = cap
        thinking = pending.get('think', self.agent.think)
        self.emit('notice', f'Solver allowance: {cap} tokens; thinking {"on" if thinking else "off"}')
        tool_history = []
        for tool_round in range(3):
            tools = self._tools() if tool_round < 2 else []
            messages = self._context(SOLVER_POLICY, task, fresh=pending['fresh'],
                                     output=cap, tools=tools, trailing_messages=tool_history)
            result = self._call('solver', messages, cap, think=thinking, tools=tools)
            if not result['calls'] or not result['complete']:
                return result
            if not tools or len(result['calls']) > 8:
                result['complete'] = False
                result['text'] += '\nTool budget exhausted; no complete proof was obtained.'
                return result
            message = {'role': 'assistant', 'content': result['text'], 'tool_calls': result['calls']}
            tool_history.append(message)
            for call in result['calls']:
                fn = call.get('function', {})
                name, args = fn.get('name', ''), fn.get('arguments', {})
                self.emit('tool', name + ' ' + json.dumps(args, ensure_ascii=False)[:240])
                try:
                    value = str(self._tool(name, args))
                except (ValueError, OSError, TypeError) as exc:
                    value = 'Tool error: ' + str(exc)
                if len(value) > 8000 and name not in {'read_proof_claim', 'read_proof_artifact'}:
                    value = value[:8000] + '\n[TRUNCATED; retrieve a smaller excerpt]'
                evidence = self.store.write_artifact('tool-result', json.dumps({'name': name, 'arguments': args, 'result': value}, ensure_ascii=False))
                self.state['pending'].setdefault('tools', []).append(evidence)
                self._save()
                tool_history.append({'role': 'tool', 'tool_name': name, 'content': value})
                self.emit('result', value[:200])
        return result

    def _read_stream(self, filename):
        result = {'text': '', 'thinking': '', 'calls': [], 'complete': False, 'stats': {}, 'stream': filename}
        for line in self.store.read_artifact(filename).splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break  # a killed process may leave a partial final JSON line
            msg = event.get('message', {})
            result['text'] += msg.get('content') or ''
            result['thinking'] += msg.get('thinking') or ''
            result['calls'].extend(msg.get('tool_calls') or [])
            if event.get('done'):
                result['stats'] = {k: event[k] for k in ('eval_count', 'prompt_eval_count', 'done_reason') if k in event}
                result['complete'] = event.get('done_reason') != 'length'
        return result

    def _recover_interrupted_call(self):
        for call in self.state['calls']:
            if call['status'] == 'running':
                # Its token reservation remains charged. Never replay tools
                # from a partial stream, nor reuse an uncommitted verdict.
                call['status'] = 'interrupted'
                if self.state['pending'] and self.state['pending']['phase'] == 'solve' and call['role'] == 'solver':
                    self.state['pending']['partial'] = call['stream']
        p = self.state['pending']
        if p and p['phase'] == 'solve':
            for call in reversed(self.state['calls']):
                if call['round'] == p['index'] and call['role'] == 'solver':
                    p['partial'] = call['stream']
                    break
        self._save()

    def _call(self, role, messages, cap, *, think=False, tools=(), format_schema=None):
        remaining = self.state['settings']['max_tokens'] - self.state['tokens_charged']
        if role == 'solver':
            reserve = min(4096, self.state['settings']['max_tokens'] // 3)
            remaining -= reserve
        cap = min(cap, remaining)
        if cap < 128:
            raise ProofBudget('Generated-token budget exhausted (includes thinking, reviews and recovery calls)')
        remaining_seconds = self.state['settings']['max_seconds'] - self._elapsed()
        if remaining_seconds <= 0:
            raise ProofBudget('Elapsed-time budget exhausted')
        payload = {'model': self.agent.model, 'messages': messages, 'stream': True,
                   'think': think, 'options': {'num_ctx': self.agent.ctx, 'num_predict': cap,
                                              'temperature': 0.6 if role == 'solver' else 0}}
        if tools:
            payload['tools'] = list(tools)
        if format_schema:
            payload['format'] = format_schema
        if len(json.dumps(payload, ensure_ascii=False).encode()) > (self.agent.ctx - cap - 256) * 3:
            raise ProofContext('Current task/tool material exceeds the approximate context budget; checkpoint retained without truncating the theorem')
        request_file = self.store.write_artifact(role + '-request', json.dumps(payload, ensure_ascii=False))
        stream_file = self.store.start_stream(role)
        call = {'role': role, 'round': self.state['rounds_started'], 'status': 'running',
                'reserved_tokens': cap, 'stream': stream_file, 'request': request_file}
        self.state['calls'].append(call)
        self.state['tokens_charged'] += cap  # reserve BEFORE dispatch, survives crashes
        self._save()
        client = self.agent.client
        old_timeout = getattr(client, 'timeout', None)
        if old_timeout is not None:
            client.timeout = max(0.1, min(old_timeout, remaining_seconds))
        started = time.monotonic()
        self.emit('start', '')
        stream = None
        try:
            stream = client.stream(payload)
            for event in stream:
                self.store.append_stream(stream_file, event)
                msg = event.get('message', {})
                if msg.get('thinking'):
                    self.emit('thinking', msg['thinking'])
                if msg.get('content') and role in {'solver', 'checkpoint'}:
                    self.emit('text', msg['content'])
                if self._elapsed() >= self.state['settings']['max_seconds']:
                    raise ProofBudget('Elapsed-time budget exhausted during generation')
            result = self._read_stream(stream_file)
            if not any(json.loads(line).get('done') for line in self.store.read_artifact(stream_file).splitlines()):
                raise AgentError('Ollama stream ended without a completion event; partial work saved')
            call['status'] = 'complete' if result['complete'] else 'truncated'
            count = result['stats'].get('eval_count')
            if type(count) is int and count >= 0:
                self.state['tokens_charged'] += count - cap
                call['charged_tokens'] = count
            else:
                call['charged_tokens'] = cap
            call['stats'] = result['stats']
            self.agent.last_stats = result['stats']
            if not result['complete']:
                self.emit('notice', f'{role.capitalize()} hit its output limit; partial work retained for a conservative checkpoint')
            return result
        except BaseException:
            call['status'] = 'interrupted'
            if role == 'solver' and self.state['pending']:
                self.state['pending']['partial'] = stream_file
            raise
        finally:
            if stream is not None and hasattr(stream, 'close'):
                stream.close()
            if old_timeout is not None:
                client.timeout = old_timeout
            call['seconds'] = time.monotonic() - started
            self._save()
            self.emit('end', '')

    def _json_call(self, role, policy, task, spec, *, complete_ledger=False, include_ledger=True, cap=2048):
        cap = min(cap, self.agent.ctx // 3)
        instructions = task + '\nReturn ONLY JSON with this schema:\n' + json.dumps(spec)
        messages = self._context(policy, instructions, complete_ledger=complete_ledger, output=cap,
                                 retrieval=False, include_ledger=include_ledger, format_schema=spec,
                                 recording=role == 'recorder')
        visible = self._visible_claims[:]
        result = self._call(role, messages, cap, format_schema=spec)
        if not result['complete']:
            raise ValueError('Truncated reviewer output cannot become a ledger verdict')
        value = json.loads(result['text'])
        _validate(value, spec)
        value['_artifact'] = result['stream']
        value['_visible_claims'] = visible
        return value

    def _excerpt_for_call(self, text, policy, prefix, cap, spec=None):
        schema_text = json.dumps(spec) if spec else ''
        # Schema occurs in both instructions and the request format. Reserve
        # both, and room for the envelope, before excerpting raw attempt text.
        overhead = len((SYSTEM + policy + self._base() + prefix + schema_text * 2).encode())
        available = max(0, (self.agent.ctx - cap - 512) * 3 - overhead - 2400)
        raw = text.encode()
        if len(raw) <= available:
            return text, False
        if available < 500:
            raise ProofContext('The original statement and review instructions need a larger context; no theorem text was dropped')
        half = max(1, (available - 160) // 2)
        return (raw[:half].decode(errors='ignore') +
                '\n[ATTEMPT EXCERPT OMITTED: no completion verdict permitted]\n' +
                raw[-half:].decode(errors='ignore')), True

    def _checkpoint(self, draft):
        cap = min(2048, self.agent.ctx // 3)
        raw = draft['thinking'] + ('\nPARTIAL WRITTEN OUTPUT:\n' + draft['text'] if draft['text'] else '')
        material, clipped = self._excerpt_for_call(raw, CHECKPOINT_POLICY, '', cap)
        messages = self._context(CHECKPOINT_POLICY, 'SOURCE ATTEMPT:\n' + material,
                                 output=cap, include_ledger=False)
        result = self._call('checkpoint', messages, cap, think=False)
        result['origin_stream'] = draft.get('stream')
        result['from_truncated_attempt'] = True
        result['material_clipped'] = clipped
        result['transport_complete'] = result['complete']
        # Salvage is a checkpoint, not a new complete proof submission. A
        # subsequent solver must write the assembled argument for full audit.
        result['complete'] = False
        return result

    def _critic(self, draft):
        cap = min(2048, self.agent.ctx // 3)
        material, clipped = self._excerpt_for_call(draft['text'] or draft['thinking'], CRITIC_POLICY, '', cap, CRITIC_SCHEMA)
        try:
            result = self._json_call('critic', CRITIC_POLICY, 'CURRENT CANDIDATE:\n' + material,
                                     CRITIC_SCHEMA, include_ledger=False, cap=cap)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {'valid_steps': [], 'first_invalid_step': '',
                      'reason': 'No valid independent critique: ' + str(exc),
                      'missing_work': 'The current argument still needs an independent mathematical check.',
                      'complete_candidate': False}
        result['ready_for_assembly'] = bool(result['complete_candidate'] and not result['first_invalid_step'].strip() and not result['reason'].strip()
            and not result['missing_work'].strip() and not clipped and draft.get('transport_complete', draft['complete']))
        if clipped or not draft['complete'] or not draft['text']:
            result['complete_candidate'] = False
        if result['first_invalid_step'].strip() or result['reason'].strip() or result['missing_work'].strip():
            result['complete_candidate'] = False
        return result

    def _review(self, draft, critique):
        cap = min(3072, self.agent.ctx // 3)
        prefix = 'FRESH INDEPENDENT CRITIQUE:\n' + json.dumps(critique, ensure_ascii=False)
        material, clipped = self._excerpt_for_call(draft['text'] or draft['thinking'], REVIEW_POLICY, prefix, cap, REVIEW_BATCH_SCHEMA)
        label = 'WRITTEN CANDIDATE' if draft['complete'] and draft['text'] else 'INCOMPLETE ATTEMPT'
        try:
            review = self._json_call('recorder', REVIEW_POLICY,
                label + '\n' + material + '\n' + prefix, REVIEW_BATCH_SCHEMA, cap=cap)
        except (json.JSONDecodeError, ValueError) as exc:
            review = {'claims': [], 'complete_candidate': False,
                      'next_task': 'Write one short, precise intermediate claim with a self-contained argument; the ledger response was invalid.',
                      'strategy_summary': 'No valid ledger checkpoint: ' + str(exc),
                      '_visible_claims': []}
        if clipped or not draft['complete'] or not draft['text'] or not critique['complete_candidate']:
            review['complete_candidate'] = False
        return review

    def _record_batch(self, review, pending, draft):
        before_stagnant = self.state['stagnant_rounds']
        previous_task = self.state['next_task']
        items = list(review['claims'])
        critique = pending['critique']
        if critique['first_invalid_step'].strip():
            # Preserve the fresh critic's concrete objection even if the
            # recorder ignores it. A repair must be written and checked later.
            items.append({'critical_claim': critique['first_invalid_step'],
                'assumptions': [], 'dependencies': [], 'argument': '',
                'disposition': 'gap', 'objection': critique['reason'],
                'evidence': 'Independent critique of the current candidate: ' + critique['reason'],
                'next_task': review['next_task'], 'new_progress': True,
                'complete_candidate': False, 'resolves': [], 'resolution': ''})
        if not items:
            items = [{'critical_claim': '', 'assumptions': [], 'dependencies': [], 'argument': '',
                'disposition': 'uncertain', 'objection': review['strategy_summary'], 'evidence': '',
                'next_task': review['next_task'], 'new_progress': False,
                'complete_candidate': False, 'resolves': [], 'resolution': ''}]
        records, made_progress = [], False
        fields = ('statement', 'status', 'assumptions', 'dependencies', 'argument', 'objection', 'evidence', 'resolves', 'resolution')
        for item in items:
            item = {**item, '_artifact': review.get('_artifact'), '_visible_claims': review.get('_visible_claims', [])}
            if (critique['first_invalid_step'].strip() and item['disposition'] == 'supported'
                    and _normal(item['critical_claim']) == _normal(critique['first_invalid_step'])):
                item['disposition'] = 'gap'
                item['objection'] = 'The independent critic rejected this inference: ' + critique['reason'] + ' Any proposed repair must be submitted for a new independent review.'
                item['resolves'] = []
            record = self._record(item, pending, draft, emit=False)
            duplicate = next((c for c in self.state['claims'][:-1]
                              if all(c.get(k) == record.get(k) for k in fields)), None)
            if duplicate:
                self.state['claims'].pop()
                record = duplicate
            elif self.state['stagnant_rounds'] == 0:
                made_progress = True
            if record['id'] not in {c['id'] for c in records}:
                records.append(record)
        self.state['stagnant_rounds'] = 0 if made_progress else before_stagnant + 1
        self.state['next_task'] = review['next_task'] or critique['missing_work'] or 'Assemble the established claims into a self-contained proof of the original goal.'
        if (not critique.get('ready_for_assembly') and critique['missing_work'].strip()
                and _normal(review['next_task']) == _normal(previous_task)):
            self.state['next_task'] = critique['missing_work']
        if critique.get('ready_for_assembly') and not draft['complete']:
            self.state['assemble_next'] = True
            self.state['next_task'] = 'The written checkpoint appears to contain a complete argument. Recheck its steps, then submit a concise self-contained proof of the entire original statement so it can receive a whole-proof audit. Do not restart the exploration.'
        return records

    def _plan(self):
        # Show the planner route descriptions rather than inherited verdicts.
        # It sees successful local arguments without endorsement labels.
        attempted = [{'task': r['task'], 'outcome': r.get('strategy_summary', '')}
                     for r in self.state['rounds'][-4:]]
        useful = [{'claim': c['statement'], 'argument': c['argument'], 'assumptions': c['assumptions']}
                  for c in self.state['claims'] if c['status'] == 'reviewed'][-3:]
        task = ('PRIOR ROUTES, TO AVOID REPEATING:\n' + json.dumps(attempted, ensure_ascii=False)
                + '\nPOTENTIALLY USEFUL LOCAL ARGUMENTS, CHECK BEFORE USING:\n' + json.dumps(useful, ensure_ascii=False))
        try:
            result = self._json_call('planner', PLAN_POLICY, task, PLAN_SCHEMA, include_ledger=False, cap=1536)
            if not all(result[k].strip() for k in ('approach', 'task', 'difference', 'deliverable')):
                raise ValueError('Planner must provide a concrete task and explain its difference')
            if any(_normal(r.get('plan', {}).get('task', r['task'])) == _normal(result['task']) for r in self.state['rounds']):
                raise ValueError('Planner repeated a previous task verbatim')
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            return {'approach': 'Independent reconstruction',
                    'task': 'Starting from the original statement, write a new self-contained proof attempt using a different intermediate claim from prior routes. Establish that claim first; do not restart a known incomplete derivation.',
                    'difference': 'Previous route retired; the planner did not return a valid new task: ' + str(exc),
                    'deliverable': 'One new precisely scoped claim, its complete derivation, and the remaining obligation.'}

    def _record(self, review, pending, draft, *, emit=True):
        previous = self.state['claims']
        by_id = {c['id']: c for c in previous}
        status = {'supported': 'reviewed', 'gap': 'gap', 'refuted': 'refuted', 'uncertain': 'uncertain'}[review['disposition']]
        objection = review['objection']
        if status == 'reviewed' and (not review['argument'].strip() or not review['critical_claim'].strip()):
            status, objection = 'uncertain', 'Reviewer supplied no exact claim and supporting argument'
        if status == 'reviewed' and objection.strip():
            status = 'gap'
        if status == 'refuted' and not review['evidence'].strip():
            status, objection = 'gap', objection or 'No explicit refutation evidence was supplied'
        deps = review['dependencies']
        if any(d not in by_id or by_id[d]['status'] != 'reviewed' for d in deps):
            status = 'gap'
            objection += '\nUnresolved or unknown dependencies; claim cannot be treated as reviewed.'
        resolves = review['resolves']
        if resolves and (not review['resolution'].strip() or any(c not in by_id for c in resolves)):
            status = 'gap'
            objection += '\nPrior objections were not explicitly resolved with valid references.'
            resolves = []
        visible = set(review.get('_visible_claims', []))
        if any(c not in visible for c in deps + resolves):
            status = 'gap'
            objection += '\nThe reviewer could not inspect all cited records; dependency or resolution remains unreviewed.'
            resolves = []
        repeated = [c for c in previous if review['critical_claim'].strip() and _normal(c['statement']) == _normal(review['critical_claim'])]
        blocked = [c for c in repeated if c['status'] in {'gap', 'refuted'} and c['id'] not in resolves]
        if blocked and status == 'reviewed':
            status = 'gap'
            objection += '\nThe same claim has recorded objections (' + ', '.join(c['id'] for c in blocked) + ') that this attempt did not explicitly answer.'
        claim = {'id': f'C{len(previous)+1}', 'round': pending['index'],
                 'statement': review['critical_claim'], 'status': status,
                 'assumptions': review['assumptions'], 'dependencies': deps,
                 'argument': review['argument'], 'objection': objection,
                 'evidence': review['evidence'], 'resolves': resolves,
                 'resolution': review['resolution'], 'candidate_artifact': pending['draft'],
                 'review_artifact': review.get('_artifact'), 'next_task': review['next_task']}
        signature = _normal(claim['statement'] + claim['objection'] + claim['evidence'])
        duplicate = any(_normal(c['statement'] + c['objection'] + c['evidence']) == signature for c in previous)
        progress = bool(review['new_progress'] and not duplicate and not blocked and
                        ((status == 'reviewed' and claim['argument']) or
                         (status in {'gap', 'refuted'} and claim['statement'] and claim['evidence'])))
        self.state['stagnant_rounds'] = 0 if progress else self.state['stagnant_rounds'] + 1
        previous.append(claim)  # append only; never silently rewrite a prior judgment
        self.state['next_task'] = review['next_task'] or 'Investigate the latest precise objection; use another approach if it cannot be repaired.'
        if self.state['stagnant_rounds'] >= 2:
            self.state['next_task'] = 'Progress has stalled. Test the blocked inference on a simple admissible case or develop a different route.\n' + self.state['next_task']
        if emit:
            self.emit('notice', f'{claim["id"]}: {status}. {objection or review["next_task"]}')
        return claim

    def _audit(self, draft):
        try:
            result = self._json_call('auditor', AUDIT_POLICY, 'SELF-CONTAINED CANDIDATE:\n' + draft['text'], AUDIT_SCHEMA, complete_ledger=True)
            if result['verdict'] == 'complete' and (result['objection'].strip() or not result['explanation'].strip()):
                result['verdict'] = 'uncertain'
                result['next_task'] = 'Resubmit the self-contained proof for audit: the previous audit returned inconsistent fields, so completion was not recorded. An approval requires a nonempty explanation and an empty objection field.'
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            return {'verdict': 'uncertain', 'explanation': 'Whole-proof audit did not yield a valid verdict: ' + str(exc),
                    'objection': 'Completion remains unverified', 'next_task': 'Produce a shorter self-contained proof that can be fully audited.'}

    def _report(self, *, detailed=False):
        s = self.state
        out = [f'# Proof {s["id"]}', '', f'Status: **{s["status"]}**', '',
               'Model reviews are fallible. No result in this report is a formal proof certificate.', '',
               'Literature tools: ' + ('enabled for this job; external results require applicability checks.' if s['settings'].get('allow_literature') else 'disabled for this job.'), '',
               '## Original goal', '', s['goal'], '', '## Budget', '',
               f'Work rounds started: {s["rounds_started"]}/{s["settings"]["max_rounds"]}; generated tokens charged: {s["tokens_charged"]}/{s["settings"]["max_tokens"]}; active elapsed seconds: {s["seconds_used"]:.1f}/{s["settings"]["max_seconds"]}.',
               'Missing token statistics or interrupted requests are conservatively charged their reserved output budget.', '',
               '## Outcome', '', s.get('stop_reason', 'A self-contained candidate passed a whole-proof model audit.' if s['status'] == 'candidate_complete' else 'Proof search is incomplete.'), '',
               '## Next task', '', ('Independently check the complete candidate below.' if s['status'] == 'candidate_complete' else s['next_task']), '',
               '## Proof ledger' if detailed else '## Mathematical progress', '',
               'Full arguments, assumptions and objection history are retained in ledger.md.', '']
        for c in s['claims']:
            if not detailed:
                excerpt = lambda text, limit: text if len(text) <= limit else text[:limit] + '… (excerpt; see ledger.md)'
                out += [f'**{c["id"]} — {c["status"]}:** ' + excerpt(c['statement'] or '(No claim extracted)', 500), '']
                if c['objection']:
                    out += ['Objection: ' + excerpt(c['objection'], 400), '']
                if c['evidence']:
                    out += ['Evidence / witness: ' + excerpt(c['evidence'], 400), '']
                if c.get('whole_proof_objection'):
                    out += ['Whole-proof objection: ' + excerpt(c['whole_proof_objection'], 400), '']
                continue
            out += [f'### {c["id"]} — {c["status"]} (round {c["round"]})', '', c['statement'] or '(No claim extracted)', '',
                    'Assumptions: ' + ('; '.join(c['assumptions']) or 'As stated in the original problem; see the argument.'),
                    'Dependencies: ' + (', '.join(c['dependencies']) or 'None cited'), '',
                    c['argument'] or '(No supporting derivation recorded)', '',
                    '**Objection:** ' + (c['objection'] or 'None recorded by this reviewer.'),
                    '**Evidence / witness:** ' + (c['evidence'] or 'None recorded.'), '',
                    f'Original candidate: artifacts/{c["candidate_artifact"]}', '']
            if c['resolves']:
                out += ['Addresses earlier objections: ' + ', '.join(c['resolves']), c['resolution'], '']
            if c.get('whole_proof_objection'):
                out += ['**Whole-proof objection:** ' + c['whole_proof_objection'], '']
        if not s['claims']:
            out += ['No reviewed mathematical progress was recorded.', '']
        if s.get('final_audit'):
            out += ['## Whole-proof audit', '', json.dumps(s['final_audit'], ensure_ascii=False, indent=2), '']
            if s['status'] == 'candidate_complete':
                draft = json.loads(self.store.read_artifact(s['final_audit']['candidate']))
                out += ['## Complete candidate (model-audited)', '', draft['text'], '']
        if s['pending']:
            out += ['## Unfinished work', '', f'Round {s["pending"]["index"]}, phase {s["pending"]["phase"]}. Partial streams and candidates remain in artifacts/.', '']
            pending = s['pending']
            if pending.get('draft'):
                draft = json.loads(self.store.read_artifact(pending['draft']))
                out += ['Unreviewed candidate: artifacts/' + pending['draft'], '',
                        'The following is unfinished source material, not an established result:', '',
                        (draft['text'] or draft['thinking'])[:3000], '']
            elif pending.get('partial'):
                out += ['Partial stream: artifacts/' + pending['partial'], '',
                        'No complete review is available; this stream must not be treated as a proved claim.', '']
        out += ['## Sources and artifacts', '']
        out += [f'- Pinned source: {src["path"]} (SHA-256 {src["sha256"]})' for src in s['sources']]
        out += ['', 'state.json retains exact claims and source snapshots; artifacts/ retains requests, attempts, streamed output and reviews.']
        if s['status'] in {'candidate_complete', 'budget_exhausted', 'stalled', 'needs_recovery'}:
            out += ['This job is terminal. Its ledger remains available; a new /prove creates a separate explicitly budgeted job.', '']
        else:
            out += [f'Resume with `/resume {s["id"]}`. Resuming does not reset budgets.', '']
        return '\n'.join(out)

    def _write_reports(self):
        report = self._report()
        self.store.write_report(report)
        self.store.write_ledger(self._report(detailed=True))

    def _result(self):
        self._write_reports()
        return {'id': self.state['id'], 'status': self.state['status'],
                'report': self._report(), 'directory': str(self.store.directory)}
