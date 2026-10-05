"""Solve, verify, and repair one proof, with immutable candidates and bounded cost.

The controller enforces provenance and budgets. A model review is not a proof
certificate, and preservation of an answer does not establish its correctness.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .agent import AgentError
from .backends import TokenBudgetError
from .ledger import LedgerError, ProofStore
from .proof_policy import (
    CONTINUE_POLICY, REPAIR_POLICY, VERIFIER_SCHEMA, initial_request, review_request,
)

POLICY_VERSION = 7
TERMINAL = {'candidate_complete', 'budget_exhausted', 'budget_violation', 'attempts_exhausted',
            'uncertain', 'review_unavailable', 'needs_context', 'needs_recovery'}


class ProofBudget(AgentError):
    pass


class ProofContext(AgentError):
    pass


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _validate(value, spec, path='response'):
    """Strictly validate the small review contract, independently of the server."""
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
    elif kind == 'string':
        if not isinstance(value, str) or not value.strip() or len(value) > spec.get('maxLength', 10000):
            raise ValueError(f'{path}: expected nonempty bounded text')
        if 'enum' in spec and value not in spec['enum']:
            raise ValueError(f'{path}: invalid value')


def validate_review(text):
    """Positive explanation is allowed; contradictory verdicts are unavailable."""
    value = json.loads(text)
    _validate(value, VERIFIER_SCHEMA)
    verdict, issues = value['verdict'], value['issues']
    if verdict == 'no_issue_found' and issues:
        raise ValueError('no_issue_found contradicts the nonempty issues list')
    if verdict == 'issues_found' and not any(i['kind'] != 'uncertainty' for i in issues):
        raise ValueError('issues_found requires a concrete mathematical issue')
    return value


def count_input(client, payload):
    """Count the full request input; return (tokens, counting method). Never clips."""
    counter = getattr(client, 'count_input_tokens', None)
    count = counter(payload) if callable(counter) else None
    if count is None:
        # A conservative fallback avoids the former optimistic chars/3 guard.
        # It can reject a request that an exact tokenizer would fit. Never clip.
        count = len(json.dumps(payload['messages'], ensure_ascii=False).encode())
        count += len(json.dumps(payload.get('format', {}), ensure_ascii=False).encode()) + 256
        method = 'conservative UTF-8 byte estimate'
    else:
        if type(count) is not int or count < 0:
            raise AgentError('Invalid exact input token count')
        method = 'server token count'
    return count, method


def check_context(client, payload):
    """Check the full request without clipping; return the counting provenance."""
    cap = payload['options']['num_predict']
    context = payload['options']['num_ctx']
    count, method = count_input(client, payload)
    if count + cap + 1 > context:
        raise ProofContext(f'Full original statement and saved work need {count} input tokens ({method}) plus {cap} output tokens, exceeding context {context}. Nothing was clipped; candidate retained.')
    return {'input_tokens': count, 'method': method}


class ProofRunner:
    def __init__(self, agent, emit=lambda kind, value: None):
        self.agent, self.emit = agent, emit
        self.store = None
        self._clock = None
        self._seconds_before = 0

    @property
    def state(self):
        return self.store.state

    def _provenance(self):
        return {
            'model': self.agent.model, 'ctx': self.agent.ctx, 'think': True,
            'seed': self.agent.seed, 'temperature': self.agent.temperature, 'top_p': self.agent.top_p,
            'backend': getattr(self.agent.client, 'backend', 'ollama'),
            'host': getattr(self.agent.client, 'host', None),
            'thinking_budget_policy': copy.deepcopy(getattr(self.agent.client, 'thinking_budget_policy', {})),
            'policy_version': POLICY_VERSION,
            'policy_sha256': hashlib.sha256(Path(__file__).with_name('proof_policy.py').read_bytes()).hexdigest(),
            'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'transport_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('backends.py', 'agent.py')},
            'harness_version': '0.5.1',
        }

    def start(self, goal, *, max_rounds=3, max_tokens=120000, max_seconds=1800,
              source_files=(), max_predict=32768, verify_tokens=16384,
              min_solve_tokens=None, verify_temperature=None, repair_tokens=None):
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 60000:
            raise ValueError('Provide a nonempty proof goal, at most 60000 characters')
        for name, value, lower, upper in (
            ('max_rounds', max_rounds, 1, 100), ('max_tokens', max_tokens, 128, 100000000),
            ('max_predict', max_predict, 128, 1000000), ('verify_tokens', verify_tokens, 128, 1000000),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{name} must be an integer between {lower} and {upper}')
        if repair_tokens is None:
            repair_tokens = max_predict
        if type(repair_tokens) is not int or not 128 <= repair_tokens <= max_predict:
            raise ValueError('repair_tokens must be an integer between 128 and max_predict')
        if min_solve_tokens is None:
            min_solve_tokens = min(16384, repair_tokens)
        if type(min_solve_tokens) is not int or not 128 <= min_solve_tokens <= repair_tokens:
            raise ValueError('min_solve_tokens must be an integer between 128 and repair_tokens')
        if verify_temperature is None:
            verify_temperature = self.agent.temperature
        if (isinstance(verify_temperature, bool) or not isinstance(verify_temperature, (int, float))
                or not math.isfinite(verify_temperature) or not 0 <= verify_temperature <= 2):
            raise ValueError('verify_temperature must be between 0 and 2')
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('Proof time budget must be finite and positive')
        sources = []
        for name in dict.fromkeys(str(n) for n in source_files):
            path = Path(name)
            if path.is_absolute():
                path = path.resolve().relative_to(self.agent.workspace.root)
            actual = self.agent.workspace.path(str(path))
            text = self.agent.workspace.text(actual)
            sources.append({'path': str(path), 'content': text, 'sha256': _digest(text)})
        settings = dict(self._provenance(), max_rounds=max_rounds, max_tokens=max_tokens,
                        max_seconds=max_seconds, max_predict=max_predict, verify_tokens=verify_tokens,
                        min_solve_tokens=min_solve_tokens, verify_temperature=verify_temperature,
                        repair_tokens=repair_tokens)
        self.store = ProofStore.create(self.agent.workspace.root, goal, settings, sources)
        self.emit('notice', f'Proof {self.state["id"]}: solve, verify, repair if needed; up to {max_rounds} attempts, subject to budget')
        return self._run()

    def resume(self, proof_id):
        self.store = ProofStore.load(self.agent.workspace.root, proof_id)
        if self.state['version'] != 2:
            raise ValueError('Legacy v1 proofs are read-only in v0.5. Inspect their saved artifacts or start a new proof; no automatic migration is performed.')
        if self.state['status'] in TERMINAL:
            return self._run()  # Viewing a finished run never needs its model server.
        saved = self.state['settings']
        for key in ('host', 'backend'):
            if saved.get(key) != self._provenance()[key]:
                raise ValueError(f'Cannot resume with changed {key}; use the original model endpoint. Saved artifacts remain readable.')
        for key in ('model', 'ctx', 'seed', 'temperature', 'top_p'):
            setattr(self.agent, key, saved[key])
        if hasattr(self.agent.client, 'thinking_budget_policy'):
            self.agent.client.thinking_budget_policy = copy.deepcopy(saved['thinking_budget_policy'])
        for key, current in self._provenance().items():
            if saved.get(key) != current:
                raise ValueError(f'Cannot resume with changed {key}; use the original proof policy. Saved artifacts remain readable.')
        return self._run()

    def _elapsed(self):
        return self._seconds_before + (time.monotonic() - self._clock if self._clock is not None else 0)

    def _save(self):
        if self._clock is not None:
            self.state['seconds_used'] = self._elapsed()
            self.state['active_checkpoint_wall'] = time.time()
        self.store.save()

    def _stop(self, status, reason):
        self.state['status'], self.state['stop_reason'] = status, reason

    def _remaining(self):
        return self.state['settings']['max_tokens'] - self.state['tokens_charged']

    def _attempt_cap(self, kind):
        settings = self.state['settings']
        return settings['max_predict'] if kind == 'initial' else settings.get('repair_tokens', settings['max_predict'])

    def _reserve_attempt(self, kind='initial'):
        settings = self.state['settings']
        solve = self._attempt_cap(kind) if kind == 'initial' else settings['min_solve_tokens']
        needed = solve + settings['verify_tokens']
        if self._remaining() < needed:
            raise ProofBudget(f'Not starting another attempt: {needed} tokens must remain for a full solve and verification; {self._remaining()} remain.')
        if self._elapsed() >= settings['max_seconds']:
            raise ProofBudget('Elapsed-time budget exhausted')
        # After one full cycle, its measured duration estimates the next one.
        solver = next((c for c in self.state['calls'] if c['role'] == 'solver' and 'seconds' in c), None)
        verifier = next((c for c in self.state['calls'] if c['role'] == 'verifier' and 'seconds' in c), None)
        if solver and verifier:
            cycle = solver['seconds'] + verifier['seconds']
            left = settings['max_seconds'] - self._elapsed()
            if left < cycle:
                raise ProofBudget(f'Not starting another attempt: the first solve and review took {cycle:.0f} s; {left:.0f} s remain.')

    def _run(self):
        with self.store.lock():
            self.store = ProofStore.load(self.agent.workspace.root, self.state['id'])
            if self.state.get('recovery_notice'):
                self._stop('needs_recovery', 'A backup was recovered and its last budget reservation may be missing. Automatic inference is disabled; inspect the saved artifacts.')
                self.store.save()
                return self._result()
            if self.state['status'] in TERMINAL:
                return self._result()
            if self.state['status'] == 'running' and self.state.get('active_checkpoint_wall'):
                self.state['seconds_used'] += max(0, time.time() - self.state['active_checkpoint_wall'])
            self._seconds_before = self.state['seconds_used']
            self._clock = time.monotonic()
            self.state['status'] = 'running'
            try:
                self._recover_calls()
                if self.state['pending'] is None:
                    self._new_attempt('initial')
                while self.state['status'] == 'running':
                    pending = self.state['pending']
                    if pending['phase'] == 'solve':
                        if self._find_call(self._call_key('solver')) is not None:
                            result = self._call('solver', None, pending.get('solver_cap', self.state['settings']['max_predict']))
                        else:
                            messages = self._solve_messages(pending)
                            result = self._call('solver', messages, self._solver_cap(messages, pending))
                        candidate = self._candidate(result, pending)
                        pending['candidate'] = candidate['id']
                        pending['phase'] = 'review' if candidate['transport_complete'] else 'continue'
                        self._save()
                    elif pending['phase'] == 'review':
                        self._review(pending)
                    elif pending['phase'] in {'continue', 'repair'}:
                        self._new_attempt(pending['phase'], pending['candidate'], pending.get('review'))
                    else:
                        raise LedgerError('Invalid saved proof phase')
            except ProofContext as exc:
                self._stop('needs_context', str(exc))
            except ProofBudget as exc:
                self._stop('budget_exhausted', str(exc))
            except TokenBudgetError as exc:
                self._stop('budget_violation', str(exc))
            except KeyboardInterrupt:
                self._stop('paused', 'Interrupted by the user. Saved calls will not be repeated; unfinished calls retain their full token reservation.')
            except AgentError as exc:
                self._stop('paused', str(exc))
            finally:
                self._save()
                self._clock = None
            return self._result()

    def _new_attempt(self, kind, parent=None, review=None):
        if self.state['rounds_started'] >= self.state['settings']['max_rounds']:
            self._stop('attempts_exhausted', 'Attempt limit reached; retained candidates and unresolved reviews remain available.')
            return
        self._reserve_attempt(kind)
        self.state['rounds_started'] += 1
        self.state['pending'] = {'phase': 'solve', 'kind': kind, 'parent': parent, 'review': review,
                                 'index': self.state['rounds_started'], 'review_attempt': 0}
        self._save()

    def _get_candidate(self, candidate_id):
        return next(c for c in self.state['candidates'] if c['id'] == candidate_id)

    def _candidate_text(self, candidate):
        text = self.store.read_artifact(candidate['artifact'])
        if _digest(text) != candidate['sha256']:
            raise LedgerError('Candidate digest changed; refusing to review or export altered evidence')
        return text

    def _find_call(self, key):
        return next((c for c in self.state['calls'] if c['key'] == key), None)

    def _payload(self, role, messages, cap, format_schema=None):
        payload = {'model': self.agent.model, 'messages': messages, 'stream': True, 'think': True,
                   'options': {'num_ctx': self.agent.ctx, 'num_predict': cap,
                               'temperature': self.agent.temperature if role == 'solver'
                               else self.state['settings'].get('verify_temperature', 0),
                               'top_p': self.agent.top_p}}
        if self.agent.seed is not None:
            payload['options']['seed'] = (self.agent.seed + len(self.state['calls'])) % (2 ** 31)
        if format_schema is not None:
            payload['format'] = format_schema
        return payload

    def _count(self, messages, cap):
        client = self.agent.client
        old_timeout = getattr(client, 'timeout', None)
        if old_timeout is not None:
            client.timeout = max(0.1, min(old_timeout, self.state['settings']['max_seconds'] - self._elapsed()))
        try:
            return count_input(client, self._payload('solver', messages, cap))[0]
        finally:
            if old_timeout is not None:
                client.timeout = old_timeout

    def _solver_cap(self, messages, pending):
        """The initial solve keeps the direct-call allowance; later calls use the repair
        allowance, fitted to the context for repairs and continuations."""
        settings = self.state['settings']
        cap = self._attempt_cap(pending['kind'])
        if pending['kind'] != 'initial':
            cap = min(cap, self._remaining() - settings['verify_tokens'])
        if pending['kind'] not in {'initial', 'retry'}:
            floor = settings.get('min_solve_tokens', cap)
            count = self._count(messages, floor)
            cap = min(cap, self.agent.ctx - count - 1)
            if cap < floor:
                raise ProofContext(f'The {pending["kind"]} request needs {count} input tokens; with context {self.agent.ctx} '
                                   f'only {max(cap, 0)} output tokens remain, below the {floor}-token minimum. Nothing was clipped; candidates retained.')
        pending['solver_cap'] = cap
        return cap

    def _solve_messages(self, pending):
        goal, sources = self.state['goal'], self.state['sources']
        messages = initial_request(goal, sources)
        if pending['kind'] in {'initial', 'retry'}:
            return messages
        prior = self._get_candidate(pending['parent'])
        text = self._candidate_text(prior)
        if pending['kind'] == 'repair':
            review = next(r for r in self.state['reviews'] if r['id'] == pending['review'])
            messages[0]['content'] += ('\n\n' + REPAIR_POLICY + '\n\nPRIOR CANDIDATE:\n' + text +
                                       '\n\nFALLIBLE REVIEW:\n' + json.dumps(review['response'], ensure_ascii=False))
            return messages
        return self._continue_messages(pending, prior, text)

    def _continue_messages(self, pending, prior, text):
        """Continue from the written text and as much of the latest notes as fits.

        Only the solver's own saved notes may be excerpted, with an explicit
        marker. If even the written text leaves too little output room, a fresh
        solve of the original problem replaces the continuation.
        """
        goal, sources = self.state['goal'], self.state['sources']
        notes = self._read_stream(self._find_call(prior['call'])['stream'])['thinking']
        floor, ctx = self.state['settings']['min_solve_tokens'], self.agent.ctx

        def build(tail):
            header = ('SAVED WORKING NOTES:\n' if len(tail) == len(notes) else
                      f'SAVED WORKING NOTES — final excerpt, earlier notes omitted ({len(tail)} of {len(notes)} characters):\n')
            messages = initial_request(goal, sources)
            messages[0]['content'] += '\n\n' + CONTINUE_POLICY + '\n\nSAVED WRITTEN TEXT:\n' + text + '\n\n' + header + tail
            return messages

        room = ctx - 1 - floor - self._count(build(''), floor) - 64
        if room > 0:
            keep = min(len(notes), int(room * 2.5))
            for _ in range(5):
                tail = notes[len(notes) - keep:] if keep else ''
                messages = build(tail)
                if self._count(messages, floor) + floor + 1 <= ctx:
                    pending.update(notes_chars_kept=len(tail), notes_chars_total=len(notes))
                    return messages
                keep = int(keep * 0.75)
        pending.update(kind='retry', continuation_fallback='fresh_solve',
                       notes_chars_kept=0, notes_chars_total=len(notes))
        self.emit('notice', 'Saved work does not fit a useful continuation; solving the original problem afresh')
        return initial_request(goal, sources)

    def _candidate(self, result, pending):
        # A completed solver call can be recovered without dispatching it again.
        call_key = self._call_key('solver')
        existing = next((c for c in self.state['candidates'] if c['call'] == call_key), None)
        if existing:
            return existing
        artifact = self.store.write_artifact('proof-candidate', result['text'])
        candidate = {'id': f'P{pending["index"]}', 'attempt': pending['index'], 'call': call_key,
                     'artifact': artifact, 'sha256': _digest(result['text']), 'parent': pending['parent'],
                     'kind': pending['kind'], 'transport_complete': bool(result['complete'] and result['text'].strip() and not result['calls']),
                     'review_status': 'unreviewed'}
        self.state['candidates'].append(candidate)
        if self.state['initial_candidate'] is None:
            self.state['initial_candidate'] = candidate['id']
        selected_id = self.state['selected_candidate']
        selected = self._get_candidate(selected_id) if selected_id else None
        if result['text'].strip() and (selected is None or
                (not selected['transport_complete'] and candidate['transport_complete'])):
            # A finished written response is preferred to an unfinished fragment.
            # Once a finished candidate exists, only a passing review replaces it.
            self._select(candidate, 'first_written_candidate' if selected is None else 'finished_response_replaces_fragment')
        self._save()
        return candidate

    def _select(self, candidate, reason):
        previous = self.state['selected_candidate']
        self.state['selected_candidate'] = candidate['id']
        self.state.setdefault('selection_history', []).append({
            'previous': previous, 'selected': candidate['id'], 'reason': reason,
            'after_call_count': len(self.state['calls']),
        })

    def _review(self, pending):
        candidate = self._get_candidate(pending['candidate'])
        messages = review_request(self.state['goal'], self.state['sources'], self._candidate_text(candidate))
        if pending.get('protocol_error'):
            messages[0]['content'] += ('\n\nYour previous review could not be interpreted: ' + pending['protocol_error'] +
                '\nReview this SAME candidate again using the exact JSON contract. A response-format error is not a mathematical objection.')
        cap = self.state['settings']['verify_tokens']
        cached = any(c['key'] == self._call_key('verifier') for c in self.state['calls'])
        if not cached and self._remaining() < cap:
            self._stop('review_unavailable', 'Insufficient budget for a full verifier call; the candidate was retained without rewriting it.')
            return
        result = self._call('verifier', messages, cap, format_schema=VERIFIER_SCHEMA)
        try:
            if not result['complete'] or result['calls']:
                raise ValueError('Review generation did not finish a standalone response')
            response = validate_review(result['text'])
        except (ValueError, TypeError, RecursionError) as exc:
            response = None
            error = str(exc)
        else:
            error = None
        review = {'id': f'V{len(self.state["reviews"]) + 1}', 'candidate': candidate['id'],
                  'candidate_sha256': candidate['sha256'], 'call': self._call_key('verifier'),
                  'response': response, 'protocol_error': error}
        self.state['reviews'].append(review)
        pending['review'] = review['id']
        if error:
            candidate['review_status'] = 'review_unavailable'
            if pending['review_attempt'] == 0 and self._remaining() >= cap:
                pending['review_attempt'] = 1
                pending['protocol_error'] = error
                self.emit('notice', 'Clarifying an unavailable review on the same candidate; no proof rewrite was requested')
            else:
                self._stop('review_unavailable', 'No consistent verifier verdict after the bounded review attempt(s). Candidate retained unchanged.')
        else:
            verdict = response['verdict']
            candidate['review_status'] = verdict
            if verdict == 'no_issue_found':
                self._select(candidate, 'whole_proof_review_found_no_issue')
                self._stop('candidate_complete', 'The whole-proof verifier found no issue. This is a model assessment, not a formal certificate.')
            elif verdict == 'uncertain':
                self._stop('uncertain', 'The verifier is uncertain; uncertainty does not trigger an automatic mathematical rewrite. Candidate and review retained.')
            else:
                pending['phase'] = 'repair'
        self._save()

    def _call_key(self, role):
        pending = self.state['pending']
        return f'{pending["index"]}-{role}-{pending["review_attempt"] if role == "verifier" else 0}'

    def _read_stream(self, filename):
        result = {'text': '', 'thinking': '', 'calls': [], 'complete': False, 'done': False,
                  'stats': {}, 'stream': filename}
        for line in self.store.read_artifact(filename).splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                break  # A killed write may leave a partial final journal line.
            if result['done'] or not isinstance(event, dict) or event.get('error'):
                raise AgentError('Invalid model event journal; saved evidence was not treated as a proof')
            message = event.get('message', {})
            for field, target in (('content', 'text'), ('thinking', 'thinking')):
                fragment = message.get(field) or ''
                if not isinstance(fragment, str):
                    raise AgentError('Model returned a non-text response fragment')
                result[target] += fragment
            result['calls'].extend(message.get('tool_calls') or [])
            if event.get('done') is True:
                result['done'] = True
                result['complete'] = event.get('done_reason') == 'stop'
                result['stats'] = {k: event[k] for k in ('eval_count', 'prompt_eval_count', 'eval_duration', 'done_reason') if k in event}
        return result

    def _charge_completed(self, call, result):
        count = result['stats'].get('eval_count')
        charged = count if type(count) is int and count >= 0 else call['reserved_tokens']
        previous = call.get('charged_tokens', call['reserved_tokens'])
        self.state['tokens_charged'] += charged - previous
        call.update(charged_tokens=charged, stats=result['stats'],
                    status='complete' if result['complete'] else 'truncated')
        if charged > call['reserved_tokens']:
            call['status'] = 'budget_violation'
            raise TokenBudgetError(result['stats'], call['reserved_tokens'])

    def _recover_calls(self):
        for call in self.state['calls']:
            if call['status'] not in {'running', 'interrupted'}:
                continue
            result = self._read_stream(call['stream'])
            if result['done']:
                self._charge_completed(call, result)
            else:
                call['status'] = 'interrupted'
                call['charged_tokens'] = call['reserved_tokens']
        self._save()

    def _call(self, role, messages, cap, format_schema=None):
        key = self._call_key(role)
        previous = next((c for c in self.state['calls'] if c['key'] == key), None)
        if previous is not None:
            return self._read_stream(previous['stream'])
        if self._remaining() < cap:
            raise ProofBudget('Insufficient budget for the full role allowance; no reduced-cap call was dispatched')
        remaining_seconds = self.state['settings']['max_seconds'] - self._elapsed()
        if remaining_seconds <= 0:
            raise ProofBudget('Elapsed-time budget exhausted')
        payload = self._payload(role, messages, cap, format_schema)
        client = self.agent.client
        old_timeout = getattr(client, 'timeout', None)
        if old_timeout is not None:
            client.timeout = max(0.1, min(old_timeout, remaining_seconds))
        try:
            context = check_context(client, payload)
        finally:
            if old_timeout is not None:
                client.timeout = old_timeout
        if self._elapsed() >= self.state['settings']['max_seconds']:
            raise ProofBudget('Elapsed-time budget exhausted before generation')
        request = self.store.write_artifact(role + '-request', json.dumps(payload, ensure_ascii=False))
        stream_file = self.store.start_stream(role)
        call = {'key': key, 'role': role, 'round': self.state['rounds_started'], 'status': 'running',
                'reserved_tokens': cap, 'request': request, 'stream': stream_file, 'context': context}
        self.state['calls'].append(call)
        self.state['tokens_charged'] += cap
        self._save()  # Durable reservation precedes inference.
        remaining_seconds = self.state['settings']['max_seconds'] - self._elapsed()
        if old_timeout is not None:
            client.timeout = max(0.1, min(old_timeout, remaining_seconds))
        started, stream = time.monotonic(), None
        self.emit('start', '')
        try:
            stream = client.stream(payload)
            for event in stream:
                self.store.append_stream(stream_file, event)
                message = event.get('message', {})
                if message.get('thinking'):
                    self.emit('thinking', message['thinking'])
                if message.get('content') and role == 'solver':
                    self.emit('text', message['content'])
                if not event.get('done') and self._elapsed() >= self.state['settings']['max_seconds']:
                    raise ProofBudget('Elapsed-time budget exhausted during generation')
            result = self._read_stream(stream_file)
            if not result['done']:
                raise AgentError('Model stream ended without completion; exact partial work was saved')
            self._charge_completed(call, result)
            self.agent.last_stats = result['stats']
            return result
        except BaseException as exc:
            if isinstance(exc, TokenBudgetError):
                previous_charge = call.get('charged_tokens', cap)
                call.update(status='budget_violation', stats=exc.stats, charged_tokens=exc.stats['eval_count'])
                self.state['tokens_charged'] += call['charged_tokens'] - previous_charge
            elif call['status'] == 'running':
                call['status'] = 'interrupted'
                call['charged_tokens'] = cap
            raise
        finally:
            if stream is not None and hasattr(stream, 'close'):
                stream.close()
            if old_timeout is not None:
                client.timeout = old_timeout
            call['seconds'] = time.monotonic() - started
            self._save()
            self.emit('end', '')

    def _result(self):
        # A stopped request may already contain useful written material. Publish
        # that exact evidence now; resume still reuses its call without dispatch.
        pending = self.state.get('pending')
        if pending and pending['phase'] == 'solve' and self.state['status'] != 'needs_recovery':
            call = next((c for c in self.state['calls'] if c['key'] == self._call_key('solver')), None)
            if call is not None:
                try:
                    result = self._read_stream(call['stream'])
                except AgentError:
                    result = None  # Keep malformed transport evidence in its journal.
                if result is not None and (result['text'] or result['thinking']):
                    self._candidate(result, pending)
        selected_id, initial_id = self.state['selected_candidate'], self.state['initial_candidate']
        selected = self._get_candidate(selected_id) if selected_id else None
        initial = self._get_candidate(initial_id) if initial_id else None
        answer = self._candidate_text(selected) if selected else ''
        lines = ['# Square Harness proof run', '', f'Status: **{self.state["status"]}**', '',
                 self.state.get('stop_reason', ''), '',
                 'Model reviews are fallible assessments, not formal proof certificates.', '',
                 f'Generated tokens charged: {self.state["tokens_charged"]} / {self.state["settings"]["max_tokens"]}.',
                 f'Elapsed active seconds charged: {self.state["seconds_used"]:.1f}.', '',
                 '## Original problem', '', self.state['goal'], '', '## Candidates', '']
        for candidate in self.state['candidates']:
            marker = ' (selected)' if candidate['id'] == selected_id else ''
            lines.append(f'- [{candidate["id"]}](artifacts/{candidate["artifact"]}){marker}: '
                         f'{candidate["review_status"]}; transport complete={candidate["transport_complete"]}; SHA-256 `{candidate["sha256"]}`.')
        lines += ['', 'The initial candidate is retained. A finished written response is preferred to an unfinished fragment; once a finished candidate exists, only a whole-proof no_issue_found verdict replaces it. '
                  'Earlier objections below remain attributed to their candidate; a later pass does not erase their history.', '']
        for selection in self.state.get('selection_history', []):
            lines.append(f'- Selection: {selection["previous"] or "none"} → {selection["selected"]}; {selection["reason"]}.')
        lines += ['', '## Reviews', '']
        for review in self.state['reviews']:
            lines += [f'### {review["id"]} on {review["candidate"]}', '']
            if review['response'] is None:
                lines += ['Review unavailable: ' + review['protocol_error'], '']
                continue
            response = review['response']
            lines += [response['verdict'] + ': ' + response['explanation'], '']
            for issue in response['issues']:
                lines += [f'- **{issue["kind"]}**, {issue["location"]}: {issue["evidence"]}']
            lines += ['']
        report = '\n'.join(lines)
        self.store.write_report(report)
        self.store.write_proof(answer)
        return {'id': self.state['id'], 'status': self.state['status'], 'stop_reason': self.state.get('stop_reason', ''),
                'directory': str(self.store.directory), 'report_path': str(self.store.directory / 'report.md'),
                'proof_path': str(self.store.directory / 'proof.md'), 'answer': answer, 'report': report,
                'initial_candidate': copy.deepcopy(initial), 'selected_candidate': copy.deepcopy(selected),
                'tokens_charged': self.state['tokens_charged'], 'seconds_used': self.state['seconds_used'],
                'rounds_started': self.state['rounds_started']}
