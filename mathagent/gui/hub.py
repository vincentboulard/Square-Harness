"""Bridge the synchronous engine to browser clients: events, one worker, approvals.

The engine is unchanged. For each task the hub builds the same objects as the
terminal interface, forwards emit() callbacks as events, answers approve()
callbacks from the browser, and pauses work by raising the KeyboardInterrupt
that the runners already treat as a checkpointed pause. The hub serves one
top-level task at a time; Assistant workers may share bounded concurrent
inference within that task. Further job starts wait in a queue.
"""
import collections
import json
from pathlib import Path
import threading
import time
import uuid

from ..agent import Agent, AgentError, Ollama
from ..assistant_budget import AssistantBudget, BudgetClient
from ..backends import LlamaCpp, OpenAICompatible, create_client
from ..ledger import ProofStore
from ..literature import LiteratureTools
from ..proof import ProofRunner
from ..litreview import ReviewRunner
from ..review import ReviewBase, runner_class as review_runner, runner_for_state
from ..refcheck import EFFORTS as REFCHECK_EFFORTS, CheckRunner, nested_checker
from ..research import ResearchRunner
from ..router import JOB_MODES
from ..orchestrator import Orchestrator
from ..tools import Workspace
from ..writeup import WriteupRunner
from ..worker_errors import WorkerInputError
from . import effort, store
from .store import one_line

RUNNING = ('starting', 'running', 'pausing')
# Task labels shown in the interface; the engine's kind stays 'referee'.
LABELS = {'literature': 'Literature report', 'referee': 'Review', 'writeup': 'Write-up', 'check': 'Literature check',
          'quick_review': 'Review · quick check', 'explain': 'Review · explanation'}
# Assistant and router modes that run as review variants of the referee kind.
REVIEW_MODES = {'referee': 'review', 'detailed_review': 'journal', 'quick_review': 'quick', 'explain': 'explain'}
LABELS['detailed_review'] = 'Review · detailed'
VARIANT_LABELS = {'review': 'Review', 'journal': LABELS['detailed_review'], 'quick': LABELS['quick_review'], 'explain': LABELS['explain']}


class Busy(Exception):
    """The model serves one task at a time; another task is still running."""


class EventBus:
    """Fan-out of JSON events with a bounded replay buffer for reconnects."""

    def __init__(self, size=4000):
        self.instance = uuid.uuid4().hex[:12]
        self._cond = threading.Condition()
        self._events = collections.deque(maxlen=size)
        self._seq = 0

    @property
    def seq(self):
        with self._cond:
            return self._seq

    def publish(self, type_, mutate=None, **data):
        # mutate() runs under the same lock as the event, so a snapshot taken
        # with snapshot() never double-counts or misses a published change.
        with self._cond:
            if mutate:
                mutate()
            self._seq += 1
            event = {'seq': self._seq, 'type': type_, **data}
            self._events.append(event)
            self._cond.notify_all()
            return event

    def snapshot(self, read):
        with self._cond:
            return self._seq, read()

    def wait(self, after, timeout):
        """Events after sequence number `after`; missed=True if some were dropped."""
        with self._cond:
            if self._seq <= after:
                self._cond.wait(timeout)
            if after > self._seq:
                return [], True
            oldest = self._events[0]['seq'] if self._events else self._seq + 1
            return [e for e in self._events if e['seq'] > after], after + 1 < oldest


class Interruptible:
    """Model client mixin that turns a pause request into the engine's pause signal.

    The check runs between streamed events, which is where Ctrl+C normally
    lands in the terminal. Tools already running finish first.
    """

    def __init__(self, host, cancel, timeout=600):
        super().__init__(host, timeout)
        self.cancel = cancel

    def stream(self, payload):
        if self.cancel.is_set():
            raise KeyboardInterrupt
        events = super().stream(payload)
        try:
            for event in events:
                if self.cancel.is_set():
                    raise KeyboardInterrupt
                yield event
        finally:
            events.close()


# Subclasses rather than wrappers: the runners set client.timeout for their time guards.
class InterruptibleOllama(Interruptible, Ollama):
    pass


class InterruptibleOpenAI(Interruptible, OpenAICompatible):
    pass


class InterruptibleLlamaCpp(Interruptible, LlamaCpp):
    pass


INTERRUPTIBLE = {'ollama': InterruptibleOllama, 'openai': InterruptibleOpenAI, 'llamacpp': InterruptibleLlamaCpp}


class Task:
    def __init__(self, kind, target, label, title='', on_update=None):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.target, self.label, self.title = kind, target, label, title
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.state = 'starting'
        self.error = None
        self.result_status = None
        self.started = time.time()
        self.finished = None
        self.activity = collections.deque(maxlen=200)
        self.live = None
        self.thread = None
        self.work = None
        self.on_update = on_update
        self.origin_chat = target if kind == 'chat' else None

    def summary(self):
        return {'id': self.id, 'kind': self.kind, 'target': self.target, 'label': self.label,
                'title': self.title[:500], 'state': self.state, 'error': self.error,
                'result_status': self.result_status, 'started': self.started, 'finished': self.finished,
                'origin_chat': self.origin_chat}


class Approval:
    def __init__(self, task, preview):
        self.id = uuid.uuid4().hex
        self.task = task
        self.preview = preview
        self.kind = ('write' if preview.startswith('WRITE ') else
                     'python' if preview.startswith('RUN PYTHON') else 'confirm')
        self.decided = threading.Event()
        self.approved = False

    def summary(self):
        return {'id': self.id, 'task': self.task.id, 'task_kind': self.task.kind,
                'target': self.task.target, 'kind': self.kind, 'preview': self.preview}


class ChatCapture:
    """Collects one chat turn from emit() for live display and the saved transcript."""

    FLUSH = 0.08

    def __init__(self, hub, task, chat_id, agent):
        self.hub, self.task, self.chat_id, self.agent = hub, task, chat_id, agent
        self.calls, self.notices = [], []
        self._buffer = {'thinking': '', 'text': ''}
        self._flushed = 0.0

    def _publish(self, kind, **data):
        steps = self.task.live['steps']

        def mutate():
            if kind == 'start':
                steps.append({'type': 'call', 'thinking': '', 'text': ''})
            elif kind == 'delta':
                step = steps[-1] if steps and steps[-1]['type'] == 'call' else None
                if step is not None:
                    step['thinking'] += data.get('thinking', '')
                    step['text'] += data.get('text', '')
            elif kind in ('tool', 'result', 'notice'):
                steps.append({'type': kind, 'text': data['text']})
        self.hub.bus.publish('chat', mutate=mutate, chat=self.chat_id, task=self.task.id, kind=kind, **data)

    def _flush(self):
        if self._buffer['thinking'] or self._buffer['text']:
            self._publish('delta', **self._buffer)
            self._buffer = {'thinking': '', 'text': ''}
        self._flushed = time.monotonic()

    def emit(self, kind, value):
        if kind == 'start':
            self._flush()
            self.calls.append({'thinking': '', 'stats': {}})
            self._publish('start')
        elif kind in ('thinking', 'text'):
            if kind == 'thinking' and self.calls:
                self.calls[-1]['thinking'] += value
            self._buffer[kind] += value
            if time.monotonic() - self._flushed >= self.FLUSH:
                self._flush()
        elif kind == 'end':
            self._flush()
            if self.calls:
                self.calls[-1]['stats'] = dict(self.agent.last_stats)
            self._publish('end')
        else:
            self._flush()
            text = str(value)[:4000]
            if kind == 'notice':
                self.notices.append(text)
            self._publish(kind if kind in ('tool', 'result', 'notice') else 'notice', text=text)


class Hub:
    def __init__(self, args, bus=None):
        self.args = args
        self.root = Path(args.workspace).resolve(strict=True)
        # Folders can be browsed and opened at or below this root only.
        self.base = Path(getattr(args, 'gui_root', None) or self.root).resolve(strict=True)
        if not self.root.is_relative_to(self.base):
            raise ValueError('The workspace must be inside --gui-root')
        self.bus = bus or EventBus()
        # An explicit --offline at launch is a hard guarantee (for unaided work);
        # otherwise online search is on, and the user may turn it off per conversation or job.
        self.online_locked = bool(getattr(args, 'offline', False))
        self._lock = threading.Lock()
        self.task = None
        self.queue = collections.deque()
        self.routing = {}
        self._approvals = {}
        self._models = (0.0, None)

    # -- task lifecycle -------------------------------------------------

    def current(self):
        with self._lock:
            return self.task

    def active(self):
        task = self.current()
        return task if task and task.state in RUNNING else None

    def _publish_task(self, task):
        self.bus.publish('task', task=task.summary())
        if task.on_update:
            try:
                task.on_update(task)
            except (OSError, ValueError, store.NotFound):
                pass

    def _publish_queue(self):
        self.bus.publish('queue', queue=[t.summary() for t in list(self.queue)])

    def _begin(self, kind, target, label, title, work, live=None, queue=False, on_update=None):
        with self._lock:
            origin_chat = target if kind == 'chat' else getattr(on_update, '_origin_chat', None)
            if origin_chat:
                # Registration and deletion share this lock. An earlier read
                # of the chat does not authorize work after it was deleted.
                store.load_chat(self.root, origin_chat)
            busy = self.task is not None and self.task.state in RUNNING
            if busy and not queue:
                raise Busy(f'The model is busy with a {self.task.label.lower()} ("{one_line(self.task.title, 60)}"). '
                           'Pause it or wait until it finishes: the model serves one task at a time '
                           'and time budgets count wall-clock time.')
            task = Task(kind, target, label, title, on_update)
            task.origin_chat = origin_chat
            task.live, task.work = live, work
            if busy:
                task.state = 'queued'
                self.queue.append(task)
            else:
                self.task = task
        if task.state == 'queued':
            self._publish_queue()
            self._publish_task(task)
        else:
            self._launch(task)
        return task

    def _launch(self, task):
        task.thread = threading.Thread(target=self._run, args=(task, task.work), name='square-' + task.kind, daemon=True)
        task.thread.start()

    def _run(self, task, work):
        if task.state == 'starting':  # a pause may already have arrived
            task.state = 'running'
        task.started = time.time()
        self._publish_task(task)
        try:
            result = work(task)
            task.result_status = result.get('status') if isinstance(result, dict) else None
            task.state = 'paused' if task.result_status == 'paused' and task.cancel.is_set() else 'done'
        except KeyboardInterrupt:
            task.state = 'paused'
        except (AgentError, OSError, ValueError) as exc:
            task.state, task.error = 'error', str(exc)
        except Exception as exc:  # keep the server alive and report the failure
            task.state, task.error = 'error', f'{type(exc).__name__}: {exc}'
        finally:
            task.finished = time.time()
            task.ready.set()
            self._publish_task(task)
            self._advance()

    def _advance(self):
        """Start the next queued job once the model is free."""
        with self._lock:
            if (self.task is not None and self.task.state in RUNNING) or not self.queue:
                return
            task = self.queue.popleft()
            task.state = 'starting'
            self.task = task
        self._publish_queue()
        self._launch(task)

    def cancel_queued(self, task_id):
        with self._lock:
            task = next((t for t in self.queue if t.id == task_id), None)
            if task is None:
                raise store.NotFound('That job is no longer waiting')
            self.queue.remove(task)
            task.state = 'cancelled'
            task.ready.set()
        self._publish_queue()
        self._publish_task(task)
        return task

    def pause(self):
        task = self.active()
        if task is None:
            raise ValueError('No task is running')
        task.cancel.set()
        task.state = 'pausing'
        self._publish_task(task)
        return task

    def shutdown(self, wait=20):
        with self._lock:
            self.queue.clear()
        task = self.active()
        if task:
            task.cancel.set()
            task.thread.join(wait)

    def snapshot(self):
        """Current task, its live chat turn, the queue and pending approvals, with the event seq."""
        def read():
            task = self.task
            return {'task': task.summary() if task else None,
                    'activity': list(task.activity) if task else [],
                    'live': _copy(task.live) if task and task.live else None,
                    'approvals': [a.summary() for a in self._approvals.values()],
                    'queue': [t.summary() for t in list(self.queue)],
                    'routing': sorted(self.routing)}
        seq, value = self.bus.snapshot(read)
        return {'seq': seq, **value}

    def delete_chat(self, chat_id):
        """Delete one idle conversation after excluding every possible writer."""
        with self._lock:
            chat = store.load_chat(self.root, chat_id)
            route_tasks = {item.get('task') for item in chat['transcript']
                           if item.get('role') == 'route' and item.get('task')}
            tasks = ([self.task] if self.task else []) + list(self.queue)
            def belongs(task):
                return (getattr(task, 'origin_chat', None) == chat_id
                        or (task.kind == 'chat' and task.target == chat_id)
                        or (task.live or {}).get('chat') == chat_id or task.id in route_tasks)
            if chat_id in self.routing or any(belongs(task) and (
                    task.state in RUNNING + ('queued',)
                    or (task.thread is not None and task.thread.is_alive())) for task in tasks):
                raise Busy('This conversation is still running or queued. Pause it and wait for it to stop, or cancel its queued work before deleting it.')
            store.delete_chat(self.root, chat_id)
            if self.task and belongs(self.task):
                self.task = None
        self.bus.publish('chat_deleted', chat=chat_id)
        return {'deleted': chat_id}

    # -- workspace --------------------------------------------------------

    def set_workspace(self, relative):
        target = store.resolve_folder(self.base, relative)
        with self._lock:
            if (self.task is not None and self.task.state in RUNNING) or self.queue:
                raise Busy('Pause the running job and clear the queue before opening another folder.')
            self.root = target
            self.args.workspace = target
        store.remember_folder(self.base, target.relative_to(self.base).as_posix())
        return target

    def workspace_files(self):
        return [f['path'] for f in store.list_files(self.root, limit=400)]

    def check_files(self, files, pdf=True):
        """Workspace-relative files that exist; used before pinning them."""
        if not isinstance(files, (list, tuple)) or len(files) > 20:
            raise ValueError('Give at most 20 files')
        workspace = Workspace(self.root)
        checked = []
        for name in files:
            path = workspace.path(name, pdf=pdf)
            if not path.is_file():
                raise ValueError(f'{name} is not a file in this folder')
            checked.append(path.relative_to(self.root).as_posix())
        return list(dict.fromkeys(checked))

    # -- engine construction (mirrors the terminal interface) ----------

    def online(self, requested=None):
        """Whether new work may search the web: on unless the user turns it off or launched --offline."""
        if self.online_locked:
            if requested:
                raise ValueError('This interface was launched with --offline, so online search stays off')
            return False
        return True if requested is None else bool(requested)

    def _engine(self, task, *, mode='prove', think=None, ctx=None, online=None):
        args = self.args
        library = LiteratureTools(self.root, online=self.online(online),
                                  max_requests=args.research_requests, max_chars=args.research_chars)
        workspace = Workspace(self.root, lambda preview: self._approve(task, preview),
                              args.allow_python, literature=library, read_types=store.read_types(self.root))
        client = INTERRUPTIBLE[args.backend](args.host, task.cancel, timeout=args.request_timeout)
        agent = Agent(client, workspace, args.model, ctx or args.ctx, args.predict,
                      (not args.no_think) if think is None else bool(think), mode, args.max_rounds,
                      seed=args.seed, temperature=args.temperature, top_p=args.top_p)
        return agent, library

    def _ensure_model(self, agent):
        models = agent.client.models()
        if agent.model not in models:
            hint = (f'Run: ollama pull {agent.model}, or restart with --model and an exact tag from ollama list.'
                    if self.args.backend == 'ollama' else
                    'Start the server with this served model name, or restart with --model and an available model ID.')
            raise AgentError(f'Model {agent.model!r} is not served. Available: {", ".join(models) or "none"}. ' + hint)

    def models(self):
        """Served models, cached briefly so status polling never hammers the model server."""
        stamp, value = self._models
        if time.monotonic() - stamp < 10 and value is not None:
            return value
        try:
            value = {'reachable': True, 'models': create_client(self.args.backend, self.args.host, timeout=3).models()}
        except AgentError as exc:
            value = {'reachable': False, 'models': [], 'error': str(exc)}
        self._models = (time.monotonic(), value)
        return value

    def _approve(self, task, preview):
        approval = Approval(task, str(preview))
        self.bus.publish('approval', mutate=lambda: self._approvals.__setitem__(approval.id, approval),
                         state='pending', approval=approval.summary())
        # Like the terminal prompt this waits for an answer; a pause denies it.
        while not approval.decided.wait(0.25):
            if task.cancel.is_set():
                break
        self.bus.publish('approval', mutate=lambda: self._approvals.pop(approval.id, None),
                         state='approved' if approval.approved else 'denied',
                         approval={'id': approval.id, 'task': task.id})
        return approval.approved

    def decide(self, approval_id, approved):
        with self.bus._cond:
            approval = self._approvals.get(approval_id)
        if approval is None:
            raise store.NotFound('This approval is no longer pending')
        approval.approved = bool(approved)
        approval.decided.set()

    def _job_emit(self, task, runner, kind):
        def emit(event, value):
            if task.target is None:
                ident = (runner.store.state['id'] if kind == 'proof' and runner.store is not None else
                         runner.state['id'] if kind == 'research' and runner.state is not None else None)
                if ident:
                    task.target = ident
                    task.ready.set()
                    self._publish_task(task)
            # Streams are published by the watcher from the saved .jsonl files,
            # identically for jobs started here or in a terminal.
            if event in ('notice', 'tool', 'result'):
                item = {'kind': event, 'text': str(value)[:4000], 'time': time.time()}
                self.bus.publish('activity', mutate=lambda: task.activity.append(item),
                                 task=task.id, job=kind, id=task.target, **item)
        return emit

    def _started(self, task):
        if task.state != 'queued':
            task.ready.wait(30)
        if task.target is None and task.error:
            raise ValueError(task.error)
        return task

    # -- proofs --------------------------------------------------------

    def start_proof(self, goal, *, source_files=(), rounds=None, tokens=None, seconds=None,
                    solve_tokens=None, verify_tokens=None, ctx=None, queue=False, on_update=None):
        # Proof work always thinks and has no tools: the model sees the goal and
        # the pinned text only, exactly as with the terminal's /prove.
        args = self.args
        rounds = args.proof_rounds if rounds is None else rounds
        tokens = args.proof_tokens if tokens is None else tokens
        seconds = args.proof_seconds if seconds is None else seconds
        solve = args.proof_solve_tokens if solve_tokens is None else solve_tokens
        verify = args.proof_verify_tokens if verify_tokens is None else verify_tokens
        _check_ctx(ctx, args.predict)
        _check_proof(args, tokens=tokens, solve=solve, verify=verify, ctx=ctx or args.ctx)

        def work(task):
            agent, _ = self._engine(task, mode='prove', ctx=ctx, online=False)
            self._ensure_model(agent)
            runner = ProofRunner(agent)
            runner.emit = self._job_emit(task, runner, 'proof')
            return runner.start(goal, max_rounds=rounds, max_tokens=tokens, max_seconds=seconds,
                                max_predict=solve, verify_tokens=verify, min_solve_tokens=args.proof_min_solve_tokens,
                                repair_tokens=args.proof_repair_tokens, verify_temperature=args.proof_verify_temperature,
                                source_files=list(source_files))
        return self._started(self._begin('proof', None, 'Proof', goal, work, queue=queue, on_update=on_update))

    def resume_proof(self, proof_id, queue=False):
        state = ProofStore.load(self.root, proof_id).state
        if state['version'] != 2:
            raise ValueError('This proof was made by the earlier v0.4 engine. It stays readable here, '
                             'but v0.5 cannot resume it: start a new proof instead.')

        def work(task):
            agent, _ = self._engine(task, online=False)
            runner = ProofRunner(agent)
            runner.emit = self._job_emit(task, runner, 'proof')
            return runner.resume(proof_id)
        task = self._begin('proof', proof_id, 'Proof', state['goal'], work, queue=queue)
        task.ready.set()
        return task

    # -- literature, referee and write-up reports -----------------------

    def _research_budgets(self, rounds=None, tokens=None, input_tokens=None, seconds=None, requests=None, chars=None):
        args = self.args
        return dict(max_rounds=args.research_rounds if rounds is None else rounds,
                    max_tokens=args.research_tokens if tokens is None else tokens,
                    max_input_tokens=args.research_input_tokens if input_tokens is None else input_tokens,
                    max_seconds=args.research_seconds if seconds is None else seconds,
                    max_requests=args.research_requests if requests is None else requests,
                    max_chars=args.research_chars if chars is None else chars)

    def review_concurrency(self):
        """Independent review calls run together only on a server that batches requests (vLLM)."""
        return getattr(self.args, 'assistant_concurrency', 2) if self.args.backend == 'openai' else 1

    def start_research(self, kind, goal, *, source_files=(), online=None, queue=False, on_update=None,
                       variant=None, target='', **limits):
        if kind not in ('literature', 'referee'):
            raise ValueError('Research kind must be literature or referee')
        budgets = self._research_budgets(**limits)
        online = self.online(online)
        variant = (variant or 'review') if kind == 'referee' else None

        def work(task):
            agent, _ = self._engine(task, mode=kind, online=online)
            self._ensure_model(agent)
            if kind == 'literature':  # a verified, themed reading list
                runner = ReviewRunner(agent)
                runner.emit = self._job_emit(task, runner, 'research')
                return runner.start(goal, source_files=list(source_files), **budgets)
            runner = review_runner(variant)(agent, concurrency=self.review_concurrency())
            runner.emit = self._job_emit(task, runner, 'research')
            return runner.start(goal, source_files=list(source_files), target=target, **budgets)
        label = VARIANT_LABELS[variant] if variant else LABELS[kind]
        return self._started(self._begin('research', None, label, goal, work, queue=queue, on_update=on_update))

    def start_writeup(self, goal, *, source_files=(), template_files=(), template=None, save_template=None,
                      notes='', output='', queue=False, on_update=None, **limits):
        budgets = self._research_budgets(**limits)
        texts = store.load_template(self.base, template) if template else []
        if save_template:
            store.save_template(self.base, Workspace(self.root), save_template, list(template_files))

        def work(task):
            # A write-up works from the pinned notes only; it never searches.
            agent, _ = self._engine(task, mode='writeup', online=False)
            self._ensure_model(agent)
            runner = WriteupRunner(agent)
            runner.emit = self._job_emit(task, runner, 'research')
            return runner.start(goal, source_files=list(source_files), template_files=list(template_files),
                                template_texts=texts, notes=notes, output=output, **budgets)
        return self._started(self._begin('research', None, LABELS['writeup'], goal, work, queue=queue, on_update=on_update))

    def resume_research(self, job_id, queue=False):
        state = store.research_state(self.root, job_id)
        runner_class = (WriteupRunner if state['kind'] == 'writeup' else
                        ReviewRunner if state.get('pipeline') == ReviewRunner.PIPELINE else runner_for_state(state))
        # A job keeps the network choice it started with (never more, per the runner).
        online = bool(state['settings'].get('online')) and not self.online_locked

        def work(task):
            agent, _ = self._engine(task, mode=state['kind'], online=online)
            runner = (runner_class(agent, concurrency=self.review_concurrency()) if issubclass(runner_class, ReviewBase)
                      else runner_class(agent))
            runner.emit = self._job_emit(task, runner, 'research')
            return runner.resume(job_id)
        label = VARIANT_LABELS.get(state.get('variant')) or LABELS.get(state['kind'], LABELS['writeup'])
        task = self._begin('research', job_id, label, state['goal'], work, queue=queue)
        task.ready.set()
        return task

    # -- conversational modes -----------------------------------------

    def chat(self, chat_id, content, *, files=(), mode=None, echo_user=True, on_update=None, queue=False):
        session = store.load_chat(self.root, chat_id)
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Write a message first')
        if len(content) > 60000:
            raise ValueError('Messages are limited to 60000 characters; pin long material as a file instead')
        mode = mode or session['mode']
        if mode not in ('critic', 'explore'):
            raise ValueError('Conversation turns use the critic or explore mode')
        files = self.check_files(list(files))
        prompt = content + (('\n\nFiles provided by the user (read them with read_file): ' + ', '.join(files)) if files else '')

        def work(task):
            # Reload under the one-task rule: the previous turn may have just been saved.
            fresh = store.load_chat(self.root, chat_id)
            agent, _ = self._engine(task, mode=mode, think=fresh['settings'].get('think'),
                                    online=self.online(fresh['settings'].get('online', False)) if not self.online_locked else False)
            agent.history = fresh['history']
            self._ensure_model(agent)
            capture = ChatCapture(self, task, chat_id, agent)
            if agent.workspace.literature.online:
                agent.workspace.checker = nested_checker(agent, capture.emit)
            try:
                answer = agent.run(prompt, capture.emit)
            except BaseException as exc:
                self._discarded(chat_id, content if echo_user else '', exc)
                raise
            store.commit_turn(self.root, chat_id, agent.history, capture.calls, capture.notices,
                              echo_user=echo_user, mode=mode)
            self.bus.publish('chat_saved', chat=chat_id)
            return {'status': 'done', 'answer': answer}
        live = {'chat': chat_id, 'kind': 'message', 'user': content if echo_user else '', 'steps': [], 'mode': mode}
        label = 'Critique' if mode == 'critic' else 'Exploration'
        return self._begin('chat', chat_id, label, content, work, live, queue=queue, on_update=on_update)

    def check(self, chat_id, question, *, effort='medium', on_update=None, queue=False):
        """A literature check answered in a Default conversation (its evidence stays in .mathagent/checks)."""
        store.load_chat(self.root, chat_id)
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise ValueError('Describe the result to find a reference for, in at most 8000 characters')
        root = self.root

        def work(task):
            fresh = store.load_chat(root, chat_id)
            online = bool(fresh['settings'].get('online')) and not self.online_locked
            agent, _ = self._engine(task, mode='check', online=online)
            self._ensure_model(agent)
            capture = ChatCapture(self, task, chat_id, agent)

            def emit(kind, value):
                # Steps and tool calls only: raw search results would flood the conversation.
                if kind in ('notice', 'tool'):
                    capture.emit(kind, value)
            runner = CheckRunner(agent, emit=emit)
            try:
                result = runner.start(question, effort=effort)
            except BaseException as exc:
                self._discarded(chat_id, '', exc)
                raise
            state = runner.state
            answer = state.get('answer') or runner._compose()
            if state['status'] in ('paused', 'error', 'budget_exhausted', 'budget_violation'):
                answer = f'*The check stopped early ({state.get("stop_reason", state["status"])}).*\n\n' + answer
            if not online:
                answer = '*Online search is off for this conversation, so only cached sources were used.*\n\n' + answer
            answer += f'\n\nFull log: `.mathagent/checks/{result["id"]}/report.md`'
            history = fresh['history'] + [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
            store.commit_turn(root, chat_id, history, [], [], echo_user=False, mode='check')
            self.bus.publish('chat_saved', chat=chat_id)
            return {'status': 'done', 'answer': answer}
        live = {'chat': chat_id, 'kind': 'message', 'user': '', 'steps': [], 'mode': 'check'}
        return self._begin('chat', chat_id, LABELS['check'], question, work, live, queue=queue, on_update=on_update)

    def review(self, chat_id):
        def last_exchange(session):
            history = session['history']
            if not history or history[-1].get('role') != 'assistant':
                raise ValueError('Ask a question and receive an answer first')
            return next(m['content'] for m in reversed(history) if m['role'] == 'user'), history[-1]['content']
        session = store.load_chat(self.root, chat_id)
        last_user, _ = last_exchange(session)

        def work(task):
            # A fresh critic context, as /review does in the terminal.
            fresh = store.load_chat(self.root, chat_id)
            last_user, last_answer = last_exchange(fresh)
            agent, _ = self._engine(task, mode='critic', think=fresh['settings'].get('think'),
                                    online=fresh['settings'].get('online', False) and not self.online_locked)
            self._ensure_model(agent)
            capture = ChatCapture(self, task, chat_id, agent)
            try:
                result = agent.run('Audit this proposed answer. Re-read relevant files if needed.\n'
                                   f'QUESTION:\n{last_user}\n\nANSWER:\n{last_answer}', capture.emit)
            except BaseException as exc:
                self._discarded(chat_id, '', exc)
                raise
            store.commit_review(self.root, chat_id, result, capture.calls, capture.notices)
            self.bus.publish('chat_saved', chat=chat_id)
            return {'status': 'done', 'answer': result}
        live = {'chat': chat_id, 'kind': 'review', 'user': '', 'steps': [], 'mode': 'critic'}
        return self._begin('chat', chat_id, 'Fresh review', last_user, work, live)

    def _discarded(self, chat_id, content, exc):
        # As in the terminal, an unfinished turn never enters the model context;
        # unlike a terminal, a chat window should still show what happened.
        reason = ('it was paused before the answer finished' if isinstance(exc, KeyboardInterrupt)
                  else str(exc) or type(exc).__name__)
        try:
            store.record_discarded(self.root, chat_id, content, reason)
            self.bus.publish('chat_saved', chat=chat_id)
        except (OSError, ValueError):
            pass

    # -- Assistant: bounded main agent and existing workers -----------------

    def route(self, chat_id, content, files=()):
        return self._assistant_turn(chat_id, content, files)

    def resume_assistant(self, chat_id):
        chat = store.load_chat(self.root, chat_id)
        state = chat.get('assistant_state') or {}
        if state.get('status') not in ('interrupted', 'running', 'incomplete'):
            raise ValueError('This Assistant turn is not paused or recoverable.')
        return self._assistant_turn(chat_id, state['query'], state.get('files', ()), resume=True)

    def _assistant_turn(self, chat_id, content, files=(), resume=False):
        chat = store.load_chat(self.root, chat_id)
        if chat['mode'] != 'free':
            raise ValueError('Only Assistant conversations orchestrate workers.')
        if not isinstance(content, str) or not content.strip() or len(content) > 60000:
            raise ValueError('Write a message of at most 60000 characters.')
        files = self.check_files(list(files))
        saved = chat.get('assistant_state') or {}
        with self._lock:
            store.load_chat(self.root, chat_id)
            pending = ([self.task] if self.task else []) + list(self.queue)
            if chat_id in self.routing or any(t.target == chat_id and t.state in RUNNING + ('queued',) for t in pending):
                raise Busy('The Assistant is still working on this conversation.')
            if not resume and saved.get('status') in ('interrupted', 'running', 'incomplete'):
                raise ValueError('Resume the pending Assistant turn first, or start a new conversation.')
            self.routing[chat_id] = time.time()
            try:
                if not resume:
                    store.append_items(self.root, chat_id, [{'role': 'user', 'content': content, 'files': files, 'time': store._now()}])
            except BaseException:
                self.routing.pop(chat_id, None)
                raise
        root = self.root
        if not resume:
            self.bus.publish('chat_saved', chat=chat_id)

        def work(task):
            fresh = store.load_chat(root, chat_id)
            previous = fresh.get('assistant_state') if resume else None
            online = bool(fresh['settings'].get('online')) and not self.online_locked
            if previous:
                online = online and bool(previous.get('online', False))
            main, _ = self._engine(task, mode='explore', think=False, online=online)
            main.workspace.pin_files(files)
            self._ensure_model(main)
            checkpoint_lock = threading.RLock()
            parent = None
            budget = None

            def checkpoint(state=None):
                with checkpoint_lock:
                    if parent is None or parent.state is None:
                        return
                    # Child usage and the shared total form one checkpoint;
                    # never combine an earlier child copy with a newer refund.
                    with budget.pool.lock:
                        state = _copy(parent.state)
                        state['budget'] = budget.snapshot()
                        state['online'] = online
                        state['web_budget'] = main.workspace.literature.snapshot()
                        store.save_assistant(root, chat_id, state)
                    self.bus.publish('chat_saved', chat=chat_id)

            concurrency = getattr(self.args, 'assistant_concurrency', 2) if self.args.backend == 'openai' else 1
            budget = BudgetClient(main.client, state=previous.get('budget') if previous else None,
                                  max_tokens=getattr(self.args, 'assistant_tokens', 60000),
                                  max_input_tokens=getattr(self.args, 'assistant_input_tokens', 240000),
                                  max_seconds=getattr(self.args, 'assistant_seconds', 900),
                                  checkpoint=checkpoint, concurrency=concurrency)
            main.client = budget
            library = main.workspace.literature
            if previous and previous.get('web_budget'):
                library.restore(previous['web_budget'])
            library.deadline = time.monotonic() + budget.remaining_seconds
            library.on_budget_change = checkpoint
            capture = ChatCapture(self, task, chat_id, main)
            def capture_main(kind, value):
                if kind == 'tool':
                    name = str(value).split(' ', 1)[0]
                    message = ('Delegating a mathematical task…' if name == 'delegate' else
                               'Reading web sources…' if name in ('open_url', 'read_page', 'search_page', 'read_references', 'search_web')
                               else 'Reading mathematical sources…')
                    capture.emit('notice', message)
                elif kind != 'result':
                    capture.emit(kind, value)
            parent = Orchestrator(main, lambda action, child, save: self._assistant_worker(
                task, root, chat_id, parent, budget, action, child, save, online, checkpoint_lock),
                emit=capture_main, checkpoint=checkpoint, concurrency=concurrency)
            checkpoint_lock = parent._state_lock
            history = fresh['history']
            if not history:
                # Legacy chats remain useful, without the old character-capped summaries.
                history = [{'role': i['role'], 'content': i.get('content', '')} for i in fresh['transcript']
                           if i.get('role') in ('user', 'assistant')]
                if not resume and history and history[-1]['role'] == 'user':
                    history.pop()  # the current message is added by the controller
            try:
                result = parent.run(content, files=files, history=history, state=previous)
            except KeyboardInterrupt:
                checkpoint()
                store.append_items(root, chat_id, [{'role': 'notice', 'content': 'Assistant paused. Its workers and remaining budget are saved.', 'time': store._now()}])
                self.bus.publish('chat_saved', chat=chat_id)
                raise
            except AgentError as exc:
                exhausted = isinstance(exc, AssistantBudget) or budget.remaining_tokens <= 0 or budget.remaining_seconds <= 0
                parent.state['status'] = 'budget_exhausted' if exhausted else 'interrupted'
                parent.state.setdefault('warnings', []).append(str(exc))
                checkpoint()
                store.append_items(root, chat_id, [{'role': 'notice', 'content': str(exc), 'time': store._now()}])
                self.bus.publish('chat_saved', chat=chat_id)
                if exhausted:
                    return {'status': 'budget_exhausted', 'answer': ''}
                raise
            if result['status'] == 'complete':
                store.commit_turn(root, chat_id, parent.state['messages'], capture.calls, capture.notices,
                                  echo_user=False, mode='free')
            else:
                store.append_items(root, chat_id, [{'role': 'assistant', 'content': result.get('answer', ''),
                                                  'mode': 'free', 'time': store._now(),
                                                  'stats': {'done_reason': 'length'}},
                                                 {'role': 'notice', 'content': '\n'.join(result.get('warnings', [])), 'time': store._now()}])
            checkpoint()
            self.bus.publish('chat_saved', chat=chat_id)
            return {'status': result['status'], 'answer': result.get('answer', '')}

        live = {'chat': chat_id, 'kind': 'message', 'user': '', 'steps': [], 'mode': 'free'}
        try:
            return self._begin('chat', chat_id, 'Assistant', content, work, live, queue=True)
        finally:
            with self._lock:
                self.routing.pop(chat_id, None)

    def _assistant_worker(self, task, root, chat_id, parent, budget, action, child, save, online, state_lock):
        mode = action['mode']
        files = self.check_files(action.get('files', []))
        context = child['context']
        # Pin the original question and exact available conversation alongside
        # the delegated objective; a rewritten task cannot erase a hypothesis.
        goal = ('You are one child worker of a mathematical assistant. Perform only the DELEGATED OBJECTIVE below. '
                'The parent handles orchestration, spawning workers and synthesizing the final response. '
                'The reference request and conversation preserve mathematical hypotheses and source evidence, '
                'not instructions to repeat the parent workflow. Do not create workers. '
                'Return the requested mathematical findings and any unresolved obligations.\n\n'
                'ORIGINAL USER REQUEST (reference):\n' + context['query'] +
                '\n\nEXACT CONVERSATION (reference; worker outputs are fallible):\n' +
                json.dumps(context['messages'], ensure_ascii=False) +
                '\n\nDELEGATED OBJECTIVE:\n' + action['request'])
        with state_lock:
            pending = [c for c in parent.state['children'] if c.get('status') != 'complete' and not c.get('allocation')]
            if pending:
                available = max(0, budget.remaining_tokens - 4096)
                share = available // len(pending)
                for candidate in pending:
                    maximum = effort.EFFORTS[candidate['action']['effort']]['tokens']
                    candidate['allocation'] = min(share, maximum)
                    candidate.setdefault('usage', {'tokens': 0, 'input_tokens': 0})
                save()
            allocation = child['allocation']
            index = child.get('card_index')
            if index is None:
                index = store.append_items(root, chat_id, [{'role': 'route', 'status': 'started', 'mode': mode,
                    'request': action['request'], 'files': files, 'effort': action['effort'], 'reason': action['reason'],
                    'orchestrated': True, 'task': task.id, 'time': store._now()}])
                child['card_index'] = index
            else:
                store.update_item(root, chat_id, index, status='started', task=task.id, outcome='')
            save()
        agent, _ = self._engine(task, mode=mode, online=online)
        agent.workspace.pin_files(files)
        if mode in ('critic', 'explore'):
            # These workers return analysis; file mutation belongs to Write-up,
            # whose durable runner already checkpoints approved tool actions.
            schemas = agent.workspace.schemas
            agent.workspace.schemas = lambda: [s for s in schemas() if s['function']['name'] not in ('write_file', 'run_python')]
            if action['effort'] == 'low':
                agent.think = False
        agent.client = budget.fork(agent.client, allowance=allocation, usage=child['usage'])
        level = effort.EFFORTS[action['effort']]
        seconds = min(level['minutes'] * 60, budget.remaining_seconds)
        runner = None

        def emit(kind, value):
            if runner is not None:
                ident = runner.state['id'] if runner.state else None
                if ident and child.get('job_id') != ident:
                    with state_lock:
                        child['job_id'] = ident
                        store.update_item(root, chat_id, index, job_id=ident)
                        save()
            if kind in ('notice', 'tool', 'result'):
                item = {'kind': kind, 'text': str(value)[:4000], 'time': time.time()}
                self.bus.publish('activity', mutate=lambda: task.activity.append(item),
                                 task=task.id, job=mode, id=child.get('job_id'), **item)

        if mode in ('critic', 'explore') and online:
            # As in a critique or exploration chat: check_reference, charged to this worker's share.
            agent.workspace.checker = nested_checker(agent, emit)
        try:
            if mode == 'prove':
                if any(f.lower().endswith('.pdf') for f in files):
                    raise ValueError('The proof worker needs text sources. Read the required PDF passages before delegating the precise statement.')
                runner = ProofRunner(agent, emit)
                if child.get('job_id'):
                    result = runner.resume(child['job_id'])
                else:
                    solve = min(self.args.proof_solve_tokens, 4096) if action['effort'] == 'low' else self.args.proof_solve_tokens
                    verify = min(self.args.proof_verify_tokens, 2048) if action['effort'] == 'low' else self.args.proof_verify_tokens
                    if allocation < 256:
                        raise AssistantBudget('No budget remains for a proof worker and its review.')
                    if solve + verify > allocation:
                        solve = max(128, allocation * solve // (solve + verify))
                        verify = max(128, min(verify, allocation - solve))
                    repair = min(solve, self.args.proof_repair_tokens or solve)
                    minimum = min(repair, self.args.proof_min_solve_tokens or min(16384, repair))
                    result = runner.start(goal, source_files=files, max_rounds=level['tries'],
                        max_tokens=allocation, max_seconds=max(0.1, seconds), max_predict=solve,
                        verify_tokens=verify, repair_tokens=repair, min_solve_tokens=minimum,
                        verify_temperature=self.args.proof_verify_temperature)
                outcome = {'candidate_complete': 'The model review found no issue.', 'uncertain': 'The model review is uncertain.',
                           'review_unavailable': 'Review unavailable; the candidate is preserved.'}.get(result['status'], result['stop_reason'] or result['status'])
                result = {'status': result['status'], 'answer': result['answer'], 'job_id': result['id'],
                          'review_status': result['status'], 'warnings': [result['stop_reason']],
                          'artifact': str(Path(result['proof_path']).relative_to(root))}
            elif mode == 'check':
                # A quick verified lookup: its own small budgets (refcheck.EFFORTS), capped by
                # this worker's share. It sees only the objective, so its queries stay public words.
                runner = CheckRunner(agent, emit)
                if child.get('job_id'):
                    result = runner.resume(child['job_id'])
                else:
                    if allocation < 2048:
                        raise AssistantBudget('No budget remains for a literature check.')
                    check = REFCHECK_EFFORTS.get(action['effort'], REFCHECK_EFFORTS['high'])
                    result = runner.start(action['request'], effort=action['effort'],
                        max_tokens=min(check['tokens'], allocation),
                        max_input_tokens=min(check['input'], budget.pool.state['max_input_tokens']),
                        max_seconds=max(0.1, min(check['seconds'], budget.remaining_seconds)))
                status = result['status']
                outcome = {'answered': 'Reference confirmed; see the answer.',
                           'partial': 'No exact place confirmed; see the levels.'}.get(status, status.replace('_', ' ').capitalize() + '.')
                result = {'status': status, 'answer': runner.state.get('answer') or runner._compose(),
                          'references': json.loads(runner.result_for_agent()), 'job_id': result['id'],
                          'artifact': str(Path(result['report_path']).relative_to(root)),
                          'warnings': [] if online else ['Online search was off; only cached sources were used.']}
            elif mode in ('literature', 'writeup') or mode in REVIEW_MODES:
                # Literature is the verified reading-list pipeline; the exact conversation sets its scope.
                # Review variants run one call at a time here: the Assistant already shares its budget.
                runner = (WriteupRunner(agent, emit) if mode == 'writeup' else ReviewRunner(agent, emit) if mode == 'literature'
                          else review_runner(REVIEW_MODES[mode])(agent, emit))
                if child.get('job_id'):
                    result = runner.resume(child['job_id'])
                else:
                    kwargs = dict(source_files=files, max_rounds=level['tries'], max_tokens=allocation,
                        max_input_tokens=min(level['input'], budget.pool.state['max_input_tokens']),
                        max_seconds=max(0.1, seconds), max_requests=level['requests'], max_chars=level['chars'])
                    if mode == 'literature':
                        kwargs['kind'] = mode
                    result = runner.start(action['request'], reference_context=context, **kwargs)
                verdict = (runner.state or {}).get('verdict')
                outcome = (result['status'].replace('_', ' ').capitalize() + (f' ({verdict.replace("_", " ")})' if verdict else '')
                           + '; see the saved report.')
                worker_error = result.get('worker_error')
                result = {'status': result['status'], 'answer': result['report'], 'job_id': result['id'],
                          'artifact': str(Path(result['report_path']).relative_to(root))}
                if worker_error:
                    result['error'] = worker_error
            else:
                agent.predict = min(self.args.predict, 2048 if action['effort'] == 'low' else 4096, agent.client.remaining_tokens)
                agent.max_rounds = min(level['tries'], 4)
                answer = agent.run(goal, emit, fresh=True)
                complete = (agent.last_stats.get('done_reason') in ('stop', 'tool_calls') and bool(answer.strip())
                            and all(type(agent.last_stats.get(key)) is int and agent.last_stats[key] >= 0
                                    for key in ('eval_count', 'prompt_eval_count')))
                result = {'status': 'complete' if complete else 'incomplete', 'answer': answer,
                          'job_id': child['id'], 'warnings': [] if complete else ['The worker answer is incomplete.']}
                outcome = 'Worker finished.' if complete else 'Worker reached its output limit.'
            if task.cancel.is_set() or result['status'] == 'paused':
                store.update_item(root, chat_id, index, status='stopped', outcome='Paused; work is saved.')
                save()
                raise KeyboardInterrupt
            store.update_item(root, chat_id, index, status='done', outcome=outcome, job_id=result['job_id'] if mode not in ('critic', 'explore') else None)
            self.bus.publish('chat_saved', chat=chat_id)
            return result
        except KeyboardInterrupt:
            store.update_item(root, chat_id, index, status='stopped', outcome='Paused; work is saved.')
            self.bus.publish('chat_saved', chat=chat_id)
            raise
        except (AgentError, ValueError, OSError) as exc:
            store.update_item(root, chat_id, index, status='failed', outcome=str(exc), error=str(exc))
            self.bus.publish('chat_saved', chat=chat_id)
            if isinstance(exc, WorkerInputError):
                error = exc.as_dict()
            elif isinstance(exc, ValueError):
                error = WorkerInputError(str(exc), details={'mode': mode}).as_dict()
            elif isinstance(exc, AssistantBudget):
                error = {'code': 'worker_budget_exhausted', 'message': str(exc), 'retryable': False,
                         'scope': 'budget', 'details': {'mode': mode}}
            else:
                error = {'code': 'worker_operation_failed', 'message': str(exc), 'retryable': True,
                         'scope': 'operation', 'details': {'mode': mode}}
            return {'status': 'failed', 'answer': '', 'warnings': [str(exc)], 'error': error,
                    'job_id': child.get('job_id') or child['id']}

    def _outcome(self, root, item):
        job = item.get('job_id')
        if not job:
            return item.get('status', '')
        try:
            if item['mode'] == 'prove':
                detail = store.proof_detail(root, job)
                if detail['status'] == 'candidate_complete' and detail['answer']:
                    return 'the model review found no issue in this answer: ' + one_line(detail['answer'], 600)
                return detail['status'] + (f'; {one_line(detail["stop_reason"], 300)}' if detail.get('stop_reason') else '')
            if item['mode'] in ('literature', 'writeup') or item['mode'] in REVIEW_MODES:
                state = store.research_state(root, job)
                text = state.get('document') or state.get('draft') or ''
                verdict = f' ({state["verdict"].replace("_", " ")})' if state.get('verdict') else ''
                return f'{state["status"]}{verdict}' + (': ' + one_line(text, 600) if text else '')
        except (store.NotFound, OSError, ValueError, KeyError):
            return ''
        return ''

    def start_route(self, chat_id, index, mode, request, files=(), limits=None):
        chat = store.load_chat(self.root, chat_id)
        if chat['mode'] != 'free':
            raise ValueError('Only Default conversations have suggestions')
        if type(index) is not int or not 0 <= index < len(chat['transcript']) or chat['transcript'][index].get('role') != 'route':
            raise store.NotFound('Unknown suggestion')
        if chat['transcript'][index].get('status') not in ('proposed', 'failed', 'cancelled'):
            raise ValueError('This suggestion has already started')
        if mode not in JOB_MODES:
            raise ValueError('Choose prove, critic, explore, check, literature, referee, detailed_review, quick_review, explain or writeup')
        if not isinstance(request, str) or not request.strip():
            raise ValueError('The request cannot be empty')
        files = self.check_files(list(files))
        root = self.root

        def on_update(task):
            fields = {'task': task.id}
            if task.target and task.kind != 'chat':
                fields.update(job_id=task.target, status='started')
            elif task.kind == 'chat' and task.state in RUNNING + ('done',):
                fields['status'] = 'started'
            if task.state == 'queued':
                fields['status'] = 'queued'
            elif task.state == 'cancelled':
                fields['status'] = 'cancelled'
            elif task.state == 'paused':
                fields['status'] = 'stopped'  # kept: a job resumes from its own page
            elif task.state == 'error' and not task.target:
                fields.update(status='failed', error=task.error)
            store.update_item(root, chat_id, index, **fields)
            self.bus.publish('chat_saved', chat=chat_id)

        limits = dict(limits or {})
        on_update._origin_chat = chat_id
        online = bool(chat['settings'].get('online')) and not self.online_locked
        store.update_item(root, chat_id, index, mode=mode, request=request, files=files, status='starting', error=None)
        try:
            text_files = [f for f in files if not f.lower().endswith('.pdf')]
            if mode == 'prove':
                task = self.start_proof(request, source_files=text_files, queue=True, on_update=on_update,
                                        **{k: v for k, v in limits.items()
                                           if k in ('rounds', 'tokens', 'seconds', 'solve_tokens', 'verify_tokens')})
            elif mode == 'literature' or mode in REVIEW_MODES:
                task = self.start_research('literature' if mode == 'literature' else 'referee', request, source_files=files,
                                           variant=REVIEW_MODES.get(mode), online=online, queue=True, on_update=on_update,
                                           **{k: v for k, v in limits.items() if k in ('rounds', 'tokens', 'input_tokens', 'seconds', 'requests', 'chars')})
            elif mode == 'check':
                task = self.check(chat_id, request, effort=limits.get('effort') or chat['transcript'][index].get('effort') or 'medium',
                                  queue=True, on_update=on_update)
            elif mode == 'writeup':
                task = self.start_writeup(request, source_files=files, notes='' if files else request,
                                          queue=True, on_update=on_update,
                                          **{k: v for k, v in limits.items() if k in ('tokens', 'input_tokens', 'seconds')})
            else:
                task = self.chat(chat_id, request, files=files, mode=mode, echo_user=False, on_update=on_update, queue=True)
        except (Busy, ValueError, OSError, AgentError) as exc:
            store.update_item(root, chat_id, index, status='failed', error=str(exc))
            self.bus.publish('chat_saved', chat=chat_id)
            raise
        return task

    def cancel_route(self, chat_id, index):
        """Cancel what a card started: remove it from the queue, or pause it if it is running."""
        chat = store.load_chat(self.root, chat_id)
        if type(index) is not int or not 0 <= index < len(chat['transcript']) or chat['transcript'][index].get('role') != 'route':
            raise store.NotFound('Unknown suggestion')
        task_id = chat['transcript'][index].get('task')
        with self._lock:
            queued = any(t.id == task_id for t in self.queue)
            running = self.task is not None and self.task.id == task_id and self.task.state in RUNNING
        if not task_id or not (queued or running):
            raise ValueError('This job is no longer running or waiting')
        return self.cancel_queued(task_id) if queued else self.pause()

    def dismiss_route(self, chat_id, index):
        chat = store.load_chat(self.root, chat_id)
        if type(index) is not int or not 0 <= index < len(chat['transcript']) or chat['transcript'][index].get('role') != 'route':
            raise store.NotFound('Unknown suggestion')
        item = store.update_item(self.root, chat_id, index, status='dismissed')
        self.bus.publish('chat_saved', chat=chat_id)
        return item


def _check_ctx(ctx, predict):
    if ctx is None:
        return
    if type(ctx) is not int or ctx < 2048 or ctx <= predict + 1024:
        raise ValueError('Context must be an integer of at least 2048 that exceeds the output budget + 1024')


def _check_proof(args, *, tokens, solve, verify, ctx):
    """The terminal's checks for a new proof, made before the job exists."""
    if max(solve, verify) + 256 >= ctx:
        raise ValueError(f'The context of {ctx} tokens must exceed each solve and review output ceiling '
                         'plus room for the input')
    if tokens < solve + verify:
        raise ValueError(f'The token budget must cover one full solve and its review: at least {solve + verify} tokens')
    repair = args.proof_repair_tokens
    if repair is not None and repair > solve:
        raise ValueError(f'The launch repair allowance (--proof-repair-tokens {repair}) exceeds the solve ceiling of {solve}')
    minimum = args.proof_min_solve_tokens
    if minimum is not None and minimum > (repair or solve):
        raise ValueError(f'The launch minimum solve allowance (--proof-min-solve-tokens {minimum}) exceeds '
                         f'the repair allowance of {repair or solve}')


def _copy(value):
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value
