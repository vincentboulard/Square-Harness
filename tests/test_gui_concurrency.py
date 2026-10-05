"""Concurrent GUI work preserves each conversation, stream and output budget."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

try:
    from .test_gui import GuiCase, MODEL, text
except ImportError:
    from test_gui import GuiCase, MODEL, text

from mathagent import cli
from mathagent.assistant_budget import AssistantBudget, BudgetClient
from mathagent.gui import store
from mathagent.gui.hub import Busy, Hub, Task


class HeldWork:
    """Keep tasks alive until the test releases or individually pauses them."""

    def __init__(self, case, *, cooperative=True):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.cooperative = cooperative
        case.addCleanup(self.release.set)

    def __call__(self, task):
        self.started.set()
        while not self.release.wait(0.01):
            if task.cancel.is_set():
                self.cancelled.set()
                if self.cooperative:
                    raise KeyboardInterrupt
        return {'status': 'done'}


class HeldModel:
    """Record actual overlapping HTTP inference before returning final tokens."""

    def __init__(self, case):
        self.requests = []
        self.lock = threading.Lock()
        self.first = threading.Event()
        self.second = threading.Event()
        self.release = threading.Event()
        self.token_counts = {}
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                body = json.dumps({'models': [{'name': MODEL}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with model.lock:
                    model.requests.append(payload)
                    model.first.set()
                    if len(model.requests) >= 2:
                        model.second.set()
                prompt = payload['messages'][-1]['content']
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.end_headers()
                try:
                    self.wfile.write(json.dumps({'message': {'content': 'Reply to '}, 'done': False}).encode() + b'\n')
                    self.wfile.flush()
                    if not model.release.wait(5):
                        return
                    self.wfile.write(json.dumps(text(prompt + '.', eval_count=model.token_counts.get(prompt, 20))[0]).encode() + b'\n')
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        self.thread.start()
        self.host = f'http://127.0.0.1:{self.server.server_port}'
        case.addCleanup(self.close)

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


class GuiConcurrencyTests(GuiCase):
    extra_args = ('--gui-concurrency', '2', '--gui-model-concurrency', '2')

    def conversation(self, mode='critic'):
        return self.ok('POST', '/api/chats', {'mode': mode, 'online': False})['id']

    def begin(self, target, *, kind='proof', queue=False, live=None, cooperative=True, on_update=None):
        work = HeldWork(self, cooperative=cooperative)
        task = self.hub._begin(kind, target, 'Test work', 'Concurrent work', work,
                               live=live, queue=queue, on_update=on_update)
        if task.state != 'queued':
            self.assertTrue(work.started.wait(1), 'task did not start')
        self.addCleanup(self.release_and_join, task, work)
        return task, work

    def release_and_join(self, task, work):
        work.release.set()
        if task.thread is not None:
            task.thread.join(1)

    def finish(self, task, work):
        work.release.set()
        task.thread.join(2)
        self.assertFalse(task.thread.is_alive())
        self.assertEqual(task.state, 'done', task.error)

    def active_ids(self):
        return {entry['task']['id'] for entry in self.hub.snapshot()['tasks']}

    def test_default_allows_two_jobs_and_third_waits_for_a_free_slot(self):
        args = cli.parser().parse_args(['--workspace', str(self.root)])
        args.workspace = self.root
        default_hub = Hub(args)
        self.addCleanup(default_hub.shutdown, 0)
        self.assertEqual(default_hub.snapshot()['concurrency']['chats'], 2)

        first, first_work = self.begin('first')
        second, second_work = self.begin('second')
        self.assertFalse(first_work.release.is_set())
        self.assertEqual(self.active_ids(), {first.id, second.id})
        with self.assertRaises(Busy):
            self.begin('capacity-without-queue')
        third, third_work = self.begin('third', queue=True)
        self.assertEqual(third.state, 'queued')
        self.assertFalse(third_work.started.is_set())
        self.assertEqual([entry['id'] for entry in self.hub.snapshot()['queue']], [third.id])

        self.finish(first, first_work)
        self.assertTrue(third_work.started.wait(1))
        self.assertEqual(second.state, 'running')
        self.assertFalse(second.cancel.is_set())
        self.assertEqual(self.active_ids(), {second.id, third.id})
        self.finish(second, second_work)
        self.finish(third, third_work)

    def test_same_chat_or_saved_job_cannot_have_two_pending_writers(self):
        chat = self.conversation()
        active, active_work = self.begin(chat, kind='chat')
        for queue in (False, True):
            with self.subTest(queue=queue), self.assertRaises(Busy):
                self.begin(chat, kind='chat', queue=queue)
        self.finish(active, active_work)

        active, active_work = self.begin('saved-proof')
        for queue in (False, True):
            with self.subTest(kind='proof', queue=queue), self.assertRaises(Busy):
                self.begin('saved-proof', queue=queue)
        self.finish(active, active_work)

        first, first_work = self.begin('first')
        second, second_work = self.begin('second')
        waiting, waiting_work = self.begin(chat, kind='chat', queue=True)
        with self.assertRaises(Busy):
            self.begin(chat, kind='chat', queue=True)
        self.hub.cancel_queued(waiting.id)
        self.assertFalse(waiting_work.started.is_set())
        self.finish(first, first_work)
        self.finish(second, second_work)

    def test_routed_job_and_chat_share_the_conversation_writer_guard(self):
        chat = self.conversation('free')

        def on_update(task):
            pass

        on_update._origin_chat = chat
        routed, routed_work = self.begin('saved-proof', on_update=on_update)
        with self.assertRaises(Busy):
            self.begin(chat, kind='chat', queue=True)
        before = store.load_chat(self.root, chat)
        status, response, _ = self.request('POST', f'/api/chats/{chat}/route',
                                         {'content': 'Must not enter this conversation.'})
        self.assertEqual(status, 409, response)
        self.assertEqual(store.load_chat(self.root, chat), before)
        self.assertEqual(self.hub.routing, {})
        self.assertFalse(routed.cancel.is_set())
        self.finish(routed, routed_work)

    def test_assistant_rejected_after_shutdown_does_not_mutate_the_conversation(self):
        chat = self.conversation('free')
        before = store.load_chat(self.root, chat)
        self.hub.shutdown(wait=0)
        status, response, _ = self.request('POST', f'/api/chats/{chat}/route',
                                         {'content': 'Must not start after shutdown.'})
        self.assertEqual(status, 409, response)
        self.assertEqual(store.load_chat(self.root, chat), before)
        self.assertEqual(self.hub.routing, {})
        self.assertEqual(self.fake.requests, [])

    def test_pause_requires_selection_and_leaves_the_other_chat_running(self):
        first, first_work = self.begin(self.conversation(), kind='chat')
        second, second_work = self.begin(self.conversation('explore'), kind='chat')
        status, value, _ = self.request('POST', '/api/task/pause', {})
        self.assertEqual(status, 400, value)
        self.assertFalse(first.cancel.is_set())
        self.assertFalse(second.cancel.is_set())
        selected = self.ok('POST', '/api/task/pause', {'task': first.id})['task']
        self.assertEqual(selected['id'], first.id)
        first.thread.join(2)
        self.assertEqual(first.state, 'paused')
        self.assertEqual(second.state, 'running')
        self.assertFalse(second.cancel.is_set())
        self.assertTrue(first_work.cancelled.is_set())

        selected = self.ok('POST', '/api/task/pause', {})['task']
        self.assertEqual(selected['id'], second.id)
        second.thread.join(2)
        self.assertEqual(second.state, 'paused')
        self.assertTrue(second_work.cancelled.is_set())

    def test_resize_drains_existing_work_then_can_fill_new_slots(self):
        first, first_work = self.begin('first')
        second, second_work = self.begin('second')
        third, third_work = self.begin('third', queue=True)
        seq = self.hub.bus.seq
        self.ok('POST', '/api/concurrency', {'chats': 1, 'requests': 1})
        settings = self.ok('GET', '/api/status')['concurrency']
        self.assertEqual((settings['chats'], settings['requests'], settings['active']), (1, 1, 2))
        self.assertFalse(first.cancel.is_set())
        self.assertFalse(second.cancel.is_set())
        events, _ = self.hub.bus.wait(seq, 0)
        self.assertTrue(any(event['type'] == 'concurrency' for event in events))

        self.finish(first, first_work)
        self.assertFalse(third_work.started.is_set())
        self.assertEqual(third.state, 'queued')
        self.finish(second, second_work)
        self.assertTrue(third_work.started.wait(1))
        fourth, fourth_work = self.begin('fourth', queue=True)
        self.assertFalse(fourth_work.started.is_set())
        self.ok('POST', '/api/concurrency', {'chats': 2, 'requests': 2})
        self.assertTrue(fourth_work.started.wait(1))
        self.assertEqual(self.active_ids(), {third.id, fourth.id})
        self.finish(third, third_work)
        self.finish(fourth, fourth_work)

    def test_resize_validation_is_atomic_and_requires_integer_limits(self):
        for body in ({'chats': 0}, {'chats': 9}, {'chats': True}, {'chats': 1.5},
                     {'requests': '2'}, {'requests': 0}, {'chats': 1, 'requests': 9}):
            with self.subTest(body=body):
                status, value, _ = self.request('POST', '/api/concurrency', body)
                self.assertEqual(status, 400, value)
                settings = self.hub.snapshot()['concurrency']
                self.assertEqual((settings['chats'], settings['requests']), (2, 2))

    def test_snapshot_keeps_each_live_stream_and_activity_separate(self):
        first_live = {'chat': self.conversation(), 'steps': [{'type': 'call', 'text': 'first output'}]}
        second_live = {'chat': self.conversation(), 'steps': [{'type': 'call', 'text': 'second output'}]}
        first, first_work = self.begin(first_live['chat'], kind='chat', live=first_live)
        second, second_work = self.begin(second_live['chat'], kind='chat', live=second_live)
        first.activity.append('first activity')
        second.activity.append('second activity')
        snapshot = self.ok('GET', '/api/task')
        entries = {entry['task']['id']: entry for entry in snapshot['tasks']}
        self.assertEqual(entries[first.id]['activity'], ['first activity'])
        self.assertEqual(entries[second.id]['activity'], ['second activity'])
        self.assertEqual(entries[first.id]['live'], first_live)
        self.assertEqual(entries[second.id]['live'], second_live)
        self.assertIn(snapshot['task']['id'], entries)
        self.assertEqual(snapshot['live'], entries[snapshot['task']['id']]['live'])
        copied = self.hub.snapshot()
        copied['tasks'][0]['live']['steps'][0]['text'] = 'mutated snapshot copy'
        self.assertEqual(first.live['steps'][0]['text'], 'first output')
        self.assertEqual(second.live['steps'][0]['text'], 'second output')
        self.finish(first, first_work)
        self.finish(second, second_work)

    def test_shutdown_cancels_every_active_job_without_advancing_queue(self):
        first, first_work = self.begin('first')
        second, second_work = self.begin('second')
        third, third_work = self.begin('third', queue=True)
        self.hub.shutdown(wait=1)
        for task, work in ((first, first_work), (second, second_work)):
            self.assertTrue(task.cancel.is_set())
            self.assertTrue(work.cancelled.is_set())
            self.assertFalse(task.thread.is_alive())
            self.assertEqual(task.state, 'paused')
        self.assertFalse(third_work.started.is_set())
        self.assertFalse(self.hub.snapshot()['queue'])

    def test_shutdown_timeout_is_total_for_uncooperative_jobs(self):
        first, first_work = self.begin('first', cooperative=False)
        second, second_work = self.begin('second', cooperative=False)
        start = time.monotonic()
        self.hub.shutdown(wait=0.15)
        self.assertLess(time.monotonic() - start, 0.27, 'shutdown waited the timeout separately for every job')
        self.assertTrue(first.cancel.is_set())
        self.assertTrue(second.cancel.is_set())
        self.assertTrue(first_work.cancelled.wait(1))
        self.assertTrue(second_work.cancelled.wait(1))
        self.finish(first, first_work)
        self.finish(second, second_work)

    def start_model_chats(self, model):
        self.hub.args.host = model.host
        prompts = ('First independent request', 'Second independent request')
        chats = [self.conversation(mode) for mode in ('critic', 'explore')]
        tasks = []
        for chat, prompt in zip(chats, prompts):
            tasks.append(self.ok('POST', f'/api/chats/{chat}/messages', {'content': prompt})['task'])
        return chats, prompts, tasks

    def assert_model_chats_saved(self, model, chats, prompts):
        for chat, prompt in zip(chats, prompts):
            def completed():
                detail = self.ok('GET', f'/api/chats/{chat}')
                return detail if len(detail['transcript']) == 2 else None
            saved = self.wait_for(completed, message='the independent saved turn')
            self.assertEqual([entry['role'] for entry in saved['transcript']], ['user', 'assistant'])
            self.assertEqual(saved['transcript'][0]['content'], prompt)
            self.assertEqual(saved['transcript'][1]['content'], 'Reply to ' + prompt + '.')
            self.assertEqual(saved['context_messages'], 2)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual([request['options']['num_predict'] for request in model.requests], [1024, 1024])
        self.assertEqual(self.fake.requests, [])

    def test_two_http_chat_streams_overlap_and_keep_full_output_ceilings(self):
        model = HeldModel(self)
        chats, prompts, tasks = self.start_model_chats(model)
        self.assertTrue(model.second.wait(2), 'second chat did not reach the model while the first was unfinished')
        self.assertEqual(self.active_ids(), {task['id'] for task in tasks})
        def both_streams():
            snapshot = self.ok('GET', '/api/task')
            return snapshot if len(snapshot['tasks']) == 2 and all(entry['live']['steps'] for entry in snapshot['tasks']) else None
        snapshot = self.wait_for(both_streams, message='both live model streams')
        self.assertEqual({entry['live']['chat'] for entry in snapshot['tasks']}, set(chats))
        model.release.set()
        self.assert_model_chats_saved(model, chats, prompts)

    def test_model_request_limit_can_serialize_inference_without_sharing_chat_budgets(self):
        self.ok('POST', '/api/concurrency', {'requests': 1})
        model = HeldModel(self)
        chats, prompts, tasks = self.start_model_chats(model)
        self.assertTrue(model.first.wait(2))
        self.assertEqual(self.active_ids(), {task['id'] for task in tasks})
        self.assertFalse(model.second.wait(0.15), 'the model request ceiling was exceeded')
        model.release.set()
        self.assertTrue(model.second.wait(2))
        self.assert_model_chats_saved(model, chats, prompts)

    def wait_for_budget_admission(self, pool, owner):
        def waiting():
            with pool._condition:
                return any(waiter.owner == owner for waiter in pool._waiters)
        self.wait_for(waiting, message='Assistant global inference admission')

    def interrupted_budget_admission(self, *, expired):
        """An actual GUI client waits without dispatch or token reservation."""
        self.hub.set_concurrency(requests=1)
        task = Task('chat', self.conversation('free'), 'Assistant')
        self.hub.inference.register(task.id)
        self.addCleanup(self.hub.inference.unregister, task.id)
        self.hub.inference.register('held-model-slot')
        self.addCleanup(self.hub.inference.unregister, 'held-model-slot')
        agent, _ = self.hub._engine(task, mode='explore', online=False)
        budget = BudgetClient(agent.client, max_tokens=60000, max_input_tokens=240000,
                              max_seconds=0.15 if expired else 10)
        payload = {'messages': [{'role': 'user', 'content': 'An undispatched request.'}],
                   'options': {'num_predict': 8192, 'num_ctx': 40960}}
        errors, completed = [], threading.Event()

        def invoke():
            try:
                list(budget.stream(payload))
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        with self.hub.inference.slot('held-model-slot', threading.Event()):
            with patch.object(agent.client, '_interruptible_stream') as dispatch:
                worker = threading.Thread(target=invoke, daemon=True)
                worker.start()
                self.addCleanup(worker.join, 1)
                self.addCleanup(task.cancel.set)
                self.wait_for_budget_admission(self.hub.inference, task.id)
                self.assertEqual(budget.snapshot()['tokens'], 0)
                self.assertEqual(budget.snapshot()['input_tokens'], 0)
                if not expired:
                    task.cancel.set()
                self.assertTrue(completed.wait(1), 'waiting inference did not cancel or expire')
                worker.join(1)
                self.assertFalse(worker.is_alive())
                dispatch.assert_not_called()
        self.assertEqual(len(errors), 1, errors)
        self.assertIsInstance(errors[0], AssistantBudget if expired else KeyboardInterrupt)
        self.assertEqual(budget.snapshot()['tokens'], 0)
        self.assertEqual(budget.snapshot()['input_tokens'], 0)
        self.assertEqual(budget.remaining_tokens, 60000)
        self.assertEqual(self.fake.requests, [])

    def test_assistant_cancelled_waiting_for_global_model_slot_keeps_budget_unspent(self):
        self.interrupted_budget_admission(expired=False)

    def test_assistant_expired_waiting_for_global_model_slot_keeps_budget_unspent(self):
        self.interrupted_budget_admission(expired=True)

    def test_two_assistant_turns_preserve_separate_full_budgets_and_usage(self):
        model = HeldModel(self)
        self.hub.args.host = model.host
        self.hub.args.ctx = 40960
        self.hub.args.assistant_tokens = 60000
        prompts = ('First independent Assistant request', 'Second independent Assistant request')
        model.token_counts = dict(zip(prompts, (37, 53)))
        chats = [self.conversation('free') for _ in prompts]
        for chat, prompt in zip(chats, prompts):
            self.ok('POST', f'/api/chats/{chat}/route', {'content': prompt})
        self.assertTrue(model.second.wait(2), 'independent Assistant turns did not overlap')
        self.assertEqual(len(self.hub.snapshot()['tasks']), 2)
        for chat in chats:
            saved = store.load_chat(self.root, chat)['assistant_state']
            self.assertEqual(saved['budget']['max_tokens'], 60000)

        model.release.set()
        for chat, prompt in zip(chats, prompts):
            def completed():
                detail = self.ok('GET', f'/api/chats/{chat}')
                return detail if detail.get('assistant_status') == 'complete' else None
            saved = self.wait_for(completed, message='independent Assistant completion')
            self.assertEqual(saved['assistant_budget']['tokens']['used'], model.token_counts[prompt])
            budget = store.load_chat(self.root, chat)['assistant_state']['budget']
            self.assertEqual(budget['max_tokens'], 60000)
            self.assertEqual(budget['tokens'], model.token_counts[prompt])
            self.assertEqual([entry['content'] for entry in saved['transcript'] if entry['role'] == 'user'], [prompt])
            self.assertEqual([entry['content'] for entry in saved['transcript'] if entry['role'] == 'assistant'],
                             ['Reply to ' + prompt + '.'])
        self.assertEqual([request['options']['num_predict'] for request in model.requests], [8192, 8192])
        self.assertEqual(self.fake.requests, [])
