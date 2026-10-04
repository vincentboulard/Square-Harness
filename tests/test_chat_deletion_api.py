"""Conversation deletion protects work and preserves independently saved artifacts."""
import json
import threading
import uuid
from unittest.mock import patch

try:
    from .test_gui import GuiCase, text, PROOF, APPROVAL
except ImportError:
    from test_gui import GuiCase, text, PROOF, APPROVAL

from mathagent.gui import store


class ChatDeletionApiTests(GuiCase):
    def conversation(self, mode='free'):
        return self.ok('POST', '/api/chats', {'mode': mode, 'online': False})['id']

    def delete(self, chat):
        return self.request('DELETE', f'/api/chats/{chat}', {})

    def held_task(self, kind, target, *, on_update=None):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def work(task):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test task release timed out')
            return {'status': 'done'}

        task = self.hub._begin(kind, target, 'Test work', 'Keep this task active', work, on_update=on_update)
        self.assertTrue(entered.wait(1))
        return task, release

    def finish(self, task, release):
        release.set()
        task.thread.join(2)
        self.assertFalse(task.thread.is_alive())
        self.assertEqual(task.state, 'done', task.error)

    def test_idle_deletion_removes_only_the_selected_sidebar_conversation(self):
        selected = self.conversation()
        other = self.conversation('critic')
        store.append_items(self.root, selected, [{'role': 'user', 'content': 'Saved question'}])
        self.assertEqual({item['id'] for item in self.ok('GET', '/api/chats')['chats']}, {selected, other})
        status, result, _ = self.delete(selected)
        self.assertEqual((status, result), (200, {'deleted': selected}))
        self.assertEqual([item['id'] for item in self.ok('GET', '/api/chats')['chats']], [other])
        self.assertEqual(self.request('GET', f'/api/chats/{selected}')[0], 404)
        self.assertEqual(self.delete(selected)[0], 404)
        self.assertEqual(self.ok('GET', f'/api/chats/{other}')['id'], other)

    def test_delete_requires_authentication_same_origin_and_valid_json(self):
        chat = self.conversation()
        path = f'/api/chats/{chat}'
        cases = [({'auth': False, 'body': {}}, 401),
                 ({'auth': False, 'body': {}, 'headers': {'Authorization': 'Bearer wrong-token'}}, 401),
                 ({'body': {}, 'headers': {'Origin': 'http://attacker.example'}}, 403),
                 ({'body': {}, 'headers': {'Sec-Fetch-Site': 'cross-site'}}, 403),
                 ({'raw': b'confirm=yes', 'headers': {'Content-Type': 'application/x-www-form-urlencoded'}}, 415),
                 ({'raw': b'{"broken":', 'headers': {'Content-Type': 'application/json'}}, 400),
                 ({'raw': b'[]', 'headers': {'Content-Type': 'application/json'}}, 400)]
        for kwargs, expected in cases:
            with self.subTest(expected=expected, kwargs=kwargs):
                self.assertEqual(self.request('DELETE', path, **kwargs)[0], expected)
                self.assertEqual(self.ok('GET', path)['id'], chat)
        status, result, _ = self.request('DELETE', path, {}, headers={
            'Origin': f'http://127.0.0.1:{self.port}', 'Sec-Fetch-Site': 'same-origin'})
        self.assertEqual((status, result), (200, {'deleted': chat}))

    def test_running_and_pausing_turn_cannot_be_deleted(self):
        chat = self.conversation('critic')
        task, release = self.held_task('chat', chat)
        self.assertEqual(self.delete(chat)[0], 409)
        self.assertEqual(self.ok('GET', f'/api/chats/{chat}')['id'], chat)
        self.ok('POST', '/api/task/pause', {})
        self.assertEqual(task.state, 'pausing')
        self.assertEqual(self.delete(chat)[0], 409)
        self.finish(task, release)
        self.assertEqual(self.delete(chat)[0], 200)
        self.assertIsNone(self.ok('GET', '/api/task')['task'])

    def test_queued_turn_is_protected_while_an_unrelated_idle_chat_is_deletable(self):
        waiting = self.conversation('critic')
        idle = self.conversation()
        active, release = self.held_task('proof', str(uuid.uuid4()))
        queued = self.hub.chat(waiting, 'A queued question', queue=True)
        self.assertEqual(queued.state, 'queued')
        self.assertEqual(self.delete(waiting)[0], 409)
        self.assertEqual(self.delete(idle)[0], 200)
        self.assertEqual(active.state, 'running')
        self.assertFalse(active.cancel.is_set())
        self.hub.cancel_queued(queued.id)
        self.finish(active, release)
        self.assertEqual(self.delete(waiting)[0], 200)
        self.assertEqual(self.fake.requests, [])

    def test_routing_claim_is_protected_until_released(self):
        chat = self.conversation()
        with self.hub._lock:
            self.hub.routing[chat] = 1
        try:
            self.assertEqual(self.delete(chat)[0], 409)
            self.assertEqual(self.ok('GET', f'/api/chats/{chat}')['id'], chat)
        finally:
            with self.hub._lock:
                self.hub.routing.pop(chat, None)
        self.assertEqual(self.delete(chat)[0], 200)

    def test_legacy_route_job_protects_its_origin_chat_without_cancelling_the_job(self):
        chat = self.conversation()

        def on_update(task):
            pass

        on_update._origin_chat = chat
        task, release = self.held_task('proof', str(uuid.uuid4()), on_update=on_update)
        self.assertEqual(self.delete(chat)[0], 409)
        self.assertFalse(task.cancel.is_set())
        self.finish(task, release)
        self.assertEqual(self.delete(chat)[0], 200)

    def test_finished_state_waits_for_the_last_worker_callback_before_deletion(self):
        chat = self.conversation()
        finishing, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        publish = self.hub._publish_task

        def delayed_publish(task):
            publish(task)
            if task.state == 'done':
                finishing.set()
                if not release.wait(5):
                    raise RuntimeError('Test callback release timed out')

        with patch.object(self.hub, '_publish_task', side_effect=delayed_publish):
            task = self.hub._begin('chat', chat, 'Test turn', 'A finished turn', lambda task: {'status': 'done'})
            self.assertTrue(finishing.wait(1))
            self.assertEqual(task.state, 'done')
            self.assertTrue(task.thread.is_alive())
            self.assertEqual(self.delete(chat)[0], 409)
            self.finish(task, release)
        self.assertEqual(self.delete(chat)[0], 200)

    def test_delete_wins_a_race_before_chat_registration_without_launching_or_resurrecting_it(self):
        self.registration_race('critic', lambda chat: self.hub.chat(chat, 'Start a turn'))

    def test_delete_wins_a_race_before_assistant_claim_without_leaking_routing(self):
        self.registration_race('free', lambda chat: self.hub.route(chat, 'Start an Assistant turn'))
        self.assertEqual(self.hub.routing, {})

    def registration_race(self, mode, start):
        chat = self.conversation(mode)
        loaded, release = threading.Event(), threading.Event()
        outcomes, errors, intercepted = [], [], [False]
        self.addCleanup(release.set)
        load_chat = store.load_chat

        def pause_first_load(root, chat_id):
            value = load_chat(root, chat_id)
            if threading.current_thread().name == 'chat-registration-race' and not intercepted[0]:
                intercepted[0] = True
                loaded.set()
                if not release.wait(5):
                    raise RuntimeError('Test registration release timed out')
            return value

        def starting():
            try:
                outcomes.append(start(chat))
            except Exception as exc:
                errors.append(exc)

        with patch.object(store, 'load_chat', side_effect=pause_first_load):
            starter = threading.Thread(target=starting, name='chat-registration-race')
            starter.start()
            self.assertTrue(loaded.wait(1))
            self.assertEqual(self.delete(chat)[0], 200)
            release.set()
            starter.join(2)
            self.assertFalse(starter.is_alive())
        self.assertEqual(outcomes, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], store.NotFound)
        self.assertEqual(self.fake.requests, [])
        self.assertIsNone(self.hub.task)
        self.assertFalse((self.root / '.mathagent' / 'chats' / (chat + '.json')).exists())

    def test_deletion_preserves_linked_jobs_their_streams_and_attached_sources(self):
        chat = self.conversation()
        source = self.root / 'notes.tex'
        source.write_text('Hypotheses and proof source.\n')
        self.fake.replies = [text(PROOF), text(json.dumps(APPROVAL))]
        proof = self.ok('POST', '/api/proofs', {'goal': 'Prove reflexivity.', 'tokens': 16000,
                                             'source_files': ['notes.tex'], 'seconds': 60})['id']
        self.assertEqual(self.wait_task()['state'], 'done')
        self.fake.replies = [text('Plan'), text('Coverage remains limited.'), text('# Partial source report'),
                             text('No complete literature audit was performed.'), text('# Partial source report')]
        research = self.ok('POST', '/api/research', {'kind': 'literature', 'goal': 'Review a mathematical topic.',
                                                   'online': False, 'rounds': 1, 'tokens': 16000,
                                                   'input_tokens': 50000, 'requests': 0, 'seconds': 60})['id']
        self.assertEqual(self.wait_task()['state'], 'done')
        store.append_items(self.root, chat, [
            {'role': 'user', 'content': 'Use the attached source.', 'files': ['notes.tex']},
            {'role': 'route', 'mode': 'prove', 'job_id': proof, 'status': 'done'},
            {'role': 'route', 'mode': 'literature', 'job_id': research, 'status': 'done'}])
        store.save_assistant(self.root, chat, {'status': 'interrupted', 'children': [
            {'job_id': proof}, {'job_id': research}], 'messages': []})
        preserved = {source: source.read_bytes()}
        for directory in (self.root / '.mathagent' / 'proofs' / proof,
                          self.root / '.mathagent' / 'research' / research):
            preserved.update({path: path.read_bytes() for path in directory.rglob('*') if path.is_file()})
        self.assertTrue(any('stream' in path.name for path in preserved))
        self.assertEqual(self.delete(chat)[0], 200)
        for path, expected in preserved.items():
            self.assertEqual(path.read_bytes(), expected, str(path))
        self.assertEqual(self.ok('GET', f'/api/proofs/{proof}')['id'], proof)
        self.assertEqual(self.ok('GET', f'/api/research/{research}')['id'], research)
        self.assertIn(proof, [job['id'] for job in self.ok('GET', '/api/proofs')['jobs']])
        self.assertIn(research, [job['id'] for job in self.ok('GET', '/api/research')['jobs']])
