import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

from mathagent.gui import store


class ChatDeletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def chat_path(self, chat_id):
        return self.root / '.mathagent' / 'chats' / (chat_id + '.json')

    def test_delete_removes_whole_conversation_and_assistant_checkpoint_only(self):
        selected = store.create_chat(self.root, 'free')
        other = store.create_chat(self.root, 'critic')
        history = [{'role': 'user', 'content': 'Prove the lemma.'},
                   {'role': 'assistant', 'content': 'Saved proof.'}]
        store.commit_turn(self.root, selected['id'], history, [], [])
        store.save_assistant(self.root, selected['id'], {'status': 'interrupted', 'messages': history,
                                                        'children': [{'job_id': 'a proof job'}]})
        source = self.root / 'attached.tex'
        source.write_text('Source hypotheses must survive.')
        preserved = {source: source.read_bytes(), self.chat_path(other['id']): self.chat_path(other['id']).read_bytes()}
        for kind in ('proofs', 'research'):
            directory = self.root / '.mathagent' / kind / str(uuid.uuid4())
            directory.mkdir(parents=True)
            for name, content in [('state.json', '{"status":"paused"}'),
                                  ('0001-solver-stream.jsonl', '{"message":{"content":"proof"}}\n')]:
                path = directory / name
                path.write_text(content)
                preserved[path] = path.read_bytes()
        store.list_chats(self.root)  # Exercise deletion after the listing cache was populated.
        result = store.delete_chat(self.root, selected['id'])
        self.assertEqual(result, {'id': selected['id'], 'deleted': True})
        self.assertFalse(self.chat_path(selected['id']).exists())
        with self.assertRaises(store.NotFound):
            store.load_chat(self.root, selected['id'])
        self.assertEqual([chat['id'] for chat in store.list_chats(self.root)], [other['id']])
        for path, data in preserved.items():
            self.assertEqual(path.read_bytes(), data, str(path))

    def test_deleted_chat_cannot_be_resurrected_by_a_late_public_update(self):
        chat = store.create_chat(self.root, 'free')
        store.list_chats(self.root)
        store.delete_chat(self.root, chat['id'])
        for update in [lambda: store.save_assistant(self.root, chat['id'], {'status': 'complete'}),
                       lambda: store.append_items(self.root, chat['id'], [{'role': 'notice', 'content': 'Late event'}]),
                       lambda: store.update_chat_settings(self.root, chat['id'], online=True)]:
            with self.assertRaises(store.NotFound):
                update()
        self.assertFalse(self.chat_path(chat['id']).exists())
        self.assertEqual(store.list_chats(self.root), [])
        replacement = store.create_chat(self.root, 'free')
        self.assertNotEqual(replacement['id'], chat['id'])
        self.assertEqual([item['id'] for item in store.list_chats(self.root)], [replacement['id']])

    def test_delete_and_public_writer_share_the_chat_lock(self):
        chat = store.create_chat(self.root, 'free')
        in_delete, release = threading.Event(), threading.Event()
        errors = []
        original_regular = store._regular

        def paused_regular(path, **kwargs):
            if threading.current_thread().name == 'delete-test':
                in_delete.set()
                if not release.wait(2):
                    raise RuntimeError('Test lock coordination timed out')
            return original_regular(path, **kwargs)

        def delete():
            try:
                store.delete_chat(self.root, chat['id'])
            except Exception as exc:
                errors.append(exc)

        def late_writer():
            try:
                store.save_assistant(self.root, chat['id'], {'status': 'complete'})
            except Exception as exc:
                errors.append(exc)

        with patch.object(store, '_regular', side_effect=paused_regular):
            remover = threading.Thread(target=delete, name='delete-test')
            remover.start()
            self.assertTrue(in_delete.wait(1))
            writer = threading.Thread(target=late_writer, name='write-test')
            writer.start()
            release.set()
            remover.join(2)
            writer.join(2)
        self.assertFalse(remover.is_alive())
        self.assertFalse(writer.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], store.NotFound)
        self.assertEqual(store.list_chats(self.root), [])

    def test_invalid_identifiers_cannot_select_other_paths(self):
        chat = store.create_chat(self.root, 'free')
        original = self.chat_path(chat['id']).read_bytes()
        for invalid in ['../' + chat['id'], chat['id'] + '.json', chat['id'] + '\n',
                        'AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA', '', None, 12, ['id']]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                store.delete_chat(self.root, invalid)
        self.assertEqual(self.chat_path(chat['id']).read_bytes(), original)

    def test_unknown_chat_does_not_create_storage_and_repeated_delete_is_not_found(self):
        with self.assertRaises(store.NotFound):
            store.delete_chat(self.root, str(uuid.uuid4()))
        self.assertFalse((self.root / '.mathagent').exists())
        chat = store.create_chat(self.root, 'critic')
        store.delete_chat(self.root, chat['id'])
        with self.assertRaises(store.NotFound):
            store.delete_chat(self.root, chat['id'])

    def test_symlinked_chat_files_including_dangling_links_are_rejected(self):
        selected = store.create_chat(self.root, 'free')
        other = store.create_chat(self.root, 'critic')
        original = self.chat_path(other['id']).read_bytes()
        path = self.chat_path(selected['id'])
        path.unlink()
        path.symlink_to(self.chat_path(other['id']))
        with self.assertRaises(ValueError):
            store.delete_chat(self.root, selected['id'])
        self.assertTrue(path.is_symlink())
        self.assertEqual(self.chat_path(other['id']).read_bytes(), original)
        path.unlink()
        path.symlink_to(self.root / 'missing.json')
        with self.assertRaises(ValueError):
            store.delete_chat(self.root, selected['id'])
        self.assertTrue(path.is_symlink())

    def test_symlinked_storage_directories_are_rejected(self):
        for component in ('.mathagent', 'chats'):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if component == 'chats':
                    (root / '.mathagent').mkdir()
                    path = root / '.mathagent' / 'chats'
                else:
                    path = root / '.mathagent'
                path.symlink_to(root / 'missing-storage', target_is_directory=True)
                with self.assertRaises(ValueError):
                    store.delete_chat(root, str(uuid.uuid4()))
                self.assertTrue(path.is_symlink())

    def test_nonregular_or_foreign_identity_records_are_not_deleted(self):
        chat = store.create_chat(self.root, 'free')
        path = self.chat_path(chat['id'])
        path.unlink()
        path.mkdir()
        with self.assertRaises(ValueError):
            store.delete_chat(self.root, chat['id'])
        self.assertTrue(path.is_dir())
        path.rmdir()
        changed = dict(chat, id=str(uuid.uuid4()))
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            store.delete_chat(self.root, chat['id'])
        self.assertTrue(path.is_file())


if __name__ == '__main__':
    unittest.main()
