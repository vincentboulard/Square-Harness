import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock
import uuid
import warnings

from mathagent.ledger import LedgerError, MAX_FILE_BYTES, ProofStore


class ProofStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = {"max_rounds": 10, "max_tokens": 100000, "max_seconds": 1200,
                         "ctx": 16000, "predict": 2000, "think": True}

    def create(self):
        return ProofStore.create(self.root, "Prove the lemma as stated.", self.settings,
                                 [{"path": "lemma.tex", "content": "Immutable original statement."}])

    def test_create_reload_and_source_snapshots(self):
        store = self.create()
        self.assertEqual(store.state["revision"], 1)
        self.assertEqual(store.state["status"], "ready")
        self.assertEqual(store.state["rounds_started"], 0)
        self.settings["max_rounds"] = 99
        self.assertEqual(store.state["settings"]["max_rounds"], 10)
        store.state["rounds_started"] = 1
        store.state["pending"] = {"round": 1, "phase": "solver"}
        store.state["tokens_charged"] = 5000
        store.save()
        loaded = ProofStore.load(self.root, store.state["id"])
        self.assertEqual(loaded.state, store.state)
        self.assertIn("Immutable original", loaded.state["sources"][0]["content"])
        backup = json.loads((store.directory / "state.backup.json").read_text())
        self.assertEqual(backup["revision"], 1)
        self.assertEqual(backup["rounds_started"], 0)

    def test_atomic_save_failure_leaves_previous_state_and_artifacts(self):
        store = self.create()
        artifact = store.write_artifact("solver", "A useful intermediate argument.")
        store.state["rounds_started"] = 1
        real_replace = __import__("os").replace

        def fail_main(source, destination):
            if Path(destination).name == "state.json":
                raise OSError("simulated interruption before atomic replacement")
            return real_replace(source, destination)

        with mock.patch("mathagent.ledger.os.replace", side_effect=fail_main):
            with self.assertRaises((LedgerError, OSError)):
                store.save()
        loaded = ProofStore.load(self.root, store.state["id"])
        self.assertEqual(loaded.state["rounds_started"], 0)
        self.assertEqual(loaded.state["revision"], 1)
        self.assertEqual(loaded.read_artifact(artifact), "A useful intermediate argument.")
        self.assertFalse(list(store.directory.glob(".state.json.*")))
        store.save()
        self.assertEqual(ProofStore.load(self.root, store.state["id"]).state["rounds_started"], 1)

    def test_corruption_recovery_is_explicit_and_does_not_destroy_backup(self):
        store = self.create()
        store.state["rounds_started"] = 1
        store.save()
        artifact = store.write_artifact("solver", "Evidence survives state recovery.")
        (store.directory / "state.json").write_text('{"incomplete":')
        with self.assertWarnsRegex(RuntimeWarning, "Recovered"):
            loaded = ProofStore.load(self.root, store.state["id"])
        self.assertEqual(loaded.state["revision"], 1)
        self.assertIn("recovery_notice", loaded.state)
        self.assertEqual(loaded.read_artifact(artifact), "Evidence survives state recovery.")
        with self.assertRaisesRegex(LedgerError, "reload"):
            store.save()
        loaded.save()
        again = ProofStore.load(self.root, store.state["id"])
        self.assertEqual(again.state["revision"], 2)
        self.assertIn("recovery_notice", again.state)
        self.assertEqual(json.loads((store.directory / "state.backup.json").read_text())["revision"], 1)

    def test_missing_state_recovers_backup_and_no_backup_is_error(self):
        store = self.create()
        (store.directory / "state.json").unlink()
        with self.assertRaisesRegex(LedgerError, "backup unavailable"):
            ProofStore.load(self.root, store.state["id"])
        other = self.create()
        other.save()
        (other.directory / "state.json").unlink()
        with self.assertWarns(RuntimeWarning):
            recovered = ProofStore.load(self.root, other.state["id"])
        self.assertEqual(recovered.state["revision"], 1)

    def test_optimistic_conflict_and_nonblocking_run_lock(self):
        first = self.create()
        second = ProofStore.load(self.root, first.state["id"])
        with first.lock():
            first.state["rounds_started"] = 1
            first.save()  # Saving while run lock is held is required.
            with self.assertRaisesRegex(LedgerError, "already in use"):
                with second.lock():
                    self.fail("Acquired a competing run lock")
            with self.assertRaisesRegex(LedgerError, "already in use"):
                with first.lock():
                    self.fail("Acquired a nested run lock")
        with second.lock():
            with self.assertRaisesRegex(LedgerError, "another process"):
                second.save()
        self.assertEqual(ProofStore.load(self.root, first.state["id"]).state["rounds_started"], 1)

    def test_locks_release_on_exception(self):
        store = self.create()
        with self.assertRaises(RuntimeError):
            with store.lock():
                raise RuntimeError("aborted run")
        with store.lock():
            store.save()

    def test_append_only_artifacts_and_streams(self):
        store = self.create()
        original = store.write_artifact("solver", "Original proof")
        second = store.write_artifact("solver", "Revised proof")
        self.assertNotEqual(original, second)
        self.assertEqual(store.read_artifact(original), "Original proof")
        stream = store.start_stream("round-1-solver")
        store.append_stream(stream, {"kind": "thinking", "text": "partial thought α"})
        before = store.read_artifact(stream)
        store.append_stream(stream, {"kind": "content", "text": "candidate"})
        after = store.read_artifact(stream)
        self.assertTrue(after.startswith(before))
        events = [json.loads(line) for line in after.splitlines()]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["text"], "partial thought α")
        loaded = ProofStore.load(self.root, store.state["id"])
        self.assertEqual(loaded.read_artifact(stream), after)
        with self.assertRaises(LedgerError):
            store.append_stream(original, {"text": "mutated"})
        self.assertEqual(store.read_artifact(original), "Original proof")

    def test_process_death_retains_stream_and_releases_lock(self):
        store = self.create()
        script = """
import os
import sys
from mathagent.ledger import ProofStore
store = ProofStore.load(sys.argv[1], sys.argv[2])
with store.lock():
    name = store.start_stream('solver')
    store.state['pending'] = {'round': 1, 'phase': 'solver', 'stream_artifact': name}
    store.state['tokens_charged'] = 8000
    store.save()
    store.append_stream(name, {'kind': 'thinking', 'text': 'saved before process death'})
    os._exit(23)
"""
        completed = subprocess.run([sys.executable, "-c", script, str(self.root), store.state["id"]],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 23, completed.stderr)
        loaded = ProofStore.load(self.root, store.state["id"])
        with loaded.lock():
            self.assertEqual(loaded.state["tokens_charged"], 8000)
            journal = loaded.read_artifact(loaded.state["pending"]["stream_artifact"])
            self.assertEqual(json.loads(journal)["text"], "saved before process death")

    def test_stream_hardlinks_cannot_modify_another_file(self):
        store = self.create()
        name = store.start_stream("solver")
        outside = self.root / "unrelated"
        __import__("os").link(store.directory / "artifacts" / name, outside)
        with self.assertRaises(LedgerError):
            store.append_stream(name, {"content": "changed"})
        self.assertEqual(outside.read_text(), "")

    def test_artifact_size_and_json_validation(self):
        store = self.create()
        with self.assertRaises(LedgerError):
            store.write_artifact("large", "x" * (MAX_FILE_BYTES + 1))
        stream = store.start_stream("solver")
        with self.assertRaises(LedgerError):
            store.append_stream(stream, {"value": float("nan")})
        self.assertEqual(store.read_artifact(stream), "")
        with self.assertRaises(LedgerError):
            store.append_stream(stream, ["not a dictionary"])

    def test_path_traversal_ids_and_artifacts(self):
        store = self.create()
        for bad in ("../outside", "/etc/passwd", "a/../b", "", "../../proof", str(uuid.uuid4()) + "/.."):
            with self.subTest(bad=bad):
                with self.assertRaises(LedgerError):
                    ProofStore.load(self.root, bad)
        for bad in ("../outside", "/etc/passwd", "0001-../secret.md", "0001-proof.md/else", "x\\y"):
            with self.subTest(bad=bad):
                with self.assertRaises(LedgerError):
                    store.read_artifact(bad)
                with self.assertRaises(LedgerError):
                    store.write_artifact(bad, "payload")
        self.assertFalse((self.root / "outside").exists())

    def test_internal_symlinks_refused(self):
        store = self.create()
        outside = self.root / "outside.json"
        outside.write_text((store.directory / "state.json").read_text())
        main = store.directory / "state.json"
        main.unlink()
        main.symlink_to(outside)
        with self.assertRaises(LedgerError):
            ProofStore.load(self.root, store.state["id"])
        with self.assertRaises(LedgerError):
            store.save()
        main.unlink()
        main.write_text(outside.read_text())
        artifact = store.directory / "artifacts" / "0001-malicious.md"
        artifact.symlink_to(outside)
        with self.assertRaises(LedgerError):
            store.read_artifact(artifact.name)
        stream = store.directory / "artifacts" / "0002-malicious.jsonl"
        stream.symlink_to(outside)
        with self.assertRaises(LedgerError):
            store.append_stream(stream.name, {"text": "payload"})
        report = store.directory / "report.md"
        report.symlink_to(outside)
        with self.assertRaises(LedgerError):
            store.write_report("payload")
        self.assertEqual(outside.read_text(), main.read_text())

    def test_symlink_storage_directory_refused_even_if_target_is_internal(self):
        (self.root / "actual").mkdir()
        (self.root / ".mathagent").symlink_to(self.root / "actual", target_is_directory=True)
        with self.assertRaises(LedgerError):
            self.create()
        with self.assertRaises(LedgerError):
            ProofStore.list(self.root)

    def test_replaced_artifact_directory_refused(self):
        store = self.create()
        artifacts = store.directory / "artifacts"
        artifacts.rmdir()
        (self.root / "other-artifacts").mkdir()
        artifacts.symlink_to(self.root / "other-artifacts", target_is_directory=True)
        with self.assertRaises(LedgerError):
            store.write_artifact("solver", "payload")
        with self.assertRaises(LedgerError):
            ProofStore.load(self.root, store.state["id"])

    def test_reports_are_atomic_replaceable_views(self):
        store = self.create()
        store.write_report("Unfinished proof")
        store.write_report("Useful partial result")
        store.write_ledger("# Proof ledger\nPending claim")
        self.assertEqual((store.directory / "report.md").read_text(), "Useful partial result")
        self.assertEqual((store.directory / "ledger.md").read_text(), "# Proof ledger\nPending claim")

    def test_validate_state_and_budget_fields(self):
        store = self.create()
        for key, bad in (("version", 2), ("rounds_started", -1), ("tokens_charged", True),
                         ("seconds_used", float("inf")), ("claims", "summary"),
                         ("pending", "solver"), ("goal", "")):
            with self.subTest(key=key):
                original = store.state[key]
                store.state[key] = bad
                with self.assertRaises(LedgerError):
                    store.save()
                store.state[key] = original
        for bad in (0, -2, False, "10"):
            with self.assertRaises(LedgerError):
                ProofStore.create(self.root, "Prove X.", {"max_rounds": bad})
        store.save()

    def test_invalid_source_snapshots_and_missing_budgets_fail(self):
        for source in ("lemma.tex", {"path": "lemma.tex"}, {"path": "../lemma.tex", "content": "Claim"},
                       {"path": "lemma.tex", "content": "Claim", "sha256": "not its digest"}):
            with self.subTest(source=source):
                with self.assertRaises(LedgerError):
                    ProofStore.create(self.root, "Prove X.", self.settings, [source])
        for key in ("max_rounds", "max_tokens", "max_seconds"):
            settings = dict(self.settings)
            settings.pop(key)
            with self.assertRaisesRegex(LedgerError, "Missing required proof budget"):
                ProofStore.create(self.root, "Prove X.", settings)
        settings = dict(self.settings, max_rounds=1.5)
        with self.assertRaisesRegex(LedgerError, "integer"):
            ProofStore.create(self.root, "Prove X.", settings)

    def test_unknown_version_cannot_be_silently_loaded(self):
        store = self.create()
        store.save()  # Even a valid older backup must not hide a newer schema.
        path = store.directory / "state.json"
        data = json.loads(path.read_text())
        data["version"] = 999
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(LedgerError, "version"):
            ProofStore.load(self.root, store.state["id"])

    def test_listing_warns_about_bad_jobs_and_skips_unrelated_entries(self):
        self.assertEqual(ProofStore.list(self.root), [])
        first = self.create()
        broken = self.create()
        (broken.directory / "state.json").write_text("corrupt")
        (first.directory.parent / "not-a-proof").mkdir()
        with self.assertWarnsRegex(RuntimeWarning, "Skipping unreadable proof"):
            jobs = ProofStore.list(self.root)
        self.assertEqual([job["id"] for job in jobs], [first.state["id"]])
        self.assertEqual(set(jobs[0]), {"id", "status", "goal", "rounds_started", "updated_at"})


if __name__ == "__main__":
    unittest.main()
