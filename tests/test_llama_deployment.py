"""Deployment contract checks without Docker, GPU access, or model downloads."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "deployment" / "llama_v100.py"
spec = importlib.util.spec_from_file_location("llama_v100", MODULE_PATH)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class Response(io.BytesIO):
    def __init__(self, payload, status, content_range=""):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.lock = deploy.load_lock()

    def small_lock(self):
        lock = dict(self.lock)
        lock.update(weights_bytes=6, weights_sha256=hashlib.sha256(b"abcdef").hexdigest())
        return lock

    def test_context_is_per_slot_and_cannot_silently_round(self):
        argv = deploy.server_arguments(self.lock, 32768, 3)
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "98304")
        self.assertEqual(argv[argv.index("--parallel") + 1], "3")
        for option in ("--no-kv-unified", "--no-context-shift", "--jinja", "--slots", "--metrics"):
            self.assertIn(option, argv)
        self.assertEqual(argv[argv.index("--fit") + 1], "off")
        self.assertEqual(argv[argv.index("--gpu-layers") + 1], "all")
        self.assertEqual(argv[argv.index("--cache-type-k") + 1], "f16")
        self.assertEqual(argv[argv.index("--cache-type-v") + 1], "f16")
        self.assertEqual(argv[argv.index("--top-k") + 1], "0")
        self.assertEqual(argv[argv.index("--min-p") + 1], "0")
        self.assertEqual(argv[argv.index("--repeat-penalty") + 1], "1")
        with self.assertRaises(ValueError):
            deploy.server_arguments(self.lock, 32001, 3)

    def test_batch_tuning_preserves_context_and_has_sensible_bounds(self):
        argv = deploy.server_arguments(self.lock, 32768, 3, 256, 64)
        self.assertEqual(argv[argv.index("--batch-size") + 1], "256")
        self.assertEqual(argv[argv.index("--ubatch-size") + 1], "64")
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "98304")
        for batch, ubatch in ((0, 128), (128, 256), (512, 0)):
            with self.assertRaises(ValueError):
                deploy.server_arguments(self.lock, 32768, 3, batch, ubatch)

    def test_build_pins_source_and_cuda_base(self):
        argv = deploy.build_command(self.lock, 8)
        self.assertIn("linux/amd64", argv)
        self.assertIn("LLAMA_COMMIT=" + self.lock["llama_commit"], argv)
        self.assertIn("@sha256:", self.lock["cuda_base_image"])
        self.assertIn("12.9.1", self.lock["cuda_base_image"])
        self.assertEqual(len(self.lock["llama_commit"]), 40)
        self.assertEqual(len(self.lock["weights_sha256"]), 64)

    def test_download_resumes_and_verifies_whole_file(self):
        lock = self.small_lock()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / (lock["model_filename"] + ".part")).write_bytes(b"abc")
            with patch.object(deploy, "urlopen", return_value=Response(b"def", 206, "bytes 3-5/6")) as request:
                path = deploy.download_weights(folder, lock)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")
            provenance = json.loads((folder / "model-provenance.json").read_text())
            self.assertEqual(provenance["weights_sha256"], lock["weights_sha256"])

    def test_range_ignored_restarts_instead_of_appending(self):
        lock = self.small_lock()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / (lock["model_filename"] + ".part")).write_bytes(b"abc")
            with patch.object(deploy, "urlopen", return_value=Response(b"abcdef", 200)):
                self.assertEqual(deploy.download_weights(folder, lock).read_bytes(), b"abcdef")

    def test_wrong_range_and_corrupt_weights_fail_closed(self):
        lock = self.small_lock()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            partial = folder / (lock["model_filename"] + ".part")
            partial.write_bytes(b"abc")
            with patch.object(deploy, "urlopen", return_value=Response(b"def", 206, "bytes 0-2/6")):
                with self.assertRaisesRegex(ValueError, "Content-Range"):
                    deploy.download_weights(folder, lock)
            self.assertEqual(partial.read_bytes(), b"abc")
            path = folder / lock["model_filename"]
            path.write_bytes(b"xxxxxx")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                deploy.verify_weights(path, lock)

    def test_hardware_profile_rejects_less_memory_or_wrong_gpu_count(self):
        valid = "0, Tesla V100S-PCIE-32GB, GPU-a, 7.0, 32510, 580.65.06\n1, Tesla V100S-PCIE-32GB, GPU-b, 7.0, 32510, 580.65.06\n"
        self.assertEqual(len(deploy.parse_gpuinfo(valid)), 2)
        with self.assertRaisesRegex(ValueError, "exactly two"):
            deploy.parse_gpuinfo(valid.splitlines()[0])
        with self.assertRaisesRegex(ValueError, "exactly two"):
            deploy.parse_gpuinfo(valid.replace("32510", "16384"))

    def test_launch_uses_verified_image_id_and_persists_requested_configuration(self):
        lock = self.small_lock()
        gpuinfo = "0, Tesla V100S-PCIE-32GB, GPU-a, 7.0, 32510, 580.65.06\n1, Tesla V100S-PCIE-32GB, GPU-b, 7.0, 32510, 580.65.06\n"
        image_id = "sha256:" + "1" * 64
        inspection = json.dumps([{"Id": image_id, "Config": {"Labels": {"org.opencontainers.image.revision": lock["llama_commit"]}}}])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / lock["model_filename"]).write_bytes(b"abcdef")
            args = argparse.Namespace(model_dir=folder, launch_record=folder / "launch.json", port=8000, context=32768, parallel=3, batch_size=256, ubatch_size=64)
            with patch.object(subprocess, "check_output", side_effect=[gpuinfo, inspection, ""]):
                command, record = deploy.prepare_launch(args, lock)
            self.assertIn(image_id, command)
            self.assertNotIn(deploy.image_name(lock), command)
            self.assertIn("127.0.0.1:8000:8000", command)
            self.assertEqual(record["context_total"], 98304)
            self.assertEqual(record["batch_size"], 256)
            self.assertEqual(record["ubatch_size"], 64)
            self.assertEqual(record["status"], "launch_requested_not_validated")
            self.assertEqual(json.loads(args.launch_record.read_text())["weights_sha256"], lock["weights_sha256"])

    def test_stale_image_is_rejected_without_overwriting_record(self):
        lock = self.small_lock()
        gpuinfo = "0, Tesla V100S-PCIE-32GB, GPU-a, 7.0, 32510, 580.65.06\n1, Tesla V100S-PCIE-32GB, GPU-b, 7.0, 32510, 580.65.06\n"
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / lock["model_filename"]).write_bytes(b"abcdef")
            args = argparse.Namespace(model_dir=folder, launch_record=folder / "launch.json", port=8000, context=32768, parallel=3, batch_size=512, ubatch_size=128)
            with patch.object(subprocess, "check_output", side_effect=[gpuinfo, '[{"Config":{"Labels":{}}}]']):
                with self.assertRaisesRegex(ValueError, "revision"):
                    deploy.prepare_launch(args, lock)
            self.assertFalse(args.launch_record.exists())


if __name__ == "__main__":
    unittest.main()
