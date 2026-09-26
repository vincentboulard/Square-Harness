"""Single-A10 deployment checks; no real GPU or network is required."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "deployment" / "llama_a10.py"
spec = importlib.util.spec_from_file_location("llama_a10", MODULE_PATH)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
GPU = "0, NVIDIA A10, GPU-a, 8.6, 23028, 580.65.06\n"


class Response(io.BytesIO):
    status = 200
    headers = {}


class A10DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.lock = deploy.load_lock()

    def small_lock(self):
        return dict(self.lock, weights_bytes=6, weights_sha256=hashlib.sha256(b"abcdef").hexdigest())

    def inspection(self, lock, **extra_labels):
        labels = {"org.opencontainers.image.revision": lock["llama_commit"],
                  "io.square-harness.profile": "a10-q4", "io.square-harness.cuda-architectures": "86"}
        labels.update(extra_labels)
        return json.dumps([{"Id": "sha256:" + "a" * 64, "Config": {"Labels": labels}}])

    def arguments(self, folder):
        return argparse.Namespace(model_dir=folder, launch_record=folder / "launch.json",
                                  port=8000, context=32768, batch_size=256, ubatch_size=64)

    def test_single_slot_preserves_context_sampling_and_gpu_placement(self):
        argv = deploy.server_arguments(self.lock, 32768)
        expected = {"--device": "CUDA0", "--split-mode": "none", "--gpu-layers": "all",
                    "--fit": "off", "--parallel": "1", "--ctx-size": "32768",
                    "--alias": "square-qwen-a10", "--cache-type-k": "f16",
                    "--cache-type-v": "f16", "--flash-attn": "off",
                    "--batch-size": "256", "--ubatch-size": "64", "--temp": "0.6",
                    "--top-p": "0.95", "--top-k": "0", "--min-p": "0"}
        for flag, value in expected.items():
            self.assertEqual(argv[argv.index(flag) + 1], value)
        for flag in ("--no-context-shift", "--no-kv-unified", "--slots", "--metrics", "--jinja"):
            self.assertIn(flag, argv)
        self.assertNotIn("--tensor-split", argv)
        self.assertNotIn("CUDA0,CUDA1", argv)

    def test_a10_and_v100_locks_remain_distinct(self):
        previous = deploy.shared.load_lock()
        self.assertEqual(previous["parallel"], 3)
        self.assertEqual(previous["quantization"], "Q8_0")
        self.assertEqual(self.lock["parallel"], 1)
        self.assertEqual(self.lock["quantization"], "Q4_K_M")
        self.assertNotEqual(self.lock["weights_sha256"], previous["weights_sha256"])
        self.assertEqual(self.lock["template_sha256"], previous["template_sha256"])
        argv = deploy.shared.server_arguments(previous, 32768, 3)
        self.assertEqual(argv[argv.index("--parallel") + 1], "3")
        self.assertEqual(argv[argv.index("--device") + 1], "CUDA0,CUDA1")

    def test_build_uses_separate_pinned_ampere_image_and_eight_jobs(self):
        argv = deploy.build_command(self.lock, 8)
        self.assertIn("BUILD_JOBS=8", argv)
        self.assertIn("LLAMA_COMMIT=" + self.lock["llama_commit"], argv)
        self.assertIn("CUDA_BASE_IMAGE=" + self.lock["cuda_base_image"], argv)
        self.assertIn(str(MODULE_PATH.with_name("Dockerfile.llama-a10")), argv)
        self.assertIn("square-harness/llama-a10:" + self.lock["llama_commit"][:16], argv)
        dockerfile = MODULE_PATH.with_name("Dockerfile.llama-a10").read_text()
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=86", dockerfile)
        self.assertIn("LLAMA_USE_PREBUILT_UI=OFF", dockerfile)

    def test_hardware_profile_rejects_v100_a100_and_insufficient_memory(self):
        self.assertEqual(deploy.parse_gpuinfo(GPU)[0]["memory_mib"], 23028)
        for invalid in (GPU + GPU, GPU.replace("A10", "A100"), GPU.replace("A10", "V100"),
                        GPU.replace("8.6", "7.0"), GPU.replace("23028", "16384")):
            with self.assertRaisesRegex(ValueError, "exactly one A10"):
                deploy.parse_gpuinfo(invalid)

    def test_launch_records_verified_q4_image_and_single_slot(self):
        lock = self.small_lock()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / lock["model_filename"]).write_bytes(b"abcdef")
            args = self.arguments(folder)
            with patch.object(subprocess, "check_output", side_effect=[GPU, self.inspection(lock), ""]):
                command, record = deploy.prepare_launch(args, lock)
            self.assertIn("sha256:" + "a" * 64, command)
            self.assertIn("square-qwen-a10", command)
            self.assertEqual(record["profile"], "a10-q4")
            self.assertEqual(record["parallel"], 1)
            self.assertEqual(record["context_total"], 32768)
            self.assertEqual(record["status"], "launch_requested_not_validated")
            self.assertEqual(json.loads(args.launch_record.read_text())["weights_sha256"], lock["weights_sha256"])

    def test_wrong_image_or_existing_container_cannot_replace_launch_record(self):
        lock = self.small_lock()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / lock["model_filename"]).write_bytes(b"abcdef")
            args = self.arguments(folder)
            args.launch_record.write_text("original")
            bad = self.inspection(lock, **{"io.square-harness.cuda-architectures": "70"})
            for responses, message in (([GPU, bad], "architecture"),
                                       ([GPU, self.inspection(lock), "running-container"], "already exists")):
                with patch.object(subprocess, "check_output", side_effect=responses):
                    with self.assertRaisesRegex(ValueError, message):
                        deploy.prepare_launch(args, lock)
                self.assertEqual(args.launch_record.read_text(), "original")

    def test_shared_download_records_q4_and_has_a10_error_guidance(self):
        lock = self.small_lock()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = io.StringIO()
            with patch.object(deploy.shared, "urlopen", return_value=Response(b"abcdef")), contextlib.redirect_stdout(output):
                deploy.shared.download_weights(folder, lock)
            self.assertIn("Q4_K_M", output.getvalue())
            self.assertNotIn("Q8_0", output.getvalue())
            self.assertEqual(json.loads((folder / "model-provenance.json").read_text())["quantization"], "Q4_K_M")
            with self.assertRaisesRegex(ValueError, "download-llama-a10.sh"):
                deploy.shared.verify_weights(folder / "missing.gguf", lock)

    def test_v100_environment_does_not_change_a10_defaults(self):
        with patch.dict(os.environ, {"MODEL_DIR": "/v100", "MODEL_PORT": "9999", "MAX_NUM_SEQS": "3", "BATCH_SIZE": "4096"}, clear=True):
            with patch.object(deploy, "prepare_launch", side_effect=ValueError("test stop")) as prepare:
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(deploy.main(["serve"]), 1)
            args = prepare.call_args.args[0]
            self.assertEqual(args.model_dir, Path("~/.cache/square-harness/models-a10"))
            self.assertEqual(args.port, 8000)
            self.assertEqual(args.batch_size, 256)
            self.assertEqual(args.launch_record, Path("~/square-runs/a10-server/launch.json"))


if __name__ == "__main__":
    unittest.main()
