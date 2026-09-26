"""Separate single-A10 Q4 profile; the two-V100 Q8 profile remains available.

One server slot executes queued requests from three independent proof branches.
Successful launch is not evidence of GPU fit for the full benchmark workload:
the separate live acceptance checks must pass before scoring.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# Reuse download/hash, atomic-write and prompt/sampling helpers without changing
# the V100 launcher's command-line defaults or its hardware validation.
_spec = importlib.util.spec_from_file_location("_llama_shared", Path(__file__).with_name("llama_v100.py"))
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)
ROOT = shared.ROOT
LOCK_PATH = Path(__file__).with_name("llama-a10.lock.json")
CONTAINER_NAME = "square-qwen-a10"


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def image_name(lock: dict) -> str:
    return "square-harness/llama-a10:" + lock["llama_commit"][:16]


def build_command(lock: dict, jobs: int) -> list[str]:
    return [
        "docker", "build", "--platform", "linux/amd64",
        "--build-arg", "CUDA_BASE_IMAGE=" + lock["cuda_base_image"],
        "--build-arg", "LLAMA_COMMIT=" + lock["llama_commit"],
        "--build-arg", f"BUILD_JOBS={jobs}",
        "-f", str(ROOT / "deployment" / "Dockerfile.llama-a10"),
        "-t", image_name(lock), str(ROOT / "deployment"),
    ]


def server_arguments(lock: dict, context: int, batch_size: int = 256,
                     ubatch_size: int = 64) -> list[str]:
    argv = shared.server_arguments(lock, context, 1, batch_size, ubatch_size)
    argv[argv.index("--device") + 1] = "CUDA0"
    argv[argv.index("--split-mode") + 1] = "none"
    split_index = argv.index("--tensor-split")
    del argv[split_index:split_index + 2]
    # This revision emits model placement and memory allocation at trace level.
    # Keep the evidence needed to verify full GPU offload in startup logs.
    argv += ["--log-verbosity", "4"]
    return argv


def parse_gpuinfo(output: str) -> list[dict]:
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 6:
            raise ValueError("Unexpected nvidia-smi GPU information.")
        index, name, uuid, capability, memory, driver = (value.strip() for value in row)
        rows.append({"index": int(index), "name": name, "uuid": uuid,
                     "compute_capability": capability, "memory_mib": int(memory), "driver_version": driver})
    if len(rows) != 1 or rows[0]["name"].split()[-1] not in ("A10", "A10G") or rows[0]["compute_capability"] != "8.6" or rows[0]["memory_mib"] < 22000:
        raise ValueError("This profile requires exactly one A10/A10G GPU with at least 22000 MiB.")
    return rows


def prepare_launch(args: argparse.Namespace, lock: dict) -> tuple[list[str], dict]:
    directory = shared.outside_repository(args.model_dir)
    record_path = shared.outside_repository(args.launch_record)
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    server_argv = server_arguments(lock, args.context, args.batch_size, args.ubatch_size)
    model_path = directory / lock["model_filename"]
    print("Verifying full Q4_K_M model SHA256 before launch...", flush=True)
    shared.verify_weights(model_path, lock)
    gpu_query = ["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"]
    gpuinfo = parse_gpuinfo(subprocess.check_output(gpu_query, text=True))
    image = image_name(lock)
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    labels = inspected.get("Config", {}).get("Labels", {})
    expected_labels = {"org.opencontainers.image.revision": lock["llama_commit"],
                       "io.square-harness.profile": "a10-q4", "io.square-harness.cuda-architectures": "86"}
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise ValueError("Docker image source/profile/architecture does not match the A10 lock; rebuild it.")
    if subprocess.check_output(["docker", "ps", "-a", "--filter", f"name=^/{CONTAINER_NAME}$", "--format", "{{.ID}}"], text=True).strip():
        raise ValueError(f"A {CONTAINER_NAME} container already exists; stop it before writing a new launch record.")
    image_id = inspected["Id"]
    command = ["docker", "run", "--rm", "--name", CONTAINER_NAME, "--gpus", "all", "--ipc=host",
               "-p", f"127.0.0.1:{args.port}:8000", "-v", f"{directory}:/models:ro", image_id, *server_argv]
    record = {
        "schema_version": 1, "profile": "a10-q4",
        "launch_requested_at": datetime.now(timezone.utc).isoformat(),
        "status": "launch_requested_not_validated", "weights_sha256": lock["weights_sha256"],
        "weights_bytes": lock["weights_bytes"], "model_alias": lock["model_alias"],
        "model_path": "/models/" + lock["model_filename"], "model_host_path": str(model_path),
        "model_source": lock["model_source"], "quantization": lock["quantization"],
        "template_sha256": lock["template_sha256"], "llama_commit": lock["llama_commit"],
        "image_id": image_id, "image_tag": image, "cuda_base_image": lock["cuda_base_image"],
        "cuda_architectures": "86", "context_per_slot": args.context, "context_total": args.context,
        "batch_size": args.batch_size, "ubatch_size": args.ubatch_size,
        "parallel": 1, "host": f"http://127.0.0.1:{args.port}",
        "cache_type_k": "f16", "cache_type_v": "f16", "flash_attention": "off",
        "context_shift": False, "gpu_layers": "all", "fit": "off", "gpuinfo": gpuinfo,
        "argv": server_argv, "docker_argv": command,
    }
    shared.write_json(record_path, record)
    return command, record


def main(argv: list[str] | None = None) -> int:
    lock = load_lock()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build", help="Build pinned CUDA 12 / SM86 llama-server image")
    builder.add_argument("--jobs", type=shared.positive, default=8)
    model_default = Path(os.environ.get("A10_MODEL_DIR", "~/.cache/square-harness/models-a10"))
    downloader = sub.add_parser("download", help="Download and hash-check the immutable Q4_K_M model")
    downloader.add_argument("--model-dir", type=Path, default=model_default)
    launcher = sub.add_parser("serve", help="Verify weights and image, then start one A10 inference slot")
    launcher.add_argument("--model-dir", type=Path, default=model_default)
    launcher.add_argument("--port", type=int, default=os.environ.get("A10_MODEL_PORT", "8000"))
    launcher.add_argument("--context", type=shared.positive, default=os.environ.get("A10_CONTEXT_PER_SLOT", str(lock["context_per_slot"])))
    launcher.add_argument("--batch-size", type=shared.positive, default=os.environ.get("A10_BATCH_SIZE", "256"))
    launcher.add_argument("--ubatch-size", type=shared.positive, default=os.environ.get("A10_UBATCH_SIZE", "64"))
    launcher.add_argument("--launch-record", type=Path, default=Path(os.environ.get("A10_LAUNCH_RECORD", "~/square-runs/a10-server/launch.json")))
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            subprocess.run(build_command(lock, args.jobs), check=True)
        elif args.command == "download":
            print(shared.download_weights(args.model_dir, lock))
        else:
            command, record = prepare_launch(args, lock)
            print(f"Starting {record['model_alias']} at {record['host']}: one slot, {args.context} context tokens.", flush=True)
            print(f"Launch record: {args.launch_record.expanduser()}. Run acceptance checks before the benchmark.", flush=True)
            os.execvp(command[0], command)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"A10 deployment error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
