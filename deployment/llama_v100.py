"""Pinned, text-only llama.cpp deployment for two 32 GiB V100 GPUs.

Uses only the Python standard library on the host. Model data and launch records
live outside the repository. A launch record describes a requested launch, not a
successful GPU acceptance test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "deployment" / "llama-v100.lock.json"


def load_lock() -> dict:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def image_name(lock: dict) -> str:
    return "square-harness/llama-v100:" + lock["llama_commit"][:16]


def positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(path: Path, lock: dict) -> None:
    if not path.is_file() or path.stat().st_size != lock["weights_bytes"]:
        raise ValueError(f"Missing or wrong-sized model: {path}. Run download-llama-model.sh.")
    if sha256_file(path) != lock["weights_sha256"]:
        raise ValueError(f"Model SHA256 mismatch: {path}. Refusing inference.")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as tmp:
        json.dump(value, tmp, indent=2)
        tmp.write("\n")
        temporary = tmp.name
    os.replace(temporary, path)


def outside_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("Keep downloaded weights and runtime records outside the Git repository.")
    return resolved


def download_weights(directory: Path, lock: dict) -> Path:
    directory = outside_repository(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / lock["model_filename"]
    if destination.exists():
        verify_weights(destination, lock)
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    expected = lock["weights_bytes"]
    if offset > expected:
        raise ValueError(f"Oversized partial download: remove {partial} before retrying.")
    if offset < expected:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = Request(lock["weights_url"], headers=headers)
        with urlopen(request, timeout=120) as response:
            if offset and response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected}"):
                    raise ValueError("Unexpected model download Content-Range; refusing to append.")
                mode = "ab"
            elif response.status == 200:
                mode = "wb"  # Server ignored Range; restart rather than corrupting the file.
            else:
                raise ValueError(f"Unexpected model download HTTP status {response.status}")
            print(f"Downloading pinned Q8_0 model to {partial}", flush=True)
            with partial.open(mode) as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    verify_weights(partial, lock)
    os.replace(partial, destination)
    write_json(directory / "model-provenance.json", {
        "schema_version": 1,
        "source": lock["model_source"],
        "source_manifest_sha256": lock["model_manifest_sha256"],
        "weights_url": lock["weights_url"],
        "weights_sha256": lock["weights_sha256"],
        "weights_bytes": lock["weights_bytes"],
        "quantization": lock["quantization"],
        "template_sha256": lock["template_sha256"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    })
    return destination


def build_command(lock: dict, jobs: int) -> list[str]:
    return [
        "docker", "build", "--platform", "linux/amd64",
        "--build-arg", "CUDA_BASE_IMAGE=" + lock["cuda_base_image"],
        "--build-arg", "LLAMA_COMMIT=" + lock["llama_commit"],
        "--build-arg", f"BUILD_JOBS={jobs}",
        "-f", str(ROOT / "deployment" / "Dockerfile.llama-v100"),
        "-t", image_name(lock), str(ROOT / "deployment"),
    ]


def server_arguments(lock: dict, context: int, parallel: int,
                     batch_size: int = 512, ubatch_size: int = 128) -> list[str]:
    # At this pinned revision, non-unified KV divides total context by slots.
    # Exact multiples of 256 prevent llama.cpp's per-sequence rounding.
    if context < 256 or context % 256 or context > 262144:
        raise ValueError("Context per slot must be a multiple of 256, between 256 and 262144.")
    if parallel < 1:
        raise ValueError("Parallel slots must be positive.")
    if not 1 <= ubatch_size <= batch_size:
        raise ValueError("Batch sizes must satisfy 1 <= ubatch-size <= batch-size.")
    return [
        "--model", "/models/" + lock["model_filename"],
        "--alias", lock["model_alias"], "--host", "0.0.0.0", "--port", "8000",
        "--device", "CUDA0,CUDA1", "--split-mode", "layer", "--tensor-split", "1,1",
        "--gpu-layers", "all", "--fit", "off", "--no-mmproj",
        "--ctx-size", str(context * parallel), "--parallel", str(parallel),
        "--no-kv-unified", "--no-context-shift", "--cont-batching",
        "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "off",
        "--batch-size", str(batch_size), "--ubatch-size", str(ubatch_size), "--ctx-checkpoints", "4",
        "--cache-ram", "0", "--threads", "15", "--threads-http", "8",
        "--jinja", "--reasoning-format", "deepseek",
        "--chat-template-kwargs", '{"enable_thinking":true,"reasoning_effort":"xhigh","preserve_thinking":true}',
        "--temp", "0.6", "--top-p", "0.95", "--top-k", "0", "--min-p", "0",
        "--repeat-penalty", "1", "--presence-penalty", "0", "--frequency-penalty", "0",
        "--slots", "--metrics", "--no-webui", "--timeout", "7200",
    ]


def parse_gpuinfo(output: str) -> list[dict]:
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 6:
            raise ValueError("Unexpected nvidia-smi GPU information.")
        index, name, uuid, capability, memory, driver = (value.strip() for value in row)
        rows.append({"index": int(index), "name": name, "uuid": uuid,
                     "compute_capability": capability, "memory_mib": int(memory), "driver_version": driver})
    if len(rows) != 2 or any("V100" not in gpu["name"] or gpu["compute_capability"] != "7.0"
                             or gpu["memory_mib"] < 30000 for gpu in rows):
        raise ValueError("This profile requires exactly two V100 GPUs with at least 30000 MiB each.")
    return rows


def prepare_launch(args: argparse.Namespace, lock: dict) -> tuple[list[str], dict]:
    directory = outside_repository(args.model_dir)
    record_path = outside_repository(args.launch_record)
    if not 1 <= args.port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    server_argv = server_arguments(lock, args.context, args.parallel, args.batch_size, args.ubatch_size)
    model_path = directory / lock["model_filename"]
    print("Verifying full model SHA256 before launch...", flush=True)
    verify_weights(model_path, lock)
    gpu_query = ["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"]
    gpuinfo = parse_gpuinfo(subprocess.check_output(gpu_query, text=True))
    image = image_name(lock)
    inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
    if inspected.get("Config", {}).get("Labels", {}).get("org.opencontainers.image.revision") != lock["llama_commit"]:
        raise ValueError("Docker image source revision does not match the lock; rebuild it.")
    if subprocess.check_output(["docker", "ps", "-a", "--filter", "name=^/square-qwen$", "--format", "{{.ID}}"], text=True).strip():
        raise ValueError("A square-qwen container already exists; stop it before writing a new launch record.")
    image_id = inspected["Id"]
    # Launch the inspected immutable image ID, not a tag that could move after inspection.
    command = ["docker", "run", "--rm", "--name", "square-qwen", "--gpus", "all", "--ipc=host",
               "-p", f"127.0.0.1:{args.port}:8000", "-v", f"{directory}:/models:ro", image_id, *server_argv]
    record = {
        "schema_version": 1, "launch_requested_at": datetime.now(timezone.utc).isoformat(),
        "status": "launch_requested_not_validated", "weights_sha256": lock["weights_sha256"],
        "weights_bytes": lock["weights_bytes"], "model_alias": lock["model_alias"],
        "model_path": "/models/" + lock["model_filename"], "model_host_path": str(model_path),
        "model_source": lock["model_source"], "quantization": lock["quantization"],
        "template_sha256": lock["template_sha256"], "llama_commit": lock["llama_commit"],
        "image_id": image_id, "image_tag": image, "cuda_base_image": lock["cuda_base_image"],
        "context_per_slot": args.context, "context_total": args.context * args.parallel,
        "batch_size": args.batch_size, "ubatch_size": args.ubatch_size,
        "parallel": args.parallel, "host": f"http://127.0.0.1:{args.port}",
        "cache_type_k": "f16", "cache_type_v": "f16", "flash_attention": "off",
        "context_shift": False, "gpu_layers": "all", "fit": "off", "gpuinfo": gpuinfo,
        "argv": server_argv, "docker_argv": command,
    }
    write_json(record_path, record)
    return command, record


def main(argv: list[str] | None = None) -> int:
    lock = load_lock()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build", help="Build pinned CUDA 12 / SM70 llama-server image")
    builder.add_argument("--jobs", type=positive, default=15)
    model_default = Path(os.environ.get("MODEL_DIR", "~/.cache/square-harness/models"))
    downloader = sub.add_parser("download", help="Download and hash-check the immutable Q8_0 model")
    downloader.add_argument("--model-dir", type=Path, default=model_default)
    launcher = sub.add_parser("serve", help="Verify weights and image, then start one three-slot server")
    launcher.add_argument("--model-dir", type=Path, default=model_default)
    launcher.add_argument("--port", type=int, default=os.environ.get("MODEL_PORT", "8000"))
    launcher.add_argument("--context", type=positive, default=os.environ.get("CONTEXT_PER_SLOT", str(lock["context_per_slot"])))
    launcher.add_argument("--parallel", type=positive, default=os.environ.get("MAX_NUM_SEQS", str(lock["parallel"])))
    launcher.add_argument("--batch-size", type=positive, default=os.environ.get("BATCH_SIZE", "512"))
    launcher.add_argument("--ubatch-size", type=positive, default=os.environ.get("UBATCH_SIZE", "128"))
    launcher.add_argument("--launch-record", type=Path, default=Path(os.environ.get("LAUNCH_RECORD", "~/square-runs/server/launch.json")))
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            subprocess.run(build_command(lock, args.jobs), check=True)
        elif args.command == "download":
            print(download_weights(args.model_dir, lock))
        else:
            command, record = prepare_launch(args, lock)
            print(f"Starting {record['model_alias']} at {record['host']}: {args.parallel} slots, {args.context} tokens each.", flush=True)
            print(f"Launch record: {args.launch_record.expanduser()}. Run acceptance checks before the benchmark.", flush=True)
            os.execvp(command[0], command)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Deployment error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
