# OVH experiment: two V100S GPUs

This Q8 profile remains available unchanged. For the current single-A10 Q4 pilot,
use the separate [A10 guide](a10.md) and its A10-specific launchers.

Target: **OVH t2-90, 2 × Tesla V100S 32 GB**, 90 GB system RAM, 30 vcores,
800 GB NVMe, x86-64 Ubuntu 22.04 or 24.04, Python 3.10+.
The scripts do not rent a server. No V100S GPU inference has yet been measured
for this project; the live checks below must pass before scoring problems.

## Frozen experimental condition

Use **one llama.cpp server and one Q8_0 model distributed across both GPUs**.
All direct answers, sequential proof searches and parallel proof searches use
this same endpoint. Three agents do not load three model copies.

| Setting | Value |
| --- | --- |
| Model | Qwen3.8-27B, official Ollama Q8_0 GGUF, text only |
| Model SHA-256 | `2bb22714289826d7b9e0ba376c3ce47d08bce39abe598745857c44d88c09bdbf` |
| llama.cpp commit | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` |
| Build | CUDA 12.9.1, explicitly compiled for V100 `sm_70` |
| Weights | Q8_0, about 29.05 GB / 27.05 GiB; download verified by SHA-256 |
| GPU allocation | All model layers, split equally across two GPUs |
| Context | **32,768 tokens per request**, including input and output |
| Concurrent slots | **3**; explicit total context 98,304, separate KV allocation |
| KV cache / Flash Attention | F16 / off for the initial compatibility configuration |
| Sampling | Solver temperature 0.6, top-p 0.95; top-k disabled, min-p 0, neutral repetition/presence/frequency penalties; reviewers temperature 0 |
| Prompt template | Embedded Qwen Jinja template, fingerprinted in the launch record |
| Context shifting / automatic fitting | Disabled; no silent truncation or context reduction |
| Endpoint | `http://127.0.0.1:8000`, served alias `square-qwen` |

The immutable build and model identifiers are stored in
[`deployment/llama-v100.lock.json`](../deployment/llama-v100.lock.json).
Q8_0 is weight quantization, not FP8 or BF16 inference. V100 does not have native
BF16 support, and the prepared vLLM 0.30.0 image requires a newer GPU architecture.
The optional `serve-vllm.sh` remains an H100 recipe; do not use it on V100.

The official Q8 artifact includes a text-model GGUF plus a separate vision
projector. We download only the text model. Its embedded template supports
thinking and tools; neither the projector nor image inputs are needed here.

The smaller weights leave substantially more room than FP16/BF16 for context,
recurrent state and runtime buffers. That is a capacity hypothesis, not a fit
measurement. System RAM does not turn two 32 GB GPUs into one 64 GB device.
The initial split uses layers and makes no assumption about NVLink connectivity.

## Prepare the server

1. Verify `nvidia-smi` shows **two V100S with about 32 GB each**. Check the driver
   supports CUDA 12.9. Install Docker and the NVIDIA Container Toolkit using their
   [official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
   Run `nvidia-smi topo -m` and retain the output; topology can affect performance.
2. Keep at least 100 GB free for model weights, image builds and logs, with more
   room for accumulated experiment results. Use a persistent SSH terminal such
   as tmux for the server and benchmark.
3. Clone and install the harness:

   ```bash
   git clone https://github.com/vincentboulard/Square-Harness.git
   cd Square-Harness
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   mkdir -p "$HOME/square-runs/server" "$HOME/square-data"
   ```

4. Build the pinned CUDA 12 server, then download and verify the model. Building
   happens on the rented Linux host, not on your Mac:

   ```bash
   bash scripts/build-llama.sh
   bash scripts/download-llama-model.sh
   ```

   CUDA 13 is not a substitute: this build explicitly targets the older V100
   architecture. The default model directory is
   `~/.cache/square-harness/models`; override `MODEL_DIR` consistently if needed.
   The download is approximately 29 GB and setup consumes paid instance time.

5. Start the model server:

   ```bash
   set -o pipefail
   bash scripts/serve-llama.sh 2>&1 | tee "$HOME/square-runs/server/startup.log"
   ```

   Startup verifies the pinned model file and built image and records the exact
   launch in `~/square-runs/server/launch.json`. Keep this terminal running.
   Inspect the log: all model layers must be on the GPUs, each slot must have
   32,768 context tokens, and both GPUs must be used. The port is published only
   on loopback. A Mac-side client can use
   `ssh -L 8000:127.0.0.1:8000 ubuntu@YOUR_SERVER`.

## Validate before the scored run

In another terminal, enter the checkout and activate `.venv`:

```bash
source .venv/bin/activate
python scripts/smoke-serving.py \
  --backend llamacpp --host http://127.0.0.1:8000 --model square-qwen \
  --ctx 32768 --parallel 3 \
  --launch-record "$HOME/square-runs/server/launch.json" \
  --output "$HOME/square-runs/server/acceptance.json"
```

This checks the running build/template against the launch record, each slot's
context size, ordinary and thinking responses, a tool round trip, structured
JSON, streamed usage and output caps. Short and representative-context batches
check that three different requests actually decode concurrently. Merely sending
three HTTP requests is not evidence of concurrent inference. Synthetic filler
is used for capacity checks; scored problems are never used for tuning.

A passing report must have `accepted_for_benchmark: true`. These checks establish
software behavior for the tested loads, not mathematical correctness or a
throughput guarantee for every future request. Save the report and startup log.
The adapter also counts the exact formatted prompt before every generation and
rejects requests whose full input plus output allowance would exceed context.

If startup or capacity checks fail, stop before scoring. Inspect the logs and
GPU memory. Adjust batch/physical batch sizes before changing scientific context
or quantization, for example `bash scripts/serve-llama.sh --batch-size 256 --ubatch-size 64`.
Stop the previous server first; any changed server launch needs a fresh acceptance
report under a new filename. Pass that report to the benchmark wrapper.
The fixed pilot wrapper requires three 32K slots. Reducing context to 16K would
also invalidate the original 17,952-token direct-answer allowance. Do not make
that change silently. A two-slot fallback would require an explicitly revised
serving profile and acceptance test, while preserving three logical branches.

## Rehearse, then run the same pilot

Create a tiny synthetic dataset and first check one harness search per problem:

```bash
python -m mathagent.benchmark --init-smoke "$HOME/square-data/smoke-statements"
python -m mathagent.benchmark \
  --manifest "$HOME/square-data/smoke-statements/manifest.json" \
  --output "$HOME/square-runs/development-q8-001" \
  --backend llamacpp --host http://127.0.0.1:8000 --model square-qwen \
  --arms sequential --ctx 32768 --tokens 6000 --selection-tokens 1536 \
  --predict 2048 --max-predict 2048 --rounds 2 --seconds 600 \
  --workers 1 --max-in-flight 1
```

This reduced development check is capped at 12,000 generated tokens across two
jobs; it does not measure the full three-method comparison. Inspect transport,
context, recorder diagnostics and `proof.md` before testing a bounded portfolio
separately. An unfinished toy is a diagnostic to inspect, not a reason to launch
the scored benchmark automatically. Keep the serving acceptance report for the
scored wrapper below.

The scored wrapper uses generous guards: 14,400 seconds per job, 7,200
per request, and up to 1,800 reserved for selection. These are upper bounds, not
runtime estimates. Fix any changed guards on synthetic inputs and use them
consistently for every arm. No unused token allowance is consumed artificially.

Copy **only the existing statement-only dataset** to the server. From your Mac,
substitute your actual SSH destination:

```bash
scp -r /Users/vincent/Documents/SquareHarness/benchmarks/pilot-10-v1 \
  ubuntu@YOUR_SERVER:~/square-data/
```

Keep annotated sources, hints and grading references on your Mac. The public
repository intentionally does not contain those private benchmark files.

On the server, validate the frozen plan without network or output writes:

```bash
python scripts/run-v100-benchmark.py \
  --manifest "$HOME/square-data/pilot-10-v1/manifest.json" \
  --output "$HOME/square-runs/pilot-q8-001" \
  --acceptance "$HOME/square-runs/server/acceptance.json" \
  --dry-run
```

Then run the same command without `--dry-run`. The wrapper rechecks the current
server and refuses missing or mismatched acceptance evidence. Its plan embeds
both serving records, model/image hashes, dataset hashes, code hashes and Git
revision. Existing output directories are never overwritten.

The scientific comparison is unchanged: **ten statements; direct best-of-three,
sequential harness and three-branch harness; 60,000 generated tokens per method
and problem, including thinking and reviews; blinded human grading**. One
replicate makes 30 jobs and authorizes at most 1.8 million generated tokens.
The numerical condition is now Q8_0; report it separately from any BF16 run.
Equal output budgets do not imply equal input-token work or equal GPU time.
See [benchmark.md](benchmark.md) for the exact allocation and failure policy.

Estimate the rental duration from measured throughput and rehearsal times.
Copy the complete results and serving logs back to your Mac and verify their
checksums before releasing the instance or its storage.

## Primary references

- [Ollama Qwen3.8-27B Q8_0 artifact](https://ollama.com/library/qwen3.8:27b-q8_0)
- [Pinned llama.cpp server API and options](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/tools/server/README.md)
- [Pinned CUDA build configuration](https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/ggml/src/ggml-cuda/CMakeLists.txt)
- [NVIDIA CUDA 12.9 driver notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)
- [vLLM 0.30.0 GPU requirements](https://docs.vllm.ai/en/v0.30.0/getting_started/installation/gpu/)

Ollama remains supported for local use. Its current Qwen architecture restriction
would serialize requests in one model instance; this deployment uses llama.cpp
directly to test shared-model concurrency.
