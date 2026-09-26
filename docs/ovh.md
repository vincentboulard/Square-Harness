# OVH H100 experiment

Target: one H100 with 80 GB GPU memory, Ubuntu 24.04, Python 3.10+.
Prepare the private dataset locally before renting. The scripts do not create
or purchase cloud resources. No H100 measurement has yet been made for this project.

## Model server

Use **one vLLM process** for all direct-model and harness requests. Multiple
clients share that model through continuous batching; do not load one model per
agent. Ollama supports parallel requests too, but its default `qwen3.8:27b` is
quantized. Keep the checkpoint, dtype and server identical across experimental arms.

Pinned starting configuration:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B` |
| Model/tokenizer revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| vLLM image | `vllm/vllm-openai:v0.30.0-cu129` (record and then pin its digest) |
| Weights | BF16, no weight quantization or CPU offload |
| Context | 32,768 tokens, including input and output |
| Active sequences | 4 |
| Scheduled batch tokens | 8,192; this is not the context limit |
| GPU allocation fraction | 0.90 |
| KV precision | `auto`; keep default recurrent-state precision |
| Reasoning / tool parser | `qwen3` / `qwen3_xml` |

The checkpoint files total about 55.6 GB. The model configuration implies about
64 KiB of full-attention KV per cached token: approximately 2 GiB for a full 32K
sequence, or 8 GiB for four. These estimates exclude recurrent-state storage,
activation buffers, CUDA graphs, allocation padding and other runtime memory.
BF16 at 4 × 32K is a starting hypothesis, not a measured fit guarantee. System
RAM helps loading and storage caching; it does not replace GPU memory.

The native 262,144-token context is unnecessary for these short statements and
would sharply reduce concurrency. Move to 64K only after observing actual
context failures and measuring memory. Do not silently switch to FP8 or a
quantized Ollama model midway through a comparison.

## Before the first launch

1. Verify `nvidia-smi` identifies the intended GPU and 80 GB memory. Install Docker
   and the NVIDIA Container Toolkit using their official Ubuntu instructions.
   Check host-driver compatibility with the pinned CUDA 12.9 image.
2. Allow room for model downloads, Docker images and result logs. About 150 GB
   of free disk is a prudent starting allowance; retain more if keeping variants.
3. Clone this repository and install its small CPU-side package:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   docker pull vllm/vllm-openai:v0.30.0-cu129
   docker image inspect vllm/vllm-openai:v0.30.0-cu129 --format '{{json .RepoDigests}}'
   ```

4. Save the returned image digest and run the server using it:

   ```bash
   export VLLM_IMAGE='vllm/vllm-openai@sha256:PASTE_VERIFIED_DIGEST'
   bash scripts/serve-vllm.sh 2>&1 | tee ~/square-vllm-startup.log
   ```

   The initial model download occurs here and takes paid instance time. Keep
   this process in a persistent terminal session. The host port is bound only
   to loopback; inference stays on the server. For a Mac-side client, use an
   SSH tunnel: `ssh -L 8000:127.0.0.1:8000 ubuntu@YOUR_SERVER`.

## Validate serving before scoring

Run the live transport checks in another terminal:

```bash
source .venv/bin/activate
python scripts/smoke-vllm.py --host http://127.0.0.1:8000 --model square-qwen
```

They check ordinary generation, reasoning, a tool round trip, JSON schema output,
streamed usage, a capped response and concurrent requests. These are software
checks, not mathematical grading. Retain the JSON output in the experiment record.

Then generate a synthetic dataset and rehearse the entire benchmark workflow
using the commands in [benchmark.md](benchmark.md). Keep scored problems out of
calibration. Inspect server startup KV capacity and logs for preemptions or OOMs.
Measure concurrency 1, 2 and 4 with representative output lengths; fix settings
before the scored run. Higher aggregate throughput can increase individual latency.

If startup fails from memory pressure, reduce context/concurrency and batch-token
size. An eager-mode run can diagnose CUDA-graph memory pressure. If parsing fails,
fix the parser/version combination before benchmarking; a failed tool or JSON
transport is not a mathematical failure. The exact-model recipe is not an H100
performance validation, so retain this live acceptance step.

## Benchmark and retain results

Copy only the prepared statement dataset to a private directory outside this
repository, for example `~/square-data/pilot-10-v1`. Keep annotated sources and
grading references elsewhere. Follow [benchmark.md](benchmark.md) for dry-run,
the one-replicate pilot and the optional three-replicate experiment.

Record the image digest, model and tokenizer revisions, driver/GPU details,
`git rev-parse HEAD`, any working-tree patch, server startup log and run manifest.
The client cannot infer a checkpoint revision from a served alias: verify it
against the actual server launch command. Seeds aid reproducibility but do not
guarantee identical output across different scheduling or runtime environments.

Save request-level prompt tokens, generated tokens (including thinking), charged
reservations, timing, all candidates, selected answers and error states. Equal
generated-token ceilings do not imply equal GPU time or equal input-token work.
Estimate rental duration from measured throughput and setup time, not model-size
arithmetic. Copy the complete results to your Mac and verify checksums before
releasing cloud storage or the instance.

## Primary references checked for this configuration

- [Pinned Qwen model card and configuration](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- [Official exact-model vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml)
- [vLLM 0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
- [Pinned engine arguments](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/arg_utils.py)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [CUDA 12.9 driver release notes](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html)
- [Ollama concurrency documentation](https://docs.ollama.com/faq)

Qwen's published thinking temperature is 1.0. This first comparison explicitly
uses the existing harness solver temperature 0.6 and top-p 0.95 across all solver
arms, with zero-temperature reviewers. A temperature change is a separately
recorded experimental configuration.
