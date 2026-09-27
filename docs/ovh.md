# Retained two-V100S llama.cpp profile

Target: two Tesla V100S GPUs with 32 GB each, originally OVH t2-90 with 90 GB RAM,
30 vcores and 800 GB NVMe. The scripts do not rent a server. This hardware profile
is retained; the v0.5 experiment now compares only direct inference and basic
proof mode. No new V100 performance claim is made by the software update.

## Frozen serving settings

| Setting | Value |
| --- | --- |
| Text model | Qwen3.8-27B Q8_0, approximately 29.05 GB |
| Model SHA-256 | `2bb22714289826d7b9e0ba376c3ce47d08bce39abe598745857c44d88c09bdbf` |
| llama.cpp | `2145525a4081d66ff1a87cf43ef809f95a85ac0c` |
| Build | CUDA 12.9.1, SM70 |
| Placement | One model split equally across both GPUs |
| Context / slots | 32,768 per request / three serving slots |
| KV cache / Flash Attention | F16 / off |
| Endpoint / model alias | `http://127.0.0.1:8000` / `square-qwen` |

Exact identities are in
[`deployment/llama-v100.lock.json`](../deployment/llama-v100.lock.json).
Q8_0 is weight quantization. No vision projector is downloaded. Context shifting
and automatic fit reduction are disabled. Three serving slots do not introduce
three proof agents; the default v0.5 benchmark submits one job at a time.
The retained vLLM serving script targets newer hardware and is not this recipe.

## Setup and check

Use Linux with a working NVIDIA driver, Docker and NVIDIA Container Toolkit.
Verify both GPUs with `nvidia-smi`, retain `nvidia-smi topo -m`, and leave about
100 GB free for weights, images and logs. Install the harness as in the
[README](../README.md), then run from the checkout:

```bash
mkdir -p "$HOME/square-runs/server"
bash scripts/build-llama.sh
bash scripts/download-llama-model.sh
set -o pipefail
bash scripts/serve-llama.sh 2>&1 | tee "$HOME/square-runs/server/startup.log"
```

The model defaults to `~/.cache/square-harness/models` and the launch record to
`~/square-runs/server/launch.json`. Keep the server running in a persistent terminal.
Inspect logs for full GPU offload and the requested context. In another terminal:

```bash
python scripts/smoke-serving.py \
  --backend llamacpp --host http://127.0.0.1:8000 --model square-qwen \
  --ctx 32768 --parallel 3 \
  --launch-record "$HOME/square-runs/server/launch.json" \
  --output "$HOME/square-runs/server/acceptance.json"
```

These checks validate the retained server capacity, reasoning, protocol and token
accounting. They do not establish proof quality or speed for every request.
If memory pressure requires changing batch settings, relaunch and create a new
acceptance record. Do not silently reduce context or change weights.

## v0.5 benchmark

The wrapper uses the same 16,384-token initial allowance in both arms, an
8,192-token verifier allowance, and a 90,000-token proof-job ceiling over at most
three attempts. The model and 32k context differ from the
[HyperQwen A10 condition](hyperqwen-a10.md); report them separately.

After the [software smoke test](benchmark.md#software-smoke-test), inspect:

```bash
python scripts/run-v100-benchmark.py \
  --manifest "$HOME/square-data/pilot-10-v1/manifest.json" \
  --output "$HOME/square-runs/v100-v05-001" \
  --launch-record "$HOME/square-runs/server/launch.json" \
  --acceptance "$HOME/square-runs/server/acceptance.json" \
  --dry-run
```

Remove `--dry-run` only when the plan and serving checks are ready. Copy only
statement files to the server, retain source/model identities, and preserve all
answers for independent grading. The old parallel-portfolio experiment is not
reproduced by this v0.5 wrapper; reproduce it using its original source snapshot.
