#!/usr/bin/env bash
# One text-only model server; all benchmark arms use this same endpoint.
set -euo pipefail

image="${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0-cu129}"
model="${MODEL_ID:-Qwen/Qwen3.8-27B}"
revision="${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
cache="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"
context="${MAX_MODEL_LEN:-32768}"
sequences="${MAX_NUM_SEQS:-4}"
batch_tokens="${MAX_BATCHED_TOKENS:-8192}"
memory="${GPU_MEMORY_UTILIZATION:-0.90}"
port="${MODEL_PORT:-8000}"

command -v docker >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
mkdir -p "$cache"

exec docker run --rm --name square-qwen --gpus all --ipc=host \
  -p "127.0.0.1:${port}:8000" \
  -v "$cache:/root/.cache/huggingface" \
  "$image" "$model" \
  --revision "$revision" --tokenizer-revision "$revision" \
  --served-model-name square-qwen --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --language-model-only --tensor-parallel-size 1 \
  --max-model-len "$context" --max-num-seqs "$sequences" \
  --max-num-batched-tokens "$batch_tokens" --gpu-memory-utilization "$memory" \
  --kv-cache-dtype auto --enable-chunked-prefill --enable-prefix-caching \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --generation-config vllm
