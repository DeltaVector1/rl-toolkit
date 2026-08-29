#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/lora-cache
install -d -o 1000 -g 1000 -m 0755 "$deployment/cache/vllm"
docker update --restart=no glm52-decensor-h200 glm52-decensor-h200-0 glm52-decensor-h200-1 >/dev/null 2>&1 || true
docker rm -f glm52-decensor-h200 glm52-decensor-h200-0 glm52-decensor-h200-1 >/dev/null 2>&1 || true
docker update --restart=no glm52-decensor-trainer >/dev/null 2>&1 || true
docker rm -f glm52-decensor-trainer >/dev/null 2>&1 || true

# Do not enable native MTP here with the pinned vLLM and FlashMLA. The MTP
# draft layer pads 8 query heads to 64. At 32,768 tokens its BF16 query has a
# 2,415,919,104-byte stride, which exceeds FlashMLA's int32 limit and
# terminates every tensor-parallel worker.
docker run -d --name glm52-decensor-h200 --restart unless-stopped \
  --log-opt max-size=512m --log-opt max-file=3 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit stack=67108864:67108864 \
  -v /data:/data \
  -v "$deployment/cache/vllm":/home/appuser/.cache \
  -v "$deployment/prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py":/app/src/prime_rl/inference/vllm/worker/filesystem.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/inference/vllm/server.py":/app/src/prime_rl/inference/vllm/server.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  --entrypoint /app/.venv/bin/inference \
  prime-rl:glm52-decensor \
  @ "$deployment/configs/runtime/glm52-inference.toml" \
  --vllm-extra '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":65536,"compilation_config":{"max_cudagraph_capture_size":256}}'
