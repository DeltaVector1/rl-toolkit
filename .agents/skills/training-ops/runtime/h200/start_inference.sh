#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/lora-cache
install -d -o 1000 -g 1000 -m 0755 "$deployment/cache/vllm"
docker update --restart=no glm52-decensor-h200 glm52-decensor-h200-0 glm52-decensor-h200-1 >/dev/null 2>&1 || true
docker rm -f glm52-decensor-h200 glm52-decensor-h200-0 glm52-decensor-h200-1 >/dev/null 2>&1 || true
docker update --restart=no glm52-decensor-trainer >/dev/null 2>&1 || true
docker rm -f glm52-decensor-trainer >/dev/null 2>&1 || true

# Bound mixed FlashMLA batches below its signed-stride limit before MTP starts.
docker run -d --name glm52-decensor-h200 --restart unless-stopped \
  --user root \
  --log-opt max-size=512m --log-opt max-file=3 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit stack=67108864:67108864 \
  -v /data:/data \
  -v glm52-fp8-model:/models/glm52-fp8:ro \
  -v "$deployment/cache":/root/.cache \
  -v "$deployment/prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py":/app/src/prime_rl/inference/vllm/worker/filesystem.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/inference/vllm/server.py":/app/src/prime_rl/inference/vllm/server.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
  -v "$deployment/prime-rl/deps/vllm/flashmla_sparse_mtp.patch":/tmp/flashmla_sparse_mtp.patch:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  --entrypoint /bin/bash \
  prime-rl:glm52-v080-glm52-h200-router \
  -lc 'set -e; patch --batch --forward -p1 -d /app/.venv/lib/python3.12/site-packages < /tmp/flashmla_sparse_mtp.patch || grep -q fp8_mixed_batch_chunk_size /app/.venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py; exec /app/.venv/bin/inference "$@"' bash \
  @ "$deployment/configs/runtime/glm52-inference.toml" \
  --vllm-extra '{"gpu_memory_utilization":0.93,"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":65536,"compilation_config":{"max_cudagraph_capture_size":256},"speculative_config":{"method":"mtp","num_speculative_tokens":1}}'
