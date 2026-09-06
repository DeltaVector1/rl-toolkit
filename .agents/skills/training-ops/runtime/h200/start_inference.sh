#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
engine_host=${ENGINE_HOST:?ENGINE_HOST must name this host in operations.toml}
runtime_output=$(python3 - "$deployment/configs/operations.toml" "$engine_host" <<'PY'
import sys
import tomllib
import json

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
host = sys.argv[2]
inference = config["inference"]
if host not in inference:
    raise SystemExit(f"unknown inference host: {host}")
ports = config["ports"]
first = ports["lora_source_first"]
last = ports["lora_source_last"]
if not isinstance(first, int) or not isinstance(last, int) or first < 1 or last < first:
    raise SystemExit("invalid lora source port range")
print(",".join(str(port) for port in range(first, last + 1)))
cache_dir = config["paths"].get("adapter_cache")
if not isinstance(cache_dir, str) or not cache_dir.startswith("/"):
    raise SystemExit("paths.adapter_cache is missing")
print(cache_dir)
disable_nccl_shm = inference[host].get("disable_nccl_shm", False)
if not isinstance(disable_nccl_shm, bool):
    raise SystemExit(f"disable_nccl_shm for {host} must be true or false")
print("1" if disable_nccl_shm else "0")
glm = inference[host]["glm52"]
tp = glm.get("tensor_parallel_size")
dp = glm.get("data_parallel_size", 1)
gpu_count = glm.get("gpu_count")
if not all(isinstance(value, int) and value > 0 for value in (tp, dp, gpu_count)):
    raise SystemExit(f"invalid GLM parallel sizes for {host}")
if tp * dp != gpu_count:
    raise SystemExit(f"GLM TP times DP must equal gpu_count for {host}")
mtp_tokens = glm.get("mtp_tokens")
if type(mtp_tokens) is not int or mtp_tokens < 0:
    raise SystemExit(f"mtp_tokens for {host} must be a non-negative integer")
vllm_extra = {
    "tensor_parallel_size": tp,
    "data_parallel_size": dp,
    "gpu_memory_utilization": 0.920,
    "kv_cache_dtype": "fp8",
    "max_num_seqs": 128,
    "max_num_batched_tokens": 32768,
    "compilation_config": {"max_cudagraph_capture_size": 128},
}
if mtp_tokens:
    vllm_extra["speculative_config"] = {
        "method": "mtp",
        "num_speculative_tokens": mtp_tokens,
    }
print(json.dumps(vllm_extra, separators=(",", ":")))
PY
)
readarray -t runtime_settings <<< "$runtime_output"
lora_source_port_list=${runtime_settings[0]:?operations config has no LoRA source ports}
adapter_cache_dir=${runtime_settings[1]:?operations config has no adapter cache path}
disable_nccl_shm=${runtime_settings[2]:?operations config has no NCCL shared-memory setting}
vllm_extra=${runtime_settings[3]:?operations config has no vLLM extra settings}
install -d -o 1000 -g 1000 -m 0755 "$adapter_cache_dir"
install -d -o 1000 -g 1000 -m 0755 "$deployment/cache/vllm"
docker update --restart=no glm52-decensor-h200 glm52-decensor-h200-0 glm52-decensor-h200-1 >/dev/null 2>&1 || true
docker rm -f glm52-decensor-h200 glm52-decensor-h200-0 glm52-decensor-h200-1 >/dev/null 2>&1 || true
docker update --restart=no glm52-decensor-trainer >/dev/null 2>&1 || true
docker rm -f glm52-decensor-trainer >/dev/null 2>&1 || true

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
  -v "$deployment/prime-rl/src/prime_rl/utils/adapter_delta.py":/app/src/prime_rl/utils/adapter_delta.py:ro \
  -v "$deployment/prime-rl/deps/vllm/flashmla_sparse_mtp.patch":/tmp/flashmla_sparse_mtp.patch:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SHM_DISABLE="$disable_nccl_shm" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e PRIME_LORA_CACHE_DIR="$adapter_cache_dir" \
  -e PRIME_LORA_DOWNLOAD_STREAMS=96 \
  -e PRIME_LORA_SOURCE_PORTS="$lora_source_port_list" \
  --entrypoint /bin/bash \
  prime-rl:glm52-v080-glm52-h200-router \
  -lc 'set -e; patch --batch --forward -p1 -d /app/.venv/lib/python3.12/site-packages < /tmp/flashmla_sparse_mtp.patch || grep -q "MIN_HEADS_FOR_BF16_PREFILL = 0" /app/.venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py; exec /app/.venv/bin/inference "$@"' bash \
  @ "$deployment/configs/runtime/glm52-inference.toml" \
  --vllm-extra "$vllm_extra"
