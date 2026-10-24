#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
engine_host=${ENGINE_HOST:?ENGINE_HOST must name this host in nodes.toml}
engine_kind=${ENGINE_KIND:-glm52}
inference_config=${inference_config:?set inference_config to the selected run config}
test -f "$inference_config" || {
  echo "inference config is missing: $inference_config" >&2
  exit 2
}
download_streams=$(python3 - "$deployment/configs/nodes.toml" <<'PY'
import sys
import tomllib
with open(sys.argv[1], "rb") as stream:
    print(tomllib.load(stream)["adapter_transport"]["download_streams"])
PY
) || exit 2
if [ -n "${PRIME_LORA_DOWNLOAD_STREAMS+x}" ] && [ "$PRIME_LORA_DOWNLOAD_STREAMS" != "$download_streams" ]; then
  echo "PRIME_LORA_DOWNLOAD_STREAMS differs from nodes.toml" >&2
  exit 2
fi
case "$download_streams" in
  ''|*[!0-9]*)
    echo "PRIME_LORA_DOWNLOAD_STREAMS must be an integer" >&2
    exit 2
    ;;
esac
if [ "${#download_streams}" -gt 3 ]; then
  echo "PRIME_LORA_DOWNLOAD_STREAMS must be between 1 and 128" >&2
  exit 2
fi
if [ "$download_streams" -lt 1 ] || [ "$download_streams" -gt 128 ]; then
  echo "PRIME_LORA_DOWNLOAD_STREAMS must be between 1 and 128" >&2
  exit 2
fi
runtime_output=$(python3 - "$deployment/configs/nodes.toml" "$engine_host" "$inference_config" <<'PY'
import sys
import tomllib
import json
import os

with open(sys.argv[1], "rb") as stream:
    config = tomllib.load(stream)
host = sys.argv[2]
inference_path = sys.argv[3]
with open(inference_path, "rb") as stream:
    selected_inference = tomllib.load(stream)
gpu_memory_utilization = selected_inference.get("gpu_memory_utilization")
if (
    isinstance(gpu_memory_utilization, bool)
    or not isinstance(gpu_memory_utilization, (int, float))
    or not 0 < gpu_memory_utilization < 1
):
    raise SystemExit(
        "gpu_memory_utilization must be a numeric value strictly between 0 and 1"
    )
inference = config["inference"]
if host not in inference:
    raise SystemExit(f"unknown inference host: {host}")
ports = config["ports"]
first = ports["lora_source_first"]
last = ports["lora_source_last"]
if not isinstance(first, int) or not isinstance(last, int) or first < 1 or last < first:
    raise SystemExit("invalid lora source port range")
# Keep the worker source order interleaved across the two forwards in each
# China supervisor.
lane_size = 2
print(",".join(
    str(port)
    for offset in range(lane_size)
    for port in range(first + offset, last + 1, lane_size)
))
cache_dir = config["paths"].get("adapter_cache")
if not isinstance(cache_dir, str) or not cache_dir.startswith("/"):
    raise SystemExit("paths.adapter_cache is missing")
print(cache_dir)
disable_nccl_shm = inference[host].get("disable_nccl_shm", False)
if not isinstance(disable_nccl_shm, bool):
    raise SystemExit(f"disable_nccl_shm for {host} must be true or false")
print("1" if disable_nccl_shm else "0")
kind = os.environ.get("ENGINE_KIND", "glm52")
profile = inference[host].get(kind)
if not isinstance(profile, dict):
    raise SystemExit(f"inference settings for {host} and {kind} are missing")
tp = profile.get("tensor_parallel_size")
dp = profile.get("data_parallel_size", 1)
gpu_count = profile.get("gpu_count")
if not all(isinstance(value, int) and value > 0 for value in (tp, dp, gpu_count)):
    raise SystemExit(f"invalid {kind} parallel sizes for {host}")
if tp * dp != gpu_count:
    raise SystemExit(f"{kind} TP times DP must equal gpu_count for {host}")
mtp_tokens = profile.get("mtp_tokens")
if type(mtp_tokens) is not int or mtp_tokens < 0:
    raise SystemExit(f"mtp_tokens for {host} must be a non-negative integer")
max_num_seqs = 128
max_num_batched_tokens = 32768
max_cudagraph_capture_size = max_num_seqs * (1 + mtp_tokens)
vllm_extra = {
    "tensor_parallel_size": tp,
    "data_parallel_size": dp,
    "gpu_memory_utilization": float(gpu_memory_utilization),
    # GLM keeps the existing FP8 cache. Trinity's H200 path uses BF16 KV
    # cache because its native FP8 cache path is not parity-qualified.
    "kv_cache_dtype": "bfloat16" if kind == "trinity" else "fp8",
    "max_num_seqs": max_num_seqs,
    "max_num_batched_tokens": max_num_batched_tokens,
    "compilation_config": {
        "max_cudagraph_capture_size": max_cudagraph_capture_size
    },
    "cudagraph_metrics": True,
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
lora_source_port_list=${runtime_settings[0]:?nodes config has no LoRA source ports}
adapter_cache_dir=${runtime_settings[1]:?nodes config has no adapter cache path}
disable_nccl_shm=${runtime_settings[2]:?nodes config has no NCCL shared-memory setting}
vllm_extra=${runtime_settings[3]:?nodes config has no vLLM extra settings}
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
  -v "$deployment/prime-rl/src/prime_rl/inference/vllm/adapter_stage_worker.py":/app/src/prime_rl/inference/vllm/adapter_stage_worker.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/adapter_delta.py":/app/src/prime_rl/utils/adapter_delta.py:ro \
  -v "$deployment/prime-rl/deps/vllm/flashmla_sparse_mtp.patch":/tmp/flashmla_sparse_mtp.patch:ro \
  -v "$deployment/prime-rl/deps/vllm/prompt_logprobs_rows.patch":/tmp/prompt_logprobs_rows.patch:ro \
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
  -e PRIME_LORA_DOWNLOAD_STREAMS="$download_streams" \
  -e PRIME_LORA_LANE_TIMEOUT_S=90 \
  -e PRIME_LORA_CHUNK_TIMEOUT_S=90 \
  -e PRIME_LORA_SOURCE_PORTS="$lora_source_port_list" \
  --entrypoint /bin/bash \
  prime-rl:glm52-v080-glm52-h200-router \
  -lc 'set -e; patch --batch --forward -p1 -d /app/.venv/lib/python3.12/site-packages < /tmp/flashmla_sparse_mtp.patch || grep -q "MIN_HEADS_FOR_BF16_PREFILL = 0" /app/.venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py; prompt_logprobs_target=/app/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py; if ! patch --batch --forward --fuzz=0 -p1 -d /app/.venv/lib/python3.12/site-packages < /tmp/prompt_logprobs_rows.patch; then test "$(grep -Fxc "            # prime-rl prompt logprob row chunking: vllm-0.26" "$prompt_logprobs_target")" -eq 1; fi; exec /app/.venv/bin/inference "$@"' bash \
  @ "$inference_config" \
  --vllm-extra "$vllm_extra"
