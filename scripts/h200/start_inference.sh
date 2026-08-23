#!/usr/bin/env bash
set -euo pipefail

deployment=${prime_deployment:-/data/glm52-rl/deployment-glm52-optims}
if [[ $# -ne 1 ]]; then
  echo "usage: $0 /data/glm52-rl/deployment-glm52-optims/<config.toml>" >&2
  exit 2
fi
deployment=$(realpath -e "$deployment")
expected_inference_config="$deployment/configs/qualification/glm52-policy-parity-h200.toml"
inference_config=$(realpath -e "$1") || {
  echo "refusing missing inference config: $1" >&2
  exit 2
}
case "$inference_config" in
  "$deployment"/*) ;;
  *)
    echo "refusing config outside isolated deployment: $inference_config" >&2
    exit 2
    ;;
esac
if [[ "$inference_config" != "$expected_inference_config" ]]; then
  echo "refusing non-parity inference config: $inference_config (expected $expected_inference_config)" >&2
  exit 2
fi
image=${prime_image:-prime-rl:glm52-v080-full}
model_volume=${prime_model_volume:-glm52-fp8-model}
adapter_ports=${prime_adapter_ports:-18100}
engine_container=${prime_inference_container:-glm52-optims-h200}
require_absent=${prime_require_absent:-0}
if [[ "$require_absent" != "0" && "$require_absent" != "1" ]]; then
  echo "refusing invalid existing-container policy" >&2
  exit 2
fi
overlay_root="$deployment/cache/vllm/flashmla512-selected-v4"
overlay_preflight="$deployment/scripts/h200/flashmla_overlay_preflight.py"
if [[ ! -f "$overlay_preflight" ]]; then
  echo "refusing missing FlashMLA overlay preflight: $overlay_preflight" >&2
  exit 2
fi
python3 "$overlay_preflight" prepare --deployment "$deployment" --overlay "$overlay_root" --image "$image"
python3 "$overlay_preflight" verify --deployment "$deployment" --overlay "$overlay_root" --image "$image"
expected_vllm_extra='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":32769,"compilation_config":{"max_cudagraph_capture_size":256}}'
vllm_extra=${prime_vllm_extra:-$expected_vllm_extra}
if [[ "$vllm_extra" != "$expected_vllm_extra" ]]; then
  echo "refusing non-parity vllm-extra: $vllm_extra" >&2
  exit 2
fi
bootstrap_token=${PRIME_PARITY_BOOTSTRAP_TOKEN:-}
if [[ -z "$bootstrap_token" || ! "$bootstrap_token" =~ ^[A-Za-z0-9._~-]{32,256}$ ]]; then
  echo "refusing parity engine without a valid bootstrap token" >&2
  exit 2
fi
install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/lora-cache
install -d -o 1000 -g 1000 -m 0755 "$deployment/cache/vllm"
for stale_container in "$engine_container" "$engine_container-0" "$engine_container-1"; do
  if docker inspect "$stale_container" >/dev/null 2>&1; then
    if [[ "$require_absent" == "1" ]]; then
      echo "refusing to replace an existing isolated container: $stale_container" >&2
      exit 2
    fi
    docker update --restart=no "$stale_container" >/dev/null 2>&1 || {
      echo "refusing to remove stale container with failed restart-policy update: $stale_container" >&2
      exit 2
    }
    docker rm -f "$stale_container" >/dev/null 2>&1 || {
      echo "refusing to launch while stale container remains: $stale_container" >&2
      exit 2
    }
    if docker inspect "$stale_container" >/dev/null 2>&1; then
      echo "refusing to launch while stale container name remains: $stale_container" >&2
      exit 2
    fi
  fi
done

# Do not enable native MTP here with the pinned vLLM and FlashMLA. The MTP
# draft layer pads 8 query heads to 64. At 32,768 tokens its BF16 query has a
# 2,415,919,104-byte stride, which exceeds FlashMLA's int32 limit and
# terminates every tensor-parallel worker.
docker run -d --name "$engine_container" --restart unless-stopped \
  --log-opt max-size=512m --log-opt max-file=3 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit stack=67108864:67108864 \
  -v /data:/data \
  -v "$model_volume":/models/glm52-fp8:ro \
  -v "$deployment/cache/vllm":/home/appuser/.cache \
  --mount "type=bind,source=$overlay_root/vllm/envs.py,target=/app/.venv/lib/python3.12/site-packages/vllm/envs.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/attention/backends/mla/flashmla_sparse.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/layers/logits_processor.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/models/deepseek_v2.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/worker/gpu_model_runner.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/core/sched/output.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/output.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/core/sched/scheduler.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py,readonly" \
  -v "$deployment/source/src/prime_rl/inference/vllm/worker/filesystem.py":/app/src/prime_rl/inference/vllm/worker/filesystem.py:ro \
  -v "$deployment/source/src/prime_rl/inference/vllm/server.py":/app/src/prime_rl/inference/vllm/server.py:ro \
  -v "$deployment/source/src/prime_rl/parity_router_trace.py":/app/src/prime_rl/parity_router_trace.py:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e PRIME_ADAPTER_PORTS="$adapter_ports" \
  -e PRIME_PARITY_BOOTSTRAP_TOKEN="$bootstrap_token" \
  --entrypoint /app/.venv/bin/inference \
  "$image" \
  @ "$inference_config" \
  --vllm-extra "$vllm_extra"
