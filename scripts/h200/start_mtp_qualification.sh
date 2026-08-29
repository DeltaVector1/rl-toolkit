#!/usr/bin/env bash
set -euo pipefail

# This launcher is for the isolated MTP qualification container only.  It has
# no production profile override and refuses an existing container name.
deployment=$(realpath -e "${prime_deployment:-/data/glm52-rl/deployment-glm52-optims}")
if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi
inference_config="$deployment/configs/qualification/glm52-mtp-h200.toml"
if [[ ! -f "$inference_config" ]]; then
  echo "refusing missing MTP qualification profile: $inference_config" >&2
  exit 2
fi
image=${prime_image:-prime-rl:glm52-v080-full}
model_volume=${prime_model_volume:-glm52-fp8-model}
container_name=${prime_mtp_container:-glm52-optims-mtp-h200}
overlay_root="$deployment/cache/vllm/flashmla512-selected-v4"
overlay_preflight="$deployment/scripts/h200/flashmla_overlay_preflight.py"
if [[ ! -f "$overlay_preflight" ]]; then
  echo "refusing missing FlashMLA overlay preflight: $overlay_preflight" >&2
  exit 2
fi
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "refusing to replace existing MTP qualification container: $container_name" >&2
  exit 2
fi
model_mount=$(docker volume inspect --format '{{.Mountpoint}}' "$model_volume" 2>/dev/null || true)
if [[ -z "$model_mount" || ! -f "$model_mount/config.json" ]]; then
  echo "refusing MTP start without the checkpoint config: $model_volume" >&2
  exit 2
fi
python3 "$deployment/scripts/glm52_h200_mtp_preflight.py" \
  --config "$inference_config" --model-config "$model_mount/config.json"
if ! python3 "$overlay_preflight" verify \
  --deployment "$deployment" --overlay "$overlay_root" --image "$image"; then
  # Refresh stale or pre-schema overlays through the pinned-image preparation
  # path.  The strict verification below remains mandatory after refresh.
  python3 "$overlay_preflight" prepare \
    --deployment "$deployment" --overlay "$overlay_root" --image "$image"
fi
python3 "$overlay_preflight" verify \
  --deployment "$deployment" --overlay "$overlay_root" --image "$image"

trace_dir=${prime_mtp_trace_dir:-}
if [[ -z "$trace_dir" || "${trace_dir:0:1}" != "/" || -L "$trace_dir" || ! -d "$trace_dir" || \
      "$(realpath -e "$trace_dir")" != "$trace_dir" ]]; then
  echo "refusing MTP start without an existing non-symlink route trace directory" >&2
  exit 2
fi
if [[ "$(id -u):$(id -g)" != "0:0" ]]; then
  echo "refusing MTP qualification launch by a non-root user" >&2
  exit 2
fi

umask 077
container_id=$(docker run -d \
  --name "$container_name" --restart=no \
  --init \
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
  --mount "type=bind,source=$overlay_root/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/layers/attention/attention.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/layers/attention/mla_attention.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/models/extract_hidden_states.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/models/extract_hidden_states.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/models/deepseek_v32/nvidia/attention.py,target=/app/.venv/lib/python3.12/site-packages/vllm/models/deepseek_v32/nvidia/attention.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/worker/gpu_model_runner.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/core/sched/output.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/output.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/v1/core/sched/scheduler.py,target=/app/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py,readonly" \
  -v "$deployment/source/src/prime_rl/glm52_routing.py":/app/.venv/lib/python3.12/site-packages/prime_rl/glm52_routing.py:ro \
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
  -e PRIME_MTP_QUALIFICATION=1 \
  -e PRIME_PARITY_ENGINE=1 \
  -e PRIME_PARITY_ROUTER_TRACE_DIR="$trace_dir" \
  -e PRIME_PARITY_QUALIFICATION_ROOT="$(dirname "$(dirname "$trace_dir")")" \
  --entrypoint /app/.venv/bin/inference \
  "$image" @ "$inference_config")
if [[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]]; then
  echo "refusing invalid MTP qualification container ID: $container_id" >&2
  exit 1
fi
printf 'mtp-qualification-container:%s\n' "$container_id"
