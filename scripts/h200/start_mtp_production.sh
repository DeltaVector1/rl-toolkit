#!/usr/bin/env bash
set -euo pipefail

# Dedicated GLM-5.2 MTP1 production inference on the isolated H200 host.
# It refuses an existing container; training_ops.start performs exact attach.
deployment=$(realpath -e "${prime_deployment:-/data/glm52-rl/deployment-glm52-optims}")
if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi
inference_config="${PRIME_MTP_PROFILE:-$deployment/configs/paused/inference.glm52-mtp20.toml}"
if [[ "$inference_config" != "$deployment/configs/paused/inference.glm52-mtp20.toml" ]]; then
  echo "refusing unexpected MTP production profile path: $inference_config" >&2
  exit 2
fi
if [[ ! -f "$inference_config" ]]; then
  echo "refusing missing MTP production profile: $inference_config" >&2
  exit 2
fi
image=${prime_image:-prime-rl:glm52-v080-full}
model_volume=${prime_model_volume:-glm52-fp8-model}
if [[ "$image" != prime-rl:glm52-v080-full || "$model_volume" != glm52-fp8-model ]]; then
  echo "refusing unexpected MTP production image or model volume" >&2
  exit 2
fi
container_name=${prime_mtp_container:-glm52-optims-mtp20-h200}
expected_image_digest=${PRIME_MTP_IMAGE_DIGEST:-sha256:2f803571329212bfb4e27982efec7559f20379bcde73ac6ccee6bb8be3c46ade}
if [[ "$expected_image_digest" != sha256:2f803571329212bfb4e27982efec7559f20379bcde73ac6ccee6bb8be3c46ade ]]; then
  echo "refusing unexpected MTP production image digest" >&2
  exit 2
fi
for identity_name in \
  PRIME_LORA_BASE_MODEL_REPO_ID \
  PRIME_LORA_BASE_MODEL_REVISION \
  PRIME_LORA_BASE_MODEL_VARIANT \
  PRIME_LORA_BASE_MODEL_CONTENT_SHA256 \
  PRIME_LORA_BASE_MODEL_MANIFEST_SHA256 \
  PRIME_LORA_BASE_MODEL_TRAINER_PATH \
  PRIME_LORA_BASE_MODEL_H200_ALIAS; do
  if [[ -z "${!identity_name:-}" ]]; then
    echo "refusing MTP production start without $identity_name" >&2
    exit 2
  fi
done
if [[ "${PRIME_GLM_ADAPTER_VALIDATE:-}" != 1 ]]; then
  echo "refusing MTP production start without PRIME_GLM_ADAPTER_VALIDATE=1" >&2
  exit 2
fi
if [[ ! "$expected_image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "refusing MTP production start without an exact image digest" >&2
  exit 2
fi
adapter_ports=${PRIME_ADAPTER_PORTS:-18200,18201,18202,18203,18204,18205,18206,18207}
if [[ "$adapter_ports" != 18200,18201,18202,18203,18204,18205,18206,18207 ]]; then
  echo "refusing MTP production start with unexpected adapter ports" >&2
  exit 2
fi
adapter_codec=${PRIME_GLM_ADAPTER_CODEC:-gzip}
if [[ "$adapter_codec" != gzip ]]; then
  echo "refusing MTP production start without gzip adapter transport" >&2
  exit 2
fi
lora_cache_dir=${PRIME_LORA_CACHE_DIR:-/data/glm52-rl/lora-cache}
if [[ "$lora_cache_dir" != /data/glm52-rl/lora-cache ]]; then
  echo "refusing MTP production start with unexpected LoRA cache directory" >&2
  exit 2
fi
overlay_root="$deployment/cache/vllm/flashmla512-selected-v4"
overlay_preflight="$deployment/scripts/h200/flashmla_overlay_preflight.py"
if [[ ! -f "$overlay_preflight" ]]; then
  echo "refusing missing FlashMLA overlay preflight: $overlay_preflight" >&2
  exit 2
fi
if docker inspect "$container_name" >/dev/null 2>&1; then
  echo "refusing to replace existing MTP production container: $container_name" >&2
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

if [[ "$(id -u):$(id -g)" != "0:0" ]]; then
  echo "refusing MTP production launch by a non-root user" >&2
  exit 2
fi

umask 077
container_id=$(docker run -d \
  --name "$container_name" --restart=unless-stopped \
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
  -v "$deployment/source/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py":/app/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:ro \
  -v "$deployment/source/packages/prime-rl-configs/src/prime_rl/configs/shared.py":/app/packages/prime-rl-configs/src/prime_rl/configs/shared.py:ro \
  -v "$deployment/source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py":/app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py:ro \
  -v "$deployment/source/src/prime_rl/_compat.py":/app/src/prime_rl/_compat.py:ro \
  -v "$deployment/source/src/prime_rl/glm52_routing.py":/app/.venv/lib/python3.12/site-packages/prime_rl/glm52_routing.py:ro \
  -v "$deployment/source/src/prime_rl/inference/vllm/adapter_validation.py":/app/src/prime_rl/inference/vllm/adapter_validation.py:ro \
  -v "$deployment/source/src/prime_rl/inference/vllm/worker/filesystem.py":/app/src/prime_rl/inference/vllm/worker/filesystem.py:ro \
  -v "$deployment/source/src/prime_rl/inference/vllm/server.py":/app/src/prime_rl/inference/vllm/server.py:ro \
  -v "$deployment/source/src/prime_rl/model_identity.py":/app/src/prime_rl/model_identity.py:ro \
  -v "$deployment/source/src/prime_rl/parity_router_trace.py":/app/src/prime_rl/parity_router_trace.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/distributed/expert_parallel.py":/app/src/prime_rl/trainer/distributed/expert_parallel.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/distributed/partial_reshard.py":/app/src/prime_rl/trainer/distributed/partial_reshard.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/distributed/pipeline_parallel.py":/app/src/prime_rl/trainer/distributed/pipeline_parallel.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/fp8_checkpoint.py":/app/src/prime_rl/trainer/fp8_checkpoint.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/lora.py":/app/src/prime_rl/trainer/lora.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/model.py":/app/src/prime_rl/trainer/model.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/kernels/fp8_indexer.py":/app/src/prime_rl/trainer/models/kernels/fp8_indexer.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/checkpointing.py":/app/src/prime_rl/trainer/models/layers/checkpointing.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/fp8_ada.py":/app/src/prime_rl/trainer/models/layers/fp8_ada.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/lora/multi_linear.py":/app/src/prime_rl/trainer/models/layers/lora/multi_linear.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py":/app/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/moe.py":/app/src/prime_rl/trainer/models/layers/moe.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/norms.py":/app/src/prime_rl/trainer/models/layers/norms.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/models/layers/rotary_emb.py":/app/src/prime_rl/trainer/models/layers/rotary_emb.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/mxfp4.py":/app/src/prime_rl/trainer/mxfp4.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/parallel_dims.py":/app/src/prime_rl/trainer/parallel_dims.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/runs.py":/app/src/prime_rl/trainer/runs.py:ro \
  -v "$deployment/source/src/prime_rl/trainer/weights.py":/app/src/prime_rl/trainer/weights.py:ro \
  -v "$deployment/source/src/prime_rl/utils/cp.py":/app/src/prime_rl/utils/cp.py:ro \
  -v "$deployment/source/src/prime_rl/utils/vlm.py":/app/src/prime_rl/utils/vlm.py:ro \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e PRIME_MTP_QUALIFICATION=0 \
  -e PRIME_PARITY_ENGINE=0 \
  -e PRIME_LORA_BASE_MODEL_REPO_ID="${PRIME_LORA_BASE_MODEL_REPO_ID}" \
  -e PRIME_LORA_BASE_MODEL_REVISION="${PRIME_LORA_BASE_MODEL_REVISION}" \
  -e PRIME_LORA_BASE_MODEL_VARIANT="${PRIME_LORA_BASE_MODEL_VARIANT}" \
  -e PRIME_LORA_BASE_MODEL_CONTENT_SHA256="${PRIME_LORA_BASE_MODEL_CONTENT_SHA256}" \
  -e PRIME_LORA_BASE_MODEL_MANIFEST_SHA256="${PRIME_LORA_BASE_MODEL_MANIFEST_SHA256}" \
  -e PRIME_LORA_BASE_MODEL_TRAINER_PATH="${PRIME_LORA_BASE_MODEL_TRAINER_PATH}" \
  -e PRIME_LORA_BASE_MODEL_H200_ALIAS="${PRIME_LORA_BASE_MODEL_H200_ALIAS}" \
  -e PRIME_GLM_ADAPTER_VALIDATE=1 \
  -e PRIME_LORA_CACHE_DIR="$lora_cache_dir" \
  -e PRIME_ADAPTER_PORTS="$adapter_ports" \
  -e PRIME_GLM_ADAPTER_CODEC="$adapter_codec" \
  -e PRIME_GLM_ADAPTER_GZIP=1 \
  --entrypoint /app/.venv/bin/inference \
  "$image" @ "$inference_config")
if [[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]]; then
  echo "refusing invalid MTP production container ID: $container_id" >&2
  exit 1
fi
actual_image_digest=$(docker inspect "$container_id" --format '{{.Image}}')
if [[ "$actual_image_digest" != "$expected_image_digest" ]]; then
  echo "refusing MTP production image digest mismatch" >&2
  docker rm -f "$container_id" >/dev/null 2>&1 || true
  exit 1
fi
printf 'mtp-production-container:%s\n' "$container_id"
