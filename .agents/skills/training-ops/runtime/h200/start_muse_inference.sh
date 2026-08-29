#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
runtime=/data/muse-glimmer/runtime
install -d -m 0755 "$runtime/prime_rl/inference/vllm/worker"
install -d -m 0755 "$runtime/prime_rl/utils"
install -d -m 0755 "$runtime/muse_file_server-0.1.dist-info"
install -d -m 0755 /data/muse-glimmer/weight-cache
install -d -m 0755 /data/muse-glimmer/lora-cache
install -m 0644 "$deployment/.agents/skills/training-ops/runtime/h200/muse_file_endpoint.py" "$runtime/muse_file_endpoint.py"
install -m 0644 "$deployment/.agents/skills/training-ops/runtime/h200/muse_file_server.entry_points.txt" \
  "$runtime/muse_file_server-0.1.dist-info/entry_points.txt"
install -m 0644 "$deployment/.agents/skills/training-ops/runtime/h200/muse_file_server.METADATA" \
  "$runtime/muse_file_server-0.1.dist-info/METADATA"
install -m 0644 "$deployment/.agents/skills/training-ops/runtime/h200/muse_transformers_sitecustomize.py" \
  "$runtime/sitecustomize.py"
install -m 0644 "$deployment/.agents/skills/training-ops/runtime/h200/muse_replica_proxy.py" \
  "$runtime/muse_replica_proxy.py"
weight_tunnel_ports=$(seq -s, 19001 19128)
vllm_image=${MUSE_VLLM_IMAGE:-vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d}
model_impl=${MUSE_MODEL_IMPL:-transformers}
tp_size=${MUSE_TP_SIZE:-1}
gpu_count=${MUSE_GPU_COUNT:-8}
max_model_len=${MUSE_MAX_MODEL_LEN:-65536}
max_num_seqs=${MUSE_MAX_NUM_SEQS:-32}
max_num_batched_tokens=${MUSE_MAX_NUM_BATCHED_TOKENS:-16384}
cudagraph_mode=${MUSE_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
case "$model_impl" in
  auto|vllm|transformers) ;;
  *)
    echo "MUSE_MODEL_IMPL must be auto, vllm, or transformers" >&2
    exit 2
    ;;
esac
case "$tp_size" in
  1|2|4|8) ;;
  *) echo "MUSE_TP_SIZE must be 1, 2, 4, or 8" >&2; exit 2 ;;
esac
case "$gpu_count" in
  ''|*[!0-9]*) echo "MUSE_GPU_COUNT must be a positive integer" >&2; exit 2 ;;
esac
if [ "$gpu_count" -lt 1 ]; then
  echo "MUSE_GPU_COUNT must be a positive integer" >&2
  exit 2
fi
case "$cudagraph_mode" in
  NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
  *)
    echo "MUSE_CUDAGRAPH_MODE must be NONE, PIECEWISE, FULL, FULL_DECODE_ONLY, or FULL_AND_PIECEWISE" >&2
    exit 2
    ;;
esac
if [ $((gpu_count % tp_size)) -ne 0 ]; then
  echo "MUSE_TP_SIZE must divide MUSE_GPU_COUNT" >&2
  exit 2
fi
replica_count=$((gpu_count / tp_size))

start_replica() {
  local replica=$1 devices=$2 port=$3
  local container="muse-glimmer-h200-$replica"
  docker update --restart=no "$container" >/dev/null 2>&1 || true
  docker rm -f "$container" >/dev/null 2>&1 || true
  docker run -d --name "$container" --restart unless-stopped \
  --log-opt max-size=512m --log-opt max-file=3 \
  --gpus "\"device=$devices\"" \
  --network host \
  --ipc host \
  --ulimit stack=67108864:67108864 \
  -v /data:/data \
  -v "$deployment/prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py":"$runtime/prime_rl/inference/vllm/worker/filesystem.py":ro \
  -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":"$runtime/prime_rl/utils/checkpoint_delta.py":ro \
  -e PYTHONPATH="$runtime" \
  -e MUSE_TP_SIZE="$tp_size" \
  -e VLLM_PLUGINS=muse_file_server \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PRIME_FULL_CHECKPOINT_CACHE_DIR=/data/muse-glimmer/weight-cache \
  -e PRIME_LORA_CACHE_DIR=/data/muse-glimmer/lora-cache \
  -e PRIME_LORA_DOWNLOAD_STREAMS=8 \
  -e PRIME_FULL_CHECKPOINT_PORTS="$weight_tunnel_ports" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 \
  -e NCCL_SOCKET_IFNAME=eth0 \
  -e GLOO_SOCKET_IFNAME=eth0 \
  -e NCCL_IB_DISABLE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  --entrypoint vllm \
  "$vllm_image" \
  serve /data/muse-glimmer/model \
  --served-model-name /data/muse-glimmer/model \
  --host 0.0.0.0 \
  --port "$port" \
  --tensor-parallel-size "$tp_size" \
  --model-impl "$model_impl" \
  --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"$cudagraph_mode\"}" \
  --dtype bfloat16 \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs "$max_num_seqs" \
  --max-num-batched-tokens "$max_num_batched_tokens" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-prefix-caching \
  --enable-lora \
  --max-loras 1 \
  --max-cpu-loras 2 \
  --max-lora-rank 128 \
  --lora-dtype bfloat16 \
  --fully-sharded-loras \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --worker-extension-cls prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker
}

if [ -n "${MUSE_REPLICA_INDEX:-}" ]; then
  case "$MUSE_REPLICA_INDEX" in
    ''|*[!0-9]*) echo "MUSE_REPLICA_INDEX must be an integer" >&2; exit 2 ;;
  esac
  if [ "$MUSE_REPLICA_INDEX" -ge "$replica_count" ]; then
    echo "MUSE_REPLICA_INDEX must be less than $replica_count for TP$tp_size" >&2
    exit 2
  fi
  first_device=$((MUSE_REPLICA_INDEX * tp_size))
  last_device=$((first_device + tp_size - 1))
  devices=$(seq -s, "$first_device" "$last_device")
  start_replica "$MUSE_REPLICA_INDEX" "$devices" "${PORT:-$((8001 + MUSE_REPLICA_INDEX))}"
  exit
fi

old_containers=$(docker ps -a --format '{{.Names}}' | grep -E '^(glm52-decensor-h200|muse-glimmer-h200(-[0-9]+)?)$' || true)
if [ -n "$old_containers" ]; then
  docker update --restart=no $old_containers >/dev/null 2>&1 || true
  docker rm -f $old_containers >/dev/null 2>&1 || true
fi

replica_backends=""
for replica in $(seq 0 $((replica_count - 1))); do
  first_device=$((replica * tp_size))
  last_device=$((first_device + tp_size - 1))
  devices=$(seq -s, "$first_device" "$last_device")
  port=$((8001 + replica))
  start_replica "$replica" "$devices" "$port"
  replica_backends="${replica_backends}${replica_backends:+,}127.0.0.1:$port"
done

docker run -d --name muse-glimmer-h200 --restart unless-stopped \
  --log-opt max-size=128m --log-opt max-file=2 \
  --network host \
  -v "$runtime/muse_replica_proxy.py":"$runtime/muse_replica_proxy.py":ro \
  -e MUSE_PROXY_PORT=8000 \
  -e MUSE_REPLICA_BACKENDS="$replica_backends" \
  --entrypoint python3 \
  "$vllm_image" \
  "$runtime/muse_replica_proxy.py"
