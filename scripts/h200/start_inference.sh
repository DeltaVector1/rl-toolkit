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
restart_policy=unless-stopped
if [[ "$require_absent" == "1" ]]; then
  restart_policy=no
fi
overlay_root="$deployment/cache/vllm/flashmla512-selected-v4"
overlay_preflight="$deployment/scripts/h200/flashmla_overlay_preflight.py"
if [[ ! -f "$overlay_preflight" ]]; then
  echo "refusing missing FlashMLA overlay preflight: $overlay_preflight" >&2
  exit 2
fi
python3 "$overlay_preflight" prepare --deployment "$deployment" --overlay "$overlay_root" --image "$image"
python3 "$overlay_preflight" verify --deployment "$deployment" --overlay "$overlay_root" --image "$image"
expected_vllm_extra='{"host":"127.0.0.1","kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":32769,"compilation_config":{"max_cudagraph_capture_size":256}}'
vllm_extra=${prime_vllm_extra:-$expected_vllm_extra}
if [[ "$vllm_extra" != "$expected_vllm_extra" ]]; then
  echo "refusing non-parity vllm-extra: $vllm_extra" >&2
  exit 2
fi
bootstrap_fifo=${prime_parity_bootstrap_fifo:-}
qualification_root=/data/glm52-rl/qualifications/glm52-policy-parity
if [[ ! "$bootstrap_fifo" =~ ^$qualification_root/\.diagnostic-run-[0-9a-f]{16}/bootstrap\.fifo$ ]]; then
  echo "refusing parity engine without an exact bootstrap FIFO" >&2
  exit 2
fi
fifo_parent=$(dirname "$bootstrap_fifo")
if [[ -L "$fifo_parent" || "$(realpath -e "$fifo_parent")" != "$fifo_parent" || \
      -L "$bootstrap_fifo" || ! -p "$bootstrap_fifo" ]]; then
  echo "refusing invalid parity bootstrap FIFO" >&2
  exit 2
fi
fifo_identity=$(stat -c '%u:%g:%a' "$bootstrap_fifo")
if [[ "$fifo_identity" != "1000:1000:600" ]]; then
  echo "refusing parity bootstrap FIFO owner or mode: $fifo_identity" >&2
  exit 2
fi
parity_trace_dir=${prime_parity_trace_dir:-}
if [[ ! "$parity_trace_dir" =~ ^$qualification_root/\.diagnostic-work-[0-9a-f]{16}/h200-router-trace$ ]]; then
  echo "refusing parity engine without an exact router trace directory" >&2
  exit 2
fi
if [[ -L "$parity_trace_dir" || ! -d "$parity_trace_dir" || \
      "$(realpath -e "$parity_trace_dir")" != "$parity_trace_dir" ]]; then
  echo "refusing invalid parity router trace directory" >&2
  exit 2
fi
trace_identity=$(stat -c '%u:%g:%a' "$parity_trace_dir")
if [[ "$trace_identity" != "1000:1000:750" || \
      -n "$(find "$parity_trace_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing parity router trace owner, mode, or content: $trace_identity" >&2
  exit 2
fi
engine_cidfile=${prime_parity_engine_cidfile:-}
engine_identity_file=${prime_parity_engine_identity_file:-}
work_parent=$(dirname "$parity_trace_dir")
if [[ "$engine_cidfile" != "$work_parent/engine.cid" || \
      "$engine_identity_file" != "$work_parent/engine-identity.json" || \
      ! "$work_parent" =~ ^$qualification_root/\.diagnostic-work-[0-9a-f]{16}$ ]]; then
  echo "refusing parity engine without exact identity paths" >&2
  exit 2
fi
for path in "$engine_cidfile" "$engine_identity_file" "$engine_identity_file.tmp"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "refusing existing parity engine identity path: $path" >&2
    exit 2
  fi
done
host_boot_id=$(cat /proc/sys/kernel/random/boot_id)
if [[ ! "$host_boot_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
  echo "refusing invalid host boot ID" >&2
  exit 1
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
container_id=$(docker run -d --cidfile "$engine_cidfile" \
  --name "$engine_container" --restart "$restart_policy" \
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
  --mount "type=bind,source=$bootstrap_fifo,target=/run/prime-parity-bootstrap" \
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
  -e PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap \
  -e PRIME_PARITY_ROUTER_TRACE_DIR="$parity_trace_dir" \
  --entrypoint /app/.venv/bin/inference \
  "$image" \
  @ "$inference_config" \
  --vllm-extra "$vllm_extra")
if [[ ! "$container_id" =~ ^[0-9a-f]{64}$ && \
      -f "$engine_cidfile" && ! -L "$engine_cidfile" ]]; then
  container_id=$(<"$engine_cidfile")
fi
if [[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]]; then
  echo "refusing invalid isolated container ID and CID file" >&2
  exit 1
fi
cleanup_unreported_container() {
  status=$?
  trap - EXIT
  if ! docker rm -f "$container_id" >/dev/null 2>&1; then
    echo "failed to remove unreported isolated container: $container_id" >&2
    exit 1
  fi
  exit "$status"
}
trap cleanup_unreported_container EXIT
if [[ ! -f "$engine_cidfile" || -L "$engine_cidfile" || \
      "$(realpath -e "$engine_cidfile")" != "$engine_cidfile" ]]; then
  echo "refusing invalid isolated container CID file" >&2
  exit 1
fi
chmod 0600 "$engine_cidfile"
if [[ "$(stat -c '%u:%g:%a' "$engine_cidfile")" != "1000:1000:600" || \
      "$(<"$engine_cidfile")" != "$container_id" || \
      "$(wc -l <"$engine_cidfile")" -ne 1 ]]; then
  echo "refusing isolated container CID file mismatch" >&2
  exit 1
fi
python3 -c 'import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); t=pathlib.Path(str(p)+".tmp")
t.write_text(json.dumps({"container_id":sys.argv[2],"host_boot_id":sys.argv[3]},sort_keys=True,separators=(",",":"))+"\n")
os.chmod(t,0o600); os.replace(t,p)' \
  "$engine_identity_file" "$container_id" "$host_boot_id"
if [[ ! -f "$engine_identity_file" || -L "$engine_identity_file" || \
      "$(realpath -e "$engine_identity_file")" != "$engine_identity_file" || \
      "$(stat -c '%u:%g:%a' "$engine_identity_file")" != "1000:1000:600" ]]; then
  echo "refusing invalid isolated container identity file" >&2
  exit 1
fi
printf 'parity-engine-identity:%s:%s\n' "$container_id" "$host_boot_id"
expected_command=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:],separators=(",",":")))' \
  @ "$inference_config" --vllm-extra "$vllm_extra")
mapfile -t identity < <(docker inspect "$container_id" --format \
  '{{.Id}}
{{.Name}}
{{.Config.Image}}
{{json .Config.Cmd}}
{{.RestartCount}}
{{.HostConfig.RestartPolicy.Name}}')
if [[ "${#identity[@]}" -ne 6 || \
      "${identity[0]}" != "$container_id" || \
      "${identity[1]}" != "/$engine_container" || \
      "${identity[2]}" != "$image" || \
      "${identity[3]}" != "$expected_command" || \
      "${identity[4]}" != "0" || \
      "${identity[5]}" != "$restart_policy" ]]; then
  echo "refusing isolated container identity mismatch" >&2
  exit 1
fi
trap - EXIT
