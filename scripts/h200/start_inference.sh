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
adapter_ports=${prime_adapter_ports:-18200,18201,18202,18203,18204,18205,18206,18207}
adapter_codec=${prime_adapter_codec:-gzip}
model_mount=$(docker volume inspect --format '{{.Mountpoint}}' "$model_volume" 2>/dev/null || true)
if [[ -z "$model_mount" || ! -f "$model_mount/config.json" ]]; then
  echo "refusing H200 start without the checkpoint config: $model_volume" >&2
  exit 2
fi
python3 "$deployment/scripts/glm52_h200_startup_preflight.py" \
  --config "$inference_config" \
  --model-config "$model_mount/config.json"
case "$adapter_codec" in
  gzip|zstd|identity) ;;
  *)
    echo "refusing invalid GLM adapter codec: $adapter_codec" >&2
    exit 2
    ;;
esac
engine_container=${prime_inference_container:-glm52-optims-h200}
require_absent=${prime_require_absent:-0}
if [[ "$require_absent" != "0" && "$require_absent" != "1" ]]; then
  echo "refusing invalid existing-container policy" >&2
  exit 2
fi
if [[ "$(id -u):$(id -g)" != "0:0" ]]; then
  echo "refusing parity engine launch by a non-root user" >&2
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
expected_vllm_extra='{"host":"127.0.0.1","kv_cache_dtype":"auto","max_num_seqs":128,"max_num_batched_tokens":32769,"compilation_config":{"max_cudagraph_capture_size":256}}'
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
read_engine_cid() {
  python3 - "$work_parent" "$(basename "$engine_cidfile")" <<'PY'
import os,re,stat,sys
parent_path,name=sys.argv[1:]
parent=os.open(parent_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
 st=os.fstat(parent)
 assert stat.S_ISDIR(st.st_mode) and st.st_uid==1000 and st.st_gid==1000
 assert stat.S_IMODE(st.st_mode)==0o750
 fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
 try:
  st=os.fstat(fd)
  assert stat.S_ISREG(st.st_mode) and st.st_uid==0 and st.st_gid==0
  assert stat.S_IMODE(st.st_mode)==0o600
  data=os.read(fd,65)
  assert os.read(fd,1)==b""
 finally:
  os.close(fd)
 assert re.fullmatch(rb"[0-9a-f]{64}",data)
 print(data.decode())
finally:
 os.close(parent)
PY
}
write_engine_identity() {
  python3 - "$work_parent" "$(basename "$engine_cidfile")" \
    "$(basename "$engine_identity_file")" "$container_id" "$host_boot_id" <<'PY'
import json,os,re,stat,sys
parent_path,cid_name,identity_name,expected_cid,boot=sys.argv[1:]
assert re.fullmatch(r"[0-9a-f]{64}",expected_cid)
assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",boot)
parent=os.open(parent_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
tmp=identity_name+".tmp"
try:
 st=os.fstat(parent)
 assert stat.S_ISDIR(st.st_mode) and st.st_uid==1000 and st.st_gid==1000
 assert stat.S_IMODE(st.st_mode)==0o750
 cid_fd=os.open(cid_name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
 try:
  st=os.fstat(cid_fd)
  assert stat.S_ISREG(st.st_mode) and st.st_uid==0 and st.st_gid==0
  assert stat.S_IMODE(st.st_mode)==0o600
  cid=os.read(cid_fd,65)
  assert os.read(cid_fd,1)==b""
 finally:
  os.close(cid_fd)
 assert cid==expected_cid.encode()
 payload=(json.dumps({"container_id":expected_cid,"host_boot_id":boot},sort_keys=True,separators=(",",":"))+"\n").encode()
 fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
 try:
  view=memoryview(payload)
  while view:
   written=os.write(fd,view)
   assert written>0
   view=view[written:]
  os.fsync(fd)
  st=os.fstat(fd)
  assert stat.S_ISREG(st.st_mode) and st.st_uid==0 and st.st_gid==0
  assert stat.S_IMODE(st.st_mode)==0o600
 finally:
  os.close(fd)
 os.replace(tmp,identity_name,src_dir_fd=parent,dst_dir_fd=parent)
 os.fsync(parent)
 final_fd=os.open(identity_name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent)
 try:
  st=os.fstat(final_fd)
  assert stat.S_ISREG(st.st_mode) and st.st_uid==0 and st.st_gid==0
  assert stat.S_IMODE(st.st_mode)==0o600
  actual=os.read(final_fd,len(payload)+1)
  assert os.read(final_fd,1)==b"" and actual==payload
 finally:
  os.close(final_fd)
finally:
 os.close(parent)
PY
}
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
cleanup_unreported_container() {
  status=$?
  trap - EXIT
  if ! docker rm -f "$container_id" >/dev/null 2>&1; then
    echo "failed to remove unreported isolated container: $container_id" >&2
    exit 1
  fi
  exit "$status"
}
umask 077
container_id=$(docker run -d --cidfile "$engine_cidfile" \
  --name "$engine_container" --restart "$restart_policy" \
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
  --mount "type=bind,source=$overlay_root/vllm/model_executor/layers/attention/attention.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/layers/attention/mla_attention.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/mla_attention.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/model_executor/models/extract_hidden_states.py,target=/app/.venv/lib/python3.12/site-packages/vllm/model_executor/models/extract_hidden_states.py,readonly" \
  --mount "type=bind,source=$overlay_root/vllm/models/deepseek_v32/nvidia/attention.py,target=/app/.venv/lib/python3.12/site-packages/vllm/models/deepseek_v32/nvidia/attention.py,readonly" \
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
  -e PRIME_GLM_ADAPTER_CODEC="$adapter_codec" \
  -e PRIME_GLM_ADAPTER_GZIP=1 \
  -e PRIME_LORA_BASE_MODEL_REPO_ID="${PRIME_LORA_BASE_MODEL_REPO_ID:-}" \
  -e PRIME_LORA_BASE_MODEL_REVISION="${PRIME_LORA_BASE_MODEL_REVISION:-}" \
  -e PRIME_LORA_BASE_MODEL_VARIANT="${PRIME_LORA_BASE_MODEL_VARIANT:-}" \
  -e PRIME_LORA_BASE_MODEL_CONTENT_SHA256="${PRIME_LORA_BASE_MODEL_CONTENT_SHA256:-}" \
  -e PRIME_LORA_BASE_MODEL_MANIFEST_SHA256="${PRIME_LORA_BASE_MODEL_MANIFEST_SHA256:-}" \
  -e PRIME_LORA_BASE_MODEL_TRAINER_PATH="${PRIME_LORA_BASE_MODEL_TRAINER_PATH:-}" \
  -e PRIME_LORA_BASE_MODEL_H200_ALIAS="${PRIME_LORA_BASE_MODEL_H200_ALIAS:-}" \
  -e PRIME_GLM_ADAPTER_VALIDATE="${PRIME_GLM_ADAPTER_VALIDATE:-0}" \
  -e PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap \
  -e PRIME_PARITY_ROUTER_TRACE_DIR="$parity_trace_dir" \
  --entrypoint /app/.venv/bin/inference \
  "$image" \
  @ "$inference_config" \
  --vllm-extra "$vllm_extra")
if [[ "$container_id" =~ ^[0-9a-f]{64}$ ]]; then
  trap cleanup_unreported_container EXIT
fi
cid_from_file=$(read_engine_cid)
if [[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]]; then
  container_id=$cid_from_file
  trap cleanup_unreported_container EXIT
fi
if [[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]]; then
  echo "refusing invalid isolated container ID and CID file" >&2
  exit 1
fi
if [[ "$cid_from_file" != "$container_id" ]]; then
  echo "refusing isolated container CID file mismatch" >&2
  exit 1
fi
write_engine_identity
printf 'parity-engine-identity:%s:%s\n' "$container_id" "$host_boot_id"
expected_command=$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:],separators=(",",":")))' \
  @ "$inference_config" --vllm-extra "$vllm_extra")
mapfile -t identity < <(docker inspect "$container_id" --format \
  '{{.Id}}
{{.Name}}
{{.Config.Image}}
{{json .Config.Cmd}}
{{.RestartCount}}
{{.HostConfig.RestartPolicy.Name}}
{{json .HostConfig.Init}}')
if [[ "${#identity[@]}" -ne 7 || \
      "${identity[0]}" != "$container_id" || \
      "${identity[1]}" != "/$engine_container" || \
      "${identity[2]}" != "$image" || \
      "${identity[3]}" != "$expected_command" || \
      "${identity[4]}" != "0" || \
      "${identity[5]}" != "$restart_policy" || \
      "${identity[6]}" != "true" ]]; then
  echo "refusing isolated container identity mismatch" >&2
  exit 1
fi
trap - EXIT
