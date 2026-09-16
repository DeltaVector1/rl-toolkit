#!/usr/bin/env bash
set -euo pipefail

image=prime-rl:glm52-fp8
deployment=${deployment_root:?set deployment_root}
storage=${storage_root:?set storage_root}
runs=${runs_path:?set runs_path}
trainer_container=${trainer_container_name:?set trainer_container_name}
master_node=${master_node_id:?set master_node_id}
RUN=${RUN:?set RUN}
run_name=${RUN_NAME:-run_$RUN}
trainer_config=${TRAINER_CONFIG:-$deployment/configs/general-agent/trainer.toml}
run_dir=$runs/$RUN/$run_name
node=${node_id:-$(ip -4 -o addr show vlan130 | awk '{split($4, a, /[.\/]/); print a[4]}')}

common=(
  # Rotate container logs: multi-day debug-level output filled node roots.
  --log-opt max-size=512m --log-opt max-file=3
  --network host
  --ipc host
  --ulimit nofile=1048576:1048576
  --ulimit memlock=-1:-1
  --ulimit stack=67108864:67108864
  # A failed 64-rank trainer can otherwise leave one multi-gigabyte Apport
  # core per worker on the host root filesystem. These dumps filled n76 and
  # prevented NCCL from writing its much smaller diagnostic trace.
  --ulimit core=0:0
  -v /data:/data
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e PYTHONPATH="$deployment/environments/general-agent:/app/src"
  -e PYTHONUNBUFFERED=1
  -e OMP_NUM_THREADS=1
  -e PRIME_TRAINER_NODE="$node"
  -e NCCL_SOCKET_IFNAME=vlan130
  -e GLOO_SOCKET_IFNAME=vlan130
  -e NCCL_IB_DISABLE=1
  -e NCCL_P2P_DISABLE=1
  -e NCCL_SOCKET_NTHREADS=1
  -e NCCL_NSOCKS_PERTHREAD=1
  # Native FP8 leaves enough memory for these communication settings.
  -e NCCL_BUFFSIZE=1048576
  # Avoid large NCCL host buffers.
  -e NCCL_CUMEM_HOST_ENABLE=0
  # One channel hangs. Four channels are faster than two.
  -e NCCL_MIN_NCHANNELS=4
  -e NCCL_MAX_NCHANNELS=4
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
  -e TORCH_NCCL_DUMP_ON_TIMEOUT=1
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  -e TORCH_NCCL_ENABLE_MONITORING=1
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
)

# A newly released model can require a newer Transformers package than the
# pinned trainer image. Set TRANSFORMERS_OVERRIDE_DIR to a host directory that
# contains the complete `transformers` Python package. The default is empty and
# therefore does not change established GLM runs.
transformers_mount=()
if [ -n "${TRANSFORMERS_OVERRIDE_DIR:-}" ]; then
  test -d "$TRANSFORMERS_OVERRIDE_DIR/transformers" || {
    echo "TRANSFORMERS_OVERRIDE_DIR must contain transformers/" >&2
    exit 1
  }
  transformers_mount=(-v "$TRANSFORMERS_OVERRIDE_DIR/transformers":/app/.venv/lib/python3.12/site-packages/transformers:ro)
fi

case ${1:-} in
  trainer)
    # trainer_nodes lists the participating nodes in rank order; a node's rank
    # is its position in that list. Set it to drop a node from the job -- e.g.
    # with n81 down, trainer_nodes="82 76 77 78 80 83 84" gives a 7-node,
    # 56-rank job. NNODES follows automatically.
    trainer_nodes=${trainer_nodes:?set trainer_nodes}
    rank=-1
    idx=0
    for n in $trainer_nodes; do
      if [ "$n" = "$node" ]; then rank=$idx; fi
      idx=$((idx + 1))
    done
    nnodes=$idx
    if [ "$rank" -lt 0 ]; then
      echo "node $node is not in trainer_nodes ($trainer_nodes)" >&2; exit 1
    fi
    trainer_env_file=$(mktemp)
    chmod 600 "$trainer_env_file"
    trap 'rm -f -- "$trainer_env_file"' EXIT
    if ! read -r rollout_transport trainer_output fake_data < <(
      python3 - "$trainer_config" "$trainer_env_file" <<'PY'
import re
import sys
import tomllib

try:
    with open(sys.argv[1], "rb") as f:
        config = tomllib.load(f)
    env_vars = config.get("env_vars", {})
    if not isinstance(env_vars, dict):
        raise ValueError
    protected = {
        "CUDA_VISIBLE_DEVICES",
        "WANDB_SHARED_MODE",
        "WANDB_SHARED_RUN_ID",
        "WANDB_SHARED_LABEL",
    }
    env_lines = []
    for key, value in env_vars.items():
        if (
            not isinstance(key, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
            or key in protected
            or not isinstance(value, str)
            or any(char in value for char in "\x00\r\n")
        ):
            raise ValueError
        env_lines.append(f"{key}={value}\n")
    with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
        f.writelines(env_lines)
except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError):
    print("invalid trainer config or env_vars", file=sys.stderr)
    raise SystemExit(2)

print(
    config["rollout_transport"]["type"],
    config["output_dir"],
    int("fake" in config.get("data", {})),
)
PY
    ); then
      echo "cannot parse trainer config" >&2
      exit 2
    fi
    trainer_env_keys=()
    while IFS='=' read -r env_key _; do
      [ -n "$env_key" ] && trainer_env_keys+=("$env_key")
    done < "$trainer_env_file"
    has_trainer_env_override() {
      local candidate=$1 env_key
      for env_key in "${trainer_env_keys[@]}"; do
        [ "$env_key" = "$candidate" ] && return 0
      done
      return 1
    }
    sudo install -d -o 1000 -g 1000 -m 0755 "$trainer_output"
    if [ "$rank" -eq 0 ]; then
      if [ "$rollout_transport" = filesystem ] && [ "$fake_data" = 0 ]; then
        trainer_sync_targets=${trainer_sync_targets:?set trainer_sync_targets}
        read -r -a sync_targets <<<"$trainer_sync_targets"
        if [ "${#sync_targets[@]}" -ne "$nnodes" ]; then
          echo "trainer sync target count (${#sync_targets[@]}) differs from trainer node count ($nnodes)" >&2
          exit 1
        fi
        sync_nodes=()
        sync_index=0
        for n in $trainer_nodes; do
          if [ "$n" != "$node" ] && [ "$n" != "$master_node" ]; then
            sync_nodes+=("${sync_targets[$sync_index]}")
          fi
          sync_index=$((sync_index + 1))
        done
        if ((${#sync_nodes[@]})); then
          sync_log="$storage/microbatch-sync-$RUN.log"
          nohup setsid env \
            run_dir="$trainer_output" nodes="${sync_nodes[*]}" \
            "$deployment/training_ops/runtime/china/sync_microbatches.sh" \
            >>"$sync_log" 2>&1 </dev/null &
        fi
      fi
    fi
    # The default fq_codel qdisc costs about a quarter of the all-to-all
    # throughput on this link (65% of line rate against 87% with fq). Applied
    # here so it is reapplied after a reboot; both calls are idempotent.
    trainer_nic=$(ip -o link show vlan130 | sed -n 's/.*vlan130@\([^:]*\):.*/\1/p')
    sudo sysctl -qw net.core.default_qdisc=fq || true
    sudo tc qdisc replace dev "$trainer_nic" root fq || true
    # TileLang compiles kernels into $HOME/.tilelang inside the container, which
    # lands on the host root filesystem. Root is small and already near full on
    # some nodes (n76 hit 100% and the trainer died with ENOSPC), while /data has
    # hundreds of GB. Keep the cache on /data.
    sudo install -d -o 1000 -g 1000 -m 0755 "$storage/cache/torchinductor"
    sudo install -d -o 1000 -g 1000 -m 0755 "$storage/cache/tilelang"
    # Triton compiles kernels into ~/.triton and filled n76 root twice.
    sudo install -d -o 1000 -g 1000 -m 0755 "$storage/cache/triton"
    sudo install -d -o 1000 -g 1000 -m 1777 "$storage/cache/tmp"
    # Batch replay cache (PRIME_BATCH_CACHE), written by the container user.
    sudo install -d -o 1000 -g 1000 -m 0755 "$storage/cache/batch_pp"
    # nccl shm segments leak on every trainer crash (~33MB each, observed
    # 1500+ per node = 50GB+); the container was just removed, so purge.
    sudo rm -f /dev/shm/nccl-* 2>/dev/null || true
    sudo docker rm -f "$trainer_container" >/dev/null 2>&1 || true
    # Keep the optional checkpoint cache disabled. It is not selected by the
    # current dequantized, node-local EP path, but its single slot per layer is
    # not safe for multiple 1F1B micro-batches.
    trainer_docker_args=(
      --gpus all "${common[@]}" \
      -v "$storage/cache/torchinductor":/tmp/torchinductor_appuser \
      -v "$storage/cache/tilelang":/home/appuser/.tilelang \
      -v "$storage/cache/triton":/home/appuser/.triton \
      -v "$storage/cache/tmp":/scratch-tmp \
      "${transformers_mount[@]}" \
      -e TMPDIR=/scratch-tmp \
      -v "$deployment/prime-rl/src/prime_rl/_compat.py":/app/src/prime_rl/_compat.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/fp8_checkpoint.py":/app/src/prime_rl/trainer/fp8_checkpoint.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/model.py":/app/src/prime_rl/trainer/model.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/layers/fp8_ada.py":/app/src/prime_rl/trainer/models/layers/fp8_ada.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/layers/moe.py":/app/src/prime_rl/trainer/models/layers/moe.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/layers/checkpointing.py":/app/src/prime_rl/trainer/models/layers/checkpointing.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/layers/norms.py":/app/src/prime_rl/trainer/models/layers/norms.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/distributed/expert_parallel.py":/app/src/prime_rl/trainer/distributed/expert_parallel.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/distributed/partial_reshard.py":/app/src/prime_rl/trainer/distributed/partial_reshard.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/distributed/pipeline_parallel.py":/app/src/prime_rl/trainer/distributed/pipeline_parallel.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/distributed/pp_runner.py":/app/src/prime_rl/trainer/distributed/pp_runner.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/parallel_dims.py":/app/src/prime_rl/trainer/parallel_dims.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/utils/cp.py":/app/src/prime_rl/utils/cp.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/utils/adapter_delta.py":/app/src/prime_rl/utils/adapter_delta.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/rl/train.py":/app/src/prime_rl/trainer/rl/train.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/rl/data.py":/app/src/prime_rl/trainer/rl/data.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/runs.py":/app/src/prime_rl/trainer/runs.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/weights.py":/app/src/prime_rl/trainer/weights.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/utils.py":/app/src/prime_rl/trainer/utils.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/rl/broadcast/filesystem.py":/app/src/prime_rl/trainer/rl/broadcast/filesystem.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/layers/lora/multi_linear.py":/app/src/prime_rl/trainer/models/layers/lora/multi_linear.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py":/app/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/lora.py":/app/src/prime_rl/trainer/lora.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py:ro \
      -v "$deployment/prime-rl/src/prime_rl/utils/vlm.py":/app/src/prime_rl/utils/vlm.py:ro \
      -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/shared.py":/app/packages/prime-rl-configs/src/prime_rl/configs/shared.py:ro \
      -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/trainer.py":/app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py:ro \
      -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py":/app/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py:ro \
      -v "$deployment/prime-rl/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py":/app/packages/prime-rl-configs/src/prime_rl/configs/algorithm.py:ro \
      --env-file /data/trinity/training.env \
      -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}" \
      -e TORCHINDUCTOR_COMPILE_THREADS=8 \
      -e PRIME_EP_PROFILE="${PRIME_EP_PROFILE:-0}" \
      -e PRIME_EP_PROFILE_EVERY="${PRIME_EP_PROFILE_EVERY:-150}" \
      -e PRIME_EP_AC_CACHE="${PRIME_EP_AC_CACHE:-0}" \
      -e PRIME_INT4_COMBINE_ROUNDS="${PRIME_INT4_COMBINE_ROUNDS:-1}" \
      -e PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD="${PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD:-0}" \
      -e PRIME_BATCH_CACHE="${PRIME_BATCH_CACHE:-}" \
      -e PRIME_PP_MAX_MICRO="${PRIME_PP_MAX_MICRO:-}" \
      -e PRIME_STATS_DEBUG="${PRIME_STATS_DEBUG:-}" \
      -e PRIME_RL_RESUME_ADAPTER="${PRIME_RL_RESUME_ADAPTER:-}" \
      -e PRIME_RL_RESUME_ADAPTER_STEP="${PRIME_RL_RESUME_ADAPTER_STEP:-}" \
      --env-file "$trainer_env_file"
    )
    trainer_filtered_args=()
    arg_index=0
    while [ "$arg_index" -lt "${#trainer_docker_args[@]}" ]; do
      if [ "${trainer_docker_args[$arg_index]}" = "-e" ] &&
        [ "$((arg_index + 1))" -lt "${#trainer_docker_args[@]}" ]; then
        env_assignment=${trainer_docker_args[$((arg_index + 1))]}
        env_key=${env_assignment%%=*}
        if has_trainer_env_override "$env_key"; then
          arg_index=$((arg_index + 2))
          continue
        fi
      fi
      trainer_filtered_args+=("${trainer_docker_args[$arg_index]}")
      arg_index=$((arg_index + 1))
    done
    sudo docker run -d --name "$trainer_container" \
      "${trainer_filtered_args[@]}" \
      --entrypoint /app/.venv/bin/torchrun "$image" \
      --role=trainer \
      --nnodes="$nnodes" \
      --nproc-per-node=8 \
      --node-rank="$rank" \
      --local-addr="10.92.20.$node" \
      --master-addr="10.92.20.$master_node" \
      --master-port=29500 \
      --log-dir="$run_dir/logs/trainer/torchrun" \
      --tee=3 \
      --redirects=3 \
      --local-ranks-filter=0 \
      -m prime_rl.trainer.rl.train \
      @ "$trainer_config"
    rm -f -- "$trainer_env_file"
    trap - EXIT
    ;;
  adapter-http)
    adapter_node=${adapter_node_id:?set adapter_node_id}
    adapter_port=${adapter_service_port:?set adapter_service_port}
    adapter_container=${adapter_container_name:?set adapter_container_name}
    test "$node" = "$adapter_node" || { echo "the adapter server runs on n$adapter_node" >&2; exit 1; }
    broadcast_root=${BROADCAST_ROOT:-$runs}
    test -d "$broadcast_root" || { echo "broadcast root does not exist: $broadcast_root" >&2; exit 1; }
    adapter_mounts=(-v "$broadcast_root":/broadcast:ro)
    adapter_dependency_mounts=()
    adapter_upgrade_args=()
    if [ "${ADAPTER_PREDICTIVE_UPGRADE:-0}" = 1 ]; then
      predictive_run=${ADAPTER_PREDICTIVE_RUN:?set ADAPTER_PREDICTIVE_RUN}
      predictive_broadcast_path=${ADAPTER_PREDICTIVE_BROADCAST_PATH:?set ADAPTER_PREDICTIVE_BROADCAST_PATH}
      predictive_source_url=${ADAPTER_PREDICTIVE_SOURCE_URL:?set ADAPTER_PREDICTIVE_SOURCE_URL}
      [[ "$predictive_run" =~ ^[A-Za-z0-9_-]+$ ]] || {
        echo "ADAPTER_PREDICTIVE_RUN contains an unsafe run name" >&2
        exit 1
      }
      expected_broadcast_path="$predictive_run/run_$predictive_run/broadcasts"
      test "$predictive_broadcast_path" = "$expected_broadcast_path" || {
        echo "ADAPTER_PREDICTIVE_BROADCAST_PATH is not the active run broadcast path" >&2
        exit 1
      }
      active_broadcasts="$broadcast_root/$predictive_broadcast_path"
      for path in \
        "$broadcast_root" \
        "$broadcast_root/$predictive_run" \
        "$broadcast_root/$predictive_run/run_$predictive_run" \
        "$active_broadcasts"; do
        test ! -L "$path" || {
          echo "active predictive broadcast path contains a symlink: $path" >&2
          exit 1
        }
      done
      sudo install -d -o 1000 -g 1000 -m 0755 \
        "$broadcast_root/$predictive_run" \
        "$broadcast_root/$predictive_run/run_$predictive_run" \
        "$active_broadcasts"
      test -d "$active_broadcasts" && test ! -L "$active_broadcasts" || {
        echo "cannot create the active predictive broadcast path: $active_broadcasts" >&2
        exit 1
      }
      predictive_source_url="${predictive_source_url%/}/run_$predictive_run/broadcasts"
      adapter_mounts+=(-v "$active_broadcasts":/broadcast/"$predictive_broadcast_path":rw)
      adapter_dependency_mounts=(
        -v "$deployment/prime-rl/src/prime_rl/utils/adapter_delta.py":/app/src/prime_rl/utils/adapter_delta.py:ro
      )
      adapter_upgrade_args=(
        --upgrade-run "$predictive_run"
        --upgrade-path "$predictive_broadcast_path"
        --source-base-url "$predictive_source_url"
      )
    elif [ -n "${ADAPTER_PREDICTIVE_RUN:-}" ] || \
      [ -n "${ADAPTER_PREDICTIVE_BROADCAST_PATH:-}" ] || \
      [ -n "${ADAPTER_PREDICTIVE_SOURCE_URL:-}" ]; then
      echo "predictive adapter settings require ADAPTER_PREDICTIVE_UPGRADE=1" >&2
      exit 1
    fi
    sudo docker rm -f "$adapter_container" >/dev/null 2>&1 || true
    sudo docker run -d --name "$adapter_container" --restart unless-stopped \
      --network host \
      "${adapter_mounts[@]}" \
      "${adapter_dependency_mounts[@]}" \
      -v "$deployment/training_ops/runtime/china/adapter_server.py":/app/adapter_server.py:ro \
      --entrypoint /app/.venv/bin/python "$image" \
      /app/adapter_server.py --port "$adapter_port" --directory /broadcast \
      "${adapter_upgrade_args[@]}"
    ;;
  *)
    echo "usage: $0 trainer|adapter-http" >&2
    exit 1
    ;;
esac
