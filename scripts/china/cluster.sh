#!/usr/bin/env bash
set -euo pipefail

image=${prime_image:-prime-rl:glm52-v080-qual}
deployment=${prime_deployment:-/data/glm52-rl/deployment-glm52-optims}
trainer_container=${prime_trainer_container:-glm52-optims-trainer}
orchestrator_container=${prime_orchestrator_container:-glm52-optims-orchestrator}
nccl_channels=${PRIME_NCCL_CHANNELS:-4}
profile_pp_stage=${PRIME_PROFILE_PP_STAGE:-0}
profile_pp_rank=${PRIME_PROFILE_PP_RANK:--1}
# RUN selects which config pair under configs/china to use and where logs land.
# Defaults keep the historical r33 layout.
RUN=${RUN:-r33}
run_name=${RUN_NAME:-run_${RUN:-r33}}
config=$deployment/configs/china
orchestrator_config=${ORCHESTRATOR_CONFIG_DIR:-$config}
# CONFIG_TAG picks configs/china/trainer${CONFIG_TAG}.toml (e.g. ".r34")
CONFIG_TAG=${CONFIG_TAG:-}
trainer_config=${TRAINER_CONFIG:-$config/trainer${CONFIG_TAG}.toml}
run_dir=/data/glm52-rl/runs/$RUN/$run_name
node=${NODE_ID:-$(ip -4 -o addr show vlan130 | awk '{split($4, a, /[.\/]/); print a[4]}')}

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
  -e PYTHONUNBUFFERED=1
  -e OMP_NUM_THREADS=1
  -e NCCL_SOCKET_IFNAME=vlan130
  -e GLOO_SOCKET_IFNAME=vlan130
  -e NCCL_IB_DISABLE=1
  -e NCCL_P2P_DISABLE=1
  -e NCCL_SOCKET_NTHREADS=1
  -e NCCL_NSOCKS_PERTHREAD=1
  # Native checkpoint FP8 restores enough VRAM headroom to use the faster x24
  # communication settings without putting routed-expert allocations at risk.
  -e NCCL_BUFFSIZE=1048576
  # NCCL >=2.19 allocates host bounce buffers via cuMem/memfd (invisible in
  # /dev/shm, counted as shmem) -- the 300GB/node "shared" spikes. Legacy path.
  -e NCCL_CUMEM_HOST_ENABLE=0
  # Four channels measured substantially faster than two. One hangs and must
  # never be used.
  -e NCCL_MIN_NCHANNELS="$nccl_channels"
  -e NCCL_MAX_NCHANNELS="$nccl_channels"
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

python_dependency_mounts=()
if [ -n "${MUSE_PYTHON_OVERRIDE_DIR:-}" ]; then
  test -d "$MUSE_PYTHON_OVERRIDE_DIR/safetensors" || {
    echo "MUSE_PYTHON_OVERRIDE_DIR must contain safetensors/" >&2
    exit 1
  }
  test -d "$MUSE_PYTHON_OVERRIDE_DIR/safetensors-0.8.0.dist-info" || {
    echo "MUSE_PYTHON_OVERRIDE_DIR must contain safetensors-0.8.0.dist-info/" >&2
    exit 1
  }
  python_dependency_mounts=(
    -v "$MUSE_PYTHON_OVERRIDE_DIR/safetensors":/app/.venv/lib/python3.12/site-packages/safetensors:ro
    -v "$MUSE_PYTHON_OVERRIDE_DIR/safetensors-0.8.0.dist-info":/app/.venv/lib/python3.12/site-packages/safetensors-0.7.0.dist-info:ro
  )
fi

verifiers_mount=()
if [ -n "${VERIFIERS_OVERRIDE_DIR:-}" ]; then
  test -d "$VERIFIERS_OVERRIDE_DIR/verifiers/v1/configs" || {
    echo "VERIFIERS_OVERRIDE_DIR must contain verifiers/v1/configs/" >&2
    exit 1
  }
  verifiers_mount=(
    -v "$VERIFIERS_OVERRIDE_DIR/verifiers":/app/deps/verifiers/verifiers:ro
  )
fi

prime_configs_mount=(
  -v "$deployment/source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py":/app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py:ro
)
if [ -n "${PRIME_CONFIGS_OVERRIDE_DIR:-}" ]; then
  test -f "$PRIME_CONFIGS_OVERRIDE_DIR/configs/trainer.py" || {
    echo "PRIME_CONFIGS_OVERRIDE_DIR must contain configs/trainer.py" >&2
    exit 1
  }
  prime_configs_mount=(
    -v "$PRIME_CONFIGS_OVERRIDE_DIR/configs":/app/packages/prime-rl-configs/src/prime_rl/configs:ro
  )
fi

case ${1:-} in
  trainer)
    # TRAINER_NODES lists the participating nodes in rank order; a node's rank
    # is its position in that list. Set it to drop a node from the job -- e.g.
    # with n81 down, TRAINER_NODES="82 76 77 78 80 83 84" gives a 7-node,
    # 56-rank job. NNODES follows automatically.
    trainer_nodes=${trainer_nodes:-${TRAINER_NODES:-"82 76 77 78 80 83 84 85"}}   # n85 = stage 7 (ex-judge); n81 still down
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
    read -r rollout_transport trainer_output trainer_only parity_trace_dir parity_trace_source parity_pp parity_cp parity_batch parity_max_calls parity_max_files parity_max_bytes parity_diagnostics parity_diagnostic_tokens parity_diagnostic_max_files parity_diagnostic_max_bytes configured_int4_combine_rounds configured_fp8_expert_activation_offload < <(
      python3 - "$trainer_config" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    config = tomllib.load(f)
data = config.get("data", {})
cache = data.get("rollout_cache", {})
trainer_only = "fake" in data or cache.get("mode") == "replay"
env_vars = config.get("env_vars", {})
model = config.get("model", {})
fake = data.get("fake", {})
memory_env = (
    env_vars.get("PRIME_INT4_COMBINE_ROUNDS", ""),
    env_vars.get("PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD", ""),
)
print(
    config["rollout_transport"]["type"],
    config["output_dir"],
    int(trainer_only),
    env_vars.get("PRIME_PARITY_ROUTER_TRACE_DIR", ""),
    env_vars.get("PRIME_PARITY_ROUTER_TRACE_SOURCE", ""),
    model.get("pp", ""),
    model.get("cp", ""),
    fake.get("batch_size", ""),
    env_vars.get("PRIME_PARITY_TRACE_MAX_CALLS", ""),
    env_vars.get("PRIME_PARITY_TRACE_MAX_FILES", ""),
    env_vars.get("PRIME_PARITY_TRACE_MAX_BYTES", ""),
    env_vars.get("PRIME_PARITY_DIAGNOSTICS", ""),
    env_vars.get("PRIME_PARITY_DIAGNOSTIC_TOKENS", ""),
    env_vars.get("PRIME_PARITY_DIAGNOSTIC_MAX_FILES", ""),
    env_vars.get("PRIME_PARITY_DIAGNOSTIC_MAX_BYTES", ""),
    *memory_env,
)
PY
    )
    if [ -n "$configured_int4_combine_rounds" ] || [ -n "$configured_fp8_expert_activation_offload" ]; then
      [ "$configured_int4_combine_rounds" = "8" ] || {
        echo "GLM trainer configs must pin PRIME_INT4_COMBINE_ROUNDS=8" >&2
        exit 1
      }
      [ "$configured_fp8_expert_activation_offload" = "0" ] || {
        echo "GLM trainer configs must pin PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD=0" >&2
        exit 1
      }
      int4_combine_rounds="$configured_int4_combine_rounds"
      fp8_expert_activation_offload="$configured_fp8_expert_activation_offload"
    else
      int4_combine_rounds="${PRIME_INT4_COMBINE_ROUNDS:-1}"
      fp8_expert_activation_offload="${PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD:-0}"
    fi
    parity_trace_runtime=""
    parity_image_digest=""
    parity_env=()
    if [ -n "$parity_trace_dir" ]; then
      [ "$parity_trace_dir" = "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces" ] || {
        echo "parity trace directory is not the pinned base directory" >&2
        exit 1
      }
      [[ "$RUN" =~ ^[A-Za-z0-9_-]+$ ]] || {
        echo "RUN is unsafe for a parity trace directory" >&2
        exit 1
      }
      case "$parity_pp:$parity_cp:$parity_batch" in
        [1-9]*:[1-9]*:[1-9]* ) ;;
        *) echo "parity trainer TOML must define model.pp, model.cp, and data.fake.batch_size" >&2; exit 1 ;;
      esac
      case "$parity_max_calls:$parity_max_files:$parity_max_bytes" in
        [1-9]*:[1-9]*:[1-9]* ) ;;
        *) echo "parity trainer TOML must define all trace limits" >&2; exit 1 ;;
      esac
      if [ -n "$parity_diagnostics" ]; then
        case "$parity_diagnostics" in
          0|1) ;;
          *)
            echo "PRIME_PARITY_DIAGNOSTICS must be 0 or 1" >&2
            exit 1
            ;;
        esac
      fi
      if [ "$parity_diagnostics" = "1" ]; then
        case "$parity_diagnostic_tokens:$parity_diagnostic_max_files:$parity_diagnostic_max_bytes" in
          [1-9]*:[1-9]*:[1-9]* ) ;;
          *) echo "parity diagnostics must define all diagnostic limits" >&2; exit 1 ;;
        esac
      fi
      case "$parity_trace_dir" in
        /*) ;;
        *) echo "PRIME_PARITY_ROUTER_TRACE_DIR must be absolute" >&2; exit 1 ;;
      esac
      test -n "$parity_trace_source" || {
        echo "parity trace source is required when parity tracing is enabled" >&2
        exit 1
      }
      test -f "$parity_trace_source/session.json" || {
        echo "parity trace session is missing: $parity_trace_source/session.json" >&2
        exit 1
      }
      parity_trace_runtime="$parity_trace_dir/$RUN"
      sudo install -d -o 1000 -g 1000 -m 0755 "$parity_trace_runtime"
      sudo find "$parity_trace_runtime" -mindepth 1 -maxdepth 1 -delete
      sudo install -o 1000 -g 1000 -m 0644 "$parity_trace_source/session.json" \
        "$parity_trace_runtime/session.json"
      sudo -u '#1000' test -w "$parity_trace_runtime" || {
        echo "parity trace directory is not writable: $parity_trace_runtime" >&2
        exit 1
      }
      parity_image_digest=$(sudo docker image inspect "$image" --format '{{.Id}}')
      if ! [[ "$parity_image_digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        echo "trainer image has no exact digest: $parity_image_digest" >&2
        exit 1
      fi
      parity_env+=(
        -e "PRIME_PARITY_ROUTER_TRACE_DIR=$parity_trace_runtime"
        -e "PRIME_PARITY_TRACE_ROOT=$parity_trace_runtime"
        -e "PRIME_PARITY_ROUTER_TRACE_SOURCE=$parity_trace_source"
        -e "PRIME_TRAINER_IMAGE_DIGEST=$parity_image_digest"
        -e "PRIME_TRAINER_HOST=n$node"
        -e "PRIME_PARITY_PP_SIZE=$parity_pp"
        -e "PRIME_PARITY_CP_SIZE=$parity_cp"
        -e "PRIME_PARITY_BATCH_SIZE=$parity_batch"
        -e "PRIME_PARITY_TRACE_MAX_CALLS=$parity_max_calls"
        -e "PRIME_PARITY_TRACE_MAX_FILES=$parity_max_files"
        -e "PRIME_PARITY_TRACE_MAX_BYTES=$parity_max_bytes"
      )
      if [ -n "$parity_diagnostics" ]; then
        parity_env+=(
          -e "PRIME_PARITY_DIAGNOSTICS=$parity_diagnostics"
          -e "PRIME_PARITY_DIAGNOSTIC_TOKENS=$parity_diagnostic_tokens"
          -e "PRIME_PARITY_DIAGNOSTIC_MAX_FILES=$parity_diagnostic_max_files"
          -e "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES=$parity_diagnostic_max_bytes"
        )
      fi
    fi
    sudo install -d -o 1000 -g 1000 -m 0755 "$trainer_output"
    if [ "$rank" -eq 0 ]; then
      if [ "$rollout_transport" = filesystem ] && [ "$trainer_only" = 0 ]; then
        sync_log="/data/glm52-rl/microbatch-sync-$RUN.log"
        nohup setsid env RUN_DIR="$trainer_output" \
          "$deployment/scripts/china/sync_microbatches.sh" \
          >>"$sync_log" 2>&1 </dev/null &
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
    sudo install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/cache/torchinductor
    sudo install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/cache/tilelang
    # Triton compiles kernels into ~/.triton and filled n76 root twice.
    sudo install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/cache/triton
    sudo install -d -o 1000 -g 1000 -m 1777 /data/glm52-rl/cache/tmp
    # Recorded rollout batches used by explicit trainer qualification configs.
    sudo install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/cache/batch_pp
    # nccl shm segments leak on every trainer crash (~33MB each, observed
    # 1500+ per node = 50GB+); the container was just removed, so purge.
    sudo rm -f /dev/shm/nccl-* 2>/dev/null || true
    sudo docker rm -f "$trainer_container" >/dev/null 2>&1 || true
    # Keep the optional checkpoint cache disabled. It is not selected by the
    # current dequantized, node-local EP path, but its single slot per layer is
    # not safe for multiple 1F1B micro-batches.
    sudo docker run -d --name "$trainer_container" \
      --gpus all "${common[@]}" \
      -v /data/glm52-rl/cache/torchinductor:/tmp/torchinductor_appuser \
      -v /data/glm52-rl/cache/tilelang:/home/appuser/.tilelang \
      -v /data/glm52-rl/cache/triton:/home/appuser/.triton \
      -v /data/glm52-rl/cache/tmp:/scratch-tmp \
      "${transformers_mount[@]}" \
      "${python_dependency_mounts[@]}" \
      "${verifiers_mount[@]}" \
      "${prime_configs_mount[@]}" \
      -e TMPDIR=/scratch-tmp \
      -e PYTHONPATH="$deployment/env":/app/packages/prime-rl-configs/src:/app/src \
      "${parity_env[@]}" \
      -v "$deployment/source/src/prime_rl/trainer/fp8_checkpoint.py":/app/src/prime_rl/trainer/fp8_checkpoint.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/model.py":/app/src/prime_rl/trainer/model.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/fp8_ada.py":/app/src/prime_rl/trainer/models/layers/fp8_ada.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/moe.py":/app/src/prime_rl/trainer/models/layers/moe.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/checkpointing.py":/app/src/prime_rl/trainer/models/layers/checkpointing.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/norms.py":/app/src/prime_rl/trainer/models/layers/norms.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/hf_ring_attn.py":/app/src/prime_rl/trainer/models/layers/hf_ring_attn.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/lm_head_gemma.py":/app/src/prime_rl/trainer/models/layers/lm_head_gemma.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/expert_parallel.py":/app/src/prime_rl/trainer/distributed/expert_parallel.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/partial_reshard.py":/app/src/prime_rl/trainer/distributed/partial_reshard.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/pipeline_parallel.py":/app/src/prime_rl/trainer/distributed/pipeline_parallel.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/pp_runner.py":/app/src/prime_rl/trainer/distributed/pp_runner.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/parallel_dims.py":/app/src/prime_rl/trainer/parallel_dims.py:ro \
      -v "$deployment/source/src/prime_rl/utils/cp.py":/app/src/prime_rl/utils/cp.py:ro \
      -v "$deployment/source/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/train.py":/app/src/prime_rl/trainer/rl/train.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/token_export.py":/app/src/prime_rl/trainer/rl/token_export.py:ro \
      -v "$deployment/source/src/prime_rl/parity_router_trace.py":/app/src/prime_rl/parity_router_trace.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/runs.py":/app/src/prime_rl/trainer/runs.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/data.py":/app/src/prime_rl/trainer/rl/data.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/weights.py":/app/src/prime_rl/trainer/weights.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/utils.py":/app/src/prime_rl/trainer/utils.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/broadcast/filesystem.py":/app/src/prime_rl/trainer/rl/broadcast/filesystem.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/lora/multi_linear.py":/app/src/prime_rl/trainer/models/layers/lora/multi_linear.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py":/app/src/prime_rl/trainer/models/layers/lora/native_fp8_moe.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/lora.py":/app/src/prime_rl/trainer/lora.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/model.py":/app/src/prime_rl/trainer/model.py:ro \
      -v "$deployment/source/src/prime_rl/utils/vlm.py":/app/src/prime_rl/utils/vlm.py:ro \
      --env-file /data/trinity/training.env \
      -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}" \
      -e TORCHINDUCTOR_COMPILE_THREADS=8 \
      -e PRIME_EP_PROFILE="${PRIME_EP_PROFILE:-0}" \
      -e PRIME_EP_PROFILE_EVERY="${PRIME_EP_PROFILE_EVERY:-150}" \
      -e PRIME_EXPERT_PROFILE_EVERY="${PRIME_EXPERT_PROFILE_EVERY:-90}" \
      -e PRIME_BIND_GPU_NUMA="${PRIME_BIND_GPU_NUMA:-0}" \
      -e PRIME_NCCL_CHANNELS="$nccl_channels" \
      -e PRIME_PROFILE_PP_STAGE="$profile_pp_stage" \
      -e PRIME_PROFILE_PP_RANK="$profile_pp_rank" \
      -e PRIME_EP_AC_CACHE="${PRIME_EP_AC_CACHE:-0}" \
      -e PRIME_INT4_COMBINE_ROUNDS="$int4_combine_rounds" \
      -e PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD="$fp8_expert_activation_offload" \
      -e PRIME_PP_MAX_MICRO="${PRIME_PP_MAX_MICRO:-}" \
      -e PRIME_STATS_DEBUG="${PRIME_STATS_DEBUG:-}" \
      --entrypoint /app/.venv/bin/torchrun "$image" \
      --role=trainer \
      --nnodes="$nnodes" \
      --nproc-per-node=8 \
      --node-rank="$rank" \
      --local-addr="10.92.20.$node" \
      --master-addr=10.92.20.82 \
      --master-port=29500 \
      --log-dir="$run_dir/logs/trainer/torchrun" \
      --tee=3 \
      --redirects=3 \
      --local-ranks-filter=0 \
      -m prime_rl.trainer.rl.train \
      @ "$trainer_config"
    ;;
  orchestrator)
    echo "the isolated GLM branch runs its orchestrator on h200a; use scripts/h200/start_orchestrator.sh" >&2
    exit 2
    test "$node" = 82 || { echo "the orchestrator runs on n82" >&2; exit 1; }
    orchestrator_output=$(
      python3 - "$orchestrator_config/orchestrator${CONFIG_TAG}.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as source:
    print(tomllib.load(source)["output_dir"])
PY
    )
    # The orchestrator writes run control data under `output_dir`, and its
    # environment workers write logs beside that directory. Both locations
    # must be writable by the container user.
    sudo install -d -o 1000 -g 1000 -m 0755 \
      "$(dirname "$orchestrator_output")" "$orchestrator_output"
    sudo docker rm -f "$orchestrator_container" >/dev/null 2>&1 || true
    sudo docker run -d --name "$orchestrator_container" \
      "${common[@]}" \
      --env-file /data/trinity/training.env \
      -e VLLM_API_KEY=unused \
      -e PYTHONPATH=/data/glm52-rl/decensor-v1:/app/src \
      -e HF_HOME=/data/glm52-rl/hf-cache \
      -e HF_HUB_OFFLINE=1 \
      -e HF_DATASETS_OFFLINE=1 \
      -v "$deployment/source/deps/renderers/renderers/base.py":/app/deps/renderers/renderers/base.py:ro \
      -v "$deployment/source/deps/renderers/renderers/client.py":/app/deps/renderers/renderers/client.py:ro \
      -v "$deployment/source/deps/verifiers/verifiers/v1/clients/eval.py":/app/deps/verifiers/verifiers/v1/clients/eval.py:ro \
      -v "$deployment/source/deps/verifiers/verifiers/v1/clients/config.py":/app/deps/verifiers/verifiers/v1/clients/config.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/algo/grpo.py":/app/src/prime_rl/orchestrator/algo/grpo.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/dispatcher.py":/app/src/prime_rl/orchestrator/dispatcher.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/orchestrator.py":/app/src/prime_rl/orchestrator/orchestrator.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/train_sink.py":/app/src/prime_rl/orchestrator/train_sink.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/train_source.py":/app/src/prime_rl/orchestrator/train_source.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/watcher.py":/app/src/prime_rl/orchestrator/watcher.py:ro \
      -v "$deployment/source/src/prime_rl/utils/client.py":/app/src/prime_rl/utils/client.py:ro \
      -v "$deployment/source/src/prime_rl/utils/checkpoint_delta.py":/app/src/prime_rl/utils/checkpoint_delta.py:ro \
      --entrypoint /app/.venv/bin/orchestrator "$image" \
      @ "$orchestrator_config/orchestrator${CONFIG_TAG}.toml"
    ;;
  gemma)
    test "$node" = 85 || { echo "the Gemma server runs on n85" >&2; exit 1; }
    sudo docker rm -f glm52-gemma glm52-gemma-user glm52-gemma-judge \
      glm52-gemma-0 glm52-gemma-1 glm52-gemma-2 glm52-gemma-3 \
      glm52-gemma-proxy >/dev/null 2>&1 || true
    run_gemma() {
      sudo docker run -d --name "glm52-gemma-$1" --restart unless-stopped \
        --gpus "\"device=$2\"" \
        --network host \
        --ipc host \
        --ulimit nofile=1048576:1048576 \
        --ulimit memlock=-1:-1 \
        --ulimit stack=67108864:67108864 \
        -v /data:/data \
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
        -e PYTHONUNBUFFERED=1 \
        -e OMP_NUM_THREADS=1 \
        -e NCCL_SOCKET_IFNAME=vlan130 \
        -e GLOO_SOCKET_IFNAME=vlan130 \
        -e NCCL_IB_DISABLE=1 \
        -e NCCL_P2P_DISABLE=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.8,max_split_size_mb:256 \
        --entrypoint /app/.venv/bin/vllm \
        "$image" \
        serve /data/glm52-rl/gemma-4-26b \
        --served-model-name google/gemma-4-26B-A4B-it \
        --host 0.0.0.0 \
        --port "$3" \
        --tensor-parallel-size 2 \
        --disable-custom-all-reduce \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.90 \
        --max-model-len 16384 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 32768 \
        --trust-remote-code
    }
    run_gemma 0 0,1 40100
    run_gemma 1 2,3 40101
    run_gemma 2 4,5 40102
    run_gemma 3 6,7 40103
    sudo docker run -d --name glm52-gemma-proxy --restart unless-stopped \
      --network host \
      --ulimit nofile=1048576:1048576 \
      -v "$deployment/scripts/china/gemma_proxy.py":/app/proxy/gemma_proxy.py:ro \
      --entrypoint /app/.venv/bin/uvicorn \
      "$image" gemma_proxy:app --app-dir /app/proxy --host 0.0.0.0 --port 40002
    ;;
  adapter-http)
    test "$node" = 82 || { echo "the adapter server runs on n82" >&2; exit 1; }
    broadcast_root=${BROADCAST_ROOT:-/data/glm52-rl/runs}
    test -d "$broadcast_root" || { echo "broadcast root does not exist: $broadcast_root" >&2; exit 1; }
    sudo docker rm -f glm52-adapter-http >/dev/null 2>&1 || true
    sudo docker run -d --name glm52-adapter-http --restart unless-stopped \
      --network host \
      -v "$broadcast_root":/broadcast:ro \
      -v "$deployment/scripts/china/adapter_server.py":/app/adapter_server.py:ro \
      --entrypoint /app/.venv/bin/python "$image" \
      /app/adapter_server.py --port 8100 --directory /broadcast
    ;;
  *)
    echo "usage: $0 trainer|orchestrator|adapter-http|gemma" >&2
    exit 1
    ;;
esac
