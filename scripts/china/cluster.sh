#!/usr/bin/env bash
set -euo pipefail

image=prime-rl:glm52-fp8
deployment=/data/glm52-rl/deployment
# RUN selects which config pair under configs/china to use and where logs land.
# Defaults keep the historical r33 layout.
RUN=${RUN:-r33}
run_name=${RUN_NAME:-run_${RUN:-r33}}
config=$deployment/configs/china
# CONFIG_TAG picks configs/china/trainer${CONFIG_TAG}.toml (e.g. ".r34")
CONFIG_TAG=${CONFIG_TAG:-}
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
  -e NCCL_MIN_NCHANNELS=4
  -e NCCL_MAX_NCHANNELS=4
  -e TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
  -e TORCH_NCCL_DUMP_ON_TIMEOUT=1
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  -e TORCH_NCCL_ENABLE_MONITORING=1
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
)

case ${1:-} in
  trainer)
    # TRAINER_NODES lists the participating nodes in rank order; a node's rank
    # is its position in that list. Set it to drop a node from the job -- e.g.
    # with n81 down, TRAINER_NODES="82 76 77 78 80 83 84" gives a 7-node,
    # 56-rank job. NNODES follows automatically.
    TRAINER_NODES=${TRAINER_NODES:-"82 76 77 78 80 83 84 85"}   # n85 = stage 7 (ex-judge); n81 still down
    rank=-1
    idx=0
    for n in $TRAINER_NODES; do
      if [ "$n" = "$node" ]; then rank=$idx; fi
      idx=$((idx + 1))
    done
    nnodes=$idx
    if [ "$rank" -lt 0 ]; then
      echo "node $node is not in TRAINER_NODES ($TRAINER_NODES)" >&2; exit 1
    fi
    if [ "$rank" -eq 0 ]; then
      read -r rollout_transport trainer_output < <(
        python3 - "$config/trainer${CONFIG_TAG}.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as f:
    config = tomllib.load(f)
print(config["rollout_transport"]["type"], config["output_dir"])
PY
      )
      if [ "$rollout_transport" = filesystem ]; then
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
    # Batch replay cache (PRIME_BATCH_CACHE), written by the container user.
    sudo install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/cache/batch_pp
    # nccl shm segments leak on every trainer crash (~33MB each, observed
    # 1500+ per node = 50GB+); the container was just removed, so purge.
    sudo rm -f /dev/shm/nccl-* 2>/dev/null || true
    sudo docker rm -f glm52-decensor-trainer >/dev/null 2>&1 || true
    # Keep the optional checkpoint cache disabled. It is not selected by the
    # current dequantized, node-local EP path, but its single slot per layer is
    # not safe for multiple 1F1B micro-batches.
    sudo docker run -d --name glm52-decensor-trainer \
      --gpus all "${common[@]}" \
      -v /data/glm52-rl/cache/torchinductor:/tmp/torchinductor_appuser \
      -v /data/glm52-rl/cache/tilelang:/home/appuser/.tilelang \
      -v /data/glm52-rl/cache/triton:/home/appuser/.triton \
      -v /data/glm52-rl/cache/tmp:/scratch-tmp \
      -e TMPDIR=/scratch-tmp \
      -v "$deployment/source/src/prime_rl/trainer/fp8_checkpoint.py":/app/src/prime_rl/trainer/fp8_checkpoint.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/fp8_ada.py":/app/src/prime_rl/trainer/models/layers/fp8_ada.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/moe.py":/app/src/prime_rl/trainer/models/layers/moe.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/checkpointing.py":/app/src/prime_rl/trainer/models/layers/checkpointing.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/norms.py":/app/src/prime_rl/trainer/models/layers/norms.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/expert_parallel.py":/app/src/prime_rl/trainer/distributed/expert_parallel.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/pipeline_parallel.py":/app/src/prime_rl/trainer/distributed/pipeline_parallel.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/distributed/pp_runner.py":/app/src/prime_rl/trainer/distributed/pp_runner.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/parallel_dims.py":/app/src/prime_rl/trainer/parallel_dims.py:ro \
      -v "$deployment/source/src/prime_rl/utils/cp.py":/app/src/prime_rl/utils/cp.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/train.py":/app/src/prime_rl/trainer/rl/train.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/data.py":/app/src/prime_rl/trainer/rl/data.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/weights.py":/app/src/prime_rl/trainer/weights.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/utils.py":/app/src/prime_rl/trainer/utils.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/rl/broadcast/filesystem.py":/app/src/prime_rl/trainer/rl/broadcast/filesystem.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py":/app/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/layers/lora/multi_linear.py":/app/src/prime_rl/trainer/models/layers/lora/multi_linear.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/lora.py":/app/src/prime_rl/trainer/lora.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_fwd.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py":/app/src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py:ro \
      -v "$deployment/source/src/prime_rl/trainer/model.py":/app/src/prime_rl/trainer/model.py:ro \
      -v "$deployment/source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py":/app/packages/prime-rl-configs/src/prime_rl/configs/trainer.py:ro \
      --env-file /data/trinity/training.env \
      -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256}" \
      -e TORCHINDUCTOR_COMPILE_THREADS=8 \
      -e PRIME_EP_PROFILE="${PRIME_EP_PROFILE:-0}" \
      -e PRIME_EP_PROFILE_EVERY="${PRIME_EP_PROFILE_EVERY:-150}" \
      -e PRIME_EP_AC_CACHE="${PRIME_EP_AC_CACHE:-0}" \
      -e PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD="${PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD:-0}" \
      -e PRIME_BATCH_CACHE="${PRIME_BATCH_CACHE:-}" \
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
      @ "$config/trainer${CONFIG_TAG}.toml"
    ;;
  orchestrator)
    test "$node" = 82 || { echo "the orchestrator runs on n82" >&2; exit 1; }
    sudo docker rm -f glm52-decensor-orchestrator >/dev/null 2>&1 || true
    sudo docker run -d --name glm52-decensor-orchestrator \
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
      -v "$deployment/source/src/prime_rl/orchestrator/train_source.py":/app/src/prime_rl/orchestrator/train_source.py:ro \
      -v "$deployment/source/src/prime_rl/orchestrator/watcher.py":/app/src/prime_rl/orchestrator/watcher.py:ro \
      -v "$deployment/source/src/prime_rl/utils/client.py":/app/src/prime_rl/utils/client.py:ro \
      --entrypoint /app/.venv/bin/orchestrator "$image" \
      @ "$config/orchestrator${CONFIG_TAG}.toml"
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
    sudo docker rm -f glm52-adapter-http >/dev/null 2>&1 || true
    sudo docker run -d --name glm52-adapter-http --restart unless-stopped \
      --network host \
      -v /data/glm52-rl/runs:/data/glm52-rl/runs:ro \
      -v "$deployment/scripts/china/adapter_server.py":/app/adapter_server.py:ro \
      --entrypoint /app/.venv/bin/python "$image" \
      /app/adapter_server.py --port 8100 --directory /data/glm52-rl/runs
    ;;
  *)
    echo "usage: $0 trainer|orchestrator|adapter-http|gemma" >&2
    exit 1
    ;;
esac
