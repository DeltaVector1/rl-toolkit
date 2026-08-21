#!/usr/bin/env bash
set -euo pipefail

model=${1:?model directory is required}
deployment=/data/glm52-rl/deployment
runtime=/data/muse-glimmer/eval-runtime
image=${MUSE_EVAL_IMAGE:-vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d}
tp_size=${MUSE_EVAL_TP_SIZE:-2}
replica_count=$((8 / tp_size))

test -f "$model/model.safetensors.index.json"
test -d /data/muse-glimmer/transformers/transformers
test -d /data/muse-glimmer/python-overrides/safetensors
test $((8 % tp_size)) -eq 0

install -d -m 0755 "$runtime"
install -m 0644 "$deployment/scripts/h200/muse_transformers_sitecustomize.py" \
  "$runtime/sitecustomize.py"

for replica in $(seq 0 7); do
  sudo docker update --restart=no "muse-merged-eval-$replica" >/dev/null 2>&1 || true
  sudo docker rm -f "muse-merged-eval-$replica" >/dev/null 2>&1 || true
done

for replica in $(seq 0 $((replica_count - 1))); do
  first_device=$((replica * tp_size))
  last_device=$((first_device + tp_size - 1))
  devices=$(seq -s, "$first_device" "$last_device")
  port=$((8201 + replica))
  sudo docker run -d --name "muse-merged-eval-$replica" \
    --log-opt max-size=512m --log-opt max-file=2 \
    --gpus "\"device=$devices\"" \
    --network host \
    --ipc host \
    --ulimit stack=67108864:67108864 \
    -v /data:/data \
    -v /data/muse-glimmer/transformers/transformers:/app/.venv/lib/python3.12/site-packages/transformers:ro \
    -v /data/muse-glimmer/python-overrides/safetensors:/app/.venv/lib/python3.12/site-packages/safetensors:ro \
    -v /data/muse-glimmer/python-overrides/safetensors-0.8.0.dist-info:/app/.venv/lib/python3.12/site-packages/safetensors-0.7.0.dist-info:ro \
    -e PYTHONPATH="$runtime" \
    -e MUSE_TP_SIZE="$tp_size" \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=1 \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_P2P_DISABLE=1 \
    --entrypoint vllm \
    "$image" \
    serve "$model" \
    --served-model-name muse15-step60-merged \
    --host 0.0.0.0 \
    --port "$port" \
    --tensor-parallel-size "$tp_size" \
    --disable-custom-all-reduce \
    --model-impl transformers \
    --enforce-eager \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer
done
