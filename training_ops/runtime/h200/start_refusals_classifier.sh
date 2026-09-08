#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
model=/data/glm52-rl/refusals-classifier-eval/model-rp-validity-396m
service=$deployment/evals/refusals-classifier/classifier_service.py
image=${VLLM_IMAGE:-vllm/vllm-openai:nightly-5a4c8d99242e9e069b604d0e9b969e77f7dd501d}
container=refusals-classifier-eval
classifier_port=$(python3 - "$deployment/configs/nodes.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    value = tomllib.load(stream)["ports"].get("refusal_classifier")
if not isinstance(value, int) or not 1 <= value <= 65535:
    raise SystemExit("nodes config has no valid refusal classifier port")
print(value)
PY
)

test -f "$model/model.safetensors"
test -f "$service"

sudo docker rm -f "$container" >/dev/null 2>&1 || true
sudo docker run -d --name "$container" --restart unless-stopped \
  --gpus '"device=0"' \
  --network host \
  --ipc host \
  --log-opt max-size=128m \
  --log-opt max-file=2 \
  -v "$model":/model:ro \
  -v "$service":/classifier_service.py:ro \
  --entrypoint python3 \
  "$image" \
  /classifier_service.py \
  --model /model \
  --device cuda:0 \
  --port "$classifier_port" \
  --batch-size 32 \
  --batch-wait-ms 20

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:$classifier_port/health" >/dev/null; then
    curl -fsS "http://127.0.0.1:$classifier_port/health"
    exit 0
  fi
  sleep 2
done

sudo docker logs --tail 100 "$container" >&2
exit 1
