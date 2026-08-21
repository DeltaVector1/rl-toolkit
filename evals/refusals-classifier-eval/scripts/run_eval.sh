#!/usr/bin/env bash
set -euo pipefail

mode=${1:-full}
model=${2:?usage: run_eval.sh full|periodic MODEL RESULT_NAME}
result_name=${3:?usage: run_eval.sh full|periodic MODEL RESULT_NAME}
deployment=${DEPLOYMENT:-/data/glm52-rl/deployment}
data_root=${DATA_ROOT:-/data/glm52-rl/refusals-classifier-eval}

case "$mode" in
  full)
    suite=$data_root/full.jsonl
    suite_sha=8ad40c2f9f66f3c786a0d10bd608be62050347a58663835342972aaa230dfcc6
    if sudo docker ps --format '{{.Names}}' | grep -qx glm52-decensor-orchestrator; then
      echo "the full refusal evaluation cannot share the production inference pool" >&2
      exit 1
    fi
    ;;
  periodic)
    suite=$data_root/periodic-4000.jsonl
    suite_sha=fec3d0af70a25e4fc95e9fccdb9437452bc15e02502a1fb76da6cf5d27b0edda
    ;;
  *)
    echo "mode must be full or periodic" >&2
    exit 1
    ;;
esac

endpoints=()
for port in $(seq 8001 8008); do
  endpoint="http://127.0.0.1:$port/v1"
  curl --fail --silent --max-time 5 "$endpoint/models" \
    | python3 -c 'import json,sys; names={row["id"] for row in json.load(sys.stdin)["data"]}; target=sys.argv[1]; raise SystemExit(0 if target in names else 1)' "$model"
  endpoints+=(--endpoint "$endpoint")
done

exec python3 "$deployment/evals/refusals-classifier-eval/runner.py" \
  --suite "$suite" \
  --suite-sha256 "$suite_sha" \
  "${endpoints[@]}" \
  --model "$model" \
  --display-name "$result_name" \
  --classifier-url http://127.0.0.1:18150 \
  --output "$data_root/results/$result_name" \
  --concurrency 128 \
  --max-tokens 8192 \
  --attempts 3
