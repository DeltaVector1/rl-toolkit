#!/usr/bin/env bash
set -euo pipefail

: "${1:?usage: start_tunnels.sh PORT...}"
script=/data/glm52-rl/deployment/scripts/h200/start_tunnel.sh

for port in "$@"; do
  tmux kill-session -t "glm52-tunnel-$port" 2>/dev/null || true
  tmux new-session -d -s "glm52-tunnel-$port" "$script $port 8000"
done
