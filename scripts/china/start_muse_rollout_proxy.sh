#!/usr/bin/env bash
set -euo pipefail

deployment=/data/glm52-rl/deployment
config=$deployment/configs/china/muse-rollout-haproxy.cfg

sudo docker run --rm \
  -v "$config":/usr/local/etc/haproxy/haproxy.cfg:ro \
  haproxy:3.2-alpine haproxy -c -f /usr/local/etc/haproxy/haproxy.cfg

sudo docker rm -f muse-rollout-proxy >/dev/null 2>&1 || true
sudo docker run -d --name muse-rollout-proxy --restart unless-stopped \
  --network host \
  --log-opt max-size=64m --log-opt max-file=2 \
  -v "$config":/usr/local/etc/haproxy/haproxy.cfg:ro \
  haproxy:3.2-alpine
