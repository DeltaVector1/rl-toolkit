#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f glm52-rollout-proxy >/dev/null 2>&1 || true
sudo docker rm -f glm52-rollout-relay >/dev/null 2>&1 || true
sudo docker run -d --name glm52-rollout-proxy --restart unless-stopped \
  --network host \
  -v /data/glm52-rl/deployment/network/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro \
  haproxy:3.2-alpine
