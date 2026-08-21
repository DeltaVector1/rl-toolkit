#!/usr/bin/env bash
set -euo pipefail

root=/data/glm52-rl/rootfs
next=/data/glm52-rl/rootfs.new
container=glm52-rootfs-export

docker rm -f "$container" >/dev/null 2>&1 || true
rm -rf "$next"
mkdir -p "$next"
docker create --name "$container" prime-rl:glm52-decensor >/dev/null
docker export "$container" | tar -xpf - -C "$next"
docker rm "$container" >/dev/null

rm -rf "$root"
mv "$next" "$root"
