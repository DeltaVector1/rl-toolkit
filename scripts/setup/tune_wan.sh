#!/usr/bin/env bash
# Tune the TCP stack for the China <-> H200 path. Run on n82 and on each H200
# host. Safe to run repeatedly. Needs sudo for sysctl only.
#
# Topology, which is why this is not obvious:
#
#   H200 (docker) --internet, 7.5% loss, 210ms RTT--> jump 202.101.73.12
#                 --LAN--> n82 10.92.20.82
#
# The rollout responses travel H200 -> n82, so the H200 is the sender for the
# bulk data. Congestion control is a sender-side choice, so the algorithm that
# matters most is the H200's. BBR cannot be loaded there: the "hosts" are
# Docker containers (systemd-detect-virt reports docker) with no
# CAP_SYS_MODULE, and the underlying host has not loaded tcp_bbr either.
#
# On n82 BBR is applied PER ROUTE, only for the two engine hosts. Do not set it
# globally. n82 is also a trainer node and its NCCL all-to-all crosses the same
# interface, where cubic measured 87% wire efficiency against BBR's 36%. See
# docs/PERFORMANCE.md.
set -uo pipefail

ENGINE_IPS=${ENGINE_IPS:-"152.236.142.239 152.236.142.243"}
ROLE=${1:-auto}

if [ "$ROLE" = "auto" ]; then
  if [ -d /data/glm52-rl/deployment/scripts/china ] && ip -brief addr | grep -q '10\.92\.20\.'; then
    ROLE=trainer
  else
    ROLE=engine
  fi
fi

say() { echo "$(date -u +%H:%M:%S) $*"; }

# Both ends. An idle keep-alive socket restarts in slow start, and on a 210ms
# RTT that costs several round trips before it carries data again. Rollout
# connections idle between requests constantly. PERFORMANCE.md records this
# setting makes no measurable difference to NCCL, so it is safe on n82.
sudo sysctl -qw net.ipv4.tcp_slow_start_after_idle=0
# Recover from an MTU black hole instead of stalling.
sudo sysctl -qw net.ipv4.tcp_mtu_probing=1
# 210ms x a few hundred Mbit needs far more than the 6 MB default ceiling.
sudo sysctl -qw net.ipv4.tcp_rmem="4096 131072 33554432"
sudo sysctl -qw net.ipv4.tcp_wmem="4096 65536 33554432"
say "$ROLE: slow_start_after_idle=0 mtu_probing=1 rmem/wmem ceiling 32MB"

if [ "$ROLE" = "trainer" ]; then
  if sudo modprobe tcp_bbr 2>/dev/null; then
    gw=$(ip route get "${ENGINE_IPS%% *}" 2>/dev/null | sed -n 's/.*via \([0-9.]*\).*/\1/p')
    dev=$(ip route get "${ENGINE_IPS%% *}" 2>/dev/null | sed -n 's/.*dev \([^ ]*\).*/\1/p')
    if [ -n "$gw" ] && [ -n "$dev" ]; then
      for ip in $ENGINE_IPS; do
        sudo ip route replace "$ip" via "$gw" dev "$dev" congctl bbr 2>/dev/null \
          && say "route $ip -> bbr (global cc stays $(sysctl -n net.ipv4.tcp_congestion_control))"
      done
    else
      say "WARNING: could not resolve gateway/device for the engine routes"
    fi
  else
    say "WARNING: tcp_bbr will not load; leaving the engine routes on cubic"
  fi
fi

say "done"
