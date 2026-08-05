#!/usr/bin/env python3
"""Cluster helper for the 4090 trainer nodes.

  cl.py run  "<cmd>" [-n n82,n76]     run a command everywhere, grouped output
  cl.py push <local> <remote> [-n ..] copy a file out, verify checksums match
  cl.py edit <deployment-relative>    push that path from the local repo
  cl.py ps                            container status on every node
  cl.py restart <RUN> <NAME> <TAG>    stop and relaunch trainer + orchestrator
  cl.py stop                          stop trainer + orchestrator
  cl.py snapshot                      tar the deployment on every node
  cl.py rollback <stamp>              restore a snapshot everywhere

Nodes default to the seven live trainer nodes (n81 is down). Everything runs in parallel.
"""

import argparse
import os
import hashlib
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# n81 is down at the OS level (shell I/O error, ssh refused); the job runs
# 7 nodes / 56 ranks until it is back. Re-add it here and in TRAINER_NODES.
NODES = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
REPO = pathlib.Path(__file__).resolve().parents[2]
DEPLOY = "/data/glm52-rl/deployment"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25"]


def sh(host, cmd, timeout=120):
    try:
        p = subprocess.run(SSH + [host, cmd], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def fan(hosts, fn):
    with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        return list(zip(hosts, ex.map(fn, hosts)))


def show(results, limit=12):
    for host, (rc, out) in results:
        lines = [l for l in out.splitlines() if l.strip()][:limit]
        tag = "" if rc == 0 else f" (rc={rc})"
        print(f"=== {host}{tag}")
        for l in lines:
            print(f"    {l}")


def cmd_run(a):
    show(fan(a.nodes, lambda h: sh(h, a.command, a.timeout)))


def cmd_push(a):
    src = pathlib.Path(a.local)
    if not src.is_file():
        sys.exit(f"no such file: {src}")
    want = hashlib.md5(src.read_bytes()).hexdigest()

    def one(host):
        subprocess.run(SSH[:1] + ["-o", "BatchMode=yes", host, f"mkdir -p {pathlib.PurePosixPath(a.remote).parent}"],
                       capture_output=True)
        p = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(src), f"{host}:{a.remote}"],
                           capture_output=True, text=True)
        if p.returncode:
            return 1, "scp failed: " + p.stderr.strip()
        rc, got = sh(host, f"md5sum {a.remote} | cut -d' ' -f1")
        return (0, "ok") if got == want else (1, f"MISMATCH {got[:8]} != {want[:8]}")

    bad = [h for h, (rc, _) in fan(a.nodes, one) if rc]
    print(f"pushed {src.name} -> {a.remote}: {len(a.nodes) - len(bad)}/{len(a.nodes)} ok"
          + (f", FAILED: {bad}" if bad else ""))


def cmd_edit(a):
    """Push a deployment-relative path straight from the local repo."""
    local = REPO / a.path
    if not local.is_file():
        sys.exit(f"not in repo: {local}")
    a.local, a.remote = str(local), f"{DEPLOY}/{a.path}"
    cmd_push(a)


def cmd_ps(a):
    show(fan(a.nodes, lambda h: sh(h, 'sudo docker ps --format "{{.Names}}: {{.Status}}"')))


def cmd_stop(a):
    sh("n82", "sudo docker rm -f glm52-decensor-orchestrator")
    fan(a.nodes, lambda h: sh(h, "sudo docker rm -f glm52-decensor-trainer"))
    print("stopped trainer + orchestrator")


def cmd_restart(a):
    cmd_stop(a)
    # TRAINER_NODES must be passed explicitly: a local export does not survive
    # the ssh hop, and cluster.sh would silently fall back to its default node
    # list, leaving torchrun waiting for a node that is not coming.
    # n82 first so global rank 0 lands there: it is the rendezvous master and
    # the node whose container logs carry the rank-0 output.
    ordered = ["n82"] + [n for n in a.nodes if n != "n82"]
    # Launch containers from the SAME list that builds TRAINER_NODES; a -n
    # subset that dropped n82 would otherwise declare an nnodes count larger
    # than the containers actually started, and torchrun waits forever.
    a.nodes = ordered
    nodes_env = " ".join(n.lstrip("n") for n in ordered)
    # PRIME_EP_AC_CACHE replays the expert all-to-all payload on recompute, which
    # only matches under SELECTIVE activation checkpointing; with full AC it trips
    # "Recomputed values ... have different metadata". It also buys little under PP,
    # where the all-to-all is node-local and already 80x faster.
    ep_cache = os.environ.get("PRIME_EP_AC_CACHE", "1")
    env = (
        f"TRAINER_NODES='{nodes_env}' RUN={a.run} RUN_NAME={a.name} CONFIG_TAG={a.tag} "
        f"PRIME_EP_PROFILE=1 PRIME_EP_PROFILE_EVERY=150 PRIME_EP_AC_CACHE={ep_cache}"
    )
    # PRIME_BATCH_CACHE is deliberately NOT forwarded: a value leaked from the
    # operator's shell would make a production run replay one frozen batch for
    # every step while looking completely healthy. Replay is banned for
    # production runs; launch with cluster.sh directly if it is ever needed.
    fan(a.nodes, lambda h: sh(h, f"{env} bash {DEPLOY}/scripts/china/cluster.sh trainer", 240))
    sh("n82", f"RUN={a.run} RUN_NAME={a.name} CONFIG_TAG={a.tag} "
              f"bash {DEPLOY}/scripts/china/cluster.sh orchestrator", 240)
    print(f"restarted as RUN={a.run} CONFIG_TAG={a.tag}")
    cmd_ps(a)


def cmd_snapshot(a):
    stamp = subprocess.run(["date", "-u", "+%Y%m%d_%H%M%S"], capture_output=True, text=True).stdout.strip()
    fan(a.nodes, lambda h: sh(h, f"cd /data/glm52-rl && tar czf deployment-{stamp}.tar.gz deployment/", 300))
    print(f"snapshot stamp: {stamp}")


def cmd_rollback(a):
    def one(host):
        return sh(host, f"cd /data/glm52-rl && test -f deployment-{a.stamp}.tar.gz && "
                        f"rm -rf deployment && tar xzf deployment-{a.stamp}.tar.gz && echo restored", 300)
    show(fan(a.nodes, one))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--nodes", type=lambda s: s.split(","), default=NODES)
    ap.add_argument("-t", "--timeout", type=int, default=120)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run"); s.add_argument("command"); s.set_defaults(fn=cmd_run)
    s = sub.add_parser("push"); s.add_argument("local"); s.add_argument("remote"); s.set_defaults(fn=cmd_push)
    s = sub.add_parser("edit"); s.add_argument("path"); s.set_defaults(fn=cmd_edit)
    sub.add_parser("ps").set_defaults(fn=cmd_ps)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    s = sub.add_parser("restart")
    s.add_argument("run"); s.add_argument("name"); s.add_argument("tag"); s.set_defaults(fn=cmd_restart)
    sub.add_parser("snapshot").set_defaults(fn=cmd_snapshot)
    s = sub.add_parser("rollback"); s.add_argument("stamp"); s.set_defaults(fn=cmd_rollback)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
