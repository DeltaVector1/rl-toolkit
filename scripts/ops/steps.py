#!/usr/bin/env python3
"""Step and micro-batch timing for the running trainer.

  steps.py                 last 10 steps, with per-micro-batch time
  steps.py -n 30           last 30 steps
  steps.py --run r35       a different run directory
  steps.py --watch         re-read every 60s

Reads trainer-metrics.jsonl for per-step timing and the container log for the
per-micro-batch all-to-all breakdown, then reports the numbers that actually
matter: seconds per micro-batch (the thing optimisation moves) and how long the
trainer sat waiting on rollouts (the thing that means the trainer is not the
bottleneck).
"""

import argparse
import json
import re
import subprocess
import time

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "n82"]


def sh(cmd, timeout=90):
    try:
        p = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except subprocess.TimeoutExpired:
        return ""


def metrics(run):
    path = f"/data/glm52-rl/runs/{run}/run_decensor_fp8_{run}/trainer-metrics.jsonl"
    raw = sh(f"sudo docker exec glm52-decensor-trainer cat {path} 2>/dev/null || cat {path} 2>/dev/null")
    steps, perf = {}, {}
    for line in raw.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "time/forward_backward" in r:
            steps[r["step"]] = r
        if "perf/throughput" in r:
            perf.setdefault(len(perf) + 1, r)
    return steps, perf


def micro_counts():
    out = sh('sudo docker logs glm52-decensor-trainer 2>&1 | grep -aoE "Received [0-9]+ micro batches for step [0-9]+"')
    counts = {}
    for m in re.finditer(r"Received (\d+) micro batches for step (\d+)", out):
        counts[int(m.group(2))] = int(m.group(1))
    return counts


def recent_microbatches(k=8):
    out = sh(f'sudo docker logs --tail 4000 glm52-decensor-trainer 2>&1 '
             f'| grep -aoE "wall=[0-9.]+s a2a=[0-9.]+s" | tail -{k}')
    vals = []
    for m in re.finditer(r"wall=([0-9.]+)s a2a=([0-9.]+)s", out):
        vals.append((float(m.group(1)), float(m.group(2))))
    return vals


def report(run, n):
    steps, _ = metrics(run)
    counts = micro_counts()
    if not steps:
        print(f"no completed steps yet for run {run}")
    else:
        print(f"{'step':>5} {'fwd+bwd':>10} {'micro':>6} {'s/micro':>9} {'wait':>9}")
        for s in sorted(steps)[-n:]:
            r = steps[s]
            fb = r["time/forward_backward"]
            mb = counts.get(s)
            per = f"{fb / mb:8.1f}s" if mb else "        -"
            print(f"{s:>5} {fb / 60:9.1f}m {mb if mb else '-':>6} {per} {r['time/wait_for_batch']:8.0f}s")

    mbs = recent_microbatches()
    if mbs:
        walls = [w for w, _ in mbs]
        a2as = [a for _, a in mbs]
        print(f"\nlast {len(mbs)} micro-batches: wall {min(walls):.0f}-{max(walls):.0f}s "
              f"(mean {sum(walls)/len(walls):.0f}s), a2a {sum(a2as)/len(a2as):.0f}s "
              f"= {100*sum(a2as)/sum(walls):.0f}% of wall")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--run", default="r37")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    while True:
        report(a.run, a.n)
        if not a.watch:
            return
        print("-" * 60, flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
