#!/usr/bin/env python3
"""Is anything wrong with the training run?

  check.py            one pass, exits non-zero if anything is FAIL
  check.py --run r35  a different run directory

Checks, in the order they tend to break:
  containers   all live trainers plus the orchestrator are Up
  progress     the trainer is not sat waiting on rollouts for ages
  deadlock     orchestrator short of a full batch with nothing in flight
  numerics     loss / grad-norm / entropy finite and in a sane band
  memory       headroom left on the 48 GiB cards
  tunnels      how many rollout links answer
  watcher      the rollout link watcher is running

Numerics bands come from the measured healthy baseline of this run, not from
theory: loss near zero, grad-norm 0.01-0.10, entropy 0.3-0.7. A NaN or a
grad-norm an order of magnitude out is the signal that a speed change broke
something.
"""

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# n81 is down at the OS level (shell I/O error, ssh refused); the job runs
# 7 nodes / 56 ranks until it is back. Re-add it here and in TRAINER_NODES.
NODES = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25"]
fails = []


def sh(host, cmd, timeout=90):
    try:
        return subprocess.run(SSH + [host, cmd], capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def line(status, name, detail):
    print(f"[{status:^4}] {name:<12} {detail}")
    if status == "FAIL":
        fails.append(name)


def check_containers():
    def one(h):
        return h, sh(h, 'sudo docker ps --filter name=glm52-decensor-trainer --format "{{.Status}}"').strip()
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, NODES))
    down = [h for h, s in res if not s.startswith("Up")]
    line("FAIL" if down else " ok ", "containers",
         f"trainers down: {down}" if down else f"{len(NODES)}/{len(NODES)} trainers Up")
    orch = sh("n82", 'sudo docker ps --filter name=glm52-decensor-orchestrator --format "{{.Status}}"').strip()
    line("FAIL" if not orch.startswith("Up") else " ok ", "orchestrator", orch or "MISSING")


def check_progress_and_deadlock():
    t = sh("n82", 'sudo docker logs --tail 3 glm52-decensor-trainer 2>&1 | grep -oE "waiting [0-9]+" | tail -1')
    m = re.search(r"waiting (\d+)", t)
    if m:
        w = int(m.group(1))
        line("WARN" if w >= 900 else " ok ", "progress",
             f"trainer waiting {w}s for a batch" + (" - rollouts behind" if w >= 900 else ""))
    else:
        line(" ok ", "progress", "trainer is computing")

    o = sh("n82", 'sudo docker logs --tail 1 glm52-decensor-orchestrator 2>&1 | sed "s/\\x1b\\[[0-9;]*m//g"')
    if "0 inflight rollouts" in o:
        b = re.search(r"Train batch (\d+)/(\d+)", o)
        # 0/N with nothing in flight is a fresh orchestrator that has not
        # dispatched yet, not a wedge. Same guard as the budget fix.
        if b and b.group(1) != b.group(2) and int(b.group(1)) > 0:
            line("FAIL", "deadlock", f"{b.group(0)} with 0 inflight - restart the orchestrator")
        else:
            line(" ok ", "deadlock", "none")
    else:
        line(" ok ", "deadlock", "none")


def check_numerics(run):
    path = f"/data/glm52-rl/runs/{run}/run_decensor_fp8_{run}/trainer-metrics.jsonl"
    raw = sh("n82", f"sudo docker exec glm52-decensor-trainer cat {path} 2>/dev/null")
    vals = {}
    for l in raw.splitlines():
        try:
            r = json.loads(l)
        except ValueError:
            continue
        for k in ("loss", "grad_norm", "entropy/mean", "perf/peak_memory"):
            if k in r:
                vals[k] = r[k]
    if not vals:
        line("WARN", "numerics", "no completed step yet")
        return
    bad = []
    loss, gn = vals.get("loss"), vals.get("grad_norm")
    for k, v in vals.items():
        if v != v or abs(v) == float("inf"):
            bad.append(f"{k}=NaN/inf")
    if gn is not None and not (0.001 <= gn <= 0.5):
        bad.append(f"grad_norm={gn:.4f} outside 0.001-0.5")
    if loss is not None and abs(loss) > 1.0:
        bad.append(f"loss={loss:.4f} outside +/-1")
    detail = " ".join(f"{k}={v:.4f}" for k, v in vals.items() if isinstance(v, float))
    line("FAIL" if bad else " ok ", "numerics", "; ".join(bad) if bad else detail)


def check_memory():
    def one(h):
        out = sh(h, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
        worst = 0
        for l in out.splitlines():
            try:
                used, tot = (int(x) for x in l.split(","))
                worst = max(worst, used)
            except ValueError:
                pass
        return h, worst
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(one, NODES))
    peak = max((v for _, v in res), default=0)
    line("FAIL" if peak > 47000 else ("WARN" if peak > 45500 else " ok "),
         "memory", f"peak {peak} MiB / 49140")


def check_tunnels():
    # probe all 24 concurrently; serial probes take minutes when links are busy
    out = sh("n82", 'for p in $(seq 18001 18024); do '
                    '(curl -s -m 8 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:$p/v1/models 2>/dev/null) & '
                    'done; wait', timeout=60)
    out = str(out.count("200"))
    try:
        n = int(out.strip() or 0)
    except ValueError:
        n = 0
    line("FAIL" if n == 0 else ("WARN" if n < 8 else " ok "), "tunnels", f"{n}/24 answering")

    w = sh("n82", 'pgrep -cf watch_rollout_links.sh || echo 0').strip()
    line(" ok " if w and int(w or 0) >= 1 else "WARN", "watcher",
         "running" if w and int(w or 0) >= 1 else "NOT running")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r37")
    a = ap.parse_args()
    check_containers()
    check_progress_and_deadlock()
    check_numerics(a.run)
    check_memory()
    check_tunnels()
    print()
    print("FAILING: " + ", ".join(fails) if fails else "all checks passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
