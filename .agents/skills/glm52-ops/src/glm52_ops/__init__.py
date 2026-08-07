"""Operations for the GLM-5.2 decensor GRPO run.

Read `SKILL.md` before you use the write functions. The read functions
(`status`, `reward`, `tunnels`, `engines`, `alive`) do not change the run.

Every function prints a report and also returns the data, so the caller can
either read the text or compute over the result.
"""

from __future__ import annotations

import datetime as _dt
import glob as _glob
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- constants

HEAD = "n82"
TRAINER_NODES = ["n76", "n77", "n78", "n80", "n82", "n83", "n84", "n85"]
ENGINE_HOSTS = ["h200a", "h200b"]

# n79 belongs to someone else. n81 is healthy but holds no authorized key, so
# it is not in the job. Adding it means changing the pipeline layout as well.
FORBIDDEN_HOSTS = {"n79"}

TUNNEL_PORTS = list(range(18001, 18025))
JUDGE_PORT = 26406
ADAPTER_PORT = 8100
BATCH_PORT = 29620

ENGINE_MODEL = "/data/glm52-rl/model-fp8"
DEPLOY = "/data/glm52-rl/deployment"
RUNS = "/data/glm52-rl/runs"
CLUSTER = f"{DEPLOY}/scripts/china/cluster.sh"

TRAINER_CT = "glm52-decensor-trainer"
ORCH_CT = "glm52-decensor-orchestrator"
ENGINE_CT = "glm52-decensor-h200"

# Run x22 gave 36 steps at batch 256: mean reward 0.1205, standard deviation
# 0.039 across steps. That is the noise floor a new batch size must beat.
# Do not compare against x21 or earlier; the adapter never reached the engines
# in those runs, so every rollout came from the base model.
BASELINE_STD = 0.039
BASELINE_RUN = "x22"

# A slope needs this many steps before it means anything, and it must clear
# twice the residual spread. A 25-step window once showed "+26 percent"; at 36
# steps the same run showed +6.3 percent against a standard deviation of 0.039.
MIN_STEPS_FOR_TREND = 10
TREND_SIGMA = 2.0

# Nothing in this run should sit still this long.
STALL_SECONDS = 600

# .../<repo>/.agents/skills/glm52-ops/src/glm52_ops/__init__.py
REPO = Path(__file__).resolve().parents[5]

_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "-o", "StrictHostKeyChecking=accept-new"]


class Refused(RuntimeError):
    """Raised when a call would break one of the rules in SKILL.md."""


# ------------------------------------------------------------------ plumbing


def _check_host(host: str) -> str:
    for bad in FORBIDDEN_HOSTS:
        if re.search(rf"\b{bad}\b", host):
            raise Refused(f"{bad} is off limits. The owner said so with no exception.")
    return host


def _ssh(host: str, script: str, timeout: int = 120) -> tuple[int, str]:
    _check_host(host)
    try:
        p = subprocess.run(_SSH + [host, script], capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _fan(hosts, fn, workers: int | None = None) -> dict:
    hosts = [_check_host(h) for h in hosts]
    with ThreadPoolExecutor(max_workers=workers or max(1, len(hosts))) as ex:
        return dict(zip(hosts, ex.map(fn, hosts)))


def _no_self_match(pattern: str) -> str:
    """Bracket the first character so a probe cannot match its own command line.

    `ssh host "pgrep -f foo"` matches the remote shell running `pgrep -f foo`,
    because the pattern sits in that command line. The count is then never
    zero and every liveness test passes. This is the same family as `pgrep
    -fc`, which additionally counts its own grep, and it hid three dead
    watchdogs for 3 hours.

    `[f]oo` still matches the string `foo` but no longer matches itself.

    Bracket the LAST token, not the first. ssh runs the probe under `bash -c`,
    so a pattern like `bash .*watchdog.sh` matches its own command line through
    the real leading `bash `, whatever is done to that word. The distinctive
    part is the script name at the end.
    """
    head, sep, tail = pattern.rpartition(" ")
    for i, ch in enumerate(tail):
        if ch.isalnum():
            return f"{head}{sep}{tail[:i]}[{ch}]{tail[i + 1:]}"
    return pattern


def _kv(out: str) -> dict[str, str]:
    """Parse `KEY value` report lines into a dict. Later lines win."""
    d: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].isupper() and parts[0].replace("_", "").isalpha():
            d[parts[0]] = parts[1].strip()
        elif len(parts) == 1 and parts[0].isupper() and parts[0]:
            d[parts[0]] = ""
    return d


def _age(rfc3339: str) -> float | None:
    """Seconds since a docker `-t` timestamp, or None if it will not parse."""
    try:
        t = rfc3339.strip().replace("Z", "+00:00")
        t = re.sub(r"\.(\d{6})\d+", r".\1", t)  # docker prints 9 fractional digits
        return (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(t)).total_seconds()
    except Exception:
        return None


def _hms(sec: float | None) -> str:
    if sec is None:
        return "?"
    sec = int(sec)
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


def _run_dir(run: str) -> str:
    """Remote shell fragment that resolves the run directory."""
    return (
        f'd=$(ls -d {RUNS}/{run}/*/ 2>/dev/null | while read x; do '
        f'[ -f "$x/rollout-metrics.jsonl" ] && echo "$x"; done | head -1)'
    )


# -------------------------------------------------------------------- probes

_HEAD_PROBE = r"""
ok=0; bad=""
for p in $(seq 18001 18024); do
  c=$(curl -s -o /dev/null -m 5 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null)
  if [ "$c" = "200" ]; then ok=$((ok+1)); else bad="$bad $p"; fi
done
echo "TUNNELS_OK $ok"
echo "TUNNELS_BAD $bad"
strip() { sed "s/\x1b\[[0-9;]*m//g"; }
echo "TRAINER_UP $(sudo docker ps --filter name=__TRAINER__ --format up 2>/dev/null | head -1)"
echo "ORCH_UP $(sudo docker ps --filter name=__ORCH__ --format up 2>/dev/null | head -1)"
echo "TRAINER_STEPS $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -ac 'SUCCESS Step')"
echo "TRAINER_LAST $(sudo docker logs __TRAINER__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "ORCH_LAST $(sudo docker logs __ORCH__ 2>&1 | strip | grep -a 'SUCCESS Step' | tail -1)"
echo "BATCH $(sudo docker logs --tail 40 __ORCH__ 2>&1 | grep -oE 'batch [0-9]+/[0-9]+[^;]*; [0-9]+ inflight' | tail -1)"
echo "FAILS5M $(sudo docker logs --since 5m __ORCH__ 2>&1 | grep -ac 'Rollout failed')"
echo "TRAINER_LOGAGE $(sudo docker logs -t --tail 1 __TRAINER__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "ORCH_LOGAGE $(sudo docker logs -t --tail 1 __ORCH__ 2>&1 | tail -1 | cut -d' ' -f1)"
echo "JUDGE $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__JUDGE__/v1/models)"
echo "ADAPTER $(curl -s -o /dev/null -m 10 -w '%{http_code}' http://127.0.0.1:__ADAPTER__/)"
echo "SUP_RUN $(pgrep -f 'bash .*[r]un_supervisor.sh' | wc -l)"
echo "SUP_TUN $(pgrep -f 'bash .*[t]unnel_watchdog.sh' | wc -l)"
echo "SUP_TUN_LOG $(tail -1 /data/glm52-rl/tunnel_watchdog.log 2>/dev/null | cut -d' ' -f1)"
"""

_NODE_PROBE = r"""
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
  | awk -F, '{m+=$1; u+=$2; n++} END{printf "GPU %d %d %d\n", m/n, u/n, n}'
free -g | sed -n 2p | awk '{print "MEM " $7 " " $5}'
echo "CT $(sudo docker ps --filter name=__TRAINER__ --format '{{.Status}}' 2>/dev/null | head -1)"
"""

_ENGINE_PROBE = r"""
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
  | awk '{u+=$1; n++} END{printf "GPU %d %d\n", u/n, n}'
s=$(date +%s%N)
c=$(curl -s -o /dev/null -m 25 -w '%{http_code}' -X POST http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"__MODEL__","prompt":"hi","max_tokens":3}')
e=$(date +%s%N)
echo "GEN $c $(( (e-s)/1000000 ))"
echo "RUNNING $(sudo docker logs --tail 20 __ENGINE__ 2>&1 | grep -aoE 'Running: [0-9]+' | tail -1 | grep -oE '[0-9]+')"
echo "TPUT $(sudo docker logs --tail 20 __ENGINE__ 2>&1 | grep -aoE 'generation throughput: [0-9.]+' | tail -1 | grep -oE '[0-9.]+')"
echo "WD $(pgrep -f 'bash .*[e]ngine_watchdog.sh' | wc -l)"
"""


def _head_script() -> str:
    return (
        _HEAD_PROBE.replace("__TRAINER__", TRAINER_CT)
        .replace("__ORCH__", ORCH_CT)
        .replace("__JUDGE__", str(JUDGE_PORT))
        .replace("__ADAPTER__", str(ADAPTER_PORT))
    )


# ------------------------------------------------------------------- results


@dataclass
class NodeState:
    host: str
    gpu_mem_mb: int = 0
    gpu_util: int = 0
    gpus: int = 0
    mem_free_gb: int = 0
    mem_shared_gb: int = 0
    container: str = ""
    reachable: bool = True


@dataclass
class EngineState:
    host: str
    gen_code: str = "000"
    gen_ms: int = 0
    gpu_util: int = 0
    running: int = 0
    throughput: float = 0.0
    watchdog: bool = False
    reachable: bool = True

    @property
    def healthy(self) -> bool:
        # `/v1/models` returning 200 proves only that the API server answers.
        # On 2026-08-06 h200a served 200 there while its EngineCore was dead,
        # with 64 requests queued and zero tokens per second. Only a real
        # completion is evidence.
        return self.gen_code == "200"


@dataclass
class Status:
    when: str = ""
    run: str = ""
    tunnels_ok: int = 0
    tunnels_bad: list[int] = field(default_factory=list)
    trainer_up: bool = False
    orch_up: bool = False
    trainer_steps: int = 0
    trainer_last: str = ""
    orch_last: str = ""
    batch: str = ""
    fails_5m: int = 0
    trainer_log_age: float | None = None
    orch_log_age: float | None = None
    judge: str = "000"
    adapter: str = "000"
    supervisor_run: bool = False
    supervisor_tunnel: bool = False
    supervisor_tunnel_log_age: float | None = None
    nodes: list[NodeState] = field(default_factory=list)
    engines: list[EngineState] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        L = [f"=== {self.when} UTC  run={self.run} ==="]
        L.append(
            f"trainer {'up' if self.trainer_up else 'DOWN'} | "
            f"orchestrator {'up' if self.orch_up else 'DOWN'} | "
            f"tunnels {self.tunnels_ok}/24 | judge {self.judge} | adapter {self.adapter}"
        )
        L.append(f"steps {self.trainer_steps} | fails/5m {self.fails_5m} | {self.batch or 'no batch line'}")
        if self.trainer_last:
            L.append(f"  trainer: {self.trainer_last[:150]}")
        if self.orch_last:
            L.append(f"  orch:    {self.orch_last[:150]}")
        L.append(
            f"  last log line: trainer {_hms(self.trainer_log_age)} ago, "
            f"orchestrator {_hms(self.orch_log_age)} ago"
        )
        L.append(
            f"  supervisors: run={'yes' if self.supervisor_run else 'NO'} "
            f"tunnel={'yes' if self.supervisor_tunnel else 'NO'} "
            f"(tunnel log {_hms(self.supervisor_tunnel_log_age)} old)"
        )
        L.append("--- trainer nodes " + "-" * 46)
        for n in self.nodes:
            if not n.reachable:
                L.append(f"{n.host:<5} UNREACHABLE")
                continue
            L.append(
                f"{n.host:<5} {n.gpu_mem_mb / 1024:5.1f} GiB/gpu  {n.gpu_util:3d}% util  "
                f"{n.mem_free_gb:4d}G free  {n.mem_shared_gb:3d}G shared  {n.container[:28]}"
            )
        L.append("--- engines " + "-" * 52)
        for e in self.engines:
            if not e.reachable:
                L.append(f"{e.host:<6} UNREACHABLE")
                continue
            L.append(
                f"{e.host:<6} gen={e.gen_code} in {e.gen_ms:>6d} ms  {e.gpu_util:3d}% util  "
                f"running={e.running:<4d} {e.throughput:8.1f} tok/s  "
                f"watchdog={'yes' if e.watchdog else 'NO'}"
            )
        for w in self.warnings:
            L.append(f"!! {w}")
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


# --------------------------------------------------------------------- reads


def status(run: str = "", quiet: bool = False) -> Status:
    """Probe every part of the run once and return the whole picture."""
    run = run or current_run()
    s = Status(when=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), run=run)

    def node(h):
        return _ssh(h, _NODE_PROBE.replace("__TRAINER__", TRAINER_CT), timeout=60)

    def engine(h):
        return _ssh(h, _ENGINE_PROBE.replace("__MODEL__", ENGINE_MODEL).replace("__ENGINE__", ENGINE_CT), timeout=90)

    with ThreadPoolExecutor(max_workers=12) as ex:
        f_head = ex.submit(_ssh, HEAD, _head_script(), 300)
        f_nodes = ex.submit(_fan, TRAINER_NODES, node)
        f_engines = ex.submit(_fan, ENGINE_HOSTS, engine)
        head_rc, head_out = f_head.result()
        nodes_out = f_nodes.result()
        engines_out = f_engines.result()

    h = _kv(head_out)
    s.tunnels_ok = int(h.get("TUNNELS_OK", 0) or 0)
    s.tunnels_bad = [int(p) for p in h.get("TUNNELS_BAD", "").split()]
    s.trainer_up = h.get("TRAINER_UP", "") == "up"
    s.orch_up = h.get("ORCH_UP", "") == "up"
    s.trainer_steps = int(h.get("TRAINER_STEPS", 0) or 0)
    s.trainer_last = h.get("TRAINER_LAST", "")
    s.orch_last = h.get("ORCH_LAST", "")
    s.batch = h.get("BATCH", "")
    s.fails_5m = int(h.get("FAILS5M", 0) or 0)
    s.trainer_log_age = _age(h.get("TRAINER_LOGAGE", ""))
    s.orch_log_age = _age(h.get("ORCH_LOGAGE", ""))
    s.judge = h.get("JUDGE", "000")
    s.adapter = h.get("ADAPTER", "000")
    s.supervisor_run = int(h.get("SUP_RUN", 0) or 0) > 0
    s.supervisor_tunnel = int(h.get("SUP_TUN", 0) or 0) > 0
    s.supervisor_tunnel_log_age = _age(h.get("SUP_TUN_LOG", ""))

    for host, (rc, out) in nodes_out.items():
        n = NodeState(host=host, reachable=(rc == 0))
        for line in out.splitlines():
            p = line.split()
            if p[:1] == ["GPU"] and len(p) >= 4:
                n.gpu_mem_mb, n.gpu_util, n.gpus = int(p[1]), int(p[2]), int(p[3])
            elif p[:1] == ["MEM"] and len(p) >= 3:
                n.mem_free_gb, n.mem_shared_gb = int(p[1]), int(p[2])
            elif p[:1] == ["CT"]:
                n.container = line[3:].strip()
        s.nodes.append(n)

    for host, (rc, out) in engines_out.items():
        e = EngineState(host=host, reachable=(rc == 0))
        for line in out.splitlines():
            p = line.split()
            if p[:1] == ["GPU"] and len(p) >= 2:
                e.gpu_util = int(p[1])
            elif p[:1] == ["GEN"] and len(p) >= 3:
                e.gen_code, e.gen_ms = p[1], int(p[2])
            elif p[:1] == ["RUNNING"] and len(p) >= 2:
                e.running = int(p[1])
            elif p[:1] == ["TPUT"] and len(p) >= 2:
                e.throughput = float(p[1])
            elif p[:1] == ["WD"] and len(p) >= 2:
                e.watchdog = int(p[1]) > 0
        s.engines.append(e)

    s.warnings = _warnings(s)
    if not quiet:
        print(s.report())
    return s


def _warnings(s: Status) -> list[str]:
    w = []
    if not s.trainer_up:
        w.append("trainer container is not up")
    if not s.orch_up:
        w.append("orchestrator container is not up")
    if s.tunnels_ok <= 10:
        # Do not read this as dead tunnels on its own. The probe shares each
        # tunnel with rollout traffic, so under load it starves on links that
        # still carry rollouts. Confirm against the engines before acting:
        # engines generating means the tunnels work and the probe is starved;
        # engines idle with many rollouts in flight means the requests are
        # stuck in the tunnels and max_inflight_rollouts is too high.
        w.append(
            f"only {s.tunnels_ok}/24 tunnels answer. Check the engines before you believe this. "
            "If they are generating, the probe is starved by load, not by dead tunnels."
        )
    elif s.tunnels_ok < 18:
        w.append(f"{s.tunnels_ok}/24 tunnels answer. Compare against the engine concurrency below.")
    # Starvation means the engines are producing nothing while many rollouts are
    # in flight. A low `running` count on its own is not evidence: a rollout in
    # flight may be at the judge or in the environment rather than generating,
    # and `running` is a single sample from the last log line. Require the
    # throughput to be near zero as well, and confirm with several samples
    # before acting on it.
    inflight = re.search(r"(\d+) inflight", s.batch or "")
    live = [e for e in s.engines if e.reachable]
    if inflight and int(inflight.group(1)) > 40 and live:
        tput = sum(e.throughput for e in live)
        if tput < 5.0 and sum(e.running for e in live) == 0:
            w.append(
                f"{inflight.group(1)} rollouts in flight but the engines generate {tput:.1f} tok/s "
                "and run nothing. Sample again before acting. If it holds, the requests are queued "
                "in the tunnels; lower max_inflight_rollouts, which is independent of batch_size."
            )
    for e in s.engines:
        if e.reachable and not e.healthy:
            w.append(f"{e.host} failed a real completion probe (code {e.gen_code}). Run repair('engine:{e.host}').")
        if e.reachable and not e.watchdog:
            w.append(f"{e.host} has no engine watchdog running")
    if s.judge != "200":
        w.append(f"judge returned {s.judge}; every reward multiplies by the judge score")
    if s.adapter != "200":
        w.append(
            f"adapter server returned {s.adapter} on port {ADAPTER_PORT}. "
            "If the engines cannot fetch the LoRA, they keep serving the base model and nothing learns."
        )
    if not s.supervisor_run:
        w.append("run supervisor is not running; a dead trainer will stay dead")
    if not s.supervisor_tunnel:
        w.append("tunnel watchdog is not running")
    elif s.supervisor_tunnel_log_age and s.supervisor_tunnel_log_age > 300:
        w.append(
            f"tunnel watchdog process exists but its log is {_hms(s.supervisor_tunnel_log_age)} old; "
            "a process that writes no log line does not work"
        )
    for name, age in (("trainer", s.trainer_log_age), ("orchestrator", s.orch_log_age)):
        if age is not None and age > STALL_SECONDS:
            w.append(f"{name} has written nothing for {_hms(age)}; the limit is {STALL_SECONDS // 60} minutes")
    for n in s.nodes:
        if n.reachable and n.mem_free_gb < 60:
            w.append(
                f"{n.host} has only {n.mem_free_gb}G of host memory free. "
                'Check that ac_offloading = "None" is set; the default fills pinned host memory.'
            )
    return w


def current_run() -> str:
    """The run name the trainer container was started with."""
    rc, out = _ssh(HEAD, f"sudo docker inspect {TRAINER_CT} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}' 2>/dev/null | grep -E '^RUN='")
    m = re.search(r"^RUN=(\S+)", out, re.M)
    if m:
        return m.group(1)
    rc, out = _ssh(HEAD, f"ls -1t {RUNS} | head -1")
    return out.strip().splitlines()[0] if out.strip() else "unknown"


def tunnels() -> dict[int, str]:
    """HTTP code for each of the 24 rollout tunnel ports."""
    rc, out = _ssh(
        HEAD,
        'for p in $(seq 18001 18024); do printf "%s " $p; '
        'curl -s -o /dev/null -m 6 -w "%{http_code}\\n" http://127.0.0.1:$p/v1/models 2>/dev/null || echo 000; done',
        timeout=240,
    )
    res = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit():
            res[int(p[0])] = p[1]
    ok = sum(1 for v in res.values() if v == "200")
    print(f"tunnels {ok}/{len(res)} answering; bad: {[p for p, v in res.items() if v != '200']}")
    return res


def engines() -> dict[str, EngineState]:
    """Probe both engines with a real 3-token completion."""
    s = status(quiet=True)
    for e in s.engines:
        print(f"{e.host}: gen={e.gen_code} in {e.gen_ms} ms, running={e.running}, {e.throughput} tok/s")
    return {e.host: e for e in s.engines}


def alive(pattern: str, host: str = HEAD, log: str = "") -> bool:
    """Test that a process runs. Never use `pgrep -fc`; it counts the grep.

    A process that runs but writes no new log line does not work, so pass
    `log` when the process has one.
    """
    rc, out = _ssh(host, f"pgrep -f {json.dumps(_no_self_match(pattern))} | wc -l; " + (f"tail -1 {log} 2>/dev/null" if log else ""))
    lines = out.strip().splitlines()
    n = int(lines[0]) if lines and lines[0].strip().isdigit() else 0
    running = n > 0
    print(f"{host}: {pattern!r} -> {n} process(es)")
    if log and len(lines) > 1:
        age = _age(lines[1].split(" ")[0])
        print(f"  last log line {_hms(age)} old")
        if age is not None and age > 300:
            print("  the process exists but its log is stale; treat it as dead")
            return False
    return running


# ------------------------------------------------------------ reward reading


@dataclass
class RewardReport:
    run: str = ""
    steps: list[int] = field(default_factory=list)
    effective: list[float] = field(default_factory=list)
    all_scope: list[float] = field(default_factory=list)
    p10: list[float] = field(default_factory=list)
    p90: list[float] = field(default_factory=list)
    error_rate: list[float] = field(default_factory=list)
    step_seconds: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    slope: float = 0.0
    total_change: float = 0.0
    residual_std: float = 0.0
    verdict: str = "too few steps"
    note: str = ""

    def report(self) -> str:
        L = [f"=== reward for run {self.run} ({len(self.steps)} steps) ==="]
        L.append(f"{'step':>5} {'effective':>10} {'all':>8} {'p10':>8} {'p90':>8} {'error':>7} {'time':>8}")
        for i, st in enumerate(self.steps):
            L.append(
                f"{st:>5} {self.effective[i]:10.4f} {self.all_scope[i]:8.4f} "
                f"{self.p10[i]:8.4f} {self.p90[i]:8.4f} "
                f"{self.error_rate[i] * 100:6.1f}% {self.step_seconds[i] / 60:7.1f}m"
            )
        L.append("")
        L.append(f"effective reward: mean {self.mean:.4f}, standard deviation {self.std:.4f} across steps")
        L.append(f"baseline {BASELINE_RUN}: standard deviation {BASELINE_STD:.4f} at batch 256 over 36 steps")
        L.append(
            f"fit: {self.total_change:+.4f} over the window, residual spread {self.residual_std:.4f}, "
            f"threshold {TREND_SIGMA:g} x residual = {TREND_SIGMA * self.residual_std:.4f}"
        )
        L.append(f"VERDICT: {self.verdict}")
        if self.note:
            L.append(self.note)
        return "\n".join(L)

    def __str__(self) -> str:
        return self.report()


def reward(run: str = "", quiet: bool = False) -> RewardReport:
    """Read the reward series and say whether it moved or only wandered.

    The verdict is deliberately conservative. A window shorter than
    MIN_STEPS_FOR_TREND has no verdict, and a change smaller than twice the
    residual spread is called noise. Report the verdict as it stands.
    """
    run = run or current_run()
    script = (
        _run_dir(run)
        + '\n[ -z "$d" ] && { echo NORUN; exit 0; }\n'
        + "python3 - \"$d\" <<'PY'\n"
        "import json,sys\n"
        "K={'eff':'train/agg/effective/reward/mean','all':'train/agg/all/reward/mean',\n"
        "   'p10':'train/agg/effective/reward/p10','p90':'train/agg/effective/reward/p90',\n"
        "   'err':'train/agg/all/has_error/mean','t':'time/step'}\n"
        "out=[]\n"
        "for line in open(sys.argv[1].rstrip('/')+'/rollout-metrics.jsonl'):\n"
        "    line=line.strip()\n"
        "    if not line: continue\n"
        "    try: r=json.loads(line)\n"
        "    except ValueError: continue\n"
        "    if 'step' not in r: continue\n"
        "    out.append({'step':r['step'],**{k:r.get(v) for k,v in K.items()}})\n"
        "print(json.dumps(out))\n"
        "PY\n"
    )
    rc, out = _ssh(HEAD, script, timeout=180)
    rep = RewardReport(run=run)
    blob = ""
    for line in out.splitlines():
        if line.startswith("["):
            blob = line
    if not blob:
        rep.verdict = "no data"
        rep.note = f"no rollout-metrics.jsonl found under {RUNS}/{run}/"
        if not quiet:
            print(rep.report())
        return rep

    rows = [r for r in json.loads(blob) if r.get("eff") is not None]
    rows.sort(key=lambda r: r["step"])
    rep.steps = [r["step"] for r in rows]
    rep.effective = [float(r["eff"]) for r in rows]
    rep.all_scope = [float(r.get("all") or 0.0) for r in rows]
    rep.p10 = [float(r.get("p10") or 0.0) for r in rows]
    rep.p90 = [float(r.get("p90") or 0.0) for r in rows]
    rep.error_rate = [float(r.get("err") or 0.0) for r in rows]
    rep.step_seconds = [float(r.get("t") or 0.0) for r in rows]

    y = rep.effective
    n = len(y)
    if n:
        rep.mean = statistics.fmean(y)
        rep.std = statistics.stdev(y) if n > 1 else 0.0
    if n >= 3:
        xs = list(range(n))
        xbar, ybar = statistics.fmean(xs), rep.mean
        denom = sum((x - xbar) ** 2 for x in xs)
        rep.slope = sum((x - xbar) * (v - ybar) for x, v in zip(xs, y)) / denom if denom else 0.0
        rep.total_change = rep.slope * (n - 1)
        resid = [v - (ybar + rep.slope * (x - xbar)) for x, v in zip(xs, y)]
        rep.residual_std = statistics.stdev(resid) if n > 2 else 0.0

    if n < MIN_STEPS_FOR_TREND:
        rep.verdict = f"too few steps ({n} of {MIN_STEPS_FOR_TREND} needed)"
    elif abs(rep.total_change) < TREND_SIGMA * rep.residual_std:
        rep.verdict = "inside the noise"
    else:
        rep.verdict = "rising" if rep.total_change > 0 else "falling"

    rep.note = _reward_note(rep)
    if not quiet:
        print(rep.report())
    return rep


def _reward_note(r: RewardReport) -> str:
    notes = []
    if len(r.steps) >= 2:
        d_err = r.error_rate[0] - r.error_rate[-1]
        d_all = r.all_scope[-1] - r.all_scope[0]
        d_eff = r.effective[-1] - r.effective[0]
        if d_err > 0.05 and d_all > d_eff:
            notes.append(
                f"The error rate fell {d_err * 100:.1f} points over this window. A failed rollout scores zero, "
                f"so the `all` series rose {d_all:+.4f} while `effective` rose only {d_eff:+.4f}. "
                "The difference is transport recovery, not learning. Quote `effective`."
            )
        if r.p90[-1] > r.p90[0] * 1.5 and r.p10[-1] < 0.01:
            notes.append(
                f"The 90th percentile moved {r.p90[0]:.4f} -> {r.p90[-1]:.4f} while the 10th percentile stayed at "
                f"{r.p10[-1]:.4f}. The reward is a product of five factors, so the bottom stays pinned at zero. "
                "The distribution is widening at the top even when the mean is flat."
            )
    if len(r.steps) >= MIN_STEPS_FOR_TREND and r.std:
        cmp = "below" if r.std < BASELINE_STD else "above"
        notes.append(
            f"Step-to-step spread is {r.std:.4f}, {cmp} the {BASELINE_RUN} baseline of {BASELINE_STD:.4f}. "
            "That comparison is the reason the batch size was raised."
        )
    return "\n".join(notes)


# ----------------------------------------------------------- active watching


def watch(hours: float = 2.0, every: int = 300, run: str = "") -> list[Status]:
    """Print the full picture every `every` seconds until `hours` elapse.

    This blocks the kernel on purpose. The owner forbade background monitor
    tools and asked for a foreground command with a sleep in it. Do not move
    this into a thread and do not run it in a background cell.

    It also compares each cycle against the last one and says when the batch
    counter or the step counter has not moved for STALL_SECONDS.
    """
    deadline = time.time() + hours * 3600
    history: list[Status] = []
    last_change = time.time()
    last_key = None
    while time.time() < deadline:
        s = status(run=run, quiet=True)
        history.append(s)
        print(s.report(), flush=True)
        key = (s.trainer_steps, s.batch)
        if key != last_key:
            last_key, last_change = key, time.time()
        else:
            still = time.time() - last_change
            if still > STALL_SECONDS:
                print(
                    f"!! STALLED: step count and batch progress have not changed for {_hms(still)}. "
                    f"Read references/FAULTS.md and repair before waiting further.",
                    flush=True,
                )
        remaining = min(every, max(0, deadline - time.time()))
        if remaining <= 0:
            break
        print(f"--- sleeping {int(remaining)}s ---", flush=True)
        time.sleep(remaining)
    return history


# -------------------------------------------------------------------- writes


def repair(what: str) -> str:
    """Apply the one correct repair for a named fault.

    what = "tunnels" | "engine:h200a" | "engine:h200b" | "supervisors"
    """
    if what == "tunnels":
        return _repair_tunnels()
    if what.startswith("engine:"):
        return _repair_engine(what.split(":", 1)[1])
    if what == "supervisors":
        return _repair_supervisors()
    raise Refused(f"unknown repair {what!r}. Use tunnels, engine:<host>, or supervisors.")


def _repair_tunnels() -> str:
    # Always rebuild. An earlier watchdog refused to act when 80 percent of the
    # tunnels failed at once, on the theory that mass failure meant a dead
    # backend. At batch 1024 all 24 wedged with healthy engines and the run sat
    # idle five hours. Rebuilding against a dead backend just fails again next
    # cycle; refusing to rebuild against a live one stops everything.
    rc, out = _ssh(
        HEAD,
        'k=0; for p in $(seq 18001 18024); do '
        'c=$(curl -s -o /dev/null -m 6 -w "%{http_code}" http://127.0.0.1:$p/v1/models 2>/dev/null); '
        'if [ "$c" != "200" ]; then '
        'pid=$(sudo ss -H -lntp "sport = :$p" | sed -n "s/.*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1); '
        '[ -n "$pid" ] && sudo kill $pid && k=$((k+1)); fi; done; echo "killed $k"',
        timeout=300,
    )
    msg = out.strip().splitlines()[-1] if out.strip() else "no output"
    print(f"tunnels: {msg}. The H200 supervisors rebuild within about 5 seconds.")
    print(f"The batch tunnel on port {BATCH_PORT} was not touched.")
    return msg


def _repair_engine(host: str) -> str:
    _check_host(host)
    if host not in ENGINE_HOSTS:
        raise Refused(f"{host} is not an engine host. Use one of {ENGINE_HOSTS}.")
    print(f"{host}: removing and restarting {ENGINE_CT}. The model load needs 5 to 10 minutes.")
    rc, out = _ssh(
        host,
        f"sudo docker rm -f {ENGINE_CT} >/dev/null 2>&1; "
        f"nohup setsid bash {DEPLOY}/scripts/h200/start_inference.sh >/root/start_inference.log 2>&1 </dev/null & "
        "sleep 3; echo issued",
        timeout=180,
    )
    print("Do not test with /v1/models. Test with engines() when the load finishes.")
    return out.strip()


def _repair_supervisors() -> str:
    out = []
    for pat, cmd, log in (
        (
            "bash .*tunnel_watchdog.sh",
            f"nohup setsid env TIMEOUT=20 INTERVAL=45 bash {DEPLOY}/scripts/china/tunnel_watchdog.sh "
            ">> /data/glm52-rl/tunnel_watchdog.log 2>&1 </dev/null & disown",
            "/data/glm52-rl/tunnel_watchdog.log",
        ),
    ):
        if not alive(pat, HEAD, log):
            _ssh(HEAD, cmd)
            out.append(f"restarted {pat}")
        else:
            out.append(f"{pat} already runs")
    for h in ENGINE_HOSTS:
        if not alive("bash .*engine_watchdog.sh", h):
            _ssh(h, f"nohup setsid bash {DEPLOY}/scripts/h200/engine_watchdog.sh >> /root/engine_watchdog.log 2>&1 </dev/null & disown")
            out.append(f"restarted engine watchdog on {h}")
    msg = "; ".join(out)
    print(msg)
    print("The run supervisor needs RUN, RUN_NAME, and CONFIG_TAG. Start it with start(run, mode='restart').")
    return msg


def mounted_files() -> list[str]:
    """The source paths that cluster.sh mounts into the trainer container.

    A change to any other file has no effect: the container keeps using the
    copy baked into the image.
    """
    sh = (REPO / "scripts" / "china" / "cluster.sh").read_text()
    return sorted(set(re.findall(r'-v "\$deployment/(\S+?)":', sh)))


def deploy(path: str, nodes: list[str] | None = None) -> bool:
    """Push one repository file to every trainer node and verify the checksum."""
    nodes = [_check_host(h) for h in (nodes or TRAINER_NODES)]
    local = REPO / path
    if not local.is_file():
        raise Refused(f"not in the repository: {local}")
    if path.startswith("source/") and path not in mounted_files():
        raise Refused(
            f"{path} is not mounted by cluster.sh, so the container would ignore it. "
            "Add a -v line to scripts/china/cluster.sh first, or pick a mounted path. "
            "Call mounted_files() to list them."
        )
    want = __import__("hashlib").md5(local.read_bytes()).hexdigest()
    remote = f"{DEPLOY}/{path}"

    def one(h):
        subprocess.run(_SSH + [h, f"mkdir -p {str(Path(remote).parent)}"], capture_output=True)
        p = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(local), f"{h}:{remote}"], capture_output=True, text=True)
        if p.returncode:
            return "scp failed"
        rc, got = _ssh(h, f"md5sum {remote} | cut -d' ' -f1")
        return "ok" if got.strip() == want else f"MISMATCH {got.strip()[:8]}"

    res = _fan(nodes, one)
    bad = [h for h, v in res.items() if v != "ok"]
    print(f"{path} -> {len(nodes) - len(bad)}/{len(nodes)} nodes" + (f", FAILED {bad}" if bad else ""))
    if not bad:
        if path.startswith("source/") or path.endswith("cluster.sh"):
            # Mounted into the container at launch, so a running container keeps
            # the old copy until it is recreated.
            print("The running container still holds the old file. Restart the trainer for this to take effect.")
        else:
            # Host-side scripts are read at each launch, so only long-lived
            # loops (the watchdogs, the supervisor) need restarting.
            print("This runs on the host. Restart the process only if it is a long-running loop.")
    return not bad


def start(run: str, mode: str = "cold", nodes: list[str] | None = None, supervisors: bool = True) -> None:
    """Start a run. `mode` is "cold" for a new run name or "restart" for an existing one.

    The two modes start the trainer and the orchestrator in opposite order, and
    both orders are correct. See SKILL.md.
    """
    if mode not in ("cold", "restart"):
        raise Refused('mode must be "cold" or "restart"')
    nodes = [_check_host(h) for h in (nodes or TRAINER_NODES)]
    if HEAD in nodes:  # rank 0 is the rendezvous master and carries the readable log
        nodes = [HEAD] + [n for n in nodes if n != HEAD]
    name, tag = f"run_{run}", f".{run}"
    trainer_nodes = " ".join(n.lstrip("n") for n in nodes)
    # PRIME_BATCH_CACHE is never forwarded. A value leaking in from the shell
    # would replay one frozen batch for every step while the run looked healthy.
    env = f"TRAINER_NODES='{trainer_nodes}' RUN={run} RUN_NAME={name} CONFIG_TAG={tag} PRIME_EP_PROFILE=1 PRIME_EP_AC_CACHE=1"

    def launch_trainer():
        print(f"starting trainer on {len(nodes)} nodes")
        _fan(nodes, lambda h: _ssh(h, f"{env} bash {CLUSTER} trainer >/dev/null 2>&1; echo done", 300))

    def launch_orch():
        print("starting orchestrator on n82")
        _ssh(HEAD, f"RUN={run} RUN_NAME={name} CONFIG_TAG={tag} bash {CLUSTER} orchestrator >/dev/null 2>&1; echo done", 300)

    print(f"stopping any previous containers for {run}")
    _ssh(HEAD, f"sudo docker rm -f {ORCH_CT} >/dev/null 2>&1; echo done")
    _fan(nodes, lambda h: _ssh(h, f"sudo pkill -9 -f PRIME-RL 2>/dev/null; sudo docker rm -f {TRAINER_CT} >/dev/null 2>&1; echo done"))
    time.sleep(20)  # let the host mirror of device memory drain before reloading

    if mode == "cold":
        # Nothing has written control/orch.toml yet and the trainer blocks
        # until it exists, so the orchestrator has to go first.
        launch_orch()
        launch_trainer()
    else:
        # orch.toml is already on disk, so the trainer can come up alone. A zmq
        # send to a dead trainer blocks the orchestrator event loop, so the
        # trainer must be listening before the orchestrator starts.
        launch_trainer()
        time.sleep(60)
        launch_orch()

    if supervisors:
        _ssh(
            HEAD,
            f"nohup setsid env RUN={run} RUN_NAME={name} CONFIG_TAG={tag} INTERVAL=120 "
            f"bash {DEPLOY}/scripts/china/run_supervisor.sh >> /data/glm52-rl/run_supervisor.log 2>&1 </dev/null & disown",
        )
        _repair_supervisors()
    print(f"started {run} in {mode} mode. Call status() in a few minutes.")


def stop() -> None:
    """Stop the orchestrator, then the trainer. Supervisors are stopped first."""
    _ssh(HEAD, "pkill -f 'bash .*[r]un_supervisor.sh'; pkill -f 'bash .*[t]unnel_watchdog.sh'; echo done")
    _ssh(HEAD, f"sudo docker rm -f {ORCH_CT} >/dev/null 2>&1; echo done")
    _fan(TRAINER_NODES, lambda h: _ssh(h, f"sudo pkill -9 -f PRIME-RL 2>/dev/null; sudo docker rm -f {TRAINER_CT} >/dev/null 2>&1; echo done"))
    print("stopped supervisors, orchestrator, and trainer. The engines are still running.")


# ------------------------------------------------------------------ entry


_ACTIONS = {
    "status": lambda a: status(*a),
    "reward": lambda a: reward(*a),
    "tunnels": lambda a: tunnels(),
    "engines": lambda a: engines(),
    "watch": lambda a: watch(*[float(x) for x in a]),
    "repair": lambda a: repair(*a),
    "deploy": lambda a: deploy(*a),
    "stop": lambda a: stop(),
    "mounts": lambda a: print("\n".join(mounted_files())),
}


def run(argstr: str = "status"):
    """Dispatcher so `await glm52_ops("reward")` works in the kernel."""
    parts = argstr.split()
    if not parts or parts[0] not in _ACTIONS:
        raise Refused(f"use one of: {', '.join(sorted(_ACTIONS))}")
    return _ACTIONS[parts[0]](parts[1:])


_USAGE = f"""glm52-ops <action> [args]

  status [run]        the whole picture in one call
  reward [run]        the reward series with a trend verdict
  tunnels             the HTTP code for each of the 24 tunnel ports
  engines             a real completion probe for each engine
  watch [hours] [s]   the picture on a loop, in the foreground
  repair <what>       tunnels | engine:<host> | supervisors
  deploy <path>       push one repository file to every trainer node
  mounts              the source paths the container mounts
  stop                stop the supervisors, orchestrator and trainer

Read {Path(__file__).resolve().parents[2] / 'SKILL.md'} first.
"""


def cli() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(_USAGE)
        sys.exit(0)
    try:
        run(" ".join(args))
    except Refused as e:
        print(f"refused: {e}\n\n{_USAGE}", file=sys.stderr)
        sys.exit(2)
