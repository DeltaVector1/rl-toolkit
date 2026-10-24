"""Review 6: after the port's real setup, run `git fsck` to completion and record its exit status.

    uv run python review6_fsck_probe.py <DATA> <out.jsonl> [concurrency=96] [fsck timeout seconds=1500]
    PROBE_IDS="id id ..." restricts the run.

The review 4 and 5 probes counted `unreachable commit` lines from `timeout 120 git fsck`, skipped
repositories of 400 MB or more, and threw the exit status away, so a timed-out or failed fsck
looked the same as a clean one. This probe has no size limit and records, per image:
  fsck_rc            124 means the timeout was hit; 0 means fsck finished without errors
  unreachable_*      unreachable objects by type (commits are the ones that would leak history)
  extra_after        commits reachable from any ref but not from HEAD
  packs, loose       object store shape after `gc --prune=now`
Resumable: rows already recorded without an error are skipped. No verifier and no model runs.
"""
import asyncio, json, os, re, sys, time
from types import SimpleNamespace

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig

DATA, OUT = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 96
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
ONLY = set(os.environ.get("PROBE_IDS", "").split())
BASE = PrimeConfig(cpu=2, disk=20, labels=["xiaomi-review"])
FSCK = r"""
echo "git_version=$(git --version)"
echo "git_mb=$(du -sm "$(git rev-parse --git-dir)" | cut -f1)"
echo "head=$(git rev-parse HEAD 2>&1)"
echo "extra_after=$(git rev-list --all --not HEAD 2>/dev/null | wc -l)"
echo "refs=$(git for-each-ref | wc -l)"
echo "shallow=$(git rev-parse --is-shallow-repository 2>/dev/null)"
s=$(date +%s)
timeout --kill-after=10 LIMIT git fsck --connectivity-only --unreachable --no-reflogs >/tmp/_fsck.out 2>/tmp/_fsck.err
echo "fsck_rc=$?"
echo "fsck_seconds=$(( $(date +%s) - s ))"
echo "unreachable_commit=$(grep -c '^unreachable commit' /tmp/_fsck.out)"
echo "unreachable_total=$(grep -c '^unreachable' /tmp/_fsck.out)"
echo "unreachable_kinds=$(awk '/^unreachable/{print $2}' /tmp/_fsck.out | sort | uniq -c | awk '{printf "%s:%s,", $2, $1}')"
echo "fsck_other=$(grep -v -E '^(unreachable|dangling) ' /tmp/_fsck.out | head -3 | tr '\n' ' ' | head -c 240)"
echo "fsck_err=$(grep -v -E '^(Checking|Verifying) ' /tmp/_fsck.err | head -3 | tr '\n' ' ' | head -c 240)"
echo "loose=$(git count-objects -v | awk '/^count:/{print $2}')"
echo "packs=$(git count-objects -v | awk '/^packs:/{print $2}')"
echo "garbage=$(git count-objects -v | awk '/^garbage:/{print $2}')"
""".replace("LIMIT", str(LIMIT))

taskset = XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{DATA}/code.parquet"))
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if "error" not in json.loads(l)}
todo = [t for t in taskset if t.data.id not in done and (not ONLY or t.data.id in ONLY)]
sem = asyncio.Semaphore(CONC)
count = 0


def fields(text):
    return dict(line.split("=", 1) for line in text.splitlines() if re.match(r"^\w+=", line))


async def work(task, rec):
    config = resolve_runtime_config(BASE, task)
    trace = SimpleNamespace(id="probe", info={}, state=SimpleNamespace(base_ref=""), last_reply="")
    async with provision_runtime(config, env=dict(task.runtime_env())) as rt:
        await rt.prepare_setup()
        started = time.monotonic()
        try:
            await type(task).setup(task, trace, rt)
        except Exception as e:
            rec["setup_error"] = repr(e)[:400]
        rec["setup_seconds"] = round(time.monotonic() - started, 1)
        rec["base_ref"] = trace.state.base_ref
        result = await rt.run(["bash", "-c", f"cd {task.data.workdir} && {FSCK}"], {})
        rec.update(fields(result.stdout))


async def probe(task):
    global count
    rec = {"id": task.data.id}
    async with sem:
        for attempt in range(2):
            rec.clear(); rec["id"] = task.data.id
            try:
                await asyncio.wait_for(work(task, rec), LIMIT + 900)
                break
            except Exception as e:
                rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    count += 1
    if count % 100 == 0:
        print(f"[{count}/{len(todo)}]", flush=True)


async def main():
    print("todo", len(todo), "done", len(done), flush=True)
    await asyncio.gather(*(probe(t) for t in todo))

asyncio.run(main())
