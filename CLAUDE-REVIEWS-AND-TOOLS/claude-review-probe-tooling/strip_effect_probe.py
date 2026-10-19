"""Review 3: does upstream's history strip change what the verifier reports?

    uv run python strip_effect_probe.py <DATA> <instance_id ...>

Per row, in one sandbox: run the hidden tests on the untouched tree, restore, strip the history
(mimoagent base.py _strip_future_commits plus one line that removes leftover loose ref files), then run the real XiaomiCodeTask setup and
reward. The two verifier runs should end the same way, and setup must no longer raise.
"""
import asyncio, json, os, re, sys
from types import SimpleNamespace
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig

IDS = sys.argv[2:]
tasks = {t.data.id: t for t in XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{sys.argv[1]}/code.parquet")) if t.data.id in IDS}
STRIP = r"""
base=$(git rev-parse HEAD)
git checkout --detach $base 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
git remote | xargs -r -n1 git remote remove 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull 2>/dev/null | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
rm -rf .git/refs/remotes .git/refs/pull .git/refs/replace .git/refs/notes .git/refs/stash
git for-each-ref --format="%(refname)" refs/tags | while read t; do if ! git merge-base --is-ancestor "$t" $base 2>/dev/null; then git tag -d "${t#refs/tags/}" >/dev/null 2>&1; fi; done || true
git reflog expire --expire=now --all 2>/dev/null || true
git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now 2>/dev/null || true
"""

def summary(text):
    lines = [l for l in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines() if re.search(r"\d+ (passed|failed|passing|failing|error)|^(ok|FAIL|PASS)\b|Tests:|tests? ran", l)]
    return [re.sub(r"\b\d+(\.\d+)? ?m?s\b|in \d+(\.\d+)?s", "", l).strip()[:110] for l in lines][-4:]

async def probe(iid):
    task = tasks[iid]; cwd = task.data.workdir
    async with provision_runtime(resolve_runtime_config(PrimeConfig(cpu=2, memory=4, disk=10, labels=["xiaomi-review"]), task), env=dict(task.runtime_env())) as rt:
        await rt.run(["git", "config", "--global", "--add", "safe.directory", cwd], {})
        await rt.write("/tmp/p.patch", task.data.test_patch.encode())
        before = await rt.run(["timeout", "1500", "bash", "-c", f"cd {cwd} && git apply /tmp/p.patch && {task.data.test_command}"], {})
        # Undo the patch the same way the task does, using HEAD as the base.
        from xiaomi_code.taskset import reset_command
        head = (await rt.run(["bash", "-c", f"cd {cwd} && git rev-parse HEAD"], {})).stdout.strip()
        await rt.run(["bash", "-c", f"cd {cwd} && " + reset_command(task.data.test_patch, head)], {})
        await rt.run(["timeout", "900", "bash", "-c", f"cd {cwd} && " + STRIP], {})
        trace = SimpleNamespace(id="probe", info={}, state=SimpleNamespace(base_ref=""))
        try:
            await task.setup(trace, rt)
            reward = await type(task).verified(task, trace, rt)
            v = trace.info["verifier"]
            after_rc, after_sum, setup = v["verifier_returncode"], summary(v["stdout"] + v["stderr"]), "ok"
        except Exception as e:
            after_rc, after_sum, setup, reward = None, [], repr(e)[:120], None
        b = summary(before.stdout + before.stderr)
        return f"{iid}: setup after strip={setup} | rc before={before.exit_code} after={after_rc} | same summary={b == after_sum}\n     before: {b}\n     after : {after_sum}"

sem = asyncio.Semaphore(int(os.environ.get("PROBE_CONCURRENCY", "10")))

async def bounded(iid):
    async with sem:
        try:
            out = await asyncio.wait_for(probe(iid), 4500)
        except Exception as e:
            out = f"{iid}: PROBE ERROR {e!r}"[:300]
        print(out, flush=True)

async def main():
    await asyncio.gather(*(bounded(i) for i in IDS))

asyncio.run(main())
