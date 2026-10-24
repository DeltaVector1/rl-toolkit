"""Review 2: drive the real task classes (setup + reward) on untouched images in Prime sandboxes.

    uv run python postfix_probe.py <DATA> <out.jsonl> code <instance_id ...>
    uv run python postfix_probe.py <DATA> <out.jsonl> terminal <instance_id ...>

No model runs, so new/hidden tests are expected to fail. What must hold:
  code      no "command not found" in verifier output (finding 1)
  terminal  no anti_hack_guard REJECT on a clean /app (finding 2), egress blocked once the
            execution policy is applied (finding 4), and no .vf-* entries in /app
"""
import asyncio, json, re, sys
from types import SimpleNamespace

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig
from xiaomi_general import GeneralTaskset
from xiaomi_general.taskset import GeneralConfig

DATA, OUT, KIND, IDS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
EGRESS = "import urllib.request as u; u.urlopen('https://pypi.org/simple/pip/', timeout=8); print('reachable')"
sem = asyncio.Semaphore(10)
# Environment-failure signatures, as opposed to hidden tests failing on an unfixed tree.
SIGNALS = {
    "command not found": r"(?:command )?not found",
    "missing python module": r"ModuleNotFoundError|No module named",
    "missing node module": r"Cannot find module|ERR_MODULE_NOT_FOUND",
    "network": r"Could not resolve host|Temporary failure in name resolution|ECONNREFUSED|ETIMEDOUT|EAI_AGAIN|Connection timed out",
    "no such file": r"No such file or directory",
    "permission": r"Permission denied",
    "timeout": r"Timed out|timed out",
}

if KIND == "code":
    taskset = XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{DATA}/code.parquet"))
    base = PrimeConfig(cpu=2, memory=4, disk=10, labels=["xiaomi-review"])
else:
    taskset = GeneralTaskset(GeneralConfig(dataset_path=f"{DATA}/general/train.parquet"))
    base = PrimeConfig(labels=["xiaomi-review"])
tasks = {task.data.id: task for task in taskset if task.data.id in IDS}


async def probe(iid):
    task = tasks[iid]
    rec = {"id": iid, "kind": KIND}
    trace = SimpleNamespace(id="probe", info={}, state=SimpleNamespace(base_ref=""), last_reply="")
    async with sem:
        try:
            config = resolve_runtime_config(base, task)
            rec["runtime"] = {k: getattr(config, k) for k in ("cpu", "memory", "disk", "allow", "block")}
            async with provision_runtime(config, env=dict(task.runtime_env())) as rt:
                await rt.prepare_setup()
                await task.setup(trace, rt)
                if KIND == "terminal":
                    rec["toolsets"] = len(task.toolsets(task.config))
                    rec["egress_during_setup"] = (await rt.run(["python3", "-c", EGRESS], {})).stdout.strip() or "blocked"
                    await rt.prepare_execution([])
                    rec["egress_during_execution"] = (await rt.run(["python3", "-c", EGRESS], {})).stdout.strip() or "blocked"
                    reward = await type(task).verifier_reward(task, trace, rt)
                    rec["vf_entries_in_app"] = (await rt.run(["sh", "-c", "ls -A /app | grep '^\\.vf-' || true"], {})).stdout.split()
                else:
                    rec["tools"] = (await rt.run(["bash", "-c", "for t in go node npm npx yarn; do printf '%s=%s ' $t \"$(command -v $t)\"; done"], {})).stdout
                    reward = await type(task).verified(task, trace, rt)
                verifier = trace.info["verifier"]
                out = (verifier.get("stdout") or "") + (verifier.get("stderr") or "")
                rec.update(
                    reward=reward,
                    returncode=verifier.get("returncode", verifier.get("verifier_returncode")),
                    not_found=sorted(set(re.findall(r"([\w./-]+): (?:command )?not found", out)))[:6],
                    guard_reject="anti_hack_guard: REJECT" in out,
                    signals=sorted({name for name, pattern in SIGNALS.items() if re.search(pattern, out)}),
                    stdout_tail=(verifier.get("stdout") or "")[-500:],
                    stderr_tail=(verifier.get("stderr") or "")[-300:],
                )
        except Exception as e:
            rec["error"] = repr(e)[:500]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    keys = ("reward", "returncode", "not_found", "guard_reject", "toolsets", "egress_during_setup", "egress_during_execution", "vf_entries_in_app", "error")
    print(iid, {k: rec[k] for k in keys if k in rec}, flush=True)


async def main():
    missing = [iid for iid in IDS if iid not in tasks]
    if missing:
        raise SystemExit(f"unknown ids: {missing}")
    await asyncio.wait_for(asyncio.gather(*(probe(iid) for iid in IDS)), 3000)


asyncio.run(main())
