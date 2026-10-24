"""Review 4: apply the four recommended changes as in-memory patches and re-run setup + verifier.

    uv run python review4_fix_check.py <DATA> <out.jsonl> <instance_id ...>

Nothing in the repository is modified. The patches mirror the fixes proposed in CLAUDE-REVIEW-4.md:
  1. bootstrap without its last line (`export BASH_ENV=...`)
  2. cache clean-up that leaves `__pycache__` alone
  3. hooks disabled for the strip's `git checkout --detach`
  4. history check decided by exit status, so text printed by a profile cannot trip it
"""
import asyncio, json, re, sys, time
from types import SimpleNamespace

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
import xiaomi_code.taskset as port
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig

DATA, OUT, IDS = sys.argv[1], sys.argv[2], set(sys.argv[3:])

lines = port.BASH_ENV_SCRIPT.splitlines(keepends=True)
assert lines[-1].startswith("export BASH_ENV=")
port.BASH_ENV_SCRIPT = "".join(lines[:-1])                                             # 1
assert "-name __pycache__ -o " in port.CACHE_CLEANUP
port.CACHE_CLEANUP = port.CACHE_CLEANUP.replace("-name __pycache__ -o ", "")            # 2
strip = port.strip_history_command
port.strip_history_command = lambda base: strip(base).replace(                          # 3
    "git checkout --detach", "git -c core.hooksPath=/dev/null checkout --detach")
run_shell = port.run_shell


async def patched_run_shell(runtime, command, *args, **kwargs):                         # 4
    found = re.fullmatch(r"git rev-list --all --not ([0-9a-f]{40})", command)
    if found:
        command = f'[ -z "$(git rev-list -n 1 --all --not {found.group(1)})" ]'
    result = await run_shell(runtime, command, *args, **kwargs)
    return type(result)(result.exit_code, "", result.stderr) if found else result

port.run_shell = patched_run_shell
tasks = [t for t in XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{DATA}/code.parquet")) if t.data.id in IDS]
sem = asyncio.Semaphore(40)


async def one(task):
    rec = {"id": task.data.id}
    cfg = resolve_runtime_config(PrimeConfig(cpu=2, disk=20, labels=["xiaomi-review"]), task)
    trace = SimpleNamespace(id="probe", info={}, state=SimpleNamespace(base_ref=""), last_reply="")
    async with sem:
        try:
            async with provision_runtime(cfg, env=dict(task.runtime_env())) as rt:
                await rt.prepare_setup()
                started = time.monotonic()
                await type(task).setup(task, trace, rt)
                rec["setup_seconds"] = round(time.monotonic() - started, 1)
                started = time.monotonic()
                rec["reward"] = await asyncio.wait_for(type(task).verified(task, trace, rt), 900)
                verifier = trace.info["verifier"]
                out = re.sub(r"\x1b\[[0-9;]*m", "", verifier["stdout"] + verifier["stderr"])
                rec.update(verify_seconds=round(time.monotonic() - started, 1), exit=verifier["verifier_returncode"],
                           not_found=sorted(set(re.findall(r"([\w./+-]+): (?:command )?not found", out)))[:5],
                           missing_module=sorted(set(re.findall(r"No module named '?([\w.]+)", out)))[:5],
                           pyenv_lock=out.count("couldn't acquire lock"),
                           summary=",".join(re.findall(r"\d+ (?:passed|failed|errors?)", out[-3000:])[-3:]), tail=out[-300:])
                check = await rt.run(["bash", "-c", f"cd {task.data.workdir} && git rev-list --all --not HEAD | grep -c ."], {})
                rec["extra_after"] = check.stdout.strip().splitlines()[-1] if check.stdout.strip() else ""
        except Exception as e:
            rec["error"] = repr(e)[:400]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print({k: v for k, v in rec.items() if k != "tail"}, flush=True)


async def main():
    await asyncio.gather(*(one(t) for t in tasks))

asyncio.run(main())
