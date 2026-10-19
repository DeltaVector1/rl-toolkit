"""Review 3: does the verifier's result change between the port's `bash -c` and upstream's `bash -lc`?

A third mode runs `bash -c` with BASH_ENV pointing at a three-line login emulation, which is one way
to give both the agent's shell and the verifier the image's intended environment.

    uv run python login_verifier_probe.py <DATA> <out.jsonl> <instance_id ...>

Per row, on an untouched image with the hidden test patch applied, run the row's test command
under both shells (same Docker-default PATH). Hidden tests fail either way; what matters is
whether the non-login run fails for an environment reason (tool or module not found).
"""
import asyncio, json, os, re, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code.taskset import reset_command
from xiaomi_code._data import resolve_image

DATA, OUT, IDS = sys.argv[1], sys.argv[2], sys.argv[3:]
insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{DATA}/code.parquet").to_pylist())}
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
sem = asyncio.Semaphore(int(os.environ.get("PROBE_CONCURRENCY", "10")))
EMU_PATH = "/tmp/xiaomi-login-env.sh"
EMU = """unset BASH_ENV
[ -r /etc/profile ] && . /etc/profile
if [ -r ~/.bash_profile ]; then . ~/.bash_profile; elif [ -r ~/.bash_login ]; then . ~/.bash_login; elif [ -r ~/.profile ]; then . ~/.profile; fi
"""

def digest(result):
    out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
    return {
        "exit": result.exit_code,
        "not_found": sorted(set(re.findall(r"([\w./-]+): (?:command )?not found", out)))[:5],
        "missing_module": sorted(set(re.findall(r"No module named '?([\w.]+)", out)))[:5],
        "summary": [l.strip()[:120] for l in out.splitlines() if re.search(r"\d+ (passed|failed|error)|^(ok|FAIL|PASS)\b|Tests:", l)][-2:],
        "tail": out[-300:],
    }

async def probe(iid):
    inst = insts[iid]; cwd = inst["cwd"]
    rec = {"id": iid}
    async with sem:
        try:
            cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=cwd, cpu=2, memory=4, disk=10, labels=["xiaomi-review"])
            async with provision_runtime(cfg, env={"PATH": PATH}) as rt:
                await rt.run(["git", "config", "--global", "--add", "safe.directory", cwd], {})
                head = (await rt.run(["bash", "-c", f"cd {cwd} && git rev-parse HEAD"], {})).stdout.strip()
                await rt.write("/tmp/p.patch", inst["test_patch"].encode())
                await rt.write(EMU_PATH, EMU.encode())
                for mode, flag, env in (("-c", "-c", {}), ("-lc", "-lc", {}), ("bash_env", "-c", {"BASH_ENV": EMU_PATH})):
                    await rt.run(["bash", "-c", f"cd {cwd} && " + reset_command(inst["test_patch"], head) + f"\ngit apply /tmp/p.patch"], {})
                    rec[mode] = digest(await rt.run(["timeout", "1500", "bash", flag, f"cd {cwd} && {inst['test_command']}"], env))
        except Exception as e:
            rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(iid, rec.get("error") or {f: (rec[f]["exit"], rec[f]["not_found"], rec[f]["missing_module"]) for f in ("-c", "-lc", "bash_env")}, flush=True)

async def main():
    await asyncio.gather(*(asyncio.wait_for(probe(i), 5400) for i in IDS), return_exceptions=True)

asyncio.run(main())
