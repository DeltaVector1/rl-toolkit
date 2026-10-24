"""Compare upstream's login shell (bash -lc) with the port's bash -c on pristine task images."""
import asyncio, json, random, re, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

D, OUT = sys.argv[1], sys.argv[2]
rows = pq.read_table(f"{D}/code.parquet").to_pylist()
insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in rows)}
TOOLS = "go cargo node npm yarn python python3 pytest java mvn ruby php gcc make"

async def probe(iid):
    inst = insts[iid]
    cwd = inst["cwd"]
    rec = {"id": iid, "cmd": inst["test_command"]}
    try:
        cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=cwd, cpu=2, memory=4, disk=10, labels=["xiaomi-review"])
        async with provision_runtime(cfg) as rt:
            for flag in ("-c", "-lc"):
                r = await rt.run(["bash", flag, f"echo PATH=$PATH; for t in {TOOLS}; do printf '%s=%s\n' $t \"$(command -v $t)\"; done"], {})
                rec[f"env{flag}"] = r.stdout
            await rt.write("/tmp/p.patch", inst["test_patch"].encode())
            a = await rt.run(["bash", "-c", f"cd {cwd} && git apply /tmp/p.patch"], {})
            rec["apply"] = a.exit_code
            # Hidden tests fail on a pristine tree; the 'base' half must pass, and 127 means a missing tool.
            cmd = inst["test_command"].split(" && ")[0]
            rec["probe_cmd"] = cmd
            for flag in ("-c", "-lc"):
                r = await rt.run(["timeout", "1500", "bash", flag, f"cd {cwd} && {cmd}"], {})
                out = r.stdout + r.stderr
                rec[f"run{flag}"] = {"exit": r.exit_code, "not_found": sorted(set(re.findall(r"([\w./-]+): (?:command )?not found", out)))[:8], "tail": out[-600:]}
    except Exception as e:
        rec["error"] = repr(e)[:500]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(iid, rec.get("error") or {k: (v["exit"], v["not_found"]) for k, v in rec.items() if k.startswith("run")}, flush=True)

async def main():
    await asyncio.wait_for(asyncio.gather(*(probe(i) for i in sys.argv[3:])), 3000)

asyncio.run(main())
