"""On pristine images the 'base' half of each verifier must pass. Compare the port as written with a Docker-default PATH."""
import asyncio, json, re, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

D, OUT = sys.argv[1], sys.argv[2]
rows = pq.read_table(f"{D}/code.parquet").to_pylist()
insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in rows)}
DOCKER = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
GO_HACK = 'if ! command -v go >/dev/null 2>&1 && test -x /usr/local/go/bin/go; then printf %s "/usr/local/go/bin:$PATH"; fi'
sem = asyncio.Semaphore(10)

async def probe(iid):
    inst = insts[iid]; cwd = inst["cwd"]
    rec = {"id": iid}
    async with sem:
        try:
            cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=cwd, cpu=2, memory=4, disk=10, labels=["xiaomi-review"])
            async with provision_runtime(cfg) as rt:
                await rt.write("/tmp/p.patch", inst["test_patch"].encode())
                rec["apply"] = (await rt.run(["bash", "-c", f"cd {cwd} && git apply /tmp/p.patch"], {})).exit_code
                cmd = inst["test_command"].split(" && ")[0]
                hack = (await rt.run(["bash", "-c", GO_HACK], {})).stdout
                modes = {"port": ("-c", {"PATH": hack} if hack else {}), "docker-c": ("-c", {"PATH": DOCKER}), "docker-lc": ("-lc", {"PATH": DOCKER})}
                for name, (flag, env) in modes.items():
                    r = await rt.run(["timeout", "1200", "bash", flag, f"cd {cwd} && {cmd}"], env)
                    out = r.stdout + r.stderr
                    rec[name] = {"exit": r.exit_code, "not_found": sorted(set(re.findall(r"([\w./-]+): (?:command )?not found", out)))[:6], "tail": out[-400:]}
        except Exception as e:
            rec["error"] = repr(e)[:400]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(iid, rec.get("error") or {k: (v["exit"], v["not_found"]) for k, v in rec.items() if isinstance(v, dict)}, flush=True)

async def main():
    await asyncio.wait_for(asyncio.gather(*(probe(i) for i in sys.argv[3:])), 3300)

asyncio.run(main())
