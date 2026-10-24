"""Review 3: does a login shell (upstream: bash -lc) see a different environment from the port's bash -c?

    uv run python login_env_probe.py <DATA> <out.jsonl> <n rows> [seed]

For each sampled Code row, in a Prime sandbox with the port's PATH: diff `env` under the two
shells, and resolve the first command word of each line of the row's test script under both.
"""
import asyncio, json, random, re, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

DATA, OUT, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 91)
insts = [json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{DATA}/code.parquet").to_pylist()]
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
IGNORE = {"SHLVL", "_", "PWD", "OLDPWD", "PS1", "LS_COLORS", "LESSOPEN", "LESSCLOSE"}
TOOLS = "python python3 pip pytest tox nox poetry uv go cargo node npm npx yarn pnpm ruby bundle rspec rake java mvn gradle php composer dotnet make cmake julia"
sem = asyncio.Semaphore(10)

def parse_env(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line))

async def probe(inst):
    rec = {"id": inst["instance_id"], "cwd": inst["cwd"]}
    async with sem:
        try:
            cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=inst["cwd"], cpu=1, memory=2, disk=10, labels=["xiaomi-review"])
            async with provision_runtime(cfg, env={"PATH": PATH}) as rt:
                plain = parse_env((await rt.run(["bash", "-c", "env"], {})).stdout)
                login = parse_env((await rt.run(["bash", "-lc", "env"], {})).stdout)
                rec["env_diff"] = {k: [plain.get(k), login.get(k)] for k in sorted(set(plain) | set(login)) if k not in IGNORE and plain.get(k) != login.get(k)}
                which = "for t in " + TOOLS + "; do printf '%s=%s\\n' $t \"$(command -v $t)\"; done"
                a = parse_env((await rt.run(["bash", "-c", which], {})).stdout)
                b = parse_env((await rt.run(["bash", "-lc", which], {})).stdout)
                rec["tool_diff"] = {k: [a.get(k), b.get(k)] for k in a if a.get(k) != b.get(k)}
        except Exception as e:
            rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["id"], sorted(rec.get("env_diff", {})), sorted(rec.get("tool_diff", {})), rec.get("error", ""), flush=True)

async def main():
    await asyncio.wait_for(asyncio.gather(*(probe(i) for i in random.sample(insts, N))), 3300)

asyncio.run(main())
