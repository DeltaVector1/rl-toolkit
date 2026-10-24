"""Review 3 follow-up: run every Code row's verifier once on its untouched image, the way the
port will after the recommended fixes (Docker-default PATH, login shell).

    uv run python verifier_health_probe.py <DATA> <out.jsonl> [concurrency=72] [cap seconds=240]

No model runs, so the hidden tests should fail. This looks for rows that are broken regardless
of the model: the test patch does not apply, a tool is missing, or the verifier already passes.
Each test command is capped; a row that hits the cap got far enough to rule out a missing tool
at start-up. Resumable: rows already recorded without an error are skipped.
"""
import asyncio, json, os, re, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

DATA, OUT = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 72
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 240
insts = [json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{DATA}/code.parquet").to_pylist()]
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if "error" not in json.loads(l)}
ONLY = set(os.environ.get("PROBE_IDS", "").split())  # optional: restrict to these ids
todo = [i for i in insts if i["instance_id"] not in done and (not ONLY or i["instance_id"] in ONLY)]
sem = asyncio.Semaphore(CONC)
count = 0

async def work(inst, rec):
    cwd = inst["cwd"]
    cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=cwd, cpu=2, memory=float(os.environ.get("PROBE_MEMORY", "4")), disk=float(os.environ.get("PROBE_DISK", "10")), labels=["xiaomi-review"])
    async with provision_runtime(cfg, env={"PATH": PATH}) as rt:
        await rt.run(["git", "config", "--global", "--add", "safe.directory", cwd], {})
        await rt.write("/tmp/p.patch", inst["test_patch"].encode())
        applied = await rt.run(["bash", "-c", f"cd {cwd} && git apply --verbose /tmp/p.patch"], {})
        rec["apply_rc"] = applied.exit_code
        if applied.exit_code:
            rec["apply_err"] = (applied.stdout + applied.stderr)[-300:]
            return
        r = await rt.run(["timeout", "--kill-after=5", str(CAP), "bash", "-lc", f"cd {cwd} && {inst['test_command']}"], {})
        out = re.sub(r"\x1b\[[0-9;]*m", "", r.stdout + r.stderr)
        rec.update(
            exit=r.exit_code,
            not_found=sorted(set(re.findall(r"([\w./+-]+): (?:command )?not found", out)))[:5],
            missing_module=sorted(set(re.findall(r"No module named '?([\w.]+)", out)))[:5],
            tail=out[-400:],
        )

async def probe(inst):
    global count
    rec = {"id": inst["instance_id"], "cwd": inst["cwd"]}
    async with sem:
        for attempt in range(2):
            rec.pop("error", None)
            try:
                await asyncio.wait_for(work(inst, rec), CAP + 600)
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
    await asyncio.gather(*(probe(i) for i in todo))

asyncio.run(main())
