import asyncio, base64, json, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

rows = pq.read_table(sys.argv[1]).to_pylist()
inst = next(i for i in (json.loads(r["extra_info"]["instance_json"]) for r in rows) if i["instance_id"] == sys.argv[2])
files = json.loads(inst["tests_files"]) if isinstance(inst["tests_files"], str) else inst["tests_files"]

async def main():
    cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir="/app", cpu=1, memory=2, disk=10, labels=["xiaomi-review"])
    async with provision_runtime(cfg) as rt:
        print((await rt.run(["sh", "-c", "echo PATH=$PATH; command -v python3; python3 -V; ls -A /app | head -20"], {})).stdout)
        for name, data in files.items():
            await rt.write(f"/tests/{name}", base64.b64decode(data))
        r = await rt.run(["sh", "-c", "mkdir -p /logs/verifier; python3 /tests/anti_hack_guard.py; echo guard_exit=$?"], {})
        print("pristine /app:", r.stdout.strip(), r.stderr.strip())
        r = await rt.run(["sh", "-c", "mkdir -p /app/.vf-venv/lib/site-packages && touch /app/.vf-venv/lib/site-packages/_virtualenv.pth; python3 /tests/anti_hack_guard.py; echo guard_exit=$?"], {})
        print("with .vf-venv only:", r.stdout.strip(), r.stderr.strip())

asyncio.run(main())
