"""Review 3: do untouched Python images keep compiled copies (.pyc) or pytest caches of the hidden tests?

    uv run python pyc_residue_probe.py <DATA> <instance_id ...>

For every .py path in the row's test patch: is there a matching .pyc, and was it compiled from a
source of a different size than the file now on disk (or from a file that no longer exists)?
Upstream's shipped Code config (anti_hack_cleanup: true) removes these with `git clean -fdx`.
"""
import asyncio, json, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code.taskset import touched_files
from xiaomi_code._data import resolve_image

insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{sys.argv[1]}/code.parquet").to_pylist())}
SCRIPT = r'''
import glob, json, os, struct, sys
out = []
for path in json.loads(sys.argv[1]):
    d, name = os.path.split(path)
    for pyc in glob.glob(os.path.join(d, "__pycache__", os.path.splitext(name)[0] + ".*.pyc")):
        with open(pyc, "rb") as f:
            header = f.read(16)
        flags, mtime, size = struct.unpack("<III", header[4:16])
        src = os.path.getsize(path) if os.path.exists(path) else None
        out.append({"path": path, "pyc": os.path.basename(pyc), "pyc_source_size": size, "source_size_now": src, "pyc_bytes": os.path.getsize(pyc)})
last = ".pytest_cache/v/cache/lastfailed"
print(json.dumps({"pyc": out, "lastfailed": list(json.load(open(last)))[:6] if os.path.exists(last) else None}))
'''

async def probe(iid):
    inst = insts[iid]
    paths = [p for p in touched_files(inst["test_patch"]) if p.endswith(".py")]
    cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=inst["cwd"], cpu=1, memory=2, disk=10, labels=["xiaomi-review"])
    async with provision_runtime(cfg, env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}) as rt:
        r = await rt.run(["python3", "-c", SCRIPT, json.dumps(paths)], {})
        try:
            data = json.loads(r.stdout)
        except ValueError:
            return f"== {iid} unparsable: {r.stdout[:200]} {r.stderr[:200]}"
        lines = [f"== {iid}: {len(paths)} .py paths in test patch; lastfailed={data['lastfailed']}"]
        for item in data["pyc"]:
            stale = item["source_size_now"] is None or item["source_size_now"] != item["pyc_source_size"]
            lines.append(f"   {'STALE' if stale else 'current'} {item['path']} pyc_source_size={item['pyc_source_size']} source_now={item['source_size_now']} pyc_bytes={item['pyc_bytes']}")
        return "\n".join(lines)

async def main():
    for out in await asyncio.gather(*(probe(i) for i in sys.argv[2:]), return_exceptions=True):
        print(out)

asyncio.run(main())
