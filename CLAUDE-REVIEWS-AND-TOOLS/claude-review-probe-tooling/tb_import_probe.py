"""Review 2: for Terminal Bench rows whose tests hit an import error on an untouched image, show
the failing import, whether the module is installed, and whether the image's own requirements list it.

    uv run python tb_import_probe.py <DATA/general/train.parquet> <module>=<instance_id> ...
"""
import asyncio, base64, json, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

rows = pq.read_table(sys.argv[1]).to_pylist()
insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in rows)}
CMD = r"""
mod="$1"
echo "## import $mod: $(python3 -c "import $mod; print('ok', getattr($mod, '__version__', ''))" 2>&1 | tail -1)"
echo "## requirements mentioning it: $(grep -ril -- "$mod" /app/requirements*.txt /app/Dockerfile /tmp/requirements*.txt 2>/dev/null | tr '\n' ' ')"
grep -i -- "$mod" /app/requirements*.txt 2>/dev/null | head -3
echo "## pip packages: $(python3 -m pip list 2>/dev/null | wc -l)"
echo "## pytest on untouched /app"
cd /app && python3 -m pytest -q -x /tests/test_outputs.py 2>&1 | grep -vE '^\s*$' | tail -14
"""

async def probe(spec):
    mod, iid = spec.split("=", 1)
    inst = insts[iid]
    files = json.loads(inst["tests_files"]) if isinstance(inst["tests_files"], str) else inst["tests_files"]
    cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir="/app", cpu=1, memory=2, disk=10, labels=["xiaomi-review"])
    async with provision_runtime(cfg) as rt:
        for name, data in files.items():
            await rt.write(f"/tests/{name}", base64.b64decode(data))
        r = await rt.run(["sh", "-c", CMD, "sh", mod], {})
        return f"================ {iid} ({inst['docker_image']})\n{r.stdout[-2600:]}\n{r.stderr[-200:]}"

async def main():
    for out in await asyncio.gather(*(probe(s) for s in sys.argv[2:]), return_exceptions=True):
        print(out)

asyncio.run(main())
