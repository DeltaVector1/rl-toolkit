"""Review 3 follow-up: on images whose sandbox died during the full pass, does the history strip
complete with more memory, or with a memory-bounded `git gc`?

    uv run python strip_heavy_probe.py <DATA> <instance_id ...>

Each row runs in two fresh sandboxes: 4 GB with a bounded gc, and 8 GB with the default gc.
"""
import asyncio, json, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{sys.argv[1]}/code.parquet").to_pylist())}
SCRIPT = r"""
git config --global --add safe.directory "$PWD"
base=$(git rev-parse HEAD)
echo "git_mb=$(du -sm .git | cut -f1) packs=$(ls .git/objects/pack/*.pack 2>/dev/null | wc -l) loose=$(git count-objects | cut -d' ' -f1) mem_mb=$(free -m | awk '/Mem:/{print $2}') disk_free_mb=$(df -m . | awk 'NR==2{print $4}')"
echo "extra_before=$(git rev-list --all --not $base | wc -l)"
git checkout --detach $base 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
git remote | xargs -r -n1 git remote remove 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull 2>/dev/null | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
git for-each-ref --format="%(refname)" refs/tags | while read t; do if ! git merge-base --is-ancestor "$t" $base 2>/dev/null; then git tag -d "${t#refs/tags/}" >/dev/null 2>&1; fi; done || true
git reflog expire --expire=now --all 2>/dev/null || true
echo "refs_done_s=$SECONDS"
git -c gc.pruneExpire=now -c gc.cruftPacks=false GCOPTS gc --prune=now 2>&1 | tail -2
echo "gc_rc=$? gc_done_s=$SECONDS"
echo "extra_after=$(git rev-list --all --not $base | wc -l) head_same=$([ "$(git rev-parse HEAD)" = "$base" ] && echo yes || echo no) dirty=$(git status --porcelain | wc -l)"
echo "git_mb_after=$(du -sm .git | cut -f1)"
"""
VARIANTS = {"4GB bounded gc": (4, "-c pack.threads=1 -c pack.windowMemory=100m -c pack.deltaCacheSize=50m"), "8GB default gc": (8, "")}

async def probe(iid, name):
    memory, opts = VARIANTS[name]
    inst = insts[iid]
    try:
        cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=inst["cwd"], cpu=2, memory=memory, disk=20, labels=["xiaomi-review"])
        async with provision_runtime(cfg, env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}) as rt:
            r = await asyncio.wait_for(rt.run(["timeout", "1500", "bash", "-c", f"cd {inst['cwd']} && " + SCRIPT.replace("GCOPTS", opts)], {}), 1700)
            return f"{iid[-6:]} [{name}] rc={r.exit_code} " + " ".join(l for l in r.stdout.splitlines() if "=" in l)
    except Exception as e:
        return f"{iid[-6:]} [{name}] SANDBOX DIED: {e!r}"[:230]

async def main():
    for out in await asyncio.gather(*(probe(i, n) for i in sys.argv[2:] for n in VARIANTS)):
        print(out, flush=True)

asyncio.run(main())
