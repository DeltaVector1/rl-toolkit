"""Review 3 follow-up: upstream's strip leaves `git gc` failing on images that carry a broken
`refs/remotes/origin/HEAD` symref (for-each-ref does not list it, so it is never deleted, and
repack dies with "bad object"). This runs the strip with one extra line that removes leftover
loose ref files, and reports whether gc succeeds and unreachable commits are gone.

    uv run python strip_fixed_probe.py <DATA> <instance_id ...>
"""
import asyncio, json, os, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

insts = {i["instance_id"]: i for i in (json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{sys.argv[1]}/code.parquet").to_pylist())}
STRIP = r"""
git config --global --add safe.directory "$PWD"
base=$(git rev-parse HEAD)
tree=$(git rev-parse HEAD^{tree}); dirty=$(git status --porcelain | wc -l)
echo "unreachable_before=$(timeout 150 git fsck --unreachable --no-reflogs 2>/dev/null | grep -c ' commit ')"
git checkout --detach $base 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
git remote | xargs -r -n1 git remote remove 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull 2>/dev/null | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
# The added line: broken symrefs such as refs/remotes/origin/HEAD are invisible to for-each-ref.
rm -rf .git/refs/remotes .git/refs/pull .git/refs/replace .git/refs/notes .git/refs/stash
git for-each-ref --format="%(refname)" refs/tags | while read t; do if ! git merge-base --is-ancestor "$t" $base 2>/dev/null; then git tag -d "${t#refs/tags/}" >/dev/null 2>&1; fi; done || true
git reflog expire --expire=now --all 2>/dev/null || true
out=$(git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now 2>&1); echo "gc_rc=$?"; echo "gc_err=$(printf '%s' "$out" | grep -iE 'fatal|error' | head -2 | tr '\n' ' ')"
echo "extra_after=$(git rev-list --all --not $base | wc -l)"
echo "unreachable_after=$(timeout 150 git fsck --unreachable --no-reflogs 2>/dev/null | grep -c ' commit ')"
echo "unchanged=$([ "$(git rev-parse HEAD)" = "$base" ] && [ "$(git rev-parse HEAD^{tree})" = "$tree" ] && [ "$(git status --porcelain | wc -l)" = "$dirty" ] && echo yes || echo no)"
echo "seconds=$SECONDS"
"""
sem = asyncio.Semaphore(int(os.environ.get("PROBE_CONCURRENCY", "24")))

async def probe(iid):
    inst = insts[iid]
    async with sem:
        try:
            cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=inst["cwd"], cpu=2, memory=4, disk=10, labels=["xiaomi-review"])
            async with provision_runtime(cfg, env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}) as rt:
                r = await asyncio.wait_for(rt.run(["timeout", "900", "bash", "-c", f"cd {inst['cwd']} && " + STRIP], {}), 1000)
                rec = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        except Exception as e:
            rec = {"error": repr(e)[:200]}
    print(json.dumps({"id": iid, **rec}), flush=True)

async def main():
    await asyncio.gather(*(probe(i) for i in sys.argv[2:]))

asyncio.run(main())
