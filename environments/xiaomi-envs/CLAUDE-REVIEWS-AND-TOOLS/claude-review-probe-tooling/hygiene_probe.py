"""Review 3: measure git-history truncation and build residue on untouched Code images.

    uv run python hygiene_probe.py <DATA> <out.jsonl> <n /workspace/repo rows> <n /testbed rows> [seed]

For each sampled row, in a Prime sandbox:
  1. Count commits reachable from any ref but outside HEAD's ancestry. Non-zero means
     XiaomiCodeTask.setup raises "Image history is not truncated" (as upstream's assertion does).
  2. List what upstream's shipped training config (anti_hack_cleanup: true) would purge.
  3. Apply upstream's documented per-rollout remedy (git_leak_prevention: strip, ported from
     mimoagent base.py _strip_future_commits) and re-count, checking HEAD and the work tree are unchanged.
"""
import asyncio, json, random, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

DATA, OUT, N_WS, N_TB = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
random.seed(int(sys.argv[5]) if len(sys.argv) > 5 else 77)
insts = [json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{DATA}/code.parquet").to_pylist()]
ws = [i for i in insts if i["cwd"] == "/workspace/repo"]
tb = [i for i in insts if i["cwd"] != "/workspace/repo"]
sample = random.sample(ws, min(N_WS, len(ws))) + random.sample(tb, min(N_TB, len(tb)))
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
KEEP = "node_modules bower_components .husky vendor third_party _deps vcpkg_installed .venv venv .gradle target .build lib .bundle Manifest.toml _build .stack-work dist-newstyle"
BEFORE = r"""
git config --global --add safe.directory "$PWD"
base=$(git rev-parse HEAD)
echo "base=$base"
echo "extra_before=$(git rev-list --all --not $base | wc -l)"
echo "refs=$(git for-each-ref | wc -l) heads=$(git for-each-ref refs/heads | wc -l) remotes=$(git for-each-ref refs/remotes | wc -l) tags=$(git for-each-ref refs/tags | wc -l)"
echo "head_subject=$(git log -1 --format=%s | cut -c1-60)"
echo "dirty_before=$(git status --porcelain | wc -l)"
echo "tree_before=$(git rev-parse HEAD^{tree})"
echo "clean_would_remove=$(git clean -fdxn """ + " ".join(f"--exclude={d}" for d in KEEP.split()) + r""" | wc -l)"
echo "clean_examples=$(git clean -fdxn """ + " ".join(f"--exclude={d}" for d in KEEP.split()) + r""" | sed 's/^Would remove //' | head -6 | tr '\n' ' ')"
echo "task_description=$(find / -maxdepth 4 -xdev -name task_description.md -path '*/.build_env/*' 2>/dev/null | head -2 | tr '\n' ' ')"
echo "build_env=$(ls -A .build_env 2>/dev/null | head -12 | tr '\n' ' ')"
echo "root_tests_logs=$(ls -d /tests /logs 2>/dev/null | tr '\n' ' ')"
echo "tmp_fstype=$(findmnt -n -o FSTYPE /tmp 2>/dev/null) tmp_entries=$(ls -A /tmp | wc -l)"
echo "m2_snapshots=$(find /root/.m2 -type f -name '*-SNAPSHOT*.jar' 2>/dev/null | wc -l)"
"""
STRIP = r"""
base=$(git rev-parse HEAD)
git checkout --detach $base 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/heads | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
git remote | xargs -r -n1 git remote remove 2>/dev/null || true
git for-each-ref --format="%(refname)" refs/replace refs/notes refs/stash refs/remotes refs/pull 2>/dev/null | while read r; do git update-ref -d "$r" 2>/dev/null; done || true
git for-each-ref --format="%(refname)" refs/tags | while read t; do if ! git merge-base --is-ancestor "$t" $base 2>/dev/null; then git tag -d "${t#refs/tags/}" >/dev/null 2>&1; fi; done || true
git reflog expire --expire=now --all 2>/dev/null || true
git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --prune=now 2>/dev/null || true
echo "extra_after=$(git rev-list --all --not $base | wc -l)"
echo "head_same=$([ "$(git rev-parse HEAD)" = "$base" ] && echo yes || echo no)"
echo "dirty_after=$(git status --porcelain | wc -l)"
echo "tree_after=$(git rev-parse HEAD^{tree})"
"""
sem = asyncio.Semaphore(10)

async def probe(inst):
    rec = {"id": inst["instance_id"], "cwd": inst["cwd"]}
    async with sem:
        try:
            cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=inst["cwd"], cpu=2, memory=4, disk=10, labels=["xiaomi-review"])
            async with provision_runtime(cfg, env={"PATH": PATH}) as rt:
                for name, script in (("before", BEFORE), ("strip", STRIP)):
                    r = await rt.run(["timeout", "900", "bash", "-c", f"cd {inst['cwd']} && " + script], {})
                    rec[name + "_rc"] = r.exit_code
                    for line in r.stdout.splitlines():
                        key, _, value = line.partition("=")
                        rec[key] = value
        except Exception as e:
            rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["id"], rec.get("cwd"), "extra", rec.get("extra_before"), "->", rec.get("extra_after"), "clean", rec.get("clean_would_remove"), rec.get("error", ""), flush=True)

async def main():
    await asyncio.wait_for(asyncio.gather(*(probe(i) for i in sample)), 3300)

asyncio.run(main())
