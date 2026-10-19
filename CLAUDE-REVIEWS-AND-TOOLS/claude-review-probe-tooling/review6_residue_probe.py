"""Review 6: after the port's real setup, look for files in `.git` that still name pruned commits.

    uv run python review6_residue_probe.py <DATA> <out.jsonl> [concurrency=96]
    PROBE_IDS="id id ..." restricts the run.
    PROBE_FIX=1 applies the change proposed in CLAUDE-REVIEW-6.md as an in-memory patch: the strip
    also deletes ORIG_HEAD, FETCH_HEAD and the commit-graph before `git gc`. Nothing in the
    repository is modified. PROBE_VERIFY=1 also runs the port's verifier and records its exit code.

The strip removes refs, reflogs and unreachable objects. This probe checks what else in the Git
directory can still name an object that no longer exists: every 40-hex string in every file outside
`objects/` (FETCH_HEAD, ORIG_HEAD, info/refs, logs, ...) is checked against the object store, and
the commit-graph is verified (it is binary, so the text scan cannot see inside it).
Per image it records
  missing_by_file    file -> number of distinct hashes in it that are not in the object store
  special            which pseudo-refs and state files exist (FETCH_HEAD, ORIG_HEAD, MERGE_HEAD, ...)
  commit_graph       whether a commit-graph file or chain exists, and `git commit-graph verify`'s status
  urls               remote URLs still present in config or FETCH_HEAD
  modules            number of submodule Git directories
  modules_extra      commits in submodule Git directories reachable from a ref but not from that HEAD
  modules_files      ORIG_HEAD / FETCH_HEAD left inside submodule Git directories
  submodule_status   `git submodule status` prefixes (' ' matches the gitlink, '+' differs, '-' absent)
  state_changed      HEAD, tree, `git describe` or `git status` differ after setup
  unreachable        unreachable objects from `git fsck --connectivity-only --unreachable --no-reflogs`
Resumable: rows already recorded without an error are skipped. No model runs.
"""
import asyncio, json, os, re, sys
from types import SimpleNamespace

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig

DATA, OUT = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 96
ONLY = set(os.environ.get("PROBE_IDS", "").split())
BASE = PrimeConfig(cpu=2, disk=20, labels=["xiaomi-review"])
SCAN = r"""
gd=$(git rev-parse --absolute-git-dir)
echo "shallow=$(git rev-parse --is-shallow-repository 2>/dev/null)"
echo "extra_after=$(git rev-list --all --not HEAD 2>/dev/null | wc -l)"
cd "$gd"
find . -path ./objects -prune -o -path ./modules -prune -o -path ./lfs -prune -o -type f -size -20M -print 2>/dev/null |
    while IFS= read -r f; do grep -a -o -H -E '\b[0-9a-f]{40}\b' "$f" 2>/dev/null; done | sort -u > /tmp/_named
for f in objects/info/packs objects/info/alternates; do [ -f "$f" ] && grep -a -o -H -E '\b[0-9a-f]{40}\b' "$f"; done >> /tmp/_named 2>/dev/null
sed 's/.*://' /tmp/_named | sort -u | git cat-file --batch-check 2>/dev/null | awk '$2=="missing"{print $1}' > /tmp/_missing
echo "named_total=$(sed 's/.*://' /tmp/_named | sort -u | wc -l)"
echo "missing_total=$(wc -l < /tmp/_missing)"
echo "missing_by_file=$(grep -F -f /tmp/_missing /tmp/_named 2>/dev/null | sed 's/:[0-9a-f]*$//' | sort | uniq -c | awk '{printf "%s=%s;", $2, $1}' | head -c 600)"
echo "special=$(for f in FETCH_HEAD ORIG_HEAD MERGE_HEAD CHERRY_PICK_HEAD REVERT_HEAD REBASE_HEAD BISECT_LOG AUTO_MERGE packed-refs shallow info/refs info/grafts gitk.cache objects/info/commit-graph objects/info/commit-graphs objects/pack/multi-pack-index rebase-merge rebase-apply logs refs/stash refs/original filter-repo svn; do [ -e "$f" ] && printf '%s,' "$f"; done)"
echo "bitmaps=$(ls objects/pack/*.bitmap 2>/dev/null | wc -l)"
echo "modules=$(find modules -name HEAD -maxdepth 6 2>/dev/null | wc -l)"
echo "logs_bytes=$(find logs -type f -exec cat {} + 2>/dev/null | wc -c)"
git commit-graph verify >/dev/null 2>/tmp/_cg; echo "commit_graph_rc=$?"
# submodule Git directories: commits reachable from their refs but not from their HEAD, and leftovers
mx=0; mf=""; for m in $(find modules -name HEAD -maxdepth 6 2>/dev/null | xargs -r -n1 dirname); do
    n=$(git --git-dir="$m" rev-list --all --not HEAD 2>/dev/null | wc -l); mx=$((mx + n))
    for f in ORIG_HEAD FETCH_HEAD; do [ -e "$m/$f" ] && mf="$mf$m/$f,"; done
done; echo "modules_extra=$mx"; echo "modules_files=$(echo "$mf" | head -c 300)"
echo "commit_graph_err=$(grep -c 'Could not read' /tmp/_cg)"
echo "urls=$( (git config --get-regexp 'url|remote\.' 2>/dev/null | head -3; grep -a -o -E '[a-z+]+://[^[:space:]]+' FETCH_HEAD 2>/dev/null | sort -u | head -2) | tr '\n' ' ' | head -c 300)"
echo "fetch_head=$(head -c 200 FETCH_HEAD 2>/dev/null | tr '\n\t' '| ')"
echo "orig_head=$(head -c 60 ORIG_HEAD 2>/dev/null | tr '\n' ' ')"
"""

# HEAD, tree, describe (without the abbreviated hash, whose length depends on object count) and status (ignoring caches setup may clean) before and after setup
STATE = ("git rev-parse HEAD HEAD^{tree}; git describe --tags --long 2>/dev/null | sed 's/-g[0-9a-f]*$//'; "
         "git status --porcelain 2>&1 | grep -v -E '__pycache__|\\.pytest_cache|\\.jest/' | sha1sum")

if os.environ.get("PROBE_FIX"):
    import xiaomi_code.taskset as port
    strip = port.strip_history_command
    GC = "git -c gc.pruneExpire=now"
    EXTRA = ('rm -f -- "$git_dir/ORIG_HEAD" "$git_dir/FETCH_HEAD"\n'
             'rm -rf -- "$git_dir/objects/info/commit-graph" "$git_dir/objects/info/commit-graphs"\n')

    def patched(base):
        command = strip(base)
        assert command.count(GC) == 1
        return command.replace(GC, EXTRA + GC)

    port.strip_history_command = patched
taskset = XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{DATA}/code.parquet"))
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if "error" not in json.loads(l)}
todo = [t for t in taskset if t.data.id not in done and (not ONLY or t.data.id in ONLY)]
sem = asyncio.Semaphore(CONC)
count = 0


def fields(text):
    return dict(line.split("=", 1) for line in text.splitlines() if re.match(r"^\w+=", line))


async def work(task, rec):
    config = resolve_runtime_config(BASE, task)
    trace = SimpleNamespace(id="probe", info={}, state=SimpleNamespace(base_ref=""), last_reply="")
    async with provision_runtime(config, env=dict(task.runtime_env())) as rt:
        await rt.prepare_setup()
        before = await rt.run(["bash", "-c", f"cd {task.data.workdir} && {STATE}"], {})
        rec["before"] = before.stdout.strip()
        try:
            await type(task).setup(task, trace, rt)
        except Exception as e:
            rec["setup_error"] = repr(e)[:300]
        rec["base_ref"] = trace.state.base_ref
        result = await rt.run(["bash", "-c", f"cd {task.data.workdir} && {SCAN}"], {})
        rec.update(fields(result.stdout))
        after = await rt.run(["bash", "-c", f"cd {task.data.workdir} && {STATE}"], {})
        rec["state_changed"] = after.stdout.strip() != rec["before"]
        check = await rt.run(["bash", "-c", f"cd {task.data.workdir} && git fsck --connectivity-only --unreachable --no-reflogs >/tmp/_f 2>&1; echo fsck_rc=$?; "
                              "echo unreachable=$(grep -c '^unreachable' /tmp/_f); echo head_is_base=$([ \"$(git rev-parse HEAD)\" = " + (trace.state.base_ref or "x") + " ] && echo 1 || echo 0); "
                              "echo submodule_status=$(git submodule status --recursive 2>/dev/null | cut -c1 | sort | uniq -c | tr -s ' ' | tr '\\n' ';'); "
                              "echo status=$(git status --porcelain 2>&1 | grep -v -E '__pycache__|\\.pytest_cache|\\.jest/' | sha1sum | cut -c1-12)"], {})
        rec.update(fields(check.stdout))
        if os.environ.get("PROBE_VERIFY") and "setup_error" not in rec:
            try:
                await asyncio.wait_for(type(task).verified(task, trace, rt), 600)
                rec["exit"] = trace.info["verifier"]["verifier_returncode"]
            except asyncio.TimeoutError:
                rec["exit"] = "cap600"


async def probe(task):
    global count
    rec = {"id": task.data.id}
    async with sem:
        for attempt in range(2):
            rec.clear(); rec["id"] = task.data.id
            try:
                await asyncio.wait_for(work(task, rec), 1500)
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
    await asyncio.gather(*(probe(t) for t in todo))

asyncio.run(main())
