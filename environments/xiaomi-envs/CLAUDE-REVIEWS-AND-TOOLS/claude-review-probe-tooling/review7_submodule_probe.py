"""Review 7: can a submodule Git directory still hold the upstream fix after setup?

    uv run python review7_submodule_probe.py <DATA> <review7_full.jsonl> <out.jsonl> [concurrency=40]

The strip prunes the superproject only; `.git/modules/*` is left as the image ships it. Those
directories hold the submodules' own upstream history, which is harmless unless a future
superproject commit moved a submodule pointer. This probe looks at the untouched image (before
setup, while future superproject commits still exist) and, for every submodule path, lists the
gitlinks that future commits (any ref, ORIG_HEAD, FETCH_HEAD, not ancestors of the base) set for
it. A future gitlink whose commit is present in that submodule's Git directory is a real leak:
setup leaves it behind. The base commit comes from the review 7 full pass. No setup, no verifier.
Per image it records
  paths              submodule paths in the base `.gitmodules`
  future_links       distinct (path, commit) gitlinks set by future superproject commits
  present            how many of those commits exist in the submodule's Git directory
  examples           up to five "path commit" pairs that are present
"""
import asyncio, json, re, shlex, sys

from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.utils.compile import resolve_runtime_config
from xiaomi_code import XiaomiCodeTaskset
from xiaomi_code._data import XiaomiTasksetConfig

DATA, FULL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 40
BASE = PrimeConfig(cpu=2, disk=20, labels=["xiaomi-review"])
rows = {r["id"]: r for r in map(json.loads, open(FULL)) if r.get("modules", "0") != "0" and r.get("base_ref")}
SCAN = r"""
base=BASE
gd=$(git rev-parse --absolute-git-dir)
tips=$( (git for-each-ref --format='%(objectname)'; cat "$gd/ORIG_HEAD" "$gd/FETCH_HEAD" 2>/dev/null | grep -oE '^[0-9a-f]{40}') | sort -u)
git config -f <(git show "$base:.gitmodules" 2>/dev/null) --get-regexp '^submodule\..*\.path$' > /tmp/_paths
echo "paths=$(wc -l < /tmp/_paths)"
: > /tmp/_links
while read -r key path; do
    name=${key#submodule.}; name=${name%.path}
    for c in $(git rev-list $tips --not "$base" -- "$path" 2>/dev/null | head -2000); do
        l=$(git rev-parse -q --verify "$c:$path" 2>/dev/null) || continue
        echo "$name $path $l"
    done
done < /tmp/_paths | sort -u > /tmp/_links
base_links=$(git ls-tree -r "$base" | awk '$2=="commit"{print $3}' | sort -u)
present=0; total=0; ex=""
while read -r name path l; do
    echo "$base_links" | grep -qx "$l" && continue
    total=$((total + 1))
    if git --git-dir="$gd/modules/$name" cat-file -e "$l^{commit}" 2>/dev/null; then
        present=$((present + 1)); [ $present -le 5 ] && ex="$ex$path $l;"
    fi
done < /tmp/_links
echo "future_links=$total"
echo "present=$present"
echo "examples=$ex"
"""

taskset = XiaomiCodeTaskset(XiaomiTasksetConfig(dataset_path=f"{DATA}/code.parquet"))
todo = [t for t in taskset if t.data.id in rows]
sem = asyncio.Semaphore(CONC)


async def probe(task):
    rec = {"id": task.data.id}
    async with sem:
        try:
            config = resolve_runtime_config(BASE, task)
            async with provision_runtime(config, env=dict(task.runtime_env())) as rt:
                await rt.prepare_setup()
                script = SCAN.replace("BASE", rows[task.data.id]["base_ref"])
                result = await rt.run(["bash", "-c", f"cd {shlex.quote(task.data.workdir)} && {script}"], {})
                rec.update(dict(l.split("=", 1) for l in result.stdout.splitlines() if re.match(r"^\w+=", l)))
                if result.exit_code:
                    rec["stderr"] = result.stderr[-300:]
        except Exception as e:
            rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec["id"][-6:], rec.get("paths"), rec.get("future_links"), rec.get("present"), rec.get("error", ""), flush=True)


async def main():
    print("todo", len(todo), flush=True)
    await asyncio.gather(*(probe(t) for t in todo))

asyncio.run(main())
