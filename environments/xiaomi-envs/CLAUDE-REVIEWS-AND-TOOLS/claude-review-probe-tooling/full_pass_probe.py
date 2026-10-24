"""Review 3 follow-up: one light pass over every Code image.

    uv run python full_pass_probe.py <DATA> <out.jsonl> [concurrency=24] [instance_id ...]

Resumable: rows already in <out.jsonl> without an error are skipped. Per untouched image it records
  history   commits outside HEAD's ancestry, before and after upstream's strip; whether the strip
            changes HEAD, the tree, `git status` or `git describe`
  shell     `env` and tool lookup under `bash -c` (the port), `bash -lc` (upstream), and `bash -c`
            with BASH_ENV pointing at a three-line login emulation (a way to give the agent's
            shell the same environment)
  residue   names of hidden tests (taken from the test patch, absent from tracked source) found in
            untracked or ignored files, and in unreachable git objects
No tests are run here.
"""
import asyncio, json, os, re, sys
import pyarrow.parquet as pq
from verifiers.v1.runtimes import provision_runtime
from verifiers.v1.runtimes.prime import PrimeConfig
from xiaomi_code._data import resolve_image

DATA, OUT = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ONLY = set(sys.argv[4:])
insts = [json.loads(r["extra_info"]["instance_json"]) for r in pq.read_table(f"{DATA}/code.parquet").to_pylist()]
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EMU_PATH = "/tmp/xiaomi-login-env.sh"
EMU = """unset BASH_ENV
[ -r /etc/profile ] && . /etc/profile
if [ -r ~/.bash_profile ]; then . ~/.bash_profile; elif [ -r ~/.bash_login ]; then . ~/.bash_login; elif [ -r ~/.profile ]; then . ~/.profile; fi
"""
IGNORE = {"SHLVL", "_", "PWD", "OLDPWD", "PS1", "LS_COLORS", "LESSOPEN", "LESSCLOSE", "BASH_ENV"}
TOOLS = "python python3 pip pytest py.test tox nox poetry uv go cargo node npm npx yarn pnpm ruby bundle rspec rake java mvn gradle php composer dotnet make cmake julia"
ENVCMD = "env; echo @@TOOLS; for t in " + TOOLS + "; do printf '%s=%s\\n' $t \"$(command -v $t)\"; done"
NAME_PATTERNS = [
    r"^\+\s*(?:async\s+)?def\s+(test_\w+)", r"^\+\s*func\s+(Test\w+)", r"^\+.*\bvoid\s+(test\w+)\s*\(",
    r"^\+\s*(?:it|test|describe)(?:\.\w+)?\(\s*['\"`]([^'\"`]{14,90})['\"`]", r"^\+\s*(?:it|describe|context|specify)\s+['\"]([^'\"]{14,90})['\"]",
]
PRE = r"""
git config --global --add safe.directory "$PWD"
base=$(git rev-parse HEAD)
echo "extra_before=$(git rev-list --all --not $base | wc -l)"
echo "refs=$(git for-each-ref | wc -l)"
echo "heads=$(git for-each-ref refs/heads | wc -l)"
echo "remotes=$(git for-each-ref refs/remotes | wc -l)"
echo "tags=$(git for-each-ref refs/tags | wc -l)"
echo "ancestor_tags=$(git tag --merged $base 2>/dev/null | wc -l)"
echo "branch_before=$(git rev-parse --abbrev-ref HEAD)"
echo "describe_before=$(git describe --tags --always --long 2>/dev/null)"
echo "dirty_before=$(git status --porcelain | wc -l)"
echo "tree_before=$(git rev-parse HEAD^{tree})"
echo "clean_would_remove=$(git clean -fdxn | wc -l)"
: > /tmp/names2.txt
if [ -s /tmp/names.txt ]; then while IFS= read -r n; do git grep -q -F -e "$n" 2>/dev/null || printf '%s\n' "$n"; done < /tmp/names.txt > /tmp/names2.txt; fi
echo "names_checked=$(wc -l < /tmp/names2.txt)"
if [ -s /tmp/names2.txt ]; then
  echo "untracked_hits=$(timeout 240 git ls-files -o -z -- . ':(exclude)node_modules' ':(exclude)**/node_modules/**' 2>/dev/null | xargs -0 -r grep -l -a -F -f /tmp/names2.txt 2>/dev/null | head -8 | tr '\n' '|')"
fi
timeout 150 git fsck --unreachable --no-reflogs 2>/dev/null > /tmp/fsck.txt; echo "fsck_rc=$?"
echo "unreachable_commits=$(grep -c ' commit ' /tmp/fsck.txt)"
echo "unreachable_blobs=$(grep -c ' blob ' /tmp/fsck.txt)"
echo "base_date=$(git log -1 --format=%cI $base)"
echo "unreachable_outside_base=$( (awk '$2=="commit"{print $3}' /tmp/fsck.txt; echo "^$base") | timeout 120 git rev-list --stdin 2>/dev/null | wc -l)"
echo "unreachable_newer_than_base=$( (awk '$2=="commit"{print $3}' /tmp/fsck.txt; echo "^$base") | timeout 120 git rev-list --stdin --since="$(git log -1 --format=%cI $base)" 2>/dev/null | wc -l)"
echo "unreachable_newest=$( (awk '$2=="commit"{print $3}' /tmp/fsck.txt; echo "^$base") | timeout 120 git rev-list --stdin --max-count=1 --format=%cI 2>/dev/null | tail -1)"
echo "pre_seconds=$SECONDS"
if [ -s /tmp/names2.txt ]; then
  echo "unreachable_hits=$(awk '$2=="blob"{print $3}' /tmp/fsck.txt | head -400 | while read b; do git cat-file -p $b 2>/dev/null | grep -q -a -F -f /tmp/names2.txt && echo $b; done | head -3 | tr '\n' '|')"
fi
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
echo "describe_after=$(git describe --tags --always --long 2>/dev/null)"
echo "dirty_after=$(git status --porcelain | wc -l)"
echo "tree_after=$(git rev-parse HEAD^{tree})"
echo "unreachable_commits_after=$(timeout 150 git fsck --unreachable --no-reflogs 2>/dev/null | grep -c ' commit ')"
echo "strip_seconds=$SECONDS"
"""

def hidden_names(patch):
    added = "\n".join(l for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = "\n".join("+" + l[1:] for l in patch.splitlines() if l.startswith("-") and not l.startswith("---"))
    names = []
    for pattern in NAME_PATTERNS:
        gone = set(re.findall(pattern, removed, re.M))
        for name in re.findall(pattern, added, re.M):
            name = name.strip()
            if len(name) >= 14 and name not in gone and name not in names:
                names.append(name)
    return names[:40]

def split_env(text):
    env_part, _, tool_part = text.partition("@@TOOLS")
    parse = lambda block: dict(line.split("=", 1) for line in block.splitlines() if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*=", line))
    return parse(env_part), parse(tool_part)

def diff(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if k not in IGNORE and a.get(k) != b.get(k)}

done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        rec = json.loads(line)
        if "error" not in rec:
            done.add(rec["id"])
todo = [i for i in insts if i["instance_id"] not in done and (not ONLY or i["instance_id"] in ONLY)]
sem = asyncio.Semaphore(CONC)
count = 0

async def work(inst, rec):
    cwd = inst["cwd"]
    cfg = PrimeConfig(image=resolve_image(inst["docker_image"]), workdir=cwd, cpu=2, memory=4, disk=float(os.environ.get("PROBE_DISK", "10")), labels=["xiaomi-review"])
    async with provision_runtime(cfg, env={"PATH": PATH}) as rt:
        names = hidden_names(inst["test_patch"])
        rec["names_from_patch"] = len(names)
        await rt.write("/tmp/names.txt", ("\n".join(names) + "\n" if names else "").encode())
        await rt.write(EMU_PATH, EMU.encode())
        for name, script, limit in (("pre", PRE, 700), ("strip", STRIP, 900)):
            if name == "strip":
                plain = split_env((await rt.run(["bash", "-c", ENVCMD], {})).stdout)
                login = split_env((await rt.run(["bash", "-lc", ENVCMD], {})).stdout)
                emu = split_env((await rt.run(["bash", "-c", ENVCMD], {"BASH_ENV": EMU_PATH})).stdout)
                rec["env_diff"] = diff(plain[0], login[0])
                rec["tool_diff"] = diff(plain[1], login[1])
                rec["emu_env_diff"] = diff(emu[0], login[0])
                rec["emu_tool_diff"] = diff(emu[1], login[1])
            r = await rt.run(["timeout", str(limit), "bash", "-c", f"cd {cwd} && " + script], {})
            rec[name + "_rc"] = r.exit_code
            for line in r.stdout.splitlines():
                key, sep, value = line.partition("=")
                if sep and re.fullmatch(r"[a-z_0-9]+", key):
                    rec[key] = value

async def probe(inst):
    global count
    rec = {"id": inst["instance_id"], "cwd": inst["cwd"]}
    async with sem:
        for attempt in range(2):
            rec.pop("error", None)
            try:
                await asyncio.wait_for(work(inst, rec), 2100)
                break
            except Exception as e:  # sandbox trouble is retried once, then recorded
                rec["error"] = repr(e)[:300]
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    count += 1
    if count % 25 == 0 or "error" in rec:
        print(f"[{count}/{len(todo)}]", rec["id"], rec.get("error", "ok"), flush=True)

async def main():
    print("todo", len(todo), "already done", len(done), flush=True)
    await asyncio.gather(*(probe(i) for i in todo))

asyncio.run(main())
