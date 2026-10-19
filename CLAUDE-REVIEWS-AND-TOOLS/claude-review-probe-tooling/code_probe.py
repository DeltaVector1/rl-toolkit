import json, sys, collections, re
import pyarrow.parquet as pq
from xiaomi_code.taskset import touched_files
D = sys.argv[1]
rows = pq.read_table(f"{D}/code.parquet").to_pylist()
insts = [json.loads(r["extra_info"]["instance_json"]) for r in rows]
print("prompt msgs", collections.Counter((len(r["prompt"]), r["prompt"][0].get("role")) for r in rows))
print("prompt content == problem_statement", sum(r["prompt"][0]["content"] == i["problem_statement"] for r, i in zip(rows, insts)))
def truth(patch):
    out = []
    for line in patch.splitlines():
        if line.startswith(("--- ", "+++ ")):
            p = line[4:].split("\t")[0]
            if p != "/dev/null":
                p = p.split("/", 1)[1] if "/" in p else p
                if p not in out: out.append(p)
        m = re.match(r"^(rename|copy) (from|to) (.+)$", line)
        if m and m.group(3) not in out: out.append(m.group(3))
    return out
mism = 0; spaces = 0
for i in insts:
    got = touched_files(i["test_patch"])
    want = truth(i["test_patch"])
    if any(" " in p for p in want): spaces += 1
    if set(got) != set(want):
        mism += 1
        if mism <= 6:
            print("MISMATCH", i["instance_id"], "\n  got ", sorted(set(got) - set(want))[:5], "\n  want", sorted(set(want) - set(got))[:5])
print("rows with spaces in paths", spaces, "mismatch rows", mism)
for i in insts:
    for line in i["test_patch"].splitlines():
        if line.startswith("diff --git ") and not line.startswith("diff --git a/"):
            print("ODD", i["instance_id"], repr(line[:200]))
# verifier scripts
kw = collections.Counter()
ex = {}
for i in insts:
    m = re.search(r"diff --git a/mimo_test_command\.sh.*?(?=\ndiff --git |\Z)", i["test_patch"], re.S)
    if not m: kw["no_script"] += 1; continue
    body = "\n".join(l[1:] for l in m.group(0).splitlines() if l.startswith("+") and not l.startswith("+++"))
    for k in ("go ", "cargo", "npm", "yarn", "pnpm", "pytest", "mvn", "gradle", "dotnet", "export PATH", "source ", ". ~/", "bashrc", "profile", "nvm", "conda", "GOROOT", "/usr/local/go", "bash -l", "set -e", "php", "ruby", "bundle", "make "):
        if k in body:
            kw[k] += 1; ex.setdefault(k, (i["instance_id"], body))
print(kw.most_common())
for k in ("go ", "cargo", "source ", "export PATH"):
    if k in ex: print("=====", k, ex[k][0]); print(ex[k][1][:900])
one = [i for i in insts if i["instance_id"] == "format-code-task-000121"][0]
print(one["test_command"], one["cwd"], one["docker_image"])
