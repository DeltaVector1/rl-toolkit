import json, sys, collections, re
import pyarrow.parquet as pq
D = sys.argv[1]
rows = pq.read_table(f"{D}/code.parquet").to_pylist()
insts = [json.loads(r["extra_info"]["instance_json"]) for r in rows]
by = {i["instance_id"]: i for i in insts}
p = by["format-code-task-000989"]["test_patch"]
for n, l in enumerate(p.splitlines()):
    if "error_wrapping_test.go" in l or l.startswith("diff --git"): print(n, repr(l[:160]))
print("-----121")
for l in by["format-code-task-000121"]["test_patch"].splitlines():
    if l.startswith(("diff --git", "--- ", "+++ ", "new file", "index ")): print(repr(l[:140]))
m = re.search(r"diff --git a/mimo_test_command\.sh.*?(?=\ndiff --git |\Z)", by["format-code-task-000121"]["test_patch"], re.S)
print("\n".join(l[1:] for l in m.group(0).splitlines() if l.startswith("+") and not l.startswith("+++"))[:1500])
c = collections.Counter(); shown = 0
for i in insts:
    m = re.search(r"diff --git a/mimo_test_command\.sh.*?(?=\ndiff --git |\Z)", i["test_patch"], re.S)
    if not m: continue
    body = "\n".join(l[1:] for l in m.group(0).splitlines() if l.startswith("+") and not l.startswith("+++"))
    for l in body.splitlines():
        if "bash -l" in l:
            c[re.sub(r"\s+", " ", l.strip())[:70]] += 1
        if ("GOROOT" in l or "profile" in l or "bashrc" in l) and shown < 12:
            shown += 1; print("ENVLINE", i["instance_id"], l.strip()[:160])
print(c.most_common(8))
langs = collections.Counter()
for i in insts:
    t = i["test_patch"]
    for ext in (".go", ".py", ".js", ".ts", ".rs", ".java", ".rb", ".php", ".c", ".cpp", ".cs", ".kt", ".swift"):
        if re.search(r"^diff --git .*" + re.escape(ext) + "$", t, re.M): langs[ext] += 1
print(langs)
