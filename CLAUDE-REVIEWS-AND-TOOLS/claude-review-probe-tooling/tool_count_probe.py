"""Finding 1: count Code verifier scripts that call /usr/local/bin tools with no PATH setup of their own.

    uv run python tool_count_probe.py <dir containing code.parquet>

Static and heuristic: it reads mimo_test_command.sh out of each test patch. Scripts that hand off
to an in-image .build_env script land in the "no /usr/local/bin tool" bucket and are not analysed.
"""
import collections, json, re, sys
import pyarrow.parquet as pq

LOCAL = ("npm", "npx", "node", "yarn", "pnpm", "dev-switch", "yq", "gofmt")
pattern = re.compile(r"(?:^|[\s;&|(`])(" + "|".join(map(re.escape, LOCAL)) + r")(?=[\s;&|)]|$)", re.M)
rows = pq.read_table(f"{sys.argv[1]}/code.parquet").to_pylist()
buckets, tools, delegating = collections.Counter(), collections.Counter(), 0
for row in rows:
    patch = json.loads(row["extra_info"]["instance_json"])["test_patch"]
    match = re.search(r"diff --git a/mimo_test_command\.sh.*?(?=\ndiff --git |\Z)", patch, re.S)
    if not match:
        buckets["no script in patch"] += 1
        continue
    body = "\n".join(line[1:] for line in match.group(0).splitlines() if line.startswith("+") and not line.startswith("+++"))
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    delegating += "/.build_env/" in code and "bash -lc" in code
    used = set(pattern.findall(code))
    if not used:
        buckets["no /usr/local/bin tool"] += 1
    elif re.search(r"PATH=", code) or "bash -lc" in code:
        buckets["uses tool but sets PATH / login shell itself"] += 1
    else:
        buckets["uses /usr/local/bin tool, no PATH setup"] += 1
        tools.update(used)
print(len(rows), "rows;", dict(buckets))
print("tools called with no PATH setup:", dict(tools))
print("scripts delegating to an in-image .build_env script via bash -lc:", delegating)
