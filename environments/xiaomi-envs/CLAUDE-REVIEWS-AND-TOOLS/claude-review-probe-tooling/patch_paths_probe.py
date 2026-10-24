"""Review 2, finding 8: check touched_files on every Code test patch against git itself.

    uv run python patch_paths_probe.py <DATA>

`git apply --numstat -z` lists every path git would touch. Each must be in the reset set.
"""
import json, os, subprocess, sys, tempfile
import pyarrow.parquet as pq
from xiaomi_code.taskset import touched_files

rows = pq.read_table(f"{sys.argv[1]}/code.parquet").to_pylist()
errors = missing = total = 0
for row in rows:
    inst = json.loads(row["extra_info"]["instance_json"])
    try:
        mine = set(touched_files(inst["test_patch"]))
    except ValueError as exc:
        errors += 1
        print("PARSE ERROR", inst["instance_id"], exc)
        continue
    total += len(mine)
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(inst["test_patch"])
    result = subprocess.run(["git", "apply", "--numstat", "-z", f.name], capture_output=True, text=True, check=False)
    os.unlink(f.name)
    if result.returncode:
        errors += 1
        print("GIT CANNOT PARSE", inst["instance_id"], result.stderr[:150])
        continue
    parts, git_paths, k = result.stdout.split("\0"), set(), 0
    while k < len(parts) and parts[k]:
        path = parts[k].split("\t", 2)[2]
        if path:
            git_paths.add(path)
            k += 1
        else:  # rename or copy: the two paths follow as separate fields
            git_paths.update(parts[k + 1 : k + 3])
            k += 3
    if not git_paths <= mine:
        missing += 1
        print("NOT RESET", inst["instance_id"], sorted(git_paths - mine)[:4])
print(f"rows {len(rows)} | parse errors {errors} | reset paths {total} | rows where git touches a path the reset misses {missing}")
