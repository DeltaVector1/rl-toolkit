"""Finding 1: tally a path_probe.py result file into the table used in CLAUDE-REVIEW-1.md.

    python3 summarize_path_probe.py results/path_probe.jsonl
"""
import collections, json, sys

records = [json.loads(line) for line in open(sys.argv[1])]
ran = [record for record in records if "error" not in record]
print("rows", len(records), "| ran", len(ran), "| sandbox errors", [record["id"] for record in records if "error" in record])
MODES = ("port", "docker-c", "docker-lc")


def bucket(result):
    if result["exit"] == 0:
        return "pass"
    return "tool not found" if result["exit"] == 127 or result["not_found"] else "tests fail"


tally = collections.Counter((mode, bucket(record[mode])) for record in ran for mode in MODES)
for mode in MODES:
    print(f"{mode:10}", {name: tally[(mode, name)] for name in ("pass", "tests fail", "tool not found")})
print("missing under port:", dict(collections.Counter(tool for record in ran for tool in record["port"]["not_found"])))
print("port fails, docker PATH passes:", sorted(record["id"] for record in ran if record["port"]["exit"] != 0 and record["docker-c"]["exit"] == 0))
print("port passes, docker PATH fails:", sorted(record["id"] for record in ran if record["port"]["exit"] == 0 and record["docker-c"]["exit"] != 0))
print("-c vs -lc differ:", sorted(record["id"] for record in ran if record["docker-c"]["exit"] != record["docker-lc"]["exit"]))
