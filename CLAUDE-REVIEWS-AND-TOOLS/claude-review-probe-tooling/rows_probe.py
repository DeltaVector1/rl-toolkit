import json, sys, collections, re
import pyarrow.parquet as pq
D = sys.argv[1]
for name in ("code.parquet", "general/train.parquet", "music.parquet"):
    t = pq.read_table(f"{D}/{name}")
    rows = t.to_pylist()
    print("=====", name, len(rows), t.schema.names)
    r = rows[0]
    print({k: (type(v).__name__, (str(v)[:200])) for k, v in r.items()})
    if name == "music.parquet":
        ids = [r["extra_info"].get("src_id") for r in rows]
        print("music src_id unique", len(set(ids)), "none", ids.count(None), "sample", ids[:5])
        print("extra_info keys", collections.Counter(k for r in rows for k in r["extra_info"]))
        print("prompt type", type(rows[0]["prompt"]), str(rows[0]["prompt"])[:1500])
        continue
    insts = [json.loads(r["extra_info"]["instance_json"]) for r in rows]
    print("keys", collections.Counter(k for i in insts for k in i))
    print("dataset_type", collections.Counter(i.get("dataset_type") for i in insts))
    ids = [i["instance_id"] for i in insts]
    print("unique ids", len(set(ids)))
    print("cwd", collections.Counter(i.get("cwd") for i in insts).most_common(8))
    print("timeouts", collections.Counter(i.get("verifier_timeout_sec") for i in insts).most_common(8))
    print("prompt type", type(rows[0]["prompt"]), str(rows[0]["prompt"])[:600])
    if name == "code.parquet":
        print("test_command", collections.Counter(re.sub(r"\d+", "N", i["test_command"]) for i in insts).most_common(10))
        hdr = collections.Counter()
        bad = 0
        for i in insts:
            for line in i["test_patch"].splitlines():
                if line.startswith("diff --git "):
                    m = re.match(r'^diff --git ([^/\s"]+)/(.+?) ([^/\s"]+)/(.+)$', line)
                    if not m: bad += 1; print("BAD", line[:200]); continue
                    hdr[(m.group(1), m.group(3))] += 1
                    if " " in line[len("diff --git "):].replace(" b/", "", 1): hdr["space"] += 1
        print("hdr prefixes", hdr, "bad", bad)
        print("prompt==problem_statement", sum(str(r["prompt"]) == i["problem_statement"] for r, i in zip(rows, insts)))
        print("image sample", insts[0]["docker_image"])
        # script inside patch?
        i = insts[0]
        print(i["test_command"]); print(i["test_patch"][:1500])
    else:
        for kind in ("general_agent", "terminal_bench"):
            sub = [i for i in insts if i["dataset_type"] == kind]
            if not sub: continue
            print("---", kind, len(sub))
            print("keys", collections.Counter(k for i in sub for k in i))
            s = sub[0]
            print({k: str(v)[:300] for k, v in s.items()})
            print("reward_binary_fullscore", collections.Counter(str(i.get("reward_binary_fullscore")) for i in sub))
            print("images", len({i["docker_image"] for i in sub}), collections.Counter(i["docker_image"] for i in sub).most_common(3))
