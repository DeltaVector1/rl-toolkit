"""Tally a full_pass_probe.py result file.

    python3 summarize_full_pass.py results/full_pass.jsonl
"""
import collections, json, re, statistics, sys

recs = [json.loads(l) for l in open(sys.argv[1])]
ok = [r for r in recs if "error" not in r]
num = lambda r, k: int(r[k]) if str(r.get(k, "")).strip().lstrip("-").isdigit() else 0
print(f"rows {len(recs)} | measured {len(ok)} | sandbox died {len(recs) - len(ok)}")

print("\n== history (finding 1)")
for label, rows in (("/workspace/repo", [r for r in ok if r["cwd"] == "/workspace/repo"]), ("/testbed", [r for r in ok if r["cwd"] != "/workspace/repo"])):
    bad = [r for r in rows if num(r, "extra_before") > 0]
    print(f"  {label}: {len(rows)} rows, history not truncated {len(bad)}, fixed by strip {sum(num(r, 'extra_after') == 0 for r in bad)}")
bad = [r for r in ok if num(r, "extra_before") > 0]
print(f"  total not truncated: {len(bad)} of {len(ok)} ({100 * len(bad) / len(ok):.1f}%)")
changed = [r["id"] for r in ok if r.get("head_same") != "yes" or r.get("tree_before") != r.get("tree_after") or r.get("dirty_before") != r.get("dirty_after")]
print(f"  strip changed HEAD, tree or git status on: {len(changed)} rows {changed[:5]}")
print(f"  strip left commits outside the base on: {sum(num(r, 'extra_after') > 0 for r in ok)} rows")

print("\n== unreachable objects (history deleted without garbage collection)")
un = [r for r in ok if num(r, "unreachable_outside_base") > 0]
newer = [r for r in ok if num(r, "unreachable_newer_than_base") > 0]
print(f"  rows with unreachable commits outside the base's ancestry: {len(un)} ({100 * len(un) / len(ok):.1f}%)")
print(f"  rows where some are newer than the base commit: {len(newer)} ({100 * len(newer) / len(ok):.1f}%)")
if un:
    print(f"  median unreachable commits on those rows: {statistics.median(num(r, 'unreachable_commits') for r in un)}")
either = [r for r in ok if num(r, "extra_before") > 0 or num(r, "unreachable_newer_than_base") > 0]
print(f"  rows with future history reachable OR unreachable: {len(either)} ({100 * len(either) / len(ok):.1f}%)")
print(f"  rows still holding unreachable commits after upstream's strip as written: {sum(num(r, 'unreachable_commits_after') > 0 for r in ok)} (gc fails on a broken refs/remotes/origin/HEAD; see strip_fixed_probe.py)")
print(f"  fsck timed out (count unknown): {sum(str(r.get('fsck_rc')) == '124' for r in ok)}")

print("\n== git describe across the strip (version detection)")
norm = lambda d: "" if re.fullmatch(r"[0-9a-f]{4,40}", d or "") else re.sub(r"-g[0-9a-f]+$", "", d or "")
tagdist = [r["id"] for r in ok if norm(r.get("describe_before")) != norm(r.get("describe_after"))]
abbrev = [r for r in ok if r.get("describe_before") != r.get("describe_after") and norm(r.get("describe_before")) == norm(r.get("describe_after"))]
print(f"  tag or distance changed: {len(tagdist)} rows {tagdist[:5]}")
print(f"  only the abbreviated hash length changed: {len(abbrev)} rows")
print(f"  branch name before the strip: {collections.Counter('detached' if r.get('branch_before') == 'HEAD' else 'on a branch' for r in ok)}")

print("\n== shells (finding 2)")
tool = [r for r in ok if r.get("tool_diff")]
only = [r for r in tool if any(not a and b for a, b in r["tool_diff"].values())]
print(f"  a tool resolves differently under bash -lc: {len(tool)} ({100 * len(tool) / len(ok):.1f}%)")
print(f"  a tool exists only under bash -lc: {len(only)} ({100 * len(only) / len(ok):.1f}%) {collections.Counter(k for r in only for k, (a, b) in r['tool_diff'].items() if not a and b).most_common(8)}")
print(f"  env keys that differ: {collections.Counter(k for r in ok for k in r.get('env_diff', {})).most_common(12)}")
worse = [r["id"] for r in ok if any(a and not b for a, b in r.get("tool_diff", {}).values())]
print(f"  a tool is LOST under bash -lc: {len(worse)} {worse[:5]}")
emu = [r["id"] for r in ok if r.get("emu_env_diff") or r.get("emu_tool_diff")]
print(f"  BASH_ENV emulation differs from bash -lc: {len(emu)} rows {emu[:5]}")

print("\n== residue (hidden test names in untracked or unreachable content)")
named = [r for r in ok if num(r, "names_checked") > 0]
hits = [r for r in named if (r.get("untracked_hits") or "").strip("|")]
kinds = collections.Counter()
for r in hits:
    for path in r["untracked_hits"].strip("|").split("|"):
        kinds[".build_env" if ".build_env" in path else ".pyc" if path.endswith(".pyc") else ".pytest_cache" if ".pytest_cache" in path else "jest cache" if ".jest" in path else "other"] += 1
print(f"  rows with at least one checkable hidden-test name: {len(named)}")
print(f"  rows where an untracked or ignored file contains one: {len(hits)} ({kinds.most_common(8)})")
print(f"  rows where an unreachable git blob contains one: {sum(bool((r.get('unreachable_hits') or '').strip('|')) for r in named)}")
print(f"  rows where git clean -fdx would remove something: {sum(num(r, 'clean_would_remove') > 0 for r in ok)}")

print("\n== cost of the strip")
t = [num(r, "strip_seconds") for r in ok]
print(f"  strip step seconds (includes one extra fsck): median {statistics.median(t)}, p95 {sorted(t)[int(len(t) * 0.95)]}, max {max(t)}")
