"""Review 4: summarise review4_full_probe.py output and compare the verifier with the review 3 baseline.

    python3 summarize_review4.py <review4_full.jsonl> <results dir with verifier_health*.jsonl>

The baseline is every row's verifier on its untouched image under `bash -lc` with the Docker PATH
(what upstream gave the images), before any of the review 3 fixes.
"""
import collections, json, re, statistics, sys

new = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    new[r["id"]] = r
base = {}
for name in ("verifier_health.jsonl", "verifier_health_reruns.jsonl"):
    for line in open(f"{sys.argv[2]}/{name}"):
        r = json.loads(line)
        if "exit" in r or r["id"] not in base:
            base[r["id"]] = r

rows = list(new.values())
print("rows", len(rows), "probe errors", sum("error" in r for r in rows))
for r in rows:
    if "error" in r:
        print("  probe error", r["id"], r["error"][:160])

print("\n== setup")
failed = [r for r in rows if "setup_error" in r]
print("setup failed:", len(failed))
kinds = collections.Counter(re.sub(r"[0-9a-f]{7,}|\d+", "N", r["setup_error"])[:150] for r in failed)
for k, v in kinds.most_common():
    print(f"  {v}  {k}")
for r in failed:
    print("   ", r["id"], r["cwd"], r.get("git_version"), r["setup_error"][:260].replace("\\n", " | "))
ok = [r for r in rows if "setup_error" not in r and "error" not in r]
secs = sorted(r["setup_seconds"] for r in ok)
if secs:
    print("setup seconds: median %.1f p95 %.1f max %.1f" % (statistics.median(secs), secs[int(len(secs) * 0.95)], secs[-1]))
    print("  slowest:", [(r["id"][-6:], r["setup_seconds"], r.get("git_mb_before"), r.get("caches_before")) for r in sorted(ok, key=lambda r: -r["setup_seconds"])[:8]])
print("git versions:", collections.Counter((r.get("git_version") or "?").split()[-1].rsplit(".", 1)[0] for r in rows).most_common(12))
for key in ("head", "tree", "status", "diff", "describe"):
    changed = [r["id"] for r in ok if r.get(f"{key}_before") != r.get(f"{key}_after")]
    print(f"{key} changed by setup: {len(changed)}", changed[:6])
print("commits outside base afterwards:", sum(r.get("extra_after", "0") != "0" for r in ok))
measured = [r for r in ok if "unreachable_after" in r]
print("unreachable commits afterwards: %d of %d measured" % (sum(r["unreachable_after"] != "0" for r in measured), len(measured)))
print("rows with caches before:", sum(int(r.get("caches_before") or 0) > 0 for r in ok), "after:", sum(int(r.get("caches_after") or 0) > 0 for r in ok),
      "| jest caches before:", sum(int(r.get("jest_caches_before") or 0) > 0 for r in ok))

print("\n== bootstrap in child shells")
print("prints to stdout:", sum(bool(r.get("noise_out")) for r in rows), "stderr:", sum(bool(r.get("noise_err")) for r in rows))
for flag in ("e", "u", "eu"):
    vals = [r.get(f"strict_{flag}", "") for r in rows]
    print(f"bash -{flag} child: nonzero exit {sum(not v.startswith('0 ') and v != '' for v in vals)}, stderr noise {sum(len(v.split(' ', 1)[-1].strip()) > 0 for v in vals if v)}")
    print("   ", collections.Counter(re.sub(r"line \d+", "line N", v.split(" ", 1)[-1].strip())[:90] for v in vals if v).most_common(4))
lost = [r for r in rows if "child_path" in r and "/zzprobe" not in r["child_path"]]
demoted = [r for r in rows if "child_path" in r and "/zzprobe" in r["child_path"] and not r["child_path"].startswith("/zzprobe")]
print("parent's PATH entry lost in child:", len(lost), collections.Counter(r["cwd"] for r in lost))
print("parent's PATH entry pushed behind profile entries:", len(demoted), collections.Counter(r["child_path"].split("/zzprobe")[0][:60] for r in demoted).most_common(6))
names = ["NODE_OPTIONS", "GRADLE_OPTS", "MAVEN_OPTS", "JAVA_HOME", "GOFLAGS", "VIRTUAL_ENV", "CARGO_HOME"]
clobber = collections.Counter()
for r in rows:
    for name, value in zip(names, r.get("child_vars", "").split("|")):
        if value != "zz":
            clobber[name] += 1
print("parent's variable overwritten in child:", dict(clobber))
ms = sorted(int(r["child_ms"]) for r in rows if r.get("child_ms", "").isdigit())
if ms:
    print("ms per child shell: median %d p95 %d max %d; over 50 ms on %d rows" % (statistics.median(ms), ms[int(len(ms) * 0.95)], ms[-1], sum(m > 50 for m in ms)))
diffs = collections.Counter()
for r in rows:
    for key, (login, port) in (r.get("env_diff") or {}).items():
        diffs[key if key != "PATH" else "PATH " + ("+cargo" if "/usr/local/cargo/bin" in (port or "") and "/usr/local/cargo/bin" not in (login or "") else "other")] += 1
print("environment vs `bash -lc`:", diffs.most_common(12))

print("\n== verifier vs review 3 baseline")
outcome = collections.Counter()
changed = []
for r in ok:
    b = base.get(r["id"], {})
    if r.get("verify") == "cap":
        now = "cap"
    elif "verify_error" in r:
        now = "error"
    else:
        now = r.get("exit")
    before = b.get("exit", "died" if "error" in b else None)
    before = "cap" if before in (124, 137) else before
    outcome[now if now in ("cap", "error") else ("pass" if now == 0 else "127" if now == 127 else "fail")] += 1
    if now != before:
        changed.append((r["id"], before, now, r.get("not_found"), r.get("missing_module"), (r.get("tail") or r.get("verify_error") or "")[-140:].replace("\n", " | ")))
print("outcomes now:", dict(outcome))
print("exit code differs from baseline:", len(changed))
for c in changed:
    print("  ", c)
newenv = [(r["id"], r["not_found"], r["missing_module"]) for r in ok if (r.get("not_found") or r.get("missing_module"))
          and (set(r.get("not_found") or []) - set(base.get(r["id"], {}).get("not_found") or []) or set(r.get("missing_module") or []) - set(base.get(r["id"], {}).get("missing_module") or []))]
print("new 'not found' / missing-module signatures:", len(newenv))
for n in newenv:
    print("  ", n)
