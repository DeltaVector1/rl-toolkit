"""Review 2: audit every general_agent manifest against the published file tree.

    uv run python manifest_audit_probe.py

Downloads only the 925 manifest.json files plus one recursive tree listing. Checks that every
upload source exists (the port raises on a missing source; upstream skips it) and tallies the
setup command, verifier command, timeouts and env settings the port depends on.
"""
import collections, json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from xiaomi_code._data import DATASET_ID, DATASET_REVISION

tree = HfApi().list_repo_tree(DATASET_ID, "general/envs", recursive=True, repo_type="dataset", revision=DATASET_REVISION)
paths = {entry.path for entry in tree}
root = Path(snapshot_download(DATASET_ID, repo_type="dataset", revision=DATASET_REVISION, allow_patterns=["general/envs/*/manifest.json"]))
tasks = sorted({path.split("/")[2] for path in paths})
tally, missing = collections.Counter(), []
for task in tasks:
    base = f"general/envs/{task}"
    manifest = json.loads((root / base / "manifest.json").read_text())
    verifier = manifest.get("verifier", {})
    for item in [*manifest.get("uploads", []), *verifier.get("uploads", [])]:
        if f"{base}/{item['source']}" not in paths:
            missing.append((task, item["source"]))
    for key, value in (
        ("setup.command", (manifest.get("setup") or {}).get("command")),
        ("verifier.command", verifier.get("command")),
        ("verifier.container", verifier.get("container")),
        ("verifier.timeout_sec", verifier.get("timeout_sec")),
        ("verifier.env_passthrough", json.dumps(verifier.get("env_passthrough"))),
        ("verifier.env", json.dumps(verifier.get("env"))),
        ("verifier.reward_file", verifier.get("reward_file")),
        ("cwd", manifest.get("cwd")),
    ):
        tally[(key, value)] += 1
print("tasks", len(tasks), "| tree entries", len(paths))
for key, count in sorted(tally.items(), key=str):
    print(" ", key, count)
print("upload sources missing from the published tree:", len(missing), missing[:5])
