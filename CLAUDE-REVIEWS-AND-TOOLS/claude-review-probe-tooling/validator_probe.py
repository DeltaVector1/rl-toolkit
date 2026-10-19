"""Finding 3: run the General submission validator on local workspaces.

    uv run python validator_probe.py

Exit 42 from the validator becomes reward 0 (invalid_submission) in GeneralTask.finalize.
"""
import os, subprocess, sys, tempfile
from pathlib import Path

from xiaomi_general.taskset import VALIDATE_SUBMISSION


def check(label, prepare):
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        (workspace / "answer.md").write_text("final answer\n")
        prepare(workspace)
        result = subprocess.run([sys.executable, "-I", "-c", VALIDATE_SUBMISSION, str(workspace)], capture_output=True, text=True, check=False)
        print(f"{label}: exit {result.returncode} | {result.stdout.strip()[:160]}")


check("plain files only", lambda ws: None)
check("internal symlink", lambda ws: os.symlink(ws / "answer.md", ws / "alias.md"))
check("virtualenv in workspace", lambda ws: subprocess.run([sys.executable, "-m", "venv", str(ws / ".venv")], check=True))
check("dangling symlink", lambda ws: os.symlink("/nonexistent", ws / "dangling"))
