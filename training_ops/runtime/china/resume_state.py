"""Read-only checks for one explicit adapter restart."""

import hashlib
import json
import re
import stat
import sys
import tomllib
from pathlib import Path


def regular(path: Path) -> None:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"restart requires a regular file: {path}")


def digest(path: Path) -> str:
    regular(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def explicit_config(text: str, step: int) -> str:
    """Change only the two restart fields, without guessing CLI syntax."""
    if type(step) is not int or step < 0:
        raise ValueError("invalid explicit restart step")
    before = tomllib.loads(text)
    expected = dict(before)
    expected["ckpt"] = dict(before.get("ckpt", {}), resume_step=step, skip_progress=True)
    lines = text.splitlines(keepends=True)
    output = []
    inside = False
    found = False
    for line in lines:
        if re.match(r"\s*\[", line):
            inside = bool(re.fullmatch(r"\s*\[ckpt\]\s*(?:#.*)?\n?", line))
            if inside:
                found = True
                output.extend([line.rstrip("\n") + "\n", f"resume_step = {step}\nskip_progress = true\n"])
                continue
        if inside and re.match(r"\s*(resume_step|skip_progress)\s*=", line):
            continue
        output.append(line)
    if not found:
        output.append(f"\n[ckpt]\nresume_step = {step}\nskip_progress = true\n")
    result = "".join(output)
    if tomllib.loads(result) != expected:
        raise ValueError("restart config changed an unrelated setting")
    return result


def check_state(service_dir: str, step: int, role: str, adapter: str, expected: str = "") -> dict:
    if type(step) is not int or step < 0 or role not in {"origin", "trainer", "orchestrator"}:
        raise ValueError("invalid explicit restart state")
    active = Path(service_dir)
    if not active.is_absolute() or active.name != f"run_{active.parent.name}":
        raise ValueError("restart run directory is not exact")
    parent = active.parent
    target = active / "broadcasts" / f"step_{step}"
    for directory in (parent, active, active / "broadcasts", target):
        if directory.is_symlink():
            raise ValueError(f"restart directory is a symlink: {directory}")
    weight = target / "adapter_model.safetensors"
    if Path(adapter) != weight:
        raise ValueError("resume adapter must be the exact requested policy path")
    # Do not remove evidence. The operator must quarantine old binaries first.
    for root in (parent / "rollouts", active / "rollouts", parent / ".incoming" / "rollouts", active / ".incoming" / "rollouts"):
        if root.is_symlink():
            raise ValueError(f"restart rollout directory is a symlink: {root}")
        if not root.exists():
            continue
        for directory in root.iterdir():
            match = re.fullmatch(r"step_(\d+)", directory.name)
            if not match or int(match[1]) <= step:
                continue
            if directory.is_symlink():
                raise ValueError(f"restart rollout step is a symlink: {directory}")
            for cached in directory.rglob("*"):
                if cached.name == "train_rollouts.bin" or re.fullmatch(r"rank_\d+\.bin", cached.name):
                    raise ValueError(f"cached rollout batch blocks restart: {cached}")
    if role != "orchestrator":
        for checkpoint in (parent / "checkpoints", active / "checkpoints"):
            if checkpoint.is_symlink() or (checkpoint.exists() and any(checkpoint.iterdir())):
                raise ValueError(f"optimizer checkpoint conflicts with adapter restart: {checkpoint}")
    if role in {"origin", "orchestrator"}:
        regular(target / "STABLE")
        for published in target.parent.glob("step_*/STABLE"):
            match = re.fullmatch(r"step_(\d+)", published.parent.name)
            if match and int(match[1]) > step:
                raise ValueError(f"later stable policy conflicts with restart: {published}")
    result = {"policy_step": step, "next_batch": step + 1}
    if role == "origin":
        regular(target / "adapter_manifest.json")
        manifest = json.loads((target / "adapter_manifest.json").read_text())
        if type(manifest.get("target_step")) is not int or manifest["target_step"] != step:
            raise ValueError("stable adapter manifest has a different policy step")
        actual = digest(weight)
        if manifest.get("target_sha256") != actual or manifest.get("target_size") != weight.stat().st_size:
            raise ValueError("stable adapter weight identity differs")
        config = target / "adapter_config.json"
        if manifest.get("config_sha256") != digest(config) or manifest.get("config_size") != config.stat().st_size:
            raise ValueError("stable adapter config identity differs")
        result["sha256"] = actual
    elif role == "trainer":
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest(weight) != expected:
            raise ValueError("trainer resume adapter differs from the stable policy")
    return result


if __name__ == "__main__":
    try:
        if sys.argv[1] == "config":
            print(explicit_config(Path(sys.argv[2]).read_text(), int(sys.argv[3])), end="")
        else:
            value = check_state(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
            print(json.dumps(value, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"adapter restart refused: {error}")
