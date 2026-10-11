"""Native adaptation of MiMo-Agent OpenSourceCodeEnvironment (MIT).

Source: XiaomiMiMo/MiMo-Agent@467f0a19016f0ac4d63b8d17a1f0da9ba07f232c,
src/mimoagent/environments/datasets/{opensource_code,base}.py. See LICENSE.mimoagent.
"""

import re
import shlex
import time
from pathlib import PurePosixPath

from pydantic import PositiveInt

import verifiers.v1 as vf
from xiaomi_common.data import XiaomiTasksetConfig, load_instance, load_rows, resolve_image
from xiaomi_common.runtime import checked_run, run_shell

ERROR_CATEGORY = "reward/testbed_corrupted"
PATCH_PATH = "/tmp/_opensource_code_test.patch"


def touched_files(patch: str) -> list[str]:
    """Read Git and unified file headers, excluding header-like hunk payload."""
    paths = []
    old_left = new_left = 0
    lines = iter(patch.splitlines())
    for line in lines:
        if old_left or new_left:
            if line.startswith("\\"):
                continue  # Git's "No newline at end of file" marker consumes no lines.
            if not line or line[0] not in " +-":
                raise ValueError(f"Unsupported test patch hunk: {line!r}")
            old_left -= line[0] in " -"
            new_left -= line[0] in " +"
            if old_left < 0 or new_left < 0:
                raise ValueError(f"Invalid test patch hunk length: {line!r}")
            continue
        if line.startswith("@@ "):
            match = re.match(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
            if not match:
                raise ValueError(f"Unsupported test patch hunk: {line!r}")
            old_left, new_left = (int(count) if count is not None else 1 for count in match.groups())
            continue
        if line.startswith("diff --git "):
            match = re.fullmatch(r'diff --git ([^\s"]+) ([^\s"]+)', line)
            if not match:
                raise ValueError(f"Unsupported test patch header: {line!r}")
            headers = match.groups()
        elif line.startswith("--- "):
            new = next(lines, "")
            if not new.startswith("+++ "):
                raise ValueError(f"Unsupported test patch header: {line!r}")
            headers = (line[4:].split("\t", 1)[0], new[4:].split("\t", 1)[0])
        else:
            continue
        for header in headers:
            if header == "/dev/null":
                continue
            # Published patches include an ab/ prefix; git apply strips one component.
            match = re.fullmatch(r'[^/\s"]+/([^\s"]+)', header)
            if not match:
                raise ValueError(f"Unsupported test patch header: {header!r}")
            path = match[1]
            if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\x00" in path:
                raise ValueError(f"Unsafe test patch path: {path!r}")
            if path not in paths:
                paths.append(path)
    if old_left or new_left:
        raise ValueError("Incomplete test patch hunk")
    if not paths:
        raise ValueError("Test patch has no resettable file headers")
    return paths


def reset_command(patch: str, base_ref: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", base_ref):
        raise ValueError("Missing or invalid captured base commit")
    commands = []
    for path in touched_files(patch):
        quoted = shlex.quote(path)
        # Preserve the original reset scope: test paths only, including both rename sides.
        commands.append(
            f"if git cat-file -e {shlex.quote(base_ref + ':' + path)} 2>/dev/null; then "
            f"git checkout {base_ref} -- {quoted} || exit $?; else "
            f"git rm -f --cached -- {quoted} >/dev/null 2>&1 || true; "
            f"rm -f -- {quoted} || exit $?; fi"
        )
    return "\n".join(commands) or "true"


class CodeData(vf.TaskData):
    test_patch: str
    test_command: str
    verifier_timeout_sec: PositiveInt


class CodeState(vf.State):
    base_ref: str = ""


class XiaomiCodeTask(vf.Task[CodeData, CodeState]):
    NEEDS_CONTAINER = True

    def runtime_env(self) -> dict[str, str]:
        # Flattened Code images omit ENV; supply the default PATH Docker would give them.
        return {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        cwd = self.data.workdir
        await checked_run(runtime, ["git", "config", "--global", "--add", "safe.directory", cwd])
        exists = await run_shell(runtime, "git rev-parse --git-dir", cwd)
        if exists.exit_code != 0:
            result = await run_shell(runtime, "git init -q && git add -A && git -c user.name=MiMo -c user.email=mimo@localhost commit -q -m baseline --allow-empty", cwd)
            if result.exit_code != 0:
                raise RuntimeError(f"Cannot initialize task repository: {result.stdout}\n{result.stderr}")
        result = await run_shell(runtime, "git rev-parse HEAD", cwd)
        hashes = re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", result.stdout)
        if result.exit_code != 0 or not hashes:
            raise RuntimeError(f"Cannot capture task base commit: {result.stdout}\n{result.stderr}")
        trace.state.base_ref = hashes[-1]
        history = await run_shell(runtime, f"git rev-list --all --not {trace.state.base_ref}", cwd)
        if history.exit_code != 0 or history.stdout.strip():
            raise RuntimeError("Image history is not truncated at the captured task base commit")
        trace.info["base_ref"] = trace.state.base_ref

    async def _reset(self, runtime: vf.Runtime, base_ref: str) -> None:
        result = await run_shell(runtime, reset_command(self.data.test_patch, base_ref), self.data.workdir)
        if result.exit_code != 0:
            raise RuntimeError(f"reset_tests_failed: {result.stdout}\n{result.stderr}")

    @vf.reward
    async def verified(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        base_ref = trace.state.base_ref
        try:
            await self._reset(runtime, base_ref)
            await runtime.write(PATCH_PATH, self.data.test_patch.encode())
            try:
                applied = await run_shell(runtime, f"git apply --verbose {PATCH_PATH}", self.data.workdir)
            finally:
                await checked_run(runtime, ["rm", "-f", PATCH_PATH])
            if applied.exit_code != 0:
                raise RuntimeError(f"apply_test_patch_failed: {applied.stdout}\n{applied.stderr}")
            started = time.monotonic()
            result = await run_shell(runtime, self.data.test_command, self.data.workdir, self.data.verifier_timeout_sec)
            reward = float(result.exit_code == 0)
            trace.info["verifier"] = {
                "verifier_returncode": result.exit_code,
                "resolved": bool(reward),
                "test_duration": time.monotonic() - started,
                "test_command": self.data.test_command,
                "stdout": result.stdout[-10000:],
                "stderr": result.stderr[-4000:],
            }
            return reward
        except Exception as exc:
            trace.info["error_category"] = ERROR_CATEGORY
            trace.info["verifier_error"] = str(exc)
            raise
        finally:
            try:
                await self._reset(runtime, base_ref)
            except Exception as exc:
                trace.info["error_category"] = ERROR_CATEGORY
                trace.info["verifier_cleanup_error"] = str(exc)
                raise


class XiaomiCodeTaskset(vf.Taskset[XiaomiCodeTask, XiaomiTasksetConfig]):
    def load(self):
        for row in load_rows("code", self.config):
            instance = load_instance(row)
            required = ("cwd", "docker_image", "instance_id", "problem_statement", "test_patch", "test_command", "verifier_timeout_sec")
            if instance.get("dataset_type") != "opensource-code" or any(not instance.get(key) for key in required):
                raise ValueError("Invalid opensource-code row")
            touched_files(instance["test_patch"])
            yield XiaomiCodeTask(CodeData(
                id=instance["instance_id"],
                prompt=instance["problem_statement"],
                image=resolve_image(instance["docker_image"], self.config.revision),
                workdir=instance["cwd"],
                test_patch=instance["test_patch"],
                test_command=instance["test_command"],
                verifier_timeout_sec=instance["verifier_timeout_sec"],
                timeout=vf.TaskTimeout(scoring=instance["verifier_timeout_sec"] + 120),
            ), self.config.task)
