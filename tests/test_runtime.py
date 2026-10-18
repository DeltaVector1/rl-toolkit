import os
import subprocess

import pytest
from verifiers.v1.runtimes import ProgramResult

from xiaomi_code._runtime import run_shell as code_run_shell
from xiaomi_general._runtime import run_shell as general_run_shell


@pytest.fixture(params=[code_run_shell, general_run_shell], ids=["code", "general"])
def run_shell(request):
    return request.param


@pytest.mark.parametrize("timeout", [None, 5.0])
async def test_nonlogin_shell_preserves_runtime_path_workdir_and_process_env(tmp_path, timeout, run_shell):
    tool_dir = tmp_path / "image-bin"
    tool_dir.mkdir()
    tool = tool_dir / "fixture-task-tool"
    tool.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$TASK_VALUE"\n')
    tool.chmod(0o755)
    workdir = tmp_path / "work ' directory"
    workdir.mkdir()
    forwarded = {"TASK_VALUE": "forwarded value"}

    class Runtime:
        async def run(self, argv, env):
            assert env == forwarded
            if timeout is not None:
                assert argv[:3] == ["timeout", "--kill-after=5", "5.0"]
                argv = argv[3:]  # Run the inner command without requiring GNU timeout on macOS.
            # Login profiles can replace the runtime's task or image PATH.
            assert argv[:2] == ["bash", "-c"]
            result = subprocess.run(
                argv, capture_output=True, text=True,
                env={**os.environ, "PATH": f"{tool_dir}:/usr/bin:/bin", **env},
            )
            return ProgramResult(result.returncode, result.stdout, result.stderr)

    result = await run_shell(Runtime(), "fixture-task-tool", str(workdir), timeout, forwarded)
    assert result.exit_code == 0, result.stderr
    assert result.stdout.splitlines() == [str(workdir), "forwarded value"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
async def test_shell_rejects_invalid_timeout_before_running(run_shell, timeout):
    class Runtime:
        async def run(self, argv, env):
            pytest.fail("invalid timeout reached the runtime")

    with pytest.raises(ValueError, match="positive and finite"):
        await run_shell(Runtime(), "echo test", timeout=timeout)
