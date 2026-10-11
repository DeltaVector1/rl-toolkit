import os
import subprocess

import pytest
from verifiers.v1.runtimes import ProgramResult

from xiaomi_common.runtime import run_shell


@pytest.mark.parametrize("timeout", [None, 5.0])
async def test_nonlogin_shell_preserves_runtime_path_workdir_and_process_env(tmp_path, timeout):
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
