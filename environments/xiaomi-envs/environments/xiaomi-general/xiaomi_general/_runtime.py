"""Small command helpers over the Runtime supplied by Verifiers."""

import math
import shlex

from verifiers.v1.runtimes import ProgramResult, Runtime


async def checked_run(runtime: Runtime, argv: list[str], env: dict[str, str] | None = None) -> ProgramResult:
    result = await runtime.run(argv, env or {})
    if result.exit_code != 0:
        raise RuntimeError(f"Command failed ({result.exit_code}): {result.stdout}\n{result.stderr}")
    return result


async def run_shell(
    runtime: Runtime,
    command: str,
    cwd: str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> ProgramResult:
    if cwd is not None:
        command = f"cd {shlex.quote(cwd)} && {command}"
    # Non-login shells preserve the runtime environment (upstream Docker uses -c, Kubernetes -lc).
    argv = ["bash", "-c", command]
    if timeout is not None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Command timeout must be positive and finite")
        # Bound the process inside the sandbox as well as the surrounding scoring stage.
        argv = ["timeout", "--kill-after=5", str(timeout), *argv]
    return await runtime.run(argv, env or {})
