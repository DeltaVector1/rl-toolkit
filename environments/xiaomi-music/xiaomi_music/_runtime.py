"""Small command helpers over the Runtime supplied by Verifiers."""

from verifiers.v1.runtimes import ProgramResult, Runtime


async def checked_run(runtime: Runtime, argv: list[str], env: dict[str, str] | None = None) -> ProgramResult:
    result = await runtime.run(argv, env or {})
    if result.exit_code != 0:
        raise RuntimeError(f"Command failed ({result.exit_code}): {result.stdout}\n{result.stderr}")
    return result
