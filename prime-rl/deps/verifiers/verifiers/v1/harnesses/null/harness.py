import json
from pathlib import Path

from openai import APIStatusError, AsyncOpenAI

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class NullHarnessConfig(HarnessConfig):
    pass


class NullHarness(Harness[NullHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True
    EXECUTES_CODE = False

    async def setup(self, runtime: Runtime) -> None:
        pass

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        if not mcp_urls:
            messages = (
                [{"role": "system", "content": system_prompt}]
                if system_prompt
                else []
            )
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            elif prompt is not None:
                messages.extend(message_to_wire(message) for message in prompt)
            client = AsyncOpenAI(base_url=endpoint, api_key=secret)
            try:
                await client.chat.completions.create(model=ctx.model, messages=messages)
            except APIStatusError:
                if trace.stop_condition is None:
                    raise
            finally:
                await client.close()
            return ProgramResult(exit_code=0, stdout="", stderr="")

        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
        ]
        if system_prompt:
            args.append(f"--system-prompt={system_prompt}")
        args.append(
            "--mcp-config="
            + json.dumps(
                {
                    "mcpServers": {
                        name: {"url": url} for name, url in mcp_urls.items()
                    }
                }
            )
        )
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(
                path,
                json.dumps([message_to_wire(message) for message in prompt]).encode(),
            )
            args.append(f"--initial-messages-file={path}")
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        return await runtime.run_program([*program, *args], env)
