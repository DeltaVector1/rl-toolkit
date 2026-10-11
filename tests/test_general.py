import base64
import json
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.utils.loaders import taskset_class
from xiaomi_general.taskset import (
    GeneralConfig,
    GeneralData,
    GeneralTask,
    GeneralTaskset,
    inline_tests,
    parse_reward,
)


def test_native_package_loads():
    assert taskset_class("xiaomi-general") is GeneralTaskset


@pytest.mark.parametrize(
    "raw,error",
    [
        ("{}", "missing_or_invalid"),
        ("not json", "missing_or_invalid"),
        ('{"reward": null}', "missing_or_invalid"),
        ('{"reward": "NaN"}', "out_of_range"),
        ('{"reward": 1.01}', "out_of_range"),
        ('{"reward": -0.1}', "out_of_range"),
    ],
)
def test_infrastructure_failure_is_not_zero(raw, error):
    with pytest.raises(RuntimeError, match=error):
        parse_reward(raw)


def test_binary_fullscore_tolerance_matches_source():
    assert parse_reward('{"reward": 0.999999}', True) == (1, {"raw_reward": 0.999999})
    assert parse_reward('{"reward": 0.999998}', True) == (0, {"raw_reward": 0.999998})
    assert parse_reward('{"reward": 0.42}') == (0.42, {})


def test_inline_paths_reject_escape():
    with pytest.raises(ValueError, match="Unsafe"):
        inline_tests({"tests_files": {"../outside": ""}})


@pytest.mark.asyncio
async def test_terminal_guard_reward_wins_over_successful_exit():
    class Runtime:
        async def run(self, argv, env):
            return ProgramResult(0, "", "")

        async def read(self, path):
            assert path == "/logs/verifier/reward.txt"
            return b"0\n"

        async def write(self, path, data):
            assert path == "/tests/test.sh"

    task = GeneralTask(
        GeneralData(
            instance={
                "dataset_type": "terminal_bench",
                "tests_files": json.dumps(
                    {"test.sh": base64.b64encode(b"exit 0").decode()}
                ),
            }
        )
    )
    assert await task.verifier_reward(SimpleNamespace(info={}), Runtime()) == 0


def test_loading_preserves_prompt_and_is_lazy(monkeypatch):
    import xiaomi_general.taskset as module

    prompt = [{"role": "user", "content": "Exact task\n"}]
    monkeypatch.setattr(module, "load_rows", lambda *_: [{"prompt": prompt}])
    monkeypatch.setattr(
        module,
        "load_instance",
        lambda _: {
            "dataset_type": "general_agent",
            "instance_id": "test",
            "docker_image": "image",
            "env_task_dir": "envs/test",
            "cwd": "/work/workspace",
        },
    )
    monkeypatch.setattr(module, "resolve_image", lambda *_: "published-image")
    monkeypatch.setattr(
        module, "download_general_task", lambda *_: pytest.fail("Eager asset download")
    )
    task = next(iter(GeneralTaskset(GeneralConfig())))
    assert task.data.prompt_text == "Exact task\n"
    assert task.data.agent_user == "xiaomi_agent"


@pytest.mark.asyncio
async def test_native_tools_keep_names_signatures_and_task_caps(tmp_path, monkeypatch):
    import inspect
    import verifiers.v1 as vf

    from xiaomi_general.servers import tool

    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "registry.py").write_text(
        'def search(query: str, limit: int = 3):\n'
        '    """Search registry."""\n'
        '    return query * limit\n'
    )
    helper = tmp_path / "mcp_http.py"
    helper.write_text(
        'import importlib.util, inspect\n'
        'def _load_tools_module(path):\n'
        '    spec = importlib.util.spec_from_file_location(path.stem, path)\n'
        '    module = importlib.util.module_from_spec(spec)\n'
        '    spec.loader.exec_module(module)\n'
        '    return module\n'
        'def _public_functions(module):\n'
        '    return [("search", module.search)]\n'
        'def _wrap_capped(fn):\n'
        '    def capped(*args, **kwargs):\n'
        '        return fn(*args, **kwargs)[:4]\n'
        '    capped.__signature__ = inspect.signature(fn)\n'
        '    return capped\n'
    )
    monkeypatch.setattr(
        tool, "Path", lambda path: tools if path == "/work/tools" else helper
    )
    registered = {}

    class MCP:
        def add_tool(self, fn, *, name, description):
            registered[name] = (fn, description)

    server = tool.GeneralTools(vf.ToolsetConfig())
    await server.setup_task(SimpleNamespace(instance={"dataset_type": "general_agent"}))
    server.register(MCP())
    fn, description = registered["mcp__registry__search"]
    assert description == "Search registry."
    assert str(inspect.signature(fn)) == "(query: str, limit: int = 3)"
    assert fn("abc", limit=3) == "abca"
    registered.clear()
    await server.setup_task(SimpleNamespace(instance={"dataset_type": "terminal_bench"}))
    server.register(MCP())
    assert registered == {}


@pytest.mark.asyncio
async def test_general_verifier_payload_stays_private_and_judge_env_is_forwarded(
    tmp_path, monkeypatch
):
    import xiaomi_general.taskset as module

    monkeypatch.setenv("GA_JUDGE_KEY", "test-key")
    monkeypatch.setenv("GA_JUDGE_URL", "http://judge.invalid/v1")
    monkeypatch.setenv("GA_JUDGE_MODEL", "test-model")
    monkeypatch.setenv("GA_JUDGE_API", "chat")
    root = tmp_path / "envs" / "test"
    root.mkdir(parents=True)
    verifier_code = (
        "from pathlib import Path\n"
        "print((Path(__file__).resolve().parent / 'verifier_meta.json').read_text())\n"
    )
    (root / "run_verify.py").write_text(verifier_code)
    (root / "verifier_meta.json").write_text('{"reward": 0.75}')
    (root / "manifest.json").write_text(json.dumps({
        "verifier": {"uploads": [
            {"source": name, "target": f"/work/{name}"}
            for name in ("run_verify.py", "verifier_meta.json")
        ]}
    }))
    calls, writes = [], {}

    class Runtime:
        supports_user = True

        async def run(self, argv, env):
            calls.append((argv, env))
            return ProgramResult(0, "", "")

        def with_user(self, user):
            assert user == "xiaomi_agent"
            return self

        async def write(self, path, data):
            writes[path] = data

        async def read(self, path):
            return b'{"reward": 0.75}'

    async def install(runtime):
        pass

    monkeypatch.setattr(module, "install_common", install)
    task = GeneralTask(GeneralData(instance={
        "dataset_type": "general_agent", "env_task_dir": "envs/test"
    }, task_root=str(tmp_path)))
    runtime = Runtime()
    trace = SimpleNamespace(info={}, id="test-trace")
    await task.setup(trace, runtime)
    assert not writes
    assert (["ln", "-sfn", "/work/_verifier/run_verify.py", "/work/run_verify.py"], {}) in calls
    assert await task.verifier_reward(trace, runtime) == 0.75
    assert writes == {
        "/work/_verifier/run_verify.py": verifier_code.encode(),
        "/work/_verifier/verifier_meta.json": b'{"reward": 0.75}',
    }
    # Upstream resolves __file__ before opening sibling metadata/helpers.
    import subprocess
    import sys

    work = tmp_path / "work"
    for path, content in writes.items():
        local = tmp_path / path.removeprefix("/")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
        (work / local.name).symlink_to(local)
    result = subprocess.check_output([sys.executable, str(work / "run_verify.py")])
    assert json.loads(result) == {"reward": 0.75}
    judge_env = next(env for _, env in calls if "GA_JUDGE_KEY" in env)
    assert judge_env["GA_JUDGE_MODEL"] == "test-model"
    assert judge_env["GA_JUDGE_URL"] == "http://judge.invalid/v1"
    assert judge_env["GA_JUDGE_API"] == "chat"
    assert judge_env["VERIFY_AGENT_JUDGE"] == "1"


@pytest.mark.asyncio
async def test_prime_harness_transcript_write_uses_agent_and_preserves_protected_uploads(
    tmp_path,
):
    import os
    import subprocess
    from verifiers.v1.runtimes.base import Runtime
    from verifiers.v1.runtimes.prime import PrimeRuntime
    from xiaomi_general.taskset import agent_message_writes

    class LocalPrime(PrimeRuntime):
        def __init__(self):
            Runtime.__init__(self)
            self.config = SimpleNamespace(workdir=str(tmp_path))
            self.uploads = []
            self.commands = []

        async def write(self, path, data):
            # Prime's gateway ignores the view's user and creates root-owned files.
            self.uploads.append((path, data, "root"))

        async def run(self, argv, env):
            self.commands.append((self.user, argv))
            assert self.user == "xiaomi_agent"
            result = subprocess.run(
                argv, env={**os.environ, **env}, capture_output=True, text=True
            )
            return ProgramResult(result.returncode, result.stdout, result.stderr)

    runtime = LocalPrime()
    tmp_path.chmod(0o1777)
    name = ".vf-initial-messages-trace.json"
    # Reproduce the provider ownership mismatch even through with_user().
    await runtime.with_user("xiaomi_agent").write(name, b"original")
    assert runtime.uploads.pop() == (name, b"original", "root")
    agent_message_writes(runtime, "trace")
    agent = runtime.with_user("xiaomi_agent")
    payload = json.dumps(
        [{"role": "user", "content": "large ' $() prompt" * 10000}]
    ).encode()
    await agent.write(name, payload)
    path = tmp_path / name
    assert path.read_bytes() == payload
    assert path.stat().st_mode & 0o777 == 0o600
    path.unlink()  # The harness consumes and removes its own transcript.
    await agent.write(str(path), b"resumed")
    assert path.read_bytes() == b"resumed"
    path.unlink()
    assert not runtime.uploads
    assert max(len(arg) for _, argv in runtime.commands for arg in argv) < 45000
    await runtime.write(".vf-src/package.whl", b"protected")
    await runtime.write(".vf-initial-messages-other.json", b"other trace")
    assert runtime.uploads == [
        (".vf-src/package.whl", b"protected", "root"),
        (".vf-initial-messages-other.json", b"other trace", "root"),
    ]
    # Another sandbox instance keeps its provider implementation.
    other = LocalPrime()
    await other.with_user("xiaomi_agent").write(name, b"other runtime")
    assert other.uploads == [(name, b"other runtime", "root")]

    protected = tmp_path / "private.py"
    protected.write_bytes(b"protected")
    path.symlink_to(protected)
    protected.chmod(0o400)
    if os.geteuid() != 0:
        with pytest.raises(RuntimeError):
            await agent.write(name, b"overwrite")
        assert protected.read_bytes() == b"protected"
        assert len(runtime.uploads) == 2  # A denied agent write never retries as root.
