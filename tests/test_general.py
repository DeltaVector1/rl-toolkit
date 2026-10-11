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
    QUIESCE_AGENT,
    VALIDATE_SUBMISSION,
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
        "verifier": {"env": {"TMPDIR": "/tmp"}, "uploads": [
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
    trace = SimpleNamespace(info={}, id="test-trace", last_reply="")
    await task.setup(trace, runtime)
    assert not writes
    assert (["ln", "-sfn", "/work/_verifier/run_verify.py", "/work/run_verify.py"], {}) in calls
    await task.finalize(trace, runtime)
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
    assert judge_env["TMPDIR"] == "/work/_verifier/tmp"
    assert any("chmod 700 /tmp/agent_output /work/_verifier/tmp" in argv[-1] for argv, _ in calls)


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


def test_submission_paths_keep_regular_files_and_internal_links(tmp_path):
    import subprocess
    import sys

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "report.txt").write_text("ordinary report")
    (workspace / "alias.txt").symlink_to("report.txt")
    (workspace / ".notes").write_text("ordinary hidden notes")
    result = subprocess.run(
        [sys.executable, "-I", "-c", VALIDATE_SUBMISSION, str(workspace)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / "report.txt").read_text() == "ordinary report"


@pytest.mark.parametrize("name", ["answer.md", ".hidden", ".vf-venv"])
def test_submission_paths_reject_private_aliases_including_hidden_names(tmp_path, name):
    import subprocess
    import sys

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = tmp_path / "private.txt"
    private.write_text("private fixture")
    (workspace / name).symlink_to(private)
    result = subprocess.run(
        [sys.executable, "-I", "-c", VALIDATE_SUBMISSION, str(workspace)],
        capture_output=True, text=True,
    )
    assert result.returncode == 42
    assert name in result.stdout
    assert private.read_text() == "private fixture"


def test_submission_paths_do_not_skip_writable_native_named_directories(tmp_path):
    import subprocess
    import sys

    workspace = tmp_path / "workspace"
    native = workspace / ".vf-venv"
    native.mkdir(parents=True)
    native.chmod(0o777)
    (native / "alias").symlink_to(tmp_path / "private")
    (tmp_path / "private").write_text("private fixture")
    result = subprocess.run(
        [sys.executable, "-I", "-c", VALIDATE_SUBMISSION, str(workspace)],
        capture_output=True, text=True,
    )
    assert result.returncode == 42


@pytest.mark.parametrize("stuck", [False, True])
def test_quiescence_rechecks_children_and_ignores_zombies(monkeypatch, stuck):
    import os
    import pathlib
    import pwd
    import signal
    import time

    class Status:
        def __init__(self, pid, state="S"):
            self.parent = SimpleNamespace(name=str(pid))
            self.state = state

        def read_text(self):
            return f"State:\t{self.state}\nUid:\t12345 12345 12345 12345\n"

    scans = iter([[Status(101)], [Status(102)], [Status(102, "Z")]])
    monkeypatch.setattr(pathlib, "Path", lambda _: SimpleNamespace(
        glob=lambda _: [Status(101)] if stuck else next(scans)
    ))
    monkeypatch.setattr(pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=12345))
    signals = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    if stuck:
        with pytest.raises(RuntimeError, match="did not stop"):
            exec(QUIESCE_AGENT, {})
        assert len(signals) == 40
        return
    exec(QUIESCE_AGENT, {})
    assert signals == [(101, signal.SIGSTOP), (101, signal.SIGKILL),
                       (102, signal.SIGSTOP), (102, signal.SIGKILL)]


@pytest.mark.asyncio
async def test_invalid_submission_scores_zero_without_privileged_upload_or_fallback(monkeypatch):
    monkeypatch.setenv("GA_JUDGE_KEY", "test-key")
    calls = []

    class Runtime:
        user = "root"

        def with_user(self, user):
            view = Runtime()
            view.user = user
            return view

        async def run(self, argv, env):
            calls.append((self.user, argv))
            if VALIDATE_SUBMISSION in argv:
                return ProgramResult(42, "Submission path is outside the workspace: answer.md", "")
            return ProgramResult(0, "", "")

        async def write(self, *args):
            pytest.fail("Invalid submission must not trigger root writes")

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    trace = SimpleNamespace(info={}, last_reply="fallback answer")
    await task.finalize(trace, Runtime())
    assert await task.verifier_reward(trace, Runtime()) == 0
    assert "answer.md" in trace.info["verifier"]["invalid_submission"]
    assert [user for user, _ in calls] == ["root", "xiaomi_agent"]
    assert QUIESCE_AGENT in calls[0][1]


@pytest.mark.asyncio
async def test_final_answer_uses_agent_chunk_writer(tmp_path):
    import os
    import subprocess

    answer = tmp_path / "answer.md"
    calls = []

    class Runtime:
        user = "root"

        def with_user(self, user):
            view = Runtime()
            view.user = user
            return view

        async def run(self, argv, env):
            calls.append((self.user, argv))
            if QUIESCE_AGENT in argv or VALIDATE_SUBMISSION in argv:
                return ProgramResult(0, "", "")
            assert self.user == "xiaomi_agent"
            argv = [str(answer) if arg == "/work/workspace/answer.md" else arg for arg in argv]
            result = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True)
            return ProgramResult(result.returncode, result.stdout, result.stderr)

        async def write(self, *args):
            pytest.fail("Fallback must never use provider root uploads")

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    payload = "large ' $() answer " * 10000
    trace = SimpleNamespace(info={}, last_reply=payload)
    await task.finalize(trace, Runtime())
    assert answer.read_text() == payload
    assert answer.stat().st_mode & 0o777 == 0o600
    trace.last_reply = "replacement"
    await task.finalize(trace, Runtime())
    assert answer.read_text() == payload  # An existing nonempty answer wins.
    assert all(user == "xiaomi_agent" for user, argv in calls if "sh" == argv[0])


def test_private_temporary_scripts_do_not_import_shared_directory_modules(tmp_path):
    import os
    import subprocess
    import sys

    shared = tmp_path / "shared"
    private = tmp_path / "private"
    shared.mkdir()
    private.mkdir(mode=0o700)
    (shared / "json.py").write_text("# An ordinary module without dumps.\n")
    script = (
        "import tempfile, subprocess, sys\n"
        "with tempfile.NamedTemporaryFile('w', suffix='.py') as f:\n"
        "    f.write('import json; print(json.dumps(1))'); f.flush()\n"
        "    result = subprocess.run([sys.executable, f.name], capture_output=True, text=True)\n"
        "    print(result.returncode, result.stdout.strip())\n"
    )
    for directory, expected in [(shared, "1"), (private, "0 1")]:
        result = subprocess.check_output(
            [sys.executable, "-I", "-c", script],
            env={**os.environ, "TMPDIR": str(directory)}, text=True,
        )
        assert result.strip() == expected
