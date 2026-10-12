import base64
import copy
import json
import pickle
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


def test_task_toolsets_survive_pickle_and_copy():
    for dataset_type, expected_count in (("terminal_bench", 0), ("general_agent", 1)):
        task = GeneralTask(GeneralData(instance={"dataset_type": dataset_type}))
        for current in (
            pickle.loads(pickle.dumps(task)),
            task.with_data(),
            copy.deepcopy(task),
        ):
            toolsets = current.toolsets(current.config)
            assert len(toolsets) == expected_count


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
            return ProgramResult(0, "key=terminal-output", "key=terminal-error")

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
    trace = SimpleNamespace(info={})
    assert await task.verifier_reward(trace, Runtime()) == 0
    assert trace.info["verifier"] == {
        "returncode": 0, "stdout": "key=terminal-output", "stderr": "key=terminal-error",
    }


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
async def test_terminal_native_selection_worker_rebuild_and_execution_limits(monkeypatch):
    import verifiers.v1 as vf
    import xiaomi_general.taskset as module
    from verifiers.v1.agent import resolve_rollout_timeouts
    from verifiers.v1.configs.agent import TimeoutConfig
    from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime
    from verifiers.v1.serve.server import EnvServer
    from verifiers.v1.utils.compile import resolve_runtime_config

    instance = {
        "dataset_type": "terminal_bench", "instance_id": "terminal-test",
        "docker_image": "image", "agent_timeout_sec": 900,
        "cpus": 1, "memory_mb": 2048, "storage_mb": 10240,
        "allow_internet": False, "verifier_timeout_sec": 120,
        "tests_files": {"test.sh": base64.b64encode(b"exit 0").decode()},
    }
    monkeypatch.setattr(module, "load_rows", lambda *_: [
        {"prompt": "Repair the tree", "extra_info": {"instance_json": json.dumps(instance)}}
    ])
    monkeypatch.setattr(module, "resolve_image", lambda *_: "published-image")
    taskset = GeneralTaskset(GeneralConfig())
    task = next(iter(taskset.include(ids=["terminal-test"])))
    server = SimpleNamespace(
        task_cls=taskset.task_type(), data_cls=taskset.task_type().data_type(),
        env=SimpleNamespace(config=SimpleNamespace(taskset=taskset.config)),
    )
    rebuilt = EnvServer._build_task(server, task.data.model_dump(mode="json"))
    assert type(task) is type(rebuilt) is GeneralTask
    for current in (task, rebuilt):
        assert current.toolsets(current.config) == []
        assert current.data.agent_user is None
        assert current.data.resources == vf.TaskResources(cpu=1, memory=2, disk=10)
        assert current.data.network_allow == []
        assert resolve_rollout_timeouts(TimeoutConfig(), current).agent == 900
        assert resolve_rollout_timeouts(TimeoutConfig(rollout=30), current).agent == 30
    assert len(GeneralTask.toolsets(taskset.config.task)) == 1  # Native class discovery.
    general = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    assert len(general.toolsets(general.config)) == 1
    config = resolve_runtime_config(PrimeConfig(), rebuilt)
    assert (config.cpu, config.memory, config.disk, config.workdir) == (1, 2, 10, "/app")
    assert config.allow == [] and config.block == ["*"]
    override = resolve_runtime_config(PrimeConfig(cpu=3, memory=4, disk=20), rebuilt)
    assert (override.cpu, override.memory, override.disk) == (3, 4, 20)

    policies = []

    class Client:
        async def set_network(self, sandbox, **policy):
            policies.append(policy)
            return SimpleNamespace(applied=True)

    runtime = object.__new__(PrimeRuntime)
    vf.Runtime.__init__(runtime)
    runtime.config = config
    runtime.info = SimpleNamespace(id="fixture")
    runtime._client = Client()
    await runtime.prepare_execution(["https://model.fixture/v1"])
    assert policies == [{"allow": ["model.fixture"]}]


@pytest.mark.asyncio
async def test_terminal_setup_and_native_bash_assets_stay_outside_workspace(tmp_path):
    import verifiers.v1 as vf
    from pathlib import PurePosixPath
    from verifiers.v1.harnesses.bash.harness import BashHarness, BashHarnessConfig
    from verifiers.v1.runtimes.prime import PrimeRuntime

    writes, calls = [], []

    class Runtime(PrimeRuntime):
        async def write(self, path, data):
            writes.append(path)
            local = tmp_path / path.lstrip("/")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)

        async def run(self, argv, env):
            calls.append((argv, self.process_env(env)))
            return ProgramResult(0, "/opt/xiaomi-uv-cache/fixture/bin/python\n", "")

    runtime = object.__new__(Runtime)
    vf.Runtime.__init__(runtime)
    task = GeneralTask(GeneralData(instance={
        "dataset_type": "terminal_bench",
        "tests_files": {"test.sh": base64.b64encode(b"exit 0").decode()},
    }, workdir="/app"))
    runtime.env = task.runtime_env()
    await task.setup(SimpleNamespace(info={}), runtime)
    assert not writes and not calls  # No common package or hidden verifier uploads.
    await BashHarness(BashHarnessConfig()).setup(runtime)
    assert writes and all(not PurePosixPath(p).is_relative_to("/app") for p in writes)
    assert calls[0][1]["UV_CACHE_DIR"] == "/opt/xiaomi-uv-cache"
    assert calls[0][1]["UV_PYTHON_INSTALL_DIR"] == "/opt/xiaomi-python"


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
    # Execute the real preparation command only after mapping every path into this fixture.
    import shlex

    prepare = next(argv[-1] for argv, _ in calls if argv[-1].startswith("rm -rf -- /tmp/agent_output"))
    agent_output = tmp_path / "agent_output"
    private_logs = tmp_path / "private_logs"
    private_logs.mkdir()
    (private_logs / "keep.jsonl").write_text("ordinary private fixture")
    prepare = prepare.replace("/tmp/agent_output", shlex.quote(str(agent_output)))
    prepare = prepare.replace("/work/_verifier/tmp", shlex.quote(str(tmp_path / "verifier_tmp")))
    prepare = prepare.replace("/logs/verifier", shlex.quote(str(tmp_path / "logs")))
    for alias in (False, True):
        if alias:
            agent_output.rmdir()
            agent_output.symlink_to(private_logs, target_is_directory=True)
        else:
            sessions = agent_output / "sessions"
            sessions.mkdir(parents=True)
            (sessions / "old.jsonl").write_text("ordinary prior session")
        subprocess.run(["sh", "-c", prepare], check=True)
        assert agent_output.is_dir() and not agent_output.is_symlink()
        assert not list(agent_output.iterdir())
        assert agent_output.stat().st_mode & 0o777 == 0o700
        assert (private_logs / "keep.jsonl").read_text() == "ordinary private fixture"


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
def test_submission_paths_remove_private_aliases_including_hidden_names(tmp_path, name):
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
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (workspace / name).is_symlink()
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
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (native / "alias").is_symlink()
    assert (tmp_path / "private").read_text() == "private fixture"


def test_submission_keeps_answer_with_real_venv_and_dangling_link(tmp_path):
    import subprocess
    import sys
    import venv

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "answer.md").write_text("ordinary answer")
    venv.EnvBuilder(with_pip=False, symlinks=True).create(workspace / ".venv")
    (workspace / "dangling").symlink_to("missing")
    (workspace / "internal").symlink_to("answer.md")
    result = subprocess.run(
        [sys.executable, "-I", "-c", VALIDATE_SUBMISSION, str(workspace)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / "answer.md").read_text() == "ordinary answer"
    assert (workspace / "internal").is_symlink()
    assert not (workspace / "dangling").is_symlink()
    assert (workspace / ".venv" / "pyvenv.cfg").is_file()
    for link in workspace.rglob("*"):
        if link.is_symlink():
            assert link.resolve(strict=True).is_relative_to(workspace)


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
                return ProgramResult(42, "Submission path is not a regular file or directory: answer.md", "")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_detail", [False, True])
@pytest.mark.parametrize("clip", [False, True])
async def test_general_trace_redacts_judge_keys_before_clipping(monkeypatch, raw_detail, clip):
    import xiaomi_general.taskset as module

    keys = {"GA_JUDGE_KEY": "fake-primary-credential", "GA_JUDGE_KEY2": "tiny",
            "GA_JUDGE_KEY3": "fake-fallback-credential", "JUDGE_API_KEY": "fake-legacy-credential"}
    monkeypatch.setenv("GA_JUDGE_KEY", keys["GA_JUDGE_KEY"])
    monkeypatch.setattr(GeneralTask, "manifest", lambda _: {"verifier": {"env": keys}})
    lines = ("[judge] chain[0]: model=fixture api=chat key=fake-primary...\n"
             "[judge] chain[1]: model=fixture api=chat key=tiny...\n"
             "[judge] switching e0->e1: model=fixture api=chat key=fake-fallbac... (last_err=HTTPError)\n")
    clean = lines.replace("fake-primary...", "<redacted>").replace("tiny...", "<redacted>").replace("fake-fallbac...", "<redacted>")
    full_error = "HTTPError: rejected " + " ".join(keys.values())
    clean_error = "HTTPError: rejected " + " ".join(["<redacted>"] * len(keys))
    stdout = lines + ("x" * 9997 if clip else full_error + "\n[verify] score=0.750")
    stderr = lines + ("y" * 3997 if clip else full_error)
    detail = {"reward": 0.75, "passed": True, "optional": None,
              "results": [{"message": lines + full_error, "score": 1}]}
    raw_prefix = "z" * 1990 + "\n"
    detail_text = raw_prefix + "key=" + keys["GA_JUDGE_KEY"] + " " + lines + full_error if raw_detail else json.dumps(detail)

    async def run_shell(runtime, command, **kwargs):
        assert {name: kwargs["env"][name] for name in keys} == keys
        return ProgramResult(0, stdout, stderr)

    monkeypatch.setattr(module, "run_shell", run_shell)

    class Runtime:
        async def run(self, argv, env):
            return ProgramResult(0, detail_text if argv[0] == "cat" else "", "")

        async def read(self, path):
            return b'{"reward": 0.75}'

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    trace = SimpleNamespace(info={"general_submission_validated": True})
    assert await task.verifier_reward(trace, Runtime()) == 0.75
    info = trace.info["verifier"]
    assert info["stdout"] == (clean + ("x" * 9997 if clip else clean_error + "\n[verify] score=0.750"))[-10000:]
    assert info["stderr"] == (clean + ("y" * 3997 if clip else clean_error))[-4000:]
    assert info["verifier_returncode"] == 0
    if raw_detail:
        assert info["reward_detail_raw"] == (raw_prefix + "key=<redacted> " + clean + clean_error)[:2000]
    else:
        detail["results"][0]["message"] = clean + clean_error
        assert info["reward_detail"] == detail
    saved = json.dumps(trace.info)
    for key in keys.values():
        assert key not in saved
    assert "fake-primary" not in saved and "fake-fallbac" not in saved


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["run", "read", "detail", "reward"])
async def test_general_grading_exception_trace_does_not_retain_judge_key(monkeypatch, failure):
    import traceback
    import xiaomi_general.taskset as module

    key = "fake-private-credential"
    monkeypatch.setenv("GA_JUDGE_KEY", key)
    monkeypatch.setattr(GeneralTask, "manifest", lambda _: {})
    error = "HTTPError: rejected " + key + " key=fake-private..."

    async def run_shell(*args, **kwargs):
        if failure == "run":
            raise ValueError(error)
        return ProgramResult(1, "", "")

    monkeypatch.setattr(module, "run_shell", run_shell)

    class Runtime:
        async def run(self, argv, env):
            if failure == "detail" and argv[0] == "cat":
                raise ValueError(error)
            return ProgramResult(0, "", "")

        async def read(self, path):
            if failure == "read":
                raise ValueError(error)
            return json.dumps({"reward": key if failure == "reward" else 0.75}).encode()

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    trace = SimpleNamespace(info={"general_submission_validated": True})
    with pytest.raises(RuntimeError) as caught:
        await task.verifier_reward(trace, Runtime())
    saved = "".join(traceback.format_exception(caught.value))
    assert key not in saved and "fake-private" not in saved
    assert str(caught.value) == (
        "RuntimeError: missing_or_invalid_reward_json" if failure in ("read", "reward")
        else "ValueError: HTTPError: rejected <redacted> key=<redacted>"
    )
    assert "verifier" not in trace.info


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [(TimeoutError(), "TimeoutError: "),
     (KeyError("fake-private-credential"), "KeyError: '<redacted>'")],
)
async def test_general_non_native_grading_errors_keep_type_and_redact_key(
    monkeypatch, error, expected
):
    import xiaomi_general.taskset as module

    key = "fake-private-credential"
    monkeypatch.setenv("GA_JUDGE_KEY", key)
    monkeypatch.setattr(module.GeneralTask, "manifest", lambda _: {})

    async def run_shell(*args, **kwargs):
        raise error

    monkeypatch.setattr(module, "run_shell", run_shell)

    class Runtime:
        async def run(self, argv, env):
            return ProgramResult(0, "", "")

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    trace = SimpleNamespace(info={"general_submission_validated": True})

    with pytest.raises(RuntimeError) as caught:
        await task.verifier_reward(trace, Runtime())

    assert str(caught.value) == expected
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert key not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["run", "read", "detail"])
@pytest.mark.parametrize("error_type", ["SandboxError", "ProviderError"])
async def test_general_redaction_preserves_native_error_retries(monkeypatch, failure, error_type):
    import verifiers.v1 as vf
    import xiaomi_general.taskset as module
    from verifiers.v1.configs.retries import RetryConfig, RetryRule
    from verifiers.v1.errors import ProviderError, SandboxError
    from verifiers.v1.utils.retries import RetryState

    key = "fake-retry-credential"
    monkeypatch.setenv("GA_JUDGE_KEY", key)
    monkeypatch.setattr(GeneralTask, "manifest", lambda _: {})
    message = "Runtime unavailable: " + key + " key=fake-retry-c..."
    native_error = (SandboxError(message) if error_type == "SandboxError"
                    else ProviderError(message, status_code=503))

    async def run_shell(*args, **kwargs):
        if failure == "run":
            raise native_error from ValueError(message)
        return ProgramResult(0, "", "")

    monkeypatch.setattr(module, "run_shell", run_shell)

    class Runtime:
        async def run(self, argv, env):
            if failure == "detail" and argv[0] == "cat":
                raise native_error from ValueError(message)
            return ProgramResult(0, "", "")

        async def read(self, path):
            if failure == "read":
                raise native_error from ValueError(message)
            return b'{"reward": 0.75}'

    task = GeneralTask(GeneralData(instance={"dataset_type": "general_agent"}))
    trace = vf.Trace(task={"type": "GeneralTask", "data": task.data},
                     agent={"config": {}}, info={"general_submission_validated": True})
    with pytest.raises(type(native_error)) as caught:
        await task.score(trace, Runtime())
    assert caught.value is native_error
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    trace.record_error(caught.value)
    assert trace.last_error.type == error_type
    assert trace.last_error.message == "Runtime unavailable: <redacted> key=<redacted>"
    assert trace.stop_condition == ("sandbox_error" if error_type == "SandboxError" else "provider_error")
    saved = trace.last_error.model_dump_json()
    assert key not in saved and "fake-retry-c" not in saved
    status = [503] if error_type == "ProviderError" else None
    assert trace.last_error.status_code == (503 if status else None)
    retry = RetryState(RetryConfig(rules=[RetryRule(type=error_type, status_code=status, max_retries=1)]))
    assert retry.next_error(trace.errors) is trace.last_error
    assert retry.next_error(trace.errors) is None
