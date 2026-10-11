"""Synthetic repositories only; no published task commands run on the host."""

import subprocess
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes import ProgramResult

from xiaomi_code.taskset import CodeData, CodeState, XiaomiCodeTask, reset_command, touched_files

PATCH = """diff --git a/check.txt b/check.txt
new file mode 100644
--- /dev/null
+++ b/check.txt
@@ -0,0 +1 @@
+hidden
"""


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@localhost")
    (tmp_path / "solution.txt").write_text("before\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


def test_reset_removes_staged_hidden_path_without_erasing_solution(repo):
    base = git(repo, "rev-parse", "HEAD")
    (repo / "solution.txt").write_text("model fix\n")
    (repo / "check.txt").write_text("agent planted\n")
    git(repo, "add", ".")
    subprocess.run(["bash", "-c", reset_command(PATCH, base)], cwd=repo, check=True)
    assert not (repo / "check.txt").exists()
    assert (repo / "solution.txt").read_text() == "model fix\n"
    assert "solution.txt" in git(repo, "diff", "--cached", "--name-only")


def test_reset_restores_both_rename_sides(repo):
    base = git(repo, "rev-parse", "HEAD")
    patch = "diff --git a/solution.txt b/renamed.txt\nsimilarity index 100%\nrename from solution.txt\nrename to renamed.txt\n"
    git(repo, "mv", "solution.txt", "renamed.txt")
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    assert (repo / "solution.txt").read_text() == "before\n"
    assert not (repo / "renamed.txt").exists()


def test_patch_paths_cannot_escape():
    with pytest.raises(ValueError, match="Unsafe"):
        touched_files("diff --git a/../../outside b/../../outside")


def test_empty_patch_cannot_be_reset():
    with pytest.raises(ValueError, match="no resettable Git diff headers"):
        reset_command("", "a" * 40)


def test_headerless_unified_patch_is_valid_git_but_cannot_be_reset(repo):
    patch = "--- /dev/null\n+++ b/check.txt\n@@ -0,0 +1 @@\n+hidden\n"
    base = git(repo, "rev-parse", "HEAD")

    subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=repo, check=True)
    assert (repo / "check.txt").read_text() == "hidden\n"
    with pytest.raises(ValueError, match="no resettable Git diff headers"):
        reset_command(patch, base)


def test_reset_covers_published_nonstandard_prefix(repo):
    name = "balance_strategy_roundrobin_global_test.go"
    patch = PATCH.replace("check.txt", name).replace("diff --git a/", "diff --git ab/")
    base = git(repo, "rev-parse", "HEAD")
    (repo / name).write_text("agent planted\n")
    for _ in range(2):
        subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
        assert not (repo / name).exists()
        subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=repo, check=True)
        assert (repo / name).read_text() == "hidden\n"
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    assert not (repo / name).exists()


@pytest.mark.parametrize("header", [
    'diff --git "a/quoted\\tpath" "b/quoted\\tpath"',
    "diff --git missing-prefix missing-prefix",
])
def test_unsupported_patch_headers_fail_instead_of_resetting_partial_paths(header):
    with pytest.raises(ValueError, match="Unsupported test patch header"):
        touched_files(PATCH + header + "\n")


class RecordingRuntime:
    def __init__(self, verifier_rc=0, apply_rc=0, transport_error=False):
        self.calls = []
        self.verifier_rc = verifier_rc
        self.apply_rc = apply_rc
        self.transport_error = transport_error

    async def run(self, argv, env):
        self.calls.append(("run", argv))
        if "git apply" in argv[-1]:
            return ProgramResult(self.apply_rc, "apply", "")
        if argv[0] == "timeout":
            if self.transport_error:
                raise OSError("sandbox disconnected")
            return ProgramResult(self.verifier_rc, "test output", "")
        return ProgramResult(0, "", "")

    async def write(self, path, content):
        self.calls.append(("write", path, content))


def task_and_trace():
    task = XiaomiCodeTask(CodeData(prompt="fix", workdir="/testbed", test_patch=PATCH, test_command="bash /testbed/check.sh", verifier_timeout_sec=10))
    return task, SimpleNamespace(state=CodeState(base_ref="a" * 40), info={})


@pytest.mark.parametrize("returncode,score", [(0, 1.0), (1, 0.0), (124, 0.0)])
async def test_original_exit_code_contract_and_late_patch(returncode, score):
    task, trace = task_and_trace()
    runtime = RecordingRuntime(verifier_rc=returncode)
    assert await task.verified(trace, runtime) == score
    assert "git cat-file" in runtime.calls[0][1][-1]
    assert runtime.calls[1][0] == "write"
    assert "git apply" in runtime.calls[2][1][-1]
    assert "git cat-file" in runtime.calls[-1][1][-1]
    assert trace.info["verifier"]["verifier_returncode"] == returncode


@pytest.mark.parametrize("runtime", [RecordingRuntime(apply_rc=1), RecordingRuntime(transport_error=True)])
async def test_broken_testbed_is_error_and_cleanup_runs(runtime):
    task, trace = task_and_trace()
    with pytest.raises((RuntimeError, OSError)):
        await task.verified(trace, runtime)
    assert trace.info["error_category"] == "reward/testbed_corrupted"
    assert "git cat-file" in runtime.calls[-1][1][-1]


async def test_history_leak_aborts_before_any_test_upload():
    task, trace = task_and_trace()
    runtime = RecordingRuntime()
    async def run(argv, env):
        if "rev-parse HEAD" in argv[-1] or "rev-list" in argv[-1]:
            return ProgramResult(0, "a" * 40 + "\n", "")
        return ProgramResult(0, "", "")
    runtime.run = run
    with pytest.raises(RuntimeError, match="history"):
        await task.setup(trace, runtime)
    assert runtime.calls == []


async def test_setup_captures_original_head_before_agent_moves_it():
    task, trace = task_and_trace()
    runtime = RecordingRuntime()
    calls = []
    async def run(argv, env):
        calls.append(argv)
        if 'rev-parse HEAD' in argv[-1]:
            return ProgramResult(0, 'shell startup message\n' + 'b' * 40 + '\n', '')
        return ProgramResult(0, '', '')
    runtime.run = run
    await task.setup(trace, runtime)
    assert trace.state.base_ref == 'b' * 40
    assert trace.info['base_ref'] == 'b' * 40
    assert 'rev-list --all --not ' + 'b' * 40 in calls[-1][-1]
    assert not any(call[0] == 'write' for call in runtime.calls)


def test_taskset_load_and_default_agent_tools(monkeypatch):
    from xiaomi_code import taskset
    from xiaomi_common.data import XiaomiTasksetConfig
    from verifiers.v1.configs.env import default_agent_harness
    from verifiers.v1.utils.loaders import load_taskset
    import json
    instance = dict(dataset_type='opensource-code', cwd='/testbed', docker_image='original',
                    instance_id='example', problem_statement='Original task\n', test_patch=PATCH,
                    test_command='bash /testbed/check.sh', verifier_timeout_sec=1800)
    monkeypatch.setattr(taskset, 'load_rows', lambda domain, config: [
        {'extra_info': {'instance_json': json.dumps(instance)}}])
    monkeypatch.setattr(taskset, 'resolve_image', lambda image, revision: 'docker.io/xiaomimimo/mimo-v2.6-rl-oss:example')
    task = next(iter(load_taskset(XiaomiTasksetConfig(id='xiaomi-code'))))
    assert task.data.id == 'example'
    assert task.data.prompt == instance['problem_statement']
    assert task.data.timeout.scoring == 1920
    assert task.data.workdir == '/testbed'
    assert task.NEEDS_CONTAINER
    harness = default_agent_harness('xiaomi-code')
    assert harness.id == 'bash' and harness.edit


@pytest.mark.parametrize("installed,visible", [(True, False), (True, True), (False, False)])
async def test_setup_repairs_only_missing_installed_go_for_agent_and_grader(tmp_path, installed, visible):
    import os
    import shlex

    from verifiers.v1.rollout import agent_runtime
    from verifiers.v1.runtimes.base import Runtime as BaseRuntime
    from verifiers.v1.runtimes.prime import PrimeRuntime

    image_bin = tmp_path / "image-bin"
    image_bin.mkdir()
    installed_go = tmp_path / "installed-go"
    for path, enabled in [(installed_go, installed), (image_bin / "go", visible)]:
        if enabled:
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
    image_path = f"{image_bin}:/usr/bin:/bin"

    class Runtime(PrimeRuntime):
        def __init__(self):
            BaseRuntime.__init__(self)
            self.calls = []
            self.env = {"TASK_ENV": "preserved"}

        async def run(self, argv, env):
            effective = self.process_env(env)
            self.calls.append((argv, effective))
            if "command -v go" in argv[-1]:
                # Exercise the actual shell condition with a harmless fixture binary.
                command = argv[-1].replace("/usr/local/go/bin/go", shlex.quote(str(installed_go)))
                result = subprocess.run(
                    [*argv[:-1], command], capture_output=True, text=True,
                    env={**os.environ, "PATH": image_path, **effective},
                )
                return ProgramResult(result.returncode, result.stdout, result.stderr)
            if "rev-parse HEAD" in argv[-1]:
                return ProgramResult(0, "a" * 40 + "\n", "")
            return ProgramResult(0, "", "")

        async def write(self, path, data):
            pass

    task, trace = task_and_trace()
    runtime = Runtime()
    await task.setup(trace, runtime)
    harness_runtime = await agent_runtime(runtime, task.data.agent_user)
    await harness_runtime.run(["bash", "-c", "go version"], {})
    assert await task.verified(trace, runtime) == 1
    expected_env = {"TASK_ENV": "preserved"}
    if installed and not visible:
        expected_env["PATH"] = f"/usr/local/go/bin:{image_path}"
    assert runtime.env == expected_env
    agent_env = next(env for argv, env in runtime.calls if argv[-1] == "go version")
    grader_env = next(env for argv, env in runtime.calls if argv[0] == "timeout")
    assert agent_env == grader_env == expected_env
