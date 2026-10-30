"""Synthetic repositories only; no published task commands run on the host."""

import os
import py_compile
import shlex
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.runtimes import ProgramResult

from xiaomi_code.taskset import (
    BASH_ENV_PATH, BASH_ENV_SCRIPT, CACHE_CLEANUP, CodeData, CodeState,
    XiaomiCodeTask, reset_command, strip_history_command, touched_files,
)

from xiaomi_code._repairs import repair_test_patch

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
    with pytest.raises(ValueError, match="no resettable file headers"):
        reset_command("", "a" * 40)


@pytest.mark.parametrize("mixed", [False, True])
def test_reset_covers_headerless_file_before_git_headers(repo, mixed):
    # The released 000989 patch starts with this file and then a Git-header script.
    name = "pkg/common/moerr/error_wrapping_test.go"
    patch = f"--- /dev/null\n+++ b/{name}\n@@ -0,0 +1 @@\n+hidden\n"
    if mixed:
        patch += PATCH
    base = git(repo, "rev-parse", "HEAD")
    hidden = repo / name
    hidden.parent.mkdir(parents=True)
    hidden.write_text("agent planted\n")
    (repo / "solution.txt").write_text("model fix\n")
    git(repo, "add", ".")
    for _ in range(2):
        subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
        assert not hidden.exists()
        subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=repo, check=True)
        assert hidden.read_text() == "hidden\n"
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    assert not hidden.exists()
    assert not (repo / "check.txt").exists()
    assert (repo / "solution.txt").read_text() == "model fix\n"


def test_headerless_deletion_resets_old_path(repo):
    patch = "--- a/solution.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-before\n"
    base = git(repo, "rev-parse", "HEAD")
    subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=repo, check=True)
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    assert (repo / "solution.txt").read_text() == "before\n"


def test_header_like_hunk_payload_does_not_reset_solution(repo):
    (repo / "check.txt").write_text("-- a/solution.txt\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "test fixture")
    base = git(repo, "rev-parse", "HEAD")
    patch = "--- a/check.txt\n+++ b/check.txt\n@@ -1 +1 @@\n--- a/solution.txt\n+++ b/solution.txt\n"
    assert touched_files(patch) == ["check.txt"]
    subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=repo, check=True)
    (repo / "solution.txt").write_text("model fix\n")
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    assert (repo / "check.txt").read_text() == "-- a/solution.txt\n"
    assert (repo / "solution.txt").read_text() == "model fix\n"


@pytest.mark.parametrize("old,new", [
    ("a/../../outside", "b/check.txt"),
    ("a/check.txt", "b/../../outside"),
    ("/dev/null", "b//absolute"),
    ("a/check.txt", "b/null\x00byte"),
])
def test_unified_patch_validates_both_paths_even_after_git_headers(old, new):
    with pytest.raises(ValueError, match="Unsafe test patch path"):
        touched_files(f"diff --git a/check.txt b/check.txt\n--- {old}\n+++ {new}\n")


@pytest.mark.parametrize("old,new", [
    ('"a/quoted\\tpath"', "b/check.txt"),
    ("/dev/null", '"b/quoted\\tpath"'),
    ("a/check.txt", "missing-prefix"),
])
def test_unsupported_unified_headers_fail(old, new):
    with pytest.raises(ValueError, match="Unsupported test patch header"):
        touched_files(f"--- {old}\n+++ {new}\n")


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
    assert "git cat-file" in runtime.calls[1][1][-1]
    assert runtime.calls[0] == ("write", BASH_ENV_PATH, BASH_ENV_SCRIPT.encode())
    assert runtime.calls[2][0] == "write"
    assert "git apply" in runtime.calls[3][1][-1]
    assert "git cat-file" in runtime.calls[-1][1][-1]
    assert trace.info["verifier"]["verifier_returncode"] == returncode


@pytest.mark.parametrize("runtime", [RecordingRuntime(apply_rc=1), RecordingRuntime(transport_error=True)])
async def test_broken_testbed_is_error_and_cleanup_runs(runtime):
    task, trace = task_and_trace()
    with pytest.raises((RuntimeError, OSError)):
        await task.verified(trace, runtime)
    assert trace.info["error_category"] == "reward/testbed_corrupted"
    assert "git cat-file" in runtime.calls[-1][1][-1]


def test_published_hcl_repair_restores_protected_upstream_fixtures(repo):
    # This patch comes unchanged from the pinned dataset revision.
    original = (Path(__file__).parent / "fixtures" / "code-001441.patch").read_text()
    assert sha256(original.encode()).hexdigest() == "f50c33e070080c593b60017eaf7ae4e10999f839f7ee1062894898b95621d87f"
    patch = repair_test_patch("format-code-task-001441", original)
    assert patch.startswith(original.rstrip("\n") + "\n")
    fixture_dir = repo / "hcl/printer/testdata"
    fixture_dir.mkdir(parents=True)
    # Source: hashicorp/hcl@78101eb3cb45bc983bceeb78993f6a263406c54e/hcl/printer/printer_test.go.
    before = (Path(__file__).parent / "fixtures" / "hcl-printer-before.go").read_bytes()
    (repo / "hcl/printer/printer_test.go").write_bytes(before)
    git(repo, "add", "hcl/printer/printer_test.go")
    git(repo, "commit", "-qm", "upstream printer test before fixture addition")
    base = git(repo, "rev-parse", "HEAD")
    for name in ("empty_block.input", "empty_block.golden"):
        (fixture_dir / name).write_text("model changed fixture\n")
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    subprocess.run(["git", "apply", "-"],
                   input=patch, text=True, cwd=repo, check=True)
    assert sha256((fixture_dir / "empty_block.input").read_bytes()).hexdigest() == "4a2b73a9a03a96f3e409d8b97ecaea0307a7e520295235873eae55184487e561"
    assert sha256((fixture_dir / "empty_block.golden").read_bytes()).hexdigest() == "ac0ab6cfb3ebc4236a0dd6658e6856e0335e516a6b62700e715026a26263fe01"
    subprocess.run(["bash", "-c", reset_command(patch, base)], cwd=repo, check=True)
    assert not (fixture_dir / "empty_block.input").exists()
    assert not (fixture_dir / "empty_block.golden").exists()
    assert (repo / "solution.txt").read_text() == "before\n"


def test_hcl_repair_rejects_a_different_published_patch():
    with pytest.raises(ValueError, match="Unknown test patch"):
        repair_test_patch("format-code-task-001441", PATCH)
    assert repair_test_patch("another-task", PATCH) == PATCH


@pytest.mark.parametrize("history", ["clean", "remaining_ref", "git_failure"])
async def test_history_check_ignores_profile_output_and_fails_closed(repo, history):
    task, trace = task_and_trace()
    task = XiaomiCodeTask(task.data.model_copy(update={"workdir": str(repo)}))
    base = git(repo, "rev-parse", "HEAD")
    if history == "remaining_ref":
        git(repo, "commit", "--allow-empty", "-qm", "remaining future commit")
    profile = repo / "chatty-profile"
    profile.write_text("echo 'JAVA_HOME='\necho 'maven settings: fixture'\n" +
                       ("git() { return 42; }\n" if history == "git_failure" else ""))
    runtime = RecordingRuntime()
    async def run(argv, env):
        if "rev-parse HEAD" in argv[-1]:
            return ProgramResult(0, base + "\n", "")
        if "rev-list" in argv[-1]:
            result = subprocess.run(argv, env={**os.environ, "BASH_ENV": str(profile)},
                                    capture_output=True, text=True)
            assert "maven settings: fixture" in result.stdout
            return ProgramResult(result.returncode, result.stdout, result.stderr)
        return ProgramResult(0, "", "")
    runtime.run = run
    if history == "clean":
        await task.setup(trace, runtime)
        assert trace.info["base_ref"] == base
    else:
        with pytest.raises(RuntimeError, match="history"):
            await task.setup(trace, runtime)
        assert "base_ref" not in trace.info
    assert runtime.calls == [("write", BASH_ENV_PATH, BASH_ENV_SCRIPT.encode())]


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
    assert any('rev-list -n 1 --all --not ' + 'b' * 40 in call[-1] for call in calls)
    assert runtime.calls == [('write', BASH_ENV_PATH, BASH_ENV_SCRIPT.encode())]


@pytest.mark.parametrize("task_id", ["example", "format-code-task-001441"])
def test_taskset_load_and_default_agent_tools(monkeypatch, task_id):
    from xiaomi_code import taskset
    from xiaomi_code._data import XiaomiTasksetConfig
    from verifiers.v1.configs.env import default_agent_harness
    from verifiers.v1.utils.loaders import load_taskset
    import json
    patch = PATCH if task_id == "example" else (
        Path(__file__).parent / "fixtures" / "code-001441.patch"
    ).read_text()
    instance = dict(dataset_type='opensource-code', cwd='/testbed', docker_image='original',
                    instance_id=task_id, problem_statement='Original task\n', test_patch=patch,
                    test_command='bash /testbed/check.sh', verifier_timeout_sec=1800)
    monkeypatch.setattr(taskset, 'load_rows', lambda domain, config: [
        {'extra_info': {'instance_json': json.dumps(instance)}}])
    monkeypatch.setattr(taskset, 'resolve_image', lambda image, revision: 'docker.io/xiaomimimo/mimo-v2.6-rl-oss:example')
    task = next(iter(load_taskset(XiaomiTasksetConfig(id='xiaomi-code'))))
    assert task.data.id == task_id
    assert task.data.test_patch == repair_test_patch(task_id, patch)
    assert task.data.prompt == instance['problem_statement']
    assert task.data.timeout.scoring == 1920
    assert task.data.timeout.setup == 600
    from verifiers.v1.runtimes import PrimeConfig
    from verifiers.v1.utils.compile import resolve_runtime_config
    assert resolve_runtime_config(PrimeConfig(), task).memory == 8
    assert resolve_runtime_config(PrimeConfig(memory=16), task).memory == 16
    assert task.data.workdir == '/testbed'
    assert task.NEEDS_CONTAINER
    harness = default_agent_harness('xiaomi-code')
    assert harness.id == 'bash' and harness.edit


async def test_docker_default_path_reaches_agent_and_verifier():
    from verifiers.v1.rollout import agent_runtime
    from verifiers.v1.runtimes.base import Runtime as BaseRuntime
    from verifiers.v1.runtimes.prime import PrimeRuntime

    class Runtime(PrimeRuntime):
        def __init__(self):
            BaseRuntime.__init__(self)
            self.calls = []

        async def run(self, argv, env):
            self.calls.append((argv, self.process_env(env)))
            if "rev-parse HEAD" in argv[-1]:
                return ProgramResult(0, "a" * 40 + "\n", "")
            return ProgramResult(0, "", "")

        async def write(self, path, data):
            pass

    task, trace = task_and_trace()
    runtime = Runtime()
    # Rollout installs the task environment before setup and creating the agent view.
    runtime.env = dict(task.runtime_env())
    await task.setup(trace, runtime)
    harness_runtime = await agent_runtime(runtime, task.data.agent_user)
    await harness_runtime.run(["bash", "-c", "npm --version; go version"], {})
    assert await task.verified(trace, runtime) == 1
    expected = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "BASH_ENV": BASH_ENV_PATH}
    agent_env = next(env for argv, env in runtime.calls if argv[-1] == "npm --version; go version")
    grader_env = next(env for argv, env in runtime.calls if argv[0] == "timeout" and task.data.test_command in argv[-1])
    assert agent_env == grader_env == expected


@pytest.mark.parametrize("returncode", [0, 1])
async def test_verifier_trace_keeps_output_tails_without_changing_reward(returncode):
    task, trace = task_and_trace()
    runtime = RecordingRuntime()
    original_run = runtime.run
    stdout = "discarded prefix" + "output" * 2000 + "final stdout"
    stderr = "discarded prefix" + "error" * 1000 + "final stderr"

    async def run(argv, env):
        if argv[0] == "timeout":
            return ProgramResult(returncode, stdout, stderr)
        return await original_run(argv, env)

    runtime.run = run
    assert await task.verified(trace, runtime) == float(returncode == 0)
    assert trace.info["verifier"]["stdout"] == stdout[-10000:]
    assert trace.info["verifier"]["stderr"] == stderr[-4000:]
    assert "git cat-file" in runtime.calls[-1][1][-1]


def test_strip_removes_future_objects_but_preserves_ancestry_and_worktree(repo):
    ancestor = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "-a", "v1", "-m", "ancestor")
    (repo / "solution.txt").write_text("base\n")
    git(repo, "commit", "-qam", "task base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "future.txt").write_text("reference fix\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "future fix")
    future = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "future")
    git(repo, "update-ref", "refs/custom/fix", future)
    git(repo, "update-ref", "refs/remotes/origin/main", future)
    git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/missing")
    git(repo, "reset", "--hard", base)
    # Replacement refs must not change the real task base tree or its ancestry.
    git(repo, "replace", base, future)
    (repo / "solution.txt").write_text("staged edit\n")
    git(repo, "add", "solution.txt")
    (repo / "solution.txt").write_text("working edit\n")
    (repo / "untracked.txt").write_text("keep\n")
    before = subprocess.run(["git", "--no-replace-objects", "status", "--porcelain"], cwd=repo,
                            capture_output=True, text=True, check=True).stdout
    hook = repo / ".git/hooks/post-checkout"
    hook.write_text("#!/bin/sh\necho ran > .git/hook-ran\nexit 1\n")
    hook.chmod(0o755)
    subprocess.run(["bash", "-c", strip_history_command(base)], cwd=repo, check=True)
    assert not (repo / ".git/hook-ran").exists()
    assert git(repo, "rev-parse", "HEAD") == base
    assert git(repo, "describe", "--tags", "--long").startswith("v1-1-g")
    assert git(repo, "rev-parse", "v1^{}") == ancestor
    assert git(repo, "for-each-ref", "--format=%(refname)") == "refs/tags/v1"
    assert git(repo, "status", "--porcelain") == before.strip()
    assert (repo / "solution.txt").read_text() == "working edit\n"
    assert git(repo, "show", ":solution.txt") == "staged edit"
    assert not (repo / ".git/refs/remotes").exists()
    assert not git(repo, "reflog", "--all")
    assert "unreachable" not in git(repo, "fsck", "--unreachable", "--no-reflogs")
    assert subprocess.run(["git", "cat-file", "-e", future], cwd=repo,
                          capture_output=True).returncode != 0


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("retain_tag", [False, True])
def test_strip_removes_future_metadata_and_stale_commit_graphs(repo, split, retain_tag):
    base = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    if retain_tag:
        git(repo, "tag", "base")
    (repo / "future.txt").write_text("reference fix\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "future fix")
    future = git(repo, "rev-parse", "HEAD")
    future_blob = git(repo, "rev-parse", "HEAD:future.txt")
    git(repo, "config", "gc.writeCommitGraph", "true")
    git(repo, "commit-graph", "write", "--reachable", *(["--split"] if split else []))
    info = repo / ".git/objects/info"
    graphs = list((info / "commit-graphs").glob("graph-*.graph")) if split else [info / "commit-graph"]
    assert graphs and any(bytes.fromhex(future) in graph.read_bytes() for graph in graphs)
    git(repo, "commit-graph", "verify")
    git(repo, "reset", "--hard", base)
    (repo / ".git/ORIG_HEAD").write_text(f"{future}\n")
    (repo / ".git/FETCH_HEAD").write_text(f"{future}\t\tbranch 'main' of https://example.invalid/upstream\n")
    (repo / "solution.txt").write_text("staged edit\n")
    git(repo, "add", "solution.txt")
    (repo / "solution.txt").write_text("working edit\n")
    (repo / "untracked.txt").write_text("keep\n")
    status = git(repo, "status", "--porcelain")

    subprocess.run(["bash", "-c", strip_history_command(base)], cwd=repo, check=True)

    assert git(repo, "rev-parse", "HEAD") == base
    assert git(repo, "rev-parse", "HEAD^{tree}") == tree
    assert git(repo, "status", "--porcelain") == status
    assert git(repo, "show", ":solution.txt") == "staged edit"
    assert (repo / "solution.txt").read_text() == "working edit\n"
    assert (repo / "untracked.txt").read_text() == "keep\n"
    assert git(repo, "for-each-ref", "--format=%(refname)") == ("refs/tags/base" if retain_tag else "")
    assert not (repo / ".git/ORIG_HEAD").exists()
    assert not (repo / ".git/FETCH_HEAD").exists()
    # GC may write a fresh graph when an ancestor tag survives.
    graphs = list(info.glob("commit-graph")) + list((info / "commit-graphs").glob("graph-*.graph"))
    assert all(bytes.fromhex(future) not in graph.read_bytes() for graph in graphs)
    git(repo, "commit-graph", "verify")
    assert "unreachable" not in git(repo, "fsck", "--unreachable", "--no-reflogs")
    for object_id in (future, future_blob):
        assert subprocess.run(["git", "cat-file", "-e", object_id], cwd=repo,
                              capture_output=True).returncode != 0


@pytest.mark.parametrize("marker", ["promisor", "keep", "config", "worktree", "linked"])
def test_strip_rejects_unprunable_or_shared_repositories(repo, tmp_path, marker):
    base = git(repo, "rev-parse", "HEAD")
    if marker in {"promisor", "keep"}:
        (repo / f".git/objects/pack/pack-future.{marker}").touch()
    elif marker == "config":
        git(repo, "config", "remote.origin.promisor", "true")
    else:
        git(repo, "worktree", "add", "--detach", str(tmp_path / "linked"), base)
    result = subprocess.run(["bash", "-c", strip_history_command(base)], cwd=repo,
                            capture_output=True, text=True)
    if marker == "linked":
        result = subprocess.run(["bash", "-c", strip_history_command(base)], cwd=tmp_path / "linked",
                                capture_output=True, text=True)
    assert result.returncode != 0
    assert "Cannot sanitize" in result.stderr
    assert git(repo, "symbolic-ref", "HEAD").startswith("refs/heads/")


@pytest.mark.parametrize("value", ["false", "no", "off", "0", "invalid"])
def test_strip_interprets_promisor_boolean_metadata(repo, value):
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "config", "remote.origin.promisor", value)
    result = subprocess.run(["bash", "-c", strip_history_command(base)], cwd=repo,
                            capture_output=True, text=True)
    if value == "invalid":
        assert result.returncode != 0
        assert "bad boolean" in result.stderr
        assert git(repo, "symbolic-ref", "HEAD").startswith("refs/heads/")
    else:
        assert result.returncode == 0, result.stderr
        assert git(repo, "rev-parse", "HEAD") == base


async def test_setup_gc_failure_is_fatal_before_agent():
    task, trace = task_and_trace()
    runtime = RecordingRuntime()
    async def run(argv, env):
        if "rev-parse HEAD" in argv[-1]:
            return ProgramResult(0, "a" * 40, "")
        if "gc --prune=now" in argv[-1]:
            return ProgramResult(128, "", "gc failed")
        return ProgramResult(0, "", "")
    runtime.run = run
    with pytest.raises(RuntimeError, match="gc failed"):
        await task.setup(trace, runtime)
    assert "base_ref" not in trace.info


def test_cleanup_preserves_tracked_caches_and_build_environment(repo):
    for path in ("pkg/__pycache__/hidden.pyc", ".pytest_cache/test_names", ".jest/cache/test_names",
                 ".build_env/.pytest_cache/required", "tracked/.pytest_cache/fixture",
                 "tracked[fixture]/.jest/cache/fixture"):
        file = repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("fixture\n")
    dependency = repo / ".venv/lib/site-packages/fixture_dep"
    dependency.mkdir(parents=True)
    source = dependency / "__init__.py"
    source.write_text("value = 'bytecode dependency'\n")
    bytecode = dependency / "__pycache__/__init__.pyc"
    py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
    source.unlink()
    git(repo, "--literal-pathspecs", "add", "tracked", "tracked[fixture]")
    subprocess.run(["bash", "-c", CACHE_CLEANUP], cwd=repo, check=True)
    assert (repo / "pkg/__pycache__/hidden.pyc").exists()
    # Published verifier scripts relocate cached bytecode to import source-less packages.
    (dependency / "__init__.pyc").write_bytes(bytecode.read_bytes())
    subprocess.run([sys.executable, "-c", "import fixture_dep; assert fixture_dep.value == 'bytecode dependency'"],
                   env={**os.environ, "PYTHONPATH": str(dependency.parent)}, check=True)
    assert not (repo / ".pytest_cache").exists()
    assert not (repo / ".jest/cache").exists()
    assert (repo / ".build_env/.pytest_cache/required").exists()
    assert (repo / "tracked/.pytest_cache/fixture").exists()
    assert (repo / "tracked[fixture]/.jest/cache/fixture").exists()


def test_code_bootstrap_loads_once_and_children_keep_parent_environment(tmp_path):
    bootstrap = tmp_path / "bootstrap with spaces"
    # Use fixture profiles, never source or modify the developer's login files.
    system_profile = tmp_path / "system-profile"
    system_profile.write_text('export PATH=/usr/bin:/bin\necho loaded >> "$HOME/profile-count"\n'
                              'export VIRTUAL_ENV=profile NODE_OPTIONS=profile GRADLE_OPTS=profile MAVEN_OPTS=profile\n'
                              'test -z "$PS1"\nbash -c true\n')
    (tmp_path / ".bash_profile").write_text("export PROJECT_PYTHON=fixture\nbash -c 'true'\n")
    tool = tmp_path / "mytool"
    tool.write_text("#!/bin/sh\necho tool-ok\n")
    tool.chmod(0o755)
    child = tmp_path / "child"
    child.write_text('#!/bin/bash\nset -eu\nmytool\n'
                     'test "$PROJECT_PYTHON" = parent\n'
                     'test "$VIRTUAL_ENV:$NODE_OPTIONS:$GRADLE_OPTS:$MAVEN_OPTS" = parent:parent:parent:parent\n'
                     'test -z "${BASH_ENV-}"\n')
    child.chmod(0o755)
    script = BASH_ENV_SCRIPT.replace("/etc/profile", str(system_profile))
    bootstrap.write_text(script)
    env = {**os.environ, "HOME": str(tmp_path), "BASH_ENV": str(bootstrap)}
    command = ('test "$PROJECT_PYTHON" = fixture && '
               'export PATH="$HOME:$PATH" PROJECT_PYTHON=parent VIRTUAL_ENV=parent '
               'NODE_OPTIONS=parent GRADLE_OPTS=parent MAVEN_OPTS=parent; '
               '"$HOME/child" && /bin/bash -eu "$HOME/child"')
    result = subprocess.run(["bash", "-c", command],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout.splitlines() == ["tool-ok", "tool-ok"]
    assert (tmp_path / "profile-count").read_text() == "loaded\n"
    # A second command launched by the same Python harness still loads its own profile.
    subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=True)
    assert (tmp_path / "profile-count").read_text() == "loaded\nloaded\n"
    from xiaomi_general.taskset import GeneralTask
    assert "BASH_ENV" not in GeneralTask.runtime_env(SimpleNamespace())


@pytest.mark.parametrize("existing_cargo", [False, True])
def test_bootstrap_recovers_only_missing_installed_tools(tmp_path, existing_cargo):
    bootstrap = tmp_path / "bootstrap"
    profile = tmp_path / "profile"
    profile.write_text("")
    cargo_home = tmp_path / "cargo"
    rustup_home = tmp_path / "rustup"
    java_home = tmp_path / "java"
    original_bin = tmp_path / "bin"
    original_bin.mkdir()
    for binary in (cargo_home / "bin/cargo", java_home / "bin/javac", tmp_path / "go/bin/okexchaincli"):
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    if existing_cargo:
        (original_bin / "cargo").write_text("#!/bin/sh\nexit 0\n")
        (original_bin / "cargo").chmod(0o755)
    script = BASH_ENV_SCRIPT.replace("/etc/profile", str(profile)).replace(
        "/usr/local/cargo", str(cargo_home)).replace("/usr/local/rustup", str(rustup_home)).replace(
        "/opt/java/openjdk", str(java_home)).replace(BASH_ENV_PATH, str(bootstrap))
    bootstrap.write_text(script)
    env = {"HOME": str(tmp_path), "PATH": str(original_bin), "BASH_ENV": str(bootstrap),
           "CARGO_HOME": "original", "RUSTUP_HOME": "original"}
    child_command = ('command -v cargo; command -v javac; command -v okexchaincli; '
                     'printf "%s\\n" "$CARGO_HOME" "$RUSTUP_HOME" "$JAVA_HOME"')
    result = subprocess.run(["/bin/bash", "-c", "/bin/bash -eu -c " + shlex.quote(child_command)],
                            env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [
        str(original_bin / "cargo" if existing_cargo else cargo_home / "bin/cargo"),
        str(java_home / "bin/javac"), str(tmp_path / "go/bin/okexchaincli"),
        "original" if existing_cargo else str(cargo_home),
        "original" if existing_cargo else str(rustup_home), str(java_home),
    ]
