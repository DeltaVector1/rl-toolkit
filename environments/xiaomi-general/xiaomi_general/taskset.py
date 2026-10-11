"""Xiaomi General task assets and inline Terminal Bench verifiers."""

import base64
import json
import math
import os
import shlex
from pathlib import Path, PurePosixPath

import verifiers.v1 as vf
from xiaomi_common.data import (
    DATASET_REVISION,
    XiaomiTasksetConfig,
    download_general_task,
    load_instance,
    load_rows,
    resolve_image,
)
from xiaomi_common.runtime import checked_run, run_shell

from .servers.tool import GeneralTools

VERIFIER_FILES = (
    "run_verify.py",
    "verify.py",
    "rubrics.json",
    "answer_key.json",
    "verifier_meta.json",
    "_helpers.py",
)
VERIFY_FORWARD = (
    "VERIFY_DETERMINISTIC",
    "VERIFY_AGENT_JUDGE",
    "GA_JUDGE_KEY",
    "GA_JUDGE_URL",
    "GA_JUDGE_MODEL",
    "GA_JUDGE_EFFORT",
    "GA_JUDGE_API",
    "GRADE_VOTES",
    "JUDGE_API_KEY",
    "JUDGE_BASE_URL",
    "JUDGE_MODEL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
)

QUIESCE_AGENT = """
import os, pwd, signal, time
from pathlib import Path
uid = pwd.getpwnam('xiaomi_agent').pw_uid
for attempt in range(20):
    live = []
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            fields = dict(line.split(':', 1) for line in path.read_text().splitlines())
        except FileNotFoundError:
            continue
        if uid in map(int, fields['Uid'].split()) and not fields['State'].strip().startswith('Z'):
            live.append(int(path.parent.name))
    if not live:
        break
    for sig in (signal.SIGSTOP, signal.SIGKILL):
        for pid in live:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
    time.sleep(0.05)
else:
    raise RuntimeError('Agent processes did not stop before grading')
"""

VALIDATE_SUBMISSION = """
import os, stat, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
def invalid(error):
    print(str(error))
    raise SystemExit(42)
try:
    for folder, dirs, files in os.walk(root, onerror=invalid):
        for name in dirs[:] + files:
            path = Path(folder) / name
            entry = path.lstat()
            if (path.parent == root and name in {'.vf-src', '.vf-venv'}
                    and stat.S_ISDIR(entry.st_mode) and entry.st_uid == 0
                    and not entry.st_mode & 0o022):
                dirs.remove(name)
                continue
            if path.is_symlink():
                try:
                    resolved = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    resolved = None
                if resolved is None or not resolved.is_relative_to(root):
                    path.unlink()
                    if name in dirs:
                        dirs.remove(name)
                    continue
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or not os.access(path, os.R_OK):
                invalid('Submission path is outside the workspace or unreadable: ' + str(path.relative_to(root)))
            if not (resolved.is_file() or resolved.is_dir()):
                invalid('Submission path is not a regular file or directory: ' + str(path.relative_to(root)))
except (OSError, RuntimeError) as exc:
    invalid(exc)
"""


def parse_reward(raw: str, binary: bool = False) -> tuple[float, dict]:
    try:
        reward = float(json.loads(raw)["reward"])
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RuntimeError("missing_or_invalid_reward_json") from exc
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise RuntimeError("reward_out_of_range")
    return (
        (float(reward >= 1 - 1e-6), {"raw_reward": reward}) if binary else (reward, {})
    )


def inline_tests(instance: dict) -> dict[str, bytes]:
    files = (
        json.loads(instance["tests_files"])
        if isinstance(instance["tests_files"], str)
        else instance["tests_files"]
    )
    decoded = {}
    for name, data in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe tests path: {name}")
        decoded[str(path)] = base64.b64decode(data, validate=True)
    if "test.sh" not in decoded:
        raise ValueError("Inline Terminal Bench verifier has no test.sh")
    return decoded


async def upload(runtime: vf.Runtime, source: Path, target: str) -> None:
    if source.is_dir():
        await checked_run(runtime, ["mkdir", "-p", target])
        for child in sorted(source.rglob("*")):
            if child.is_file():
                await runtime.write(
                    str(PurePosixPath(target) / child.relative_to(source).as_posix()),
                    child.read_bytes(),
                )
    else:
        await runtime.write(target, source.read_bytes())


async def write_as_agent(runtime: vf.Runtime, target: str, data: bytes) -> None:
    """Prime uploads ignore agent_user, so use bounded writes under that user."""
    agent = runtime.with_user("xiaomi_agent")
    await checked_run(agent, ["sh", "-c", 'umask 077; : > "$1"', "write", target])
    for offset in range(0, len(data), 32768):
        encoded = base64.b64encode(data[offset : offset + 32768]).decode()
        await checked_run(
            agent,
            ["sh", "-c", 'printf %s "$2" | base64 -d >> "$1"', "write", target, encoded],
        )


def agent_message_writes(runtime: vf.Runtime, trace_id: str) -> None:
    """Keep native harness transcripts agent-owned in the sticky workspace."""
    from verifiers.v1.runtimes.prime import PrimeRuntime

    if not isinstance(runtime, PrimeRuntime):
        return
    original_write = runtime.write
    name = f".vf-initial-messages-{trace_id}.json"
    target = str(PurePosixPath(runtime.config.workdir) / name)

    async def write(path: str, data: bytes) -> None:
        if path not in (name, target):
            await original_write(path, data)
            return
        # shortcut: Prime uploads ignore agent_user; remove when native writes honor it.
        await write_as_agent(runtime, target, data)

    runtime.write = write


async def install_common(runtime: vf.Runtime) -> None:
    # Native toolset bundling ships its own package only; common is a local sibling.
    # shortcut: private bundler helpers match both pinned runtimes; revisit on a Verifiers upgrade.
    from verifiers.v1.mcp.launch import _cached_package, _package_dir

    name, archive = await _cached_package(Path(_package_dir(XiaomiTasksetConfig)))
    target = str(PurePosixPath(runtime.config.workdir) / ".vf-src" / name)
    await runtime.write(target, archive)
    venv = str(PurePosixPath(runtime.config.workdir) / ".vf-venv")
    await runtime.prepare_uv_script(
        '# /// script\n# requires-python = ">=3.11"\n# dependencies = []\n# ///\n'
    )
    await checked_run(
        runtime,
        [
            "sh",
            "-c",
            'export PATH="$HOME/.local/bin:$PATH"; '
            + f"uv venv --allow-existing {shlex.quote(venv)} && uv pip install --python {shlex.quote(venv)} --no-deps {shlex.quote(target)}",
        ],
    )


class GeneralData(vf.TaskData):
    instance: dict
    revision: str = DATASET_REVISION
    task_root: str | None = None


class GeneralTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(colocated=True)


class GeneralConfig(XiaomiTasksetConfig):
    task: GeneralTaskConfig = GeneralTaskConfig()
    task_root: str | None = None
    """Local General directory containing envs/, otherwise fetch pinned task assets."""


def _no_toolsets(config):
    return []


class GeneralTask(vf.Task[GeneralData, vf.State, GeneralTaskConfig]):
    NEEDS_CONTAINER = True

    def __init__(self, data: GeneralData, config: GeneralTaskConfig | None = None) -> None:
        super().__init__(data, config)
        if data.instance["dataset_type"] == "terminal_bench":
            # Workers/replay reconstruct GeneralTask; a loader-only subclass loses this.
            self.toolsets = _no_toolsets

    def runtime_env(self):
        return {
            "UV_CACHE_DIR": "/opt/xiaomi-uv-cache",
            "UV_PYTHON_INSTALL_DIR": "/opt/xiaomi-python",
        }

    @classmethod
    def toolsets(cls, config):
        return [GeneralTools(config.tools)]

    def task_dir(self) -> Path:
        relative = self.data.instance["env_task_dir"]
        if self.data.task_root:
            root = Path(self.data.task_root).resolve()
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError("env_task_dir escapes task_root")
            return path
        return download_general_task(relative, self.data.revision)

    def manifest(self) -> dict:
        path = self.task_dir() / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text())
        root = self.task_dir()
        return {
            "uploads": [
                {
                    "source": name,
                    "target": f"/work/{name}",
                    "container": "main" if name == "workspace" else "sidecar",
                }
                for name in ("workspace", "system", "tools")
                if (root / name).exists()
            ]
            + [
                {
                    "source": "mcp_http.py",
                    "target": "/installed-agent/mcp_http.py",
                    "container": "sidecar",
                }
            ],
            "verifier": {
                "uploads": [
                    {"source": name, "target": f"/work/{name}"}
                    for name in VERIFIER_FILES
                    if (root / name).exists()
                ]
            },
        }

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.instance["dataset_type"] == "terminal_bench":
            inline_tests(self.data.instance)
            return
        if not runtime.supports_user:
            raise RuntimeError(
                "General requires a runtime supporting an unprivileged agent_user"
            )
        if not os.environ.get("GA_JUDGE_KEY"):
            raise RuntimeError(
                "judge_key_missing: set GA_JUDGE_KEY in the controller environment"
            )
        manifest = self.manifest()
        setup = manifest.get("setup") or {}
        if setup.get("command") not in (
            None,
            "python3 /installed-agent/sidecar_entrypoint.py --start-and-detach",
        ):
            raise ValueError(
                "Unsupported task-specific setup command; native MCP adapter requires the released sidecar entrypoint"
            )
        await checked_run(
            runtime,
            [
                "sh",
                "-c",
                "id -u xiaomi_agent >/dev/null 2>&1 || useradd -m -s /bin/bash xiaomi_agent; mkdir -p /work/workspace /work/system /work/tools /work/_verifier /installed-agent /opt/xiaomi-uv-cache /opt/xiaomi-python; chmod 700 /work/_verifier; chmod 755 /opt/xiaomi-uv-cache /opt/xiaomi-python; chmod 700 /work/system /work/tools /installed-agent; chmod 755 /work",
            ],
        )
        root = self.task_dir()
        for item in manifest.get("uploads", []):
            relative = PurePosixPath(item["source"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Manifest upload escapes task directory")
            # HF snapshot files are symlinks into the cache's shared blobs directory.
            source = root / relative
            await upload(runtime, source, item["target"])
        await checked_run(
            runtime,
            [
                "sh",
                "-c",
                "chown -R root:root /work/system /work/tools /installed-agent; chmod -R go-rwx /work/system /work/tools /installed-agent; chown -R xiaomi_agent: /work/workspace; chown root:root /work/workspace; chmod 1777 /work/workspace",
            ],
        )
        # Protect payloads even if the provider's upload API recreates files as 0644.
        for item in manifest.get("verifier", {}).get("uploads", []):
            target = item["target"]
            if PurePosixPath(target).parent != PurePosixPath("/work"):
                raise ValueError("Verifier uploads must target root-owned /work files")
            await checked_run(
                runtime,
                ["ln", "-sfn", f"/work/_verifier/{PurePosixPath(target).name}", target],
            )
        # Fail before rollout if the permission boundary is ineffective.
        probe = await runtime.with_user("xiaomi_agent").run(
            [
                "sh",
                "-c",
                "test ! -r /work/system && test ! -r /work/tools && test ! -r /installed-agent",
            ],
            {},
        )
        if probe.exit_code:
            raise RuntimeError("General protected assets are accessible to agent_user")
        await install_common(runtime)
        agent_message_writes(runtime, trace.id)
        trace.info["general_isolation"] = (
            "root-owned assets; unprivileged xiaomi_agent; native colocated MCP"
        )

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.instance["dataset_type"] != "general_agent":
            return
        # Native rollout.close has closed the harness and tool servers already.
        await checked_run(runtime, ["python3", "-I", "-c", QUIESCE_AGENT])
        submission = await runtime.with_user("xiaomi_agent").run(
            ["python3", "-I", "-c", VALIDATE_SUBMISSION, "/work/workspace"], {}
        )
        if submission.exit_code == 42:
            trace.info["verifier"] = {"invalid_submission": submission.stdout[-2000:]}
            trace.info["general_submission_validated"] = False
            return
        if submission.exit_code:
            raise RuntimeError(f"Submission validation failed: {submission.stderr[-2000:]}")
        trace.info["general_submission_validated"] = True
        if trace.last_reply:
            exists = await runtime.with_user("xiaomi_agent").run(["test", "-s", "/work/workspace/answer.md"], {})
            if exists.exit_code:
                await write_as_agent(
                    runtime, "/work/workspace/answer.md", trace.last_reply.encode()
                )

    @vf.reward
    async def verifier_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        instance = self.data.instance
        if instance["dataset_type"] == "terminal_bench":
            await checked_run(
                runtime,
                [
                    "sh",
                    "-c",
                    "mkdir -p /logs/verifier; rm -f /logs/verifier/reward.txt",
                ],
            )
            for name, data in inline_tests(instance).items():
                await runtime.write(f"/tests/{name}", data)
            result = await run_shell(
                runtime,
                "bash /tests/test.sh",
                cwd=self.data.workdir,
                timeout=instance.get("verifier_timeout_sec") or 900,
            )
            try:
                reward = float(
                    (await runtime.read("/logs/verifier/reward.txt")).decode().strip()
                )
            except Exception as exc:
                raise RuntimeError("missing_or_invalid_terminal_reward") from exc
            if not math.isfinite(reward) or not 0 <= reward <= 1:
                raise RuntimeError("terminal_reward_out_of_range")
            trace.info["verifier"] = {
                "returncode": result.exit_code,
                "stdout": result.stdout[-10000:],
                "stderr": result.stderr[-4000:],
            }
            return reward
        if not os.environ.get("GA_JUDGE_KEY"):
            raise RuntimeError("judge_key_missing")
        if "general_submission_validated" not in trace.info:
            raise RuntimeError("General submission was not validated before grading")
        if not trace.info["general_submission_validated"]:
            return 0.0
        manifest = self.manifest()
        verifier = manifest.get("verifier", {})
        if verifier.get("container", "sidecar") == "main":
            raise ValueError("Verifier must remain privileged")
        for item in verifier.get("uploads", []):
            await upload(
                runtime,
                self.task_dir() / item["source"],
                f"/work/_verifier/{PurePosixPath(item['target']).name}",
            )
        env = {"VERIFY_DETERMINISTIC": "1", "VERIFY_AGENT_JUDGE": "1"}
        env.update(
            {
                key: os.environ[key]
                for key in verifier.get("env_passthrough") or VERIFY_FORWARD
                if os.environ.get(key)
            }
        )
        env.update(
            {key: str(value) for key, value in (verifier.get("env") or {}).items()}
        )
        # Rule subprocesses import from their script directory; shared /tmp is agent-writable.
        env["TMPDIR"] = "/work/_verifier/tmp"
        reward_path = verifier.get("reward_file", "/logs/verifier/reward.json")
        detail_path = verifier.get(
            "reward_detail_file", "/logs/verifier/reward_detail.json"
        )
        await checked_run(
            runtime,
            [
                "sh",
                "-c",
                f"rm -rf -- /tmp/agent_output && mkdir -p {shlex.quote(str(PurePosixPath(reward_path).parent))} /tmp/agent_output /work/_verifier/tmp && rm -f {shlex.quote(reward_path)} {shlex.quote(detail_path or '')} && chmod 700 /tmp/agent_output /work/_verifier/tmp",
            ],
        )
        result = await run_shell(
            runtime,
            verifier.get("command", "python3 /work/run_verify.py"),
            cwd="/work",
            timeout=verifier.get("timeout_sec")
            or instance.get("verifier_timeout_sec")
            or 900,
            env=env,
        )
        try:
            raw = (await runtime.read(reward_path)).decode()
        except Exception as exc:
            raise RuntimeError("missing_or_invalid_reward_json") from exc
        reward, info = parse_reward(raw, bool(instance.get("reward_binary_fullscore")))
        info.update(
            verifier_returncode=result.exit_code,
            stdout=result.stdout[-10000:],
            stderr=result.stderr[-4000:],
        )
        if detail_path:
            detail = await runtime.run(["cat", detail_path], {})
            if detail.exit_code == 0:
                try:
                    info["reward_detail"] = json.loads(detail.stdout)
                except ValueError:
                    info["reward_detail_raw"] = detail.stdout[:2000]
        trace.info["verifier"] = info
        return reward


class GeneralTaskset(vf.Taskset[GeneralTask, GeneralConfig]):
    def load(self):
        for row in load_rows("general", self.config):
            instance = load_instance(row)
            kind = instance["dataset_type"]
            if kind not in {"general_agent", "terminal_bench"}:
                raise ValueError(f"Unknown General dataset_type: {kind}")
            yield GeneralTask(
                GeneralData(
                    id=instance["instance_id"],
                    prompt=row["prompt"],
                    instance=instance,
                    image=resolve_image(instance["docker_image"], self.config.revision),
                    workdir=instance.get("cwd")
                    or ("/work/workspace" if kind == "general_agent" else "/app"),
                    agent_user="xiaomi_agent" if kind == "general_agent" else None,
                    timeout=vf.TaskTimeout(
                        setup=900,
                        agent=instance.get("agent_timeout_sec"),
                        scoring=(instance.get("verifier_timeout_sec") or 900) + 120,
                    ),
                    resources=vf.TaskResources(
                        cpu=instance.get("cpus"),
                        memory=instance["memory_mb"] / 1024
                        if instance.get("memory_mb") else None,
                        disk=instance["storage_mb"] / 1024
                        if instance.get("storage_mb") else None,
                    ),
                    network_allow=[] if instance.get("allow_internet") is False else ["*"],
                    revision=self.config.revision,
                    task_root=self.config.task_root,
                ),
                self.config.task,
            )
