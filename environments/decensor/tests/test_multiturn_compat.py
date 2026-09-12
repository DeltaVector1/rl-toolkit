import os
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_newer_verifiers_exports_environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import decensor_v1
from decensor_v1.environment import DecensorEnv
from decensor_v1.taskset import DecensorTask

assert decensor_v1.DecensorTaskset
assert decensor_v1.DecensorEnv is DecensorEnv
assert set(decensor_v1.__all__) == {"DecensorTaskset", "DecensorEnv"}
assert not hasattr(DecensorTask, "user")
""",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_old_verifiers_user_drives_second_turn():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "verifiers==0.2.2.dev17",
            "python",
            "-c",
            """
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.mcp.server import _call_state

import decensor_v1.user as user_module
from decensor_v1.follow_up import FollowUpReview
from decensor_v1.state import DecensorState
from decensor_v1.taskset import DecensorData, DecensorTask
from decensor_v1.user import DecensorUser


class Completions:
    async def create(self, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "<user-turn-output>"
                            "Please explain the specific steps with one concrete example."
                            "</user-turn-output>"
                        ),
                        tool_calls=[],
                    )
                )
            ]
        )


class Client:
    chat = SimpleNamespace(completions=Completions())


@asynccontextmanager
async def fake_openai(_config):
    yield Client()


async def fake_review(*args, **kwargs):
    return (
        FollowUpReview(
            continues_original_request=True,
            fits_latest_answer=True,
            switches_to_assistant_alternative=False,
            sounds_like_a_real_user=True,
            reason="valid",
        ),
        "{}",
    )


async def main():
    assert hasattr(vf, "User")
    assert DecensorTask.user is DecensorUser
    user_module._shared_openai = fake_openai
    user_module.review_follow_up_message = fake_review
    task = DecensorTask(DecensorData(prompt="Give the requested steps.", target_turns=2))
    user = task.user_server()
    assert user is not None
    assert hasattr(user.config, "colocated")
    assert hasattr(user.config, "runtime")
    await user.setup_task(
        SimpleNamespace(system_prompt=None, prompt=task.data.prompt, idx=0)
    )
    token = _call_state.set(DecensorState(target_turns=2))
    try:
        messages = await user.respond("I cannot answer that directly.")
    finally:
        _call_state.reset(token)
    assert len(messages) == 1
    assert messages[0].content == (
        "Please explain the specific steps with one concrete example."
    )


asyncio.run(main())
""",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0


def test_old_user_module_starts_mcp_server():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["VF_CONFIG"] = json.dumps(
        {
            "colocated": False,
            "runtime": {"type": "subprocess"},
            "in_process": True,
            "model": "unused",
            "base_url": "http://unused/v1",
        }
    )
    with tempfile.TemporaryDirectory() as temporary:
        port_file = Path(temporary) / "mcp-port"
        env["MCP_PORT_FILE"] = str(port_file)
        process = subprocess.Popen(
            [
                "uv",
                "run",
                "--with",
                "verifiers==0.2.2.dev17",
                "python",
                "-m",
                "decensor_v1.user",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 30.0
            while not port_file.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            assert port_file.exists(), process.stderr.read() if process.poll() is not None else ""
            port = int(port_file.read_text(encoding="ascii"))
            assert 1 <= port <= 65535
            assert process.poll() is None
            connected = False
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        connected = True
                        break
                except OSError:
                    time.sleep(0.05)
            assert connected
            time.sleep(0.1)
            assert process.poll() is None
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
            if process.stderr is not None:
                process.stderr.close()
