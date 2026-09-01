import ast
import asyncio
import httpx
import math
import os
import time
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"


class _ManifestProbeResponse:
    def __init__(self, version):
        self.version = version

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "version": self.version,
            "transport": (
                "chunked-bf16-linear-zigzag-sparse-zstd"
                if self.version == 2
                else "chunked-shuffled-xor-zstd"
            ),
        }


class _ManifestProbeClient:
    requests = []

    def __init__(self, version):
        self.version = version

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, *_args, **_kwargs):
        self.requests.append(_kwargs["timeout"])
        return _ManifestProbeResponse(self.version)


def _load_stage_function(probe_version=1):
    source = CLIENT_PATH.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage_lora_adapter"
    )
    namespace = {
        "AsyncClient": httpx.AsyncClient,
        "httpx": type(
            "ProbeHttpx",
            (),
            {
                "AsyncClient": lambda *_args, **_kwargs: _ManifestProbeClient(probe_version),
                "HTTPError": httpx.HTTPError,
                "Timeout": httpx.Timeout,
            },
        ),
        "Path": Path,
        "asyncio": asyncio,
        "time": time,
        "LORA_STAGE_ATTEMPT_TIMEOUT_S": 30.0,
        "LORA_STAGE_TOTAL_TIMEOUT_S": 30.0,
        "_delta_stage_timeout_s": lambda: 30.0,
        "_admin_post_response": None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(CLIENT_PATH), "exec"), namespace)
    return namespace["stage_lora_adapter"], namespace


def _load_delta_timeout_function():
    source = CLIENT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_delta_stage_timeout_s", "_is_finite_number"}
    ]
    namespace = {
        "os": os,
        "Path": Path,
        "tomllib": tomllib,
        "math": math,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(CLIENT_PATH), "exec"), namespace)
    return namespace["_delta_stage_timeout_s"]


class StageLoraAdapterTest(unittest.TestCase):
    def test_client_rejects_non_finite_delta_limit(self):
        timeout = _load_delta_timeout_function()
        old_value = os.environ.get("PRIME_LORA_DELTA_STAGE_TIMEOUT_S")
        try:
            for value in ("nan", "inf", "-inf"):
                os.environ["PRIME_LORA_DELTA_STAGE_TIMEOUT_S"] = value
                with self.assertRaisesRegex(RuntimeError, "must be positive"):
                    timeout()
        finally:
            if old_value is None:
                os.environ.pop("PRIME_LORA_DELTA_STAGE_TIMEOUT_S", None)
            else:
                os.environ["PRIME_LORA_DELTA_STAGE_TIMEOUT_S"] = old_value

    def test_distinct_ports_each_stage(self):
        stage_lora_adapter, namespace = _load_stage_function()
        calls = []

        class Response:
            def json(self):
                return {"status": "ok", "stage_seconds": 1.0}

        async def fake_admin_post_response(client, path, **kwargs):
            calls.append(client)
            return Response()

        namespace["_admin_post_response"] = fake_admin_post_response
        clients = [
            httpx.AsyncClient(base_url="http://127.0.0.1:8000"),
            httpx.AsyncClient(base_url="http://127.0.0.1:18049"),
        ]
        result = asyncio.run(stage_lora_adapter(clients, "http://adapter/step_1"))

        self.assertEqual(calls, clients)
        self.assertEqual(result["host_count"], 2)

    def test_exact_duplicate_origins_stage_once(self):
        stage_lora_adapter, namespace = _load_stage_function()
        calls = []

        class Response:
            def json(self):
                return {"status": "ok", "stage_seconds": 1.0}

        async def fake_admin_post_response(client, path, **kwargs):
            calls.append(client)
            return Response()

        namespace["_admin_post_response"] = fake_admin_post_response
        clients = [
            httpx.AsyncClient(base_url="http://127.0.0.1:8000"),
            httpx.AsyncClient(base_url="http://127.0.0.1:8000"),
        ]
        result = asyncio.run(stage_lora_adapter(clients, "http://adapter/step_1"))

        self.assertEqual(calls, [clients[0]])
        self.assertEqual(result["host_count"], 1)

    def test_configured_origin_stages_before_peer(self):
        stage_lora_adapter, namespace = _load_stage_function()
        calls = []

        class Response:
            def json(self):
                return {"status": "ok", "stage_seconds": 1.0}

        async def fake_admin_post_response(client, path, **kwargs):
            calls.append(client)
            return Response()

        namespace["_admin_post_response"] = fake_admin_post_response
        origin = httpx.AsyncClient(base_url="http://127.0.0.1:8000")
        peer = httpx.AsyncClient(base_url="http://127.0.0.1:18049")
        result = asyncio.run(
            stage_lora_adapter(
                [peer, origin],
                "http://adapter/step_1",
                origin_base_urls=["http://127.0.0.1:8000"],
            )
        )

        self.assertEqual(calls, [origin, peer])
        self.assertEqual(result["origin_host_count"], 1)
        self.assertEqual(result["peer_host_count"], 1)

    def test_delta_limit_covers_origin_and_peer_together(self):
        stage_lora_adapter, namespace = _load_stage_function()

        class Response:
            def json(self):
                return {"status": "ok", "stage_seconds": 0.03}

        async def fake_admin_post_response(_client, _path, **_kwargs):
            await asyncio.sleep(0.03)
            return Response()

        namespace["_admin_post_response"] = fake_admin_post_response
        namespace["_delta_stage_timeout_s"] = lambda: 0.05
        origin = httpx.AsyncClient(base_url="http://127.0.0.1:8000")
        peer = httpx.AsyncClient(base_url="http://127.0.0.1:18049")

        with self.assertRaisesRegex(TimeoutError, "exceeded its time limit"):
            asyncio.run(
                stage_lora_adapter(
                    [origin, peer],
                    "http://adapter/step_1",
                    require_delta=True,
                    origin_base_urls=["http://127.0.0.1:8000"],
                )
            )

    def test_delta_peer_receives_only_remaining_time(self):
        stage_lora_adapter, namespace = _load_stage_function()
        calls = []

        class Response:
            def json(self):
                return {"status": "ok", "stage_seconds": 0.0}

        origin = httpx.AsyncClient(base_url="http://127.0.0.1:8000")
        peer = httpx.AsyncClient(base_url="http://127.0.0.1:18049")

        async def fake_admin_post_response(client, _path, **kwargs):
            calls.append((client, kwargs))
            if client is origin:
                await asyncio.sleep(0.03)
            return Response()

        namespace["_admin_post_response"] = fake_admin_post_response
        namespace["_delta_stage_timeout_s"] = lambda: 0.15
        result = asyncio.run(
            stage_lora_adapter(
                [peer, origin],
                "http://adapter/step_1",
                require_delta=True,
                origin_base_urls=["http://127.0.0.1:8000"],
            )
        )

        self.assertEqual([client for client, _ in calls], [origin, peer])
        self.assertEqual(result["origin_host_count"], 1)
        self.assertEqual(result["peer_host_count"], 1)
        origin_request = calls[0][1]
        peer_request = calls[1][1]
        self.assertEqual(
            origin_request["json"]["stage_timeout_seconds"], origin_request["timeout_s"]
        )
        self.assertEqual(
            peer_request["json"]["stage_timeout_seconds"], peer_request["timeout_s"]
        )
        self.assertEqual(origin_request["total_timeout_s"], origin_request["timeout_s"])
        self.assertEqual(peer_request["total_timeout_s"], peer_request["timeout_s"])
        self.assertLess(peer_request["timeout_s"], origin_request["timeout_s"])
        self.assertLessEqual(origin_request["timeout_s"], 0.15)
        self.assertLessEqual(peer_request["timeout_s"], 0.15)

    def test_predictive_origin_and_peer_stage_concurrently(self):
        _ManifestProbeClient.requests.clear()
        stage_lora_adapter, namespace = _load_stage_function(probe_version=2)
        calls = []

        class Response:
            def json(self):
                return {"status": "ok", "stage_seconds": 0.03}

        async def fake_admin_post_response(client, _path, **_kwargs):
            calls.append((client, time.perf_counter()))
            await asyncio.sleep(0.03)
            return Response()

        namespace["_admin_post_response"] = fake_admin_post_response
        origin = httpx.AsyncClient(base_url="http://127.0.0.1:8000")
        peer = httpx.AsyncClient(base_url="http://127.0.0.1:18049")
        result = asyncio.run(
            stage_lora_adapter(
                [peer, origin],
                "http://adapter/step_2",
                require_delta=True,
                origin_base_urls=["http://127.0.0.1:8000"],
            )
        )

        self.assertEqual({client for client, _ in calls}, {origin, peer})
        self.assertLess(abs(calls[0][1] - calls[1][1]), 0.02)
        self.assertEqual(result["origin_host_count"], 1)
        self.assertEqual(result["peer_host_count"], 1)
        self.assertLess(result["wire_stage_seconds"], 0.10)
        self.assertEqual(len(_ManifestProbeClient.requests), 1)
        self.assertGreater(_ManifestProbeClient.requests[0].read, 10.0)
        self.assertLessEqual(_ManifestProbeClient.requests[0].read, 30.0)


if __name__ == "__main__":
    unittest.main()
