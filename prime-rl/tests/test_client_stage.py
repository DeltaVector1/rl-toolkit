import ast
import asyncio
import httpx
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"


def _load_stage_function():
    source = CLIENT_PATH.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage_lora_adapter"
    )
    namespace = {
        "AsyncClient": httpx.AsyncClient,
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


class StageLoraAdapterTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
