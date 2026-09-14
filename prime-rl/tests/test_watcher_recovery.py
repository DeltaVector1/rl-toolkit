import ast
import asyncio
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _TransportError(Exception):
    pass


class _ConnectError(_TransportError):
    pass


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def load_watcher_start(logger):
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "start"
    )

    async def run_once(_name, operation, *, attempts=3):
        del attempts
        return await operation()

    namespace = {
        "asyncio": asyncio,
        "time": time,
        "httpx": types.SimpleNamespace(
            TransportError=_TransportError,
            ConnectError=_ConnectError,
        ),
        "_run_with_retries": run_once,
        "adapter_stage_limit_seconds": lambda _step: 1.0,
        "get_logger": lambda: logger,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["start"]


class WatcherRecoveryTest(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_repair_waits_for_health_then_reconciles(self):
        logger = _Logger()
        start = load_watcher_start(logger)
        stopped = asyncio.Event()
        health_checks = []
        reconcile_calls = 0

        async def wait_for_ready(model_name):
            health_checks.append(model_name)

        async def reconcile():
            nonlocal reconcile_calls
            reconcile_calls += 1
            if reconcile_calls == 1:
                raise _ConnectError("port 8000 is unavailable")
            stopped.set()

        watcher = types.SimpleNamespace(
            task=None,
            stopped=stopped,
            bootstrap_staged=True,
            poll_interval=0.0,
            compute_next_ckpt_step=lambda: 1,
            ckpt_step=1,
            lora_name="policy-lora",
            last_reconcile_check_at=0.0,
            reconcile_interval=0.0,
            apply_policy_update=None,
            reconcile_current_policy=reconcile,
            inference=types.SimpleNamespace(wait_for_ready=wait_for_ready),
            config=types.SimpleNamespace(model=types.SimpleNamespace(name="base-model")),
        )

        await start(watcher)

        self.assertEqual(reconcile_calls, 2)
        self.assertEqual(health_checks, ["base-model"])
        self.assertIn("inference endpoint outage", logger.warnings[0])

    async def test_reconciliation_timeout_stays_terminal(self):
        logger = _Logger()
        start = load_watcher_start(logger)
        stopped = asyncio.Event()

        async def wait_for_ready(_model_name):
            raise AssertionError("a reconciliation timeout must not enter health recovery")

        async def reconcile():
            raise TimeoutError("adapter stage deadline exceeded")

        watcher = types.SimpleNamespace(
            task=None,
            stopped=stopped,
            bootstrap_staged=True,
            poll_interval=0.0,
            compute_next_ckpt_step=lambda: 1,
            ckpt_step=1,
            lora_name="policy-lora",
            last_reconcile_check_at=0.0,
            reconcile_interval=0.0,
            apply_policy_update=None,
            reconcile_current_policy=reconcile,
            inference=types.SimpleNamespace(wait_for_ready=wait_for_ready),
            config=types.SimpleNamespace(model=types.SimpleNamespace(name="base-model")),
        )

        with self.assertRaisesRegex(TimeoutError, "adapter stage deadline"):
            await start(watcher)


if __name__ == "__main__":
    unittest.main()
