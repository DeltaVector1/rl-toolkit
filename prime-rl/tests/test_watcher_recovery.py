import ast
import asyncio
import httpx
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


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

    def info(self, message):
        pass


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
    async def test_policy_update_is_never_reentered_by_health_recovery(self):
        start = load_watcher_start(_Logger())
        stopped = asyncio.Event()
        calls = []

        async def apply(step):
            calls.append(step)
            raise _ConnectError("load failed")

        watcher = types.SimpleNamespace(
            task=None, stopped=stopped, bootstrap_staged=True, poll_interval=0.0,
            compute_next_ckpt_step=lambda: 1, ckpt_step=0, lora_name="policy-lora",
            apply_policy_update=apply,
        )
        with self.assertRaisesRegex(_ConnectError, "load failed"):
            await start(watcher)
        self.assertEqual(calls, [1])

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


class SteadyStageBudgetTest(unittest.IsolatedAsyncioTestCase):
    def harness(self, outcomes, *, log_seconds=0.0):
        path = ROOT / "src/prime_rl/orchestrator/watcher.py"
        names = {"_stage_policy_adapter", "apply_policy_update"}
        nodes = [node for node in ast.walk(ast.parse(path.read_text()))
                 if isinstance(node, ast.AsyncFunctionDef) and node.name in names]
        clock, calls, phases, partials = [100.0], [], [], {}
        logger = _Logger()

        async def sleep(delay):
            clock[0] += delay

        async def stage(clients, url, **kwargs):
            step = kwargs["policy_step"]
            if step in partials:
                self.assertEqual(partials[step], (url, "verified-chunk"))
            calls.append((url, kwargs.copy(), clock[0]))
            phases.append(("stage", step))
            seconds, error = outcomes[len(calls) - 1]
            clock[0] += seconds
            if error:
                partials[step] = (url, "verified-chunk")
                raise error
            return {"adapter_identity": {"source_url": url}, "bytes_done": 23}

        def warning(message):
            logger.warnings.append(message)
            clock[0] += log_seconds

        logger.warning = warning
        namespace = dict(
            asyncio=types.SimpleNamespace(sleep=sleep), httpx=httpx,
            time=types.SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]),
            adapter_stage_limit_seconds=lambda _step: 600.0, stage_lora_adapter=stage,
            get_logger=lambda: logger, format_time=str,
            get_broadcast_dir=lambda path: path,
            get_step_path=lambda path, step: path / f"step_{step}",
            inference_weight_path=lambda _config, path, _name: f"http://source/{path.name}",
            _write_receivers_ready=lambda *_args, **_kwargs: phases.append(("ack", None)),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        return namespace, clock, calls, phases, partials, logger

    async def test_two_consecutive_policies_resume_before_boundary_and_load(self):
        await self.exercise_updates()

    async def test_load_failure_does_not_restage_or_advance_version(self):
        await self.exercise_updates(fail_load=True)

    async def exercise_updates(self, *, fail_load=False):
        namespace, clock, calls, phases, partials, logger = self.harness([
            (600.0, TimeoutError("stage timeout")), (150.0, None),
            (600.0, TimeoutError("stage timeout")), (150.0, None),
        ] if not fail_load else [(1.0, None)])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 2):
                (root / f"step_{step}").mkdir()
                (root / f"step_{step}" / "STABLE").touch()

            async def prewarm(*_args):
                return "manifest"

            async def boundary(step):
                phases.append(("boundary", step))

            async def pending(step):
                phases.append(("pending", step))

            async def update(*_args, step, **_kwargs):
                phases.extend((name, step) for name in ("pause", "load", "resume"))
                if fail_load:
                    raise TimeoutError("load timeout")
                return []

            async def ready(_name):
                phases.append(("ready", None))

            async def notified(step):
                phases.append(("version", step))

            logger.debug = lambda _message: None
            watcher = types.SimpleNamespace(
                update_lock=asyncio.Lock(), ckpt_step=0, model_express=None,
                config=types.SimpleNamespace(output_dir=root), lora_name="policy",
                _prewarm_adapter_manifest=prewarm, last_adapter_delta_build_time=0.0,
                policy=types.SimpleNamespace(version=0, model_name="base"), update_count=0,
                inference=types.SimpleNamespace(admin_clients=[], adapter_origin_base_urls=[],
                    update_weights=update, update_model_name=lambda _name: None, wait_for_ready=ready),
                observers=[types.SimpleNamespace(wait_for_policy_update_boundary=boundary,
                    on_version_pending=pending, on_new_version=notified)],
            )
            if fail_load:
                with self.assertRaisesRegex(TimeoutError, "load timeout"):
                    await namespace["apply_policy_update"](watcher, 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(watcher.policy.version, 0)
                self.assertNotIn(("ack", None), phases)
                return
            for step in (1, 2):
                await namespace["apply_policy_update"](watcher, step)
                self.assertEqual(watcher.policy.version, step)
            self.assertEqual(watcher.update_count, 2)
        self.assertEqual([call[1]["delta_deadline"] for call in calls],
                         [700.0, 1000.0, 1450.5, 1750.5])
        for step in (1, 2):
            self.assertEqual([name for name, version in phases if version == step],
                             ["stage", "stage", "boundary", "pending", "pause", "load", "resume", "version"])
        self.assertEqual(len(partials), 2)
        self.assertEqual(len(logger.warnings), 2)

    async def test_second_timeout_is_terminal_with_930s_response_bound(self):
        namespace, clock, calls, *_ = self.harness([
            (630.0, TimeoutError("first")), (299.5, TimeoutError("second")),
        ])
        start = load_watcher_start(_Logger())

        async def apply(step):
            await namespace["_stage_policy_adapter"]([], "http://source/step_1",
                policy_step=step, delta_deadline=700.0, origin_base_urls=[])

        watcher = types.SimpleNamespace(task=None, stopped=asyncio.Event(), bootstrap_staged=True,
            poll_interval=0.0, compute_next_ckpt_step=lambda: 1, ckpt_step=0,
            lora_name="policy", apply_policy_update=apply)
        with self.assertRaisesRegex(TimeoutError, "second"):
            await start(watcher)
        self.assertEqual(len(calls), 2)
        self.assertEqual(clock[0] - 100.0, 930.0)
        self.assertEqual([call[1]["delta_deadline"] for call in calls], [700.0, 1000.0])

    async def test_fatal_matrix_never_retries(self):
        request = httpx.Request("POST", "http://host/stage_lora_adapter")
        errors = [RuntimeError(name) for name in
                  ("manifest", "base", "config", "target", "checksum", "protocol", "receipt")]
        errors += [OSError("storage"), httpx.ConnectError("connection"),
                   asyncio.CancelledError(), httpx.ReadTimeout("unknown remote state")]
        errors += [httpx.HTTPStatusError("fatal", request=request,
                   response=httpx.Response(status, json=payload, request=request))
                   for status, payload in ((500, {"error": "storage"}), (400, {}), (504, {}),
                                            (504, {"error_type": "TimeoutError", "phase": "load"}))]
        for error in errors:
            with self.subTest(error=repr(error)):
                namespace, _clock, calls, *_ = self.harness([(1.0, error)])
                with self.assertRaises(type(error)):
                    await namespace["_stage_policy_adapter"]([], "http://source/step_1",
                        policy_step=1, delta_deadline=700.0, origin_base_urls=[])
                self.assertEqual(len(calls), 1)

    async def test_confirmed_504_has_fresh_bounded_attempt(self):
        request = httpx.Request("POST", "http://host/stage_lora_adapter")
        error = httpx.HTTPStatusError("timeout", request=request, response=httpx.Response(
            504, request=request, json={"error_type": "TimeoutError", "phase": "staging"}))
        namespace, _clock, calls, *_ = self.harness([(20.0, error), (1.0, None)])
        await namespace["_stage_policy_adapter"]([], "http://source/step_1",
            policy_step=1, delta_deadline=700.0, origin_base_urls=[])
        self.assertEqual([call[1]["delta_deadline"] for call in calls], [700.0, 720.5])
        self.assertTrue(all(call[1]["delta_deadline"] - call[2] <= 600.0 for call in calls))

    async def test_backoff_and_logging_cannot_extend_total_budget(self):
        for log_seconds, expected_calls in ((10.0, 2), (270.0, 1)):
            namespace, _clock, calls, *_ = self.harness(
                [(630.0, TimeoutError("timeout")), (1.0, None)], log_seconds=log_seconds)
            operation = namespace["_stage_policy_adapter"]([], "http://source/step_1",
                policy_step=1, delta_deadline=700.0, origin_base_urls=[])
            if expected_calls == 1:
                with self.assertRaisesRegex(TimeoutError, "deadline expired"):
                    await operation
            else:
                await operation
                self.assertEqual(calls[1][1]["delta_deadline"], 1000.0)
                self.assertEqual(calls[1][2], 740.5)
            self.assertEqual(len(calls), expected_calls)


if __name__ == "__main__":
    unittest.main()
