import ast
import asyncio
from collections.abc import Mapping
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import test_source_deadline
from test_resume_recovery import ORCHESTRATOR_PATH, CLIENT_PATH


def startup_verifier():
    node = next(node for node in ast.parse(ORCHESTRATOR_PATH.read_text()).body
                if isinstance(node, ast.AsyncFunctionDef)
                and node.name == "_verify_startup_adapter_delta")
    namespace = test_source_deadline.manifest_readers()
    namespace.update(
        OrchestratorConfig=object,
        get_broadcast_dir=lambda path: path,
        get_step_path=lambda path, step: path / f"step_{step}",
        inference_weight_path=lambda *_args: "http://unused",
        adapter_stage_limit_seconds=lambda _step: 600.0,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ORCHESTRATOR_PATH), "exec"), namespace)
    return namespace["_verify_startup_adapter_delta"]


class StartupManifestTest(test_source_deadline.ManifestDeadlineTest):
    async def probe(self, parts, *, timeout=0.15, error=TimeoutError):
        # Run the shared transport fixtures through the actual startup verifier.
        original = test_source_deadline.manifest_readers
        verifier = startup_verifier()

        async def verify(url, deadline):
            return await verifier(SimpleNamespace(output_dir=Path("/unused")),
                                  "adapter", 1, Path("/unused"), url, deadline)

        test_source_deadline.manifest_readers = lambda: {"_read_remote_adapter_manifest": verify}
        try:
            await super().probe(parts, timeout=timeout, error=error)
        finally:
            test_source_deadline.manifest_readers = original

    async def test_local_oversize_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "adapter_manifest.json").write_bytes(b" " * (2 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(RuntimeError, "size limit"):
                await startup_verifier()(SimpleNamespace(output_dir=path), "adapter", 1, path)


class StartupBudgetTest(unittest.IsolatedAsyncioTestCase):
    def stage_loop(self, outcomes, *, require_delta=True, warning_seconds=0.0):
        tree = ast.parse(ORCHESTRATOR_PATH.read_text())
        block = next(node.body for node in ast.walk(tree) if isinstance(node, ast.If)
                     and any(isinstance(child, ast.For)
                             and isinstance(child.target, ast.Name)
                             and child.target.id == "stage_attempt" for child in node.body))
        last = next(index for index, node in enumerate(block) if isinstance(node, ast.For))
        calls = []
        clock = [100.0]
        partial = {}
        identity = {"source_url": "http://source/step_2", "target_sha256": "a" * 64}

        async def stage(_clients, path, **kwargs):
            calls.append((path, kwargs.copy(), clock[0]))
            duration, error = outcomes[len(calls) - 1]
            if partial:
                self.assertEqual(partial, {"identity": identity, "verified_chunks": [0]})
                self.assertEqual(path, identity["source_url"])
            clock[0] += duration
            if error:
                partial.update(identity=identity, verified_chunks=[0])
                raise error
            return {"adapter_identity": identity, "cache_hit": False}

        def warning(*_args):
            clock[0] += warning_seconds

        namespace = dict(time=SimpleNamespace(monotonic=lambda: clock[0]),
                         startup_delta_deadline=700.0 if require_delta else None,
                         adapter_stage_limit_seconds=lambda _step: 600.0,
                         stage_lora_adapter=stage, weights_path=identity["source_url"],
                         require_delta=require_delta, sync_version=2,
                         self=SimpleNamespace(policy_inference=SimpleNamespace(
                             admin_clients=[], adapter_origin_base_urls=[])),
                         get_logger=lambda: SimpleNamespace(warning=warning))
        function = ast.AsyncFunctionDef(name="run", args=ast.arguments(
            posonlyargs=[], args=[ast.arg(arg="startup_delta_deadline")], kwonlyargs=[],
            kw_defaults=[], defaults=[ast.Name(id="startup_delta_deadline", ctx=ast.Load())]),
            body=[*block[:last + 1], ast.Return(value=ast.Name(id="stage_result", ctx=ast.Load()))],
            decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(ORCHESTRATOR_PATH), "exec"), namespace)
        return namespace["run"], calls, clock, partial, identity

    async def test_first_timeout_resumes_same_partial_with_fresh_deadline(self):
        run, calls, clock, partial, identity = self.stage_loop([
            (600.0, TimeoutError("stage timed out")), (20.0, None),
        ])
        self.assertEqual(await run(), {"adapter_identity": identity, "cache_hit": False})
        self.assertEqual([call[1]["delta_deadline"] for call in calls], [700.0, 1000.0])
        self.assertEqual([call[1]["policy_step"] for call in calls], [2, 2])
        self.assertTrue(all(call[1]["require_delta"] for call in calls))
        self.assertEqual(partial["verified_chunks"], [0])
        self.assertEqual(clock[0] - 100.0, 620.0)

    async def test_second_timeout_fails_after_exactly_two_attempts(self):
        run, calls, clock, _partial, _identity = self.stage_loop([
            (630.0, TimeoutError("first timeout")),
            (300.0, TimeoutError("second timeout")),
        ])
        with self.assertRaisesRegex(TimeoutError, "second timeout"):
            await run()
        self.assertEqual(len(calls), 2)
        self.assertEqual([call[1]["delta_deadline"] for call in calls], [700.0, 1000.0])
        self.assertEqual([call[1]["delta_deadline"] - call[2] for call in calls], [600.0, 270.0])
        self.assertEqual(clock[0] - 100.0, 930.0)

    async def test_retry_bookkeeping_cannot_extend_total_deadline(self):
        run, calls, _clock, _partial, _identity = self.stage_loop([
            (630.0, TimeoutError("stage timed out")), (1.0, None),
        ], warning_seconds=10.0)
        await run()
        self.assertEqual(calls[1][1]["delta_deadline"], 1000.0)
        self.assertEqual(calls[1][1]["delta_deadline"] - calls[1][2], 260.0)

    async def test_expired_total_budget_cannot_start_another_attempt(self):
        run, calls, *_rest = self.stage_loop([
            (630.0, TimeoutError("stage timed out")),
        ], warning_seconds=270.0)
        with self.assertRaisesRegex(TimeoutError, "staging deadline expired"):
            await run()
        self.assertEqual(len(calls), 1)

    async def test_early_transport_error_still_gets_fresh_bounded_retry(self):
        run, calls, *_rest = self.stage_loop([
            (20.0, OSError("connection closed")), (1.0, None),
        ])
        await run()
        self.assertEqual([call[1]["delta_deadline"] for call in calls], [700.0, 720.0])

    async def test_cancellation_does_not_retry(self):
        run, calls, *_rest = self.stage_loop([(1.0, asyncio.CancelledError())])
        with self.assertRaises(asyncio.CancelledError):
            await run()
        self.assertEqual(len(calls), 1)

    async def test_integrity_errors_and_wrong_cached_identity_do_not_retry(self):
        for message in ("identity mismatch in cached adapter", "checksum mismatch",
                        "policy step differs", "manifest is invalid"):
            with self.subTest(message=message):
                run, calls, *_rest = self.stage_loop([(1.0, RuntimeError(message))])
                with self.assertRaisesRegex(RuntimeError, message):
                    await run()
                self.assertEqual(len(calls), 1)

    async def test_full_adapter_still_has_only_one_attempt(self):
        run, calls, *_rest = self.stage_loop([
            (1.0, TimeoutError("full stage timed out")),
        ], require_delta=False)
        with self.assertRaisesRegex(TimeoutError, "full stage timed out"):
            await run()
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0][1]["delta_deadline"])

    async def test_first_success_does_not_retry(self):
        run, calls, *_rest = self.stage_loop([(1.0, None)])
        await run()
        self.assertEqual(len(calls), 1)

    def test_verifier_and_first_stage_share_deadline(self):
        tree = ast.parse(ORCHESTRATOR_PATH.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "sync_version")
        source = ast.unparse(loop)
        self.assertLess(source.index("time.monotonic() + adapter_stage_limit_seconds(sync_version)",
                                     source.index("source_weights_path =")),
                        source.index("await _verify_startup_adapter_delta"))
        retry_loop = next(node for node in ast.walk(loop) if isinstance(node, ast.For)
                          and isinstance(node.target, ast.Name) and node.target.id == "stage_attempt")
        self.assertEqual(ast.unparse(retry_loop.body[0].test),
                         "stage_attempt > 1 and startup_delta_deadline is not None")


class UpdateTimingTest(unittest.IsolatedAsyncioTestCase):
    async def test_separate_phases_and_failed_load_cleanup(self):
        node = next(node for node in ast.parse(CLIENT_PATH.read_text()).body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "update_weights")
        for fail in (False, True):
            clock = [10.0]
            phases = []

            async def phase(name, seconds, fail=False):
                phases.append(name)
                clock[0] += seconds
                if fail:
                    raise RuntimeError("load failed")
                return ["receipt"]

            namespace = dict(AsyncClient=object, Path=Path, Mapping=Mapping,
                             AdapterIdentity=dict, POLICY_PHASE_TIMEOUT_S=600.0,
                             get_logger=lambda: None,
                             time=SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0]),
                             _pause_engines=lambda *_a, **_k: phase("pause", 2.0),
                             load_lora_adapter=lambda *_a, **_k: phase("adapter_load", 5.0, fail),
                             _resume_engines=lambda *_a, **_k: phase("resume", 3.0))
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(CLIENT_PATH), "exec"), namespace)
            timings = {}
            operation = namespace["update_weights"]([], "http://source", lora_name="adapter", timings=timings)
            if fail:
                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    await operation
            else:
                self.assertEqual(await operation, ["receipt"])
            self.assertEqual(phases, ["pause", "adapter_load", "resume"])
            self.assertEqual(timings, {"pause": 2.0, "adapter_load": 5.0, "resume": 3.0})
