import ast
import asyncio
from collections.abc import Mapping
from pathlib import Path
import tempfile
import time
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
    async def run_stage_loop(self, duration):
        tree = ast.parse(ORCHESTRATOR_PATH.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "stage_attempt")
        deadlines = []
        clock = [100.0]

        async def stage(*_args, **kwargs):
            deadlines.append(kwargs["delta_deadline"])
            clock[0] += duration
            if len(deadlines) == 1:
                raise TimeoutError("connection closed")
            return {"adapter_identity": {}}

        namespace = dict(time=SimpleNamespace(monotonic=lambda: clock[0]),
                         stage_attempts=2, startup_delta_deadline=700.0,
                         stage_lora_adapter=stage, weights_path="http://source",
                         require_delta=True, sync_version=2,
                         self=SimpleNamespace(policy_inference=SimpleNamespace(
                             admin_clients=[], adapter_origin_base_urls=[])),
                         get_logger=lambda: SimpleNamespace(warning=lambda *_args: None))
        function = ast.AsyncFunctionDef(name="run", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[loop], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(ORCHESTRATOR_PATH), "exec"), namespace)
        try:
            await namespace["run"]()
        except TimeoutError:
            self.assertEqual(deadlines, [700.0])
            raise
        return deadlines

    async def test_retries_share_original_deadline(self):
        self.assertEqual(await self.run_stage_loop(200.0), [700.0, 700.0])

    async def test_expired_budget_cannot_start_second_attempt(self):
        with self.assertRaisesRegex(TimeoutError, "ten-minute"):
            await self.run_stage_loop(600.0)

    def test_verifier_and_stage_share_deadline(self):
        tree = ast.parse(ORCHESTRATOR_PATH.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "sync_version")
        source = ast.unparse(loop)
        self.assertLess(source.index("time.monotonic() + adapter_stage_limit_seconds(sync_version)",
                                     source.index("source_weights_path =")),
                        source.index("await _verify_startup_adapter_delta"))
        self.assertNotIn("startup_delta_deadline =", source[source.index("await _verify_startup_adapter_delta"):])


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

    async def test_watcher_separates_receipt_from_load_and_keeps_full_total(self):
        path = ORCHESTRATOR_PATH.with_name("watcher.py")
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "apply_policy_update")
        block = next(node.body for node in ast.walk(method) if isinstance(node, ast.AsyncWith)
                     and any(isinstance(child, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "t1" for target in child.targets)
                             for child in node.body))
        first = next(i for i, node in enumerate(block) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "t1" for target in node.targets))
        last = next(i for i, node in enumerate(block) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Attribute) and target.attr == "last_update_weights_time"
                            for target in node.targets))
        clock = [0.0]

        async def update(*_args, timings, **_kwargs):
            timings.update(pause=2.0, adapter_load=5.0, resume=3.0)
            clock[0] += 10.0
            return []

        def write(*_args, **_kwargs):
            clock[0] += 4.0

        watcher = SimpleNamespace(lora_name="adapter", config=SimpleNamespace(output_dir=Path("/run")),
                                  inference=SimpleNamespace(update_weights=update, admin_clients=[]))
        namespace = dict(time=SimpleNamespace(perf_counter=lambda: clock[0]), self=watcher,
                         next_step=2, weights_path="http://source", local_weights_path=Path("/run/step_2"),
                         require_delta=True, adapter_identity={}, _write_receivers_ready=write)
        function = ast.AsyncFunctionDef(name="run", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=block[first:last + 1], decorator_list=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
        await namespace["run"]()
        self.assertEqual(watcher.last_adapter_load_time, 5.0)
        self.assertEqual(watcher.last_adapter_pause_time, 2.0)
        self.assertEqual(watcher.last_adapter_resume_time, 3.0)
        self.assertEqual(watcher.last_adapter_receipt_time, 4.0)
        self.assertEqual(watcher.last_update_weights_time, 14.0)
