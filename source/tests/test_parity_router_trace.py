import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prime_rl.parity_router_trace import (
    TraceSession,
    assemble_chunks,
    clear_trace,
    configure_process_trace,
    h200_trace_request,
    install_vllm_parity_trace,
    mark_h200_trace_flags,
    read_h200_runtime_marker,
    sigmoid_router_reference,
    trace_dir_env,
    trace_session_identity,
    validate_sparse_routes,
    write_route_chunk,
    write_session,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def to(self, *, dtype, device):
        del device
        return FakeTensor(self.value.astype(dtype))

    def numpy(self):
        return self.value


def fake_torch(state=None):
    state = state or {"compiling": False, "capturing": False}
    module = types.ModuleType("torch")
    module.int32 = np.int32
    module.float32 = np.float32
    module.compiler = types.SimpleNamespace(
        is_compiling=lambda: state["compiling"]
    )
    module.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        is_current_stream_capturing=lambda: state["capturing"],
    )
    return module


def session() -> TraceSession:
    return TraceSession(
        side="trainer",
        fixture_sha256="1" * 64,
        token_ids_sha256="2" * 64,
        model_revision="3" * 40,
        adapter_sha256="4" * 64,
        image="prime-rl:test",
        expected_tokens=4,
        hidden_layers=3,
        dense_layers=1,
        topk=2,
        experts=4,
        route_output_scale=2.5,
        weight_sum_tolerance=0.005,
    )


class ParityRouterTraceTest(unittest.TestCase):
    def test_bias_selects_and_unbiased_sigmoid_weights_score(self):
        logits = np.array([[2.0, 1.0, -1.0, -2.0]], dtype=np.float32)
        bias = np.array([-10.0, 0.0, 10.0, 0.0], dtype=np.float32)
        ids, weights = sigmoid_router_reference(logits, bias, topk=2)
        np.testing.assert_array_equal(ids, [[2, 1]])
        raw = 1.0 / (1.0 + np.exp(-logits[0, [2, 1]]))
        expected = raw / raw.sum()
        np.testing.assert_allclose(weights[0], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(weights.dtype, np.float32)

    def test_ties_are_stable_and_weights_sum_to_one(self):
        logits = np.zeros((2, 4), dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)
        ids, weights = sigmoid_router_reference(logits, bias, topk=2)
        np.testing.assert_array_equal(ids, [[0, 1], [0, 1]])
        np.testing.assert_allclose(weights, np.full((2, 2), 0.5, np.float32))
        np.testing.assert_allclose(weights.sum(axis=-1), [1.0, 1.0])

    def test_chunks_preserve_cp_call_and_layer_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = session()
            write_session(root, metadata)
            for layer in (1, 2):
                for cp in (0, 1):
                    ids = np.array(
                        [[cp, cp + 2]], dtype=np.int32
                    )
                    weights = np.array([[0.4, 0.6]], dtype=np.float32)
                    path = root / f"trainer-rank-{cp}-pp-0-cp-{cp}-layer-{layer}-call-0.npz"
                    np.savez(
                        path,
                        ids=ids,
                        weights=weights,
                    )
                    path = root / f"trainer-rank-{cp}-pp-0-cp-{cp}-layer-{layer}-call-1.npz"
                    np.savez(
                        path,
                        ids=ids[::-1],
                        weights=weights[::-1],
                    )
            ids, weights = assemble_chunks(root, metadata)
            self.assertEqual(ids.shape, (4, 3, 2))
            self.assertEqual(weights.shape, (4, 3, 2))
            np.testing.assert_array_equal(ids[:, 1, :], [[0, 2], [0, 2], [1, 3], [1, 3]])
            np.testing.assert_allclose(weights[:, 2, :].sum(axis=1), 1.0)

    def test_disabled_default_does_not_import_vllm(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(configure_process_trace("trainer", rank=0))
            self.assertFalse(install_vllm_parity_trace())

    def test_session_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = session()
            write_session(root, first)
            changed = TraceSession(**{**first.__dict__, "adapter_sha256": "5" * 64})
            with self.assertRaisesRegex(ValueError, "does not match"):
                write_session(root, changed)

    def test_new_h200_session_restarts_chunk_order(self):
        torch = fake_torch()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = session()
            first = TraceSession(**{**first.__dict__, "side": "h200"})
            second = TraceSession(**{**first.__dict__, "fixture_sha256": "5" * 64})
            ids = FakeTensor([[0, 1]] * 4)
            weights = FakeTensor([[0.4, 0.6]] * 4)
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, first)
                configure_process_trace("h200", rank=0)
                mark_h200_trace_flags(trace_session_identity(first), "1" * 32)
                with h200_trace_request(trace_session_identity(first), "1" * 32):
                    write_route_chunk(
                        1, ids, weights, route_output_scale=first.route_output_scale
                    )
                clear_trace(root)
                write_session(root, second)
                mark_h200_trace_flags(trace_session_identity(second), "2" * 32)
                with h200_trace_request(trace_session_identity(second), "2" * 32):
                    write_route_chunk(
                        1, ids, weights, route_output_scale=second.route_output_scale
                    )
            self.assertTrue((root / "h200-rank-0-pp-0-cp-0-layer-1-call-0.npz").is_file())
            self.assertFalse((root / "h200-rank-0-pp-0-cp-0-layer-1-call-1.npz").exists())

    def test_compile_and_cuda_capture_data_are_not_written(self):
        state = {"compiling": True, "capturing": False}
        torch = fake_torch(state)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            stale_ids = FakeTensor([[3, 2]] * 4)
            fresh_ids = FakeTensor([[0, 1]] * 4)
            weights = FakeTensor([[0.4, 0.6]] * 4)
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                mark_h200_trace_flags(trace_session_identity(metadata), "3" * 32)
                with h200_trace_request(trace_session_identity(metadata), "3" * 32):
                    write_route_chunk(
                        1,
                        stale_ids,
                        weights,
                        route_output_scale=metadata.route_output_scale,
                    )
                    state["compiling"] = False
                    state["capturing"] = True
                    write_route_chunk(
                        1,
                        stale_ids,
                        weights,
                        route_output_scale=metadata.route_output_scale,
                    )
                    state["capturing"] = False
                    write_route_chunk(
                        1,
                        fresh_ids,
                        weights,
                        route_output_scale=metadata.route_output_scale,
                    )
            path = root / "h200-rank-0-pp-0-cp-0-layer-1-call-0.npz"
            with np.load(path, allow_pickle=False) as values:
                np.testing.assert_array_equal(values["ids"], fresh_ids.value)
            self.assertFalse(
                (root / "h200-rank-0-pp-0-cp-0-layer-1-call-1.npz").exists()
            )

    def test_request_chunks_update_in_order_for_every_layer(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                mark_h200_trace_flags(trace_session_identity(metadata), "4" * 32)
                for chunk in range(2):
                    with h200_trace_request(
                        trace_session_identity(metadata), "4" * 32
                    ):
                        for layer in (1, 2):
                            ids = FakeTensor([[chunk, chunk + 2]] * 2)
                            weights = FakeTensor([[0.4, 0.6]] * 2)
                            write_route_chunk(
                                layer,
                                ids,
                                weights,
                                route_output_scale=metadata.route_output_scale,
                            )
                ids, weights = assemble_chunks(root, metadata)
                marker = read_h200_runtime_marker(root, metadata)
            np.testing.assert_array_equal(
                ids[:, 1, :], [[0, 2], [0, 2], [1, 3], [1, 3]]
            )
            np.testing.assert_allclose(weights.sum(axis=-1)[:, 1:], 1.0)
            self.assertEqual(marker["layer_calls"], {"1": 2, "2": 2})
            self.assertEqual(
                marker["id_sources"], {"1": "modular_actual", "2": "modular_actual"}
            )
            self.assertEqual(
                marker["route_output_scales"], {"1": 2.5, "2": 2.5}
            )

    def test_lifecycle_reports_each_missing_runtime_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                with self.assertRaisesRegex(ValueError, "flags not received"):
                    read_h200_runtime_marker(root, metadata)
                mark_h200_trace_flags(trace_session_identity(metadata), "6" * 32)
                with self.assertRaisesRegex(ValueError, "context not entered"):
                    read_h200_runtime_marker(root, metadata)
                with h200_trace_request(trace_session_identity(metadata), "6" * 32):
                    pass
                with self.assertRaisesRegex(ValueError, "router not called"):
                    read_h200_runtime_marker(root, metadata)

    def test_lifecycle_marker_keeps_rank_session_and_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True):
                write_session(root, metadata)
                configure_process_trace("h200", rank=3, world_size=4)
                mark_h200_trace_flags(trace_session_identity(metadata), "7" * 32)
                with h200_trace_request(trace_session_identity(metadata), "7" * 32):
                    pass
            marker = __import__("json").loads(
                (root / "h200-lifecycle-rank-3.json").read_text()
            )
            self.assertEqual(marker["rank"], 3)
            self.assertEqual(marker["session_identity"], trace_session_identity(metadata))
            self.assertEqual(marker["request_identity"], "7" * 32)
            self.assertTrue(marker["context_entered"])

    def test_modular_router_wrapper_records_executed_routes(self):
        torch = fake_torch()

        class FusedMoERouter:
            def select_experts(self):
                return self._select_experts()

        class BaseRouter(FusedMoERouter):
            def __init__(self, ids):
                self.ids = ids
                self.routed_scaling_factor = 1.0

            def _select_experts(self):
                return FakeTensor([[0.4, 0.6]] * 4), FakeTensor([self.ids] * 4)

        class MoERunner:
            def __init__(self, layer_id, ids):
                self.layer_id = layer_id
                self.router = BaseRouter(ids)
                self.routed_scaling_factor = 2.5

        class Model:
            def __init__(self):
                self.runners = [MoERunner(1, [0, 2]), MoERunner(2, [1, 3])]

            def modules(self):
                return self.runners

        class GPUModelRunner:
            def __init__(self):
                self.model = Model()

            def _bind_routed_experts_capturer(self, capturer):
                del capturer
                return "bound"

        modules = {}
        for name in (
            "vllm",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.fused_moe",
            "vllm.model_executor.layers.fused_moe.router",
            "vllm.model_executor.layers.fused_moe.runner",
            "vllm.distributed",
            "vllm.v1",
            "vllm.v1.worker",
        ):
            modules[name] = types.ModuleType(name)
        base_module = types.ModuleType(
            "vllm.model_executor.layers.fused_moe.router.base_router"
        )
        base_module.BaseRouter = BaseRouter
        modules[base_module.__name__] = base_module
        fused_module = types.ModuleType(
            "vllm.model_executor.layers.fused_moe.router.fused_moe_router"
        )
        fused_module.FusedMoERouter = FusedMoERouter
        modules[fused_module.__name__] = fused_module
        parallel_module = types.ModuleType("vllm.distributed.parallel_state")
        parallel_module.get_tensor_model_parallel_rank = lambda: 0
        parallel_module.get_tensor_model_parallel_world_size = lambda: 1
        modules[parallel_module.__name__] = parallel_module
        layer_module = types.ModuleType("vllm.model_executor.layers.fused_moe.layer")
        layer_module.MoERunner = MoERunner
        modules[layer_module.__name__] = layer_module
        gpu_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")
        gpu_module.GPUModelRunner = GPUModelRunner
        modules[gpu_module.__name__] = gpu_module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with (
                patch.dict(
                    os.environ,
                    {trace_dir_env: str(root), "RANK": "0"},
                    clear=True,
                ),
                patch.dict(sys.modules, {**modules, "torch": torch}),
            ):
                write_session(root, metadata)
                self.assertTrue(install_vllm_parity_trace())
                self.assertTrue(
                    getattr(FusedMoERouter, "_prime_parity_trace_installed", False)
                )
                self.assertNotIn("_prime_parity_trace_installed", BaseRouter.__dict__)
                runner = GPUModelRunner()
                self.assertEqual(runner._bind_routed_experts_capturer(None), "bound")
                mark_h200_trace_flags(trace_session_identity(metadata), "8" * 32)
                with h200_trace_request(trace_session_identity(metadata), "8" * 32):
                    for module in runner.model.modules():
                        module.router.select_experts()
                ids, _weights = assemble_chunks(root, metadata)
                marker = read_h200_runtime_marker(root, metadata)
        np.testing.assert_array_equal(ids[:, 1, :], [[0, 2]] * 4)
        np.testing.assert_array_equal(ids[:, 2, :], [[1, 3]] * 4)
        self.assertEqual(
            marker["id_sources"], {"1": "modular_actual", "2": "modular_actual"}
        )

    def test_missing_tensor_parallel_rank_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0, world_size=2)
                mark_h200_trace_flags(trace_session_identity(metadata), "9" * 32)
                with h200_trace_request(trace_session_identity(metadata), "9" * 32):
                    pass
                with self.assertRaisesRegex(ValueError, "rank coverage is incomplete"):
                    read_h200_runtime_marker(root, metadata)

    def test_all_tensor_parallel_workers_enter_the_same_context(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            session_id = trace_session_identity(metadata)
            request_id = "a" * 32
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, metadata)
                for rank in range(8):
                    configure_process_trace("h200", rank=rank, world_size=8)
                    mark_h200_trace_flags(session_id, request_id)
                    with h200_trace_request(session_id, request_id):
                        if rank == 0:
                            for layer in (1, 2):
                                write_route_chunk(
                                    layer,
                                    FakeTensor([[0, 1]] * 4),
                                    FakeTensor([[0.4, 0.6]] * 4),
                                    route_output_scale=metadata.route_output_scale,
                                )
                marker = read_h200_runtime_marker(root, metadata)
            self.assertEqual(marker["ranks"], list(range(8)))

    def test_h200_request_session_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                with (
                    self.assertRaisesRegex(ValueError, "session does not match"),
                ):
                    mark_h200_trace_flags("9" * 64, "5" * 32)

    def test_normal_h200_request_does_not_write_trace_data(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                write_route_chunk(
                    1,
                    FakeTensor([[0, 1]] * 4),
                    FakeTensor([[0.4, 0.6]] * 4),
                    route_output_scale=metadata.route_output_scale,
                )
            self.assertEqual(list(root.glob("h200-*.npz")), [])

    def test_vllm_patch_forces_only_trace_request_to_raw_eager_model(self):
        root = Path(__file__).resolve().parents[2]
        patch_text = (
            root / "patches/vllm/0004-v026-request-exact-router-trace.patch"
        ).read_text()
        self.assertIn('extra_args.get("prime_router_trace") is True', patch_text)
        self.assertIn("an exact router trace request must run alone", patch_text)
        self.assertIn("force_eager=eager_router_trace", patch_text)
        self.assertIn(
            "skip_compiled=has_encoder_input or eager_router_trace", patch_text
        )
        self.assertIn("router_trace_context = h200_trace_request", patch_text)
        self.assertIn("mark_h200_trace_flags(session, request)", patch_text)

    def test_monolithic_route_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ID source is invalid"):
            write_route_chunk(
                1,
                FakeTensor([[0, 1]]),
                FakeTensor([[0.4, 0.6]]),
                route_output_scale=2.5,
                id_source="monolithic_actual",
            )

    def test_invalid_weight_sum_is_rejected(self):
        ids = np.array([[0, 1]], dtype=np.int32)
        weights = np.array([[0.2, 0.3]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "sum error"):
            validate_sparse_routes(ids, weights, topk=2, experts=4)

    def test_selected_router_weights_sum_to_one(self):
        ids = np.array([[0, 1]], dtype=np.int32)
        weights = np.array([[0.4, 0.6]], dtype=np.float32)
        _, checked = validate_sparse_routes(ids, weights, topk=2, experts=4)
        np.testing.assert_allclose(checked.sum(axis=1), 1.0)

    def test_route_output_scale_is_required_and_valid(self):
        for scale in (None, 0.0, -1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "scale must be finite and positive"):
                TraceSession(
                    **{**session().__dict__, "route_output_scale": scale}
                ).validate()


if __name__ == "__main__":
    unittest.main()
