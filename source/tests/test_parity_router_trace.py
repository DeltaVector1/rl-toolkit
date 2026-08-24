import os
import json
import base64
import io
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts.glm52_policy_parity import (
    Profile,
    capture_h200,
    read_secret_stdin,
    validate_diagnostic_file,
    validate_diagnostic_source,
)

from prime_rl.parity_router_trace import (
    TraceSession,
    assemble_chunks,
    clear_trace,
    configure_process_trace,
    configure_hidden_diagnostics,
    h200_trace_request,
    h200_diagnostic_request,
    install_vllm_parity_trace,
    mark_h200_trace_flags,
    read_h200_runtime_marker,
    read_diagnostic_manifest,
    sigmoid_router_reference,
    trace_dir_env,
    trace_session_identity,
    validate_sparse_routes,
    router_debug_enabled,
    write_route_chunk,
    write_router_debug_chunk,
    write_hidden_diagnostic,
    run_h200_router_diagnostic,
    bootstrap_readiness,
    diagnostic_authorization_valid,
    diagnostic_capture_preflight,
    consume_single_use_diagnostic_token,
    consume_single_use_bootstrap_token,
    read_one_use_bootstrap_fifo,
    validate_encoded_diagnostic_input,
    validate_diagnostic_content_length,
    validate_isolated_parity_root,
    validate_h200_diagnostic_results,
    _capture_h200_decoder_input,
    _capture_h200_layer_boundary,
    _capture_h200_attention_boundary,
    _capture_h200_mlp_boundary,
    _slice_diagnostic_tokens,
    _hidden_stream,
    _router_logits,
    diagnostics_enabled,
    write_session,
    _is_mounted,
    _trainer_host_name,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.dtype = np.float32

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def shape(self):
        return self.value.shape

    def detach(self):
        return self

    def to(self, *args, dtype=None, device=None):
        del device
        if args:
            dtype = args[0]
        return FakeTensor(self.value.astype(dtype))

    def numpy(self):
        return self.value

    def __array__(self, dtype=None):
        return np.asarray(self.value, dtype=dtype)

    def reshape(self, *shape):
        return FakeTensor(self.value.reshape(*shape))

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def __mul__(self, other):
        return FakeTensor(self.value * other)

    def __rmul__(self, other):
        return FakeTensor(other * self.value)

    def __add__(self, other):
        return FakeTensor(self.value + (other.value if isinstance(other, FakeTensor) else other))

    def __radd__(self, other):
        return FakeTensor((other.value if isinstance(other, FakeTensor) else other) + self.value)


def fake_torch(state=None):
    state = state or {"compiling": False, "capturing": False}
    module = types.ModuleType("torch")
    module.int32 = np.int32
    module.float32 = np.float32
    module.sigmoid = lambda value: value
    module.tanh = lambda value: value
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
    def test_glm_diagnostic_plan_fits_existing_caps(self):
        metadata = TraceSession(
            **{
                **session().__dict__,
                "hidden_layers": 78,
                "dense_layers": 3,
                "experts": 256,
                "topk": 8,
            }
        )
        environment = {
            "PRIME_PARITY_DIAGNOSTICS": "1",
            "PRIME_PARITY_DIAGNOSTIC_TOKENS": "256",
            "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "64",
            "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "100000000",
        }
        with patch.dict(os.environ, environment, clear=True):
            trainer = diagnostic_capture_preflight(metadata, "trainer")
            h200 = diagnostic_capture_preflight(metadata, "h200")
        self.assertEqual(trainer["files"], 14)
        self.assertEqual(trainer["estimated_bytes"], 88_997_888)
        self.assertEqual(h200["files"], 20)
        self.assertEqual(h200["estimated_bytes"], 96_485_376)
        self.assertEqual(len(h200["names"]), len(set(h200["names"])))

    def test_diagnostic_plan_refuses_existing_cap_overflow(self):
        metadata = TraceSession(
            **{
                **session().__dict__,
                "hidden_layers": 78,
                "dense_layers": 3,
                "experts": 256,
                "topk": 8,
            }
        )
        environment = {
            "PRIME_PARITY_DIAGNOSTICS": "1",
            "PRIME_PARITY_DIAGNOSTIC_TOKENS": "256",
            "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "64",
            "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "96485375",
        }
        with patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
            ValueError, "exceeds"
        ):
            diagnostic_capture_preflight(metadata, "h200")

    def test_direct_router_replay_requires_cp_zero_layer_input(self):
        validate_diagnostic_source("layer3_router_input-rank-0-pp-0-cp-0", 3)
        for name in (
            "layer3_router_input-rank-1-pp-0-cp-1",
            "layer2_router_input-rank-0-pp-0-cp-0",
        ):
            with self.assertRaisesRegex(ValueError, "CP0"):
                validate_diagnostic_source(name, 3)

    def test_hidden_diagnostics_are_off_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(diagnostics_enabled())

    def test_hidden_diagnostic_is_bounded_and_hashed(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_session(root, session())
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_DIAGNOSTICS": "1",
                "PRIME_PARITY_DIAGNOSTIC_TOKENS": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "2000000",
            }
            values = FakeTensor(np.ones((3, 6144), dtype=np.float32))
            with patch.dict(os.environ, environment, clear=True), patch.dict(
                sys.modules, {"torch": torch}
            ):
                configure_process_trace("trainer", rank=0)
                write_hidden_diagnostic("layer0_input", values)
                write_hidden_diagnostic("layer0_input", values)
            path = root / "trainer-rank-0-pp-0-cp-0-diagnostic-layer0_input.npz"
            self.assertTrue(path.is_file())
            manifest = json.loads((root / "parity-diagnostics.json").read_text())
            self.assertEqual(len(manifest["entries"]), 1)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["values"].shape, (2, 6144))
                self.assertEqual(saved["session_identity"].item(), trace_session_identity(session()))

    def test_hidden_diagnostic_rejects_file_limit(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_session(root, session())
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_DIAGNOSTICS": "1",
                "PRIME_PARITY_DIAGNOSTIC_TOKENS": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "1",
                "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "2000000",
            }
            values = FakeTensor(np.ones((2, 6144), dtype=np.float32))
            with patch.dict(os.environ, environment, clear=True), patch.dict(
                sys.modules, {"torch": torch}
            ):
                configure_process_trace("trainer", rank=0)
                write_hidden_diagnostic("layer0_input", values)
                with self.assertRaisesRegex(ValueError, "file or byte limit"):
                    write_hidden_diagnostic("layer0_output", values)

    def test_h200_diagnostic_requires_explicit_enable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            write_session(root, metadata)
            with patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True):
                configure_process_trace("h200", rank=0)
                with self.assertRaisesRegex(ValueError, "diagnostics are disabled"):
                    run_h200_router_diagnostic(object(), {})

    def test_diagnostic_input_rejects_wrong_size_before_decode(self):
        data = base64.b64encode(b"too short").decode()
        with self.assertRaisesRegex(ValueError, "length is invalid"):
            validate_encoded_diagnostic_input(
                {"data": data, "dtype": "float32", "shape": [1, 6144]}
            )

    def test_diagnostic_authentication_rejects_missing_or_wrong_token(self):
        self.assertFalse(diagnostic_authorization_valid(None, "secret"))
        self.assertFalse(diagnostic_authorization_valid("Bearer wrong", "secret"))
        self.assertTrue(diagnostic_authorization_valid("Bearer secret", "secret"))

    def test_diagnostic_token_expires_and_cannot_be_replayed(self):
        tokens = {"session": "secret"}
        expiry = {"session": 10.0}
        self.assertFalse(
            consume_single_use_diagnostic_token(tokens, expiry, "session", "Bearer secret", 10.0)
        )
        tokens["session"] = "secret"
        expiry["session"] = 20.0
        self.assertTrue(
            consume_single_use_diagnostic_token(tokens, expiry, "session", "Bearer secret", 11.0)
        )
        self.assertFalse(
            consume_single_use_diagnostic_token(tokens, expiry, "session", "Bearer secret", 11.0)
        )

    def test_bootstrap_fifo_is_exact_mode_and_removed_after_one_read(self):
        token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "bootstrap.fifo"
            os.mkfifo(fifo, 0o600)
            os.chmod(fifo, 0o600)

            def write():
                fifo.write_text(token + "\n")

            writer = threading.Thread(target=write)
            writer.start()
            observed = read_one_use_bootstrap_fifo(fifo, fifo, os.geteuid())
            writer.join(timeout=5)
            self.assertFalse(writer.is_alive())
            self.assertEqual(observed, token)
            fifo.unlink()
            self.assertFalse(fifo.exists())

    def test_bootstrap_fifo_rejects_regular_file_and_wrong_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bootstrap.fifo"
            path.write_text("not a fifo\n")
            with self.assertRaisesRegex(ValueError, "type, mode, or owner"):
                read_one_use_bootstrap_fifo(path, path, os.geteuid())
            path.unlink()
            os.mkfifo(path, 0o644)
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ValueError, "type, mode, or owner"):
                read_one_use_bootstrap_fifo(path, path, os.geteuid())

    def test_bootstrap_fifo_rejects_insufficient_token_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "bootstrap.fifo"
            os.mkfifo(fifo, 0o600)
            os.chmod(fifo, 0o600)
            writer = threading.Thread(target=lambda: fifo.write_text("a" * 32 + "\n"))
            writer.start()
            with self.assertRaisesRegex(ValueError, "below 384 bits"):
                read_one_use_bootstrap_fifo(fifo, fifo, os.geteuid())
            writer.join(timeout=5)
            self.assertFalse(writer.is_alive())

    def test_bootstrap_authorization_cannot_be_replayed(self):
        token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        authorization = f"Bearer {token}"
        self.assertIsNone(consume_single_use_bootstrap_token(token, authorization))
        token = None
        with self.assertRaisesRegex(ValueError, "authorization"):
            consume_single_use_bootstrap_token(token, authorization)

    def test_bootstrap_readiness_auth_does_not_consume_token(self):
        token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        authorization = f"Bearer {token}"
        self.assertEqual(bootstrap_readiness(None, authorization), "not_ready")
        self.assertEqual(bootstrap_readiness(token, "Bearer wrong"), "unauthorized")
        self.assertEqual(bootstrap_readiness(token, authorization), "ready")
        self.assertEqual(bootstrap_readiness(token, authorization), "ready")
        token = consume_single_use_bootstrap_token(token, authorization)
        self.assertIsNone(token)
        self.assertEqual(bootstrap_readiness(token, authorization), "not_ready")

    def test_secret_stdin_is_bounded_and_has_no_cli_value(self):
        token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        stdin = io.TextIOWrapper(io.BytesIO((token + "\n").encode("ascii")))
        with patch.object(sys, "stdin", stdin):
            self.assertEqual(read_secret_stdin("bootstrap token", 64, 64), token)
        oversized = io.TextIOWrapper(io.BytesIO(("a" * 257 + "\n").encode("ascii")))
        with patch.object(sys, "stdin", oversized), self.assertRaisesRegex(
            ValueError, "too long"
        ):
            read_secret_stdin("diagnostic token", 32, 256)
        source = (Path(__file__).resolve().parents[2] / "scripts/glm52_policy_parity.py").read_text()
        self.assertNotIn("--diagnostic-token", source)
        self.assertNotIn("--bootstrap-token-env", source)

    def test_h200_capture_uses_reset_token_for_direct_replay_without_saving_it(self):
        import scripts.glm52_policy_parity as parity_script

        bootstrap_token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"

        route_ids = np.asarray(
            [[[0, 0], [1, 2]], [[0, 0], [2, 3]]], dtype=np.uint8
        )
        route_weights = np.asarray(
            [[[0.0, 0.0], [0.5, 0.5]], [[0.0, 0.0], [0.5, 0.5]]],
            dtype=np.float32,
        )
        encoded_ids = {
            "data": base64.b64encode(route_ids.tobytes()).decode(),
            "dtype": route_ids.dtype.name,
            "shape": list(route_ids.shape),
            "start": 0,
        }
        encoded_weights = {
            "data": base64.b64encode(route_weights.tobytes()).decode(),
            "dtype": route_weights.dtype.name,
            "shape": list(route_weights.shape),
        }
        active_session = {}
        generate_count = 0
        reset_count = 0
        events = []

        def request(url, body, api_key, timeout):
            nonlocal generate_count, reset_count
            del timeout
            if url.endswith("/parity/bootstrap-ready"):
                self.assertEqual(api_key, bootstrap_token)
                events.append("bootstrap_ready")
                return {"status": "ready"}
            if url.endswith("/v1/models"):
                events.append("models")
                return {"data": [{"id": "model", "max_model_len": 3}]}
            if url.endswith("/parity/router-trace/reset"):
                self.assertEqual(api_key, bootstrap_token)
                reset_count += 1
                events.extend(("reset", "token_issue"))
                active_session.update(body)
                return {"session": body, "diagnostic_token": "one-use-secret"}
            if url.endswith("/inference/v1/generate"):
                generate_count += 1
                events.append("graph_capture" if generate_count == 1 else "eager_capture")
                result = {"request_id": f"request-{generate_count}"}
                if generate_count == 2:
                    result["choices"] = [{"routed_experts": encoded_ids}]
                return result
            if url.endswith("/parity/router-trace"):
                events.append("trace")
                return {
                    "session": active_session,
                    "runtime_marker": {"request_identity": "trace-request"},
                    "router_ids": encoded_ids,
                    "router_weights": encoded_weights,
                }
            raise AssertionError(url)

        def replay_diagnostic(*args):
            self.assertEqual(args[4], "one-use-secret")
            events.append("replay")
            return {"passed": True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "profile.toml"
            profile_path.write_text("test = true\n")
            profile = Profile(
                profile_path,
                {
                    "h200_scoring_context_length": 3,
                    "context_length": 2,
                    "hidden_layers": 2,
                    "dense_layers": 1,
                    "experts_per_token": 2,
                    "experts": 4,
                    "router_scale": 2.5,
                    "max_router_weight_sum_error": 0.01,
                    "max_output_length": 1,
                },
            )
            arrays = {
                "token_ids": np.asarray([1, 2], dtype=np.int64),
                "selected_indices": np.asarray([1], dtype=np.int64),
                "advantages": np.asarray([1.0], dtype=np.float32),
            }
            output = root / "output"
            with patch.object(
                parity_script, "load_manifest", return_value={"seed": 1}
            ), patch.object(
                parity_script, "check_fresh"
            ), patch.object(
                parity_script, "fixture_digest", return_value="fixture-sha"
            ), patch.object(
                parity_script, "load_array", side_effect=lambda fixture, manifest, name: arrays[name]
            ), patch.object(
                parity_script, "request_json", side_effect=request
            ), patch.object(
                parity_script, "prompt_logprobs", return_value=np.asarray([0.0, -0.2])
            ), patch.object(
                parity_script, "graph_sampling_params", return_value={}
            ), patch.object(
                parity_script,
                "trace_sampling_params",
                return_value={"extra_args": {"prime_router_trace_request": "trace-request"}},
            ), patch.object(
                parity_script, "inference_path_metrics", return_value={"passed": True}
            ), patch.object(
                parity_script,
                "run_h200_router_diagnostic",
                side_effect=replay_diagnostic,
            ) as replay:
                manifest = capture_h200(
                    profile,
                    root / "fixture",
                    output,
                    "http://h200",
                    "model",
                    "a" * 40,
                    "b" * 64,
                    "image",
                    None,
                    bootstrap_token,
                    root / "input.npz",
                    30.0,
                )
        self.assertEqual(reset_count, 1)
        self.assertEqual(
            events,
            [
                "bootstrap_ready",
                "models",
                "graph_capture",
                "reset",
                "token_issue",
                "eager_capture",
                "trace",
                "replay",
            ],
        )
        self.assertEqual(replay.call_args.args[4], "one-use-secret")
        self.assertEqual(manifest["direct_router_diagnostic"], {"passed": True})
        self.assertNotIn("one-use-secret", json.dumps(manifest, sort_keys=True))

    def test_router_owner_gate_tuple_and_bias_are_checked(self):
        class Gate:
            def __call__(self, value):
                return value, FakeTensor([1.0, 2.0])

        class Router:
            e_score_correction_bias = FakeTensor([1.0, 2.0])

        class Owner:
            gate = Gate()

        torch = fake_torch()
        torch.equal = lambda left, right: np.array_equal(left.value, right.value)
        with patch.dict(sys.modules, {"torch": torch}):
            result = _router_logits(Owner(), Router(), FakeTensor([[1.0, 2.0]]))
        self.assertIsInstance(result, FakeTensor)
        Router.e_score_correction_bias = FakeTensor([1.0, 3.0])
        with patch.dict(sys.modules, {"torch": torch}), self.assertRaisesRegex(
            ValueError, "correction bias"
        ):
            _router_logits(Owner(), Router(), FakeTensor([[1.0, 2.0]]))

    def test_two_stream_boundary_requires_two_tensors(self):
        first = FakeTensor([[1.0, 2.0]])
        second = FakeTensor([[3.0, 4.0]])
        self.assertIs(_hidden_stream((first, second), "layer"), first)
        with self.assertRaisesRegex(ValueError, "known hidden-state"):
            _hidden_stream((first,), "layer")

    def test_router_diagnostics_capture_only_first_routed_layer(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_session(root, session())
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_DIAGNOSTICS": "1",
                "PRIME_PARITY_DIAGNOSTIC_TOKENS": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "4",
                "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "100000000",
            }
            router_input = FakeTensor(np.ones((2, 6144), dtype=np.float32))
            router_values = FakeTensor(np.ones((2, 4), dtype=np.float32))
            with patch.dict(os.environ, environment, clear=True), patch.dict(
                sys.modules, {"torch": torch}
            ):
                configure_process_trace("trainer", rank=0)
                write_router_debug_chunk(2, router_input, router_values, router_values, router_values)
                self.assertFalse(list(root.glob("*diagnostic-layer2_router_input.npz")))
                write_router_debug_chunk(1, router_input, router_values, router_values, router_values)
            self.assertTrue(list(root.glob("*diagnostic-layer1_router_input.npz")))

    def test_mount_classification_requires_exact_mountpoint(self):
        self.assertTrue(_is_mounted(Path("/app/src/prime_rl/model.py"), {Path("/app/src/prime_rl/model.py")}))
        self.assertFalse(_is_mounted(Path("/app/src/prime_rl/model.py"), {Path("/app/src")}))

    def test_trainer_host_identity_comes_from_launcher(self):
        with patch.dict(os.environ, {"PRIME_TRAINER_HOST": "n82"}, clear=True):
            self.assertEqual(_trainer_host_name(), "n82")
        with patch.dict(os.environ, {"PRIME_TRAINER_HOST": "container-id"}, clear=True):
            with self.assertRaisesRegex(ValueError, "configured trainer host"):
                _trainer_host_name()

    def test_trainer_trace_bound_rejects_second_chunk(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = session()
            write_session(root, metadata)
            ids = FakeTensor([[0, 1]] * 4)
            weights = FakeTensor([[0.4, 0.6]] * 4)
            env = {
                trace_dir_env: str(root),
                "PRIME_PARITY_TRACE_MAX_CALLS": "1",
                "PRIME_PARITY_TRACE_MAX_FILES": "1",
                "PRIME_PARITY_TRACE_MAX_BYTES": "100000",
            }
            with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"torch": torch}):
                configure_process_trace("trainer", rank=1, pp_rank=0, cp_rank=1, world_size=2, pp_size=1, cp_size=2)
                write_route_chunk(1, ids, weights, route_output_scale=2.5)
                with self.assertRaisesRegex(ValueError, "limit exceeded"):
                    write_route_chunk(1, ids, weights, route_output_scale=2.5)

    def test_cp_nonzero_skips_hidden_debug_but_keeps_route_trace(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = session()
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_TRACE_MAX_CALLS": "4",
                "PRIME_PARITY_TRACE_MAX_FILES": "4",
                "PRIME_PARITY_TRACE_MAX_BYTES": "100000",
                "PRIME_PARITY_DIAGNOSTICS": "1",
                "PRIME_PARITY_DIAGNOSTIC_TOKENS": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "4",
                "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "10000000",
                "PRIME_PARITY_ROUTER_DEBUG_TOKENS": "2",
            }
            ids = FakeTensor([[0, 1]] * 4)
            weights = FakeTensor([[0.4, 0.6]] * 4)
            hidden = FakeTensor(np.ones((4, 6144), dtype=np.float32))
            router = FakeTensor(np.ones((4, 4), dtype=np.float32))
            with patch.dict(os.environ, environment, clear=True), patch.dict(
                sys.modules, {"torch": torch}
            ):
                configure_process_trace(
                    "trainer",
                    rank=1,
                    pp_rank=0,
                    cp_rank=1,
                    world_size=2,
                    pp_size=1,
                    cp_size=2,
                )
                write_session(root, metadata)
                write_hidden_diagnostic("layer0_input", hidden)
                write_router_debug_chunk(1, hidden, router, router, router)
                write_route_chunk(1, ids, weights, route_output_scale=2.5)
            self.assertEqual(list(root.glob("*-diagnostic-*.npz")), [])
            self.assertEqual(list(root.glob("*-router-debug.npz")), [])
            self.assertEqual(len(list(root.glob("*-layer-1-call-0.npz"))), 1)

    def test_hidden_hooks_install_only_on_owning_pp_stage(self):
        class Target:
            def __init__(self, layer_idx=None):
                self.layer_idx = layer_idx
                self.self_attn = self
                self.mlp = self
                self.hooks = 0

            def register_forward_hook(self, _hook):
                self.hooks += 1
                return types.SimpleNamespace(remove=lambda: None)

            def register_forward_pre_hook(self, _hook):
                self.hooks += 1
                return types.SimpleNamespace(remove=lambda: None)

        embed = Target()
        layer = Target(layer_idx=0)
        model = types.SimpleNamespace(embed_tokens=embed, layers={0: layer})
        with patch.dict(os.environ, {"PRIME_PARITY_DIAGNOSTICS": "1"}, clear=True):
            configure_hidden_diagnostics(model, 10, 20)
        self.assertEqual(embed.hooks, 0)
        self.assertEqual(layer.hooks, 0)

    def test_router_debug_is_bounded_and_cleared(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = session()
            write_session(root, metadata)
            values = FakeTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_ROUTER_DEBUG_TOKENS": "2",
                "PRIME_PARITY_ROUTER_DEBUG_MAX_BYTES": "100000",
            }
            with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, {"torch": torch}):
                configure_process_trace("trainer", rank=0)
                bias = FakeTensor([1.1, 2.2, 3.3, 4.4])
                write_router_debug_chunk(
                    1,
                    values,
                    values,
                    values,
                    values,
                    bias,
                    "a" * 64,
                    "a" * 64,
                    True,
                )
            path = root / "trainer-rank-0-pp-0-cp-0-layer-1-router-debug.npz"
            self.assertTrue(path.is_file())
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["router_input"].shape, (2, 4))
                self.assertEqual(saved["router_logits"].shape, (2, 4))
                self.assertEqual(saved["pre_bias_scores"].shape, (2, 4))
                self.assertEqual(saved["selection_scores"].shape, (2, 4))
                np.testing.assert_allclose(saved["selection_bias"], bias.value)
                self.assertEqual(saved["selection_bias_data_ptr"].item(), 0)
                self.assertEqual(saved["selection_bias_source_sha256"].item(), "a" * 64)
                self.assertEqual(saved["selection_bias_cache_sha256"].item(), "a" * 64)
                self.assertTrue(saved["selection_bias_is_static"].item())
            clear_trace(root)
            self.assertFalse(path.exists())

    def test_router_debug_slices_before_transfer(self):
        events = []

        class TrackingTensor(FakeTensor):
            def __init__(self, value):
                super().__init__(value)

            def __getitem__(self, key):
                events.append("slice")
                return TrackingTensor(self.value[key])

            def to(self, *args, dtype=None, device=None):
                events.append("transfer")
                return super().to(*args, dtype=dtype, device=device)

        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_session(root, session())
            value = TrackingTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
            environment = {trace_dir_env: str(root), "PRIME_PARITY_ROUTER_DEBUG_TOKENS": "2"}
            with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, {"torch": torch}):
                configure_process_trace("trainer", rank=0)
                write_router_debug_chunk(1, value, value, value, value)
        self.assertEqual(events, ["slice", "transfer"] * 4)

    def test_router_debug_records_the_selection_bias_operand(self):
        torch = fake_torch()
        logits = np.zeros((4, 4), dtype=np.float32)
        exact_bias = np.asarray([0.0, 0.0001, 0.0, 0.0], dtype=np.float32)
        rounded_bias = np.zeros(4, dtype=np.float32)
        exact_ids, exact_weights = sigmoid_router_reference(logits, exact_bias, 2)
        rounded_ids, _ = sigmoid_router_reference(logits, rounded_bias, 2)
        self.assertFalse(np.array_equal(exact_ids, rounded_ids))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_session(root, session())
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_TRACE_MAX_CALLS": "2",
                "PRIME_PARITY_TRACE_MAX_FILES": "2",
                "PRIME_PARITY_TRACE_MAX_BYTES": "100000",
                "PRIME_PARITY_ROUTER_DEBUG_TOKENS": "4",
            }
            pre_bias = FakeTensor(logits)
            selection = FakeTensor(logits + exact_bias[None, :])
            with patch.dict(os.environ, environment, clear=True), patch.dict(sys.modules, {"torch": torch}):
                configure_process_trace("trainer", rank=0)
                write_route_chunk(
                    1,
                    FakeTensor(exact_ids),
                    FakeTensor(exact_weights),
                    route_output_scale=2.5,
                )
                write_router_debug_chunk(
                    1,
                    pre_bias,
                    pre_bias,
                    pre_bias,
                    selection,
                    FakeTensor(exact_bias),
                    "a" * 64,
                    "a" * 64,
                    True,
                )
            debug_path = root / "trainer-rank-0-pp-0-cp-0-layer-1-router-debug.npz"
            route_path = root / "trainer-rank-0-pp-0-cp-0-layer-1-call-0.npz"
            with np.load(debug_path, allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["selection_bias"], exact_bias)
                np.testing.assert_allclose(
                    saved["selection_scores"] - saved["pre_bias_scores"],
                    np.broadcast_to(exact_bias, (4, 4)),
                )
                self.assertFalse(np.allclose(saved["selection_bias"], rounded_bias))
            with np.load(route_path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ids"], exact_ids)
                np.testing.assert_allclose(saved["weights"], exact_weights)

    def test_router_debug_is_disabled_without_trace_root(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_process_trace("trainer", rank=0)
            self.assertFalse(router_debug_enabled(1))

    def test_h200_router_debug_waits_for_an_active_request(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            values = FakeTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                write_router_debug_chunk(1, values, values, values, values)
                self.assertFalse(list(root.glob("*-router-debug.npz")))
                mark_h200_trace_flags(trace_session_identity(metadata), "a" * 32)
                with h200_trace_request(trace_session_identity(metadata), "a" * 32):
                    write_router_debug_chunk(1, values, values, values, values)
            self.assertEqual(len(list(root.glob("*-router-debug.npz"))), 1)

    def test_h200_router_debug_rearms_for_a_new_session(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = TraceSession(**{**session().__dict__, "side": "h200"})
            second = TraceSession(**{**first.__dict__, "fixture_sha256": "5" * 64})
            values = FakeTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, first)
                configure_process_trace("h200", rank=0)
                mark_h200_trace_flags(trace_session_identity(first), "b" * 32)
                with h200_trace_request(trace_session_identity(first), "b" * 32):
                    write_router_debug_chunk(1, values, values, values, values)
                clear_trace(root)
                write_session(root, second)
                mark_h200_trace_flags(trace_session_identity(second), "c" * 32)
                with h200_trace_request(trace_session_identity(second), "c" * 32):
                    write_router_debug_chunk(1, values, values, values, values)
            self.assertEqual(len(list(root.glob("*-router-debug.npz"))), 1)

    def test_h200_router_debug_writes_once_per_session(self):
        torch = fake_torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = TraceSession(**{**session().__dict__, "side": "h200"})
            values = FakeTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
            with (
                patch.dict(os.environ, {trace_dir_env: str(root)}, clear=True),
                patch.dict(sys.modules, {"torch": torch}),
            ):
                write_session(root, metadata)
                configure_process_trace("h200", rank=0)
                mark_h200_trace_flags(trace_session_identity(metadata), "d" * 32)
                with h200_trace_request(trace_session_identity(metadata), "d" * 32):
                    write_router_debug_chunk(1, values, values, values, values)
                    write_router_debug_chunk(1, values, values, values, values)
            self.assertEqual(len(list(root.glob("*-router-debug.npz"))), 1)

    def test_bias_selects_and_unbiased_sigmoid_weights_score(self):
        logits = np.array([[2.0, 1.0, -1.0, -2.0]], dtype=np.float32)
        bias = np.array([-10.0, 0.0, 10.0, 0.0], dtype=np.float32)
        ids, weights = sigmoid_router_reference(logits, bias, topk=2)
        np.testing.assert_array_equal(ids, [[2, 1]])
        raw = 1.0 / (1.0 + np.exp(-logits[0, [2, 1]]))
        expected = raw / raw.sum()
        np.testing.assert_allclose(weights[0], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(weights.dtype, np.float32)

    def test_measured_fp32_bias_changes_layer_three_route_selection(self):
        logits = np.array(
            [[
                2.7464626, -0.7243003, -0.7251897, -0.7362717,
                -0.7686051, -0.6138942, -0.73032445, -0.7487934,
                -0.011393953, -0.024174858, -0.030296402, -0.045026578,
                -0.05115481, -0.057988662, -0.070849776,
            ]],
            dtype=np.float32,
        )
        bias = np.array(
            [
                34.18921, 34.60598, 34.601982, 34.601982,
                34.60798, 34.572998, 34.598984, 34.601982,
                34.383102, 34.412086, 34.379105, 34.381104,
                34.397095, 34.397095, 34.420082,
            ],
            dtype=np.float32,
        )
        full_ids, _ = sigmoid_router_reference(logits, bias, topk=8)
        raw = bias.view(np.uint32)
        low = raw & np.uint32(0xFFFF)
        increment = (
            (low > np.uint32(0x8000))
            | ((low == np.uint32(0x8000)) & (((raw >> np.uint32(16)) & 1) != 0))
        ).astype(np.uint32)
        bf16_bias = ((raw + (increment << np.uint32(16))) & np.uint32(0xFFFF0000)).view(np.float32)
        rounded_ids, _ = sigmoid_router_reference(logits, bf16_bias, topk=8)
        np.testing.assert_array_equal(full_ids[0], np.arange(8, dtype=np.int32))
        np.testing.assert_array_equal(rounded_ids[0], [0, 8, 9, 10, 11, 12, 13, 14])

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
            def select_experts(self, hidden_states=None, router_logits=None, **kwargs):
                del kwargs
                if hidden_states is None:
                    hidden_states = FakeTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
                if router_logits is None:
                    router_logits = hidden_states
                return self._select_experts(hidden_states, router_logits)

        class BaseRouter(FusedMoERouter):
            def __init__(self, ids):
                self.ids = ids
                self.routed_scaling_factor = 1.0

            def _select_experts(self, hidden_states, router_logits):
                del hidden_states, router_logits
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
                        values = FakeTensor([[0.1, 0.2, 0.3, 0.4]] * 4)
                        module.router.select_experts(values, values)
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
                ids, weights = assemble_chunks(root, metadata)
            self.assertEqual(marker["ranks"], list(range(8)))
            self.assertEqual(ids.shape, (4, 3, 2))
            np.testing.assert_allclose(weights[:, 1:, :].sum(axis=-1), 1.0)

    def test_direct_diagnostic_accepts_all_tp8_results_but_writes_once(self):
        result = {
            "session_identity": "a" * 64,
            "request_identity": "b" * 32,
            "layer": 3,
            "input_shape": [2, 6144],
            "logits_sha256": "c" * 64,
            "route_ids_sha256": "d" * 64,
            "route_weights_sha256": "e" * 64,
        }
        self.assertEqual(validate_h200_diagnostic_results([result.copy() for _ in range(8)]), result)
        changed = result.copy()
        changed["route_ids_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "worker results differ"):
            validate_h200_diagnostic_results([result] * 7 + [changed])

    def test_direct_diagnostic_nonzero_worker_does_not_write(self):
        metadata = TraceSession(**{**session().__dict__, "side": "h200"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_DIAGNOSTICS": "1",
                "PRIME_PARITY_DIAGNOSTIC_TOKENS": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "4",
                "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "10000000",
            }
            with patch.dict(os.environ, environment, clear=True), patch.dict(
                sys.modules, {"torch": fake_torch()}
            ):
                configure_process_trace("h200", rank=1, world_size=8)
                write_session(root, metadata)
                with h200_diagnostic_request(trace_session_identity(metadata), "a" * 32):
                    write_hidden_diagnostic(
                        "layer3_h200_input", FakeTensor(np.ones((2, 6144), dtype=np.float32))
                    )
            self.assertEqual(list(root.glob("*-diagnostic-*.npz")), [])

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

    def test_chunked_diagnostic_request_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "chunked"):
            validate_diagnostic_content_length("100", "chunked", 1024)
        with self.assertRaisesRegex(ValueError, "Content-Length"):
            validate_diagnostic_content_length(None, None, 1024)
        with self.assertRaisesRegex(ValueError, "too large"):
            validate_diagnostic_content_length("1025", None, 1024)

    def test_h200_boundaries_use_logical_sums(self):
        captured = []
        hidden = FakeTensor([[1.0, 2.0]])
        residual = FakeTensor([[3.0, 4.0]])
        with patch(
            "prime_rl.parity_router_trace.write_hidden_diagnostic",
            lambda name, value, **kwargs: captured.append((name, value.value.copy())),
        ):
            _capture_h200_decoder_input(None, (None, hidden, residual), 1)
            _capture_h200_layer_boundary(None, (), (hidden, residual), 1)
        self.assertEqual([name for name, _ in captured], ["layer1_input", "layer1_output"])
        np.testing.assert_allclose(captured[0][1], [[4.0, 6.0]])
        np.testing.assert_allclose(captured[1][1], [[4.0, 6.0]])

    def test_h200_attention_and_mlp_names_are_raw_outputs(self):
        captured = []
        value = FakeTensor([[1.0, 2.0]])
        with patch(
            "prime_rl.parity_router_trace.write_hidden_diagnostic",
            lambda name, tensor, **kwargs: captured.append((name, tensor)),
        ):
            _capture_h200_attention_boundary(None, (), value, 0)
            _capture_h200_mlp_boundary(None, (), value, 0)
        self.assertEqual([name for name, _ in captured], ["layer0_attention_raw", "layer0_mlp_raw"])

    def test_direct_h200_diagnostic_creates_and_clears_session_context(self):
        import prime_rl.parity_router_trace as trace_module

        metadata = TraceSession(**{**session().__dict__, "side": "h200"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {
                trace_dir_env: str(root),
                "PRIME_PARITY_DIAGNOSTICS": "1",
                "PRIME_PARITY_DIAGNOSTIC_TOKENS": "2",
                "PRIME_PARITY_DIAGNOSTIC_MAX_FILES": "4",
                "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES": "10000000",
            }
            with patch.dict(os.environ, environment, clear=True):
                configure_process_trace("h200", rank=0)
                write_session(root, metadata)
                with h200_diagnostic_request(trace_session_identity(metadata), "a" * 32):
                    self.assertEqual(
                        trace_module._active_h200_session,
                        trace_session_identity(metadata),
                    )
                self.assertIsNone(trace_module._active_h200_session)
                with self.assertRaisesRegex(ValueError, "manifest is missing"):
                    read_diagnostic_manifest(root, metadata)

    def test_boundary_slices_each_operand_before_addition(self):
        events = []

        class TrackingTensor(FakeTensor):
            def __getitem__(self, key):
                events.append("slice")
                return TrackingTensor(self.value[key])

            def __add__(self, other):
                events.append("add")
                return super().__add__(other)

        first = TrackingTensor(np.ones((1024, 6144), dtype=np.float32))
        second = TrackingTensor(np.ones((1024, 6144), dtype=np.float32))
        sliced_first = _slice_diagnostic_tokens(first)
        sliced_second = _slice_diagnostic_tokens(second)
        _ = sliced_first + sliced_second
        self.assertEqual(events, ["slice", "slice", "add"])
        self.assertEqual(sliced_first.shape[0], 256)

    def test_reset_body_limit_is_small_and_exact(self):
        with self.assertRaisesRegex(ValueError, "too large"):
            validate_diagnostic_content_length("65537", None, 65536)
        self.assertEqual(validate_diagnostic_content_length("12", None, 65536), 12)

    def test_bound_worker_root_detects_directory_replacement(self):
        import prime_rl.parity_router_trace as trace_module

        metadata = session()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trace"
            root.mkdir()
            environment = {trace_dir_env: str(root)}
            with patch.dict(os.environ, environment, clear=True):
                configure_process_trace("trainer", rank=0, pp_size=1, cp_size=1)
                write_session(root, metadata)
                old = Path(temporary) / "old"
                root.rename(old)
                root.mkdir()
                with self.assertRaisesRegex(ValueError, "replaced"):
                    trace_module.read_session(root)

    def test_isolated_root_rejects_equal_or_symlink_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "qualification"
            run = base / "run"
            run.mkdir(parents=True)
            self.assertEqual(validate_isolated_parity_root(run, base), run.resolve())
            with self.assertRaisesRegex(ValueError, "outside"):
                validate_isolated_parity_root(base, base)
            link = Path(temporary) / "link"
            link.symlink_to(run, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                validate_isolated_parity_root(link, base)

    def test_diagnostic_npz_rejects_compression_bomb_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bomb.npz"
            np.savez_compressed(path, values=np.zeros((256, 6144), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "compression ratio"):
                validate_diagnostic_file(path)

    def test_diagnostic_npy_requires_exact_data_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "values.npy"
            np.save(path, np.zeros((1, 6144), dtype=np.float32))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "byte count"):
                validate_diagnostic_file(path)


if __name__ == "__main__":
    unittest.main()
