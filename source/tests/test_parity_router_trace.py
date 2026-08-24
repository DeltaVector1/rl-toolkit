import os
import json
import base64
import io
import inspect
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import prime_rl.parity_router_trace as parity_trace
from scripts.glm52_policy_parity import (
    Profile,
    capture_h200,
    retain_h200_diagnostics,
    read_secret_stdin,
    validate_diagnostic_file,
    validate_diagnostic_source,
    validate_retained_h200_diagnostics,
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
    issue_one_use_bootstrap_token,
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
    _h200_router_forward_context,
    diagnostics_enabled,
    write_session,
    _is_mounted,
    _trainer_host_name,
    _validate_trainer_bundle_inputs,
    _write_trainer_bundle,
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


class CollectiveTensor:
    def __init__(self, value, *, fail_to=False):
        self.value = np.asarray(value)
        self.device = types.SimpleNamespace(type="cuda")
        self.fail_to = fail_to

    @property
    def dtype(self):
        return self.value.dtype

    def to(self, *args, **kwargs):
        del args, kwargs
        if self.fail_to:
            raise RuntimeError("legacy metadata transfer stopped")
        return self

    def view(self, dtype):
        return CollectiveTensor(self.value.view(dtype))

    def reshape(self, *shape):
        return CollectiveTensor(self.value.reshape(*shape))

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def numel(self):
        return self.value.size

    def copy_(self, other):
        self.value[...] = other.value
        return self

    def zero_(self):
        self.value.fill(0)
        return self

    def fill_(self, value):
        self.value.fill(value)
        return self

    def item(self):
        return self.value.item()

    def __getitem__(self, key):
        return CollectiveTensor(self.value[key])

    def __setitem__(self, key, value):
        self.value[key] = value


class FakeBundleCollectives(types.ModuleType):
    def __init__(
        self,
        rank,
        *,
        gathered_metadata=None,
        source_values=None,
        remote_failure=False,
        remote_controls=None,
    ):
        super().__init__("torch.distributed")
        self.rank = rank
        self.gathered_metadata = gathered_metadata
        self.source_values = source_values
        self.remote_failure = remote_failure
        self.remote_controls = list(remote_controls or ())
        self.calls = []
        self.gather_count = 0
        self.all_reduce_count = 0
        self.source_broadcast_count = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return 64

    def all_gather_into_tensor(self, output, local):
        self.gather_count += 1
        label = "gather-metadata" if self.gather_count == 1 else "gather-errors"
        self.calls.append(label)
        if self.gather_count == 1 and self.gathered_metadata is not None:
            output.value[...] = self.gathered_metadata.reshape(-1)
        else:
            output.value[...] = np.tile(local.value, 64)

    def all_reduce(self, value, op):
        del op
        label = (
            "all-reduce-readiness"
            if self.all_reduce_count == 0
            else (
                "all-reduce-slab"
                if self.all_reduce_count == 1
                else "all-reduce-errors"
            )
        )
        self.calls.append(label)
        if self.all_reduce_count < len(self.remote_controls):
            value.value[...] = np.maximum(
                value.value, self.remote_controls[self.all_reduce_count]
            )
        self.all_reduce_count += 1

    def broadcast(self, value, src):
        if src == 56:
            self.calls.append("broadcast-source")
            if self.source_values is not None:
                value.value[...] = self.source_values[self.source_broadcast_count]
            self.source_broadcast_count += 1
            return
        self.calls.append("broadcast-status")
        if self.rank != 0 and self.remote_failure:
            value.value[...] = 1


def fake_collective_torch(
    distributed,
    *,
    fail_source_copy=False,
    fail_slab_allocation=False,
    fail_legacy_allocation=None,
):
    module = types.ModuleType("torch")
    module.distributed = distributed
    module.Tensor = CollectiveTensor
    module.dtype = object
    module.int64 = np.int64
    module.uint8 = np.uint8
    module.float32 = np.float32
    module.int32 = np.int32
    distributed.ReduceOp = types.SimpleNamespace(MAX="max")
    module.device = lambda *_args, **_kwargs: "cuda:0"
    module.cuda = types.SimpleNamespace(current_device=lambda: 0)

    def from_numpy(value):
        array = np.asarray(value)
        if fail_source_copy and array.dtype != np.uint8:
            raise RuntimeError("source copy stopped")
        return CollectiveTensor(
            array.copy(),
            fail_to=(
                fail_legacy_allocation == "metadata_to"
                and array.dtype == np.uint8
                and array.size == 16 * 1024
            ),
        )

    module.from_numpy = from_numpy
    def empty(shape, dtype, device):
        del device
        if fail_slab_allocation and shape == (parity_trace.trainer_bundle_slab_bytes,):
            raise RuntimeError("communication slab stopped")
        if fail_legacy_allocation == "metadata_empty" and shape == (64 * 16 * 1024,):
            raise RuntimeError("legacy metadata gather allocation stopped")
        return CollectiveTensor(np.empty(shape, dtype=dtype))

    module.empty = empty
    def zeros(shape, dtype, device):
        del device
        if fail_legacy_allocation == "shape2" and shape == (2,):
            raise RuntimeError("legacy shape-2 allocation stopped")
        if fail_legacy_allocation == "shape1" and shape == (1,):
            raise RuntimeError("legacy shape-1 allocation stopped")
        return CollectiveTensor(np.zeros(shape, dtype=dtype))

    def full(shape, fill, dtype, device):
        del device
        return CollectiveTensor(np.full(shape, fill, dtype=dtype))

    module.zeros = zeros
    module.full = full
    module.tensor = lambda value, dtype, device: CollectiveTensor(
        np.asarray(value, dtype=dtype)
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
    def _trainer_bundle_values(self, expected_tokens=8):
        token_ids = np.arange(expected_tokens, dtype=np.int64)
        metadata = TraceSession(
            **{
                **session().__dict__,
                "token_ids_sha256": parity_trace.sha256_tokens(token_ids),
                "expected_tokens": expected_tokens,
                "hidden_layers": 78,
                "dense_layers": 3,
            }
        )
        hosts = ["n82", "n76", "n77", "n78", "n80", "n83", "n84", "n85"]
        layer_parts = np.array_split(np.arange(3, 78), 8)
        modules = {
            "mounted": {"location": "mounted", "sha256": "a" * 64},
            "image": {"location": "unmounted", "sha256": "b" * 64},
        }
        gathered = []
        for rank in range(64):
            pp_rank, cp_rank = divmod(rank, 8)
            record = None
            if cp_rank == 0:
                record = {
                    "schema_version": 1,
                    "kind": "glm52_policy_parity_trainer_runtime",
                    "host": hosts[pp_rank],
                    "pp_rank": pp_rank,
                    "cp_rank": 0,
                    "global_rank": rank,
                    "pp_size": 8,
                    "cp_size": 8,
                    "batch_size": 8,
                    "image_digest": "sha256:" + str(pp_rank) * 64,
                    "owned_layers": layer_parts[pp_rank].tolist(),
                    "modules": modules,
                }
            gathered.append(
                {
                    "rank": rank,
                    "session_identity": trace_session_identity(metadata),
                    "provenance": record,
                    "has_compact_source": rank == 56,
                    "error": None,
                }
            )
        selected = np.arange(expected_tokens // 2, expected_tokens, dtype=np.int64)
        arrays = {
            "token_ids": token_ids,
            "positions": np.arange(expected_tokens, dtype=np.int64),
            "selected_indices": selected,
            "advantages": np.ones(selected.size, dtype=np.float32),
        }
        return metadata, gathered, arrays

    def _encoded_bundle_metadata(self, gathered):
        rows = []
        for record in gathered:
            payload = parity_trace._canonical_json_bytes(record)
            encoded = len(payload).to_bytes(4, "little") + payload
            rows.append(
                np.frombuffer(
                    encoded + b"\0" * (16 * 1024 - len(encoded)), dtype=np.uint8
                )
            )
        return np.stack(rows)

    def _bundle_collective_context(self, rank, distributed, torch_module):
        return (
            patch.dict(
                sys.modules,
                {"torch": torch_module, "torch.distributed": distributed},
            ),
            patch.object(parity_trace, "_process_side", "trainer"),
            patch.object(parity_trace, "_process_rank", rank),
            patch.object(parity_trace, "_process_pp_rank", rank // 8),
            patch.object(parity_trace, "_process_cp_rank", rank % 8),
            patch.object(parity_trace, "_process_world_size", 64),
            patch.object(parity_trace, "_process_pp_size", 8),
            patch.object(parity_trace, "_process_cp_size", 8),
        )

    def _bundle_resources(self, torch_module):
        slab = torch_module.empty(
            (parity_trace.trainer_bundle_slab_bytes,),
            dtype=torch_module.uint8,
            device="cuda:0",
        )
        return (
            parity_trace.TrainerBundleCommunication(slab=slab, error=None),
            CollectiveTensor(np.zeros(2, dtype=np.int64)),
        )

    def test_trainer_bundle_validates_pp8_cp8_and_compact_source(self):
        metadata, gathered, arrays = self._trainer_bundle_values()
        with patch.object(parity_trace, "_process_pp_size", 8), patch.object(
            parity_trace, "_process_cp_size", 8
        ):
            records, checked, binding = _validate_trainer_bundle_inputs(
                gathered, metadata, "n82", arrays
            )
        self.assertEqual([record["pp_rank"] for record in records], list(range(8)))
        self.assertEqual(checked["selected_indices"].tolist(), [4, 5, 6, 7])
        self.assertRegex(binding, r"^[0-9a-f]{64}$")
        self.assertLess(sum(value.nbytes for value in checked.values()), 18 * 1024 * 1024)

    def test_trainer_bundle_refuses_missing_rank_and_mismatched_modules(self):
        metadata, gathered, arrays = self._trainer_bundle_values()
        cases = []
        missing = list(gathered)
        missing[24] = {**missing[24], "provenance": None}
        cases.append((missing, "CP0 provenance"))
        mismatched = [dict(value) for value in gathered]
        mismatched[8] = {
            **mismatched[8],
            "provenance": {
                **mismatched[8]["provenance"],
                "modules": {
                    "different": {"location": "mounted", "sha256": "f" * 64}
                },
            },
        }
        cases.append((mismatched, "module hashes differ"))
        with patch.object(parity_trace, "_process_pp_size", 8), patch.object(
            parity_trace, "_process_cp_size", 8
        ):
            for values, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    _validate_trainer_bundle_inputs(values, metadata, "n82", arrays)

    def test_trainer_bundle_writes_stable_marker_last_with_bound_hashes(self):
        metadata, gathered, arrays = self._trainer_bundle_values()
        with patch.object(parity_trace, "_process_pp_size", 8), patch.object(
            parity_trace, "_process_cp_size", 8
        ):
            records, checked, binding = _validate_trainer_bundle_inputs(
                gathered, metadata, "n82", arrays
            )
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_session(root, metadata)
                replacements = []
                real_replace = os.replace

                def replace(source, destination):
                    replacements.append(Path(destination).name)
                    real_replace(source, destination)

                with patch.object(parity_trace.os, "replace", side_effect=replace):
                    _write_trainer_bundle(root, metadata, records, checked, binding)
                self.assertEqual(replacements[-1], "STABLE")
                stable = json.loads((root / "STABLE").read_text())
                union = root / "trainer-provenance-union.json"
                source = root / "trainer-fixture-source.npz"
                manifest = root / "trainer-fixture-source.json"
                self.assertEqual(
                    stable["provenance_union_sha256"], parity_trace._file_sha256(union)
                )
                self.assertEqual(
                    stable["fixture_source_sha256"], parity_trace._file_sha256(source)
                )
                self.assertEqual(
                    stable["fixture_source_manifest_sha256"],
                    parity_trace._file_sha256(manifest),
                )
                self.assertLess(source.stat().st_size, 18 * 1024 * 1024)

    def test_trainer_bundle_does_not_mark_partial_publish_stable(self):
        metadata, gathered, arrays = self._trainer_bundle_values()
        with patch.object(parity_trace, "_process_pp_size", 8), patch.object(
            parity_trace, "_process_cp_size", 8
        ):
            records, checked, binding = _validate_trainer_bundle_inputs(
                gathered, metadata, "n82", arrays
            )
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_session(root, metadata)
                real_replace = os.replace

                def replace(source, destination):
                    if Path(destination).name == "trainer-fixture-source.json":
                        raise OSError("publish stopped")
                    real_replace(source, destination)

                with patch.object(
                    parity_trace.os, "replace", side_effect=replace
                ), self.assertRaisesRegex(OSError, "publish stopped"):
                    _write_trainer_bundle(root, metadata, records, checked, binding)
                self.assertFalse((root / "STABLE").exists())

    def test_trainer_bundle_uses_fixed_cuda_collectives_and_status_broadcast(self):
        source = inspect.getsource(parity_trace.publish_trainer_bundle)
        self.assertIn("all_gather_into_tensor", source)
        self.assertNotIn("gather_object", source)
        self.assertIn("metadata_limit = 16 * 1024", source)
        self.assertIn("error_limit = 1024", source)
        self.assertNotIn("torch.zeros(", source)
        self.assertNotIn("torch.empty(", source)
        self.assertNotIn(".to(device=device)", source)
        self.assertEqual(
            source.count("dist.all_reduce(readiness, op=dist.ReduceOp.MAX)"), 3
        )
        self.assertLess(
            source.index("dist.all_reduce(readiness"),
            source.index("dist.all_gather_into_tensor"),
        )
        self.assertLess(
            source.rindex("dist.all_reduce(readiness"),
            source.index("if not any_error:"),
        )
        self.assertLess(
            source.index("dist.broadcast(status"), source.index("if int(status.item())")
        )
        train_source = Path(
            "source/src/prime_rl/trainer/rl/train.py"
        ).read_text()
        prepare = train_source.index(
            "parity_bundle_communication = prepare_trainer_bundle_communication"
        )
        world_count = train_source.index("dist.all_reduce(_n, op=dist.ReduceOp.MAX)")
        readiness_source = train_source.index("parity_bundle_readiness = _n", world_count)
        barrier = train_source.index("dist.barrier()", train_source.index("if config.enable_token_export"))
        publish = train_source.index("publish_trainer_bundle", barrier)
        stable = train_source.index("token_exporter.mark_stable", publish)
        self.assertLess(prepare, barrier)
        self.assertLess(world_count, readiness_source)
        self.assertLess(readiness_source, barrier)
        self.assertLess(barrier, publish)
        self.assertLess(publish, stable)

    def test_trainer_bundle_is_off_without_parity_trace(self):
        with patch.object(parity_trace, "validated_trace_root", return_value=None):
            parity_trace.publish_trainer_bundle(None, False)

    def test_trainer_bundle_rank_local_failures_keep_collective_order(self):
        metadata, _, _ = self._trainer_bundle_values(32768)
        expected = [
            "all-reduce-readiness",
            "all-reduce-slab",
            "gather-metadata",
            "all-reduce-errors",
            "gather-errors",
            "broadcast-status",
        ]
        canonical_json = parity_trace._canonical_json_bytes

        def fail_first_json():
            calls = 0

            def encode(value):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise TypeError("json stopped")
                return canonical_json(value)

            return encode

        cases = (
            (
                "root",
                patch.object(
                    parity_trace,
                    "validated_trace_root",
                    side_effect=ValueError("root stopped"),
                ),
                patch.object(parity_trace, "read_session", return_value=metadata),
                patch.object(
                    parity_trace, "_canonical_json_bytes", side_effect=canonical_json
                ),
            ),
            (
                "session",
                patch.object(parity_trace, "validated_trace_root", return_value=Path("/trace")),
                patch.object(
                    parity_trace, "read_session", side_effect=ValueError("session stopped")
                ),
                patch.object(
                    parity_trace, "_canonical_json_bytes", side_effect=canonical_json
                ),
            ),
            (
                "json",
                patch.object(parity_trace, "validated_trace_root", return_value=Path("/trace")),
                patch.object(parity_trace, "read_session", return_value=metadata),
                patch.object(
                    parity_trace, "_canonical_json_bytes", side_effect=fail_first_json()
                ),
            ),
        )
        for name, root_patch, session_patch, json_patch in cases:
            with self.subTest(name=name):
                distributed = FakeBundleCollectives(0)
                torch_module = fake_collective_torch(distributed)
                communication, readiness = self._bundle_resources(torch_module)
                with ExitStack() as stack:
                    for context in self._bundle_collective_context(
                        0, distributed, torch_module
                    ):
                        stack.enter_context(context)
                    stack.enter_context(root_patch)
                    stack.enter_context(session_patch)
                    stack.enter_context(json_patch)
                    with self.assertRaisesRegex(RuntimeError, "bundle failed"):
                        parity_trace.publish_trainer_bundle(
                            None, True, communication, readiness
                        )
                self.assertEqual(distributed.calls, expected)

    def test_trainer_bundle_source_failures_keep_collective_order(self):
        metadata, _, arrays = self._trainer_bundle_values(32768)
        expected = [
            "all-reduce-readiness",
            "all-reduce-slab",
            "gather-metadata",
            "all-reduce-errors",
            "gather-errors",
            "broadcast-status",
        ]
        oversized = dict(arrays)
        oversized["selected_indices"] = np.arange(32769, dtype=np.int64)
        oversized["advantages"] = np.ones(32769, dtype=np.float32)
        for name, compact_source, fail_copy in (
            ("copy", arrays, True),
            ("oversize", oversized, False),
        ):
            with self.subTest(name=name):
                distributed = FakeBundleCollectives(56, remote_failure=True)
                torch_module = fake_collective_torch(
                    distributed, fail_source_copy=fail_copy
                )
                communication, readiness = self._bundle_resources(torch_module)
                with ExitStack() as stack:
                    for context in self._bundle_collective_context(
                        56, distributed, torch_module
                    ):
                        stack.enter_context(context)
                    stack.enter_context(
                        patch.object(
                            parity_trace,
                            "validated_trace_root",
                            return_value=Path("/trace"),
                        )
                    )
                    stack.enter_context(
                        patch.object(parity_trace, "read_session", return_value=metadata)
                    )
                    with self.assertRaisesRegex(RuntimeError, "trainer head"):
                        parity_trace.publish_trainer_bundle(
                            compact_source, True, communication, readiness
                        )
                self.assertEqual(distributed.calls, expected)

    def test_trainer_bundle_slab_failure_is_uniformly_reported(self):
        for rank, remote_controls in (
            (0, ()),
            (56, ()),
            (0, ((1, 0),)),
        ):
            name = f"rank-{rank}-remote-{bool(remote_controls)}"
            with self.subTest(name=name):
                distributed = FakeBundleCollectives(
                    rank, remote_controls=remote_controls
                )
                torch_module = fake_collective_torch(
                    distributed, fail_slab_allocation=not remote_controls
                )
                with ExitStack() as stack:
                    for context in self._bundle_collective_context(
                        rank, distributed, torch_module
                    ):
                        stack.enter_context(context)
                    communication = parity_trace.prepare_trainer_bundle_communication(
                        True, "cuda:0"
                    )
                    if remote_controls:
                        communication, _ = self._bundle_resources(torch_module)
                    readiness = CollectiveTensor(np.zeros(2, dtype=np.int64))
                    with self.assertRaisesRegex(
                        RuntimeError, "communication is unavailable"
                    ):
                        parity_trace.publish_trainer_bundle(
                            None, True, communication, readiness
                        )
                self.assertEqual(distributed.calls, ["all-reduce-readiness"])

    def test_trainer_bundle_avoids_legacy_cuda_allocations(self):
        metadata, _, _ = self._trainer_bundle_values(32768)
        expected = [
            "all-reduce-readiness",
            "all-reduce-slab",
            "gather-metadata",
            "all-reduce-errors",
            "gather-errors",
            "broadcast-status",
        ]
        for fault in ("shape2", "shape1", "metadata_to", "metadata_empty"):
            with self.subTest(fault=fault):
                distributed = FakeBundleCollectives(0)
                clean_torch = fake_collective_torch(distributed)
                communication, readiness = self._bundle_resources(clean_torch)
                torch_module = fake_collective_torch(
                    distributed, fail_legacy_allocation=fault
                )
                with ExitStack() as stack:
                    for context in self._bundle_collective_context(
                        0, distributed, torch_module
                    ):
                        stack.enter_context(context)
                    stack.enter_context(
                        patch.object(
                            parity_trace,
                            "validated_trace_root",
                            side_effect=ValueError("root stopped"),
                        )
                    )
                    with self.assertRaisesRegex(RuntimeError, "bundle failed"):
                        parity_trace.publish_trainer_bundle(
                            None, True, communication, readiness
                        )
                self.assertEqual(distributed.calls, expected)

    def test_trainer_bundle_rank_zero_write_failure_reaches_status_broadcast(self):
        metadata, gathered, arrays = self._trainer_bundle_values(32768)
        selected_count = arrays["selected_indices"].size
        selected = np.full(32768, -1, dtype=np.int64)
        selected[:selected_count] = arrays["selected_indices"]
        advantages = np.zeros(32768, dtype=np.float32)
        advantages[:selected_count] = arrays["advantages"]
        distributed = FakeBundleCollectives(
            0,
            gathered_metadata=self._encoded_bundle_metadata(gathered),
            source_values=(
                arrays["token_ids"],
                arrays["positions"],
                selected,
                advantages,
            ),
        )
        torch_module = fake_collective_torch(distributed)
        communication, readiness = self._bundle_resources(torch_module)
        with ExitStack() as stack:
            for context in self._bundle_collective_context(
                0, distributed, torch_module
            ):
                stack.enter_context(context)
            stack.enter_context(
                patch.object(
                    parity_trace, "validated_trace_root", return_value=Path("/trace")
                )
            )
            stack.enter_context(
                patch.object(parity_trace, "read_session", return_value=metadata)
            )
            stack.enter_context(
                patch.object(
                    parity_trace,
                    "_write_trainer_bundle",
                    side_effect=OSError("write stopped"),
                )
            )
            stack.enter_context(
                patch.dict(os.environ, {"PRIME_TRAINER_HEAD": "n82"}, clear=True)
            )
            with self.assertRaisesRegex(RuntimeError, "bundle failed"):
                parity_trace.publish_trainer_bundle(
                    None, True, communication, readiness
                )
        self.assertEqual(
            distributed.calls,
            [
                "all-reduce-readiness",
                "all-reduce-slab",
                "gather-metadata",
                "all-reduce-errors",
                "gather-errors",
                "broadcast-source",
                "broadcast-source",
                "broadcast-source",
                "broadcast-source",
                "broadcast-status",
            ],
        )

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

    def test_h200_diagnostics_are_retained_with_content_validation(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace"
            output = root / "capture"
            trace.mkdir()
            output.mkdir()
            values = np.arange(3 * 6144, dtype=np.float32).reshape(3, 6144)
            filename = "h200-rank-0-pp-0-cp-0-diagnostic-layer0_attention_raw.npz"
            name = "layer0_attention_raw-rank-0-pp-0-cp-0"
            np.savez(
                trace / filename,
                values=values,
                name=np.asarray(name),
                source_dtype=np.asarray("torch.bfloat16"),
            )
            entry = {
                "name": name,
                "file": filename,
                "bytes": (trace / filename).stat().st_size,
                "shape": [3, 6144],
                "source_dtype": "torch.bfloat16",
                "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            }
            route_values = np.arange(3 * 8, dtype=np.int32).reshape(3, 8)
            route_filename = (
                "h200-rank-0-pp-0-cp-0-diagnostic-layer3_h200_route_ids.npz"
            )
            route_name = "layer3_h200_route_ids-rank-0-pp-0-cp-0"
            np.savez(
                trace / route_filename,
                values=route_values,
                name=np.asarray(route_name),
                source_dtype=np.asarray("torch.int32"),
            )
            route_entry = {
                "name": route_name,
                "file": route_filename,
                "bytes": (trace / route_filename).stat().st_size,
                "shape": [3, 8],
                "source_dtype": "torch.int32",
                "sha256": hashlib.sha256(route_values.tobytes()).hexdigest(),
            }
            diagnostics = {
                "schema_version": 1,
                "kind": "glm52_policy_parity_diagnostics",
                "session_identity": "a" * 64,
                "side": "h200",
                "entries": [entry, route_entry],
            }
            (trace / "parity-diagnostics.json").write_text(
                json.dumps(diagnostics, sort_keys=True) + "\n"
            )
            retained = retain_h200_diagnostics(
                {"diagnostics": diagnostics}, output, trace, 64, 100_000_000
            )
            copied = output / retained["entries"][0]["file"]
            self.assertTrue(copied.is_file())
            self.assertEqual(retained["files"], 2)
            self.assertEqual(
                retained["bytes"], entry["bytes"] + route_entry["bytes"]
            )
            with np.load(copied, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored["values"], values)
            manifest = {
                "direct_router_diagnostic": {"diagnostics": diagnostics},
                "retained_diagnostics": retained,
            }
            validate_retained_h200_diagnostics(output, manifest)

            with (output / retained["entries"][1]["file"]).open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                validate_retained_h200_diagnostics(output, manifest)

            diagnostics["entries"][0]["sha256"] = "0" * 64
            (trace / "parity-diagnostics.json").write_text(
                json.dumps(diagnostics, sort_keys=True) + "\n"
            )
            (root / "bad").mkdir()
            with self.assertRaisesRegex(ValueError, "content differs"):
                retain_h200_diagnostics(
                    {"diagnostics": diagnostics}, root / "bad", trace, 64, 100_000_000
                )

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

    def test_api_process_issues_the_exact_token_received_by_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "bootstrap.fifo"
            os.mkfifo(fifo, 0o600)
            os.chmod(fifo, 0o600)
            child = os.fork()
            if child == 0:
                try:
                    issue_one_use_bootstrap_token(fifo, fifo, os.geteuid())
                except BaseException:
                    os._exit(1)
                else:
                    os._exit(0)
            with fifo.open("rb", buffering=0) as stream:
                received = stream.read()
            finished, status = os.waitpid(child, 0)
            self.assertEqual(finished, child)
            self.assertEqual(status, 0)
            self.assertEqual(len(received), 65)
            self.assertTrue(received.endswith(b"\n"))
            self.assertEqual(len(base64.urlsafe_b64decode(received[:-1])), 48)

    def test_bootstrap_issue_rejects_regular_file_and_wrong_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bootstrap.fifo"
            path.write_text("not a fifo\n")
            with self.assertRaisesRegex(ValueError, "type, mode, or owner"):
                issue_one_use_bootstrap_token(path, path, os.geteuid())
            path.unlink()
            os.mkfifo(path, 0o644)
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ValueError, "type, mode, or owner"):
                issue_one_use_bootstrap_token(path, path, os.geteuid())

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
                self.assertTrue(url.startswith("http://control"))
                self.assertEqual(api_key, bootstrap_token)
                events.append("bootstrap_ready")
                return {
                    "status": "ready",
                    "process_id": 22,
                    "boot_nonce": "c" * 32,
                    "token_sha256": parity_script.hashlib.sha256(
                        bootstrap_token.encode("ascii")
                    ).hexdigest(),
                }
            if url.endswith("/v1/models"):
                self.assertTrue(url.startswith("http://h200"))
                events.append("models")
                return {"data": [{"id": "model", "max_model_len": 3}]}
            if url.endswith("/parity/router-trace/reset"):
                self.assertTrue(url.startswith("http://control"))
                self.assertEqual(api_key, bootstrap_token)
                reset_count += 1
                events.extend(("reset", "token_issue"))
                active_session.update(body)
                return {"session": body, "diagnostic_token": "one-use-secret"}
            if url.endswith("/inference/v1/generate"):
                self.assertTrue(url.startswith("http://h200"))
                generate_count += 1
                events.append("graph_capture" if generate_count == 1 else "eager_capture")
                result = {"request_id": f"request-{generate_count}"}
                if generate_count == 2:
                    result["choices"] = [{"routed_experts": encoded_ids}]
                return result
            if url.endswith("/parity/router-trace"):
                self.assertTrue(url.startswith("http://control"))
                events.append("trace")
                return {
                    "session": active_session,
                    "runtime_marker": {"request_identity": "trace-request"},
                    "router_ids": encoded_ids,
                    "router_weights": encoded_weights,
                }
            raise AssertionError(url)

        def replay_diagnostic(*args):
            self.assertEqual(args[2], "http://control")
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
            ) as replay, patch.object(
                parity_script,
                "h200_diagnostic_retention_settings",
                return_value=(root / "trace", 64, 100_000_000),
            ), patch.object(
                parity_script,
                "retain_h200_diagnostics",
                return_value={"files": 1, "bytes": 1, "entries": []},
            ):
                manifest = capture_h200(
                    profile,
                    root / "fixture",
                    output,
                    "http://h200",
                    "http://control",
                    "model",
                    "a" * 40,
                    "b" * 64,
                    "image",
                    None,
                    bootstrap_token,
                    22,
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
        self.assertEqual(manifest["retained_diagnostics"]["files"], 1)
        self.assertEqual(manifest["bootstrap_identity"]["boot_nonce"], "c" * 32)
        self.assertNotIn("one-use-secret", json.dumps(manifest, sort_keys=True))

    def test_bootstrap_identity_waits_only_for_not_ready(self):
        import scripts.glm52_policy_parity as parity_script

        token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
        ready = {
            "status": "ready",
            "process_id": 22,
            "boot_nonce": "d" * 32,
            "token_sha256": parity_script.hashlib.sha256(token.encode("ascii")).hexdigest(),
        }
        with patch.object(
            parity_script,
            "request_json",
            side_effect=[ValueError("H200 request failed with HTTP 503"), ready],
        ) as request, patch.object(parity_script.time, "sleep") as sleep:
            result = parity_script.wait_for_bootstrap_identity(
                "http://h200", token, 22, 10
            )
        self.assertEqual(result, ready)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1)

        with patch.object(
            parity_script,
            "request_json",
            side_effect=ValueError("H200 request failed with HTTP 401"),
        ) as request, patch.object(parity_script.time, "sleep") as sleep, self.assertRaisesRegex(
            ValueError, "identity check failed"
        ):
            parity_script.wait_for_bootstrap_identity(
                "http://h200", token, 22, 10
            )
        request.assert_called_once()
        sleep.assert_not_called()

        with patch.object(
            parity_script,
            "request_json",
            return_value={**ready, "process_id": 23},
        ), self.assertRaisesRegex(ValueError, "differs from the isolated API"):
            parity_script.wait_for_bootstrap_identity(
                "http://h200", token, 22, 10
            )

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

    def test_direct_router_replay_uses_vllm_026_forward_context(self):
        calls = []
        config = object()

        @contextmanager
        def set_forward_context(
            attn_metadata,
            vllm_config,
            num_tokens=None,
            num_tokens_across_dp=None,
            cudagraph_runtime_mode=None,
            batch_descriptor=None,
            ubatch_slices=None,
            slot_mapping=None,
            skip_compiled=False,
            is_padding=None,
        ):
            calls.append(
                (
                    "enter",
                    attn_metadata,
                    vllm_config,
                    num_tokens,
                    skip_compiled,
                )
            )
            yield
            calls.append(("exit",))

        vllm = types.ModuleType("vllm")
        forward_context = types.ModuleType("vllm.forward_context")
        forward_context.set_forward_context = set_forward_context
        runner = types.SimpleNamespace(vllm_config=config)
        with patch.dict(
            sys.modules,
            {"vllm": vllm, "vllm.forward_context": forward_context},
        ):
            with _h200_router_forward_context(runner, 256):
                calls.append(("router",))
        self.assertEqual(
            calls,
            [("enter", None, config, 256, True), ("router",), ("exit",)],
        )

    def test_direct_router_replay_refuses_incompatible_context_api(self):
        def set_forward_context(attn_metadata, vllm_config, num_tokens=None):
            del attn_metadata, vllm_config, num_tokens

        vllm = types.ModuleType("vllm")
        forward_context = types.ModuleType("vllm.forward_context")
        forward_context.set_forward_context = set_forward_context
        with patch.dict(
            sys.modules,
            {"vllm": vllm, "vllm.forward_context": forward_context},
        ), self.assertRaisesRegex(ValueError, "API is incompatible"):
            _h200_router_forward_context(
                types.SimpleNamespace(vllm_config=object()), 256
            )
        with self.assertRaisesRegex(ValueError, "no vLLM configuration"):
            _h200_router_forward_context(object(), 256)

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
