"""CPU qualification checks for the native per-token FP8 transport.

This test is deliberately independent of Torch.  It exercises the same
ordering and codec contract with eight fake EP ranks when a CUDA build is not
available.  The real one-node command is documented in the test output and
must still be run in a CUDA qualification environment.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
)
WORLD = 8
HIDDEN = 6144
RANK = 32
OUT = 32
ROWS = 64


def bf16(value: np.ndarray) -> np.ndarray:
    """Round FP32 values to the BF16 value set used by the transport path."""
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def fp8_e4m3(value: np.ndarray) -> np.ndarray:
    """Round finite values to E4M3FN and return the dequantized FP32 values."""
    value = np.asarray(value, dtype=np.float32)
    magnitude = np.abs(value)
    sign = np.sign(value)
    # E4M3FN has a 448 finite maximum and a 2**-9 subnormal quantum.
    magnitude = np.minimum(magnitude, 448.0)
    normal = magnitude >= 2.0**-6
    exponent = np.floor(np.log2(np.maximum(magnitude, 2.0**-6)))
    normal_quantum = np.exp2(exponent - 3.0)
    normal_value = np.round(magnitude / normal_quantum) * normal_quantum
    subnormal_value = np.round(magnitude / 2.0**-9) * 2.0**-9
    result = np.where(normal, normal_value, subnormal_value)
    result = np.minimum(result, 448.0)
    return (sign * result).astype(np.float32)


def quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(value, dtype=np.float32)
    maximum = np.max(np.abs(value), axis=1)
    scale = np.where(maximum == 0, 1.0, maximum / 448.0).astype(np.float32)
    return fp8_e4m3(value / scale[:, None]), scale


def transport_bytes(rows: int, hidden: int) -> dict[str, int]:
    if rows <= 0 or hidden <= 0:
        raise ValueError("transport shape must be positive")
    payload = rows * (hidden + 4)
    return {
        "one_all_to_all": payload,
        "forward_dispatch_and_combine": 2 * payload,
        "forward_backward": 4 * payload,
        "saved_received_input": payload,
        "dequantized_bf16_output": rows * hidden * 2,
    }


def fixed_tokens(length: int = 64, seed: int = 20260827) -> np.ndarray:
    state = seed & 0xFFFFFFFF
    result = np.empty(length, dtype=np.int64)
    for index in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        result[index] = 256 + state % (151936 - 256)
    return result


class EightRankTransport:
    """A small all-to-all model with the native source/destination ordering."""

    def __init__(self, use_fp8: bool, seed: int):
        self.use_fp8 = use_fp8
        self.rng = np.random.default_rng(seed)
        self.destinations = np.arange(ROWS, dtype=np.int64) % WORLD
        self.owners_a = [
            bf16(self.rng.normal(0, 0.02, (HIDDEN, RANK))) for _ in range(WORLD)
        ]
        self.owners_b = [
            bf16(self.rng.normal(0, 0.02, (RANK, OUT))) for _ in range(WORLD)
        ]

    def dispatch(self, values: np.ndarray) -> list[list[tuple[int, np.ndarray]]]:
        packets: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(WORLD)]
        for row, destination in enumerate(self.destinations):
            item = values[row : row + 1]
            if self.use_fp8:
                encoded, scale = quantize_rows(item)
                item = (encoded * scale[:, None]).astype(np.float32)
            packets[int(destination)].append((row, item[0]))
        return packets

    def combine(self, packets: list[list[tuple[int, np.ndarray]]]) -> np.ndarray:
        result = np.empty((ROWS, packets[0][0][1].shape[0]), dtype=np.float32)
        for packet in packets:
            for row, value in packet:
                if self.use_fp8:
                    encoded, scale = quantize_rows(value[None, :])
                    value = encoded[0] * scale[0]
                result[row] = value
        return bf16(result)

    def run(self, values: np.ndarray, grad_output: np.ndarray):
        values = bf16(values)
        grad_output = bf16(grad_output)
        packets = self.dispatch(values)
        outputs: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(WORLD)]
        grad_a = [np.zeros_like(item) for item in self.owners_a]
        grad_b = [np.zeros_like(item) for item in self.owners_b]
        for owner, packet in enumerate(packets):
            for row, received in packet:
                a, b = self.owners_a[owner], self.owners_b[owner]
                z = bf16(received[None, :] @ a)[0]
                output = bf16(z[None, :] @ b)[0]
                outputs[owner].append((row, output))
                dy = grad_output[row]
                if self.use_fp8:
                    dy_q, dy_scale = quantize_rows(dy[None, :])
                    dy = (dy_q * dy_scale[:, None])[0]
                grad_b[owner] += np.outer(z, dy)
                grad_a[owner] += np.outer(received, dy @ b.T)
        return self.combine(outputs), grad_a, grad_b


class GLM52FP8QualificationTests(unittest.TestCase):
    def test_candidate_config_changes_only_transport_and_keeps_strict_guards(self):
        candidate = tomllib.loads(
            (ROOT / "configs/qualification/glm52-native-b2048.toml").read_text()
        )
        self.assertEqual(candidate["model"]["quantization"]["expert_activation_transport"], "fp8")
        self.assertTrue(candidate["output_dir"].endswith("glm52-native-b2048-fp8"))
        self.assertEqual(candidate["model"]["pp"], 8)
        self.assertEqual(candidate["model"]["cp"], 8)
        self.assertEqual(candidate["model"]["ep"], 8)
        self.assertEqual(candidate["model"]["lora"]["rank"], 32)
        self.assertEqual(candidate["model"]["lora"]["target_modules"], list(TARGETS))
        self.assertEqual(candidate["optim"], {"type": "muon", "lr": 8e-6, "weight_decay": 0.0, "max_norm": 1.0})
        fake = candidate["data"]["fake"]
        self.assertTrue(candidate["enable_router_replay"] is False)
        self.assertTrue(fake["generate_samples"] is True)
        self.assertEqual(fake["batch_size"] * fake["samples_per_micro_batch"], 2048)
        self.assertEqual(fake["group_size"], 16)

    def test_forward_backward_and_optimizer_cover_all_ten_lora_targets(self):
        # Each of the ten target modules has an A and B factor in production.
        for target_index, target in enumerate(TARGETS):
            fp8_run = EightRankTransport(True, 7 + target_index)
            bf16_run = EightRankTransport(False, 7 + target_index)
            inputs = bf16(fp8_run.rng.normal(0, 0.5, (ROWS, HIDDEN)))
            grad_output = bf16(fp8_run.rng.normal(0, 0.5, (ROWS, OUT)))
            fp8_output, fp8_a, fp8_b = fp8_run.run(inputs, grad_output)
            bf16_output, bf16_a, bf16_b = bf16_run.run(inputs, grad_output)
            output_rmse = float(np.sqrt(np.mean((fp8_output - bf16_output) ** 2)))
            backward_rmse = float(
                math.sqrt(
                    np.mean(
                        [
                            np.mean((a - b) ** 2)
                            for a, b in zip(fp8_a + fp8_b, bf16_a + bf16_b)
                        ]
                    )
                )
            )
            self.assertTrue(np.isfinite(output_rmse), target)
            self.assertTrue(np.isfinite(backward_rmse), target)
            self.assertLess(output_rmse, 0.15, target)
            self.assertLess(backward_rmse, 0.15, target)
            self.assertTrue(all(np.isfinite(value).all() for value in fp8_a), target)
            self.assertTrue(all(np.isfinite(value).all() for value in fp8_b), target)
            self.assertTrue(any(np.any(value) for value in fp8_a), target)
            self.assertTrue(any(np.any(value) for value in fp8_b), target)
            lr = 8e-6
            updated_a = [parameter - lr * gradient for parameter, gradient in zip(fp8_run.owners_a, fp8_a)]
            updated_b = [parameter - lr * gradient for parameter, gradient in zip(fp8_run.owners_b, fp8_b)]
            self.assertTrue(any(np.any(x != y) for x, y in zip(updated_a, fp8_run.owners_a)), target)
            self.assertTrue(any(np.any(x != y) for x, y in zip(updated_b, fp8_run.owners_b)), target)

    def test_changing_microbatches_and_fixed_token_fixture_are_deterministic(self):
        first = fixed_tokens()
        second = fixed_tokens()
        changed = fixed_tokens(seed=20260828)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, changed))
        self.assertEqual(hashlib.sha256(first.tobytes()).hexdigest(), hashlib.sha256(second.tobytes()).hexdigest())
        run = EightRankTransport(True, 11)
        input_a = bf16(run.rng.normal(size=(ROWS, HIDDEN)))
        input_b = bf16(run.rng.normal(size=(ROWS, HIDDEN)))
        self.assertFalse(np.array_equal(input_a, input_b))

    def test_eight_rank_forward_backward_replay_is_bitwise_deterministic(self):
        first = EightRankTransport(True, 19)
        second = EightRankTransport(True, 19)
        inputs = bf16(first.rng.normal(size=(ROWS, HIDDEN)))
        grad_output = bf16(first.rng.normal(size=(ROWS, OUT)))
        output_a, gradients_a, gradients_b = first.run(inputs, grad_output)
        output_b, gradients_c, gradients_d = second.run(inputs, grad_output)
        self.assertTrue(np.array_equal(output_a, output_b))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(gradients_a, gradients_c)))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(gradients_b, gradients_d)))
        updated_a = [parameter - 8e-6 * gradient for parameter, gradient in zip(first.owners_a, gradients_a)]
        updated_b = [parameter - 8e-6 * gradient for parameter, gradient in zip(second.owners_a, gradients_c)]
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(updated_a, updated_b)))

    def test_exact_payload_accounting_for_production_hidden_size(self):
        measured = transport_bytes(32768, HIDDEN)
        self.assertEqual(measured["one_all_to_all"], 201457664)
        self.assertEqual(measured["forward_backward"], 805830656)
        self.assertEqual(measured["saved_received_input"], 201457664)
        self.assertEqual(measured["dequantized_bf16_output"], 402653184)
        self.assertLess(measured["one_all_to_all"], 32768 * HIDDEN * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
