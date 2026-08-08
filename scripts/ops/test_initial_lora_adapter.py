#!/usr/bin/env python3
"""Smoke test for policy-only LoRA restart initialization."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from prime_rl.trainer.lora import load_initial_lora_adapter
from prime_rl.trainer.models.layers.lora import MultiLoRALinear


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = MultiLoRALinear(
            nn.Linear(16, 8, bias=False, dtype=torch.bfloat16),
            rank=8,
            n_adapters=1,
            alpha=16,
        )


class InitialAdapterTest(unittest.TestCase):
    def test_loads_exact_exported_weights(self):
        model = _Model()
        expected_a = torch.full_like(model.proj.lora_A[0], 0.25)
        expected_b = torch.full_like(model.proj.lora_B[0], -0.5)
        with tempfile.TemporaryDirectory() as td:
            save_file(
                {
                    "proj.lora_A.weight": expected_a,
                    "proj.lora_B.weight": expected_b,
                },
                str(Path(td) / "adapter_model.safetensors"),
            )
            load_initial_lora_adapter(model, td)
        torch.testing.assert_close(model.proj.lora_A[0], expected_a)
        torch.testing.assert_close(model.proj.lora_B[0], expected_b)

    def test_rejects_partial_adapter(self):
        model = _Model()
        with tempfile.TemporaryDirectory() as td:
            save_file(
                {"proj.lora_A.weight": torch.zeros_like(model.proj.lora_A[0])},
                str(Path(td) / "adapter_model.safetensors"),
            )
            with self.assertRaisesRegex(ValueError, "missing="):
                load_initial_lora_adapter(model, td)


if __name__ == "__main__":
    unittest.main()
