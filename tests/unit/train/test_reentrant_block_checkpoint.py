import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTING_PATH = (
    REPOSITORY_ROOT
    / "source"
    / "src"
    / "prime_rl"
    / "trainer"
    / "models"
    / "layers"
    / "checkpointing.py"
)


def _load_checkpointing_module():
    spec = importlib.util.spec_from_file_location(
        "_prime_rl_checkpointing_under_test", CHECKPOINTING_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CHECKPOINTING_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checkpointing = _load_checkpointing_module()


class _KeywordBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False)
        self.dropout = nn.Dropout(p=0.25)
        self.forward_calls = 0
        self.parameters_restored = False
        self.require_restore_on_recompute = False

    def forward(self, *, hidden_states, metadata, multiplier=1.0):
        self.forward_calls += 1
        if (
            self.require_restore_on_recompute
            and self.forward_calls == 2
            and not self.parameters_restored
        ):
            raise RuntimeError("recompute started before the outer pre-backward hook")
        value = self.dropout(self.projection(hidden_states))
        return value.mul(multiplier).add(metadata["residual"])


class ReentrantBlockCheckpointTest(unittest.TestCase):
    def test_keyword_and_nested_tensor_gradients_match_eager_with_rng(self):
        torch.manual_seed(11)
        eager = _KeywordBlock()
        recomputed = _KeywordBlock()
        recomputed.load_state_dict(eager.state_dict())
        checkpointing.set_reentrant_block_checkpointing(recomputed)

        eager_hidden = torch.randn(4, 8, requires_grad=True)
        eager_residual = torch.randn(4, 8, requires_grad=True)
        checkpoint_hidden = eager_hidden.detach().clone().requires_grad_()
        checkpoint_residual = eager_residual.detach().clone().requires_grad_()

        torch.manual_seed(19)
        eager_output = eager(
            hidden_states=eager_hidden,
            metadata={"residual": eager_residual},
            multiplier=0.75,
        )
        eager_output.square().sum().backward()

        torch.manual_seed(19)
        checkpoint_output = recomputed(
            hidden_states=checkpoint_hidden,
            metadata={"residual": checkpoint_residual},
            multiplier=0.75,
        )
        checkpoint_output.square().sum().backward()

        torch.testing.assert_close(checkpoint_output, eager_output)
        torch.testing.assert_close(checkpoint_hidden.grad, eager_hidden.grad)
        torch.testing.assert_close(checkpoint_residual.grad, eager_residual.grad)
        torch.testing.assert_close(
            recomputed.projection.weight.grad, eager.projection.weight.grad
        )
        self.assertEqual(eager.forward_calls, 1)
        self.assertEqual(recomputed.forward_calls, 2)

    def test_outer_output_hook_runs_before_reentrant_recompute(self):
        block = _KeywordBlock()
        block.dropout.p = 0.0
        block.require_restore_on_recompute = True
        checkpointing.set_reentrant_block_checkpointing(block)

        hidden = torch.randn(2, 8, requires_grad=True)
        residual = torch.randn(2, 8, requires_grad=True)
        output = block(
            hidden_states=hidden,
            metadata={"residual": residual},
        )

        # FSDP attaches an equivalent hook to the checkpoint boundary output.
        # It must run before CheckpointFunction invokes the block a second time.
        output.register_hook(
            lambda gradient: (
                setattr(block, "parameters_restored", True) or gradient
            )
        )
        output.sum().backward()

        self.assertTrue(block.parameters_restored)
        self.assertEqual(block.forward_calls, 2)

    def test_frozen_input_still_recomputes_trainable_parameters(self):
        torch.manual_seed(31)
        eager = _KeywordBlock()
        recomputed = _KeywordBlock()
        recomputed.load_state_dict(eager.state_dict())
        eager.dropout.p = 0.0
        recomputed.dropout.p = 0.0
        checkpointing.set_reentrant_block_checkpointing(recomputed)

        hidden = torch.randn(4, 8)
        residual = torch.randn(4, 8)
        eager(hidden_states=hidden, metadata={"residual": residual}).sum().backward()
        recomputed(
            hidden_states=hidden,
            metadata={"residual": residual},
        ).sum().backward()

        torch.testing.assert_close(
            recomputed.projection.weight.grad,
            eager.projection.weight.grad,
        )
        self.assertEqual(recomputed.forward_calls, 2)

    def test_reentrant_context_covers_forward_and_recompute_only(self):
        phases = []

        class ContextBlock(nn.Module):
            def forward(self, hidden_states):
                phases.append(checkpointing.in_reentrant_block_checkpoint())
                return hidden_states.square()

        block = ContextBlock()
        checkpointing.set_reentrant_block_checkpointing(block)
        value = torch.randn(3, requires_grad=True)
        block(value).sum().backward()

        self.assertEqual(phases, [True, True])
        self.assertFalse(checkpointing.in_reentrant_block_checkpoint())

    def test_eval_mode_does_not_checkpoint(self):
        block = _KeywordBlock().eval()
        checkpointing.set_reentrant_block_checkpointing(block)
        hidden = torch.randn(2, 8, requires_grad=True)
        residual = torch.randn(2, 8, requires_grad=True)

        block(hidden_states=hidden, metadata={"residual": residual}).sum().backward()

        self.assertEqual(block.forward_calls, 1)

    def test_reentrant_attention_recomputes_attention_but_not_mlp(self):
        phases = []

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(8, 8, bias=False)
                self.calls = 0

            def forward(self, *, hidden_states, metadata):
                self.calls += 1
                phases.append(checkpointing.in_reentrant_block_checkpoint())
                return self.projection(hidden_states) + metadata["bias"]

        class Mlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(8, 8, bias=False)
                self.calls = 0

            def forward(self, value):
                self.calls += 1
                phases.append(checkpointing.in_reentrant_block_checkpoint())
                return self.projection(value).relu()

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.mlp = Mlp()

            def forward(self, hidden_states, bias):
                value = self.self_attn(
                    hidden_states=hidden_states,
                    metadata={"bias": bias},
                )
                return self.mlp(value)

        block = Block()
        checkpointing.set_reentrant_attention_checkpointing(block)
        hidden = torch.randn(4, 8, requires_grad=True)
        bias = torch.randn(4, 8, requires_grad=True)
        block(hidden, bias).sum().backward()

        self.assertEqual(block.self_attn.calls, 2)
        self.assertEqual(block.mlp.calls, 1)
        self.assertEqual(phases, [True, False, True])
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(torch.isfinite(bias.grad).all())

    def test_configuration_and_model_dispatch_expose_reentrant_mode(self):
        config_source = (
            REPOSITORY_ROOT
            / "source"
            / "packages"
            / "prime-rl-configs"
            / "src"
            / "prime_rl"
            / "configs"
            / "trainer.py"
        ).read_text()
        model_source = (
            REPOSITORY_ROOT
            / "source"
            / "src"
            / "prime_rl"
            / "trainer"
            / "model.py"
        ).read_text()

        self.assertIn('"reentrant_attention"', config_source)
        self.assertIn('ac_config.mode == "reentrant_block"', model_source)
        self.assertIn('ac_config.mode == "reentrant_attention"', model_source)
        self.assertIn(
            "set_reentrant_block_checkpointing(transformer_block)", model_source
        )
        self.assertIn(
            "set_reentrant_attention_checkpointing(transformer_block)",
            model_source,
        )


if __name__ == "__main__":
    unittest.main()
