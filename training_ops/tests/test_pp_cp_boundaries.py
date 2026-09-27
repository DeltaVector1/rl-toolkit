"""CPU checks for packed attention state during PP and delayed recompute."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

root = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PipelineBoundariesTest(unittest.TestCase):
    def setUp(self):
        ring = ModuleType("ring_flash_attn")
        ring.update_ring_flash_attn_params = lambda *args: None
        replacements = {
            "ring_flash_attn": ring,
            "prime_rl._compat": ModuleType("prime_rl._compat"),
        }
        names = [
            *replacements,
            "prime_rl.utils.cp",
            "prime_rl.trainer.models.layers.ulysses_attn",
        ]
        previous = {name: sys.modules.get(name) for name in names}

        def restore():
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

        self.addCleanup(restore)
        sys.modules.update(replacements)
        self.cp = load("prime_rl.utils.cp", "prime-rl/src/prime_rl/utils/cp.py")
        self.checkpointing = load(
            "boundary_checkpointing",
            "prime-rl/src/prime_rl/trainer/models/layers/checkpointing.py",
        )
        self.pp = load(
            "boundary_pp_runner",
            "prime-rl/src/prime_rl/trainer/distributed/pp_runner.py",
        )
        import prime_rl

        ulysses = load(
            "prime_rl.trainer.models.layers.ulysses_attn",
            Path(next(iter(prime_rl.__path__)))
            / "trainer/models/layers/ulysses_attn.py",
        )
        self.params = ulysses.ULYSSES_PARAMS
        self.cp.setup_cp_attention_params_for_length(
            8, None, seq_lens=torch.tensor([8]), cp_style="ulysses"
        )

    def boundaries(self):
        return self.params["cu_seqlens"].tolist()

    def test_alternating_stages_and_delayed_checkpoint_recompute(self):
        for reentrant in (True, False):
            seen = []
            owner = self

            class Attention(nn.Module):
                def forward(self, x):
                    seen.append(owner.boundaries())
                    factor = float(len(owner.boundaries()))
                    return x.square() * factor

            class Decoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attention = Attention()

                def forward(self, inputs_embeds=None, **kwargs):
                    return SimpleNamespace(
                        last_hidden_state=self.attention(inputs_embeds)
                    )

            decoder = Decoder()
            self.checkpointing.checkpoint_method(
                decoder.attention, "forward", use_reentrant=reentrant
            )
            model = nn.Module()
            model.model = decoder
            stage = self.pp.PipelineStageModule(model, is_first=False, is_last=False)
            layouts = [[3, 5], [8], [2, 2, 4]]
            inputs, outputs = [], []
            for layout in layouts:
                side = dict(
                    position_ids=torch.arange(4)[None],
                    seq_lens=torch.tensor(layout),
                    sequence_lengths=layout,
                    seq_lens_are_pre_shard=True,
                )
                stage._next_side = lambda side=side: side
                x = torch.tensor([2.0], requires_grad=True)
                inputs.append(x)
                outputs.append(stage._forward_inner(x))
                self.assertEqual(owner.boundaries(), [0, 8])
            self.assertEqual(seen, [[0, 3, 8], [0, 8], [0, 2, 4, 8]])
            # Backward for the first packed record occurs after two newer forwards.
            for index in (0, 2, 1):
                outputs[index].sum().backward()
                self.assertEqual(
                    inputs[index].grad.item(), 4 * (len(layouts[index]) + 1)
                )
                self.assertEqual(owner.boundaries(), [0, 8])
            self.assertEqual(seen[3:], [[0, 3, 8], [0, 2, 4, 8], [0, 8]])

    def test_missing_or_invalid_state_fails_closed(self):
        for layout in ([], [0, 8], [3, 4]):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                self.cp.cp_attention_state_for_lengths(8, torch.tensor(layout))
        self.cp._cp_attention_state = None
        with self.assertRaises(RuntimeError):
            self.cp.cp_attention_state_for_lengths(8, torch.tensor([8]))

    def test_nested_context_and_exception_restore(self):
        packed = self.cp.cp_attention_state_for_lengths(8, torch.tensor([3, 5]))
        other = self.cp.cp_attention_state_for_lengths(8, torch.tensor([2, 6]))
        with self.cp.use_cp_attention_state(packed):
            self.assertEqual(self.boundaries(), [0, 3, 8])
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with self.cp.use_cp_attention_state(other):
                    self.assertEqual(self.boundaries(), [0, 2, 8])
                    raise RuntimeError("probe")
            self.assertEqual(self.boundaries(), [0, 3, 8])
        self.assertEqual(self.boundaries(), [0, 8])


if __name__ == "__main__":
    unittest.main()
