import sys
import types

import torch
from torch.utils.checkpoint import checkpoint

from prime_rl.trainer.models.layers.checkpointing import (
    ring_attention_checkpoint_context_fn,
)


def test_checkpoint_recompute_restores_each_microbatch_ring_state(monkeypatch) -> None:
    data_params: dict[str, object] = {}
    adapter = types.ModuleType("ring_flash_attn.adapters.hf_adapter")
    adapter.DATA_PARAMS = data_params
    adapters = types.ModuleType("ring_flash_attn.adapters")
    adapters.hf_adapter = adapter
    package = types.ModuleType("ring_flash_attn")
    package.adapters = adapters
    monkeypatch.setitem(sys.modules, "ring_flash_attn", package)
    monkeypatch.setitem(sys.modules, "ring_flash_attn.adapters", adapters)
    monkeypatch.setitem(
        sys.modules, "ring_flash_attn.adapters.hf_adapter", adapter
    )

    class UsesGlobalRingState(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            scale = data_params["scale"]
            assert isinstance(scale, torch.Tensor)
            return value * scale

    module = UsesGlobalRingState()
    value = torch.ones((), requires_grad=True)

    data_params["scale"] = torch.tensor(2.0)
    first = checkpoint(
        module,
        value,
        use_reentrant=False,
        context_fn=ring_attention_checkpoint_context_fn,
    )
    data_params["scale"] = torch.tensor(3.0)
    second = checkpoint(
        module,
        value,
        use_reentrant=False,
        context_fn=ring_attention_checkpoint_context_fn,
    )

    # A later pipeline microbatch has replaced the process-global state before
    # either earlier checkpoint recomputes.
    data_params["scale"] = torch.tensor(11.0)
    (first + second).backward()

    assert value.grad is not None
    assert value.grad.item() == 5.0
    assert data_params["scale"].item() == 11.0
