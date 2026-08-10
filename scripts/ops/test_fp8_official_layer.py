"""Two-rank gate for an official GLM-5.2 FP8 decoder layer.

Run in the trainer image with the production FP8 checkpoint mounted at
``/model``. The gate loads layer 0 through the production DCP reader, so it
tests native block-FP8 matrices and the partial-block BF16 projections.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from prime_rl.trainer.fp8_checkpoint import FP8CheckpointStorageReader
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import (
    GlmMoeDsaConfig,
)
from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
)
from prime_rl.trainer.models.layers.checkpointing import (
    set_reentrant_block_checkpointing,
)
from prime_rl.trainer.models.layers.fp8_ada import replace_linear
from prime_rl.trainer.models.layers.rotary_emb import (
    RotaryEmbedding,
    RotaryEmbeddingConfig,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--memory-budget-gib", type=float, default=23.0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.seq_len < 2048:
        raise ValueError("the official layer gate requires at least 2,048 tokens")
    if args.memory_budget_gib <= 0:
        raise ValueError("memory budget must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability(local_rank) != (8, 9):
        raise RuntimeError("the official layer gate requires SM89 GPUs")
    dist.init_process_group("nccl")
    if dist.get_world_size() != 2:
        raise RuntimeError("run this gate with exactly two ranks")

    start = time.perf_counter()

    def progress(message: str) -> None:
        if dist.get_rank() == 0:
            print(
                f"[official-fp8-layer] {time.perf_counter() - start:.1f}s {message}",
                flush=True,
            )

    config = GlmMoeDsaConfig.from_pretrained(args.model)
    config.fp8_source_checkpoint = True
    config.fp8_checkpoint = True
    config.fp8_checkpoint_linears = True
    config.fp8_checkpoint_moe = True
    config.fp8_checkpoint_routed_moe = True
    config.fp8_checkpoint_shared_moe = True
    config.fp8_checkpoint_dequantized = False
    config.use_grouped_mm = True

    class LayerContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [GlmMoeDsaDecoderLayer(config, layer_idx=0)]
            )

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            container = LayerContainer()
    finally:
        torch.set_default_dtype(previous_dtype)
    replace_linear(container)
    container.to_empty(device=torch.device("cuda", local_rank))

    reader = FP8CheckpointStorageReader(
        path=args.model,
        dequantize_all=False,
    )
    dcp_load(container.state_dict(), storage_reader=reader)
    for parameter in container.parameters():
        parameter.requires_grad_(False)
    progress("official layer-0 checkpoint tensors loaded")

    layer = container.model.layers[0]
    set_reentrant_block_checkpointing(layer)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    fully_shard(
        layer,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(),
        reshard_after_forward=True,
    )
    progress("layer sharded with reshard_after_forward")

    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_type = (
        rope_parameters.get("rope_type", "default")
        if isinstance(rope_parameters, dict)
        else "default"
    )
    rotary = RotaryEmbedding(
        RotaryEmbeddingConfig(
            max_position_embeddings=config.max_position_embeddings,
            rope_type=rope_type,
            model_config=config,
        )
    ).cuda(local_rank)

    generator = torch.Generator(device="cuda").manual_seed(20260810)
    template = torch.randn(
        1,
        args.seq_len,
        config.hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    position_ids = torch.arange(
        args.seq_len,
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    position_embeddings = rotary(template, position_ids)
    ks = torch.zeros(args.seq_len, device="cuda", dtype=torch.int32)
    ke = torch.arange(
        1,
        args.seq_len + 1,
        device="cuda",
        dtype=torch.int32,
    )

    def iteration(checkpoint_enabled: bool):
        layer.train(checkpoint_enabled)
        value = template.detach().clone().requires_grad_()
        torch.cuda.reset_peak_memory_stats(local_rank)
        output, _ = layer(
            value,
            position_embeddings=position_embeddings,
            ks=ks,
            ke=ke,
        )
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("official layer forward produced nonfinite values")
        output.float().square().mean().backward()
        torch.cuda.synchronize(local_rank)
        if value.grad is None or not bool(torch.isfinite(value.grad).all()):
            raise RuntimeError("official layer backward produced a nonfinite or missing gradient")
        if not bool(value.grad.abs().max() > 0):
            raise RuntimeError("official layer backward produced an all-zero gradient")
        return (
            output.detach(),
            value.grad.detach(),
            torch.cuda.max_memory_allocated(local_rank),
        )

    eager_output, eager_gradient, eager_peak = iteration(False)
    progress("eager official-layer forward and backward passed")
    checkpoint_output, checkpoint_gradient, checkpoint_peak = iteration(True)
    progress("reentrant official-layer forward and backward passed")

    torch.testing.assert_close(checkpoint_output, eager_output, rtol=0, atol=0)
    gradient_diff = checkpoint_gradient.float() - eager_gradient.float()
    gradient_rel_rmse = float(
        gradient_diff.square().mean().sqrt()
        / eager_gradient.float().square().mean().sqrt().clamp_min(1e-30)
    )
    torch.testing.assert_close(
        checkpoint_gradient,
        eager_gradient,
        rtol=1e-4,
        atol=1e-8,
    )
    local_peak = max(eager_peak, checkpoint_peak)
    peak = torch.tensor(local_peak, device="cuda", dtype=torch.int64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    peak_gib = int(peak.item()) / 1024**3
    if peak_gib > args.memory_budget_gib:
        raise RuntimeError(
            f"official layer peak {peak_gib:.2f} GiB exceeds "
            f"{args.memory_budget_gib:.2f} GiB"
        )
    if dist.get_rank() == 0:
        print(
            f"PASS official GLM layer 0 | seq={args.seq_len} | "
            f"peak={peak_gib:.2f} GiB | "
            f"grad_rel_rmse={gradient_rel_rmse:.6e} | "
            f"grad_absmax={float(checkpoint_gradient.abs().max()):.6e}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
