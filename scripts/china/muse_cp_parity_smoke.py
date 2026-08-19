"""CUDA smoke test for Muse Hugging Face ring context parallelism.

This test uses a small random Muse text model so four replicas fit on one
trainer node.  Rank zero computes the ordinary Hugging Face FlashAttention
reference.  All four ranks then compute contiguous ring-CP shards and gather
the output for a direct comparison.
"""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

import prime_rl._compat  # noqa: F401
from prime_rl.utils.cp import gather_for_cp_wo_grad, setup_cp_params
from prime_rl.trainer.models.layers.hf_ring_attn import substitute_hf_flash_attn
from transformers.models.muse_glimmer.configuration_muse_glimmer import (
    MuseGlimmerTextConfig,
)
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerTextModel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", choices=("full", "sliding"), default="full")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    config = MuseGlimmerTextConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        sliding_window=32,
        layer_types=[f"{args.attention}_attention"],
        layer_rope_theta=[0 if args.attention == "full" else 500000.0],
        use_cache=False,
    )
    config._attn_implementation = "flash_attention_2"
    model = MuseGlimmerTextModel(config).to(device=device, dtype=torch.bfloat16)
    model.eval()
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)

    # Keep every local shard longer than the 32-token window.  The upstream HF
    # ring adapter decides whether to pass a window from the local K length.
    length = 256
    input_ids = torch.randint(0, config.vocab_size, (1, length), device=device)
    dist.broadcast(input_ids, src=0)
    sequence_lengths = (80, length - 80)
    position_ids = torch.cat(
        [torch.arange(n, device=device) for n in sequence_lengths]
    ).unsqueeze(0)
    seq_lens = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)

    reference = None
    reference_grads = None
    if rank == 0:
        model.zero_grad(set_to_none=True)
        # vLLM scores each rollout as an independent sequence. A normal HF
        # forward over the concatenated tokens would allow attention across the
        # packed-document boundary and is not a valid reference for Prime-RL's
        # varlen ring call. Score each document separately, then concatenate in
        # the same source order used by the trainer packer.
        reference_parts = []
        offset = 0
        for document_length in sequence_lengths:
            document_ids = input_ids[:, offset : offset + document_length]
            document_positions = position_ids[:, offset : offset + document_length]
            reference_parts.append(
                model(
                    input_ids=document_ids,
                    position_ids=document_positions,
                    use_cache=False,
                ).last_hidden_state
            )
            offset += document_length
        reference_output = torch.cat(reference_parts, dim=1)
        reference = reference_output.detach().float()
        reference_output.float().square().sum().backward()
        reference_grads = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        model.zero_grad(set_to_none=True)
    dist.barrier()

    substitute_hf_flash_attn(dist.group.WORLD, heads_k_stride=1)
    local_ids, local_positions = setup_cp_params(
        input_ids,
        position_ids,
        rank,
        world_size,
        dist.group.WORLD,
        seq_lens=seq_lens,
        cp_style="ring",
    )
    local = model(
        input_ids=local_ids,
        position_ids=local_positions,
        use_cache=False,
    ).last_hidden_state
    actual = gather_for_cp_wo_grad(
        local.detach().float(), world_size, dist.group.WORLD
    )
    local.float().square().sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, group=dist.group.WORLD)

    if rank == 0:
        assert reference is not None
        delta = actual - reference
        print(
            f"MUSE_CP_PARITY attention={args.attention} "
            f"mean_abs={delta.abs().mean().item():.8f} "
            f"p95_abs={delta.abs().quantile(0.95).item():.8f} "
            f"max_abs={delta.abs().max().item():.8f}"
        )
        if delta.abs().mean().item() > 0.02:
            raise RuntimeError("Muse ring-CP output does not match the HF reference")
        assert reference_grads is not None
        grad_deltas = []
        for name, parameter in model.named_parameters():
            if name in reference_grads:
                grad_deltas.append(
                    (parameter.grad.float() - reference_grads[name]).abs().flatten()
                )
        grad_delta = torch.cat(grad_deltas)
        print(
            f"MUSE_CP_BACKWARD attention={args.attention} "
            f"mean_abs={grad_delta.mean().item():.8f} "
            f"p95_abs={grad_delta.quantile(0.95).item():.8f} "
            f"max_abs={grad_delta.max().item():.8f}"
        )
        if grad_delta.mean().item() > 0.02:
            raise RuntimeError("Muse ring-CP gradients do not match the HF reference")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
