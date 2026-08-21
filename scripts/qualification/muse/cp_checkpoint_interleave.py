"""Reproduce a 1F1B checkpoint recompute after CP boundaries change.

The two forwards use different packed-document counts. Backward recomputes the
first forward after the second forward replaced ring_flash_attn's global state.
The test fails with CheckpointError unless every checkpoint invocation restores
its own ring-attention parameters.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

import prime_rl._compat  # noqa: F401
from prime_rl.trainer.models.layers.checkpointing import (
    ring_attention_checkpoint_context_fn,
)
from prime_rl.trainer.models.layers.hf_ring_attn import substitute_hf_flash_attn
from prime_rl.utils.cp import setup_cp_params
from transformers.models.muse_glimmer.configuration_muse_glimmer import (
    MuseGlimmerTextConfig,
)
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerTextModel,
)


def _inputs(
    config: MuseGlimmerTextConfig,
    lengths: tuple[int, ...],
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = sum(lengths)
    input_ids = torch.randint(
        0, config.vocab_size, (1, total), device=torch.device("cuda", rank)
    )
    dist.broadcast(input_ids, src=0)
    position_ids = torch.cat(
        [torch.arange(length, device=input_ids.device) for length in lengths]
    ).unsqueeze(0)
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device=input_ids.device)
    return setup_cp_params(
        input_ids,
        position_ids,
        rank,
        world_size,
        dist.group.WORLD,
        seq_lens=seq_lens,
        cp_style="ring",
    )


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"this smoke test requires CP4, got {world_size}")
    torch.cuda.set_device(rank)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)

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
        layer_types=["sliding_attention"],
        layer_rope_theta=[500000.0],
        use_cache=False,
    )
    config._attn_implementation = "flash_attention_2"
    model = MuseGlimmerTextModel(config).to(
        device=torch.device("cuda", rank), dtype=torch.bfloat16
    )
    model.train()
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)

    substitute_hf_flash_attn(dist.group.WORLD, heads_k_stride=1)
    layer = model.layers[0]
    model.layers[0] = checkpoint_wrapper(
        layer,
        preserve_rng_state=False,
        context_fn=ring_attention_checkpoint_context_fn,
    )

    ids_a, positions_a = _inputs(config, (80, 176), rank, world_size)
    output_a = model(
        input_ids=ids_a, position_ids=positions_a, use_cache=False
    ).last_hidden_state

    # The second microbatch has a different number of documents and therefore
    # replaces the saved cu_seqlens tensors with different-shaped tensors.
    ids_b, positions_b = _inputs(config, (48, 80, 128), rank, world_size)
    output_b = model(
        input_ids=ids_b, position_ids=positions_b, use_cache=False
    ).last_hidden_state

    (output_a.float().square().mean() + output_b.float().square().mean()).backward()
    finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    flag = torch.tensor(int(finite), device=output_a.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise RuntimeError("nonfinite gradient in checkpoint interleave smoke test")
    if rank == 0:
        print("MUSE_CP_CHECKPOINT_INTERLEAVE PASS")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
