"""Compare Muse Hugging Face logits with and without ring context parallelism.

Run this with four H200 processes. Rank zero first runs normal Hugging Face
FlashAttention. All ranks then run the same weights and tokens through
Prime-RL's ring-CP adapter and compare the gathered logits and sampled-token
log probabilities with the reference. Deterministic synthetic tokens are the
default, so this test does not need or replay an RL rollout batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import msgspec
import torch
import torch.distributed as dist

import prime_rl._compat  # noqa: F401; patch ring-flash-attn compatibility first
from prime_rl.transport.types import TrainingBatch
from prime_rl.trainer.models.layers.hf_ring_attn import substitute_hf_flash_attn
from prime_rl.utils.cp import gather_for_cp_wo_grad, setup_cp_params
from transformers import AutoModelForImageTextToText


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/muse-glimmer/model")
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument(
        "--sequence-lengths",
        help="comma-separated packed document lengths; defaults to one document",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    if args.batch is not None:
        batch = msgspec.msgpack.decode(args.batch.read_bytes(), type=TrainingBatch)
        sample = batch.examples[0]
        length = min(args.tokens, len(sample.token_ids))
        token_ids = sample.token_ids[:length]
    else:
        length = args.tokens
        generator = torch.Generator().manual_seed(20260815)
        token_ids = torch.randint(0, 199000, (length,), generator=generator).tolist()
    length -= length % world_size
    if length <= 0:
        raise ValueError("the selected rollout is shorter than the CP world size")

    input_ids = torch.tensor(token_ids[:length], device=device).unsqueeze(0)
    if args.sequence_lengths:
        sequence_lengths = tuple(int(value) for value in args.sequence_lengths.split(","))
        if sum(sequence_lengths) != length:
            raise ValueError(
                f"sequence lengths sum to {sum(sequence_lengths)}, expected {length}"
            )
    else:
        sequence_lengths = (length,)
    position_ids = torch.cat(
        [torch.arange(n, device=device) for n in sequence_lengths]
    ).unsqueeze(0)
    seq_lens = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.eval()

    reference = None
    if rank == 0:
        with torch.inference_mode():
            reference_parts = []
            offset = 0
            for document_length in sequence_lengths:
                reference_parts.append(
                    model(
                        input_ids=input_ids[:, offset : offset + document_length],
                        position_ids=position_ids[:, offset : offset + document_length],
                        use_cache=False,
                    ).logits.float()
                )
                offset += document_length
            reference = torch.cat(reference_parts, dim=1)
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
    with torch.inference_mode():
        local_logits = model(
            input_ids=local_ids,
            position_ids=local_positions,
            use_cache=False,
        ).logits.float()
    cp_logits = gather_for_cp_wo_grad(local_logits, world_size, dist.group.WORLD)

    if rank == 0:
        assert reference is not None
        delta = cp_logits - reference
        # Compare each document's next-token probabilities without crossing a
        # packed boundary.
        ref_lp_parts = []
        cp_lp_parts = []
        offset = 0
        for document_length in sequence_lengths:
            labels = input_ids[:, offset + 1 : offset + document_length]
            ref_lp_parts.append(
                torch.log_softmax(
                    reference[:, offset : offset + document_length - 1], dim=-1
                ).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            )
            cp_lp_parts.append(
                torch.log_softmax(
                    cp_logits[:, offset : offset + document_length - 1], dim=-1
                ).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            )
            offset += document_length
        ref_lp = torch.cat(ref_lp_parts, dim=1)
        cp_lp = torch.cat(cp_lp_parts, dim=1)
        lp_delta = cp_lp - ref_lp
        print(f"tokens={length} documents={sequence_lengths} cp={world_size}")
        print(
            "logits "
            f"mean_abs={delta.abs().mean().item():.8f} "
            f"max_abs={delta.abs().max().item():.8f}"
        )
        print(
            "token_logprobs "
            f"mean_abs={lp_delta.abs().mean().item():.8f} "
            f"p95_abs={lp_delta.abs().quantile(0.95).item():.8f} "
            f"max_abs={lp_delta.abs().max().item():.8f}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
