"""Rescore one token sequence with the unmodified Hugging Face Muse forward."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from transformers import AutoModelForImageTextToText
from transformers.utils import logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/muse-glimmer/model")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens is not None:
        token_ids = [(index * 7919 + 17) % 199000 for index in range(args.tokens)]
    else:
        token_ids = json.load(sys.stdin)
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("stdin must be a nonempty JSON token-id list")

    logging.set_verbosity_error()
    torch.set_grad_enabled(False)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        # The vLLM image uses its own attention kernels and does not include
        # the flash-attn Python wheel.  SDPA is exact enough for this short
        # diagnostic sequence and still exercises Muse's stock HF forward.
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    input_ids = torch.tensor(token_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    logits = model(input_ids=input_ids, use_cache=False).logits.float()
    labels = input_ids[:, 1:]
    logprobs = torch.log_softmax(logits[:, :-1], dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(0).squeeze(-1)
    values = logprobs.cpu().tolist()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "input_ids": token_ids,
                    # The trainer aligns each next-token probability with its
                    # sampled token after shifting right.
                    "inference_logprobs": [
                        math.log(1.0 / model.config.text_config.vocab_size),
                        *values,
                    ],
                    "sequence_lengths": [len(token_ids)],
                }
            )
        )
        print(f"MUSE_HF_REFERENCE={args.output} tokens={len(token_ids)}")
    else:
        print("MUSE_HF_LOGPROBS=" + json.dumps(values))


if __name__ == "__main__":
    main()
