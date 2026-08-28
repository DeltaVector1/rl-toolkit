#!/usr/bin/env python3
"""No-start checks for the GLM H200 parity profile.

This check validates the parity invariants and, when a measured free-memory
value is supplied, fails closed before a service start if the profile cannot
accommodate the observed warm-up allocation.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "src"))

from prime_rl.glm52_routing import glm_routing_errors


TARGET_MODULES = {
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
}
CONTEXT_TOKENS = 32768
ONE_OUTPUT = 1
PROFILE_MAX_NUM_BATCHED_TOKENS = CONTEXT_TOKENS + ONE_OUTPUT
PROFILE_MAX_NUM_SEQS = 128
PROFILE_ALLOCATION_MIB = 768.0
POST_ALLOCATION_CUSHION_MIB = 1024.0


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def load_model_config(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid GLM model config: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"GLM model config must be an object: {path}")
    return value


def static_errors(config: dict, max_num_batched_tokens: int, max_num_seqs: int) -> list[str]:
    errors: list[str] = []
    model = config.get("model", {})
    parallel = config.get("parallel", {})
    if model.get("max_model_len") != CONTEXT_TOKENS + ONE_OUTPUT:
        errors.append("max_model_len must be 32769")
    if max_num_batched_tokens != PROFILE_MAX_NUM_BATCHED_TOKENS:
        errors.append("max_num_batched_tokens must be exactly 32769")
    if max_num_seqs != PROFILE_MAX_NUM_SEQS:
        errors.append("max_num_seqs must be exactly 128")
    if parallel.get("tp") != 8 or parallel.get("dp") != 1:
        errors.append("parallel topology must be TP8/DP1")
    if config.get("max_lora_rank") != 32:
        errors.append("max_lora_rank must be 32")
    if set(config.get("lora_target_modules", [])) != TARGET_MODULES:
        errors.append("LoRA target modules do not match the required full set")
    if config.get("enable_lora") is not True:
        errors.append("LoRA must be enabled")
    if config.get("enable_return_routed_experts") is not True:
        errors.append("routed-expert trace must be enabled")
    if model.get("dtype") != "bfloat16":
        errors.append("model compute dtype must remain bfloat16")
    return errors


def budget_report(free_mib: float) -> tuple[bool, float, float]:
    required = PROFILE_ALLOCATION_MIB + POST_ALLOCATION_CUSHION_MIB
    deficit = max(0.0, required - free_mib)
    return deficit == 0.0, required, deficit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=PROFILE_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--max-num-seqs", type=int, default=PROFILE_MAX_NUM_SEQS)
    parser.add_argument(
        "--free-mib",
        type=float,
        help="Measured free memory before profile_run; omit for static checks only.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="GLM checkpoint config.json used by the H200 server.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    errors = static_errors(config, args.max_num_batched_tokens, args.max_num_seqs)
    try:
        errors.extend(glm_routing_errors(load_model_config(args.model_config)))
    except ValueError as error:
        errors.append(str(error))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        "static parity preflight passed: "
        f"context={CONTEXT_TOKENS}+{ONE_OUTPUT}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens}, "
        f"max_num_seqs={args.max_num_seqs}, TP8/DP1, FP8 path, router trace"
    )
    if args.free_mib is None:
        return 0

    safe, required, deficit = budget_report(args.free_mib)
    post = args.free_mib - PROFILE_ALLOCATION_MIB
    print(
        f"measured budget: free={args.free_mib:.2f} MiB, "
        f"profile_request={PROFILE_ALLOCATION_MIB:.2f} MiB, "
        f"post-request={post:.2f} MiB, "
        f"required_free={required:.2f} MiB, "
        f"required_post-request_cushion={POST_ALLOCATION_CUSHION_MIB:.2f} MiB"
    )
    if not safe:
        print(
            f"ERROR: measured free memory is short by {deficit:.2f} MiB; "
            "refuse startup preflight",
            file=sys.stderr,
        )
        return 2
    print("measured startup budget preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
