#!/usr/bin/env python3
"""Validate the isolated GLM-5.2 H200 MTP profile before a start."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "src"))

from prime_rl.glm52_routing import glm_routing_errors


PROFILE_NAME = "glm52-mtp-h200"
CONTEXT_TOKENS = 32768
SUPPORTED_BLOCK_SIZES = frozenset({64})
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
EXPECTED_SPECULATIVE = {
    "method": "mtp",
    "num_speculative_tokens": 1,
    "draft_tensor_parallel_size": 8,
    "max_model_len": CONTEXT_TOKENS,
    "attention_backend": "FLASH_ATTN_MLA_SPARSE",
    "kv_cache_dtype": "bfloat16",
    "enforce_eager": True,
}


def load_config(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"invalid MTP profile: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"MTP profile must be a TOML table: {path}")
    return value


def load_model_config(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            value = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid GLM model config: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"GLM model config must be an object: {path}")
    return value


def static_errors(config: dict) -> list[str]:
    errors: list[str] = []
    model = config.get("model", {})
    parallel = config.get("parallel", {})
    extra = config.get("vllm_extra", {})
    speculative = extra.get("speculative_config", {})
    attention = extra.get("attention_config", {})

    if config.get("enable_lora") is not True:
        errors.append("LoRA must be enabled")
    if config.get("max_lora_rank") != 32:
        errors.append("max_lora_rank must be 32")
    if set(config.get("lora_target_modules", [])) != TARGET_MODULES:
        errors.append("LoRA target modules do not match the required full set")
    if config.get("enable_return_routed_experts") is not True:
        errors.append("return_routed_experts must be enabled")
    if config.get("enable_prefix_caching") is not True:
        errors.append("prefix caching must be enabled")
    if config.get("enable_expert_parallel") is not False:
        errors.append("expert parallelism must remain disabled for this qualification")
    if model.get("name") != "/models/glm52-fp8":
        errors.append("model path must be /models/glm52-fp8")
    if model.get("dtype") != "bfloat16":
        errors.append("model compute dtype must remain bfloat16")
    if model.get("max_model_len") != CONTEXT_TOKENS:
        errors.append("model max_model_len must be 32768")
    if "enforce_eager" in model:
        errors.append("target model must not override CUDA graphs with enforce_eager")
    if parallel != {"tp": 8, "dp": 1}:
        errors.append("parallel topology must be TP8/DP1")
    if extra.get("kv_cache_dtype") != "bfloat16":
        errors.append("target KV cache dtype must be bfloat16")
    target_block_size = extra.get("block_size")
    draft_block_size = speculative.get("block_size", target_block_size)
    if target_block_size not in SUPPORTED_BLOCK_SIZES:
        errors.append("target block_size must be the supported value 64")
    if not isinstance(target_block_size, int) or target_block_size <= 0 or CONTEXT_TOKENS % target_block_size:
        errors.append("target block_size must divide the 32768-token context")
    if draft_block_size not in SUPPORTED_BLOCK_SIZES:
        errors.append("draft block_size must be the supported value 64")
    if not isinstance(draft_block_size, int) or draft_block_size <= 0 or CONTEXT_TOKENS % draft_block_size:
        errors.append("draft block_size must divide the 32768-token context")
    if target_block_size != draft_block_size:
        errors.append("target and draft block_size must be identical")
    if extra.get("max_num_seqs") != 128:
        errors.append("max_num_seqs must be 128")
    if extra.get("max_num_batched_tokens") != CONTEXT_TOKENS:
        errors.append("max_num_batched_tokens must be 32768")
    if attention != {
        "backend": "FLASH_ATTN_MLA_SPARSE",
        "mla_prefill_backend": "FLASH_ATTN",
    }:
        errors.append("target attention backend is not the bounded sparse FlashAttention path")
    if speculative != EXPECTED_SPECULATIVE:
        errors.append("MTP speculative_config differs from the TP8/MTP1 contract")
    if extra.get("compilation_config") != {"max_cudagraph_capture_size": 256}:
        errors.append("CUDA graph capture size must be 256")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    errors = static_errors(config)
    try:
        errors.extend(glm_routing_errors(load_model_config(args.model_config)))
    except ValueError as error:
        errors.append(str(error))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(
        "static MTP preflight passed: "
        "TP8/DP1, context=32768, MTP1, draft TP8, "
        "dynamic LoRA rank=32, target-only route trace"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
