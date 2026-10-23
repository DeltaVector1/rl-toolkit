#!/usr/bin/env python3
"""Benchmark the opt-in partitioned dKV prototype against the baseline."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch


def load_callable(path: Path, name: str, attribute: str):
    spec = importlib.util.spec_from_file_location(name, path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, attribute)


def reference_case(q, kv, do, indices, scale):
    value_dim = do.shape[-1]
    sentinel = kv.shape[1] - 1
    q_ref = q.float().requires_grad_()
    kv_ref = kv.float().requires_grad_()
    idx = indices[:, :, 0, :].long()
    valid = idx != sentinel
    batch_ids = torch.arange(q.shape[0], device=q.device)[:, None, None]
    gathered = kv_ref[batch_ids, idx, 0]
    logits = torch.einsum("bshd,bskd->bshk", q_ref, gathered) * scale
    valid_mask = valid[:, :, None, :]
    has_valid = valid.any(dim=-1)[:, :, None, None]
    logits = logits.masked_fill(~valid_mask, -torch.inf).masked_fill(~has_valid, 0)
    probs = torch.softmax(logits, -1) * valid_mask
    out = torch.einsum("bshk,bskd->bshd", probs, gathered[..., :value_dim])
    dq, dkv = torch.autograd.grad((out * do.float()).sum(), (q_ref, kv_ref))
    return out.detach().cpu(), dq.detach().cpu(), dkv.detach().cpu()


def make_case(shape, generator, device, route_case):
    batch, sequence, heads, kv_length, topk, total_dim, value_dim = shape
    q = torch.randn((batch, sequence, heads, total_dim), device=device, dtype=torch.bfloat16, generator=generator)
    kv = torch.randn((batch, kv_length, 1, total_dim), device=device, dtype=torch.bfloat16, generator=generator)
    do = torch.randn((batch, sequence, heads, value_dim), device=device, dtype=torch.bfloat16, generator=generator)
    sentinel = kv_length - 1
    slots = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, 1, topk)
    if route_case == "allvalid":
        indices = slots.expand(batch, sequence, 1, topk).clone()
    elif route_case == "invalidsuffix25":
        indices = slots.expand(batch, sequence, 1, topk).clone()
        indices[..., topk * 3 // 4 :] = sentinel
    elif route_case == "mostlysentinel":
        indices = slots.expand(batch, sequence, 1, topk).clone()
        indices[..., topk // 4 :] = sentinel
    elif route_case == "duplicates":
        indices = torch.zeros((batch, sequence, 1, topk), device=device, dtype=torch.int32)
    elif route_case == "mixedholes":
        indices = (slots % max(1, topk // 8)).expand(batch, sequence, 1, topk).clone()
        indices[..., 2::5] = sentinel
    elif route_case == "allmasked":
        indices = torch.full((batch, sequence, 1, topk), sentinel, device=device, dtype=torch.int32)
    else:
        raise ValueError(f"unknown route case: {route_case}")
    kv[:, -1].zero_()
    return q.contiguous(), kv.contiguous(), do.contiguous(), indices.contiguous()


def errors(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    denom = torch.linalg.vector_norm(expected.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return delta.max().item(), (torch.linalg.vector_norm(delta) / denom).item()


def assert_finite(label, values):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise AssertionError(f"{label} returned non-finite values")


def invoke(fn, q, kv, out, do, indices, lse, scale, partitions, direct_dkv):
    os.environ["prime_sparse_bwd_direct_dkv"] = "1" if direct_dkv else "0"
    kwargs = {"sm_scale": scale}
    if partitions != 1:
        kwargs["dkv_partitions"] = partitions
    return fn(q, kv, out, do, indices, lse, **kwargs)


def run(label, fn, partitions, direct_dkv, q, kv, out, do, indices, lse, scale, warmup, iterations):
    for _ in range(warmup):
        invoke(fn, q, kv, out, do, indices, lse, scale, partitions, direct_dkv)
    torch.cuda.synchronize(q.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(q.device)
    times = []
    result = None
    for _ in range(iterations):
        torch.cuda.synchronize(q.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = invoke(fn, q, kv, out, do, indices, lse, scale, partitions, direct_dkv)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    assert result is not None
    assert_finite(label, result)
    median = sorted(times)[len(times) // 2]
    print(
        f"{label} partitions={partitions} times_s={[round(x, 6) for x in times]} "
        f"median_s={median:.6f} peak_allocated={torch.cuda.max_memory_allocated(q.device)} "
        f"peak_reserved={torch.cuda.max_memory_reserved(q.device)}",
        flush=True,
    )
    saved = tuple(value.detach().cpu() for value in result)
    del result
    torch.cuda.empty_cache()
    return saved


def compare(label, actual, expected, strict):
    values = errors(actual, expected)
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{label} has invalid error {values}")
    if strict:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
    print(f"{label} maxabs_relL2={values}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-module", required=True, type=Path)
    parser.add_argument("--candidate-module", required=True, type=Path)
    parser.add_argument("--forward-module", type=Path)
    parser.add_argument("--sm-scale", type=float, default=0.0625)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--partitions", nargs="+", type=int, choices=(2, 4), default=(2, 4))
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--skip-sentinel", action="store_true")
    parser.add_argument("--direct-dkv", action="store_true", help="enable the opt-in SM89 direct dKV candidate")
    args = parser.parse_args()
    if args.warmup < 3 or args.iterations < 5:
        parser.error("warmup must be at least 3 and iterations must be at least 5")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("an SM89 CUDA device is required")
    baseline_path = args.baseline_module.resolve()
    candidate_path = args.candidate_module.resolve()
    forward_path = (args.forward_module or baseline_path.with_name("sparse_mla_fwd.py")).resolve()
    baseline_fn = load_callable(baseline_path, "sparse_mla_bwd_baseline", "sparse_mla_bwd")
    candidate_fn = load_callable(candidate_path, "sparse_mla_bwd_candidate", "sparse_mla_bwd")
    forward_fn = load_callable(forward_path, "sparse_mla_fwd_reference", "sparse_mla_fwd_interface")
    os.environ["PRIME_SPARSE_BWD_BLOCK_SIZE"] = "16"
    os.environ["PRIME_SPARSE_BWD_BLOCK_HEADS"] = "16"
    os.environ["PRIME_SPARSE_BWD_THREADS"] = "64"
    os.environ["prime_sparse_bwd_skip_sentinel"] = "1" if args.skip_sentinel else "0"
    os.environ["prime_sparse_bwd_dkv_partitions"] = "1"
    print(
        f"baseline={baseline_path} candidate={candidate_path} forward={forward_path} "
        f"routes=6 skip_sentinel={args.skip_sentinel} direct_dkv={args.direct_dkv} "
        f"order={'candidate,baseline' if args.candidate_first else 'baseline,candidate'}",
        flush=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(19)
    device = torch.device("cuda")
    small = (1, 4, 32, 65, 64, 576, 512)
    small_batch2 = (2, 4, 32, 65, 64, 576, 512)
    actual = (1, 4096, 64, 32769, 2048, 576, 512)
    routes = ("invalidsuffix25", "allvalid", "mostlysentinel", "duplicates", "mixedholes", "allmasked")
    for case_name, shape, needs_reference in (
        ("small", small, True),
        ("small_batch2", small_batch2, True),
        ("actual", actual, False),
    ):
        for route_case in routes:
            q, kv, do, indices = make_case(shape, generator, device, route_case)
            with torch.inference_mode():
                out, lse = forward_fn(q, kv, indices, sm_scale=args.sm_scale)
            if not bool(torch.isfinite(out).all()) or bool(torch.isnan(lse).any()):
                raise AssertionError(f"{case_name}_{route_case} forward returned invalid values")
            reference = reference_case(q, kv, do, indices, args.sm_scale) if needs_reference else None
            variants = [("baseline", baseline_fn, 1, False), ("candidate", candidate_fn, 1, args.direct_dkv)]
            variants.extend(
                (f"candidate_p{partitions}", candidate_fn, partitions, args.direct_dkv)
                for partitions in args.partitions
            )
            if args.candidate_first:
                variants[0], variants[1] = variants[1], variants[0]
            results = {}
            for label, fn, partitions, direct_dkv in variants:
                results[label] = run(
                    f"{case_name}_{route_case}_{label}",
                    fn,
                    partitions,
                    direct_dkv,
                    q,
                    kv,
                    out,
                    do,
                    indices,
                    lse,
                    args.sm_scale,
                    args.warmup,
                    args.iterations,
                )
            compare(f"{case_name}_{route_case}_candidate_default_vs_baseline_dq", results["candidate"][0], results["baseline"][0], False)
            compare(f"{case_name}_{route_case}_candidate_default_vs_baseline_dkv", results["candidate"][1], results["baseline"][1], False)
            for partitions in args.partitions:
                compare(f"{case_name}_{route_case}_candidate_p{partitions}_vs_baseline_dq", results[f"candidate_p{partitions}"][0], results["baseline"][0], needs_reference)
                compare(f"{case_name}_{route_case}_candidate_p{partitions}_vs_baseline_dkv", results[f"candidate_p{partitions}"][1], results["baseline"][1], needs_reference)
            if reference is not None:
                compare(f"{case_name}_{route_case}_baseline_dq_vs_oracle", results["baseline"][0], reference[1], True)
                compare(f"{case_name}_{route_case}_baseline_dkv_vs_oracle", results["baseline"][1], reference[2], True)
                compare(f"{case_name}_{route_case}_candidate_output_vs_oracle", out.cpu(), reference[0], True)
            del q, kv, do, indices, out, lse, reference
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
