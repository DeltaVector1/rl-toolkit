#!/usr/bin/env python3
"""Compare original and sentinel-skip SM89 sparse MLA backward kernels."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "prime-rl" / "src"))


def load_callable(path: Path, name: str, attribute: str):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, path)
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
    do_ref = do.float()
    idx = indices[:, :, 0, :].long()
    valid = idx != sentinel
    gathered = kv_ref[0, idx[0], 0].unsqueeze(0)
    logits = torch.einsum("bshd,bskd->bshk", q_ref, gathered) * scale
    valid_mask = valid[:, :, None, :]
    has_valid = valid.any(dim=-1)[:, :, None, None]
    masked_logits = logits.masked_fill(~valid_mask, -torch.inf)
    # Keep the all-sentinel reference finite before zeroing its probabilities.
    masked_logits = masked_logits.masked_fill(~has_valid, 0)
    probs = torch.softmax(masked_logits, -1) * valid_mask
    out = torch.einsum("bshk,bskd->bshd", probs, gathered[..., :value_dim])
    dq, dkv = torch.autograd.grad((out * do_ref).sum(), (q_ref, kv_ref))
    return out.detach().cpu(), dq.detach().cpu(), dkv.detach().cpu()


def make_case(shape, generator, device, route_case):
    batch, sequence, heads, kv_length, topk, total_dim, value_dim = shape
    q = torch.randn(
        (batch, sequence, heads, total_dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    kv = torch.randn(
        (batch, kv_length, 1, total_dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    do = torch.randn(
        (batch, sequence, heads, value_dim),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    sentinel = kv_length - 1
    unique_routes = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, 1, topk)
    if route_case == "allvalid":
        indices = unique_routes.expand(batch, sequence, 1, topk).clone()
    elif route_case == "invalidsuffix25":
        indices = unique_routes.expand(batch, sequence, 1, topk).clone()
        indices[..., topk * 3 // 4 :] = sentinel
    elif route_case == "mostlysentinel":
        indices = unique_routes.expand(batch, sequence, 1, topk).clone()
        indices[..., topk // 4 :] = sentinel
    elif route_case == "duplicates":
        indices = torch.zeros((batch, sequence, 1, topk), device=device, dtype=torch.int32)
    elif route_case == "mixedholes":
        slots = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, 1, topk)
        indices = (slots % max(1, topk // 8)).expand(batch, sequence, 1, topk).clone()
        indices[..., 2::5] = sentinel
    elif route_case == "allmasked":
        indices = torch.full((batch, sequence, 1, topk), sentinel, device=device, dtype=torch.int32)
    else:
        raise ValueError(f"unknown route case: {route_case}")
    assert indices.dtype == torch.int32
    kv[:, -1].zero_()
    return q.contiguous(), kv.contiguous(), do.contiguous(), indices.contiguous()


def error(a, b):
    delta = (a.float() - b.float()).abs()
    denom = torch.linalg.vector_norm(b.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return delta.max().item(), (torch.linalg.vector_norm(delta) / denom).item()


def checked_error(label, actual, expected, assert_close=False):
    values = error(actual, expected)
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"{label} has non-finite reference error: {values}")
    if assert_close:
        try:
            torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.02)
        except AssertionError as exc:
            raise AssertionError(f"{label} exceeds rtol=0.02 atol=0.02: {values}") from exc
    return values


def run(
    label,
    fn,
    block_heads,
    threads,
    q,
    kv,
    out,
    do,
    indices,
    lse,
    scale,
    warmup,
    iterations,
    direct=False,
    skip_sentinel=False,
):
    os.environ["PRIME_SPARSE_BWD_BLOCK_SIZE"] = "16"
    os.environ["PRIME_SPARSE_BWD_BLOCK_HEADS"] = str(block_heads)
    os.environ["PRIME_SPARSE_BWD_THREADS"] = str(threads)
    os.environ["prime_sparse_bwd_direct_dkv"] = "1" if direct else "0"
    os.environ["prime_sparse_bwd_skip_sentinel"] = "1" if skip_sentinel else "0"
    os.environ["prime_sparse_bwd_dkv_partitions"] = "1"
    for _ in range(warmup):
        fn(q, kv, out, do, indices, lse, sm_scale=scale)
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
        result = fn(q, kv, out, do, indices, lse, sm_scale=scale)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)
    assert result is not None
    dq, dkv = result
    if not bool(torch.isfinite(dq).all()) or not bool(torch.isfinite(dkv).all()):
        raise AssertionError(f"{label} returned non-finite gradients")
    median = sorted(times)[len(times) // 2]
    print(
        f"{label} block_size=16 block_H={block_heads} threads={threads} "
        f"times_s={[round(x, 6) for x in times]} median_s={median:.6f} "
        f"peak_allocated={torch.cuda.max_memory_allocated(q.device)} "
        f"peak_reserved={torch.cuda.max_memory_reserved(q.device)} "
        f"finite_dq={bool(torch.isfinite(dq).all())} "
        f"finite_dkv={bool(torch.isfinite(dkv).all())}",
        flush=True,
    )
    saved = dq.detach().cpu(), dkv.detach().cpu()
    del dq, dkv
    torch.cuda.empty_cache()
    return saved


def try_candidate(label, *args):
    try:
        return run(label, *args)
    except Exception as exc:
        detail = str(exc).splitlines()[0][:240] or "no detail"
        print(f"{label} ERROR {type(exc).__name__}: {detail}", flush=True)
        raise


def run_baseline(label, fn, q, kv, out, do, indices, lse, scale, warmup, iterations):
    return run(label, fn, 16, 64, q, kv, out, do, indices, lse, scale, warmup, iterations)


def run_candidate(label, fn, q, kv, out, do, indices, lse, scale, warmup, iterations, direct=False):
    return try_candidate(label, fn, 16, 64, q, kv, out, do, indices, lse, scale, warmup, iterations, direct, True)


def print_small_comparisons(route_case, results, output, reference):
    original64_a, original64_b, candidate16 = results
    ref_out, ref_dq, ref_dkv = reference
    values = [f"small_{route_case}_reference output_maxabs_relL2={checked_error('output', output, ref_out, True)}"]
    if original64_a is not None:
        values.append(f"original16_threads64_dq={checked_error('baseline dQ', original64_a[0], ref_dq, True)} original16_threads64_dkv={checked_error('baseline dKV', original64_a[1], ref_dkv, True)}")
    if original64_b is not None:
        values.append(f"original16_threads64_repeat_dq={checked_error('baseline repeat dQ', original64_b[0], ref_dq, True)} original16_threads64_repeat_dkv={checked_error('baseline repeat dKV', original64_b[1], ref_dkv, True)}")
    if candidate16 is not None:
        values.append(f"candidate16_skip_sentinel_dq={checked_error('candidate dQ', candidate16[0], ref_dq, True)} candidate16_skip_sentinel_dkv={checked_error('candidate dKV', candidate16[1], ref_dkv, True)}")
    print(" ".join(values), flush=True)
    if original64_a is not None and original64_b is not None:
        print(
            f"small_{route_case}_original_threads64_repeat dq_maxabs_relL2={checked_error('baseline repeat dQ', original64_b[0], original64_a[0], True)} "
            f"dkv_maxabs_relL2={checked_error('baseline repeat dKV', original64_b[1], original64_a[1], True)}",
            flush=True,
        )
    if candidate16 is not None and original64_a is not None:
        print(
            f"small_{route_case}_candidate16_skip_sentinel_vs_original16_threads64 dq_maxabs_relL2={checked_error('candidate dQ', candidate16[0], original64_a[0], True)} "
            f"dkv_maxabs_relL2={checked_error('candidate dKV', candidate16[1], original64_a[1], True)}",
            flush=True,
        )


def compare_actual(route_case, original64_a, original64_b, candidate16):
    if original64_a is not None and original64_b is not None:
        print(
            f"actual_{route_case}_original_threads64_repeat dq_maxabs_relL2={checked_error('baseline repeat dQ', original64_b[0], original64_a[0])} "
            f"dkv_maxabs_relL2={checked_error('baseline repeat dKV', original64_b[1], original64_a[1])}",
            flush=True,
        )
    if candidate16 is not None and original64_a is not None:
        print(
            f"actual_{route_case}_candidate16_skip_sentinel_vs_original16_threads64 dq_maxabs_relL2={checked_error('candidate dQ', candidate16[0], original64_a[0])} "
            f"dkv_maxabs_relL2={checked_error('candidate dKV', candidate16[1], original64_a[1])}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-module", required=True, type=Path)
    parser.add_argument("--candidate-module", required=True, type=Path)
    parser.add_argument("--forward-module", type=Path)
    parser.add_argument("--sm-scale", type=float, default=0.0625)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--candidate-first", action="store_true", help="run the candidate before the baseline variants")
    parser.add_argument("--direct-dkv", action="store_true", help="enable the opt-in SM89 direct dKV candidate")
    args = parser.parse_args()
    baseline_path = args.baseline_module.resolve()
    candidate_path = args.candidate_module.resolve()
    forward_path = (args.forward_module or baseline_path.with_name("sparse_mla_fwd.py")).resolve()
    for label, path in (("baseline", baseline_path), ("candidate", candidate_path), ("forward", forward_path)):
        if not path.is_file():
            parser.error(f"{label} module does not exist: {path}")
    if baseline_path == candidate_path:
        parser.error("baseline and candidate modules must be different files")
    if args.warmup < 3 or args.iterations < 5:
        parser.error("warmup must be at least 3 and iterations must be at least 5")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (8, 9):
        raise SystemExit(f"expected SM89, got {torch.cuda.get_device_capability(device)}")
    baseline_fn = load_callable(baseline_path, "sparse_mla_bwd_baseline", "sparse_mla_bwd")
    candidate_fn = load_callable(candidate_path, "sparse_mla_bwd_candidate", "sparse_mla_bwd")
    forward_fn = load_callable(forward_path, "sparse_mla_fwd_baseline", "sparse_mla_fwd_interface")
    print(
        f"baseline_module={baseline_path} candidate_module={candidate_path} "
        f"forward_module={forward_path} block_size=16 baseline_block_H=16 baseline_threads=64 "
        f"candidate_block_H=16 candidate_threads=64 skip_sentinel=true direct_dkv={args.direct_dkv} sm_scale={args.sm_scale} "
        f"variant_order={'candidate,baseline,baseline' if args.candidate_first else 'baseline,baseline,candidate'}",
        flush=True,
    )
    generator = torch.Generator(device=device).manual_seed(19)
    small = (1, 4, 32, 65, 64, 576, 512)
    actual = (1, 4096, 64, 32769, 2048, 576, 512)
    for case_name, shape, needs_reference in (("small", small, True), ("actual", actual, False)):
        for route_case in ("invalidsuffix25", "allvalid", "mostlysentinel", "duplicates", "mixedholes", "allmasked"):
            q, kv, do, indices = make_case(shape, generator, device, route_case)
            with torch.inference_mode():
                out, lse = forward_fn(q, kv, indices, sm_scale=args.sm_scale)
            if not bool(torch.isfinite(out).all()) or bool(torch.isnan(lse).any()):
                raise AssertionError(f"{case_name}_{route_case} forward returned invalid values")
            if needs_reference:
                reference = reference_case(q, kv, do, indices, args.sm_scale)
                if not all(bool(torch.isfinite(value).all()) for value in reference):
                    raise AssertionError(f"{case_name}_{route_case} reference returned non-finite values")
            else:
                reference = None
            print(
                f"shape case={case_name} routes={route_case} q={tuple(q.shape)} kv={tuple(kv.shape)} "
                f"do={tuple(do.shape)} indices={tuple(indices.shape)} sm_scale={args.sm_scale}",
                flush=True,
            )
            if args.candidate_first:
                candidate16 = run_candidate(
                    f"{case_name} {route_case} candidate16 skip_sentinel threads64",
                    candidate_fn,
                    q,
                    kv,
                    out,
                    do,
                    indices,
                    lse,
                    args.sm_scale,
                    args.warmup,
                    args.iterations,
                    args.direct_dkv,
                )
                original64_a = run_baseline(
                    f"{case_name} {route_case} original16 threads64 #1",
                    baseline_fn,
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
                original64_b = run_baseline(
                    f"{case_name} {route_case} original16 threads64 #2",
                    baseline_fn,
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
            else:
                original64_a = run_baseline(
                    f"{case_name} {route_case} original16 threads64 #1",
                    baseline_fn,
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
                original64_b = run_baseline(
                    f"{case_name} {route_case} original16 threads64 #2",
                    baseline_fn,
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
                candidate16 = run_candidate(
                    f"{case_name} {route_case} candidate16 skip_sentinel threads64",
                    candidate_fn,
                    q,
                    kv,
                    out,
                    do,
                    indices,
                    lse,
                    args.sm_scale,
                    args.warmup,
                    args.iterations,
                    args.direct_dkv,
                )
            results = (original64_a, original64_b, candidate16)
            if needs_reference:
                print_small_comparisons(route_case, results, out.cpu(), reference)
            else:
                compare_actual(route_case, original64_a, original64_b, candidate16)
            del q, kv, do, indices, out, lse, reference
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
