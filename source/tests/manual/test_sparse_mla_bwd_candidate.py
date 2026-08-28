"""CUDA-gated checks for the opt-in sparse-MLA backward store candidate.

Run on an idle RTX 4090 with the trainer environment, for example:

    PRIME_SPARSE_BWD_REUSE_DQ_SHARED=1 python3 -m \
        source.tests.manual.test_sparse_mla_bwd_candidate

The test compares the candidate with the reference in one process.  dQ must
be exact; dKV uses a small tolerance because both paths use duplicate-index
FP32 atomics and CUDA does not promise their global order.
"""

from __future__ import annotations

import os

import torch

from prime_rl.trainer.models.kernels.sparse_mla_bwd import bwd, sparse_mla_bwd
from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface


def _set_candidate(enabled: bool) -> None:
    os.environ["PRIME_SPARSE_BWD_REUSE_DQ_SHARED"] = "1" if enabled else "0"


def _assert_close(candidate, reference, label: str) -> None:
    try:
        torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-3)
    except AssertionError as error:
        raise AssertionError(label) from error


def _production_case():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260826)
    batch, sequence, heads, kv_length, topk = 1, 4, 16, 9, 32
    query = torch.randn(
        batch,
        sequence,
        heads,
        576,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value = torch.randn(
        batch,
        kv_length,
        1,
        576,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(0.02)
    key_value[:, -1].zero_()
    indices = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .remainder(kv_length - 1)
        .view(1, 1, 1, topk)
        .expand(batch, sequence, 1, topk)
        .contiguous()
    )
    output, lse = sparse_mla_fwd_interface(
        query, key_value, indices, sm_scale=576**-0.5
    )
    output_gradient = torch.randn(
        output.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    ).mul_(1e-3)
    return query, key_value, output, output_gradient, indices, lse


def test_duplicate_parity_and_repeated_calls() -> None:
    query, key_value, output, output_gradient, indices, lse = _production_case()
    _set_candidate(False)
    reference_q, reference_kv = sparse_mla_bwd(
        query, key_value, output, output_gradient, indices, lse
    )
    _set_candidate(True)
    candidate_q, candidate_kv = sparse_mla_bwd(
        query, key_value, output, output_gradient, indices, lse
    )
    repeat_q, repeat_kv = sparse_mla_bwd(
        query, key_value, output, output_gradient, indices, lse
    )
    torch.cuda.synchronize()
    if not torch.equal(candidate_q, repeat_q):
        raise AssertionError("repeated candidate dQ is not deterministic")
    _assert_close(candidate_q, reference_q, "candidate dQ parity")
    _assert_close(candidate_kv, reference_kv, "candidate dKV duplicate parity")
    _assert_close(repeat_kv, candidate_kv, "candidate dKV repeated-call parity")


def test_malformed_indices_and_nonfinite_lse_are_fail_closed() -> None:
    device = torch.device("cuda")
    batch, sequence, heads, kv_length, topk = 1, 2, 16, 9, 32
    query = torch.ones(
        batch, sequence, heads, 576, device=device, dtype=torch.bfloat16
    )
    key_value = torch.ones(
        batch, kv_length, 1, 576, device=device, dtype=torch.bfloat16
    )
    key_value[:, -1].zero_()
    malformed = torch.tensor(
        [-1, kv_length - 1, kv_length, 2**31 - 1],
        device=device,
        dtype=torch.int32,
    ).repeat(topk // 4)
    indices = malformed.view(1, 1, 1, topk).expand(batch, sequence, 1, topk).contiguous()
    output = torch.zeros(batch, sequence, heads, 512, device=device, dtype=torch.bfloat16)
    output_gradient = torch.ones_like(output)
    lse = torch.full((batch, sequence, heads), float("inf"), device=device)
    _set_candidate(True)
    query_gradient, key_value_gradient = sparse_mla_bwd(
        query, key_value, output, output_gradient, indices, lse
    )
    torch.cuda.synchronize()
    if not bool(torch.isfinite(query_gradient).all()):
        raise AssertionError("malformed rows produced nonfinite dQ")
    if not bool(torch.isfinite(key_value_gradient).all()):
        raise AssertionError("malformed rows produced nonfinite dKV")
    if bool(query_gradient.any()) or bool(key_value_gradient.any()):
        raise AssertionError("malformed/all-masked rows were not zero")


def test_tail_shapes_match_reference() -> None:
    device = torch.device("cuda")
    batch, sequence, heads, kv_length, topk = 1, 2, 16, 9, 32
    generator = torch.Generator(device=device).manual_seed(23)
    indices = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .remainder(kv_length - 1)
        .view(1, 1, 1, topk)
        .expand(batch, sequence, 1, topk)
        .contiguous()
    )
    for tail in (16, 64, 128):
        query = torch.randn(
            batch,
            sequence,
            heads,
            512 + tail,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        key_value = torch.randn(
            batch,
            kv_length,
            1,
            512 + tail,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        key_value[:, -1].zero_()
        output = torch.randn(
            batch, sequence, heads, 512, device=device, dtype=torch.bfloat16, generator=generator
        )
        output_gradient = torch.randn_like(output)
        lse = torch.zeros(batch, sequence, heads, device=device, dtype=torch.float32)
        _set_candidate(False)
        reference_q, reference_kv = sparse_mla_bwd(
            query, key_value, output, output_gradient, indices, lse
        )
        _set_candidate(True)
        candidate_q, candidate_kv = sparse_mla_bwd(
            query, key_value, output, output_gradient, indices, lse
        )
        _assert_close(candidate_q, reference_q, f"tail={tail} dQ parity")
        _assert_close(candidate_kv, reference_kv, f"tail={tail} dKV parity")


def test_compile_paths() -> None:
    for heads, topk in ((16, 32), (64, 128)):
        for reuse in (False, True):
            bwd(
                heads,
                512,
                64,
                topk,
                1,
                576**-0.5,
                True,
                block_size=16,
                block_H=16,
                threads=64,
                vectorized_atomic=False,
                reuse_dq_shared=reuse,
            )


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available")
        return
    test_compile_paths()
    if torch.cuda.get_device_capability() != (8, 9):
        print("sparse MLA backward candidate compile paths: PASS")
        return
    test_duplicate_parity_and_repeated_calls()
    test_malformed_indices_and_nonfinite_lse_are_fail_closed()
    test_tail_shapes_match_reference()
    print("sparse MLA backward candidate: PASS")


if __name__ == "__main__":
    main()
