from __future__ import annotations

import json
import statistics
from types import SimpleNamespace

import torch

import prime_rl.trainer.models.layers.fp8_ada as fp8_ada


ROWS = 8192
EXPERTS = 32
RANK = 32
CHUNK_ROWS = 4096
WARMUP = 1
MEASURED = 3


class ReceivedInput:
    row_index_device = "cuda"
    local_recompute = True
    save_for_backward = True

    def __init__(self, source):
        self.source = source.detach().clone()

    def make_workspace(self, _chunk_rows, _device):
        return None

    def reconstruct_rows(self, indices, destination, _workspace):
        safe = indices.clamp(0, self.source.shape[0] - 1).long()
        destination.copy_(self.source.index_select(0, safe))


def make_case(device):
    generator = torch.Generator(device="cpu").manual_seed(101)

    def rand(*shape, scale=1.0):
        return (
            torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
            * scale
        )

    route = torch.randint(EXPERTS, (ROWS,), generator=generator)
    order = torch.argsort(route, stable=True).to(device)
    counts = torch.bincount(route, minlength=EXPERTS).to(device)
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
    x = rand(ROWS, 6144)
    w1 = rand(EXPERTS, 2048, 6144, scale=0.02).to(torch.float8_e4m3fn)
    w3 = rand(EXPERTS, 2048, 6144, scale=0.02).to(torch.float8_e4m3fn)
    w2 = rand(EXPERTS, 6144, 2048, scale=0.02).to(torch.float8_e4m3fn)
    s1 = torch.ones(EXPERTS, 16, 48, device=device, dtype=torch.float32)
    s3 = torch.ones_like(s1)
    s2 = torch.ones(EXPERTS, 48, 16, device=device, dtype=torch.float32)
    w1_a = rand(EXPERTS, RANK, 6144, scale=0.02)
    w1_b = rand(EXPERTS, 2048, RANK, scale=0.02)
    w3_a = rand(EXPERTS, RANK, 6144, scale=0.02)
    w3_b = rand(EXPERTS, 2048, RANK, scale=0.02)
    w2_a = rand(EXPERTS, RANK, 2048, scale=0.02)
    w2_b = rand(EXPERTS, 6144, RANK, scale=0.02)
    grad = rand(ROWS, 6144)
    scores = torch.ones(ROWS, device=device, dtype=torch.float32)
    base_layer = SimpleNamespace(
        w1=w1,
        w1_scale_inv=s1,
        w2=w2,
        w2_scale_inv=s2,
        w3=w3,
        w3_scale_inv=s3,
    )
    owner = SimpleNamespace(
        base_layer=base_layer,
        w1_lora_A=(w1_a,),
        w1_lora_B=(w1_b,),
        w2_lora_A=(w2_a,),
        w2_lora_B=(w2_b,),
        w3_lora_A=(w3_a,),
        w3_lora_B=(w3_b,),
    )
    return (
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        w3_a,
        w3_b,
        order,
        offsets,
        grad,
        scores,
        owner,
    )


def run(case, paired, backward):
    (
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        w3_a,
        w3_b,
        order,
        offsets,
        grad,
        scores,
        owner,
    ) = case
    if backward:
        x = (x.clone().requires_grad_(True) + 0)
        x.retain_grad()
        factors = [
            value.clone().requires_grad_(True)
            for value in (w1_a, w1_b, w2_a, w2_b, w3_a, w3_b)
        ]
        w1_a, w1_b, w2_a, w2_b, w3_a, w3_b = factors
        owner = SimpleNamespace(
            base_layer=owner.base_layer,
            w1_lora_A=(w1_a,),
            w1_lora_B=(w1_b,),
            w2_lora_A=(w2_a,),
            w2_lora_B=(w2_b,),
            w3_lora_A=(w3_a,),
            w3_lora_B=(w3_b,),
        )
    else:
        x = x.clone()
    received = ReceivedInput(x)
    grad = grad.clone()
    output = fp8_ada._InplacePermutedLoRAExperts.apply(
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        w1_a,
        w1_b,
        w2_a,
        w2_b,
        w3_a,
        w3_b,
        1.0,
        order,
        offsets,
        owner,
        received,
        scores,
        CHUNK_ROWS,
    )
    if backward:
        torch.autograd.backward(output, grad)
        return (
            output.detach(),
            x.grad.detach(),
            tuple(value.grad.detach() for value in factors),
        )
    return output.detach()


def time_run(case, paired, backward):
    values = []
    peak = 0
    for index in range(WARMUP + MEASURED):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = run(case, paired, backward)
        end.record()
        end.synchronize()
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**2)
        if index >= WARMUP:
            values.append(start.elapsed_time(end))
        del result
    return statistics.median(values), peak


def relative_rmse(value, reference):
    difference = value.float() - reference.float()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float(difference.square().mean().sqrt() / denominator)


def gradient_errors(candidate, baseline):
    return [relative_rmse(got, want) for got, want in zip(candidate, baseline, strict=True)]


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("this benchmark requires an RTX 4090 SM89 GPU")
    device = torch.device("cuda")
    case = make_case(device)
    original_pair = fp8_ada._grouped_lora_projection_pair

    def separate_pair(x, a1, b1, a3, b3, offsets):
        return (
            fp8_ada._grouped_lora_projection(x, a1, b1, offsets),
            fp8_ada._grouped_lora_projection(x, a3, b3, offsets),
        )

    fp8_ada._grouped_lora_projection_pair = separate_pair
    baseline_forward = run(case, False, False)
    baseline_backward, baseline_input_grad, baseline_grads = run(case, False, True)
    baseline_forward_ms, baseline_peak = time_run(case, False, False)
    baseline_backward_ms, baseline_backward_peak = time_run(case, False, True)

    fp8_ada._grouped_lora_projection_pair = original_pair
    candidate_forward = run(case, True, False)
    candidate_backward, candidate_input_grad, candidate_grads = run(case, True, True)
    candidate_forward_ms, candidate_peak = time_run(case, True, False)
    candidate_backward_ms, candidate_backward_peak = time_run(case, True, True)

    errors = gradient_errors(candidate_grads, baseline_grads)
    print(json.dumps({
        "device": torch.cuda.get_device_name(),
        "rows": ROWS,
        "experts": EXPERTS,
        "rank": RANK,
        "native_fp8_base": True,
        "forward_relative_rmse": relative_rmse(candidate_forward, baseline_forward),
        "gradient_relative_rmse": max(errors),
        "gradient_relative_rmse_by_factor": errors,
        "input_gradient_relative_rmse": relative_rmse(
            candidate_input_grad,
            baseline_input_grad,
        ),
        "baseline_forward_ms_median": baseline_forward_ms,
        "candidate_forward_ms_median": candidate_forward_ms,
        "baseline_backward_ms_median": baseline_backward_ms,
        "candidate_backward_ms_median": candidate_backward_ms,
        "baseline_forward_peak_mib": baseline_peak,
        "candidate_forward_peak_mib": candidate_peak,
        "baseline_backward_peak_mib": baseline_backward_peak,
        "candidate_backward_peak_mib": candidate_backward_peak,
    }, indent=2))


if __name__ == "__main__":
    main()
