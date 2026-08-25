"""Two-rank CUDA check for the bounded interleaved pipeline stages.

Run inside the qualification trainer image with two GPUs:

    torchrun --standalone --nproc-per-node=2 source/tests/test_interleaved_cuda.py
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining import ScheduleInterleaved1F1B

from prime_rl.trainer.distributed.pp_runner import (
    _bounded_pipeline_stage_cls,
    derive_interleaved_transport_capacities,
    _sparse_cache_pipeline_stage_cls,
)


STAGES = 4
HIDDEN = 4


class Affine(nn.Module):
    def __init__(self, stage_index: int, sparse: bool):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(stage_index + 2)))
        self.bias = nn.Parameter(torch.tensor(float(stage_index)))
        self.sparse = sparse

    def set_active_microbatch(self, _microbatch: int) -> None:
        pass

    def clear_active_microbatch(self) -> None:
        pass

    def forward(self, hidden, cache=None):
        output = hidden * self.scale + self.bias
        if not self.sparse:
            return output
        if cache is None:
            cache = torch.empty(
                (hidden.shape[0], 2), device=hidden.device, dtype=hidden.dtype
            )
            cache.copy_(hidden.detach()[:, :2])
        return output, cache.detach()


def _expected_hidden(inputs: torch.Tensor) -> torch.Tensor:
    result = inputs
    for stage_index in range(STAGES):
        result = result * (stage_index + 2) + stage_index
    return result


def _expected_parameter_grads(inputs: torch.Tensor, stage_index: int) -> tuple[float, float]:
    value = inputs
    for index in range(stage_index):
        value = value * (index + 2) + index
    downstream = 1.0
    for index in range(stage_index + 1, STAGES):
        downstream *= index + 2
    return (float((value * downstream).sum()), float(inputs.shape[0] * HIDDEN * downstream))


def _run(rank: int, sparse: bool, microbatches: int) -> dict[str, object] | None:
    device = torch.device("cuda", torch.cuda.current_device())
    stage_cls = _sparse_cache_pipeline_stage_cls() if sparse else _bounded_pipeline_stage_cls()
    hidden_example = torch.empty((1, HIDDEN), device=device)
    cache_example = torch.empty((1, 2), device=device)
    stages = []
    modules = []
    for local_index in range(2):
        stage_index = rank + local_index * 2
        module = Affine(stage_index, sparse).to(device)
        modules.append(module)
        first = stage_index == 0
        input_example = hidden_example if first else (
            (hidden_example, cache_example) if sparse else hidden_example
        )
        output_example = (
            (hidden_example, cache_example)
            if sparse
            else hidden_example
        )
        stages.append(
            stage_cls(
                module,
                stage_index=stage_index,
                num_stages=STAGES,
                device=device,
                group=dist.group.WORLD,
                input_args=input_example,
                output_args=output_example,
            )
        )
    schedule = ScheduleInterleaved1F1B(
        stages,
        n_microbatches=microbatches,
        loss_fn=(lambda output, _target: output[0].sum()) if sparse else (
            lambda output, _target: output.sum()
        ),
        scale_grads=False,
    )
    capacities = derive_interleaved_transport_capacities(
        schedule.pipeline_order_with_comms,
        schedule.stage_index_to_group_rank,
        num_stages=STAGES,
        num_microbatches=microbatches,
    )
    for stage in stages:
        stage.require_exact_transport_capacities()
        stage.set_transport_capacities(capacities[stage.stage_index])
    inputs = torch.arange(microbatches * HIDDEN, device=device, dtype=torch.float32)
    inputs = inputs.view(microbatches, HIDDEN)
    expected = _expected_hidden(inputs)
    before = torch.cuda.memory_allocated()
    if rank == 0:
        result = schedule.step(inputs, return_outputs=False)
    else:
        result = schedule.step(
            target=torch.zeros(microbatches, device=device), return_outputs=True
        )
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    if rank == 1:
        output = result[0] if sparse else result
        if output.shape != expected.shape:
            raise AssertionError(f"output shape {output.shape} differs from {expected.shape}")
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    for local_index, module in enumerate(modules):
        stage_index = rank + local_index * 2
        if module.scale.grad is None or module.bias.grad is None:
            raise AssertionError(f"stage {stage_index} has a missing gradient")
        if not torch.isfinite(module.scale.grad) or not torch.isfinite(module.bias.grad):
            raise AssertionError(f"stage {stage_index} has a non-finite gradient")
        expected_scale, expected_bias = _expected_parameter_grads(inputs, stage_index)
        torch.testing.assert_close(
            module.scale.grad,
            torch.tensor(expected_scale, device=device),
            rtol=0,
            atol=1e-5,
        )
        torch.testing.assert_close(
            module.bias.grad,
            torch.tensor(expected_bias, device=device),
            rtol=0,
            atol=1e-5,
        )

    for stage in stages:
        expected = capacities[stage.stage_index]
        if stage._transport_slot_count != max(expected.values()):
            raise AssertionError("transport slot summary is incorrect")
        if len(stage._forward_slots) != expected["forward"]:
            raise AssertionError("hidden-state receive ring has the wrong size")
        if len(stage._backward_slots) != expected["backward"]:
            raise AssertionError("gradient receive ring has the wrong size")
        if sparse and not stage.is_first and expected["sparse"] > 0:
            if len(stage._sparse_cache_slot_owner) != expected["sparse"]:
                raise AssertionError("sparse-cache receive ring has the wrong size")

    if rank == 0:
        ring_bytes = sum(
            tensor.numel() * tensor.element_size()
            for stage in stages
            for buffers in (getattr(stage, "_forward_slots", []), getattr(stage, "_backward_slots", []))
            for tensor in buffers
        )
        if sparse:
            ring_bytes += sum(
                tensor.numel() * tensor.element_size()
                for stage in stages
                if not stage.is_first
                for tensor in stage._sparse_cache_slots
            )
        return {
            "sparse": sparse,
            "torch": torch.__version__,
            "microbatches": microbatches,
            "transport_capacities": capacities,
            "ring_bytes_hidden_and_gradient": ring_bytes,
            "cuda_allocated_before": before,
            "cuda_allocated_after": after,
            "cuda_peak_allocated": peak,
            "out_of_order_schedule": True,
        }
    return None


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    reports = []
    try:
        only = os.environ.get("PRIME_INTERLEAVED_ONLY")
        sparse_cases = (False,) if only == "nonsparse" else (True,) if only == "sparse" else (False, True)
        microbatch_cases = tuple(
            int(value)
            for value in os.environ.get("PRIME_INTERLEAVED_MICROBATCHES", "16,64").split(",")
            if value
        )
        for microbatches in microbatch_cases:
            for sparse in sparse_cases:
                report = _run(rank, sparse, microbatches)
                if report is not None:
                    reports.append(report)
        if rank == 0:
            print(json.dumps({"status": "pass", "reports": reports}, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
