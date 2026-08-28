"""Final PP16 CUDA gate for the GLM virtual-chunk schedule.

Run on one 8-GPU host in the qualification image:

    torchrun --standalone --nproc-per-node=8 source/tests/test_final_pp16_cuda.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.pipelining import ScheduleInterleaved1F1B

from prime_rl.trainer.distributed import pp_runner as runner


PHYSICAL = 8
VIRTUAL = 16
MICROBATCHES = 95
SEQ_LEN = 4
HIDDEN = 4
INDEX_TOPK = 2
LEARNING_RATE = 1e-4
VIRTUAL_BOUNDARIES = (0, 3, 7, 11, 15, 19, 23, 27, 31, 36, 42, 48, 54, 60, 66, 72, 78)
FULL_INDEX_LAYERS = frozenset(
    (0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74)
)
HIDDEN_SLOT_BYTES = 4096 * 6144 * 2
SPARSE_SLOT_BYTES = 4096 * 2048 * 2
MAX_PROJECTED_BYTES = int(1.28125 * 2**30)
PROCESS_GROUP_TIMEOUT_ENV = "PRIME_PP_CUDA_PROCESS_GROUP_TIMEOUT_SECONDS"
RESULT_PATH_ENV = "PRIME_PP_CUDA_RESULT_PATH"


def side_value(stage: int, microbatch: int) -> float:
    return float((stage + 1) * 1e-3 + microbatch * 1e-5)


def stage_sparse_flags(stage: int) -> tuple[bool, bool]:
    """Return the accepted production receive/send flags for one stage."""
    start = VIRTUAL_BOUNDARIES[stage]
    receive = stage > 0 and start not in FULL_INDEX_LAYERS
    send = stage < VIRTUAL - 1 and VIRTUAL_BOUNDARIES[stage + 1] not in FULL_INDEX_LAYERS
    return receive, send


def cache_source_stage(stage: int) -> int:
    """Return the preceding stage that generated the sparse-cache bits."""
    for candidate in range(stage - 1, -1, -1):
        receive, _ = stage_sparse_flags(candidate)
        if candidate == 0 or not receive:
            return candidate
    raise AssertionError(f"stage {stage} has no sparse-cache source")


def cache_bits(stage: int, device: torch.device) -> torch.Tensor:
    """Create deterministic FP16 transport bits like production conversion."""
    count = SEQ_LEN * INDEX_TOPK
    bits = torch.arange(count, device=device, dtype=torch.int16)
    bits.add_(stage * 17 + 1)
    return bits.view(torch.float16).view(1, SEQ_LEN, 1, INDEX_TOPK)


class SideAffine(nn.Module):
    """Small deterministic layer with production-style keyed side inputs."""

    def __init__(self, stage: int, receive_sparse: bool, send_sparse: bool, is_last: bool):
        super().__init__()
        self.stage = stage
        self.receive_sparse = receive_sparse
        self.send_sparse = send_sparse
        self.is_last = is_last
        self.scale = nn.Parameter(torch.tensor(1.0 + stage * 0.01))
        self.bias = nn.Parameter(torch.tensor(stage * 0.01))
        self._side: dict[tuple[int, int], dict] = {}
        self._consumed: list[tuple[int, int]] = []
        self._active: tuple[int, int] | None = None
        self._offset = 0.0

    def set_side_inputs(self, side: dict[tuple[int, int], dict]) -> None:
        expected = {(self.stage, microbatch) for microbatch in range(MICROBATCHES)}
        if set(side) != expected:
            raise AssertionError(f"stage {self.stage} received incorrect side keys")
        self._side = dict(side)
        self._consumed.clear()
        self._active = None

    def set_active_microbatch(self, microbatch: int) -> None:
        if self._active is not None:
            raise AssertionError(f"stage {self.stage} has nested active microbatches")
        key = (self.stage, int(microbatch))
        if key in self._consumed or key not in self._side:
            raise AssertionError(f"invalid side-input consumption {key}")
        value = self._side.pop(key)
        if value["marker"] != f"stage={self.stage};microbatch={microbatch}":
            raise AssertionError(f"side-input key mismatch for {key}")
        self._active = key
        self._offset = float(value["offset"])
        self._consumed.append(key)

    def clear_active_microbatch(self) -> None:
        self._active = None

    def forward(self, hidden, cache=None):
        if self._active is None:
            raise AssertionError(f"stage {self.stage} ran without a side-input key")
        output = hidden * self.scale + self.bias + self._offset
        if self.is_last:
            if not self.receive_sparse:
                raise AssertionError("last production stage must receive sparse cache")
            return output.sum()
        if self.send_sparse:
            if self.receive_sparse:
                if cache is None:
                    raise AssertionError(f"stage {self.stage} lost its sparse cache")
                expected = cache_bits(cache_source_stage(self.stage), cache.device)
                torch.testing.assert_close(cache, expected, rtol=0, atol=0)
                cache_out = cache.detach()
            else:
                cache_out = cache_bits(self.stage, hidden.device)
        else:
            # Production still returns a second value when transfer is
            # enabled. It is a one-element placeholder when this stage does
            # not send a real sparse cache.
            cache_out = torch.zeros(1, dtype=torch.float16, device=hidden.device)
        return output, cache_out

    def assert_side_inputs_freed(self) -> None:
        expected = [(self.stage, microbatch) for microbatch in range(MICROBATCHES)]
        if self._side or sorted(self._consumed) != expected or self._active is not None:
            raise AssertionError(
                f"stage {self.stage} side-input lifecycle is invalid: "
                f"remaining={sorted(self._side)}, consumed={len(self._consumed)}"
            )


@dataclass
class Reference:
    loss: float
    output: torch.Tensor
    input_grad: torch.Tensor
    scale_grads: dict[int, torch.Tensor]
    bias_grads: dict[int, torch.Tensor]
    scale_updates: dict[int, torch.Tensor]
    bias_updates: dict[int, torch.Tensor]


class LossCounter:
    def __init__(self):
        self.calls = 0
        self.total = 0.0

    def __call__(self, output, _target):
        self.calls += 1
        loss = output if output.ndim == 0 else output.sum()
        self.total += float(loss.detach())
        return loss


def make_side_inputs(stage: int) -> dict[tuple[int, int], dict]:
    return {
        (stage, microbatch): {
            "marker": f"stage={stage};microbatch={microbatch}",
            "offset": side_value(stage, microbatch),
        }
        for microbatch in range(MICROBATCHES)
    }


def make_inputs(device: torch.device) -> torch.Tensor:
    return torch.arange(
        MICROBATCHES * SEQ_LEN * HIDDEN, device=device, dtype=torch.float32
    ).view(MICROBATCHES, SEQ_LEN, HIDDEN).div_(1000).requires_grad_()


def ordinary_reference(device: torch.device) -> Reference:
    inputs = make_inputs(device)
    modules = [
        SideAffine(stage, *stage_sparse_flags(stage), is_last=stage == VIRTUAL - 1).to(device)
        for stage in range(VIRTUAL)
    ]
    outputs = []
    for microbatch in range(MICROBATCHES):
        value = inputs[microbatch : microbatch + 1]
        for module in modules:
            value = value * module.scale + module.bias + side_value(module.stage, microbatch)
        # Production calls backward once per real micro-batch. Keep this
        # order because BF16 stage arithmetic and FP32 gradient accumulation
        # are not bit-equivalent to one concatenated backward.
        loss = value.sum()
        outputs.append(loss.detach())
        loss.backward()
    losses = torch.stack(outputs)
    scale_grads = {module.stage: module.scale.grad.detach().cpu().clone() for module in modules}
    bias_grads = {module.stage: module.bias.grad.detach().cpu().clone() for module in modules}
    scale_updates = {}
    bias_updates = {}
    for module in modules:
        scale_updates[module.stage] = module.scale.detach().clone()
        scale_updates[module.stage].add_(module.scale.grad, alpha=-LEARNING_RATE)
        bias_updates[module.stage] = module.bias.detach().clone()
        bias_updates[module.stage].add_(module.bias.grad, alpha=-LEARNING_RATE)
    return Reference(
        loss=float(losses.sum()),
        output=losses.detach().cpu(),
        input_grad=inputs.grad.detach().cpu(),
        scale_grads=scale_grads,
        bias_grads=bias_grads,
        scale_updates={key: value.cpu() for key, value in scale_updates.items()},
        bias_updates={key: value.cpu() for key, value in bias_updates.items()},
    )


def verify_schedule_actions(schedule) -> dict[str, int]:
    positions = runner._schedule_action_positions(schedule.pipeline_order_with_comms)
    synthetic = [
        key
        for key in positions
        if key[2] in {"F", "B"} and key[1] >= MICROBATCHES
    ]
    if synthetic:
        raise AssertionError(f"synthetic microbatches reached model actions: {synthetic[:4]}")
    for stage in range(VIRTUAL):
        forward = [
            microbatch
            for microbatch in range(MICROBATCHES)
            if (stage, microbatch, "F") in positions
        ]
        backward = [
            microbatch
            for microbatch in range(MICROBATCHES)
            if (stage, microbatch, "B") in positions
        ]
        if forward != list(range(MICROBATCHES)) or backward != list(range(MICROBATCHES)):
            raise AssertionError(f"stage {stage} has dropped or repeated microbatches")
    return {
        "real_forward_actions": VIRTUAL * MICROBATCHES,
        "real_backward_actions": VIRTUAL * MICROBATCHES,
        "all_action_entries": sum(len(actions) for actions in schedule.pipeline_order_with_comms.values()),
    }


def run_repeat(rank: int, reference: Reference, repeat: int) -> dict[str, object]:
    device = torch.device("cuda", torch.cuda.current_device())
    stages = []
    modules = []
    hidden_example = torch.empty((1, SEQ_LEN, HIDDEN), device=device)
    sparse_cache_example = torch.empty(
        (1, SEQ_LEN, 1, INDEX_TOPK), dtype=torch.float16, device=device
    )
    placeholder_example = torch.empty((1,), dtype=torch.float16, device=device)
    for local_index in range(2):
        stage_index = rank + local_index * PHYSICAL
        receive_sparse, send_sparse = stage_sparse_flags(stage_index)
        module = SideAffine(
            stage_index,
            receive_sparse=receive_sparse,
            send_sparse=send_sparse,
            is_last=stage_index == VIRTUAL - 1,
        ).to(device)
        module.set_side_inputs(make_side_inputs(stage_index))
        if stage_index == 0:
            input_args = hidden_example
        else:
            input_args = (
                hidden_example,
                sparse_cache_example if receive_sparse else placeholder_example,
            )
        if stage_index == VIRTUAL - 1:
            output_args = torch.empty((), dtype=torch.float32, device=device)
        else:
            output_args = (
                hidden_example,
                sparse_cache_example if send_sparse else placeholder_example,
            )
        modules.append(module)
        stages.append(
            runner._sparse_cache_pipeline_stage_cls()(
                module,
                stage_index=stage_index,
                num_stages=VIRTUAL,
                device=device,
                group=dist.group.WORLD,
                input_args=input_args,
                output_args=output_args,
            )
        )

    flexible = runner._flexible_interleaved_schedule_cls(ScheduleInterleaved1F1B)
    loss_counter = LossCounter()
    schedule = flexible(
        stages,
        n_microbatches=MICROBATCHES,
        loss_fn=loss_counter,
        scale_grads=False,
    )
    action_report = verify_schedule_actions(schedule)
    capacities = runner.derive_interleaved_transport_capacities(
        schedule.pipeline_order_with_comms,
        schedule.stage_index_to_group_rank,
        num_stages=VIRTUAL,
        num_microbatches=MICROBATCHES,
    )
    for stage in stages:
        stage.require_exact_transport_capacities()
        stage.set_transport_capacities(capacities[stage.stage_index])

    inputs = make_inputs(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    losses = [] if rank == PHYSICAL - 1 else None
    if rank == 0:
        result = schedule.step(inputs, return_outputs=False)
    else:
        result = schedule.step(
            target=torch.zeros(MICROBATCHES, device=device),
            losses=losses,
            return_outputs=False,
        )
    end.record()
    end.synchronize()
    wall_ms = start.elapsed_time(end)
    peak_delta = torch.cuda.max_memory_allocated(device) - before

    if rank == 0:
        torch.testing.assert_close(inputs.grad.detach().cpu(), reference.input_grad, rtol=0, atol=0)
    if rank == PHYSICAL - 1:
        if losses is None or len(losses) != MICROBATCHES:
            raise AssertionError(f"loss list has {len(losses or [])} chunks, expected {MICROBATCHES}")
        torch.testing.assert_close(
            torch.stack([loss.detach() for loss in losses]).cpu(),
            reference.output,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            torch.tensor(loss_counter.total),
            torch.tensor(reference.loss),
            rtol=0,
            atol=2e-4,
        )
        if loss_counter.calls != MICROBATCHES:
            raise AssertionError(
                f"loss saw {loss_counter.calls} chunks, expected {MICROBATCHES}"
            )

    for module in modules:
        module.assert_side_inputs_freed()
        expected_scale_grad = reference.scale_grads[module.stage].to(device)
        expected_bias_grad = reference.bias_grads[module.stage].to(device)
        torch.testing.assert_close(module.scale.grad, expected_scale_grad, rtol=0, atol=2e-5)
        torch.testing.assert_close(module.bias.grad, expected_bias_grad, rtol=0, atol=2e-5)
        module.scale.data.add_(module.scale.grad, alpha=-LEARNING_RATE)
        module.bias.data.add_(module.bias.grad, alpha=-LEARNING_RATE)
        torch.testing.assert_close(
            module.scale.cpu(), reference.scale_updates[module.stage], rtol=0, atol=2e-6
        )
        torch.testing.assert_close(
            module.bias.cpu(), reference.bias_updates[module.stage], rtol=0, atol=2e-6
        )
        if any(microbatch >= MICROBATCHES for _, microbatch in module._consumed):
            raise AssertionError("synthetic microbatch reached a model")

    ring_bytes = sum(
        tensor.numel() * tensor.element_size()
        for stage in stages
        for buffers in (
            getattr(stage, "_forward_slots", []),
            getattr(stage, "_backward_slots", []),
            getattr(stage, "_sparse_cache_slots", []),
        )
        for tensor in buffers
    )
    projected_bytes = sum(
        (capacities[stage_index]["forward"] + capacities[stage_index]["backward"]) * HIDDEN_SLOT_BYTES
        + capacities[stage_index]["sparse"] * SPARSE_SLOT_BYTES
        for stage_index in (rank, rank + PHYSICAL)
    )
    peak = torch.tensor(int(peak_delta), device=device, dtype=torch.int64)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    tiny_ring = torch.tensor(ring_bytes, device=device, dtype=torch.int64)
    dist.all_reduce(tiny_ring, op=dist.ReduceOp.MAX)
    projected = torch.tensor(projected_bytes, device=device, dtype=torch.int64)
    dist.all_reduce(projected, op=dist.ReduceOp.MAX)
    if int(projected.item()) > MAX_PROJECTED_BYTES:
        raise AssertionError(f"projected ring peak exceeds bound: {int(projected.item())}")
    report = {
        "repeat": repeat,
        "schedule": type(schedule).__name__,
        "wall_ms": wall_ms,
        "peak_delta_bytes_max": int(peak.item()),
        "ring_bytes_tiny_max": int(tiny_ring.item()),
        "projected_peak_bytes_max": int(projected.item()),
        "virtual_boundaries": list(VIRTUAL_BOUNDARIES),
        **action_report,
    }
    del result, schedule, stages, modules, inputs
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return report


def distributed_launch() -> tuple[int, torch.device, int]:
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        timeout_seconds = int(os.environ.get(PROCESS_GROUP_TIMEOUT_ENV, "120"))
    except (KeyError, ValueError) as error:
        raise RuntimeError("the CUDA qualification launch environment is invalid") from error
    if world_size != PHYSICAL or local_world_size != PHYSICAL:
        raise RuntimeError(
            f"expected exactly {PHYSICAL} local ranks, got "
            f"world={world_size} local_world={local_world_size}"
        )
    if not 0 <= local_rank < PHYSICAL:
        raise RuntimeError(f"local CUDA rank is invalid: {local_rank}")
    if torch.cuda.device_count() != PHYSICAL:
        raise RuntimeError(
            f"expected exactly {PHYSICAL} CUDA devices, got {torch.cuda.device_count()}"
        )
    if not 1 <= timeout_seconds <= 300:
        raise RuntimeError("process-group timeout must be from 1 to 300 seconds")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return local_rank, device, timeout_seconds


def main() -> None:
    # The pinned runtime prints the full 1,145-step schedule on an exception.
    # Keep the failure traceback readable during this gate.
    logging.getLogger("torch.distributed.pipelining.schedules").setLevel(logging.CRITICAL)
    local_rank, device, timeout_seconds = distributed_launch()
    ready = False
    try:
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=timeout_seconds),
            device_id=device,
        )
        ready = True
        rank = dist.get_rank()
        if rank != int(os.environ["RANK"]) or dist.get_world_size() != PHYSICAL:
            raise RuntimeError("initialized process-group identity differs from torchrun")
        if torch.cuda.current_device() != local_rank:
            raise RuntimeError("initialized process group changed the local CUDA device")
        reports = []
        previous = None
        for repeat in range(3):
            reference = ordinary_reference(torch.device("cuda", torch.cuda.current_device()))
            try:
                report = run_repeat(rank, reference, repeat)
            except BaseException as exc:
                print(
                    f"rank={rank} repeat={repeat} failure={type(exc).__name__}: {exc}",
                    flush=True,
                )
                raise
            if previous is not None and rank == 0:
                if report["real_forward_actions"] != previous["real_forward_actions"]:
                    raise AssertionError("repeated schedule changed real action count")
                if abs(report["wall_ms"] - previous["wall_ms"]) > 10000:
                    raise AssertionError("repeated schedule timing is invalid")
            previous = report
            if rank == 0:
                reports.append(report)
        if rank == 0:
            result = {
                "status": "pass",
                "ranks": PHYSICAL,
                "repeats": len(reports),
                "reports": reports,
            }
            result_path = os.environ.get(RESULT_PATH_ENV)
            if result_path:
                destination = Path(result_path)
                if destination != Path("/output/result.json"):
                    raise RuntimeError("the CUDA qualification result path is invalid")
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, sort_keys=True) + "\n")
                os.replace(temporary, destination)
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if ready:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
