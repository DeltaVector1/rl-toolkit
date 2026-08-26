import math
import os
import re
import weakref

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    get_default_config,
    moe_kernel_quantize_input,
)

from prime_rl.trainer.models.layers.checkpointing import (
    in_reentrant_block_checkpoint,
)
from prime_rl.trainer.models.layers.lm_head import (
    PrimeLmOutput,
    _online_logsumexp_and_weighted_update,
)

block_shape = [128, 128]
_FP8_W13_CHUNK_ROWS = 4096
_FP8_W2_CHUNK_ROWS = 2048
_FP8_EXPERT_BACKWARD_CHUNK_ROWS = 4096
_FP8_ROUTE_SCALE_CHUNK_ROWS = 4096
_FP8_EXPERT_ACTIVATION_OFFLOAD = (
    os.environ.get("PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD", "") == "1"
)
_DIAGNOSE_LINEAR_BACKWARD = False
_REPORTED_LINEAR_BACKWARD: set[tuple[tuple[int, ...], bool, bool]] = set()
_REPORTED_INPUT_GRAD: set[str] = set()
_REPORTED_W8A16_INPUT_GRAD: set[tuple[str, bool, bool]] = set()
_EXPERT_PROFILE = os.environ.get("PRIME_EP_PROFILE", "") == "1"
_EXPERT_BACKWARD_EVENTS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_EXPERT_PROFILE_EVERY = int(os.environ.get("PRIME_EXPERT_PROFILE_EVERY", "90"))
_EXPERT_PHASE_PROFILE_VALUE = os.environ.get("PRIME_EXPERT_PHASE_PROFILE", "0")
if _EXPERT_PHASE_PROFILE_VALUE not in {"0", "1"}:
    raise ValueError("PRIME_EXPERT_PHASE_PROFILE must be 0 or 1")
_EXPERT_PHASE_PROFILE = _EXPERT_PHASE_PROFILE_VALUE == "1"
_EXPERT_PHASE_PROFILE_EVERY = int(
    os.environ.get("PRIME_EXPERT_PHASE_PROFILE_EVERY", "180")
)
if _EXPERT_PHASE_PROFILE_EVERY <= 0:
    raise ValueError("PRIME_EXPERT_PHASE_PROFILE_EVERY must be positive")
_EXPERT_PHASE_EVENTS: dict[
    str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
] = {}
_EXPERT_PHASE_CALLS: dict[str, int] = {"forward": 0, "backward": 0}
_EXPERT_PHASE_LAST_END: dict[str, torch.cuda.Event] = {}
# Fallback for direct function calls that have no owning module. Production
# modules reacquire current unsharded FSDP parameters from their owner.
# Values are [weight, scale, refcount].
_LINEAR_SNAPSHOT_CACHE: dict[str, list] = {}


class W8A16BackwardFence:
    def __init__(self) -> None:
        self.event: torch.cuda.Event | None = None
        self.pending = False
        self.producer_stream: torch.cuda.Stream | None = None

    def record(self, device: torch.device) -> None:
        stream = torch.cuda.current_stream(device)
        if self.pending and self.producer_stream != stream:
            raise RuntimeError("one W8A16 FSDP unit used more than one CUDA stream")
        if self.event is None:
            self.event = torch.cuda.Event()
        self.event.record(stream)
        self.producer_stream = stream
        self.pending = True

    def wait_before_reshard(self, group) -> None:
        if not self.pending:
            return
        if self.event is None:
            raise RuntimeError("W8A16 backward event is missing")
        streams = {
            group.device_handle.current_stream(),
            group.comm_ctx.all_gather_copy_in_stream,
            group.comm_ctx.all_gather_stream,
        }
        for stream in streams:
            stream.wait_event(self.event)
        self.pending = False
        self.producer_stream = None


def exact_module_patterns(module_names, *, label="FP8") -> list[str]:
    if len(module_names) != len(set(module_names)):
        raise ValueError(f"{label} module names must be unique")
    for name in module_names:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError(f"Invalid {label} module name: {name!r}")
    return [rf"(?:^|\.){re.escape(name)}$" for name in module_names]


def _acquire_linear_snapshot(
    fqn: str,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    entry = _LINEAR_SNAPSHOT_CACHE.get(fqn)
    if entry is None:
        entry = [weight.detach().clone(), scale.detach().clone(), 0]
        _LINEAR_SNAPSHOT_CACHE[fqn] = entry
    elif tuple(entry[0].shape) != tuple(weight.shape) or tuple(entry[1].shape) != tuple(
        scale.shape
    ):
        raise RuntimeError(f"FP8 snapshot cache shape changed for {fqn}")
    entry[2] += 1
    return entry[0], entry[1]


def _release_linear_snapshot(fqn: str) -> None:
    entry = _LINEAR_SNAPSHOT_CACHE[fqn]
    entry[2] -= 1
    if entry[2] == 0:
        del _LINEAR_SNAPSHOT_CACHE[fqn]


def _profile_expert_backward(start, end) -> None:
    if not _EXPERT_PROFILE:
        return
    _EXPERT_BACKWARD_EVENTS.append((start, end))
    if len(_EXPERT_BACKWARD_EVENTS) < _EXPERT_PROFILE_EVERY:
        return
    end.synchronize()
    elapsed = sum(first.elapsed_time(last) for first, last in _EXPERT_BACKWARD_EVENTS)
    _EXPERT_BACKWARD_EVENTS.clear()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            f"[fp8-expert-prof] rank={os.environ.get('RANK', '0')} "
            f"backward_calls={_EXPERT_PROFILE_EVERY} "
            f"cuda={elapsed / 1000:.2f}s",
            flush=True,
        )


def _expert_phase_start(phase: str):
    """Start one routed-expert CUDA phase, or return None when disabled."""
    if not _EXPERT_PHASE_PROFILE:
        return None
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    return phase, start


def _expert_phase_end(timer) -> None:
    """End one routed-expert CUDA phase without synchronizing the stream."""
    if timer is None:
        return
    phase, start = timer
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    _EXPERT_PHASE_EVENTS.setdefault(phase, []).append((start, end))
    direction = phase.split(".", 1)[0]
    _EXPERT_PHASE_LAST_END[direction] = end


def _profile_expert_phases_tick(direction: str) -> None:
    """Report aggregate routed-expert phase CUDA time at a low frequency."""
    if not _EXPERT_PHASE_PROFILE:
        return
    _EXPERT_PHASE_CALLS[direction] += 1
    if _EXPERT_PHASE_CALLS[direction] % _EXPERT_PHASE_PROFILE_EVERY:
        return
    end = _EXPERT_PHASE_LAST_END.get(direction)
    if end is None:
        return
    end.synchronize()
    prefix = f"{direction}."
    totals = {}
    for phase, events in _EXPERT_PHASE_EVENTS.items():
        if phase.startswith(prefix):
            totals[phase[len(prefix) :]] = sum(
                start.elapsed_time(stop) for start, stop in events
            ) / 1000.0
    for phase in tuple(_EXPERT_PHASE_EVENTS):
        if phase.startswith(prefix):
            del _EXPERT_PHASE_EVENTS[phase]
    _EXPERT_PHASE_LAST_END.pop(direction, None)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        parts = " ".join(
            f"{name}={value:.2f}s" for name, value in sorted(totals.items())
        )
        print(
            f"[fp8-expert-phase-prof] rank={os.environ.get('RANK', '0')} "
            f"direction={direction} calls={_EXPERT_PHASE_PROFILE_EVERY} {parts}",
            flush=True,
        )


if hasattr(triton, "jit"):
    @triton.jit
    def _block_fp8_mm_kernel(
        x,
        weight,
        scale,
        out,
        rows,
        output_size: tl.constexpr,
        reduction_size: tl.constexpr,
        stride_xm,
        stride_xk,
        stride_w0,
        stride_w1,
        stride_s0,
        stride_s1,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """BF16 tensor-core GEMM which dequantizes one FP8 block at a time.

        BLOCK_N and BLOCK_K are the checkpoint scale block dimensions. One FP32
        inverse scale applies to every weight in each dot-product fragment and
        can be applied to the FP32 accumulator fragment without materializing
        a dequantized matrix.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, reduction_size, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            activation = tl.load(
                x + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < rows) & (offs_k[None, :] < reduction_size),
                other=0.0,
            ).to(tl.bfloat16)
            weight_fragment = tl.load(
                weight + offs_n[None, :] * stride_w0 + offs_k[:, None] * stride_w1,
                mask=(offs_n[None, :] < output_size)
                & (offs_k[:, None] < reduction_size),
                other=0.0,
            ).to(tl.bfloat16)
            inverse_scale = tl.load(
                scale + (pid_n * stride_s0) + ((k_start // 128) * stride_s1)
            )
            block_product = tl.dot(
                activation,
                weight_fragment,
                out_dtype=tl.float32,
            )
            accumulator = accumulator + block_product * inverse_scale

        tl.store(
            out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < output_size),
        )

    @triton.jit
    def _w8a16_input_grad_kernel(
        grad,
        weight,
        scale,
        out,
        rows,
        input_size: tl.constexpr,
        output_size: tl.constexpr,
        stride_gm,
        stride_gk,
        stride_wk,
        stride_wn,
        stride_sk,
        stride_sn,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute dX from BF16 dY and checkpoint-layout block-FP8 W."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, output_size, BLOCK_K, num_stages=2):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            grad_fragment = tl.load(
                grad + offs_m[:, None] * stride_gm + offs_k[None, :] * stride_gk,
                mask=(offs_m[:, None] < rows) & (offs_k[None, :] < output_size),
                other=0.0,
            ).to(tl.bfloat16)
            weight_fragment = tl.load(
                weight + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=(offs_k[:, None] < output_size)
                & (offs_n[None, :] < input_size),
                other=0.0,
            ).to(tl.bfloat16)
            inverse_scale = tl.load(
                scale + (k_start // 128) * stride_sk + pid_n * stride_sn
            ).to(tl.bfloat16)
            dequantized_weight = (weight_fragment * inverse_scale).to(tl.bfloat16)
            accumulator += tl.dot(grad_fragment, dequantized_weight)

        tl.store(
            out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < input_size),
        )

    @triton.jit
    def _grouped_outer_accumulate_kernel(
        left,
        right,
        offsets,
        output,
        left_size: tl.constexpr,
        right_size: tl.constexpr,
        stride_lm,
        stride_ln,
        stride_rm,
        stride_rn,
        stride_oe,
        stride_ol,
        stride_or,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_LEFT: tl.constexpr,
        BLOCK_RIGHT: tl.constexpr,
    ):
        expert = tl.program_id(0)
        left_block = tl.program_id(1)
        right_block = tl.program_id(2)
        expert_start = tl.load(
            offsets + expert - 1,
            mask=expert > 0,
            other=0,
        )
        expert_end = tl.load(offsets + expert)
        left_columns = left_block * BLOCK_LEFT + tl.arange(0, BLOCK_LEFT)
        right_columns = right_block * BLOCK_RIGHT + tl.arange(0, BLOCK_RIGHT)
        accumulator = tl.zeros((BLOCK_LEFT, BLOCK_RIGHT), dtype=tl.float32)

        for row_start in tl.range(expert_start, expert_end, BLOCK_ROWS):
            row_indices = row_start + tl.arange(0, BLOCK_ROWS)
            row_mask = row_indices < expert_end
            left_values = tl.load(
                left
                + row_indices[:, None] * stride_lm
                + left_columns[None, :] * stride_ln,
                mask=row_mask[:, None] & (left_columns[None, :] < left_size),
                other=0.0,
            ).to(tl.bfloat16)
            right_values = tl.load(
                right
                + row_indices[:, None] * stride_rm
                + right_columns[None, :] * stride_rn,
                mask=row_mask[:, None] & (right_columns[None, :] < right_size),
                other=0.0,
            ).to(tl.bfloat16)
            accumulator += tl.dot(tl.trans(left_values), right_values)

        output_offsets = (
            expert * stride_oe
            + left_columns[:, None] * stride_ol
            + right_columns[None, :] * stride_or
        )
        output_mask = (left_columns[:, None] < left_size) & (
            right_columns[None, :] < right_size
        )
        previous = tl.load(output + output_offsets, mask=output_mask, other=0.0)
        tl.store(output + output_offsets, previous + accumulator, mask=output_mask)

else:
    _block_fp8_mm_kernel = None
    _w8a16_input_grad_kernel = None
    _grouped_outer_accumulate_kernel = None


def set_diagnose_linear_backward(enabled: bool) -> None:
    global _DIAGNOSE_LINEAR_BACKWARD
    _DIAGNOSE_LINEAR_BACKWARD = enabled


def _native_block_fp8_mm(x, weight, scale):
    """Compute x @ weight.T from output-major block-E4M3 weights."""
    if _block_fp8_mm_kernel is None:
        raise RuntimeError("Triton is unavailable")
    if x.ndim != 2 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("native block-FP8 GEMM requires 2-D tensors")
    reduction_size = x.shape[1]
    output_size = weight.shape[0]
    expected_reduction = weight.shape[1]
    expected_scale = (
        (output_size + 127) // 128,
        (reduction_size + 127) // 128,
    )
    if reduction_size != expected_reduction:
        raise ValueError(
            f"reduction mismatch: activation {reduction_size}, weight {tuple(weight.shape)}"
        )
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not cover weight "
            f"{tuple(weight.shape)}; expected {expected_scale}"
        )
    out = torch.empty(x.shape[0], output_size, device=x.device, dtype=x.dtype)
    # Small LoRA projections need a short M tile. Long context shards need a
    # taller tile so each program reuses the same checkpoint weight fragment
    # across more token rows instead of reloading it for every 16 rows.
    if x.shape[0] <= 64:
        block_m, num_warps = 16, 4
    elif x.shape[0] <= 512:
        block_m, num_warps = 32, 4
    elif x.shape[0] <= 2048:
        block_m, num_warps = 64, 8
    else:
        block_m, num_warps = 128, 8
    grid = (
        (x.shape[0] + block_m - 1) // block_m,
        (output_size + 127) // 128,
    )
    _block_fp8_mm_kernel[grid](
        x,
        weight,
        scale,
        out,
        x.shape[0],
        output_size,
        reduction_size,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def _native_w8a16_input_grad(grad, weight, scale):
    """Compute dX with BF16 activations and block-FP8 checkpoint weights."""
    if _w8a16_input_grad_kernel is None:
        raise RuntimeError("Triton is unavailable")
    if grad.ndim != 2 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("native W8A16 input gradient requires 2-D tensors")
    if grad.dtype != torch.bfloat16:
        raise TypeError(f"W8A16 gradient must be BF16, got {grad.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"W8A16 weight must be FP8, got {weight.dtype}")
    if scale.dtype != torch.float32:
        raise TypeError(f"W8A16 scale must be FP32, got {scale.dtype}")
    if not (grad.is_cuda and weight.is_cuda and scale.is_cuda):
        raise ValueError("native W8A16 input gradient requires CUDA tensors")
    if grad.device != weight.device or grad.device != scale.device:
        raise ValueError("W8A16 gradient, weight, and scale must share one device")
    output_size, input_size = weight.shape
    if grad.shape[1] != output_size:
        raise ValueError(
            f"reduction mismatch: gradient {grad.shape[1]}, weight {tuple(weight.shape)}"
        )
    expected_scale = (
        (output_size + 127) // 128,
        (input_size + 127) // 128,
    )
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not cover weight "
            f"{tuple(weight.shape)}; expected {expected_scale}"
        )

    out = torch.empty(
        grad.shape[0], input_size, device=grad.device, dtype=torch.bfloat16
    )
    if grad.shape[0] <= 64:
        block_m, num_warps = 16, 4
    elif grad.shape[0] <= 512:
        block_m, num_warps = 32, 4
    elif grad.shape[0] <= 2048:
        block_m, num_warps = 64, 8
    else:
        block_m, num_warps = 128, 8
    grid = (
        triton.cdiv(grad.shape[0], block_m),
        triton.cdiv(input_size, 128),
    )
    stream = torch.cuda.current_stream(grad.device)
    grad.record_stream(stream)
    weight.record_stream(stream)
    scale.record_stream(stream)
    _w8a16_input_grad_kernel[grid](
        grad,
        weight,
        scale,
        out,
        grad.shape[0],
        input_size,
        output_size,
        grad.stride(0),
        grad.stride(1),
        weight.stride(0),
        weight.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=num_warps,
    )
    return out


def _native_sm89_eligible(activation, weight, scale):
    if (
        not activation.is_cuda
        or not weight.is_cuda
        or not scale.is_cuda
        or activation.device != weight.device
        or activation.device != scale.device
    ):
        return False, "not-cuda"
    if torch.cuda.get_device_capability(activation.device) != (8, 9):
        return False, "not-sm89"
    if activation.dtype not in (torch.bfloat16, torch.float16):
        return False, "activation-dtype"
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        return False, "checkpoint-dtype"
    if _block_fp8_mm_kernel is None:
        return False, "triton-unavailable"
    return True, ""


def _linear(x, weight, scale):
    shape = (*x.shape[:-1], weight.shape[0])
    x = x.reshape(-1, x.shape[-1]).contiguous()
    # Use the checkpoint block-FP8 W8A8 kernel for every production forward.
    # The old short-row branch converted both operands to BF16 inside Triton.
    # It was accurate, but it was not an FP8 forward and introduced a precision
    # change at an arbitrary row-count boundary.
    offsets = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int32)
    return _grouped_mm(x, weight.unsqueeze(0), scale.unsqueeze(0), offsets).view(shape)


def _w8a16_linear(x, weight, scale):
    if x.dtype != torch.bfloat16:
        raise TypeError(f"W8A16 input must be BF16, got {x.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"W8A16 weight must be FP8, got {weight.dtype}")
    if scale.dtype != torch.float32:
        raise TypeError(f"W8A16 scale must be FP32, got {scale.dtype}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"W8A16 input width {x.shape[-1]} does not match weight width {weight.shape[1]}"
        )
    shape = (*x.shape[:-1], weight.shape[0])
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    return _native_block_fp8_mm(x_2d, weight, scale).view(shape)


def _w8a16_input_grad(
    grad, weight, scale, *, fqn="<unassigned-fp8-linear>"
):
    if grad.dtype != torch.bfloat16:
        raise TypeError(f"W8A16 gradient must be BF16, got {grad.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"W8A16 weight must be FP8, got {weight.dtype}")
    if scale.dtype != torch.float32:
        raise TypeError(f"W8A16 scale must be FP32, got {scale.dtype}")
    if grad.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"W8A16 gradient width {grad.shape[-1]} does not match "
            f"weight output width {weight.shape[0]}"
        )
    shape = (*grad.shape[:-1], weight.shape[1])
    grad_2d = grad.reshape(-1, grad.shape[-1]).contiguous()
    input_grad = _native_w8a16_input_grad(grad_2d, weight, scale).view(shape)
    if (
        _DIAGNOSE_LINEAR_BACKWARD
        or os.environ.get("PRIME_FP8_DIAG_LINEAR_BACKWARD", "") == "1"
    ):
        incoming_finite = bool(torch.isfinite(grad).all())
        output_finite = bool(torch.isfinite(input_grad).all())
        report_key = (fqn, incoming_finite, output_finite)
        if (
            (not incoming_finite or not output_finite)
            and report_key not in _REPORTED_W8A16_INPUT_GRAD
        ):
            _REPORTED_W8A16_INPUT_GRAD.add(report_key)
            finite_grad = grad[torch.isfinite(grad)]
            finite_output = input_grad[torch.isfinite(input_grad)]
            print(
                "[fp8-w8a16-input-grad] "
                f"fqn={fqn} weight={tuple(weight.shape)} grad={tuple(grad.shape)} "
                f"incoming_finite={incoming_finite} "
                f"incoming_absmax={float(finite_grad.abs().max()) if finite_grad.numel() else float('nan'):.6e} "
                f"output_finite={output_finite} "
                f"output_absmax={float(finite_output.abs().max()) if finite_output.numel() else float('nan'):.6e}",
                flush=True,
            )
    return input_grad


def _linear_input_grad_oracle(
    grad, weight, scale, chunk_rows=1024, fqn="<unassigned-fp8-linear>"
):
    """Compute dX = dY @ dequant(W) without a transposed FP8 kernel.

    The forward fused kernel accepts checkpoint layout [out, in]. Its
    transposed-weight use in backward is not a supported checkpoint layout and
    produced non-finite LoRA gradients on the 64-GPU Ada job. Dequantize bounded
    row chunks and accumulate in FP32 instead.
    """
    shape = (*grad.shape[:-1], weight.shape[1])
    grad_2d = grad.reshape(-1, grad.shape[-1])
    compute_dtype = (
        torch.float64
        if grad.dtype == torch.float64 or weight.dtype == torch.float64
        else torch.float32
    )
    result = torch.zeros(
        grad_2d.shape[0],
        weight.shape[1],
        device=grad.device,
        dtype=compute_dtype,
    )
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(start + chunk_rows, weight.shape[0])
        dequantized = _dequant_weight(weight, scale, start, end, compute_dtype)
        result.addmm_(grad_2d[:, start:end].to(compute_dtype), dequantized)
    converted = result.to(grad.dtype)
    if (
        (
            _DIAGNOSE_LINEAR_BACKWARD
            or os.environ.get("PRIME_FP8_DIAG_LINEAR_BACKWARD", "") == "1"
        )
        and not bool(torch.isfinite(converted).all())
        and fqn not in _REPORTED_INPUT_GRAD
    ):
        _REPORTED_INPUT_GRAD.add(fqn)
        finite_result = result[torch.isfinite(result)]
        weight_float = weight.float()
        finite_weight = weight_float[torch.isfinite(weight_float)]
        print(
            "[fp8-linear-input-grad] "
            f"fqn={fqn} weight={tuple(weight.shape)} weight_finite={bool(torch.isfinite(weight_float).all())} "
            f"weight_absmax={float(finite_weight.abs().max()) if finite_weight.numel() else float('nan'):.6e} "
            f"scale_finite={bool(torch.isfinite(scale).all())} "
            f"scale_min={float(scale[torch.isfinite(scale)].min()):.6e} "
            f"scale_max={float(scale[torch.isfinite(scale)].max()):.6e} "
            f"result_finite={bool(torch.isfinite(result).all())} "
            f"result_nan={int(torch.isnan(result).sum())} result_inf={int(torch.isinf(result).sum())} "
            f"result_absmax={float(finite_result.abs().max()) if finite_result.numel() else float('nan'):.6e} "
            f"converted_nan={int(torch.isnan(converted).sum())} converted_inf={int(torch.isinf(converted).sum())}",
            flush=True,
        )
    return converted.view(shape)


def _linear_input_grad(
    grad, weight, scale, chunk_rows=1024, fqn="<unassigned-fp8-linear>"
):
    """Compute dX with W8A8 block-FP8 GEMM, or use the FP32 oracle.

    The checkpoint stores W as ``[out, in]`` with a ``[out/128, in/128]``
    inverse-scale grid. Transposing both views gives the exact block metadata
    required for ``dY @ W``. The fused grouped kernel dynamically quantizes dY
    and performs the matrix multiply on FP8 tensor cores. This replaces the
    previous SM89 path, which converted each weight fragment to BF16 before
    ``tl.dot`` and therefore was not an FP8 backward.
    """
    eligible, _ = _native_sm89_eligible(grad, weight, scale)
    if eligible:
        shape = (*grad.shape[:-1], weight.shape[1])
        grad_2d = grad.reshape(-1, grad.shape[-1]).contiguous()
        offsets = torch.tensor(
            [grad_2d.shape[0]], device=grad.device, dtype=torch.int32
        )
        return _grouped_mm(
            grad_2d,
            weight.t().unsqueeze(0),
            scale.t().unsqueeze(0),
            offsets,
        ).view(shape)
    return _linear_input_grad_oracle(
        grad, weight, scale, chunk_rows=chunk_rows, fqn=fqn
    )


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, fqn, owner):
        # FSDP2 releases the unsharded frozen parameters after forward. A saved
        # tensor alias continues to point at that released allocation. An
        # owned production module can reacquire the current storage after the
        # outer block pre-backward hook unshards it. Direct function calls have
        # no owner, so they keep a bounded fallback snapshot instead.
        ctx.owner = owner
        if owner is not None:
            # The decoder block is the FSDP unit. Its output pre-backward hook
            # restores current parameter storage before an inner backward.
            # Keep the owner, not this forward's view, because resharding
            # releases that view.
            ctx.weight_snapshot = None
            ctx.scale_snapshot = None
            ctx.owns_snapshot = False
            ctx.snapshot_key = None
        elif in_reentrant_block_checkpoint():
            # A whole-block reentrant checkpoint starts recompute only after
            # the matching FSDP pre-backward hook unshards this block. FSDP
            # keeps the parameters valid until the block backward completes,
            # so these inner linear contexts can borrow the restored storage.
            # The first no-grad checkpoint call also discards its contexts.
            ctx.weight_snapshot = weight
            ctx.scale_snapshot = scale
            ctx.owns_snapshot = False
            ctx.snapshot_key = None
        elif ctx.needs_input_grad[0]:
            ctx.weight_snapshot, ctx.scale_snapshot = _acquire_linear_snapshot(
                fqn, weight, scale
            )
            ctx.owns_snapshot = True
            ctx.snapshot_key = fqn
        else:
            # Autograd will not call backward when the activation does not
            # require a gradient. Do not retain an owned weight in that case.
            ctx.weight_snapshot = weight
            ctx.scale_snapshot = scale
            ctx.owns_snapshot = False
            ctx.snapshot_key = None
        ctx.fqn = fqn
        try:
            return _linear(x, weight, scale)
        except BaseException:
            if ctx.snapshot_key is not None:
                _release_linear_snapshot(ctx.snapshot_key)
                ctx.snapshot_key = None
            raise

    @staticmethod
    def backward(ctx, grad):
        if ctx.owner is not None:
            owner, weight_name, scale_name = ctx.owner
            weight = getattr(owner, weight_name)
            scale = getattr(owner, scale_name)
            if isinstance(weight, DTensor):
                weight = weight.to_local()
            if isinstance(scale, DTensor):
                scale = scale.to_local()
        else:
            weight = ctx.weight_snapshot
            scale = ctx.scale_snapshot
        try:
            input_grad = _linear_input_grad(grad, weight, scale, fqn=ctx.fqn)
        finally:
            if ctx.snapshot_key is not None:
                _release_linear_snapshot(ctx.snapshot_key)
            del ctx.weight_snapshot, ctx.scale_snapshot, ctx.owner
        if (
            _DIAGNOSE_LINEAR_BACKWARD
            or os.environ.get("PRIME_FP8_DIAG_LINEAR_BACKWARD", "") == "1"
        ):
            incoming_finite = bool(torch.isfinite(grad).all())
            output_finite = bool(torch.isfinite(input_grad).all())
            report_key = (ctx.fqn, incoming_finite, output_finite)
            if (
                not incoming_finite or not output_finite
            ) and report_key not in _REPORTED_LINEAR_BACKWARD:
                _REPORTED_LINEAR_BACKWARD.add(report_key)
                finite_grad = grad[torch.isfinite(grad)]
                finite_output = input_grad[torch.isfinite(input_grad)]
                print(
                    "[fp8-linear-bwd] "
                    f"fqn={ctx.fqn} weight={tuple(weight.shape)} "
                    f"grad={tuple(grad.shape)} "
                    f"incoming_finite={incoming_finite} "
                    f"incoming_absmax={float(finite_grad.abs().max()) if finite_grad.numel() else float('nan'):.6e} "
                    f"output_finite={output_finite} "
                    f"output_absmax={float(finite_output.abs().max()) if finite_output.numel() else float('nan'):.6e}",
                    flush=True,
                )
        return input_grad, None, None, None, None


class _W8A16Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, fqn, owner):
        ctx.owner = owner
        ctx.fqn = fqn
        if owner is not None:
            ctx.weight_snapshot = None
            ctx.scale_snapshot = None
            ctx.owns_snapshot = False
            ctx.snapshot_key = None
        elif in_reentrant_block_checkpoint():
            ctx.weight_snapshot = weight
            ctx.scale_snapshot = scale
            ctx.owns_snapshot = False
            ctx.snapshot_key = None
        elif ctx.needs_input_grad[0]:
            ctx.weight_snapshot, ctx.scale_snapshot = _acquire_linear_snapshot(
                fqn, weight, scale
            )
            ctx.owns_snapshot = True
            ctx.snapshot_key = fqn
        else:
            ctx.weight_snapshot = weight
            ctx.scale_snapshot = scale
            ctx.owns_snapshot = False
            ctx.snapshot_key = None
        try:
            return _w8a16_linear(x, weight, scale)
        except BaseException:
            if ctx.snapshot_key is not None:
                _release_linear_snapshot(ctx.snapshot_key)
                ctx.snapshot_key = None
            raise

    @staticmethod
    def backward(ctx, grad):
        if ctx.owner is not None:
            owner, weight_name, scale_name = ctx.owner
            weight = getattr(owner, weight_name)
            scale = getattr(owner, scale_name)
            if isinstance(weight, DTensor):
                weight = weight.to_local()
            if isinstance(scale, DTensor):
                scale = scale.to_local()
        else:
            weight = ctx.weight_snapshot
            scale = ctx.scale_snapshot
        try:
            input_grad = _w8a16_input_grad(grad, weight, scale, fqn=ctx.fqn)
            if ctx.owner is not None:
                fence = getattr(ctx.owner[0], "_w8a16_backward_fence", None)
                if fence is not None:
                    fence.record(input_grad.device)
        finally:
            if ctx.snapshot_key is not None:
                _release_linear_snapshot(ctx.snapshot_key)
            del ctx.weight_snapshot, ctx.scale_snapshot, ctx.owner
        return input_grad, None, None, None, None


class _FusedQKVALinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, fqn, owner):
        ctx.owner = owner
        ctx.fqn = fqn
        if owner is not None:
            ctx.weight_snapshot = None
            ctx.scale_snapshot = None
            ctx.snapshot_key = None
        elif in_reentrant_block_checkpoint():
            ctx.weight_snapshot = weight
            ctx.scale_snapshot = scale
            ctx.snapshot_key = None
        elif ctx.needs_input_grad[0]:
            ctx.weight_snapshot, ctx.scale_snapshot = _acquire_linear_snapshot(
                fqn, weight, scale
            )
            ctx.snapshot_key = fqn
        else:
            ctx.weight_snapshot = weight
            ctx.scale_snapshot = scale
            ctx.snapshot_key = None
        try:
            return _linear(x, weight, scale)
        except BaseException:
            if ctx.snapshot_key is not None:
                _release_linear_snapshot(ctx.snapshot_key)
                ctx.snapshot_key = None
            raise

    @staticmethod
    def backward(ctx, grad):
        if ctx.owner is not None:
            owner, weight_name, scale_name = ctx.owner
            weight = getattr(owner, weight_name)
            scale = getattr(owner, scale_name)
            if isinstance(weight, DTensor):
                weight = weight.to_local()
            if isinstance(scale, DTensor):
                scale = scale.to_local()
        else:
            weight = ctx.weight_snapshot
            scale = ctx.scale_snapshot
        try:
            input_grad = _w8a16_input_grad(grad, weight, scale, fqn=ctx.fqn)
            if ctx.owner is not None:
                fence = getattr(ctx.owner[0], "_w8a16_backward_fence", None)
                if fence is not None:
                    fence.record(input_grad.device)
        finally:
            if ctx.snapshot_key is not None:
                _release_linear_snapshot(ctx.snapshot_key)
            del ctx.weight_snapshot, ctx.scale_snapshot, ctx.owner
        return input_grad, None, None, None, None


class FP8CheckpointLinear(nn.Linear):
    def forward(self, x):
        fqn = getattr(self, "_fp8_fqn", f"<unassigned-fp8-linear:{id(self)}>")
        if getattr(self, "w8a16", False):
            out = _W8A16Linear.apply(
                x,
                self.weight,
                self.weight_scale_inv,
                fqn,
                (self, "weight", "weight_scale_inv"),
            )
            return out if self.bias is None else out + self.bias
        out = _Linear.apply(
            x,
            self.weight,
            self.weight_scale_inv,
            fqn,
            (self, "weight", "weight_scale_inv"),
        )
        return out if self.bias is None else out + self.bias

    @classmethod
    def from_linear(cls, linear, *, w8a16=False):
        with torch.device("meta"):
            module = cls(
                linear.in_features,
                linear.out_features,
                bias=linear.bias is not None,
            )
        module.weight = nn.Parameter(
            torch.empty_like(linear.weight, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module.weight_scale_inv = nn.Parameter(
            torch.empty(
                linear.out_features // 128,
                linear.in_features // 128,
                device=linear.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        module.bias = linear.bias
        module.w8a16 = bool(w8a16)
        return module


class FP8CheckpointFusedQKVA(nn.Module):
    input_size = 6144
    q_output_size = 2048
    kv_output_size = 576

    def __init__(self, *, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.in_features = self.input_size
        self.out_features = self.q_output_size
        self.bias = None
        output_size = self.q_output_size + self.kv_output_size
        self.weight = nn.Parameter(
            torch.empty(
                output_size,
                self.input_size,
                device=device,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.empty(
                math.ceil(output_size / 128),
                self.input_size // 128,
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @classmethod
    def from_linears(cls, q_projection: nn.Linear, kv_projection: nn.Linear):
        if q_projection.bias is not None or kv_projection.bias is not None:
            raise ValueError("fused GLM Q-A and KV-A projections do not support bias")
        expected = (
            cls.input_size,
            cls.q_output_size,
            cls.kv_output_size,
        )
        actual = (
            q_projection.in_features,
            q_projection.out_features,
            kv_projection.out_features,
        )
        if (
            kv_projection.in_features != q_projection.in_features
            or actual != expected
        ):
            raise ValueError(
                "fused GLM Q-A and KV-A shapes must be "
                f"input={expected[0]}, q={expected[1]}, kv={expected[2]}; got "
                f"q={tuple(q_projection.weight.shape)}, "
                f"kv={tuple(kv_projection.weight.shape)}"
            )
        return cls(device=q_projection.weight.device)

    def combined(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise TypeError(
                f"fused GLM Q-A and KV-A input must be BF16, got {x.dtype}"
            )
        if self.weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "fused GLM Q-A and KV-A weight must be FP8 E4M3FN, "
                f"got {self.weight.dtype}"
            )
        if self.weight_scale_inv.dtype != torch.float32:
            raise TypeError(
                "fused GLM Q-A and KV-A scales must be FP32, "
                f"got {self.weight_scale_inv.dtype}"
            )
        if self.weight.device != x.device or self.weight_scale_inv.device != x.device:
            raise ValueError(
                "fused GLM Q-A and KV-A input, weight, and scales must use one device"
            )
        expected_weight = (
            self.q_output_size + self.kv_output_size,
            self.input_size,
        )
        expected_scale = (
            math.ceil(expected_weight[0] / 128),
            self.input_size // 128,
        )
        if tuple(self.weight.shape) != expected_weight:
            raise ValueError(
                f"fused GLM Q-A and KV-A weight has shape {tuple(self.weight.shape)}; "
                f"expected {expected_weight}"
            )
        if tuple(self.weight_scale_inv.shape) != expected_scale:
            raise ValueError(
                "fused GLM Q-A and KV-A scale has shape "
                f"{tuple(self.weight_scale_inv.shape)}; expected {expected_scale}"
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"fused GLM Q-A and KV-A input width is {x.shape[-1]}; "
                f"expected {self.input_size}"
            )
        fqn = getattr(self, "_fp8_fqn", "<unassigned-fused-qkv-a>")
        return _FusedQKVALinear.apply(
            x,
            self.weight,
            self.weight_scale_inv,
            fqn,
            (self, "weight", "weight_scale_inv"),
        )

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.combined(x).split(
            (self.q_output_size, self.kv_output_size), dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.split(x)[0]


class FP8CheckpointFusedQKVASlice(nn.Module):
    def __init__(self, owner: FP8CheckpointFusedQKVA) -> None:
        super().__init__()
        object.__setattr__(self, "_owner_ref", weakref.ref(owner))
        self.in_features = owner.input_size
        self.out_features = owner.kv_output_size
        self.bias = None

    @property
    def owner(self) -> FP8CheckpointFusedQKVA:
        owner = self._owner_ref()
        if owner is None:
            raise RuntimeError("fused GLM Q-A and KV-A owner no longer exists")
        return owner

    @property
    def weight(self) -> torch.Tensor:
        return self.owner.weight.narrow(
            0, self.owner.q_output_size, self.owner.kv_output_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.owner.split(x)[1]


def replace_glm_qkv_a(model: nn.Module) -> int:
    if getattr(getattr(model, "config", None), "model_type", None) != "glm_moe_dsa":
        raise ValueError("fused Q-A and KV-A FP8 is only valid for GLM-MoE-DSA")
    replaced = 0
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] != "q_a_proj":
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"{name} must be nn.Linear before fused FP8 replacement")
        parent_name = name.rsplit(".", 1)[0]
        parent = model.get_submodule(parent_name)
        kv_projection = getattr(parent, "kv_a_proj_with_mqa", None)
        if not isinstance(kv_projection, nn.Linear):
            raise TypeError(
                f"{parent_name}.kv_a_proj_with_mqa must be nn.Linear before "
                "fused FP8 replacement"
            )
        fused = FP8CheckpointFusedQKVA.from_linears(module, kv_projection)
        fused._fp8_fqn = name
        parent.q_a_proj = fused
        parent.kv_a_proj_with_mqa = FP8CheckpointFusedQKVASlice(fused)
        replaced += 1
    expected = sum(
        1
        for module in model.modules()
        if type(module).__name__ == "GlmMoeDsaDecoderLayer"
    )
    if expected == 0 or replaced != expected:
        raise ValueError(
            f"fused Q-A and KV-A FP8 replaced {replaced} pairs; expected {expected}"
        )
    return replaced


def install_w8a16_reshard_wait(module: nn.Module) -> bool:
    linears = [
        child
        for child in module.modules()
        if (
            isinstance(child, FP8CheckpointLinear)
            and getattr(child, "w8a16", False)
        )
        or isinstance(child, FP8CheckpointFusedQKVA)
    ]
    if not linears:
        return False
    state = module._get_fsdp_state()
    group = state._fsdp_param_group
    if group is None:
        raise RuntimeError("W8A16 reshard wait requires an FSDP parameter group")
    if getattr(group, "_prime_w8a16_reshard_wait", False):
        raise RuntimeError("W8A16 reshard wait is already installed")

    fence = W8A16BackwardFence()
    for linear in linears:
        linear._w8a16_backward_fence = fence
    original_post_backward = group.post_backward

    def post_backward(*args, **kwargs):
        fence.wait_before_reshard(group)
        return original_post_backward(*args, **kwargs)

    group.post_backward = post_backward
    group._prime_w8a16_fence = fence
    group._prime_w8a16_reshard_wait = True
    return True


class FP8CheckpointEmbedding(nn.Embedding):
    def forward(self, input):
        quantized = (
            self.weight.view(torch.float8_e4m3fn)
            .view(self.num_embeddings, self.embedding_dim)
            .index_select(0, input.flatten())
            .view(*input.shape, self.embedding_dim)
            .to(torch.bfloat16)
        )
        scales = self.weight_scale_inv.index_select(0, input.flatten()).view(
            input.shape
        )
        return quantized * scales.to(quantized.dtype).unsqueeze(-1)

    @classmethod
    def from_embedding(cls, embedding):
        with torch.device("meta"):
            module = cls(
                embedding.num_embeddings,
                embedding.embedding_dim,
                embedding.padding_idx,
                embedding.max_norm,
                embedding.norm_type,
                embedding.scale_grad_by_freq,
                embedding.sparse,
            )
        module.weight = nn.Parameter(
            torch.empty(
                embedding.num_embeddings,
                embedding.embedding_dim // 2,
                device=embedding.weight.device,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        module.weight_scale_inv = nn.Parameter(
            torch.empty(
                embedding.num_embeddings,
                device=embedding.weight.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        return module


def _dequant_weight(weight, scale, start, end, dtype):
    if not (0 <= start <= end <= weight.shape[0]):
        raise ValueError(
            f"invalid row range [{start}, {end}) for weight {tuple(weight.shape)}"
        )
    expected = (
        (weight.shape[0] + 127) // 128,
        (weight.shape[1] + 127) // 128,
    )
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not cover "
            f"weight {tuple(weight.shape)}; expected {expected}"
        )
    if start % 128 == 0 and end % 128 == 0 and weight.shape[1] % 128 == 0:
        return (
            weight[start:end]
            .to(dtype)
            .view(-1, 128, weight.shape[1] // 128, 128)
            .mul(scale[start // 128 : end // 128].to(dtype)[:, None, :, None])
            .flatten(0, 1)
            .flatten(1, 2)
        )
    rows = torch.arange(start, end, device=scale.device) // 128
    columns = torch.arange(weight.shape[1], device=scale.device) // 128
    expanded_scale = scale.index_select(0, rows).index_select(1, columns)
    return weight[start:end].to(dtype).mul(expanded_scale.to(dtype))


def _scaled_mm(x, weight, weight_scale):
    scale = (
        x.float()
        .abs()
        .amax(1)
        .clamp_min(1e-12)
        .div(torch.finfo(torch.float8_e4m3fn).max)
    )
    quantized = (
        x.float()
        .div(scale[:, None])
        .clamp(
            torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max
        )
        .to(torch.float8_e4m3fn)
    )
    matrix = weight.t()
    if matrix.stride(0) != 1:
        matrix = matrix.t().contiguous().t()
    if weight_scale is None:
        weight_scale = torch.ones(matrix.shape[1], device=x.device, dtype=torch.float32)
    return torch._scaled_mm(
        quantized,
        matrix,
        scale[:, None],
        weight_scale[None],
        out_dtype=x.dtype,
        use_fast_accum=True,
    )


class _FP8LogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, scale, labels, inv_temperature, chunk_size):
        n = hidden.shape[0]
        vocab = weight.shape[0]
        logprobs = torch.empty(n, device=hidden.device, dtype=torch.float32)
        entropy = torch.empty_like(logprobs)
        logz = torch.empty_like(logprobs)
        mean_scaled = torch.empty_like(logprobs)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t = inv_temperature[start:end].unsqueeze(-1)
            m = torch.full(
                (end - start,), float("-inf"), device=hidden.device, dtype=torch.float32
            )
            s = torch.zeros_like(m)
            t = torch.zeros_like(m)
            target = torch.zeros_like(m)
            for vocab_start in range(0, vocab, 16384):
                vocab_end = min(vocab_start + 16384, vocab)
                logits = _scaled_mm(
                    hidden_chunk,
                    weight[vocab_start:vocab_end],
                    scale[vocab_start:vocab_end],
                )
                scaled = logits.float() * inv_t
                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled)
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                row = torch.arange(end - start, device=hidden.device)
                index = (
                    (labels_chunk - vocab_start)
                    .clamp(0, vocab_end - vocab_start - 1)
                    .long()
                )
                target = torch.where(mask, scaled[row, index], target)
            logz[start:end] = m + torch.log(s)
            logprobs[start:end] = target - logz[start:end]
            mean_scaled[start:end] = t / s
            entropy[start:end] = logz[start:end] - mean_scaled[start:end]
        ctx.save_for_backward(
            hidden, weight, scale, labels, inv_temperature, logz, mean_scaled
        )
        ctx.chunk_size = chunk_size
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, grad_entropy):
        hidden, weight, scale, labels, inv_temperature, logz, mean_scaled = (
            ctx.saved_tensors
        )
        grad_hidden = torch.zeros_like(hidden)
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, hidden.shape[0])
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_logprob_chunk = (
                torch.zeros(end - start, device=hidden.device, dtype=torch.float32)
                if grad_logprobs is None
                else grad_logprobs[start:end].float()
            )
            grad_entropy_chunk = (
                torch.zeros(end - start, device=hidden.device, dtype=torch.float32)
                if grad_entropy is None
                else grad_entropy[start:end].float()
            )
            inv_t = inv_temperature[start:end].unsqueeze(-1)
            for vocab_start in range(0, weight.shape[0], 16384):
                vocab_end = min(vocab_start + 16384, weight.shape[0])
                weight_chunk = weight[vocab_start:vocab_end]
                scale_chunk = scale[vocab_start:vocab_end]
                scaled = (
                    _scaled_mm(hidden_chunk, weight_chunk, scale_chunk).float() * inv_t
                )
                probability = torch.exp(scaled - logz[start:end].unsqueeze(-1))
                # d log p_y / dz = one_hot(y) - p and
                # d H / dz = p * (E[z] - z), where z is the
                # temperature-scaled logit.
                grad_logits = probability * (
                    -grad_logprob_chunk.unsqueeze(-1)
                    + grad_entropy_chunk.unsqueeze(-1)
                    * (mean_scaled[start:end].unsqueeze(-1) - scaled)
                )
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                row = torch.arange(end - start, device=hidden.device)
                index = (
                    (labels_chunk - vocab_start)
                    .clamp(0, vocab_end - vocab_start - 1)
                    .long()
                )
                grad_logits[row, index] += grad_logprob_chunk * mask
                grad_hidden[start:end].add_(
                    _scaled_mm(
                        (grad_logits * inv_t * scale_chunk).to(hidden.dtype),
                        weight_chunk.t(),
                        None,
                    )
                )
        return grad_hidden, None, None, None, None, None


class _AllToAllRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        out = x.new_empty((sum(output_splits),) + x.shape[1:])
        dist.all_to_all_single(
            out, x.contiguous(), output_splits, input_splits, group=group
        )
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.group = group
        return out

    @staticmethod
    def backward(ctx, grad):
        out = grad.new_empty((sum(ctx.input_splits),) + grad.shape[1:])
        dist.all_to_all_single(
            out,
            grad.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            group=ctx.group,
        )
        return out, None, None, None


def _balanced_splits(counts, rank):
    world = len(counts)
    total = sum(counts)
    targets = [total // world + (i < total % world) for i in range(world)]
    source_start = sum(counts[:rank])
    source_end = source_start + counts[rank]
    target_starts = [sum(targets[:i]) for i in range(world)]
    sends = [
        max(0, min(source_end, start + size) - max(source_start, start))
        for start, size in zip(target_starts, targets)
    ]
    target_start = target_starts[rank]
    target_end = target_start + targets[rank]
    source_starts = [sum(counts[:i]) for i in range(world)]
    receives = [
        max(0, min(target_end, start + size) - max(target_start, start))
        for start, size in zip(source_starts, counts)
    ]
    return sends, receives


def _fp8_lm_head_forward(self, hidden_states, labels=None, temperature=None):
    b, s, h = hidden_states.shape
    hidden_states = hidden_states.reshape(b * s, h)
    labels = labels.reshape(b * s)
    temperature = temperature.reshape(b * s)
    valid = labels != -100
    hidden = hidden_states[valid].contiguous()
    target = labels[valid].contiguous()
    inv_temperature = (1.0 / temperature[valid]).contiguous()
    sends = receives = None
    if getattr(self, "cp_group", None) is not None:
        count = torch.tensor([hidden.shape[0]], device=hidden.device)
        counts = [torch.empty_like(count) for _ in range(self.cp_size)]
        dist.all_gather(counts, count, group=self.cp_group)
        sends, receives = _balanced_splits(
            [int(x.item()) for x in counts], self.cp_rank
        )
        hidden = _AllToAllRows.apply(hidden, receives, sends, self.cp_group)
        moved_target = target.new_empty(sum(receives))
        moved_temperature = inv_temperature.new_empty(sum(receives))
        dist.all_to_all_single(
            moved_target, target, receives, sends, group=self.cp_group
        )
        dist.all_to_all_single(
            moved_temperature, inv_temperature, receives, sends, group=self.cp_group
        )
        target = moved_target
        inv_temperature = moved_temperature
    scored, scored_entropy = _FP8LogProb.apply(
        hidden,
        self.weight.view(torch.float8_e4m3fn).view(self.out_features, self.in_features),
        self.weight_scale_inv,
        target,
        inv_temperature,
        self.chunk_size,
    )
    if sends is not None:
        scored = _AllToAllRows.apply(scored, sends, receives, self.cp_group)
        restored_entropy = scored_entropy.new_empty(sum(sends))
        dist.all_to_all_single(
            restored_entropy, scored_entropy, sends, receives, group=self.cp_group
        )
        scored_entropy = restored_entropy
    logprobs = (
        torch.zeros(b * s, device=hidden_states.device, dtype=torch.float32)
        + scored.sum() * 0
    )
    entropy = torch.zeros_like(logprobs) + scored_entropy.sum() * 0
    logprobs[valid], entropy[valid] = scored, scored_entropy
    return PrimeLmOutput(logprobs=logprobs.view(b, s), entropy=entropy.view(b, s))


def validate_w8a16_matches(
    matched_pattern_ids: set[int],
    patterns: tuple[str, ...] | list[str],
    *,
    process_group=None,
) -> None:
    """Require every configured W8A16 pattern locally or across PP stages."""
    if not patterns:
        return

    invalid = set(matched_pattern_ids).difference(range(len(patterns)))
    if invalid:
        raise ValueError(f"invalid W8A16 pattern IDs: {sorted(invalid)}")

    if process_group is None:
        present = set(matched_pattern_ids)
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Pipeline-aware W8A16 validation requires an initialized process group"
            )
        backend = str(dist.get_backend(process_group)).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        present_mask = torch.zeros(len(patterns), dtype=torch.int32, device=device)
        if matched_pattern_ids:
            present_mask[list(matched_pattern_ids)] = 1
        dist.all_reduce(present_mask, op=dist.ReduceOp.MAX, group=process_group)
        present = {
            index
            for index, value in enumerate(present_mask.detach().cpu().tolist())
            if value
        }

    missing = [
        pattern
        for index, pattern in enumerate(patterns)
        if index not in present
    ]
    if missing:
        scope = (
            "configured W8A16 modules were not installed across pipeline stages"
            if process_group is not None
            else "configured W8A16 modules were not installed"
        )
        raise ValueError(f"{scope}: {missing}")


def replace_linear(
    model,
    embed=False,
    lm_head=False,
    ignore_patterns=(),
    w8a16_patterns=(),
    validate_w8a16_patterns=True,
    fused_qkv_a_fp8=False,
):
    if fused_qkv_a_fp8:
        replace_glm_qkv_a(model)
    exclusions = set(model.config.quantization_config["modules_to_not_convert"])
    matched_w8a16 = set()
    for name, module in list(model.named_modules()):
        if (
            not isinstance(module, nn.Linear)
            or name in exclusions
            or name.replace(".mlp.router.gate", ".mlp.gate") in exclusions
            or any(re.search(pattern, name) for pattern in ignore_patterns)
            or name.endswith((".self_attn.kv_b_proj", ".self_attn.indexer.wk"))
        ):
            continue
        if module.in_features % 128 or module.out_features % 128:
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        matched = tuple(
            index
            for index, pattern in enumerate(w8a16_patterns)
            if re.search(pattern, name)
        )
        matched_w8a16.update(matched)
        setattr(
            parent,
            attr,
            FP8CheckpointLinear.from_linear(module, w8a16=bool(matched)),
        )

    if validate_w8a16_patterns:
        validate_w8a16_matches(matched_w8a16, w8a16_patterns)

    # Shape-only diagnostics are ambiguous in GLM because many projections
    # share geometry. Persist the model path on every native checkpoint
    # linear and shared expert so backward reports identify one module.
    for fqn, module in model.named_modules():
        if isinstance(module, FP8CheckpointLinear) or all(
            hasattr(module, attr)
            for attr in ("w1_scale_inv", "w2_scale_inv", "w3_scale_inv")
        ):
            module._fp8_fqn = fqn

    if embed:
        model.model.embed_tokens = FP8CheckpointEmbedding.from_embedding(
            model.model.embed_tokens
        )
    if not lm_head:
        return matched_w8a16
    model.lm_head.weight = nn.Parameter(
        torch.empty(
            model.lm_head.out_features,
            model.lm_head.in_features // 2,
            device=model.lm_head.weight.device,
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    model.lm_head.weight_scale_inv = nn.Parameter(
        torch.empty(
            model.lm_head.out_features,
            device=model.lm_head.weight.device,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    model.lm_head.forward = _fp8_lm_head_forward.__get__(model.lm_head)
    return matched_w8a16


def _prepare_grouped_mm(x, weight, offsets):
    experts, output_size, input_size = weight.shape
    config = get_default_config(
        x.shape[0], experts, output_size, input_size, 1, "fp8_w8a8", block_shape
    )
    rows = torch.arange(x.shape[0], device=x.device)
    ids = (
        torch.searchsorted(offsets, rows, right=True)
        .clamp_max(experts - 1)
        .int()
        .view(-1, 1)
    )
    sorted_ids, expert_ids, padded = _prepare_expert_assignment(
        ids,
        config,
        x.shape[0],
        1,
        experts,
        None,
        block_shape=block_shape,
        ignore_invalid_experts=True,
    )
    return config, rows, sorted_ids, expert_ids, padded


def _quantize_grouped_mm_input(x):
    try:
        return moe_kernel_quantize_input(
            x,
            None,
            torch.float8_e4m3fn,
            False,
            block_shape,
        )
    except torch.OutOfMemoryError:
        # The quantizer allocates its FP8 output before launching its kernel.
        # A routed-row peak can leave enough bytes in unused allocator blocks
        # but no block that fits that output. Release only unused cache and
        # retry the identical quantization once. A live-set OOM still fails on
        # the second call.
        torch.cuda.empty_cache()
        return moe_kernel_quantize_input(
            x,
            None,
            torch.float8_e4m3fn,
            False,
            block_shape,
        )


def _dispatch_grouped_mm(
    x,
    weight,
    scale,
    config,
    rows,
    sorted_ids,
    expert_ids,
    padded,
    quantized,
    activation_scale,
    offsets,
):
    output_size = weight.shape[1]
    try:
        out = torch.empty(
            x.shape[0], 1, output_size, device=x.device, dtype=x.dtype
        )
    except torch.OutOfMemoryError:
        # A routed-row peak can coincide with allocator fragmentation. x72p
        # rank 18 failed a 24 MiB allocation while PyTorch reported 1.16 GiB
        # reserved but unallocated. Release only unused cached blocks and make
        # one bounded retry. A true live-set OOM still fails immediately on
        # the second allocation.
        torch.cuda.empty_cache()
        out = torch.empty(
            x.shape[0], 1, output_size, device=x.device, dtype=x.dtype
        )
    dispatch_fused_moe_kernel(
        quantized,
        weight,
        out,
        activation_scale,
        scale,
        None,
        None,
        sorted_ids,
        expert_ids,
        padded,
        False,
        1,
        config,
        tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16,
        True,
        False,
        False,
        False,
        False,
        block_shape,
    )
    # Expert-parallel dispatch can append rows after the last real token.  The
    # assignment above must give every row an expert id for the fused kernel,
    # but padded rows are not tokens and must not contribute in forward or
    # backward.  The dense expert implementation has the same zero-padding
    # contract.
    result = out[:, 0]
    # ``out`` is new storage owned by this dispatch. An out-of-place mask made
    # a second 118-to-524 MiB expert result at the production shape and OOMed
    # x60l after the inverse-permutation allocation was removed.
    result.masked_fill_((rows >= offsets[-1]).unsqueeze(-1), 0)
    return result


@torch.compiler.disable
def _grouped_mm(x, weight, scale, offsets):
    prepared = _prepare_grouped_mm(x, weight, offsets)
    quantized, activation_scale = _quantize_grouped_mm_input(x)
    return _dispatch_grouped_mm(
        x,
        weight,
        scale,
        *prepared,
        quantized,
        activation_scale,
        offsets,
    )


@torch.compiler.disable
def _grouped_mm_pair(
    x1,
    w1,
    s1,
    x2,
    w2,
    s2,
    offsets,
    *,
    shared_input=False,
):
    """Run two equal-shape expert GEMMs with one assignment preparation.

    W1 and W3 use the same input in forward, so they also reuse one dynamic
    FP8 quantization. Their input-gradient GEMMs use different activations but
    still have identical expert assignments and kernel configurations.
    """
    if not x1.is_cuda:
        return (
            _grouped_mm(x1, w1, s1, offsets),
            _grouped_mm(x2, w2, s2, offsets),
        )
    if w1.shape != w2.shape or x1.shape != x2.shape:
        raise ValueError(
            "paired FP8 expert GEMMs require equal input and weight shapes"
        )
    prepared = _prepare_grouped_mm(x1, w1, offsets)
    quantized1, activation_scale1 = _quantize_grouped_mm_input(x1)
    if shared_input:
        quantized2, activation_scale2 = quantized1, activation_scale1
    else:
        quantized2, activation_scale2 = _quantize_grouped_mm_input(x2)
    return (
        _dispatch_grouped_mm(
            x1,
            w1,
            s1,
            *prepared,
            quantized1,
            activation_scale1,
            offsets,
        ),
        _dispatch_grouped_mm(
            x2,
            w2,
            s2,
            *prepared,
            quantized2,
            activation_scale2,
            offsets,
        ),
    )


def _expert_chunk_offsets(offsets, start, end):
    expert_starts = torch.cat((offsets.new_zeros(1), offsets[:-1]))
    counts = (
        offsets.clamp(min=start, max=end) - expert_starts.clamp(min=start, max=end)
    ).clamp_min_(0)
    return torch.cumsum(counts, 0, dtype=torch.int32)


def _index_select_rows_with_cache_retry(x, indices):
    """Gather one bounded expert row chunk, with one fragmentation retry.

    The production allocator can have enough unused reserved memory in total
    while no cached block is large enough for a contiguous BF16 row gather.
    x74p rank 17 failed a 48 MiB gather with 179 MiB reserved but unused. A
    one-time cache release preserves the bounded-row path and turns that
    allocator-fragmentation case into physically available memory. A true
    live-set OOM still fails on the second allocation.
    """
    try:
        return x.index_select(0, indices)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return x.index_select(0, indices)


def _index_copy_real_rows_with_cache_retry(destination, indices, source, real):
    """Scatter one bounded real-row chunk with one fragmentation retry.

    Keep this on PyTorch's already-loaded indexing path. A custom Triton
    scatter first encountered a new kernel variant during x79p step 4 and
    failed while loading its CUDA module at the full model live set. The
    4,096-row bound limits the boolean source gather to 48 MiB here.
    """
    try:
        destination.index_copy_(0, indices[real].long(), source[real])
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        destination.index_copy_(0, indices[real].long(), source[real])
    return destination


def _grouped_mm_pair_chunks(x, w1, s1, w3, s3, offsets, chunk_rows):
    """Yield bounded W1/W3 outputs with one quantization for each row chunk."""
    for start in range(0, x.shape[0], chunk_rows):
        end = min(start + chunk_rows, x.shape[0])
        chunk_offsets = _expert_chunk_offsets(offsets, start, end)
        z1, z3 = _grouped_mm_pair(
            x[start:end],
            w1,
            s1,
            x[start:end],
            w3,
            s3,
            chunk_offsets,
            shared_input=True,
        )
        yield start, end, z1, z3


def _grouped_mm_chunked(x, weight, scale, offsets, chunk_rows=None):
    """Run a grouped matrix multiply with bounded token-row workspaces.

    The routed-expert W2 projection can start while delayed expert unshards are
    still resident. Splitting only its token rows keeps the fused-kernel
    workspace bounded without changing expert ownership or output order.
    """
    chunk_rows = chunk_rows or _FP8_W2_CHUNK_ROWS
    if chunk_rows <= 0:
        raise ValueError("FP8 W2 chunk rows must be positive")
    if x.shape[0] <= chunk_rows:
        return _grouped_mm(x, weight, scale, offsets)

    out = x.new_empty(x.shape[0], weight.shape[1])
    for start in range(0, x.shape[0], chunk_rows):
        end = min(start + chunk_rows, x.shape[0])
        chunk_offsets = _expert_chunk_offsets(offsets, start, end)
        out[start:end].copy_(
            _grouped_mm(
                x[start:end],
                weight,
                scale,
                chunk_offsets,
            )
        )
    return out


def _silu_glu_backward_inplace(hidden_grad, z1, z3):
    """Return SiLU-GLU input gradients with no extra gate-gradient tensor.

    The inputs and returned tensors are backward-only temporaries. Reuse
    ``hidden_grad`` for the gate gradient and ``z3_grad`` for the other
    product. The operation order matches the original expression. This
    removes the large expression temporaries that caused a random production
    OOM near the 48 GiB limit.
    """
    sigmoid = torch.sigmoid(z1)
    z3_grad = F.silu(z1)
    z3_grad.mul_(hidden_grad)
    hidden_grad.mul_(z3).mul_(sigmoid)
    # z1 <- 1 + z1 * (1 - sigmoid(z1))
    sigmoid.neg_().add_(1)
    z1.mul_(sigmoid).add_(1)
    hidden_grad.mul_(z1)
    return hidden_grad, z3_grad


class _Experts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, s1, w2, s2, w3, s3, offsets, owner):
        offload = _FP8_EXPERT_ACTIVATION_OFFLOAD and x.is_cuda
        if offload:
            # An eager 1F1B stage retains one expert context for each live
            # microbatch. Keeping q1 and q3 on the GPU used several GiB and
            # left x60m unable to allocate even a 48 MiB W2 chunk. Stream the
            # W1/W3 pair, retain its exact per-token FP8 values in pinned host
            # memory, and keep only the small inverse scales on the GPU.
            gated = x.new_empty(x.shape[0], w1.shape[1])
            q1_host = q3_host = None
            qs1_parts = []
            qs3_parts = []
            for start, end, z1, z3 in _grouped_mm_pair_chunks(
                x,
                w1,
                s1,
                w3,
                s3,
                offsets,
                _FP8_W13_CHUNK_ROWS,
            ):
                q1_chunk, qs1_chunk = ops.scaled_fp8_quant(
                    z1, use_per_token_if_dynamic=True
                )
                q3_chunk, qs3_chunk = ops.scaled_fp8_quant(
                    z3, use_per_token_if_dynamic=True
                )
                if q1_host is None:
                    q1_host = torch.empty(
                        x.shape[0],
                        z1.shape[1],
                        device="cpu",
                        dtype=q1_chunk.dtype,
                        pin_memory=True,
                    )
                    q3_host = torch.empty_like(q1_host, pin_memory=True)
                q1_host[start:end].copy_(q1_chunk, non_blocking=True)
                q3_host[start:end].copy_(q3_chunk, non_blocking=True)
                qs1_parts.append(qs1_chunk)
                qs3_parts.append(qs3_chunk)
                F.silu(z1, inplace=True)
                z1.mul_(z3)
                gated[start:end].copy_(z1)
            if q1_host is None or q3_host is None:
                raise RuntimeError("streamed FP8 experts received no token rows")
            q1, q3 = q1_host, q3_host
            qs1 = torch.cat(qs1_parts, dim=0)
            qs3 = torch.cat(qs3_parts, dim=0)
            offload_done = torch.cuda.Event()
            offload_done.record(torch.cuda.current_stream(x.device))
            z1 = gated
        else:
            z1, z3 = _grouped_mm_pair(
                x,
                w1,
                s1,
                x,
                w3,
                s3,
                offsets,
                shared_input=True,
            )
            q1, qs1 = ops.scaled_fp8_quant(z1, use_per_token_if_dynamic=True)
            q3, qs3 = ops.scaled_fp8_quant(z3, use_per_token_if_dynamic=True)
            F.silu(z1, inplace=True)
            z1.mul_(z3)
            del z3
            offload_done = None
        ctx.owner = owner
        ctx.expert_activation_offload = offload
        ctx.offload_done = offload_done
        if owner is None:
            ctx.save_for_backward(q1, qs1, q3, qs3, offsets, w1, s1, w2, s2, w3, s3)
        else:
            # Reacquire current unsharded owner parameters in backward. Saved
            # forward views still point at the released FSDP all-gather
            # allocation after resharding.
            ctx.save_for_backward(q1, qs1, q3, qs3, offsets)
        return _grouped_mm_chunked(z1, w2, s2, offsets)

    @staticmethod
    def backward(ctx, grad):
        if _EXPERT_PROFILE:
            profile_start = torch.cuda.Event(enable_timing=True)
            profile_end = torch.cuda.Event(enable_timing=True)
            profile_start.record()
        if ctx.owner is None:
            q1, qs1, q3, qs3, offsets, w1, s1, w2, s2, w3, s3 = ctx.saved_tensors
        else:
            q1, qs1, q3, qs3, offsets = ctx.saved_tensors
            owner = ctx.owner
            w1, s1, w2, s2, w3, s3 = (
                getattr(owner, name)
                for name in (
                    "w1",
                    "w1_scale_inv",
                    "w2",
                    "w2_scale_inv",
                    "w3",
                    "w3_scale_inv",
                )
            )
            w1, s1, w2, s2, w3, s3 = (
                value.to_local() if isinstance(value, DTensor) else value
                for value in (w1, s1, w2, s2, w3, s3)
            )
        if ctx.expert_activation_offload:
            torch.cuda.current_stream(grad.device).wait_event(ctx.offload_done)
            q1_device = q1.to(grad.device, non_blocking=True)
            z1 = q1_device.to(grad.dtype).mul_(qs1.to(grad.dtype))
            del q1_device
            q3_device = q3.to(grad.device, non_blocking=True)
            z3 = q3_device.to(grad.dtype).mul_(qs3.to(grad.dtype))
            del q3_device
        else:
            z1 = q1.to(grad.dtype).mul_(qs1.to(grad.dtype))
            z3 = q3.to(grad.dtype).mul_(qs3.to(grad.dtype))
        hidden_grad = _grouped_mm_chunked(
            grad, w2.transpose(-2, -1), s2.transpose(-2, -1), offsets
        )
        z1_grad, z3_grad = _silu_glu_backward_inplace(hidden_grad, z1, z3)
        del hidden_grad, z1, z3
        x_grad, x_grad_w3 = _grouped_mm_pair(
            z1_grad,
            w1.transpose(-2, -1),
            s1.transpose(-2, -1),
            z3_grad,
            w3.transpose(-2, -1),
            s3.transpose(-2, -1),
            offsets,
        )
        del z1_grad, z3_grad
        x_grad.add_(x_grad_w3)
        if _EXPERT_PROFILE:
            profile_end.record()
            _profile_expert_backward(profile_start, profile_end)
        del ctx.owner, ctx.offload_done
        return x_grad, None, None, None, None, None, None, None, None


class _InplacePermutedExperts(torch.autograd.Function):
    """Run EP experts without a full source-major permutation allocation.

    The EP permutation is a bijection for real token rows.  Each gathered row
    can therefore be processed once and written back to its original source
    row without changing an input that a later chunk needs.  Reusing the
    routed-input allocation for the source-major result removes one complete
    hidden-size tensor from both forward and backward.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        permuted_indices,
        offsets,
        owner,
        chunk_rows,
    ):
        if not x.is_cuda:
            raise RuntimeError("in-place streamed EP experts require CUDA")
        if chunk_rows <= 0:
            raise ValueError("FP8 EP permutation chunk rows must be positive")

        rows = permuted_indices.shape[0]
        q1_host = torch.empty(
            rows,
            w1.shape[1],
            device="cpu",
            dtype=torch.float8_e4m3fn,
            pin_memory=True,
        )
        q3_host = torch.empty_like(q1_host, pin_memory=True)
        qs1_parts = []
        qs3_parts = []

        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            indices = permuted_indices[start:end]
            real = (indices >= 0) & (indices < x.shape[0])
            safe_indices = indices.clamp(min=0, max=x.shape[0] - 1).long()
            chunk_x = _index_select_rows_with_cache_retry(x, safe_indices)
            chunk_x.masked_fill_((~real).unsqueeze(-1), 0)
            chunk_offsets = _expert_chunk_offsets(offsets, start, end)
            z1, z3 = _grouped_mm_pair(
                chunk_x,
                w1,
                s1,
                chunk_x,
                w3,
                s3,
                chunk_offsets,
                shared_input=True,
            )
            # The W2 result is hidden-size wide. Release the gathered input
            # before W2 allocates its output. In x72p step 39, retaining this
            # 24 MiB tensor left only 1.5 MiB free and the following 24 MiB
            # grouped-matrix output failed on rank 18.
            del chunk_x, safe_indices
            q1, qs1 = ops.scaled_fp8_quant(z1, use_per_token_if_dynamic=True)
            q3, qs3 = ops.scaled_fp8_quant(z3, use_per_token_if_dynamic=True)
            q1_host[start:end].copy_(q1, non_blocking=True)
            q3_host[start:end].copy_(q3, non_blocking=True)
            qs1_parts.append(qs1)
            qs3_parts.append(qs3)
            # The copies are ordered on the current CUDA stream. Dropping the
            # source references here lets the caching allocator reuse their
            # storage after the copies finish, before the W2 allocation.
            del q1, q3
            F.silu(z1, inplace=True)
            z1.mul_(z3)
            del z3
            chunk_out = _grouped_mm_chunked(
                z1,
                w2,
                s2,
                chunk_offsets,
            )
            _index_copy_real_rows_with_cache_retry(x, indices, chunk_out, real)
            # Do not retain one completed W2 output while Python evaluates
            # the next W1/W3 call. At 4,096 rows the old gated activation and
            # W2 output otherwise keep another 56 MiB live across chunks.
            del z1, chunk_out, indices, real, chunk_offsets

        qs1 = torch.cat(qs1_parts, dim=0)
        qs3 = torch.cat(qs3_parts, dim=0)
        offload_done = torch.cuda.Event()
        offload_done.record(torch.cuda.current_stream(x.device))
        ctx.owner = owner
        ctx.offload_done = offload_done
        ctx.source_rows = x.shape[0]
        ctx.chunk_rows = chunk_rows
        ctx.backward_chunk_rows = min(chunk_rows, _FP8_EXPERT_BACKWARD_CHUNK_ROWS)
        if owner is None:
            ctx.save_for_backward(
                q1_host,
                qs1,
                q3_host,
                qs3,
                permuted_indices,
                offsets,
                w1,
                s1,
                w2,
                s2,
                w3,
                s3,
            )
        else:
            ctx.save_for_backward(
                q1_host,
                qs1,
                q3_host,
                qs3,
                permuted_indices,
                offsets,
            )
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        if _EXPERT_PROFILE:
            profile_start = torch.cuda.Event(enable_timing=True)
            profile_end = torch.cuda.Event(enable_timing=True)
            profile_start.record()
        if ctx.owner is None:
            (
                q1,
                qs1,
                q3,
                qs3,
                permuted_indices,
                offsets,
                w1,
                s1,
                w2,
                s2,
                w3,
                s3,
            ) = ctx.saved_tensors
        else:
            q1, qs1, q3, qs3, permuted_indices, offsets = ctx.saved_tensors
            owner = ctx.owner
            w1, s1, w2, s2, w3, s3 = (
                getattr(owner, name)
                for name in (
                    "w1",
                    "w1_scale_inv",
                    "w2",
                    "w2_scale_inv",
                    "w3",
                    "w3_scale_inv",
                )
            )
            w1, s1, w2, s2, w3, s3 = (
                value.to_local() if isinstance(value, DTensor) else value
                for value in (w1, s1, w2, s2, w3, s3)
            )

        torch.cuda.current_stream(grad.device).wait_event(ctx.offload_done)
        grad = grad.contiguous()
        rows = permuted_indices.shape[0]
        for start in range(0, rows, ctx.backward_chunk_rows):
            end = min(start + ctx.backward_chunk_rows, rows)
            indices = permuted_indices[start:end]
            real = (indices >= 0) & (indices < ctx.source_rows)
            safe_indices = indices.clamp(min=0, max=ctx.source_rows - 1).long()
            grad_chunk = _index_select_rows_with_cache_retry(grad, safe_indices)
            grad_chunk.masked_fill_((~real).unsqueeze(-1), 0)
            chunk_offsets = _expert_chunk_offsets(offsets, start, end)

            q1_device = q1[start:end].to(grad.device, non_blocking=True)
            z1 = q1_device.to(grad.dtype).mul_(qs1[start:end].to(grad.dtype))
            del q1_device
            q3_device = q3[start:end].to(grad.device, non_blocking=True)
            z3 = q3_device.to(grad.dtype).mul_(qs3[start:end].to(grad.dtype))
            del q3_device
            hidden_grad = _grouped_mm_chunked(
                grad_chunk,
                w2.transpose(-2, -1),
                s2.transpose(-2, -1),
                chunk_offsets,
            )
            z1_grad, z3_grad = _silu_glu_backward_inplace(hidden_grad, z1, z3)
            del hidden_grad, z1, z3
            x_grad, x_grad_w3 = _grouped_mm_pair(
                z1_grad,
                w1.transpose(-2, -1),
                s1.transpose(-2, -1),
                z3_grad,
                w3.transpose(-2, -1),
                s3.transpose(-2, -1),
                chunk_offsets,
            )
            x_grad.add_(x_grad_w3)
            _index_copy_real_rows_with_cache_retry(grad, indices, x_grad, real)
            # Release this chunk before the next chunk allocates. Assignment
            # keeps the previous value alive until the new RHS completes, so
            # relying on name reuse briefly retains two chunks.
            del (
                indices,
                real,
                safe_indices,
                grad_chunk,
                chunk_offsets,
                z1_grad,
                z3_grad,
                x_grad,
                x_grad_w3,
            )

        if _EXPERT_PROFILE:
            profile_end.record()
            _profile_expert_backward(profile_start, profile_end)
        del ctx.owner, ctx.offload_done
        return grad, None, None, None, None, None, None, None, None, None, None


def _grouped_lora_projection(x, lora_a, lora_b, offsets):
    """Run one expert LoRA projection without changing the FP8 base path."""
    output, _ = _grouped_lora_projection_with_low_rank(
        x,
        lora_a,
        lora_b,
        offsets,
    )
    return output


def _grouped_lora_projection_pair(
    x,
    first_lora_a,
    first_lora_b,
    second_lora_a,
    second_lora_b,
    offsets,
):
    """Share the first grouped GEMM for two projections with one input.

    The W1 and W3 LoRA projections have the same input and row assignment.
    Concatenating only their A factors removes one grouped launch. The two B
    GEMMs remain separate because their output matrices are different.
    """
    first_output, second_output, _, _ = _grouped_lora_projection_pair_with_low_rank(
        x,
        first_lora_a,
        first_lora_b,
        second_lora_a,
        second_lora_b,
        offsets,
    )
    return first_output, second_output


def _grouped_lora_projection_pair_with_low_rank(
    x,
    first_lora_a,
    first_lora_b,
    second_lora_a,
    second_lora_b,
    offsets,
):
    """Return paired LoRA outputs and both rank-sized intermediates."""
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"expert LoRA input must be BF16 or FP16, got {x.dtype}")
    if first_lora_a.shape != second_lora_a.shape:
        raise ValueError("paired expert LoRA A factors must have equal shapes")
    if first_lora_b.shape != second_lora_b.shape:
        raise ValueError("paired expert LoRA B factors must have equal shapes")
    if first_lora_a.shape[1] != second_lora_a.shape[1]:
        raise ValueError("paired expert LoRA ranks must match")
    if first_lora_a.shape[2] != x.shape[1]:
        raise ValueError("paired expert LoRA input widths do not match x")
    if first_lora_a.dtype != x.dtype or second_lora_a.dtype != x.dtype:
        raise TypeError("paired expert LoRA A factors must use the input dtype")
    if first_lora_b.dtype != x.dtype or second_lora_b.dtype != x.dtype:
        raise TypeError("paired expert LoRA B factors must use the input dtype")
    if offsets.shape[0] != first_lora_a.shape[0]:
        raise ValueError("paired expert LoRA factors and offsets have different expert counts")

    joined_a = torch.cat((first_lora_a, second_lora_a), dim=1)
    joined_low_rank = torch._grouped_mm(
        x,
        joined_a.transpose(-2, -1),
        offs=offsets,
    )
    rank = first_lora_a.shape[1]
    first_low_rank, second_low_rank = joined_low_rank.split(rank, dim=-1)
    return (
        torch._grouped_mm(
            first_low_rank,
            first_lora_b.transpose(-2, -1),
            offs=offsets,
        ),
        torch._grouped_mm(
            second_low_rank,
            second_lora_b.transpose(-2, -1),
            offs=offsets,
        ),
        first_low_rank,
        second_low_rank,
    )


def _grouped_lora_projection_with_low_rank(x, lora_a, lora_b, offsets):
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"expert LoRA input must be BF16 or FP16, got {x.dtype}")
    if lora_a.dtype != x.dtype or lora_b.dtype != x.dtype:
        raise TypeError(
            "expert LoRA factors must use the activation dtype: "
            f"x={x.dtype}, A={lora_a.dtype}, B={lora_b.dtype}"
        )
    if lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError("expert LoRA factors must be expert-major three-dimensional tensors")
    if lora_a.shape[0] != lora_b.shape[0] or offsets.shape[0] != lora_a.shape[0]:
        raise ValueError("expert LoRA factors and offsets have different expert counts")
    if lora_a.shape[1] != lora_b.shape[2]:
        raise ValueError("expert LoRA A and B ranks do not match")
    if lora_a.shape[2] != x.shape[1]:
        raise ValueError("expert LoRA A input width does not match the activation")

    low_rank = torch._grouped_mm(
        x,
        lora_a.transpose(-2, -1),
        offs=offsets,
    )
    return (
        torch._grouped_mm(
            low_rank,
            lora_b.transpose(-2, -1),
            offs=offsets,
        ),
        low_rank,
    )


def _grouped_outer_accumulate(left, right, offsets, output):
    if _grouped_outer_accumulate_kernel is None:
        raise RuntimeError("Triton is unavailable for expert LoRA backward")
    if not all(value.is_cuda for value in (left, right, offsets, output)):
        raise RuntimeError("expert LoRA backward requires CUDA tensors")
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise TypeError("expert LoRA backward inputs must be BF16")
    if output.dtype != torch.bfloat16:
        raise TypeError("expert LoRA parameter gradients must be BF16")
    if offsets.dtype != torch.int32 or offsets.ndim != 1:
        raise TypeError("expert LoRA offsets must be a one-dimensional INT32 tensor")
    if left.ndim != 2 or right.ndim != 2 or output.ndim != 3:
        raise ValueError("expert LoRA outer product has invalid tensor dimensions")
    if left.shape[0] != right.shape[0]:
        raise ValueError("expert LoRA outer-product inputs have different row counts")
    expected = (offsets.shape[0], left.shape[1], right.shape[1])
    if tuple(output.shape) != expected:
        raise ValueError(
            f"expert LoRA gradient shape {tuple(output.shape)} does not match {expected}"
        )

    block_left = 32
    block_right = 32
    grid = (
        offsets.shape[0],
        triton.cdiv(left.shape[1], block_left),
        triton.cdiv(right.shape[1], block_right),
    )
    _grouped_outer_accumulate_kernel[grid](
        left,
        right,
        offsets,
        output,
        left.shape[1],
        right.shape[1],
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_ROWS=32,
        BLOCK_LEFT=block_left,
        BLOCK_RIGHT=block_right,
        num_warps=4,
    )


def _grouped_lora_backward(
    x,
    lora_a,
    lora_b,
    offsets,
    grad_output,
    grad_a,
    grad_b,
    low_rank=None,
    scaling=1.0,
):
    """Compute one grouped LoRA backward without nested autograd."""
    if grad_output.dtype != x.dtype:
        raise TypeError("expert LoRA output gradient must use the activation dtype")
    if tuple(grad_a.shape) != tuple(lora_a.shape) or tuple(grad_b.shape) != tuple(
        lora_b.shape
    ):
        raise ValueError("expert LoRA gradient buffers do not match their factors")
    if lora_a.shape[1] != 32:
        raise RuntimeError(
            f"native expert LoRA backward supports rank 32, got {lora_a.shape[1]}"
        )

    if low_rank is None:
        low_rank = torch._grouped_mm(
            x,
            lora_a.transpose(-2, -1),
            offs=offsets,
        )
    elif (
        low_rank.device != x.device
        or low_rank.dtype != x.dtype
        or low_rank.shape != (x.shape[0], lora_a.shape[1])
    ):
        raise ValueError("saved expert LoRA low-rank activation is invalid")
    low_rank_grad = torch._grouped_mm(
        grad_output,
        lora_b,
        offs=offsets,
    )
    # Scale the intermediate in place. This avoids a full-width temporary for
    # each expert projection while preserving the usual LoRA gradient scale.
    if scaling != 1.0:
        low_rank_grad.mul_(scaling)
    input_grad = torch._grouped_mm(
        low_rank_grad,
        lora_a,
        offs=offsets,
    )
    _grouped_outer_accumulate(low_rank_grad, x, offsets, grad_a)
    if scaling != 1.0:
        low_rank.mul_(scaling)
    _grouped_outer_accumulate(grad_output, low_rank, offsets, grad_b)
    return input_grad


class _InplacePermutedLoRAExperts(torch.autograd.Function):
    """Native block-FP8 experts with standard LoRA on W1, W2, and W3."""

    @staticmethod
    def forward(
        ctx,
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        w1_lora_a,
        w1_lora_b,
        w2_lora_a,
        w2_lora_b,
        w3_lora_a,
        w3_lora_b,
        scaling,
        permuted_indices,
        offsets,
        owner,
        received_input,
        received_scores,
        chunk_rows,
    ):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("native routed-expert LoRA requires CUDA BF16 activations")
        if received_input is None:
            raise RuntimeError(
                "native routed-expert LoRA requires the exact received dispatch input"
            )
        if chunk_rows <= 0:
            raise ValueError("native routed-expert LoRA chunk size must be positive")
        scaling = float(scaling)
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("native routed-expert LoRA scaling must be finite and positive")
        local_recompute = bool(
            getattr(received_input, "local_recompute", False)
        )
        if local_recompute != (received_scores is not None):
            raise RuntimeError(
                "native expert local recompute state does not match its route scores"
            )
        if local_recompute and (
            received_scores.device != x.device
            or received_scores.dtype != torch.float32
            or received_scores.shape != (x.shape[0],)
            or not received_scores.is_contiguous()
        ):
            raise ValueError(
                "native expert local recompute requires contiguous CUDA FP32 route scores"
            )

        rows = permuted_indices.shape[0]
        save_backward_state = bool(
            getattr(received_input, "save_for_backward", True)
        )
        save_intermediates = save_backward_state and not local_recompute
        if save_intermediates:
            q1_host = torch.empty(
                rows,
                w1.shape[1],
                device="cpu",
                dtype=torch.float8_e4m3fn,
                pin_memory=True,
            )
            q3_host = torch.empty_like(q1_host, pin_memory=True)
            qs1_parts = []
            qs3_parts = []

        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            indices = permuted_indices[start:end]
            real = (indices >= 0) & (indices < x.shape[0])
            safe_indices = indices.clamp(min=0, max=x.shape[0] - 1).long()
            phase_timer = _expert_phase_start("forward.input_gather")
            chunk_x = _index_select_rows_with_cache_retry(x, safe_indices)
            chunk_x.masked_fill_((~real).unsqueeze(-1), 0)
            _expert_phase_end(phase_timer)
            chunk_offsets = _expert_chunk_offsets(offsets, start, end)

            phase_timer = _expert_phase_start("forward.w13_base")
            z1, z3 = _grouped_mm_pair(
                chunk_x,
                w1,
                s1,
                chunk_x,
                w3,
                s3,
                chunk_offsets,
                shared_input=True,
            )
            _expert_phase_end(phase_timer)
            phase_timer = _expert_phase_start("forward.w13_lora")
            z1_lora, z3_lora = _grouped_lora_projection_pair(
                chunk_x,
                w1_lora_a,
                w1_lora_b,
                w3_lora_a,
                w3_lora_b,
                chunk_offsets,
            )
            z1.add_(
                z1_lora,
                alpha=scaling,
            )
            z3.add_(
                z3_lora,
                alpha=scaling,
            )
            del z1_lora, z3_lora
            _expert_phase_end(phase_timer)

            # The gathered source rows are no longer needed after W1/W3 and
            # their LoRA updates. Release this bounded copy before W2 allocates
            # its hidden-width output. This is safe for local recompute because
            # route scores and the source input remain separate tensors.
            del chunk_x, safe_indices
            if save_intermediates:
                phase_timer = _expert_phase_start("forward.quantize_offload")
                q1, qs1 = ops.scaled_fp8_quant(z1, use_per_token_if_dynamic=True)
                q3, qs3 = ops.scaled_fp8_quant(z3, use_per_token_if_dynamic=True)
                q1_host[start:end].copy_(q1, non_blocking=True)
                q3_host[start:end].copy_(q3, non_blocking=True)
                qs1_parts.append(qs1)
                qs3_parts.append(qs3)
                del q1, q3
                _expert_phase_end(phase_timer)

            phase_timer = _expert_phase_start("forward.glu")
            F.silu(z1, inplace=True)
            z1.mul_(z3)
            _expert_phase_end(phase_timer)
            del z3
            phase_timer = _expert_phase_start("forward.w2_base")
            chunk_out = _grouped_mm_chunked(
                z1,
                w2,
                s2,
                chunk_offsets,
            )
            _expert_phase_end(phase_timer)
            phase_timer = _expert_phase_start("forward.w2_lora")
            chunk_out.add_(
                _grouped_lora_projection(
                    z1,
                    w2_lora_a,
                    w2_lora_b,
                    chunk_offsets,
                ),
                alpha=scaling,
            )
            _expert_phase_end(phase_timer)
            phase_timer = _expert_phase_start("forward.combine")
            _index_copy_real_rows_with_cache_retry(x, indices, chunk_out, real)
            _expert_phase_end(phase_timer)
            del (
                z1,
                chunk_out,
                indices,
                real,
                chunk_offsets,
            )

        if local_recompute:
            for start in range(0, x.shape[0], chunk_rows):
                end = min(start + chunk_rows, x.shape[0])
                x[start:end].copy_(
                    (
                        x[start:end].float()
                        * received_scores[start:end, None]
                    ).to(x.dtype)
                )

        if not save_backward_state:
            _profile_expert_phases_tick("forward")
            ctx.has_backward_state = False
            ctx.mark_dirty(x)
            return x

        if save_intermediates:
            qs1 = torch.cat(qs1_parts, dim=0)
            qs3 = torch.cat(qs3_parts, dim=0)
        row_index_device = getattr(received_input, "row_index_device", None)
        if row_index_device == "cuda":
            saved_permuted_indices = permuted_indices.to(torch.int64)
        elif row_index_device == "cpu":
            saved_permuted_indices = torch.empty_like(
                permuted_indices,
                device="cpu",
                pin_memory=True,
            )
            saved_permuted_indices.copy_(permuted_indices, non_blocking=True)
        else:
            raise TypeError("native expert input has no supported row-index device")
        _profile_expert_phases_tick("forward")
        offload_done = torch.cuda.Event()
        offload_done.record(torch.cuda.current_stream(x.device))
        ctx.has_backward_state = True
        ctx.owner = owner
        ctx.received_input = received_input
        ctx.offload_done = offload_done
        ctx.source_rows = x.shape[0]
        ctx.row_index_device = row_index_device
        ctx.chunk_rows = min(chunk_rows, _FP8_EXPERT_BACKWARD_CHUNK_ROWS)
        ctx.scaling = scaling
        ctx.local_recompute = local_recompute
        ctx.lora_parameter_metadata = tuple(
            (tuple(value.shape), value.dtype)
            for value in (
                w1_lora_a,
                w1_lora_b,
                w2_lora_a,
                w2_lora_b,
                w3_lora_a,
                w3_lora_b,
            )
        )
        if local_recompute:
            ctx.save_for_backward(
                saved_permuted_indices,
                offsets,
                received_scores,
            )
        else:
            ctx.save_for_backward(
                q1_host,
                qs1,
                q3_host,
                qs3,
                saved_permuted_indices,
                offsets,
            )
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        if not ctx.has_backward_state:
            raise RuntimeError(
                "native expert LoRA backward state was not saved during checkpoint forward"
            )
        if _EXPERT_PROFILE:
            profile_start = torch.cuda.Event(enable_timing=True)
            profile_end = torch.cuda.Event(enable_timing=True)
            profile_start.record()
        if ctx.local_recompute:
            saved_permuted_indices, offsets, received_scores = ctx.saved_tensors
            q1 = qs1 = q3 = qs3 = None
        else:
            (
                q1,
                qs1,
                q3,
                qs3,
                saved_permuted_indices,
                offsets,
            ) = ctx.saved_tensors
            received_scores = None
        owner = ctx.owner
        w1, s1, w2, s2, w3, s3 = (
            getattr(owner.base_layer, name)
            for name in (
                "w1",
                "w1_scale_inv",
                "w2",
                "w2_scale_inv",
                "w3",
                "w3_scale_inv",
            )
        )
        w1, s1, w2, s2, w3, s3 = (
            value.to_local() if isinstance(value, DTensor) else value
            for value in (w1, s1, w2, s2, w3, s3)
        )
        (
            w1_lora_a,
            w1_lora_b,
            w2_lora_a,
            w2_lora_b,
            w3_lora_a,
            w3_lora_b,
        ) = (
            getattr(owner, f"{projection}_lora_{factor}")[0]
            for projection in ("w1", "w2", "w3")
            for factor in ("A", "B")
        )
        w1_lora_a, w1_lora_b, w2_lora_a, w2_lora_b, w3_lora_a, w3_lora_b = (
            value.to_local() if isinstance(value, DTensor) else value
            for value in (
                w1_lora_a,
                w1_lora_b,
                w2_lora_a,
                w2_lora_b,
                w3_lora_a,
                w3_lora_b,
            )
        )
        lora_parameters = (
            w1_lora_a,
            w1_lora_b,
            w2_lora_a,
            w2_lora_b,
            w3_lora_a,
            w3_lora_b,
        )
        current_metadata = tuple(
            (tuple(value.shape), value.dtype) for value in lora_parameters
        )
        if current_metadata != ctx.lora_parameter_metadata:
            raise RuntimeError("native expert LoRA parameter metadata changed before backward")

        torch.cuda.current_stream(grad.device).wait_event(ctx.offload_done)
        if ctx.row_index_device == "cpu":
            ctx.offload_done.synchronize()
        elif ctx.row_index_device != "cuda":
            raise RuntimeError("native expert row-index device changed before backward")
        grad = grad.contiguous()
        workspace = ctx.received_input.make_workspace(ctx.chunk_rows, grad.device)
        parameter_grads = [torch.zeros_like(value) for value in lora_parameters]
        score_grads = (
            torch.empty_like(received_scores)
            if ctx.local_recompute
            else None
        )

        rows = saved_permuted_indices.shape[0]
        for start in range(0, rows, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, rows)
            reconstruction_indices = saved_permuted_indices[start:end]
            indices = (
                reconstruction_indices
                if ctx.row_index_device == "cuda"
                else reconstruction_indices.to(grad.device, non_blocking=True)
            )
            real = (indices >= 0) & (indices < ctx.source_rows)
            safe_indices = indices.clamp(min=0, max=ctx.source_rows - 1).long()
            phase_timer = _expert_phase_start("backward.input_gather")
            grad_chunk = _index_select_rows_with_cache_retry(grad, safe_indices)
            grad_chunk.masked_fill_((~real).unsqueeze(-1), 0)
            _expert_phase_end(phase_timer)
            chunk_offsets = _expert_chunk_offsets(offsets, start, end)
            recomputed_w1_low_rank = None
            recomputed_w3_low_rank = None

            if ctx.local_recompute:
                phase_timer = _expert_phase_start("backward.reconstruct")
                reconstructed_x = torch.empty_like(grad_chunk)
                ctx.received_input.reconstruct_rows(
                    reconstruction_indices,
                    reconstructed_x,
                    workspace,
                )
                _expert_phase_end(phase_timer)
                phase_timer = _expert_phase_start("backward.w13_base")
                z1, z3 = _grouped_mm_pair(
                    reconstructed_x,
                    w1,
                    s1,
                    reconstructed_x,
                    w3,
                    s3,
                    chunk_offsets,
                    shared_input=True,
                )
                _expert_phase_end(phase_timer)
                phase_timer = _expert_phase_start("backward.w13_lora")
                (
                    z1_lora,
                    z3_lora,
                    recomputed_w1_low_rank,
                    recomputed_w3_low_rank,
                ) = _grouped_lora_projection_pair_with_low_rank(
                    reconstructed_x,
                    w1_lora_a,
                    w1_lora_b,
                    w3_lora_a,
                    w3_lora_b,
                    chunk_offsets,
                )
                z1.add_(
                    z1_lora,
                    alpha=ctx.scaling,
                )
                z3.add_(
                    z3_lora,
                    alpha=ctx.scaling,
                )
                del z1_lora, z3_lora
                _expert_phase_end(phase_timer)
                phase_timer = _expert_phase_start("backward.activation_restore")
                forward_hidden = F.silu(z1) * z3
                _expert_phase_end(phase_timer)
                phase_timer = _expert_phase_start("backward.w2_base")
                chunk_out = _grouped_mm_chunked(
                    forward_hidden,
                    w2,
                    s2,
                    chunk_offsets,
                )
                _expert_phase_end(phase_timer)
                phase_timer = _expert_phase_start("backward.w2_lora")
                chunk_out.add_(
                    _grouped_lora_projection(
                        forward_hidden,
                        w2_lora_a,
                        w2_lora_b,
                        chunk_offsets,
                    ),
                    alpha=ctx.scaling,
                )
                quantized_out, output_scales = ops.scaled_fp8_quant(
                    chunk_out,
                    use_per_token_if_dynamic=True,
                )
                chunk_out = quantized_out.to(chunk_out.dtype).mul_(
                    output_scales.to(chunk_out.dtype)
                )
                _expert_phase_end(phase_timer)
                chunk_score_grads = (
                    grad_chunk.float() * chunk_out.float()
                ).sum(dim=-1)
                _index_copy_real_rows_with_cache_retry(
                    score_grads,
                    indices,
                    chunk_score_grads,
                    real,
                )
                del (
                    chunk_out,
                    quantized_out,
                    output_scales,
                    chunk_score_grads,
                )
                chunk_scores = received_scores.index_select(0, safe_indices)
                grad_chunk.copy_(
                    (grad_chunk.float() * chunk_scores[:, None]).to(grad.dtype)
                )
                grad_chunk.masked_fill_((~real).unsqueeze(-1), 0)
                quantized_grad, grad_scales = ops.scaled_fp8_quant(
                    grad_chunk,
                    use_per_token_if_dynamic=True,
                )
                grad_chunk.copy_(
                    quantized_grad.to(grad.dtype).mul_(
                        grad_scales.to(grad.dtype)
                    )
                )
                del quantized_grad, grad_scales, chunk_scores
                quantized_z1, z1_scales = ops.scaled_fp8_quant(
                    z1,
                    use_per_token_if_dynamic=True,
                )
                z1 = quantized_z1.to(z1.dtype).mul_(
                    z1_scales.to(z1.dtype)
                )
                del quantized_z1, z1_scales
                quantized_z3, z3_scales = ops.scaled_fp8_quant(
                    z3,
                    use_per_token_if_dynamic=True,
                )
                z3 = quantized_z3.to(z3.dtype).mul_(
                    z3_scales.to(z3.dtype)
                )
                del quantized_z3, z3_scales
                hidden = F.silu(z1) * z3
                del forward_hidden
            else:
                phase_timer = _expert_phase_start("backward.activation_restore")
                q1_device = q1[start:end].to(grad.device, non_blocking=True)
                z1 = q1_device.to(grad.dtype).mul_(
                    qs1[start:end].to(grad.dtype)
                )
                del q1_device
                q3_device = q3[start:end].to(grad.device, non_blocking=True)
                z3 = q3_device.to(grad.dtype).mul_(
                    qs3[start:end].to(grad.dtype)
                )
                del q3_device
                hidden = F.silu(z1) * z3
                _expert_phase_end(phase_timer)

            phase_timer = _expert_phase_start("backward.w2_base")
            hidden_grad = _grouped_mm_chunked(
                grad_chunk,
                w2.transpose(-2, -1),
                s2.transpose(-2, -1),
                chunk_offsets,
            )
            _expert_phase_end(phase_timer)
            phase_timer = _expert_phase_start("backward.w2_lora")
            lora_hidden_grad = _grouped_lora_backward(
                hidden,
                w2_lora_a,
                w2_lora_b,
                chunk_offsets,
                grad_chunk,
                parameter_grads[2],
                parameter_grads[3],
                scaling=ctx.scaling,
            )
            hidden_grad.add_(lora_hidden_grad)
            _expert_phase_end(phase_timer)
            del lora_hidden_grad, hidden

            phase_timer = _expert_phase_start("backward.glu")
            z1_grad, z3_grad = _silu_glu_backward_inplace(
                hidden_grad,
                z1,
                z3,
            )
            _expert_phase_end(phase_timer)
            del hidden_grad, z1, z3
            phase_timer = _expert_phase_start("backward.w13_base")
            x_grad, x_grad_w3 = _grouped_mm_pair(
                z1_grad,
                w1.transpose(-2, -1),
                s1.transpose(-2, -1),
                z3_grad,
                w3.transpose(-2, -1),
                s3.transpose(-2, -1),
                chunk_offsets,
            )
            _expert_phase_end(phase_timer)
            x_grad.add_(x_grad_w3)
            del x_grad_w3

            if not ctx.local_recompute:
                reconstructed_x = grad_chunk
                phase_timer = _expert_phase_start("backward.reconstruct")
                ctx.received_input.reconstruct_rows(
                    reconstruction_indices,
                    reconstructed_x,
                    workspace,
                )
                _expert_phase_end(phase_timer)
            phase_timer = _expert_phase_start("backward.w1_lora")
            lora_x_grad = _grouped_lora_backward(
                reconstructed_x,
                w1_lora_a,
                w1_lora_b,
                chunk_offsets,
                z1_grad,
                parameter_grads[0],
                parameter_grads[1],
                low_rank=recomputed_w1_low_rank,
                scaling=ctx.scaling,
            )
            x_grad.add_(lora_x_grad)
            _expert_phase_end(phase_timer)
            del lora_x_grad
            phase_timer = _expert_phase_start("backward.w3_lora")
            lora_x_grad = _grouped_lora_backward(
                reconstructed_x,
                w3_lora_a,
                w3_lora_b,
                chunk_offsets,
                z3_grad,
                parameter_grads[4],
                parameter_grads[5],
                low_rank=recomputed_w3_low_rank,
                scaling=ctx.scaling,
            )
            x_grad.add_(lora_x_grad)
            _expert_phase_end(phase_timer)
            phase_timer = _expert_phase_start("backward.combine")
            _index_copy_real_rows_with_cache_retry(grad, indices, x_grad, real)
            _expert_phase_end(phase_timer)
            del (
                indices,
                real,
                safe_indices,
                grad_chunk,
                chunk_offsets,
                z1_grad,
                z3_grad,
                x_grad,
                reconstructed_x,
                lora_x_grad,
                recomputed_w1_low_rank,
                recomputed_w3_low_rank,
            )

        _profile_expert_phases_tick("backward")
        if _EXPERT_PROFILE:
            profile_end.record()
            _profile_expert_backward(profile_start, profile_end)
        del (
            ctx.owner,
            ctx.received_input,
            ctx.offload_done,
            ctx.lora_parameter_metadata,
        )
        return (
            grad,
            None,
            None,
            None,
            None,
            None,
            None,
            *parameter_grads,
            None,
            None,
            None,
            None,
            None,
            score_grads,
            None,
        )


class _ScaleRoutedOutput(torch.autograd.Function):
    """Apply FP32 route scores with bounded temporary storage.

    The ordinary expression materializes the complete routed output in FP32,
    the complete FP32 product, and the final BF16 tensor together. Production
    routed output can exceed 500 MiB, so that eager expression does not fit.
    """

    @staticmethod
    def forward(ctx, x, scores, chunk_rows):
        flat_scores = scores.reshape(-1)
        if x.shape[0] != flat_scores.shape[0]:
            raise ValueError(
                f"route score rows {flat_scores.shape[0]} do not match expert rows {x.shape[0]}"
            )
        inplace_offload = _FP8_EXPERT_ACTIVATION_OFFLOAD and x.is_cuda
        if inplace_offload:
            saved_x = torch.empty_like(x, device="cpu", pin_memory=True)
            saved_x.copy_(x, non_blocking=True)
            offload_done = torch.cuda.Event()
            offload_done.record(torch.cuda.current_stream(x.device))
        else:
            saved_x = x
            offload_done = None
        ctx.save_for_backward(saved_x, flat_scores)
        ctx.score_shape = scores.shape
        ctx.chunk_rows = chunk_rows
        ctx.inplace_offload = inplace_offload
        ctx.offload_done = offload_done
        out = x if inplace_offload else torch.empty_like(x)
        for start in range(0, x.shape[0], chunk_rows):
            end = min(start + chunk_rows, x.shape[0])
            out[start:end].copy_(
                (x[start:end].float() * flat_scores[start:end, None]).to(x.dtype)
            )
        if inplace_offload:
            ctx.mark_dirty(x)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, scores = ctx.saved_tensors
        if ctx.inplace_offload:
            torch.cuda.current_stream(grad.device).wait_event(ctx.offload_done)
            grad_x = grad.contiguous()
        else:
            grad_x = torch.empty_like(x)
        grad_scores = torch.empty_like(scores)
        for start in range(0, x.shape[0], ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, x.shape[0])
            grad_chunk = grad[start:end].float()
            if ctx.inplace_offload:
                x_chunk = x[start:end].to(grad.device, non_blocking=True).float()
            else:
                x_chunk = x[start:end].float()
            grad_x[start:end].copy_((grad_chunk * scores[start:end, None]).to(x.dtype))
            grad_scores[start:end].copy_((grad_chunk * x_chunk).sum(dim=-1))
        del ctx.offload_done
        return grad_x, grad_scores.view(ctx.score_shape), None


def scale_routed_output(x, scores, chunk_rows=None):
    return _ScaleRoutedOutput.apply(
        x,
        scores,
        chunk_rows or _FP8_ROUTE_SCALE_CHUNK_ROWS,
    )


def fp8_experts(
    w1,
    w2,
    w3,
    s1,
    s2,
    s3,
    x,
    num_tokens_per_expert,
    *,
    owner=None,
):
    """Run checkpoint-FP8 experts with TorchTitan's EP row permutation.

    After the EP all-to-all, rows are grouped first by sending rank and then
    by this rank's local experts.  ``num_tokens_per_expert`` consequently has
    ``ep_size * local_experts`` entries while the local weight shard has only
    ``local_experts`` matrices.  The ordinary grouped-expert path uses
    TorchTitan's ``expert_parallel`` decorator to transpose that logical
    layout, combine counts for each local expert, and add aligned padding.
    The native FP8 path used to unwrap the DTensors and call the kernel
    directly.  Most rows were then assigned expert ids beyond the local
    weight array and were ignored by the fused kernel.
    """
    if isinstance(w1, DTensor):
        w1, w2, w3, s1, s2, s3 = (
            value.to_local() for value in (w1, w2, w3, s1, s2, s3)
        )

    if x.shape[0] == 0:
        return x

    local_experts = w1.shape[0]
    if num_tokens_per_expert.shape[0] == local_experts:
        return _Experts.apply(
            x,
            w1,
            s1,
            w2,
            s2,
            w3,
            s3,
            torch.cumsum(num_tokens_per_expert, 0, dtype=torch.int32),
            owner,
        )

    if num_tokens_per_expert.shape[0] % local_experts:
        raise ValueError(
            "EP token counts must be a whole number of local expert groups: "
            f"{num_tokens_per_expert.shape[0]} counts for {local_experts} experts"
        )

    from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    ep_size = num_tokens_per_expert.shape[0] // local_experts
    with torch.no_grad():
        permuted_indices, local_counts, _ = generate_permute_indices(
            num_tokens_per_expert,
            local_experts,
            ep_size,
            x.shape[0] + local_experts * TOKEN_GROUP_ALIGN_SIZE_M,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    offsets = torch.cumsum(local_counts, 0, dtype=torch.int32)
    if _FP8_EXPERT_ACTIVATION_OFFLOAD and x.is_cuda:
        return _InplacePermutedExperts.apply(
            x,
            w1,
            s1,
            w2,
            s2,
            w3,
            s3,
            permuted_indices,
            offsets,
            owner,
            _FP8_W13_CHUNK_ROWS,
        )

    # TorchTitan versions use either -1 or x.shape[0] as the shared zero
    # sentinel for alignment padding. Clamp both forms for the gather, then
    # zero those gathered rows in place. This avoids a complete padded copy of
    # the 426 MiB production input.
    real_rows = (permuted_indices >= 0) & (permuted_indices < x.shape[0])
    safe_indices = permuted_indices.clamp(min=0, max=x.shape[0] - 1)
    permuted_x = x[safe_indices, :]
    permuted_x.masked_fill_((~real_rows).unsqueeze(-1), 0)
    out = _Experts.apply(
        permuted_x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        offsets,
        owner,
    )
    # Expert forward no longer needs permuted input values, and its custom
    # backward keeps only quantized intermediates. Reuse this buffer for the
    # inverse permutation.
    out_unpermuted = permuted_x
    out_unpermuted[permuted_indices[real_rows], :] = out[real_rows]
    return out_unpermuted[: x.shape[0]]


def fp8_lora_experts(
    w1,
    w2,
    w3,
    s1,
    s2,
    s3,
    w1_lora_a,
    w1_lora_b,
    w2_lora_a,
    w2_lora_b,
    w3_lora_a,
    w3_lora_b,
    scaling,
    x,
    num_tokens_per_expert,
    received_input,
    received_scores=None,
    *,
    owner,
):
    """Run expert-sharded standard LoRA inside the native FP8 expert path."""
    values = (
        w1,
        w2,
        w3,
        s1,
        s2,
        s3,
        w1_lora_a,
        w1_lora_b,
        w2_lora_a,
        w2_lora_b,
        w3_lora_a,
        w3_lora_b,
    )
    values = tuple(
        value.to_local() if isinstance(value, DTensor) else value
        for value in values
    )
    (
        w1,
        w2,
        w3,
        s1,
        s2,
        s3,
        w1_lora_a,
        w1_lora_b,
        w2_lora_a,
        w2_lora_b,
        w3_lora_a,
        w3_lora_b,
    ) = values

    if x.shape[0] == 0:
        return x
    local_experts = w1.shape[0]
    if num_tokens_per_expert.shape[0] == local_experts:
        raise RuntimeError(
            "native routed-expert LoRA qualification requires real EP dispatch; "
            "the direct local path has no captured INT4 input"
        )
    if num_tokens_per_expert.shape[0] % local_experts:
        raise ValueError(
            "EP token counts must be a whole number of local expert groups: "
            f"{num_tokens_per_expert.shape[0]} counts for {local_experts} experts"
        )

    from torchtitan.distributed.expert_parallel import TOKEN_GROUP_ALIGN_SIZE_M
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    ep_size = num_tokens_per_expert.shape[0] // local_experts
    with torch.no_grad():
        permuted_indices, local_counts, _ = generate_permute_indices(
            num_tokens_per_expert,
            local_experts,
            ep_size,
            x.shape[0] + local_experts * TOKEN_GROUP_ALIGN_SIZE_M,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )
    offsets = torch.cumsum(local_counts, 0, dtype=torch.int32)
    return _InplacePermutedLoRAExperts.apply(
        x,
        w1,
        s1,
        w2,
        s2,
        w3,
        s3,
        w1_lora_a,
        w1_lora_b,
        w2_lora_a,
        w2_lora_b,
        w3_lora_a,
        w3_lora_b,
        scaling,
        permuted_indices,
        offsets,
        owner,
        received_input,
        received_scores,
        _FP8_W13_CHUNK_ROWS,
    )
