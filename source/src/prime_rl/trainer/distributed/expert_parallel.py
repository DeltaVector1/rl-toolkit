import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import triton
import triton.language as tl
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.distributed.tensor import DeviceMesh, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle
from torchtitan.distributed.expert_parallel import ExpertParallel
from triton.language.extra import libdevice
from vllm import _custom_ops as ops


@triton.jit
def _quantize_int4_kernel(x, packed, scale, cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    pair = tl.arange(0, block)
    col0 = pair * 2
    col1 = col0 + 1
    x0 = tl.load(x + row * cols + col0, mask=col0 < cols, other=0.0).to(tl.float32)
    x1 = tl.load(x + row * cols + col1, mask=col1 < cols, other=0.0).to(tl.float32)
    row_scale = tl.maximum(tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 7.0, 1e-8)
    q0 = tl.maximum(-7.0, tl.minimum(7.0, libdevice.rint(x0 / row_scale))).to(tl.int32)
    q1 = tl.maximum(-7.0, tl.minimum(7.0, libdevice.rint(x1 / row_scale))).to(tl.int32)
    tl.store(packed + row * ((cols + 1) // 2) + pair, (q0 + 8) | ((q1 + 8) << 4), mask=col0 < cols)
    tl.store(scale + row, row_scale)


@triton.jit
def _dequantize_int4_kernel(packed, scale, output, cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    pair = tl.arange(0, block)
    col0 = pair * 2
    value = tl.load(packed + row * ((cols + 1) // 2) + pair, mask=col0 < cols, other=0).to(tl.int32)
    row_scale = tl.load(scale + row)
    tl.store(output + row * cols + col0, ((value & 15) - 8) * row_scale, mask=col0 < cols)
    tl.store(output + row * cols + col0 + 1, ((value >> 4) - 8) * row_scale, mask=col0 + 1 < cols)


# When selective activation checkpointing targets "routed_experts", the whole
# dispatch -> experts -> combine region is re-run during backward, which repeats
# both all-to-alls. On this cluster the all-to-all is the dominant cost, so we
# keep the *received* INT4 payload from the forward pass and reuse it on the
# recompute. This is numerically exact: the recompute would dequantize the same
# bytes. The payload is 1/4 the size of the dequantized bf16 activation.
_EP_AC_CACHE = os.environ.get("PRIME_EP_AC_CACHE", "") == "1"


def _in_backward() -> bool:
    """True while executing an autograd graph task, i.e. during a checkpoint recompute."""
    return torch._C._current_graph_task_id() != -1


_EP_PROFILE = os.environ.get("PRIME_EP_PROFILE", "") == "1"
_EP_PROFILE_EVERY = int(os.environ.get("PRIME_EP_PROFILE_EVERY", "150"))
# tag -> list of (start_event, end_event); summed and cleared on each report
_ep_events: dict[str, list] = {}
_ep_calls = 0
_ep_t0 = time.perf_counter()


class _EPTimer:
    """CUDA-event timer for one all-to-all. No-op unless PRIME_EP_PROFILE=1."""

    __slots__ = ("tag", "start")

    def __init__(self, tag: str):
        self.tag = tag
        self.start = None

    def __enter__(self):
        if _EP_PROFILE:
            self.start = torch.cuda.Event(enable_timing=True)
            self.start.record()
        return self

    def __exit__(self, *exc):
        if _EP_PROFILE and self.start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            _ep_events.setdefault(self.tag, []).append((self.start, end))
        return False


def _ep_profile_tick() -> None:
    """Count a dispatch call; periodically report accumulated all-to-all time."""
    global _ep_calls, _ep_t0
    if not _EP_PROFILE:
        return
    _ep_calls += 1
    if _ep_calls % _EP_PROFILE_EVERY:
        return
    torch.cuda.synchronize()
    wall = time.perf_counter() - _ep_t0
    totals = {}
    for tag, pairs in _ep_events.items():
        totals[tag] = sum(s.elapsed_time(e) for s, e in pairs) / 1000.0
    _ep_events.clear()
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        a2a = sum(totals.values())
        parts = " ".join(f"{k}={v:.1f}s" for k, v in sorted(totals.items()))
        print(
            f"[ep-prof] rank={os.environ.get('RANK', '0')} "
            f"{_EP_PROFILE_EVERY} dispatches | wall={wall:.1f}s "
            f"a2a={a2a:.1f}s ({100 * a2a / max(wall, 1e-9):.0f}% of wall) | {parts}",
            flush=True,
        )
    _ep_t0 = time.perf_counter()


class _INT4Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, routed_input, output_splits, input_splits, group_name, tag="a2a", cache=None):
        ctx.tag = tag
        rows, cols = routed_input.shape
        packed_cols = (cols + 1) // 2
        block = triton.next_power_of_2(packed_cols)
        recompute = cache is not None and _in_backward() and cache[0] is not None
        if recompute:
            # Reuse the payload this rank received during the forward pass; the
            # quantize and both all-to-alls would reproduce it byte for byte.
            output, output_scale = cache[0]
            cache[0] = None
        else:
            quantized = torch.empty((rows, packed_cols), device=routed_input.device, dtype=torch.uint8)
            scale = torch.empty(rows, device=routed_input.device, dtype=torch.float16)
            _quantize_int4_kernel[(rows,)](routed_input, quantized, scale, cols, block, num_warps=8)
            output = quantized.new_empty((sum(output_splits), packed_cols))
            output_scale = scale.new_empty(sum(output_splits))
            group = _resolve_process_group(group_name)
            with _EPTimer(f"fwd_{tag}"):
                dist.all_to_all_single(output, quantized, output_splits, input_splits, group=group)
                dist.all_to_all_single(output_scale, scale, output_splits, input_splits, group=group)
            # NOTE: do not test torch.is_grad_enabled() here. autograd.Function
            # forward always runs with grad disabled, so that check would never
            # pass and nothing would ever be cached. The caller decides whether
            # caching applies and passes cache=None when it does not.
            if cache is not None and not _in_backward():
                cache[0] = (output, output_scale)
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group_name = group_name
        ctx.cols = cols
        dequantized = torch.empty((sum(output_splits), cols), device=output.device, dtype=routed_input.dtype)
        _dequantize_int4_kernel[(sum(output_splits),)](
            output, output_scale, dequantized, cols, block, num_warps=8
        )
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        quantized, scale = ops.scaled_fp8_quant(grad_output, use_per_token_if_dynamic=True)
        scale = scale.flatten()
        grad_input = quantized.new_empty((sum(ctx.input_splits), quantized.shape[1]))
        grad_scale = scale.new_empty(sum(ctx.input_splits))
        group = _resolve_process_group(ctx.group_name)
        with _EPTimer(f"bwd_{ctx.tag}"):
            dist.all_to_all_single(
                grad_input, quantized, ctx.input_splits, ctx.output_splits, group=group
            )
            dist.all_to_all_single(
                grad_scale, scale, ctx.input_splits, ctx.output_splits, group=group
            )
        grad_input = grad_input.to(grad_output.dtype)
        grad_input.mul_(grad_scale.to(grad_output.dtype)[:, None])
        return grad_input, None, None, None, None, None


class MXFP8AllToAllExpertParallel(ExpertParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One instance is created per MoE layer, so these slots are per-layer.
        # Each holds the payload received in the forward pass until the matching
        # activation-checkpoint recompute consumes it.
        self._dispatch_cache = [None] if _EP_AC_CACHE else None
        self._combine_cache = [None] if _EP_AC_CACHE else None
        self._meta_cache = None

    def _cache_slot(self, slot):
        """Caching only applies while building a graph that will be recomputed."""
        if slot is None or not torch.is_grad_enabled():
            return None
        return slot

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        if _EP_AC_CACHE and _in_backward() and self._meta_cache is not None:
            num_tokens_per_expert_group, self.input_splits, self.output_splits = self._meta_cache
            self._meta_cache = None
        else:
            with torch.no_grad():
                num_tokens_per_expert_group = num_tokens_per_expert.new_empty(num_tokens_per_expert.shape[0])
                with _EPTimer("meta"):
                    dist.all_to_all_single(
                        num_tokens_per_expert_group, num_tokens_per_expert, group=device_mesh.get_group()
                    )
                self.input_splits = num_tokens_per_expert.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
                self.output_splits = num_tokens_per_expert_group.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
            if _EP_AC_CACHE and not _in_backward() and torch.is_grad_enabled():
                self._meta_cache = (num_tokens_per_expert_group, self.input_splits, self.output_splits)
        _ep_profile_tick()
        routed_input = _INT4Dispatch.apply(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group().group_name,
            "dispatch",
            self._cache_slot(self._dispatch_cache),
        )
        return routed_input, num_tokens_per_expert_group

    def _token_combine(self, mod, routed_output, device_mesh):
        return _INT4Dispatch.apply(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group().group_name,
            "combine",
            self._cache_slot(self._combine_cache),
        )


class DeepEPExpertParallel(ParallelStyle):
    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(
                param_name,
                nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(0)]),
                    requires_grad=param.requires_grad,
                ),
            )
        mod._ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)


def get_ep_group(experts: nn.Module) -> ProcessGroup:
    return experts._ep_group
