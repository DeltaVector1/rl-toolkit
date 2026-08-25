import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import triton
import triton.language as tl
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _resolve_process_group
from torch.distributed.tensor import (
    DeviceMesh,
    Shard,
    distribute_module,
    distribute_tensor,
)
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
    row_scale = tl.maximum(
        tl.maximum(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 7.0, 1e-8
    )
    q0 = tl.maximum(-7.0, tl.minimum(7.0, libdevice.rint(x0 / row_scale))).to(tl.int32)
    q1 = tl.maximum(-7.0, tl.minimum(7.0, libdevice.rint(x1 / row_scale))).to(tl.int32)
    tl.store(
        packed + row * ((cols + 1) // 2) + pair,
        (q0 + 8) | ((q1 + 8) << 4),
        mask=col0 < cols,
    )
    tl.store(scale + row, row_scale)


@triton.jit
def _dequantize_int4_kernel(
    packed, scale, output, cols: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    pair = tl.arange(0, block)
    col0 = pair * 2
    value = tl.load(
        packed + row * ((cols + 1) // 2) + pair, mask=col0 < cols, other=0
    ).to(tl.int32)
    row_scale = tl.load(scale + row)
    tl.store(
        output + row * cols + col0, ((value & 15) - 8) * row_scale, mask=col0 < cols
    )
    tl.store(
        output + row * cols + col0 + 1,
        ((value >> 4) - 8) * row_scale,
        mask=col0 + 1 < cols,
    )


# When selective activation checkpointing targets "routed_experts", the whole
# dispatch -> experts -> combine region is re-run during backward, which repeats
# both all-to-alls. On this cluster the all-to-all is the dominant cost, so we
# keep the *received* INT4 payload from the forward pass and reuse it on the
# recompute. This is numerically exact: the recompute would dequantize the same
# bytes. The payload is 1/4 the size of the dequantized bf16 activation.
_EP_AC_CACHE = os.environ.get("PRIME_EP_AC_CACHE", "") == "1"
# The dispatch and combine outputs are full BF16 activations. A one-shot INT4
# all-to-all holds the full packed send and receive buffers while it allocates
# that activation. On a 32K-row GLM microbatch this adds about 192 MiB
# immediately before a 384 to 500 MiB allocation. Split both directions into
# a fixed number of rounds so every rank executes the same collective sequence
# and peak packed-buffer memory falls in proportion to this value.
_INT4_COMBINE_ROUNDS = int(os.environ.get("PRIME_INT4_COMBINE_ROUNDS", "1"))
if _INT4_COMBINE_ROUNDS < 1:
    raise ValueError("PRIME_INT4_COMBINE_ROUNDS must be at least 1")


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


def _empty_with_cache_retry(*size, **kwargs):
    """Allocate once more after releasing only unused CUDA allocator blocks."""
    try:
        return torch.empty(*size, **kwargs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return torch.empty(*size, **kwargs)
    _ep_t0 = time.perf_counter()


def _round_splits(
    splits: list[int], round_idx: int, rounds: int
) -> tuple[list[int], list[int]]:
    """Return this round's peer counts and offsets in the full peer-major buffer."""
    counts = []
    offsets = []
    peer_base = 0
    for split in splits:
        width, extra = divmod(split, rounds)
        counts.append(width + int(round_idx < extra))
        offsets.append(peer_base + round_idx * width + min(round_idx, extra))
        peer_base += split
    return counts, offsets


class ReceivedINT4ExpertInput:
    """Own the exact compressed expert input received by one dispatch call."""

    __slots__ = ("values", "scales", "columns", "copy_done", "copy_ready")

    def __init__(self) -> None:
        self.values = None
        self.scales = None
        self.columns = 0
        self.copy_done = None
        self.copy_ready = False

    def prepare(self, rows: int, columns: int) -> None:
        if self.values is not None:
            raise RuntimeError("received INT4 expert input was prepared more than once")
        if rows < 0 or columns <= 0:
            raise ValueError("received INT4 expert input shape is invalid")
        self.columns = columns
        self.values = torch.empty(
            (rows, (columns + 1) // 2),
            device="cpu",
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.scales = torch.empty(
            rows,
            device="cpu",
            dtype=torch.float16,
            pin_memory=True,
        )

    def copy_slices(
        self,
        values: torch.Tensor,
        scales: torch.Tensor,
        counts: list[int],
        source_offsets: list[int],
        destination_offsets: list[int],
    ) -> None:
        if self.values is None or self.scales is None:
            raise RuntimeError("received INT4 expert input was not prepared")
        if not (len(counts) == len(source_offsets) == len(destination_offsets)):
            raise ValueError("received INT4 slice metadata lengths do not match")
        for count, source, destination in zip(
            counts,
            source_offsets,
            destination_offsets,
        ):
            if count < 0 or source < 0 or destination < 0:
                raise ValueError("received INT4 slice metadata cannot be negative")
            if source + count > values.shape[0] or source + count > scales.shape[0]:
                raise ValueError("received INT4 source slice is out of bounds")
            if (
                destination + count > self.values.shape[0]
                or destination + count > self.scales.shape[0]
            ):
                raise ValueError("received INT4 destination slice is out of bounds")
            if not count:
                continue
            self.values[destination : destination + count].copy_(
                values[source : source + count],
                non_blocking=True,
            )
            self.scales[destination : destination + count].copy_(
                scales[source : source + count],
                non_blocking=True,
            )

    def finish_copy(self, device: torch.device) -> None:
        if self.values is None or self.scales is None:
            raise RuntimeError("received INT4 expert input was not prepared")
        if self.copy_done is not None:
            raise RuntimeError(
                "received INT4 expert input copy was finished more than once"
            )
        self.copy_done = torch.cuda.Event()
        self.copy_done.record(torch.cuda.current_stream(device))

    def reconstruct_rows(
        self,
        row_indices: torch.Tensor,
        output: torch.Tensor,
        workspace,
    ) -> None:
        if self.values is None or self.scales is None or self.copy_done is None:
            raise RuntimeError("received INT4 expert input is incomplete")
        if row_indices.device.type != "cpu" or row_indices.ndim != 1:
            raise ValueError(
                "received INT4 row indices must be a one-dimensional CPU tensor"
            )
        if output.device.type != "cuda" or output.dtype != torch.bfloat16:
            raise ValueError(
                "received INT4 rows must be reconstructed into CUDA BF16 storage"
            )
        if output.shape != (row_indices.shape[0], self.columns):
            raise ValueError("received INT4 reconstruction output has the wrong shape")
        workspace.check(row_indices.shape[0], self.columns, output.device)
        if not row_indices.shape[0]:
            return
        if not self.values.shape[0]:
            raise ValueError(
                "cannot reconstruct rows from an empty received INT4 input"
            )

        if not self.copy_ready:
            self.copy_done.synchronize()
            self.copy_ready = True
        real_rows = (row_indices >= 0) & (row_indices < self.values.shape[0])
        safe_indices = row_indices.clamp(0, self.values.shape[0] - 1).long()
        host_values = workspace.host_values[: row_indices.shape[0]]
        host_scales = workspace.host_scales[: row_indices.shape[0]]
        torch.index_select(self.values, 0, safe_indices, out=host_values)
        torch.index_select(self.scales, 0, safe_indices, out=host_scales)

        device_values = workspace.device_values[: row_indices.shape[0]]
        device_scales = workspace.device_scales[: row_indices.shape[0]]
        device_values.copy_(host_values, non_blocking=True)
        device_scales.copy_(host_scales, non_blocking=True)
        _dequantize_int4_kernel[(row_indices.shape[0],)](
            device_values,
            device_scales,
            output,
            self.columns,
            triton.next_power_of_2((self.columns + 1) // 2),
            num_warps=8,
        )
        output.masked_fill_((~real_rows).to(output.device).unsqueeze(-1), 0)

    def make_workspace(
        self, row_capacity: int, device: torch.device
    ) -> "INT4ExpertInputWorkspace":
        return INT4ExpertInputWorkspace(row_capacity, self.columns, device)


class INT4ExpertInputWorkspace:
    """Bound host and GPU scratch for compressed expert-input reconstruction."""

    __slots__ = (
        "row_capacity",
        "columns",
        "device",
        "host_values",
        "host_scales",
        "device_values",
        "device_scales",
    )

    def __init__(self, row_capacity: int, columns: int, device: torch.device) -> None:
        if row_capacity <= 0 or columns <= 0:
            raise ValueError("INT4 expert input workspace shape is invalid")
        self.row_capacity = row_capacity
        self.columns = columns
        self.device = torch.device(device)
        packed_columns = (columns + 1) // 2
        self.host_values = torch.empty(
            (row_capacity, packed_columns),
            device="cpu",
            dtype=torch.uint8,
            pin_memory=True,
        )
        self.host_scales = torch.empty(
            row_capacity,
            device="cpu",
            dtype=torch.float16,
            pin_memory=True,
        )
        self.device_values = torch.empty(
            (row_capacity, packed_columns),
            device=self.device,
            dtype=torch.uint8,
        )
        self.device_scales = torch.empty(
            row_capacity,
            device=self.device,
            dtype=torch.float16,
        )

    def check(self, rows: int, columns: int, device: torch.device) -> None:
        if rows > self.row_capacity:
            raise ValueError("INT4 expert input workspace row capacity is too small")
        if columns != self.columns or torch.device(device) != self.device:
            raise ValueError("INT4 expert input workspace does not match the output")


class ReceivedBF16ExpertInput:
    """Own the exact BF16 expert input received by one dispatch call."""

    __slots__ = (
        "values",
        "columns",
        "copy_done",
        "copy_ready",
        "save_for_backward",
    )

    def __init__(self, save_for_backward: bool = True) -> None:
        self.values = None
        self.columns = 0
        self.copy_done = None
        self.copy_ready = False
        self.save_for_backward = save_for_backward

    def prepare(self, values: torch.Tensor) -> None:
        if self.values is not None:
            raise RuntimeError("received BF16 expert input was prepared more than once")
        if values.device.type != "cuda" or values.dtype != torch.bfloat16:
            raise TypeError("received BF16 expert input must be a CUDA BF16 tensor")
        if values.ndim != 2 or values.shape[1] <= 0:
            raise ValueError("received BF16 expert input shape is invalid")
        self.columns = values.shape[1]
        if not self.save_for_backward:
            return
        self.values = torch.empty(
            values.shape,
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        self.values.copy_(values, non_blocking=True)
        self.copy_done = torch.cuda.Event()
        self.copy_done.record(torch.cuda.current_stream(values.device))

    def reconstruct_rows(
        self,
        row_indices: torch.Tensor,
        output: torch.Tensor,
        workspace,
    ) -> None:
        if self.values is None or self.copy_done is None:
            raise RuntimeError("received BF16 expert input is incomplete")
        if row_indices.device.type != "cpu" or row_indices.ndim != 1:
            raise ValueError(
                "received BF16 row indices must be a one-dimensional CPU tensor"
            )
        if output.device.type != "cuda" or output.dtype != torch.bfloat16:
            raise ValueError(
                "received BF16 rows must be reconstructed into CUDA BF16 storage"
            )
        if output.shape != (row_indices.shape[0], self.columns):
            raise ValueError("received BF16 reconstruction output has the wrong shape")
        workspace.check(row_indices.shape[0], self.columns, output.device)
        if not row_indices.shape[0]:
            return
        if not self.values.shape[0]:
            raise ValueError("cannot reconstruct rows from an empty received BF16 input")

        if not self.copy_ready:
            self.copy_done.synchronize()
            self.copy_ready = True
        real_rows = (row_indices >= 0) & (row_indices < self.values.shape[0])
        safe_indices = row_indices.clamp(0, self.values.shape[0] - 1).long()
        host_values = workspace.host_values[: row_indices.shape[0]]
        torch.index_select(self.values, 0, safe_indices, out=host_values)
        output.copy_(host_values, non_blocking=True)
        output.masked_fill_((~real_rows).to(output.device).unsqueeze(-1), 0)

    def make_workspace(
        self, row_capacity: int, device: torch.device
    ) -> "BF16ExpertInputWorkspace":
        return BF16ExpertInputWorkspace(row_capacity, self.columns, device)


class BF16ExpertInputWorkspace:
    """Bound host scratch for exact BF16 expert-input reconstruction."""

    __slots__ = ("row_capacity", "columns", "device", "host_values")

    def __init__(self, row_capacity: int, columns: int, device: torch.device) -> None:
        if row_capacity <= 0 or columns <= 0:
            raise ValueError("BF16 expert input workspace shape is invalid")
        self.row_capacity = row_capacity
        self.columns = columns
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("BF16 expert input workspace requires a CUDA device")
        self.host_values = torch.empty(
            (row_capacity, columns),
            device="cpu",
            dtype=torch.bfloat16,
            pin_memory=True,
        )

    def check(self, rows: int, columns: int, device: torch.device) -> None:
        if rows > self.row_capacity:
            raise ValueError("BF16 expert input workspace row capacity is too small")
        if columns != self.columns or torch.device(device) != self.device:
            raise ValueError("BF16 expert input workspace does not match the output")


def bf16_expert_transport_memory_bytes(
    input_rows: int,
    output_rows: int,
    columns: int,
    reconstruction_rows: int,
) -> dict[str, int]:
    """Return the added BF16 transport storage for one expert call."""
    if min(input_rows, output_rows) < 0 or columns <= 0 or reconstruction_rows <= 0:
        raise ValueError("BF16 expert transport memory shape is invalid")
    element_bytes = torch.tensor([], dtype=torch.bfloat16).element_size()
    return {
        "dispatch_output_cuda": output_rows * columns * element_bytes,
        "combine_output_cuda": input_rows * columns * element_bytes,
        "saved_dispatch_input_host": output_rows * columns * element_bytes,
        "reconstruction_workspace_host": reconstruction_rows
        * columns
        * element_bytes,
    }


def _streaming_int4_forward(
    routed_input,
    output_splits,
    input_splits,
    group_name,
    tag,
    rounds,
    reuse_input,
    received_input=None,
):
    """Run an INT4 all-to-all with bounded send and receive rounds."""
    rows, cols = routed_input.shape
    packed_cols = (cols + 1) // 2
    block = triton.next_power_of_2(packed_cols)
    output_rows = sum(output_splits)
    if received_input is not None:
        received_input.prepare(output_rows, cols)

    # The operation is the sole consumer of its input, and its backward does
    # not need the forward values. If the input allocation is large enough for
    # the returned rows, reuse it for the BF16 result. Quantize the complete
    # source before overwriting it in that case. If routing imbalance requires
    # a larger result, allocate that result before any packed transport buffer
    # and quantize only one send round at a time. The latter ordering removed a
    # measured step-8 dispatch peak where a 498 MiB result had only 413.5 MiB
    # physically free while full packed send and receive buffers were live.
    if reuse_input:
        # The input and final output share storage.  Do not overwrite any BF16
        # source rows until every round has sent them.  Retain the compact INT4
        # receive payload, quantize only one send round at a time, and expand
        # into the reserved BF16 storage after all collectives finish.  This
        # replaces a complete INT4 send copy with one round of send scratch.
        received_all = _empty_with_cache_retry(
            (output_rows, packed_cols),
            device=routed_input.device,
            dtype=torch.uint8,
        )
        received_scale_all = _empty_with_cache_retry(
            output_rows, device=routed_input.device, dtype=torch.float16
        )
        dequantized = routed_input
        storage_bytes = routed_input.untyped_storage().nbytes()
        logical_bytes = rows * cols * routed_input.element_size()
        tail_bytes = max(0, storage_bytes - logical_bytes)
    else:
        # Dispatch is followed by the inverse combine.  An EP rank can receive
        # fewer rows during dispatch than it must return during combine.  Give
        # the dispatch result enough storage for both shapes while memory is
        # still available.  The native-FP8 expert path operates in place and
        # preserves this storage, so combine can reuse it instead of trying to
        # allocate a complete hidden-size tensor at the later memory peak.
        capacity_rows = (
            max(rows, output_rows) if tag.startswith("dispatch") else output_rows
        )
        dequantized = _empty_with_cache_retry(
            (capacity_rows, cols), device=routed_input.device, dtype=routed_input.dtype
        )
        if capacity_rows != output_rows:
            dequantized.as_strided_((output_rows, cols), (cols, 1))
    group = _resolve_process_group(group_name)

    for round_idx in range(rounds):
        input_counts, input_offsets = _round_splits(input_splits, round_idx, rounds)
        output_counts, output_offsets = _round_splits(output_splits, round_idx, rounds)
        send_rows = sum(input_counts)
        receive_rows = sum(output_counts)

        round_packed_bytes = send_rows * packed_cols
        if reuse_input and round_packed_bytes <= tail_bytes:
            # Dispatch reserved BF16 capacity for a larger inverse-combine
            # result.  Before expansion, that tail is outside the live input.
            # Use it as the current round's packed-send scratch.  No source row
            # is overwritten, and all sends finish before final dequantization
            # writes into the expanded BF16 result.
            round_quantized = torch.empty(
                0, device=routed_input.device, dtype=torch.uint8
            ).set_(
                routed_input.untyped_storage(),
                logical_bytes,
                (send_rows, packed_cols),
                (packed_cols, 1),
            )
        else:
            round_quantized = _empty_with_cache_retry(
                (send_rows, packed_cols),
                device=routed_input.device,
                dtype=torch.uint8,
            )
        round_scale = _empty_with_cache_retry(
            send_rows, device=routed_input.device, dtype=torch.float16
        )
        send = []
        send_scale = []
        send_cursor = 0
        for count, offset in zip(input_counts, input_offsets):
            packed_part = round_quantized[send_cursor : send_cursor + count]
            scale_part = round_scale[send_cursor : send_cursor + count]
            if count:
                _quantize_int4_kernel[(count,)](
                    routed_input[offset : offset + count],
                    packed_part,
                    scale_part,
                    cols,
                    block,
                    num_warps=8,
                )
            send.append(packed_part)
            send_scale.append(scale_part)
            send_cursor += count

        if reuse_input:
            received = received_all
            received_scale = received_scale_all
            received_offsets = output_offsets
        else:
            received = _empty_with_cache_retry(
                (receive_rows, packed_cols),
                device=routed_input.device,
                dtype=torch.uint8,
            )
            received_scale = _empty_with_cache_retry(
                receive_rows, device=routed_input.device, dtype=torch.float16
            )
            received_offsets = []
            cursor = 0
            for count in output_counts:
                received_offsets.append(cursor)
                cursor += count
        receive = [
            received[offset : offset + count]
            for count, offset in zip(output_counts, received_offsets)
        ]
        receive_scale = [
            received_scale[offset : offset + count]
            for count, offset in zip(output_counts, received_offsets)
        ]
        with _EPTimer(f"fwd_{tag}"):
            dist.all_to_all(receive, send, group=group)
            dist.all_to_all(receive_scale, send_scale, group=group)

        if received_input is not None:
            if reuse_input:
                receive_source_offsets = output_offsets
            else:
                receive_source_offsets = []
                receive_cursor = 0
                for count in output_counts:
                    receive_source_offsets.append(receive_cursor)
                    receive_cursor += count
            received_input.copy_slices(
                received,
                received_scale,
                output_counts,
                receive_source_offsets,
                output_offsets,
            )

        if not reuse_input:
            cursor = 0
            for count, offset in zip(output_counts, output_offsets):
                if count:
                    _dequantize_int4_kernel[(count,)](
                        received[cursor : cursor + count],
                        received_scale[cursor : cursor + count],
                        dequantized[offset : offset + count],
                        cols,
                        block,
                        num_warps=8,
                    )
                    cursor += count

        del send, send_scale, receive, receive_scale, round_quantized, round_scale
        if not reuse_input:
            del received, received_scale

    if reuse_input:
        # as_strided_ changes only tensor metadata.  The dispatch allocation
        # reserved enough rows for the inverse-combine result.
        routed_input.as_strided_((output_rows, cols), (cols, 1))
        _dequantize_int4_kernel[(output_rows,)](
            received_all,
            received_scale_all,
            routed_input,
            cols,
            block,
            num_warps=8,
        )

    if received_input is not None:
        received_input.finish_copy(routed_input.device)

    return dequantized


def _streaming_fp8_backward(
    grad_output,
    output_splits,
    input_splits,
    group_name,
    tag,
    rounds,
):
    """Return the inverse all-to-all gradient with bounded receive memory.

    The forward combine output and its input gradient normally have the same
    32K routed-row capacity. Reuse the consumed incoming-gradient buffer when
    it is large enough. This avoids allocating a second 384 MiB BF16 tensor at
    the backward peak. Imbalanced ranks whose returned gradient is larger use
    a new result, but still receive only one FP8 round at a time.
    """
    rows, cols = grad_output.shape
    input_rows = sum(input_splits)
    storage_rows = 0
    if grad_output.is_contiguous():
        storage_elements = (
            grad_output.untyped_storage().nbytes() // grad_output.element_size()
        )
        storage_rows = (storage_elements - grad_output.storage_offset()) // cols
    reuse_output = input_rows <= storage_rows
    if reuse_output:
        quantized, scale = ops.scaled_fp8_quant(
            grad_output, use_per_token_if_dynamic=True
        )
        scale = scale.flatten()
        grad_output.as_strided_((input_rows, cols), (cols, 1))
        dequantized = grad_output
    else:
        dequantized = _empty_with_cache_retry(
            (input_rows, cols), device=grad_output.device, dtype=grad_output.dtype
        )
    group = _resolve_process_group(group_name)

    # Backward reverses forward: send the forward receive segments and receive
    # the forward send segments. Every rank executes the same round count.
    for round_idx in range(rounds):
        send_counts, send_offsets = _round_splits(output_splits, round_idx, rounds)
        receive_counts, receive_offsets = _round_splits(input_splits, round_idx, rounds)
        receive_rows = sum(receive_counts)

        if reuse_output:
            send = [
                quantized[offset : offset + count]
                for count, offset in zip(send_counts, send_offsets)
            ]
            send_scale = [
                scale[offset : offset + count]
                for count, offset in zip(send_counts, send_offsets)
            ]
        else:
            # Allocate the BF16 result before this loop, then retain only the
            # current round's FP8 sources. Per-token scaling makes independent
            # peer-slice quantization identical to quantizing the full tensor.
            send = []
            send_scale = []
            for count, offset in zip(send_counts, send_offsets):
                if count:
                    quantized_part, scale_part = ops.scaled_fp8_quant(
                        grad_output[offset : offset + count],
                        use_per_token_if_dynamic=True,
                    )
                    scale_part = scale_part.flatten()
                else:
                    quantized_part = torch.empty(
                        (0, cols), device=grad_output.device, dtype=torch.float8_e4m3fn
                    )
                    scale_part = torch.empty(
                        0, device=grad_output.device, dtype=torch.float32
                    )
                send.append(quantized_part)
                send_scale.append(scale_part)

        # A production rank failed here with enough reserved-but-unused memory
        # for this bounded receive. A true live-memory OOM remains fatal on the
        # retry because the requested shape and lifetime do not change.
        received = _empty_with_cache_retry(
            (receive_rows, cols), device=grad_output.device, dtype=send[0].dtype
        )
        received_scale = torch.empty(
            receive_rows, device=grad_output.device, dtype=send_scale[0].dtype
        )
        receive_round_offsets = []
        cursor = 0
        for count in receive_counts:
            receive_round_offsets.append(cursor)
            cursor += count
        receive = [
            received[offset : offset + count]
            for count, offset in zip(receive_counts, receive_round_offsets)
        ]
        receive_scale = [
            received_scale[offset : offset + count]
            for count, offset in zip(receive_counts, receive_round_offsets)
        ]
        with _EPTimer(f"bwd_{tag}"):
            dist.all_to_all(receive, send, group=group)
            dist.all_to_all(receive_scale, send_scale, group=group)

        cursor = 0
        for count, offset in zip(receive_counts, receive_offsets):
            if count:
                target = dequantized[offset : offset + count]
                target.copy_(received[cursor : cursor + count])
                target.mul_(
                    received_scale[cursor : cursor + count].to(target.dtype)[:, None]
                )
                cursor += count

        del received, received_scale, send, send_scale, receive, receive_scale

    return dequantized


def _validate_bf16_dispatch(
    value: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    group_name: str,
) -> ProcessGroup:
    if value.device.type != "cuda" or value.dtype != torch.bfloat16:
        raise TypeError("BF16 expert transport requires a CUDA BF16 tensor")
    if value.ndim != 2 or not value.is_contiguous() or value.shape[1] <= 0:
        raise ValueError("BF16 expert transport requires a contiguous matrix")
    if any(split < 0 for split in (*output_splits, *input_splits)):
        raise ValueError("BF16 expert transport splits cannot be negative")
    group = _resolve_process_group(group_name)
    world_size = dist.get_world_size(group)
    if len(output_splits) != world_size or len(input_splits) != world_size:
        raise ValueError("BF16 expert transport needs one split for every EP rank")
    if sum(input_splits) != value.shape[0]:
        raise ValueError("BF16 expert transport input splits do not match the input")
    return group


class _BF16Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        routed_input,
        output_splits,
        input_splits,
        group_name,
        tag="a2a",
        received_input=None,
    ):
        group = _validate_bf16_dispatch(
            routed_input,
            output_splits,
            input_splits,
            group_name,
        )
        output = _empty_with_cache_retry(
            (sum(output_splits), routed_input.shape[1]),
            device=routed_input.device,
            dtype=torch.bfloat16,
        )
        with _EPTimer(f"fwd_{tag}"):
            dist.all_to_all_single(
                output,
                routed_input,
                output_splits,
                input_splits,
                group=group,
            )
        if received_input is not None:
            if not isinstance(received_input, ReceivedBF16ExpertInput):
                raise TypeError("BF16 expert dispatch received the wrong saved-input type")
            received_input.prepare(output)
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group_name = group_name
        ctx.tag = tag
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = _validate_bf16_dispatch(
            grad_output,
            ctx.input_splits,
            ctx.output_splits,
            ctx.group_name,
        )
        grad_input = _empty_with_cache_retry(
            (sum(ctx.input_splits), grad_output.shape[1]),
            device=grad_output.device,
            dtype=torch.bfloat16,
        )
        with _EPTimer(f"bwd_{ctx.tag}"):
            dist.all_to_all_single(
                grad_input,
                grad_output,
                ctx.input_splits,
                ctx.output_splits,
                group=group,
            )
        return grad_input, None, None, None, None, None


class _INT4Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        routed_input,
        output_splits,
        input_splits,
        group_name,
        tag="a2a",
        cache=None,
        received_input=None,
    ):
        ctx.tag = tag
        ctx.rounds = (
            _INT4_COMBINE_ROUNDS
            if tag in {"dispatch", "combine"} and cache is None
            else 1
        )
        if ctx.rounds > 1:
            ctx.input_splits = input_splits
            ctx.output_splits = output_splits
            ctx.group_name = group_name
            ctx.cols = routed_input.shape[1]
            output_rows = sum(output_splits)
            # Dispatch receives an FSDP post-backward view. Returning and
            # modifying that view from this custom autograd function is
            # forbidden because it would replace FSDP's custom backward.
            # Expert-output combine owns its input allocation and can reuse it.
            storage_rows = 0
            if routed_input.is_contiguous() and routed_input.storage_offset() == 0:
                storage_rows = (
                    routed_input.untyped_storage().nbytes()
                    // routed_input.element_size()
                    // routed_input.shape[1]
                )
            reuse_input = tag == "combine" and output_rows <= storage_rows
            if reuse_input:
                ctx.mark_dirty(routed_input)
            return _streaming_int4_forward(
                routed_input,
                output_splits,
                input_splits,
                group_name,
                tag,
                ctx.rounds,
                reuse_input,
                received_input,
            )
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
            quantized = torch.empty(
                (rows, packed_cols), device=routed_input.device, dtype=torch.uint8
            )
            scale = torch.empty(rows, device=routed_input.device, dtype=torch.float16)
            _quantize_int4_kernel[(rows,)](
                routed_input, quantized, scale, cols, block, num_warps=8
            )
            output = quantized.new_empty((sum(output_splits), packed_cols))
            output_scale = scale.new_empty(sum(output_splits))
            group = _resolve_process_group(group_name)
            with _EPTimer(f"fwd_{tag}"):
                dist.all_to_all_single(
                    output, quantized, output_splits, input_splits, group=group
                )
                dist.all_to_all_single(
                    output_scale, scale, output_splits, input_splits, group=group
                )
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
        dequantized = torch.empty(
            (sum(output_splits), cols), device=output.device, dtype=routed_input.dtype
        )
        _dequantize_int4_kernel[(sum(output_splits),)](
            output, output_scale, dequantized, cols, block, num_warps=8
        )
        if received_input is not None:
            received_input.prepare(sum(output_splits), cols)
            received_input.copy_slices(
                output,
                output_scale,
                [sum(output_splits)],
                [0],
                [0],
            )
            received_input.finish_copy(output.device)
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.rounds > 1:
            return (
                _streaming_fp8_backward(
                    grad_output,
                    ctx.output_splits,
                    ctx.input_splits,
                    ctx.group_name,
                    ctx.tag,
                    ctx.rounds,
                ),
                None,
                None,
                None,
                None,
                None,
                None,
            )
        quantized, scale = ops.scaled_fp8_quant(
            grad_output, use_per_token_if_dynamic=True
        )
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
        return grad_input, None, None, None, None, None, None


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

    def _exchange_metadata(self, num_tokens_per_expert, device_mesh):
        if _EP_AC_CACHE and _in_backward() and self._meta_cache is not None:
            num_tokens_per_expert_group, self.input_splits, self.output_splits = (
                self._meta_cache
            )
            self._meta_cache = None
        else:
            with torch.no_grad():
                num_tokens_per_expert_group = num_tokens_per_expert.new_empty(
                    num_tokens_per_expert.shape[0]
                )
                with _EPTimer("meta"):
                    dist.all_to_all_single(
                        num_tokens_per_expert_group,
                        num_tokens_per_expert,
                        group=device_mesh.get_group(),
                    )
                self.input_splits = (
                    num_tokens_per_expert.view(device_mesh.shape[0], -1)
                    .sum(dim=1)
                    .tolist()
                )
                self.output_splits = (
                    num_tokens_per_expert_group.view(device_mesh.shape[0], -1)
                    .sum(dim=1)
                    .tolist()
                )
            if _EP_AC_CACHE and not _in_backward() and torch.is_grad_enabled():
                self._meta_cache = (
                    num_tokens_per_expert_group,
                    self.input_splits,
                    self.output_splits,
                )
        return num_tokens_per_expert_group

    def _token_dispatch(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        num_tokens_per_expert_group = self._exchange_metadata(
            num_tokens_per_expert,
            device_mesh,
        )
        _ep_profile_tick()
        received_input = (
            ReceivedINT4ExpertInput()
            if getattr(mod, "requires_received_int4_input", False)
            else None
        )
        routed_input = _INT4Dispatch.apply(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group().group_name,
            "dispatch",
            self._cache_slot(self._dispatch_cache),
            received_input,
        )
        if received_input is None:
            return routed_input, num_tokens_per_expert_group
        return routed_input, num_tokens_per_expert_group, received_input

    def _token_combine(self, mod, routed_output, device_mesh):
        return _INT4Dispatch.apply(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group().group_name,
            "combine",
            self._cache_slot(self._combine_cache),
            None,
        )


class NativeFP8LoRAAllToAllExpertParallel(MXFP8AllToAllExpertParallel):
    """Shard the native FP8 base and every routed-expert LoRA factor."""

    def __init__(self, expert_activation_transport: str):
        super().__init__()
        if expert_activation_transport not in {"bf16", "int4"}:
            raise ValueError(
                "native FP8 routed-expert LoRA transport must be 'bf16' or 'int4'"
            )
        self.expert_activation_transport = expert_activation_transport

    def _token_dispatch(self, mod, inputs, device_mesh):
        if not getattr(mod, "requires_received_expert_input", False):
            raise TypeError("native FP8 routed-expert LoRA needs its received input")
        routed_input, num_tokens_per_expert = inputs
        num_tokens_per_expert_group = self._exchange_metadata(
            num_tokens_per_expert,
            device_mesh,
        )
        _ep_profile_tick()
        if self.expert_activation_transport == "bf16":
            received_input = ReceivedBF16ExpertInput(
                save_for_backward=torch.is_grad_enabled()
            )
            routed_input = _BF16Dispatch.apply(
                routed_input,
                self.output_splits,
                self.input_splits,
                device_mesh.get_group().group_name,
                "dispatch",
                received_input,
            )
        elif self.expert_activation_transport == "int4":
            received_input = ReceivedINT4ExpertInput()
            routed_input = _INT4Dispatch.apply(
                routed_input,
                self.output_splits,
                self.input_splits,
                device_mesh.get_group().group_name,
                "dispatch",
                self._cache_slot(self._dispatch_cache),
                received_input,
            )
        else:
            raise RuntimeError("native FP8 routed-expert LoRA transport is invalid")
        return routed_input, num_tokens_per_expert_group, received_input

    def _token_combine(self, mod, routed_output, device_mesh):
        if self.expert_activation_transport == "bf16":
            return _BF16Dispatch.apply(
                routed_output,
                self.input_splits,
                self.output_splits,
                device_mesh.get_group().group_name,
                "combine",
                None,
            )
        if self.expert_activation_transport == "int4":
            return _INT4Dispatch.apply(
                routed_output,
                self.input_splits,
                self.output_splits,
                device_mesh.get_group().group_name,
                "combine",
                self._cache_slot(self._combine_cache),
                None,
            )
        raise RuntimeError("native FP8 routed-expert LoRA transport is invalid")

    @staticmethod
    def _shard_direct_parameters(mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for parameter_name, parameter in list(mod.named_parameters(recurse=False)):
            mod.register_parameter(
                parameter_name,
                nn.Parameter(
                    distribute_tensor(parameter, device_mesh, [Shard(0)]),
                    requires_grad=parameter.requires_grad,
                ),
            )

    @classmethod
    def _partition_fn(
        cls,
        name: str,
        mod: nn.Module,
        device_mesh: DeviceMesh,
    ) -> None:
        if not getattr(mod, "requires_received_expert_input", False):
            return
        base_layer = getattr(mod, "base_layer", None)
        if base_layer is None or not getattr(base_layer, "fp8_checkpoint", False):
            raise TypeError("native FP8 LoRA expert parallelism requires an FP8 base layer")
        cls._shard_direct_parameters(base_layer, device_mesh)
        for projection in ("w1", "w2", "w3"):
            for factor in ("A", "B"):
                parameter_list = getattr(mod, f"{projection}_lora_{factor}", None)
                if not isinstance(parameter_list, nn.ParameterList):
                    raise TypeError(
                        f"native FP8 LoRA expert module has no {projection} factor {factor}"
                    )
                cls._shard_direct_parameters(parameter_list, device_mesh)
        mod._ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
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
