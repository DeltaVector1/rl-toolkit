"""Exact FP8 routed-expert transport with source-side LoRA-A projections.

The native FP8 expert kernel quantizes each input row before its FP8 weight
GEMM.  This module moves that same ``(E4M3 value, FP32 row scale)`` pair over
expert parallel transport.  The two rank-sized gate/up LoRA-A projections are
computed while the original BF16 input is still on the source rank.

This is an opt-in protocol.  The existing BF16 transport remains the caller's
default path.  This module does not silently fall back to BF16 or reconstruct
LoRA gradients from a dequantized FP8 input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch


TransportMode = Literal["bf16", "fp8_lora_a"]
DEFAULT_TRANSPORT_MODE: TransportMode = "bf16"
FP8_LORA_A_TRANSPORT_MODE: TransportMode = "fp8_lora_a"


def _check_splits(
    input_rows: int,
    output_splits: Sequence[int],
    input_splits: Sequence[int],
) -> None:
    if not output_splits or len(output_splits) != len(input_splits):
        raise ValueError("expert transport needs one input and output split per rank")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (*output_splits, *input_splits)
    ):
        raise ValueError("expert transport splits must be non-negative integers")
    if sum(input_splits) != input_rows:
        raise ValueError(
            f"expert transport input splits sum to {sum(input_splits)}, expected {input_rows}"
        )


def _round_splits(
    splits: Sequence[int], round_index: int, rounds: int
) -> tuple[list[int], list[int]]:
    """Return peer-major row counts and offsets for one fixed collective round."""

    counts: list[int] = []
    offsets: list[int] = []
    peer_base = 0
    for split in splits:
        width, extra = divmod(split, rounds)
        counts.append(width + int(round_index < extra))
        offsets.append(peer_base + round_index * width + min(round_index, extra))
        peer_base += split
    return counts, offsets


def _validate_rounds(rounds: int) -> None:
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise ValueError("expert transport rounds must be a positive integer")


def _check_matrix(value: torch.Tensor, name: str, *, dtype: torch.dtype | None = None) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous matrix")
    if value.shape[1] <= 0:
        raise ValueError(f"{name} must have a positive width")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}, got {value.dtype}")


def _check_vector(value: torch.Tensor, name: str, rows: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows},)")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _default_rowwise_fp8_quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the same vLLM rowwise FP8 quantizer used by native experts.

    Importing vLLM lazily keeps the shape/accounting helpers usable by CPU
    tests.  A production caller must provide the same quantizer implementation
    used by its accepted native grouped GEMM; no approximate replacement is
    selected here.
    """

    try:
        from vllm import _custom_ops as ops
    except ImportError as exc:  # pragma: no cover - exercised only without runtime deps
        raise RuntimeError("fp8_lora_a transport requires vLLM FP8 quantization") from exc
    quantized, scale = ops.scaled_fp8_quant(
        value,
        use_per_token_if_dynamic=True,
    )
    scale = scale.reshape(-1)
    if quantized.dtype != torch.float8_e4m3fn:
        raise TypeError("native FP8 row transport requires E4M3 values")
    if scale.dtype != torch.float32 or scale.shape != (value.shape[0],):
        raise TypeError("native FP8 row transport requires one FP32 scale per row")
    return quantized, scale


def _all_to_all_single(
    receive: torch.Tensor,
    send: torch.Tensor,
    output_splits: Sequence[int],
    input_splits: Sequence[int],
    *,
    group: Any,
    collective: Any,
) -> None:
    """Run one real or test-injected all-to-all.

    ``collective`` is intentionally explicit.  Tests use a deterministic
    distributed fake; production passes ``None`` and uses the configured
    process group.  A one-rank CPU call is allowed without initialising a
    process group.
    """

    if collective is not None:
        collective.all_to_all_single(
            receive,
            send,
            output_splits,
            input_splits,
            group=group,
        )
        return
    if group is None and len(output_splits) == 1 and output_splits[0] == input_splits[0]:
        receive.copy_(send)
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("expert transport requires an initialized process group")
    torch.distributed.all_to_all_single(
        receive,
        send,
        output_splits,
        input_splits,
        group=group,
    )


def _all_to_all_rounds(
    send: torch.Tensor,
    output_splits: Sequence[int],
    input_splits: Sequence[int],
    rounds: int,
    *,
    group: Any,
    collective: Any,
) -> torch.Tensor:
    """Exchange rows in fixed rounds while retaining only compact payloads."""

    _check_splits(send.shape[0], output_splits, input_splits)
    _validate_rounds(rounds)
    receive = torch.empty(
        (sum(output_splits), send.shape[1]),
        device=send.device,
        dtype=send.dtype,
    )
    if rounds == 1:
        _all_to_all_single(
            receive,
            send,
            output_splits,
            input_splits,
            group=group,
            collective=collective,
        )
        return receive
    for round_index in range(rounds):
        send_counts, send_offsets = _round_splits(input_splits, round_index, rounds)
        receive_counts, receive_offsets = _round_splits(output_splits, round_index, rounds)
        send_rows = sum(send_counts)
        receive_rows = sum(receive_counts)
        round_send = torch.empty(
            (send_rows, send.shape[1]), device=send.device, dtype=send.dtype
        )
        round_receive = torch.empty(
            (receive_rows, send.shape[1]), device=send.device, dtype=send.dtype
        )
        cursor = 0
        for count, offset in zip(send_counts, send_offsets):
            if count:
                round_send[cursor : cursor + count].copy_(send[offset : offset + count])
            cursor += count
        _all_to_all_single(
            round_receive,
            round_send,
            receive_counts,
            send_counts,
            group=group,
            collective=collective,
        )
        cursor = 0
        for count, offset in zip(receive_counts, receive_offsets):
            if count:
                receive[offset : offset + count].copy_(round_receive[cursor : cursor + count])
            cursor += count
        del round_send, round_receive
    return receive


class _RowsAllToAll(torch.autograd.Function):
    """All-to-all for BF16 LoRA projections with an exact reverse VJP."""

    @staticmethod
    def forward(
        ctx,
        value: torch.Tensor,
        output_splits: Sequence[int],
        input_splits: Sequence[int],
        group: Any,
        collective: Any,
    ) -> torch.Tensor:
        _check_matrix(value, "LoRA projection", dtype=torch.bfloat16)
        _check_splits(value.shape[0], output_splits, input_splits)
        output = torch.empty(
            (sum(output_splits), value.shape[1]),
            device=value.device,
            dtype=value.dtype,
        )
        _all_to_all_single(
            output,
            value,
            output_splits,
            input_splits,
            group=group,
            collective=collective,
        )
        ctx.output_splits = tuple(output_splits)
        ctx.input_splits = tuple(input_splits)
        ctx.group = group
        ctx.collective = collective
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _check_matrix(grad_output, "LoRA projection gradient", dtype=torch.bfloat16)
        if grad_output.shape[0] != sum(ctx.output_splits):
            raise ValueError("LoRA projection gradient has the wrong row count")
        grad_input = torch.empty(
            (sum(ctx.input_splits), grad_output.shape[1]),
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        _all_to_all_single(
            grad_input,
            grad_output,
            ctx.input_splits,
            ctx.output_splits,
            group=ctx.group,
            collective=ctx.collective,
        )
        return grad_input, None, None, None, None


class _FP8InputAllToAll(torch.autograd.Function):
    """Move FP8 rows, but return the base GEMM VJP to the BF16 source rows.

    The FP8 values and scales have no differentiable path.  The expert kernel
    supplies a BF16 gradient for its dequantized input; this function sends
    that gradient back to the source rank.  This is the base-input VJP and is
    separate from the LoRA projection VJP handled by ``_RowsAllToAll``.
    """

    @staticmethod
    def forward(
        ctx,
        source_value: torch.Tensor,
        values: torch.Tensor,
        scales: torch.Tensor,
        output_splits: Sequence[int],
        input_splits: Sequence[int],
        group: Any,
        collective: Any,
        rounds: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_matrix(source_value, "source activation", dtype=torch.bfloat16)
        _check_matrix(values, "FP8 values", dtype=torch.float8_e4m3fn)
        if values.shape != source_value.shape:
            raise ValueError("FP8 values must have the source activation shape")
        _check_vector(scales, "FP8 row scales", source_value.shape[0])
        if scales.dtype != torch.float32 or scales.device != source_value.device:
            raise TypeError("FP8 row scales must be device-local FP32")
        _check_splits(source_value.shape[0], output_splits, input_splits)
        _validate_rounds(rounds)
        received_values = _all_to_all_rounds(
            values,
            output_splits,
            input_splits,
            rounds,
            group=group,
            collective=collective,
        )
        # The row scales are a vector; use the same round schedule so every
        # value keeps its scale association.  The helper expects a matrix.
        received_scales = _all_to_all_rounds(
            scales.unsqueeze(1),
            output_splits,
            input_splits,
            rounds,
            group=group,
            collective=collective,
        ).reshape(-1)
        # This dequantization is the input representation consumed by the
        # accepted native grouped GEMM.  LoRA backward never uses it to form
        # dA or dInput; those use source-side BF16 rows below.
        output = received_values.to(torch.bfloat16)
        output.mul_(received_scales.to(output.dtype).unsqueeze(-1))
        ctx.input_splits = tuple(input_splits)
        ctx.output_splits = tuple(output_splits)
        ctx.group = group
        ctx.collective = collective
        ctx.rounds = rounds
        # Keep the received payload available to a native grouped GEMM.  The
        # first output is its BF16 view; the latter two outputs are metadata,
        # not differentiable tensors.
        return output, received_values, received_scales

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        _grad_received_values: torch.Tensor | None,
        _grad_received_scales: torch.Tensor | None,
    ):
        _check_matrix(grad_output, "base input gradient", dtype=torch.bfloat16)
        if grad_output.shape[0] != sum(ctx.output_splits):
            raise ValueError("base input gradient has the wrong row count")
        grad_source = _all_to_all_rounds(
            grad_output,
            ctx.input_splits,
            ctx.output_splits,
            ctx.rounds,
            group=ctx.group,
            collective=ctx.collective,
        )
        return grad_source, None, None, None, None, None, None, None


def _select_lora_a(
    source_value: torch.Tensor,
    expert_indices: torch.Tensor,
    lora_a: torch.Tensor,
    name: str,
) -> torch.Tensor:
    _check_matrix(source_value, "source activation", dtype=torch.bfloat16)
    if expert_indices.ndim != 1 or expert_indices.shape[0] != source_value.shape[0]:
        raise ValueError(f"{name} expert indices must have one entry per source row")
    if expert_indices.dtype not in (torch.int32, torch.int64) or not expert_indices.is_contiguous():
        raise TypeError(f"{name} expert indices must be contiguous int32 or int64")
    if lora_a.ndim != 3 or lora_a.dtype != torch.bfloat16 or not lora_a.is_contiguous():
        raise TypeError(f"{name} LoRA A must be a contiguous BF16 [expert, rank, hidden] tensor")
    if lora_a.shape[2] != source_value.shape[1]:
        raise ValueError(f"{name} LoRA A hidden width does not match source activation")
    if expert_indices.numel() and (
        int(expert_indices.min()) < 0 or int(expert_indices.max()) >= lora_a.shape[0]
    ):
        raise ValueError(f"{name} expert index is outside the LoRA A table")
    selected = lora_a.index_select(0, expert_indices.to(torch.int64))
    # A has [rank, hidden], so x @ A.T is [rows, rank].  bmm retains the
    # source-side autograd path needed for dA and the original BF16 dInput.
    return torch.bmm(source_value.unsqueeze(1), selected.transpose(1, 2)).squeeze(1)


@dataclass
class FP8LoRAATransportState:
    """Forward state owned by one source/owner pair.

    ``source_value`` and the two ``source_*_projection`` tensors remain on the
    source rank.  The received rows are owner-side views.  They are separate
    so the backward graph can return d(projection) to the source without using
    dequantized FP8 input for dA or dInput.
    """

    source_value: torch.Tensor
    source_expert_indices: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor
    received_value: torch.Tensor
    received_values: torch.Tensor
    received_scales: torch.Tensor
    received_gate_projection: torch.Tensor
    received_up_projection: torch.Tensor
    output_splits: tuple[int, ...]
    input_splits: tuple[int, ...]
    group: Any
    collective: Any
    score_before_experts: bool
    route_scores: torch.Tensor | None


def prepare_fp8_lora_a_transport(
    source_value: torch.Tensor,
    expert_indices: torch.Tensor,
    gate_lora_a: torch.Tensor,
    up_lora_a: torch.Tensor,
    output_splits: Sequence[int],
    input_splits: Sequence[int],
    *,
    group: Any = None,
    collective: Any = None,
    route_scores: torch.Tensor | None = None,
    score_before_experts: bool = False,
    rounds: int = 1,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    | None = None,
) -> FP8LoRAATransportState:
    """Prepare the loss-minimizing forward payload.

    ``gate_lora_a`` and ``up_lora_a`` are expert-major full tables.  A caller
    with expert-sharded parameters must make these two rank-sized tables
    available on the source rank before this call (for example, by a separate
    setup-time all-gather).  Sending BF16 hidden rows to the owner to compute
    these projections is not this protocol.
    """

    _check_matrix(source_value, "source activation", dtype=torch.bfloat16)
    rows, hidden = source_value.shape
    if source_value.device.type not in {"cuda", "cpu"}:
        raise ValueError("source activation has an unsupported device")
    if expert_indices.device != source_value.device:
        raise ValueError("expert indices must be on the source activation device")
    _check_splits(rows, output_splits, input_splits)
    _validate_rounds(rounds)
    if score_before_experts:
        if route_scores is None:
            raise ValueError("score-before-experts transport requires route scores")
        _check_vector(route_scores, "route scores", rows)
        if not route_scores.dtype.is_floating_point or route_scores.device != source_value.device:
            raise TypeError("route scores must be device-local floating point")
        projected_source = (
            source_value.float() * route_scores.unsqueeze(-1).float()
        ).to(source_value.dtype)
    else:
        if route_scores is not None:
            _check_vector(route_scores, "route scores", rows)
            if not route_scores.dtype.is_floating_point or route_scores.device != source_value.device:
                raise TypeError("route scores must be device-local floating point")
        projected_source = source_value

    # The quantizer is called exactly once on the same rows that the accepted
    # grouped GEMM would quantize.  No BF16 hidden activation is sent.
    values, scales = (quantize or _default_rowwise_fp8_quantize)(projected_source)
    _check_matrix(values, "FP8 values", dtype=torch.float8_e4m3fn)
    if values.shape != (rows, hidden):
        raise ValueError("FP8 quantizer changed the source activation shape")
    _check_vector(scales, "FP8 row scales", rows)
    if scales.dtype != torch.float32 or scales.device != source_value.device:
        raise TypeError("FP8 row scales must be device-local FP32")
    gate_projection = _select_lora_a(
        projected_source,
        expert_indices,
        gate_lora_a,
        "gate",
    )
    up_projection = _select_lora_a(
        projected_source,
        expert_indices,
        up_lora_a,
        "up",
    )
    (
        received_value,
        received_values,
        received_scales,
    ) = _FP8InputAllToAll.apply(
        projected_source,
        values.detach(),
        scales.detach(),
        tuple(output_splits),
        tuple(input_splits),
        group,
        collective,
        rounds,
    )
    received_gate_projection = _RowsAllToAll.apply(
        gate_projection,
        tuple(output_splits),
        tuple(input_splits),
        group,
        collective,
    )
    received_up_projection = _RowsAllToAll.apply(
        up_projection,
        tuple(output_splits),
        tuple(input_splits),
        group,
        collective,
    )
    return FP8LoRAATransportState(
        source_value=source_value,
        source_expert_indices=expert_indices,
        values=values,
        scales=scales,
        received_value=received_value,
        received_values=received_values,
        received_scales=received_scales,
        received_gate_projection=received_gate_projection,
        received_up_projection=received_up_projection,
        output_splits=tuple(output_splits),
        input_splits=tuple(input_splits),
        group=group,
        collective=collective,
        score_before_experts=score_before_experts,
        route_scores=route_scores,
    )


BaseMatmul = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def _validate_expert_rows(expert_indices: torch.Tensor, rows: int, name: str) -> None:
    if expert_indices.ndim != 1 or expert_indices.shape[0] != rows:
        raise ValueError(f"{name} must have one expert index per row")
    if expert_indices.dtype not in (torch.int32, torch.int64) or not expert_indices.is_contiguous():
        raise TypeError(f"{name} must be contiguous int32 or int64")


def _quantized_view(
    value: torch.Tensor,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an exact QDQ forward view with an STE BF16 VJP for local GEMMs."""

    quantized, scales = quantize(value)
    _check_matrix(quantized, "local FP8 values", dtype=torch.float8_e4m3fn)
    _check_vector(scales, "local FP8 row scales", value.shape[0])
    if scales.dtype != torch.float32 or scales.device != value.device:
        raise TypeError("local FP8 row scales must be device-local FP32")
    dequantized = quantized.to(value.dtype)
    dequantized.mul_(scales.to(value.dtype).unsqueeze(-1))
    # The accepted native expert backward treats activation quantization as a
    # straight-through operation.  This keeps base dInput in BF16 while the
    # forward value remains the exact FP8 QDQ result.
    value_for_grad = value + (dequantized - value).detach()
    return value_for_grad, quantized, scales


def _dense_fp8_base_matmul(
    value: torch.Tensor,
    _values: torch.Tensor,
    _scales: torch.Tensor,
    weights: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """Reference base GEMM callback used by CPU/distributed-fake tests."""

    selected = weights.index_select(0, expert_indices.to(torch.int64))
    return torch.bmm(value.unsqueeze(1), selected.transpose(1, 2)).squeeze(1)


def run_fp8_lora_a_experts(
    state: FP8LoRAATransportState,
    local_expert_indices: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    *,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    base_matmul: BaseMatmul | None = None,
) -> torch.Tensor:
    """Run the owner-side expert MLP and combine rows back to the source.

    ``local_expert_indices`` is in the stable owner-side row order produced by
    the accepted route permutation.  Duplicate indices are valid and are not
    coalesced.  ``base_matmul`` receives ``(BF16 QDQ view, FP8 values, FP32
    scales, expert weights, expert indices)``.  A native callback can consume
    the latter two payload tensors directly; the reference callback uses the
    QDQ view and its exact BF16 VJP.
    """

    rows = state.received_value.shape[0]
    _validate_expert_rows(local_expert_indices, rows, "owner expert indices")
    for name, value in {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "gate LoRA B": gate_lora_b,
        "up LoRA B": up_lora_b,
        "down LoRA A": down_lora_a,
        "down LoRA B": down_lora_b,
    }.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous expert-major rank-3 tensor")
        if value.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use BF16")
    if any(
        local_expert_indices.numel()
        and (
            int(local_expert_indices.min()) < 0
            or int(local_expert_indices.max()) >= value.shape[0]
        )
        for value in (w1, w2, w3, gate_lora_b, up_lora_b, down_lora_a, down_lora_b)
    ):
        raise ValueError("owner expert index is outside an expert table")
    if w1.shape[2] != state.received_value.shape[1] or w3.shape[2] != w1.shape[2]:
        raise ValueError("gate/up base input widths do not match transport width")
    rank = state.received_gate_projection.shape[1]
    if gate_lora_b.shape[2] != rank or up_lora_b.shape[2] != rank:
        raise ValueError("gate/up LoRA B rank does not match transported projections")
    if down_lora_a.shape[1] != rank or down_lora_b.shape[2] != rank:
        raise ValueError("down LoRA rank does not match transported projections")
    if down_lora_a.shape[2] != w1.shape[1] or down_lora_b.shape[1] != w2.shape[1]:
        raise ValueError("down LoRA dimensions do not match the expert MLP")
    if w2.shape[1] != state.received_value.shape[1]:
        raise ValueError("down base output width does not match transport width")
    if state.received_gate_projection.shape != (
        rows,
        rank,
    ) or state.received_up_projection.shape != (rows, rank):
        raise ValueError("received LoRA-A projection shape is invalid")

    quantizer = quantize or _default_rowwise_fp8_quantize
    base_kernel = base_matmul or _dense_fp8_base_matmul
    gate_base = base_kernel(
        state.received_value,
        state.received_values,
        state.received_scales,
        w1,
        local_expert_indices,
    )
    up_base = base_kernel(
        state.received_value,
        state.received_values,
        state.received_scales,
        w3,
        local_expert_indices,
    )
    gate_b = gate_lora_b.index_select(0, local_expert_indices.to(torch.int64))
    up_b = up_lora_b.index_select(0, local_expert_indices.to(torch.int64))
    gate = gate_base + torch.bmm(
        state.received_gate_projection.unsqueeze(1), gate_b.transpose(1, 2)
    ).squeeze(1)
    up = up_base + torch.bmm(
        state.received_up_projection.unsqueeze(1), up_b.transpose(1, 2)
    ).squeeze(1)
    hidden = torch.nn.functional.silu(gate) * up

    hidden_for_base, hidden_values, hidden_scales = _quantized_view(hidden, quantizer)
    down_base = base_kernel(
        hidden_for_base,
        hidden_values,
        hidden_scales,
        w2,
        local_expert_indices,
    )
    down_a = down_lora_a.index_select(0, local_expert_indices.to(torch.int64))
    down_b = down_lora_b.index_select(0, local_expert_indices.to(torch.int64))
    hidden_low_rank = torch.bmm(hidden.unsqueeze(1), down_a.transpose(1, 2)).squeeze(1)
    down_lora = torch.bmm(hidden_low_rank.unsqueeze(1), down_b.transpose(1, 2)).squeeze(1)
    owner_output = down_base + down_lora

    # Combine is the inverse route.  The owner has output_splits rows from the
    # source; the source must receive input_splits rows in its original stable
    # route order.  This is a distinct all-to-all from projection backward.
    source_output = _RowsAllToAll.apply(
        owner_output,
        state.input_splits,
        state.output_splits,
        state.group,
        state.collective,
    )
    if not state.score_before_experts and state.route_scores is not None:
        source_output = (
            source_output.float() * state.route_scores.reshape(-1, 1).float()
        ).to(source_output.dtype)
    return source_output


def transport_mode(mode: str | None = None) -> TransportMode:
    """Validate the explicit mode without changing the default BF16 path."""

    selected = DEFAULT_TRANSPORT_MODE if mode is None else mode
    if selected not in {"bf16", "fp8_lora_a"}:
        raise ValueError("expert activation transport mode is invalid")
    return selected  # type: ignore[return-value]


def transport_bytes(hidden_size: int, rank: int = 32) -> dict[str, int]:
    """Return exact payload bytes for one routed token in each direction.

    Route metadata (expert counts, stable permutation, and duplicate row
    indices) is already exchanged by the accepted EP route and is not repeated
    in this activation payload.  The FP32 row scale is included.
    """

    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
        raise ValueError("hidden size must be a positive integer")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    return {
        "bf16_forward": 2 * hidden_size,
        "fp8_lora_a_forward": hidden_size + 4 + 2 * rank + 2 * rank,
        "bf16_backward": 2 * hidden_size,
        "fp8_lora_a_backward": 2 * hidden_size + 2 * rank + 2 * rank,
        # Named route directions.  Combine remains the accepted BF16 output
        # transport because quantizing it would change the expert result.
        "dispatch_forward": hidden_size + 4 + 4 * rank,
        "combine_forward": 2 * hidden_size,
        "dispatch_backward": 2 * hidden_size + 4 * rank,
        "combine_backward": 2 * hidden_size,
    }


def bounded_workspace_bytes(
    send_rows: int,
    receive_rows: int,
    hidden_size: int,
    rank: int = 32,
    *,
    rounds: int = 1,
) -> dict[str, int]:
    """Return the live transport scratch bound for one collective round.

    The result counts retained send and receive payloads, owner-side BF16
    input reconstruction, and the temporary FP8/BF16 row buffers used by a
    multi-round collective. It excludes existing expert weights, source BF16
    input, and output storage. For one round the collective writes directly to
    its full receive buffer, so it has no extra row scratch.
    """

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (send_rows, receive_rows)
    ):
        raise ValueError("workspace row counts must be non-negative integers")
    _validate_rounds(rounds)
    payload = transport_bytes(hidden_size, rank)
    forward_send = send_rows * payload["fp8_lora_a_forward"]
    forward_receive = receive_rows * payload["fp8_lora_a_forward"]
    owner_dequantized = receive_rows * 2 * hidden_size
    backward_send = send_rows * payload["fp8_lora_a_backward"]
    backward_receive = receive_rows * payload["fp8_lora_a_backward"]
    if rounds == 1:
        forward_round_scratch = 0
        backward_round_scratch = 0
    else:
        round_send = (send_rows + rounds - 1) // rounds
        round_receive = (receive_rows + rounds - 1) // rounds
        # The FP8 value round is the largest temporary in forward. Scales use
        # four bytes per row and are exchanged after the value scratch is
        # released. Backward transports the BF16 base-input VJP in rounds.
        forward_round_scratch = (round_send + round_receive) * hidden_size
        backward_round_scratch = (round_send + round_receive) * 2 * hidden_size
    return {
        "forward_send_bytes": forward_send,
        "forward_receive_bytes": forward_receive,
        "forward_owner_bf16_bytes": owner_dequantized,
        "forward_round_scratch_bytes": forward_round_scratch,
        "forward_peak_bytes": (
            forward_send
            + forward_receive
            + owner_dequantized
            + forward_round_scratch
        ),
        "backward_send_bytes": backward_send,
        "backward_receive_bytes": backward_receive,
        "backward_round_scratch_bytes": backward_round_scratch,
        "backward_peak_bytes": backward_send + backward_receive + backward_round_scratch,
    }


__all__ = [
    "DEFAULT_TRANSPORT_MODE",
    "FP8_LORA_A_TRANSPORT_MODE",
    "FP8LoRAATransportState",
    "TransportMode",
    "bounded_workspace_bytes",
    "prepare_fp8_lora_a_transport",
    "run_fp8_lora_a_experts",
    "transport_bytes",
    "transport_mode",
]
