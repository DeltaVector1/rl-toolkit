import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor
from prime_rl.trainer.models.layers.checkpointing import (
    in_reentrant_block_checkpoint,
)
from prime_rl.trainer.models.layers.lm_head import (
    PrimeLmOutput,
    _online_logsumexp_and_weighted_update,
)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    dispatch_fused_moe_kernel,
    get_default_config,
    moe_kernel_quantize_input,
)


block_shape = [128, 128]
_FP8_W13_CHUNK_ROWS = 2048
_FP8_W2_CHUNK_ROWS = 2048
_FP8_EXPERT_BACKWARD_CHUNK_ROWS = 2048
_FP8_ROUTE_SCALE_CHUNK_ROWS = 4096
_FP8_EXPERT_ACTIVATION_OFFLOAD = (
    os.environ.get("PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD", "") == "1"
)
_DIAGNOSE_LINEAR_BACKWARD = False
_REPORTED_LINEAR_BACKWARD: set[tuple[tuple[int, ...], bool, bool]] = set()
_REPORTED_INPUT_GRAD: set[tuple[int, ...]] = set()
_EXPERT_PROFILE = os.environ.get("PRIME_EP_PROFILE", "") == "1"
_EXPERT_BACKWARD_EVENTS: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_EXPERT_PROFILE_EVERY = int(os.environ.get("PRIME_EXPERT_PROFILE_EVERY", "90"))
# Fallback for direct function calls that have no owning module. Production
# modules reacquire current unsharded FSDP parameters from their owner.
# Values are [weight, scale, refcount].
_LINEAR_SNAPSHOT_CACHE: dict[str, list] = {}


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
        weight_is_output_major: tl.constexpr,
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
            if weight_is_output_major:
                weight_fragment = tl.load(
                    weight + offs_n[None, :] * stride_w0 + offs_k[:, None] * stride_w1,
                    mask=(offs_n[None, :] < output_size)
                    & (offs_k[:, None] < reduction_size),
                    other=0.0,
                ).to(tl.bfloat16)
                inverse_scale = tl.load(
                    scale + (pid_n * stride_s0) + ((k_start // 128) * stride_s1)
                )
            else:
                weight_fragment = tl.load(
                    weight + offs_k[:, None] * stride_w0 + offs_n[None, :] * stride_w1,
                    mask=(offs_k[:, None] < reduction_size)
                    & (offs_n[None, :] < output_size),
                    other=0.0,
                ).to(tl.bfloat16)
                inverse_scale = tl.load(
                    scale + ((k_start // 128) * stride_s0) + (pid_n * stride_s1)
                )
            accumulator += tl.dot(activation, weight_fragment) * inverse_scale

        tl.store(
            out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < output_size),
        )

else:
    _block_fp8_mm_kernel = None


def set_diagnose_linear_backward(enabled: bool) -> None:
    global _DIAGNOSE_LINEAR_BACKWARD
    _DIAGNOSE_LINEAR_BACKWARD = enabled


def _native_block_fp8_mm(x, weight, scale, *, weight_is_output_major):
    """Multiply by serialized block-E4M3 weights without dequantizing them.

    With ``weight_is_output_major=True`` this computes ``x @ weight.T`` (the
    checkpoint forward layout).  False computes ``x @ weight`` and is the dX
    layout.  The output is always contiguous; x may have arbitrary positive
    two-dimensional strides.
    """
    if _block_fp8_mm_kernel is None:
        raise RuntimeError("Triton is unavailable")
    if x.ndim != 2 or weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("native block-FP8 GEMM requires 2-D tensors")
    reduction_size = x.shape[1]
    if weight_is_output_major:
        output_size = weight.shape[0]
        expected_reduction = weight.shape[1]
        expected_scale = (
            (output_size + 127) // 128,
            (reduction_size + 127) // 128,
        )
    else:
        expected_reduction = weight.shape[0]
        output_size = weight.shape[1]
        expected_scale = (
            (reduction_size + 127) // 128,
            (output_size + 127) // 128,
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
        weight_is_output_major,
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=128,
        num_warps=num_warps,
        num_stages=2,
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


class FP8CheckpointLinear(nn.Linear):
    def forward(self, x):
        fqn = getattr(self, "_fp8_fqn", f"<unassigned-fp8-linear:{id(self)}>")
        out = _Linear.apply(
            x,
            self.weight,
            self.weight_scale_inv,
            fqn,
            (self, "weight", "weight_scale_inv"),
        )
        return out if self.bias is None else out + self.bias

    @classmethod
    def from_linear(cls, linear):
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
        return module


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


def replace_linear(model, embed=False, lm_head=False):
    exclusions = set(model.config.quantization_config["modules_to_not_convert"])
    for name, module in list(model.named_modules()):
        if (
            not isinstance(module, nn.Linear)
            or name in exclusions
            or name.replace(".mlp.router.gate", ".mlp.gate") in exclusions
            or name.endswith((".self_attn.kv_b_proj", ".self_attn.indexer.wk"))
        ):
            continue
        if module.in_features % 128 or module.out_features % 128:
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, FP8CheckpointLinear.from_linear(module))

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
        return
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
    2,048-row bound limits the boolean source gather to 24 MiB here.
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
    return torch._grouped_mm(
        low_rank,
        lora_b.transpose(-2, -1),
        offs=offsets,
    )


def _grouped_lora_vjp(x, lora_a, lora_b, offsets, grad_output):
    """Recompute one LoRA projection and return its exact local VJP."""
    with torch.enable_grad():
        x_ref = x.detach().requires_grad_(True)
        a_ref = lora_a.detach().requires_grad_(True)
        b_ref = lora_b.detach().requires_grad_(True)
        output = _grouped_lora_projection(x_ref, a_ref, b_ref, offsets)
        return torch.autograd.grad(
            output,
            (x_ref, a_ref, b_ref),
            grad_outputs=grad_output,
            allow_unused=False,
        )


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
        chunk_rows,
    ):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            raise TypeError("native routed-expert LoRA requires CUDA BF16 activations")
        if received_input is None:
            raise RuntimeError(
                "native routed-expert LoRA requires the exact received INT4 dispatch input"
            )
        if chunk_rows <= 0:
            raise ValueError("native routed-expert LoRA chunk size must be positive")
        scaling = float(scaling)
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("native routed-expert LoRA scaling must be finite and positive")

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
            z1.add_(
                _grouped_lora_projection(
                    chunk_x,
                    w1_lora_a,
                    w1_lora_b,
                    chunk_offsets,
                ),
                alpha=scaling,
            )
            z3.add_(
                _grouped_lora_projection(
                    chunk_x,
                    w3_lora_a,
                    w3_lora_b,
                    chunk_offsets,
                ),
                alpha=scaling,
            )

            q1, qs1 = ops.scaled_fp8_quant(z1, use_per_token_if_dynamic=True)
            q3, qs3 = ops.scaled_fp8_quant(z3, use_per_token_if_dynamic=True)
            q1_host[start:end].copy_(q1, non_blocking=True)
            q3_host[start:end].copy_(q3, non_blocking=True)
            qs1_parts.append(qs1)
            qs3_parts.append(qs3)
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
            chunk_out.add_(
                _grouped_lora_projection(
                    z1,
                    w2_lora_a,
                    w2_lora_b,
                    chunk_offsets,
                ),
                alpha=scaling,
            )
            _index_copy_real_rows_with_cache_retry(x, indices, chunk_out, real)
            del (
                chunk_x,
                safe_indices,
                z1,
                chunk_out,
                indices,
                real,
                chunk_offsets,
            )

        qs1 = torch.cat(qs1_parts, dim=0)
        qs3 = torch.cat(qs3_parts, dim=0)
        offload_done = torch.cuda.Event()
        offload_done.record(torch.cuda.current_stream(x.device))
        ctx.owner = owner
        ctx.received_input = received_input
        ctx.offload_done = offload_done
        ctx.source_rows = x.shape[0]
        ctx.chunk_rows = min(chunk_rows, _FP8_EXPERT_BACKWARD_CHUNK_ROWS)
        ctx.scaling = scaling
        ctx.save_for_backward(
            q1_host,
            qs1,
            q3_host,
            qs3,
            permuted_indices.detach().to("cpu"),
            offsets,
            w1_lora_a,
            w1_lora_b,
            w2_lora_a,
            w2_lora_b,
            w3_lora_a,
            w3_lora_b,
        )
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        (
            q1,
            qs1,
            q3,
            qs3,
            permuted_indices_host,
            offsets,
            w1_lora_a,
            w1_lora_b,
            w2_lora_a,
            w2_lora_b,
            w3_lora_a,
            w3_lora_b,
        ) = ctx.saved_tensors
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

        from prime_rl.trainer.distributed.expert_parallel import (
            INT4ExpertInputWorkspace,
        )

        torch.cuda.current_stream(grad.device).wait_event(ctx.offload_done)
        grad = grad.contiguous()
        workspace = INT4ExpertInputWorkspace(
            ctx.chunk_rows,
            grad.shape[1],
            grad.device,
        )
        parameter_grads = [
            torch.zeros_like(value)
            for value in (
                w1_lora_a,
                w1_lora_b,
                w2_lora_a,
                w2_lora_b,
                w3_lora_a,
                w3_lora_b,
            )
        ]

        rows = permuted_indices_host.shape[0]
        for start in range(0, rows, ctx.chunk_rows):
            end = min(start + ctx.chunk_rows, rows)
            host_indices = permuted_indices_host[start:end]
            indices = host_indices.to(grad.device, non_blocking=True)
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
            hidden = F.silu(z1) * z3

            hidden_grad = _grouped_mm_chunked(
                grad_chunk,
                w2.transpose(-2, -1),
                s2.transpose(-2, -1),
                chunk_offsets,
            )
            lora_hidden_grad, grad_a, grad_b = _grouped_lora_vjp(
                hidden,
                w2_lora_a,
                w2_lora_b,
                chunk_offsets,
                grad_chunk.mul(ctx.scaling),
            )
            hidden_grad.add_(lora_hidden_grad)
            parameter_grads[2].add_(grad_a)
            parameter_grads[3].add_(grad_b)
            del lora_hidden_grad, grad_a, grad_b, hidden

            z1_grad, z3_grad = _silu_glu_backward_inplace(
                hidden_grad,
                z1,
                z3,
            )
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

            reconstructed_x = grad_chunk
            ctx.received_input.reconstruct_rows(
                host_indices,
                reconstructed_x,
                workspace,
            )
            lora_x_grad, grad_a, grad_b = _grouped_lora_vjp(
                reconstructed_x,
                w1_lora_a,
                w1_lora_b,
                chunk_offsets,
                z1_grad.mul(ctx.scaling),
            )
            x_grad.add_(lora_x_grad)
            parameter_grads[0].add_(grad_a)
            parameter_grads[1].add_(grad_b)
            del lora_x_grad, grad_a, grad_b
            lora_x_grad, grad_a, grad_b = _grouped_lora_vjp(
                reconstructed_x,
                w3_lora_a,
                w3_lora_b,
                chunk_offsets,
                z3_grad.mul(ctx.scaling),
            )
            x_grad.add_(lora_x_grad)
            x_grad.add_(x_grad_w3)
            parameter_grads[4].add_(grad_a)
            parameter_grads[5].add_(grad_b)
            _index_copy_real_rows_with_cache_retry(grad, indices, x_grad, real)
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
                reconstructed_x,
                lora_x_grad,
                grad_a,
                grad_b,
            )

        del ctx.owner, ctx.received_input, ctx.offload_done
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
        _FP8_W13_CHUNK_ROWS,
    )
