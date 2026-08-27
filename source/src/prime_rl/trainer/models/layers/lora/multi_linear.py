import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.fp8_ada import (
    FP8CheckpointFusedQKVA,
    FP8CheckpointFusedQKVASlice,
)
from prime_rl.trainer.models.layers.lora.base import MultiLoRAModule, get_lora_num_tokens, get_multilora_scaling
from prime_rl.trainer.phase_profiler import phase_span


_REPORTED_NONFINITE_LORA: set[str] = set()


if hasattr(triton, "jit"):

    @triton.jit
    def _lora_mm_kernel(
        left,
        right,
        out,
        rows,
        columns,
        reduction,
        stride_lm,
        stride_lk,
        stride_rk,
        stride_rn,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, reduction, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            left_fragment = tl.load(
                left + offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lk,
                mask=(offs_m[:, None] < rows) & (offs_k[None, :] < reduction),
                other=0.0,
            )
            right_fragment = tl.load(
                right + offs_k[:, None] * stride_rk + offs_n[None, :] * stride_rn,
                mask=(offs_k[:, None] < reduction) & (offs_n[None, :] < columns),
                other=0.0,
            )
            accumulator += tl.dot(left_fragment, right_fragment)

        tl.store(
            out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            accumulator,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < columns),
        )

else:
    _lora_mm_kernel = None


@dataclass(frozen=True)
class _LoraMmConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int


_LEGACY_LORA_MM_CONFIGS = {
    "rank_output": _LoraMmConfig(32, 32, 32, 4),
    "paired_rank_output": _LoraMmConfig(64, 64, 32, 4),
    "short_rows": _LoraMmConfig(32, 128, 32, 4),
    "wide_output": _LoraMmConfig(64, 128, 32, 8),
}

_SM89_R32_LORA_MM_CONFIGS = {
    "rank_output": _LoraMmConfig(128, 32, 64, 8),
    "rank_output_short": _LoraMmConfig(32, 32, 32, 4),
    "paired_rank_output": _LoraMmConfig(64, 64, 64, 4),
    "short_rows": _LoraMmConfig(32, 128, 32, 4),
    "short_rows_long_reduction": _LoraMmConfig(32, 128, 64, 4),
    "wide_output": _LoraMmConfig(64, 128, 32, 8),
    "wide_output_long_reduction": _LoraMmConfig(64, 128, 64, 8),
}


def _lora_mm_config(
    rows: int,
    reduction: int,
    columns: int,
    profile: str | None = None,
) -> _LoraMmConfig:
    profile = profile or os.environ.get("PRIME_LORA_MM_PROFILE", "legacy")
    if profile not in {"auto", "legacy", "sm89-r32"}:
        raise ValueError(
            "PRIME_LORA_MM_PROFILE must be auto, legacy, or sm89-r32"
        )
    configs = (
        _LEGACY_LORA_MM_CONFIGS
        if profile == "legacy"
        else _SM89_R32_LORA_MM_CONFIGS
    )
    if columns <= 32:
        if configs is _SM89_R32_LORA_MM_CONFIGS and rows < 1024:
            return configs["rank_output_short"]
        return configs["rank_output"]
    if columns <= 64:
        return configs["paired_rank_output"]
    if rows <= 32:
        if configs is _SM89_R32_LORA_MM_CONFIGS and reduction >= 2048:
            return configs["short_rows_long_reduction"]
        return configs["short_rows"]
    if configs is _SM89_R32_LORA_MM_CONFIGS and reduction >= 2048:
        return configs["wide_output_long_reduction"]
    return configs["wide_output"]


def _lora_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """BF16/FP16 GEMM with FP32 tensor-core accumulation and no cuBLAS handle.

    Reentrant backward runs on an autograd worker thread. Creating a new cuBLAS
    handle there can fail when the 32K working set is near the device limit.
    The direct Triton path has bounded launch state and does not materialize
    full FP32 copies of the activation or output gradient.
    """

    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError(
            f"LoRA matrix mismatch: {tuple(left.shape)} @ {tuple(right.shape)}"
        )
    eligible = (
        _lora_mm_kernel is not None
        and left.is_cuda
        and right.is_cuda
        and left.device == right.device
        and left.dtype in (torch.bfloat16, torch.float16)
        and right.dtype == left.dtype
        and torch.cuda.get_device_capability(left.device) == (8, 9)
    )
    if not eligible:
        return left @ right

    rows, reduction = left.shape
    columns = right.shape[1]
    config = _lora_mm_config(rows, reduction, columns)
    out = torch.empty(rows, columns, device=left.device, dtype=left.dtype)
    grid = (
        triton.cdiv(rows, config.block_m),
        triton.cdiv(columns, config.block_n),
    )
    _lora_mm_kernel[grid](
        left,
        right,
        out,
        rows,
        columns,
        reduction,
        left.stride(0),
        left.stride(1),
        right.stride(0),
        right.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=2,
    )
    return out


def _run_lora_grouped_mm(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.LongTensor,
) -> torch.Tensor:
    return torch._grouped_mm(
        torch._grouped_mm(x, lora_A.transpose(-2, -1), offsets),
        lora_B.transpose(-2, -1),
        offsets,
    )


def _run_lora_for_loop(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    offsets: torch.LongTensor,
) -> torch.Tensor:
    lora_out_splits = []
    for i in range(offsets.shape[0]):
        start = 0 if i == 0 else offsets[i - 1]
        a_out = torch.matmul(x[start : offsets[i]], lora_A[i].transpose(-2, -1))
        lora_out_splits.append(torch.matmul(a_out, lora_B[i].transpose(-2, -1)))
    return torch.cat(lora_out_splits, dim=0)


class _SingleLoRA(torch.autograd.Function):
    @staticmethod
    @phase_span("lora_forward")
    def forward(ctx, x, lora_A, lora_B, fqn):
        a_out = _lora_mm(x, lora_A.T)
        ctx.save_for_backward(x, lora_A, lora_B, a_out)
        ctx.fqn = fqn
        return _lora_mm(a_out, lora_B.T)

    @staticmethod
    @phase_span("lora_backward")
    def backward(ctx, grad):
        x, lora_A, lora_B, a_out = ctx.saved_tensors
        if grad.dtype in (torch.float16, torch.bfloat16):
            # The Triton dot operations accumulate each product in FP32, then
            # write the BF16/FP16 gradients expected by the parameters. Avoid
            # explicit full-size FP32 activation and output-gradient copies.
            grad_a = _lora_mm(grad, lora_B)
            x_grad = _lora_mm(grad_a, lora_A).to(x.dtype)
            a_grad = _lora_mm(grad_a.T, x).to(lora_A.dtype)
            b_grad = _lora_mm(grad.T, a_out).to(lora_B.dtype)
            if (
                os.environ.get("PRIME_FP8_DIAG_LINEAR_BACKWARD", "") == "1"
                and not all(
                    bool(torch.isfinite(value).all())
                    for value in (grad, grad_a, x_grad, a_grad, b_grad)
                )
                and ctx.fqn not in _REPORTED_NONFINITE_LORA
            ):
                _REPORTED_NONFINITE_LORA.add(ctx.fqn)
                print(
                    "[lora-bwd] "
                    f"fqn={ctx.fqn} incoming={bool(torch.isfinite(grad).all())} "
                    f"grad_a={bool(torch.isfinite(grad_a).all())} "
                    f"x_grad={bool(torch.isfinite(x_grad).all())} "
                    f"a_grad={bool(torch.isfinite(a_grad).all())} "
                    f"b_grad={bool(torch.isfinite(b_grad).all())}",
                    flush=True,
                )
            return (
                x_grad,
                a_grad,
                b_grad,
                None,
            )

        # Preserve exact FP32/FP64 behavior for CPU use and gradcheck.
        grad_a = grad @ lora_B
        return (
            (grad_a @ lora_A).to(x.dtype),
            (grad_a.T @ x).to(lora_A.dtype),
            (grad.T @ a_out).to(lora_B.dtype),
            None,
        )


def _local_frozen_weight(module: nn.Linear) -> torch.Tensor:
    weight = module.weight
    if weight.requires_grad:
        raise ValueError("fused Q/KV-A base weights must be frozen")
    if isinstance(weight, DTensor):
        weight = weight.to_local()
    if weight.dtype != torch.bfloat16:
        raise TypeError(
            "fused Q/KV-A base weights must be BF16, " f"got {weight.dtype}"
        )
    return weight


class _FusedFrozenLinearPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, first_owner, second_owner):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"fused Q/KV-A input must be BF16, got {x.dtype}")
        first_weight = _local_frozen_weight(first_owner)
        second_weight = _local_frozen_weight(second_owner)
        if first_weight.shape[1] != second_weight.shape[1]:
            raise ValueError(
                "fused Q/KV-A input sizes differ: "
                f"{first_weight.shape[1]} and {second_weight.shape[1]}"
            )
        if first_weight.shape[1] != x.shape[-1]:
            raise ValueError(
                "fused Q/KV-A input size mismatch: "
                f"input={x.shape[-1]}, weight={first_weight.shape[1]}"
            )
        ctx.first_owner = first_owner
        ctx.second_owner = second_owner
        ctx.input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        fused_weight = torch.cat((first_weight, second_weight), dim=0)
        try:
            output = F.linear(x, fused_weight)
            return output.view(*ctx.input_shape[:-1], output.shape[-1])
        finally:
            del fused_weight

    @staticmethod
    def backward(ctx, grad_output):
        first_weight = _local_frozen_weight(ctx.first_owner)
        second_weight = _local_frozen_weight(ctx.second_owner)
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        fused_weight = torch.cat((first_weight, second_weight), dim=0)
        try:
            input_grad = F.linear(grad_output, fused_weight.t()).view(ctx.input_shape)
        finally:
            del (
                fused_weight,
                ctx.first_owner,
                ctx.second_owner,
                ctx.input_shape,
            )
        return input_grad, None, None


def fused_frozen_linear_pair(
    x: torch.Tensor,
    first: nn.Linear | FP8CheckpointFusedQKVA,
    second: nn.Linear | FP8CheckpointFusedQKVASlice,
) -> tuple[torch.Tensor, torch.Tensor]:
    if first.bias is not None or second.bias is not None:
        raise ValueError("fused Q/KV-A base projections do not support bias")
    if isinstance(first, FP8CheckpointFusedQKVA):
        if not isinstance(second, FP8CheckpointFusedQKVASlice):
            raise TypeError("fused FP8 Q-A requires its fused KV-A slice")
        if second.owner is not first:
            raise ValueError("fused FP8 Q-A and KV-A do not share one owner")
        return first.split(x)
    if isinstance(second, FP8CheckpointFusedQKVASlice):
        raise TypeError("fused FP8 KV-A requires its fused Q-A owner")
    combined = _FusedFrozenLinearPair.apply(x, first, second)
    return combined.split((first.out_features, second.out_features), dim=-1)


class MultiLoRALinear(MultiLoRAModule):
    """Linear + multi-LoRA with grouped GEMM."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_grouped_mm: bool = True,
    ):
        super().__init__(base_layer)
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be > 0")
        if rank % 8 != 0 or base_layer.in_features % 8 != 0 or base_layer.out_features % 8 != 0:
            use_grouped_mm = False

        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_grouped_mm = use_grouped_mm
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()
        lora_dtype = (
            torch.bfloat16 if base_layer.weight.dtype == torch.float8_e4m3fn else base_layer.weight.dtype
        )
        self.lora_A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        rank,
                        self.in_features,
                        device=self.base_layer.weight.device,
                        dtype=lora_dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.lora_B = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.out_features,
                        rank,
                        device=self.base_layer.weight.device,
                        dtype=lora_dtype,
                    )
                )
                for _ in range(n_adapters)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        if index is None:
            for i in range(self.n_adapters):
                self.reset_parameters(i)
        else:
            nn.init.kaiming_uniform_(self.lora_A[index], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[index])

    def named_parameters_for_adapter(self, idx: int) -> list[tuple[str, nn.Parameter]]:
        return [("lora_A", self.lora_A[idx]), ("lora_B", self.lora_B[idx])]

    def get_lora_param_counts(self) -> tuple[int, int]:
        return (
            self.lora_A[0].numel() + self.lora_B[0].numel(),
            self.base_layer.weight.numel(),
        )

    def active_lora_factors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the active factors for code paths that absorb a linear weight."""
        if not isinstance(self.lora_dropout, nn.Identity):
            raise ValueError("absorbed LoRA projections require dropout=0")
        adapter_index = int(self._lora_num_tokens.argmax().item())
        return (
            self.lora_A[adapter_index],
            self.lora_B[adapter_index],
            self._scaling_factors[adapter_index],
        )

    def _adapter_output_2d(self, x: torch.Tensor) -> torch.Tensor:
        lora_x = self.lora_dropout(x)

        if self.n_adapters == 1:
            lora_out = _SingleLoRA.apply(
                lora_x,
                self.lora_A[0],
                self.lora_B[0],
                getattr(self.base_layer, "_fp8_fqn", "<unassigned-lora-linear>"),
            )
            return self._scaling_factors[0] * lora_out

        offsets = self._lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
        assert offsets[-1] == x.shape[0], f"offsets: {offsets}, x.shape: {x.shape}"
        lora_A = torch.stack(list(self.lora_A))
        lora_B = torch.stack(list(self.lora_B))
        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(lora_x, lora_A, lora_B, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, lora_A, lora_B, offsets)
        scaling = torch.repeat_interleave(
            self._scaling_factors, self._lora_num_tokens
        ).unsqueeze(-1)
        return scaling * lora_out

    def adapter_output(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        return self._adapter_output_2d(x).view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        return (self.base_layer(x) + self._adapter_output_2d(x)).view(shape)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )
