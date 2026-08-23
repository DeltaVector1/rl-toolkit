# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.distributed.expert_parallel import expert_parallel

from prime_rl.configs.trainer import EPCommBackend
from prime_rl.trainer.models.layers.fp8_ada import (
    _Linear,
    fp8_experts,
    scale_routed_output,
)
from prime_rl.trainer.mxfp4 import dequant_mxfp4


def _require_static_expert_bias(module: nn.Module) -> torch.Tensor:
    """Return the frozen routing bias only when its recorded bytes are intact."""
    value = getattr(module, "_fp32_expert_bias", None)
    source_hash = getattr(module, "_fp32_expert_bias_source_sha256", None)
    cache_hash = getattr(module, "_fp32_expert_bias_cache_sha256", None)
    if value is None:
        raise RuntimeError("Frozen GLM routing requires an FP32 bias cache")
    if value.dtype != torch.float32:
        raise RuntimeError("Frozen GLM routing bias cache must remain float32")
    if not source_hash or not cache_hash or source_hash != cache_hash:
        raise RuntimeError("Frozen GLM routing bias hashes are incomplete")
    local_value = value.to_local() if hasattr(value, "to_local") else value
    raw = local_value.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash != source_hash or actual_hash != cache_hash:
        raise RuntimeError("Frozen GLM routing bias cache changed after source capture")
    return value


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3
    fp8: bool = False  # use FP8 grouped GEMM via DeepGEMM (requires SM90)
    fp8_checkpoint: bool = False
    fp8_checkpoint_routed: bool | None = None
    fp8_checkpoint_shared: bool | None = None
    mxfp4: bool = False


# can be used as dense FFN layer or shared experts in MoE layers
class FeedForward(nn.Module):
    """
    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class BCFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(hidden_dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(F.silu(torch.matmul(x, self.w1.T)) * torch.matmul(x, self.w3.T), self.w2.T)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class MXFP4FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_dim, dim // 2, dtype=torch.uint8), requires_grad=False)
        self.w1_scale = nn.Parameter(torch.empty(hidden_dim, dim // 32, dtype=torch.uint8), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(dim, hidden_dim // 2, dtype=torch.uint8), requires_grad=False)
        self.w2_scale = nn.Parameter(torch.empty(dim, hidden_dim // 32, dtype=torch.uint8), requires_grad=False)
        self.w3 = nn.Parameter(torch.empty(hidden_dim, dim // 2, dtype=torch.uint8), requires_grad=False)
        self.w3_scale = nn.Parameter(torch.empty(hidden_dim, dim // 32, dtype=torch.uint8), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            F.silu(F.linear(x, dequant_mxfp4(self.w1, self.w1_scale)))
            * F.linear(x, dequant_mxfp4(self.w3, self.w3_scale)),
            dequant_mxfp4(self.w2, self.w2_scale),
        )


class FP8CheckpointFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_dim, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.w1_scale_inv = nn.Parameter(torch.empty(hidden_dim // 128, dim // 128, dtype=torch.float32), requires_grad=False)
        self.w2 = nn.Parameter(torch.empty(dim, hidden_dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.w2_scale_inv = nn.Parameter(torch.empty(dim // 128, hidden_dim // 128, dtype=torch.float32), requires_grad=False)
        self.w3 = nn.Parameter(torch.empty(hidden_dim, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.w3_scale_inv = nn.Parameter(torch.empty(hidden_dim // 128, dim // 128, dtype=torch.float32), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fqn = getattr(self, "_fp8_fqn", "<unassigned-fp8-shared-expert>")
        return _Linear.apply(
            F.silu(
                _Linear.apply(
                    x,
                    self.w1,
                    self.w1_scale_inv,
                    f"{fqn}.w1",
                    (self, "w1", "w1_scale_inv"),
                )
            )
            * _Linear.apply(
                x,
                self.w3,
                self.w3_scale_inv,
                f"{fqn}.w3",
                (self, "w3", "w3_scale_inv"),
            ),
            self.w2,
            self.w2_scale_inv,
            f"{fqn}.w2",
            (self, "w2", "w2_scale_inv"),
        )


# TODO: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


@expert_parallel
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_experts_for_loop_impl(w1, w2, w3, x, num_tokens_per_expert)


@expert_parallel
def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_experts_grouped_mm_impl(w1, w2, w3, x, num_tokens_per_expert)


@expert_parallel
def _run_experts_fp8_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_experts_grouped_mm_impl(w1, w2, w3, x, num_tokens_per_expert, fp8=True)


def _run_experts_grouped_mm_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    fp8: bool = False,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    # grouped mm between a 2D tensor and a 3D tensor
    assert x.dim() == 2

    if fp8:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        h = F.silu(grouped_fp8_gemm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offsets))
        h = h * grouped_fp8_gemm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offsets)
        out = grouped_fp8_gemm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)
    else:
        h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
        h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets)
        out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


@torch.compiler.disable
def _run_experts_mxfp4(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w3_scale: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if isinstance(w1, DTensor):
        w1, w2, w3, w1_scale, w2_scale, w3_scale = (
            weight.to_local() for weight in (w1, w2, w3, w1_scale, w2_scale, w3_scale)
        )
    return _run_experts_grouped_mm(
        dequant_mxfp4(w1, w1_scale),
        dequant_mxfp4(w2, w2_scale),
        dequant_mxfp4(w3, w3_scale),
        x,
        num_tokens_per_expert,
    )


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
        fp8: bool = False,
        fp8_checkpoint: bool = False,
        mxfp4: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        if fp8_checkpoint:
            self.w1 = nn.Parameter(
                torch.empty(num_experts, hidden_dim, dim, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.w1_scale_inv = nn.Parameter(torch.empty(num_experts, hidden_dim // 128, dim // 128, dtype=torch.float32), requires_grad=False)
            self.w2 = nn.Parameter(
                torch.empty(num_experts, dim, hidden_dim, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.w2_scale_inv = nn.Parameter(torch.empty(num_experts, dim // 128, hidden_dim // 128, dtype=torch.float32), requires_grad=False)
            self.w3 = nn.Parameter(
                torch.empty(num_experts, hidden_dim, dim, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            self.w3_scale_inv = nn.Parameter(torch.empty(num_experts, hidden_dim // 128, dim // 128, dtype=torch.float32), requires_grad=False)
        elif mxfp4:
            self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim // 2, dtype=torch.uint8), requires_grad=False)
            self.w1_scale = nn.Parameter(
                torch.empty(num_experts, hidden_dim, dim // 32, dtype=torch.uint8), requires_grad=False
            )
            self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim // 2, dtype=torch.uint8), requires_grad=False)
            self.w2_scale = nn.Parameter(
                torch.empty(num_experts, dim, hidden_dim // 32, dtype=torch.uint8), requires_grad=False
            )
            self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim // 2, dtype=torch.uint8), requires_grad=False)
            self.w3_scale = nn.Parameter(
                torch.empty(num_experts, hidden_dim, dim // 32, dtype=torch.uint8), requires_grad=False
            )
        else:
            self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
            self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
            self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm
        self.fp8 = fp8
        self.fp8_checkpoint = fp8_checkpoint
        self.mxfp4 = mxfp4
        self.ep_comm_backend: EPCommBackend = "torch"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        weights = (self.w1, self.w2, self.w3)
        w1, w2, w3 = (
            weight.to_local() if isinstance(weight, DTensor) else weight
            for weight in weights
        )
        if self.fp8_checkpoint:
            scales = (self.w1_scale_inv, self.w2_scale_inv, self.w3_scale_inv)
            w1_scale, w2_scale, w3_scale = (
                scale.to_local() if isinstance(scale, DTensor) else scale for scale in scales
            )
            return fp8_experts(
                w1,
                w2,
                w3,
                w1_scale,
                w2_scale,
                w3_scale,
                x,
                num_tokens_per_expert,
                owner=self,
            )
        if self.mxfp4:
            scales = (self.w1_scale, self.w2_scale, self.w3_scale)
            w1_scale, w2_scale, w3_scale = (
                scale.to_local() if isinstance(scale, DTensor) else scale
                for scale in scales
            )
            return _run_experts_mxfp4(
                w1,
                w2,
                w3,
                w1_scale,
                w2_scale,
                w3_scale,
                x,
                num_tokens_per_expert,
            )
        if self.use_grouped_mm:
            return _run_experts_grouped_mm_impl(w1, w2, w3, x, num_tokens_per_expert, fp8=self.fp8)
        return _run_experts_for_loop_impl(w1, w2, w3, x, num_tokens_per_expert)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.fp8_checkpoint:
            weights = (
                weight.to_local() if isinstance(weight, DTensor) else weight
                for weight in (
                    self.w1,
                    self.w2,
                    self.w3,
                    self.w1_scale_inv,
                    self.w2_scale_inv,
                    self.w3_scale_inv,
                )
            )
            return fp8_experts(
                *weights,
                x,
                num_tokens_per_expert,
                owner=self,
            )
        if self.mxfp4:
            return _run_experts_mxfp4(
                self.w1,
                self.w2,
                self.w3,
                self.w1_scale,
                self.w2_scale,
                self.w3_scale,
                x,
                num_tokens_per_expert,
            )
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        if self.use_grouped_mm:
            if self.fp8:
                return _run_experts_fp8_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
            return _run_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


# GPT-OSS activation constants. Both clamping limit and the sigmoid alpha live here
# rather than as instance attrs so the function is JIT/compile-friendly.
GPT_OSS_LIMIT = 7.0
GPT_OSS_ALPHA = 1.702


def _gpt_oss_apply_gate(gate_up: torch.Tensor) -> torch.Tensor:
    """GPT-OSS expert activation: clamped sigmoid-glu over interleaved gate/up channels."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=GPT_OSS_LIMIT)
    up = up.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
    glu = gate * torch.sigmoid(gate * GPT_OSS_ALPHA)
    return (up + 1) * glu


def _broadcast_expert_bias(bias: torch.Tensor, num_tokens_per_expert: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Repeat per-expert bias to per-token, padding to target_rows if EP added padding rows."""
    # repeat_interleave on CUDA requires int counts; histc/router output is float.
    bias_per_token = torch.repeat_interleave(bias, num_tokens_per_expert.to(torch.int64), dim=0)
    if bias_per_token.shape[0] < target_rows:
        pad_rows = target_rows - bias_per_token.shape[0]
        bias_per_token = F.pad(bias_per_token, (0, 0, 0, pad_rows))
    return bias_per_token


def _run_gpt_oss_experts_for_loop_impl(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    n = num_tokens_per_expert.tolist()
    num_padding = x.shape[0] - sum(n)
    x_split = torch.split(x[: sum(n)], split_size_or_sections=n, dim=0)
    out_splits = []
    for e, x_e in enumerate(x_split):
        gate_up = x_e @ gate_up_proj[e] + gate_up_proj_bias[e]
        h = _gpt_oss_apply_gate(gate_up)
        out = h @ down_proj[e] + down_proj_bias[e]
        out_splits.append(out)
    out = torch.cat(out_splits, dim=0)
    return torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))


@expert_parallel
def _run_gpt_oss_experts_for_loop(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_gpt_oss_experts_for_loop_impl(
        gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, num_tokens_per_expert
    )


def _run_gpt_oss_experts_grouped_mm_impl(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2

    gate_up = torch._grouped_mm(x.bfloat16(), gate_up_proj.bfloat16(), offs=offsets)
    gate_up = gate_up + _broadcast_expert_bias(gate_up_proj_bias, num_tokens_per_expert, gate_up.shape[0]).bfloat16()
    h = _gpt_oss_apply_gate(gate_up)
    out = torch._grouped_mm(h, down_proj.bfloat16(), offs=offsets)
    out = out + _broadcast_expert_bias(down_proj_bias, num_tokens_per_expert, out.shape[0]).bfloat16()
    return out.type_as(x)


@expert_parallel
def _run_gpt_oss_experts_grouped_mm(
    gate_up_proj: torch.Tensor,
    gate_up_proj_bias: torch.Tensor,
    down_proj: torch.Tensor,
    down_proj_bias: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_gpt_oss_experts_grouped_mm_impl(
        gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, num_tokens_per_expert
    )


class GptOssGroupedExperts(nn.Module):
    """GPT-OSS-style grouped experts.

    Mirrors HF's `GptOssExperts` parameter naming (gate_up_proj/down_proj plus per-expert
    biases, fused interleaved gate/up channels) so the unsloth BF16 checkpoint loads with
    no key conversion. Forward signature matches `GroupedExperts` (`x`, `num_tokens_per_expert`)
    so the surrounding MoE plumbing and LoRA wrapper follow the same convention.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.use_grouped_mm = use_grouped_mm
        self.ep_comm_backend: EPCommBackend = "torch"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        gate_up_proj = self.gate_up_proj.to_local()
        gate_up_proj_bias = self.gate_up_proj_bias.to_local()
        down_proj = self.down_proj.to_local()
        down_proj_bias = self.down_proj_bias.to_local()
        if self.use_grouped_mm:
            return _run_gpt_oss_experts_grouped_mm_impl(
                gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, num_tokens_per_expert
            )
        return _run_gpt_oss_experts_for_loop_impl(
            gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, num_tokens_per_expert
        )

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)

        if self.use_grouped_mm:
            return _run_gpt_oss_experts_grouped_mm(
                self.gate_up_proj, self.gate_up_proj_bias, self.down_proj, self.down_proj_bias, x, num_tokens_per_expert
            )
        return _run_gpt_oss_experts_for_loop(
            self.gate_up_proj, self.gate_up_proj_bias, self.down_proj, self.down_proj_bias, x, num_tokens_per_expert
        )

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.zeros_(self.gate_up_proj_bias)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)
        nn.init.zeros_(self.down_proj_bias)


def _selected_probability_mass_sum(
    scores: torch.Tensor, top_scores: torch.Tensor, score_func: Literal["softmax", "sigmoid"]
) -> torch.Tensor:
    with torch.no_grad():
        if score_func == "softmax":
            return top_scores.sum()
        selected_prob_mass = top_scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
        return selected_prob_mass.sum(dim=-1).sum()


class _BF16RouterGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
        if gate_weight.requires_grad:
            raise RuntimeError("BF16 router GEMM requires frozen router weights")
        ctx.input_dtype = hidden_states.dtype
        ctx.save_for_backward(gate_weight)
        return torch.mm(hidden_states, gate_weight.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (gate_weight,) = ctx.saved_tensors
        hidden_grad = torch.mm(
            grad_output.to(torch.float32), gate_weight.to(torch.float32)
        )
        return hidden_grad.to(ctx.input_dtype), None


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        score_func (Literal["softmax", "sigmoid"]): Whether to use sigmoid or softmax for router scores.
        route_norm (bool): Whether to normalize the routing scores when using sigmoid.
        route_scale (float): Scaling factor applied to the routing scores.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        score_func: Literal["softmax", "sigmoid"],
        route_norm: bool,
        route_scale: float,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.force_balanced = False
        self.parity_trace_layer: int | None = None
        self.use_fp32_router_output = False

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
        expert_bias_is_static: bool = False,
        expert_bias_source_sha256: str | None = None,
        expert_bias_cache_sha256: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs * slen, top_k)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
                - routing_confidence_sum (torch.Tensor):
                    Sum over tokens of the selected-expert probability mass before route normalization/scaling.
        """
        # scores shape (bs*slen, num_experts)
        assert routed_experts is None or routed_experts.shape[-1] == self.top_k, (
            f"routed_experts shape: {routed_experts.shape}, top_k: {self.top_k}"
        )
        router_debug_active = False
        if self.parity_trace_layer is not None:
            from prime_rl.parity_router_trace import router_debug_enabled

            router_debug_active = router_debug_enabled(self.parity_trace_layer)
        if self.use_fp32_router_output:
            gate_weight = self.gate.weight
            if gate_weight.dtype == torch.bfloat16:
                gate_input = x.to(torch.bfloat16)
                scores = _BF16RouterGemm.apply(gate_input, gate_weight)
            else:
                gate_input = x.to(gate_weight.dtype)
                scores = self.gate(gate_input)
        else:
            scores = self.gate(x)
        if router_debug_active:
            raw_router_logits = scores.to(torch.float32)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = scores.to(torch.float32)
            if self.use_fp32_router_output:
                # vLLM's GLM grouped-topk kernel uses this tanhf form.
                scores = 0.5 * torch.tanh(0.5 * scores) + 0.5
            else:
                scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        selection_bias = expert_bias
        selection_scores = scores
        if selection_bias is not None:
            selection_scores = selection_scores + selection_bias.to(torch.float32)

        if router_debug_active:
            from prime_rl.parity_router_trace import write_router_debug_chunk

            write_router_debug_chunk(
                self.parity_trace_layer,
                x,
                raw_router_logits,
                scores,
                selection_scores,
                selection_bias,
                expert_bias_source_sha256,
                expert_bias_cache_sha256,
                expert_bias_is_static,
            )
            del raw_router_logits

        if routed_experts is not None:
            selected_experts_indices = routed_experts
            if self.top_k > 1:
                padding = routed_experts[:, 1:].eq(routed_experts[:, :1]).all(dim=1, keepdim=True)
                selected_experts_indices = torch.where(
                    padding,
                    torch.topk(scores, k=self.top_k, dim=1).indices,
                    routed_experts,
                )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        elif self.force_balanced:
            num_tokens = scores.shape[0]
            arange = torch.arange(num_tokens * self.top_k, device=scores.device)
            selected_experts_indices = (arange % self.num_experts).view(num_tokens, self.top_k)
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        elif expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                selection_scores, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        routing_confidence_sum = _selected_probability_mass_sum(scores, top_scores, self.score_func)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator

        if self.parity_trace_layer is not None:
            from prime_rl.parity_router_trace import write_route_chunk

            write_route_chunk(
                self.parity_trace_layer,
                selected_experts_indices,
                top_scores,
                route_output_scale=self.route_scale,
            )

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size*seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        selected_experts_indices = selected_experts_indices.reshape(-1)
        num_tokens_per_expert = torch.histc(
            selected_experts_indices,
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        expert_order = torch.argsort(selected_experts_indices, stable=True)

        top_scores_experts_sorted = top_scores.view(-1)[expert_order]
        token_indices_experts_sorted = expert_order // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            expert_order,
        )


def combine_expert_outputs(
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
    expert_order: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Add expert outputs in the model's fixed top-k order."""
    token_count, width = shared_output.shape
    expected_rows = token_count * top_k
    if routed_output.shape != (expected_rows, width):
        raise ValueError(
            f"routed output shape {tuple(routed_output.shape)} does not match "
            f"({expected_rows}, {width})"
        )
    if expert_order.shape != (expected_rows,):
        raise ValueError(
            f"expert order shape {tuple(expert_order.shape)} does not match "
            f"({expected_rows},)"
        )

    sorted_row_for_token_slot = torch.empty_like(expert_order)
    sorted_row_for_token_slot[expert_order] = torch.arange(
        expected_rows,
        device=expert_order.device,
        dtype=expert_order.dtype,
    )
    sorted_row_for_token_slot = sorted_row_for_token_slot.view(token_count, top_k)

    output = shared_output
    for slot in range(top_k):
        rows = sorted_row_for_token_slot[:, slot]
        contribution = routed_output.index_select(0, rows)
        output = output + contribution
    return output


class MoE(nn.Module):
    def __init__(self, moe_args: MoEArgs, dim: int, hidden_dim: int):
        super().__init__()

        num_experts = moe_args.num_experts
        fp8_checkpoint_routed = (
            moe_args.fp8_checkpoint
            if moe_args.fp8_checkpoint_routed is None
            else moe_args.fp8_checkpoint_routed
        )
        fp8_checkpoint_shared = (
            moe_args.fp8_checkpoint
            if moe_args.fp8_checkpoint_shared is None
            else moe_args.fp8_checkpoint_shared
        )
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=moe_args.use_grouped_mm,
            fp8=moe_args.fp8,
            fp8_checkpoint=fp8_checkpoint_routed,
            mxfp4=moe_args.mxfp4,
        )
        self.ep_comm_backend: EPCommBackend = "torch"
        self.experts.set_ep_comm_backend(self.ep_comm_backend)
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=moe_args.top_k,
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
        )
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=moe_args.top_k)
        # TODO: Add the s back and use FF when the weights support it
        self.shared_expert = (
            (
                FP8CheckpointFeedForward
                if fp8_checkpoint_shared
                else MXFP4FeedForward if moe_args.mxfp4 else BCFeedForward
            )(
                dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts
            )
            if moe_args.num_shared_experts > 0
            else None
        )
        self.score_before_experts = moe_args.score_before_experts
        self.deepep_token_chunk_size: int | None = None

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = moe_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("routing_confidence_sum", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_deepep_token_chunk_size(self, chunk_size: int | None) -> None:
        self.deepep_token_chunk_size = chunk_size

    def _run_local_routed_experts(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, num_tokens_per_expert)

    def _run_routed_experts(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        routed_indices = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=routed_indices)

        if self.score_before_experts:
            routed_input = (routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

        routed_output = self.experts(routed_input, num_tokens_per_expert)

        if not self.score_before_experts:
            if self.experts.fp8_checkpoint:
                routed_output = scale_routed_output(
                    routed_output,
                    top_scores_experts_sorted,
                )
            else:
                routed_output = (
                    routed_output.to(torch.float32)
                    * top_scores_experts_sorted.reshape(-1, 1)
                ).to(x.dtype)

        return routed_output

    def _run_deepep_routed_experts(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.distributed.deepep import (
            combine_tokens,
            dispatch_tokens_async,
            finalize_dispatch_tokens,
            sync_combine,
        )
        from prime_rl.trainer.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            return x.new_zeros(x.shape)

        group = get_ep_group(self.experts)
        chunk_size = min(self.deepep_token_chunk_size or x.shape[0], x.shape[0])
        routed_outputs: list[torch.Tensor] = []

        for chunk_start in range(0, x.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, x.shape[0])
            hidden_states, num_tokens_per_expert, dispatch_state = finalize_dispatch_tokens(
                dispatch_tokens_async(
                    x[chunk_start:chunk_end],
                    selected_experts_indices[chunk_start:chunk_end],
                    top_scores[chunk_start:chunk_end],
                    num_experts=self.experts.num_experts,
                    group=group,
                    score_before_experts=self.score_before_experts,
                )
            )
            routed_outputs.append(
                combine_tokens(self._run_local_routed_experts(hidden_states, num_tokens_per_expert), dispatch_state)
            )
            sync_combine()

        routed_output = routed_outputs[0] if len(routed_outputs) == 1 else torch.cat(routed_outputs, dim=0)
        return routed_output

    def forward(
        self,
        x: torch.Tensor,
        routed_experts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            routed_experts (torch.Tensor | None, optional): Optional tensor with shape ``(bs, slen, top_k)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        if routed_experts is not None:
            _, _, top_k = routed_experts.shape
            routed_experts = routed_experts.reshape(
                -1, top_k
            )  # we have to reshape here because the original is non-contiguous

        # top_scores and selected_experts_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        static_expert_bias = getattr(self, "_use_static_expert_bias", False)
        if static_expert_bias and getattr(self, "_fp32_expert_bias_source_sha256", None) is None:
            raise RuntimeError("Frozen GLM routing requires an exact source bias hash")
        if static_expert_bias:
            expert_bias = _require_static_expert_bias(self)
        else:
            expert_bias = self.expert_bias
        if expert_bias is not None and expert_bias.device != x.device:
            expert_bias = expert_bias.to(x.device)
            if static_expert_bias:
                self._fp32_expert_bias = expert_bias
        (
            top_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            routing_confidence_sum,
        ) = self.router(
            x,
            expert_bias,
            routed_experts=routed_experts,
            expert_bias_is_static=static_expert_bias,
            expert_bias_source_sha256=getattr(self, "_fp32_expert_bias_source_sha256", None),
            expert_bias_cache_sha256=getattr(self, "_fp32_expert_bias_cache_sha256", None),
        )

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # Full block checkpointing can double count tokens_per_expert because it reruns the router
        # in backward. The selective MoE path avoids that by checkpointing only the
        # routed expert compute below.
        with torch.no_grad():
            if self.load_balance_coeff is not None:
                self.tokens_per_expert.add_(num_tokens_per_expert)
            self.routing_confidence_sum.add_(routing_confidence_sum)

        if self.ep_comm_backend == "deepep":
            routed_output = self._run_deepep_routed_experts(x, selected_experts_indices, top_scores)
            routed_output = routed_output * self.router.route_scale
            if self.shared_expert is not None:
                routed_output = self.shared_expert(x) + routed_output
            return routed_output.reshape(bs, slen, dim)

        # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            expert_order,
        ) = self.reorderer(top_scores, selected_experts_indices)

        routed_output = self._run_routed_experts(
            x,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )
        routed_output = routed_output * self.router.route_scale
        if self.shared_expert is not None:
            out = self.shared_expert(x)
        else:
            out = torch.zeros_like(x)

        out = combine_expert_outputs(
            out,
            routed_output,
            expert_order,
            self.router.top_k,
        )
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            self.routing_confidence_sum = torch.tensor(0.0, dtype=torch.float32)
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)


@torch.compile(dynamic=True)
def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


def _run_nongated_experts_for_loop_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    num_tokens_per_expert = num_tokens_per_expert.tolist()
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = relu2(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


@expert_parallel
def _run_nongated_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_nongated_experts_for_loop_impl(w1, w2, _w3, x, num_tokens_per_expert)


def _run_nongated_experts_grouped_mm_impl(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    fp8: bool = False,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    assert x.dim() == 2

    if fp8:
        from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

        h = relu2(grouped_fp8_gemm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offsets))
        out = grouped_fp8_gemm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)
    else:
        h = relu2(torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets))
        out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)
    return out


@expert_parallel
def _run_nongated_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_nongated_experts_grouped_mm_impl(w1, w2, _w3, x, num_tokens_per_expert)


@expert_parallel
def _run_nongated_experts_fp8_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return _run_nongated_experts_grouped_mm_impl(w1, w2, _w3, x, num_tokens_per_expert, fp8=True)


class NonGatedGroupedExperts(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
        fp8: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_dim, input_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, input_dim, intermediate_dim))
        # Dummy w3 for @expert_parallel decorator compatibility (expects w1, w2, w3 signature)
        self.w3 = nn.Parameter(torch.empty(0))
        self.use_grouped_mm = use_grouped_mm
        self.fp8 = fp8
        self.ep_comm_backend: EPCommBackend = "torch"

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend

    def _forward_deepep(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        w1 = self.w1.to_local()
        w2 = self.w2.to_local()
        w3 = self.w3.to_local()
        if self.use_grouped_mm:
            return _run_nongated_experts_grouped_mm_impl(w1, w2, w3, x, num_tokens_per_expert, fp8=self.fp8)
        return _run_nongated_experts_for_loop_impl(w1, w2, w3, x, num_tokens_per_expert)

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_comm_backend == "deepep":
            return self._forward_deepep(x, num_tokens_per_expert)
        if self.use_grouped_mm:
            if self.fp8:
                return _run_nongated_experts_fp8_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
            return _run_nongated_experts_grouped_mm(self.w1, self.w2, self.w3, x, num_tokens_per_expert)
        else:
            return _run_nongated_experts_for_loop(self.w1, self.w2, self.w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)


class NemotronHRouter(nn.Module):
    """Sigmoid router with group-based expert selection and e_score_correction_bias.

    Follows the DeepseekV3 routing pattern: sigmoid scoring, group-based top-k selection,
    and bias correction for load balancing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
    ):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(num_experts, dim))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        routed_experts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.gate.float()).sigmoid()

        if routed_experts is not None:
            # Router replay: reuse the inference engine's expert selection and
            # only recompute the gating weights from the trainer's scores. The
            # correction/load-balancing biases only affect selection, so they
            # are intentionally skipped here.
            selected_experts_indices = routed_experts
        else:
            scores_for_choice = scores + self.e_score_correction_bias.to(torch.float32)

            if expert_bias is not None:
                scores_for_choice = scores_for_choice + expert_bias.to(torch.float32)

            # Group-based routing
            if self.n_group > 1:
                group_scores = (
                    scores_for_choice.view(-1, self.n_group, self.num_experts // self.n_group)
                    .topk(2, dim=-1)[0]
                    .sum(dim=-1)
                )
                group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
                group_mask = torch.zeros_like(group_scores)
                group_mask.scatter_(1, group_idx, 1)
                score_mask = (
                    group_mask.unsqueeze(-1)
                    .expand(-1, self.n_group, self.num_experts // self.n_group)
                    .reshape(-1, self.num_experts)
                )
                scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

            selected_experts_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

        top_scores = scores.gather(1, selected_experts_indices)
        routing_confidence_sum = _selected_probability_mass_sum(scores, top_scores, "sigmoid")

        if self.norm_topk_prob:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator

        num_tokens_per_expert = torch.histc(
            selected_experts_indices.reshape(-1).float(),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate, mean=0.0, std=init_std)


class BCNonGatedFeedForward(nn.Module):
    """Non-gated feed-forward network used as the shared expert in NemotronH.

    Uses relu2 activation: down_proj(relu2(up_proj(x))).
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(relu2(self.up_proj(x)))


class LatentMoE(nn.Module):
    """NemotronH-style Mixture of Experts with latent projections.

    The input is projected to a latent space before expert computation,
    and the output is projected back. Experts use relu2 activation without gating.
    """

    def __init__(
        self,
        dim: int,
        latent_dim: int | None,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
        use_grouped_mm: bool,
        load_balance_coeff: float | None,
        fp8: bool = False,
    ):
        super().__init__()
        effective_latent_dim = latent_dim if latent_dim is not None else dim

        self.router = NemotronHRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = NonGatedGroupedExperts(
            input_dim=effective_latent_dim,
            intermediate_dim=moe_intermediate_size,
            num_experts=num_experts,
            use_grouped_mm=use_grouped_mm,
            fp8=fp8,
        )
        self.ep_comm_backend: EPCommBackend = "torch"
        self.experts.set_ep_comm_backend(self.ep_comm_backend)
        self.reorderer = TokenReorderer(num_experts=num_experts, top_k=top_k)
        self.shared_expert = BCNonGatedFeedForward(dim=dim, hidden_dim=shared_expert_intermediate_size)
        self.deepep_token_chunk_size: int | None = None

        if latent_dim is not None:
            self.fc1_latent_proj = nn.Linear(dim, latent_dim, bias=False)
            self.fc2_latent_proj = nn.Linear(latent_dim, dim, bias=False)
        else:
            self.fc1_latent_proj = nn.Identity()
            self.fc2_latent_proj = nn.Identity()

        self.routed_scaling_factor = routed_scaling_factor
        self.load_balance_coeff = load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("routing_confidence_sum", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def set_ep_comm_backend(self, backend: EPCommBackend) -> None:
        self.ep_comm_backend = backend
        self.experts.set_ep_comm_backend(backend)

    def set_deepep_token_chunk_size(self, chunk_size: int | None) -> None:
        self.deepep_token_chunk_size = chunk_size

    def _run_local_routed_experts(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return self.experts(x, num_tokens_per_expert)

    def _run_routed_experts(
        self,
        x: torch.Tensor,
        token_indices_experts_sorted: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores_experts_sorted: torch.Tensor,
    ) -> torch.Tensor:
        dim = x.shape[-1]
        token_indices_expanded = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices_expanded)

        routed_input = self.fc1_latent_proj(routed_input)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        routed_output = (routed_output.float() * top_scores_experts_sorted.reshape(-1, 1)).to(routed_output.dtype)
        routed_output = routed_output * self.routed_scaling_factor

        routed_output = self.fc2_latent_proj(routed_output)
        return routed_output

    def _run_deepep_routed_experts(
        self,
        x: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        from prime_rl.trainer.distributed.deepep import (
            combine_tokens,
            dispatch_tokens_async,
            finalize_dispatch_tokens,
            sync_combine,
        )
        from prime_rl.trainer.distributed.expert_parallel import get_ep_group

        if x.shape[0] == 0:
            return self.shared_expert(x)

        group = get_ep_group(self.experts)
        # Project before dispatch so DeepEP communicates the smaller latent activations.
        latent_x = self.fc1_latent_proj(x)
        chunk_size = min(self.deepep_token_chunk_size or latent_x.shape[0], latent_x.shape[0])
        routed_outputs: list[torch.Tensor] = []

        for chunk_start in range(0, latent_x.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, latent_x.shape[0])
            hidden_states, num_tokens_per_expert, dispatch_state = finalize_dispatch_tokens(
                dispatch_tokens_async(
                    latent_x[chunk_start:chunk_end],
                    selected_experts_indices[chunk_start:chunk_end],
                    top_scores[chunk_start:chunk_end],
                    num_experts=self.experts.num_experts,
                    group=group,
                    score_before_experts=False,
                )
            )
            routed_outputs.append(
                combine_tokens(self._run_local_routed_experts(hidden_states, num_tokens_per_expert), dispatch_state)
            )
            sync_combine()

        shared_output = self.shared_expert(x)
        routed_output = routed_outputs[0] if len(routed_outputs) == 1 else torch.cat(routed_outputs, dim=0)
        routed_output = routed_output * self.routed_scaling_factor
        routed_output = self.fc2_latent_proj(routed_output)
        return shared_output + routed_output

    def forward(self, x: torch.Tensor, routed_experts: torch.Tensor | None = None) -> torch.Tensor:
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)

        if routed_experts is not None:
            # Flatten to (bs * slen, top_k); reshape (not view) since the slice is non-contiguous.
            _, _, top_k = routed_experts.shape
            routed_experts = routed_experts.reshape(-1, top_k)

        static_expert_bias = getattr(self, "_use_static_expert_bias", False)
        if static_expert_bias and getattr(self, "_fp32_expert_bias_source_sha256", None) is None:
            raise RuntimeError("Frozen GLM routing requires an exact source bias hash")
        if static_expert_bias:
            expert_bias = _require_static_expert_bias(self)
        else:
            expert_bias = self.expert_bias
        if expert_bias is not None and expert_bias.device != x_flat.device:
            expert_bias = expert_bias.to(x_flat.device)
            if static_expert_bias:
                self._fp32_expert_bias = expert_bias
        top_scores, selected_experts_indices, num_tokens_per_expert, routing_confidence_sum = self.router(
            x_flat, expert_bias, routed_experts=routed_experts
        )

        with torch.no_grad():
            if self.load_balance_coeff is not None:
                self.tokens_per_expert.add_(num_tokens_per_expert)
            self.routing_confidence_sum.add_(routing_confidence_sum)

        if self.ep_comm_backend == "deepep":
            routed_output = self._run_deepep_routed_experts(x_flat, selected_experts_indices, top_scores)
            return routed_output.reshape(bs, slen, dim)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            expert_order,
        ) = self.reorderer(top_scores, selected_experts_indices)

        routed_output = self._run_routed_experts(
            x_flat,
            token_indices_experts_sorted,
            num_tokens_per_expert,
            top_scores_experts_sorted,
        )

        out = self.shared_expert(x_flat)

        out = combine_expert_outputs(
            out,
            routed_output,
            expert_order,
            self.router.top_k,
        )
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(self, init_std: float, buffer_device: torch.device):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(self.experts.num_experts, dtype=torch.float32)
            self.routing_confidence_sum = torch.tensor(0.0, dtype=torch.float32)
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(self.experts.num_experts, dtype=torch.float32)
