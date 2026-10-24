from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers.integrations import use_kernel_forward_from_hub


@lru_cache(maxsize=1)
def _get_quack_rmsnorm():
    """Lazy-load quack rmsnorm. Returns None if unavailable or GPU is pre-Hopper."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return None
    try:
        from quack import rmsnorm

        return rmsnorm
    except ImportError:
        return None


@dataclass
class RMSNormConfig:
    hidden_size: int
    eps: float = 1e-6


@use_kernel_forward_from_hub("RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        quack_fn = _get_quack_rmsnorm() if hidden_states.is_cuda else None
        if quack_fn is not None:
            return quack_fn(hidden_states, self.weight, eps=self.variance_epsilon)
        input_dtype = hidden_states.dtype
        # Under `quantization.dequantize_checkpoint = false` the transformer
        # blocks are wrapped with a bare MixedPrecisionPolicy() (model.py:813),
        # so FSDP does not cast params to bf16 and float8_e4m3fn tensors reach
        # this norm. `self.weight * hidden_states` then raises "Promotion for
        # Float8 Types is not supported". Torch has no fp8 type promotion at
        # all, so both operands must land in a real float dtype first.
        #
        # Widening fp8 -> bf16/fp32 is exact (every e4m3 value is representable),
        # and the loader never bitcasts -- `bitcast` is hardcoded False in
        # fp8_checkpoint.py:111 -- so fp8 storage always holds true values, not
        # reinterpreted bytes. This changes no numerics when nothing is fp8.
        if input_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            input_dtype = self.weight.dtype
            if input_dtype not in (torch.float32, torch.float16, torch.bfloat16):
                input_dtype = torch.bfloat16
        weight = self.weight
        if weight.dtype != input_dtype:
            weight = weight.to(input_dtype)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        # `.type_as(x)` PROPAGATES float8: an fp8 input yields an fp8 output,
        # which then blows up at the next op that mixes it with bf16. This norm
        # already computes in fp32, so there is nothing to preserve by casting
        # back to fp8 -- it only re-narrows the result to 3 mantissa bits.
        # Normalize to a real float dtype instead and stop the leak here.
        out = F.layer_norm(x.float(), (self.dim,), self.weight.float(), self.bias.float(), self.eps)
        if x.dtype in (torch.float32, torch.float16, torch.bfloat16):
            return out.type_as(x)
        return out.to(torch.bfloat16)
