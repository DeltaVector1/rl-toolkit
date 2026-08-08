import math

import torch
from torch import nn

from prime_rl.trainer.models.layers.lora.base import MultiLoRAModule, get_lora_num_tokens, get_multilora_scaling


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
    def forward(ctx, x, lora_A, lora_B):
        a_out = x @ lora_A.T
        ctx.save_for_backward(x, lora_A, lora_B, a_out)
        return a_out @ lora_B.T

    @staticmethod
    def backward(ctx, grad):
        x, lora_A, lora_B, a_out = ctx.saved_tensors
        # Accumulate low-precision LoRA gradients in FP32. Preserve FP32/FP64
        # inputs so the custom backward has the same derivative as the two
        # matmuls and remains gradcheckable outside the production BF16 path.
        compute_dtype = (
            torch.float32
            if grad.dtype in (torch.float16, torch.bfloat16)
            else grad.dtype
        )
        grad_compute = grad.to(compute_dtype)
        x_compute = x.to(compute_dtype)
        lora_A_compute = lora_A.to(compute_dtype)
        lora_B_compute = lora_B.to(compute_dtype)
        a_out_compute = a_out.to(compute_dtype)
        grad_a = grad_compute @ lora_B_compute
        return (
            (grad_a @ lora_A_compute).to(x.dtype),
            (grad_a.T @ x_compute).to(lora_A.dtype),
            (grad_compute.T @ a_out_compute).to(lora_B.dtype),
        )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        base_out = self.base_layer(x)
        lora_x = self.lora_dropout(x)

        if self.n_adapters == 1:
            lora_out = _SingleLoRA.apply(lora_x, self.lora_A[0], self.lora_B[0])
            return (base_out + self._scaling_factors[0] * lora_out).view(shape)

        offsets = self._lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
        assert offsets[-1] == x.shape[0], f"offsets: {offsets}, x.shape: {x.shape}"
        lora_A = torch.stack(list(self.lora_A))
        lora_B = torch.stack(list(self.lora_B))
        if self.use_grouped_mm:
            lora_out = _run_lora_grouped_mm(lora_x, lora_A, lora_B, offsets)
        else:
            lora_out = _run_lora_for_loop(lora_x, lora_A, lora_B, offsets)
        scaling = torch.repeat_interleave(self._scaling_factors, self._lora_num_tokens).unsqueeze(-1)
        return (base_out + scaling * lora_out).view(shape)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base={self.base_layer}, rank={self.rank}, "
            f"n_adapters={self.n_adapters}, alpha={self.alpha}, dropout={self.lora_dropout})"
        )
