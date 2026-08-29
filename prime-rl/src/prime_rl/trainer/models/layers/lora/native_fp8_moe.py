import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.layers.fp8_ada import fp8_lora_experts
from prime_rl.trainer.models.layers.lora.base import (
    MultiLoRAModule,
    get_lora_num_tokens,
    get_multilora_scaling,
)
from prime_rl.trainer.models.layers.moe import GroupedExperts


class NativeFP8MultiLoRAGroupedExperts(MultiLoRAModule):
    """Standard expert LoRA with a frozen native block-FP8 base."""

    requires_received_expert_input = True
    fp8_checkpoint = True
    local_recompute_backward = False

    def __init__(
        self,
        base_layer: GroupedExperts,
        rank: int,
        n_adapters: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(base_layer)
        if not base_layer.fp8_checkpoint:
            raise ValueError("native FP8 expert LoRA requires block-FP8 checkpoint weights")
        if rank <= 0 or n_adapters <= 0:
            raise ValueError("rank and n_adapters must be positive")
        if rank % 8:
            raise ValueError("native FP8 routed-expert LoRA rank must be a multiple of 8")
        if n_adapters != 1:
            raise ValueError("native FP8 routed-expert LoRA supports one run at a time")
        if dropout != 0.0:
            raise ValueError("native FP8 routed-expert LoRA requires dropout=0")

        self.num_experts = base_layer.num_experts
        self.dim = base_layer.w1.shape[2]
        self.hidden_dim = base_layer.w1.shape[1]
        self.rank = rank
        self.n_adapters = n_adapters
        self.alpha = alpha
        self._lora_num_tokens = get_lora_num_tokens()
        self._scaling_factors = get_multilora_scaling()

        for projection, input_width, output_width in (
            ("w1", self.dim, self.hidden_dim),
            ("w2", self.hidden_dim, self.dim),
            ("w3", self.dim, self.hidden_dim),
        ):
            setattr(
                self,
                f"{projection}_lora_A",
                nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.empty(
                                self.num_experts,
                                rank,
                                input_width,
                                device=base_layer.w1.device,
                                dtype=torch.bfloat16,
                            )
                        )
                        for _ in range(n_adapters)
                    ]
                ),
            )
            setattr(
                self,
                f"{projection}_lora_B",
                nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.empty(
                                self.num_experts,
                                output_width,
                                rank,
                                device=base_layer.w1.device,
                                dtype=torch.bfloat16,
                            )
                        )
                        for _ in range(n_adapters)
                    ]
                ),
            )
        self.reset_parameters()

    def reset_parameters(self, index: int | None = None) -> None:
        indices = range(self.n_adapters) if index is None else (index,)
        for adapter_index in indices:
            for projection in ("w1", "w2", "w3"):
                nn.init.kaiming_uniform_(
                    getattr(self, f"{projection}_lora_A")[adapter_index],
                    a=math.sqrt(5),
                )
                nn.init.zeros_(
                    getattr(self, f"{projection}_lora_B")[adapter_index]
                )

    def named_parameters_for_adapter(
        self, index: int
    ) -> list[tuple[str, nn.Parameter]]:
        return [
            (f"{projection}_lora_{factor}", getattr(self, f"{projection}_lora_{factor}")[index])
            for projection in ("w1", "w2", "w3")
            for factor in ("A", "B")
        ]

    def get_lora_param_counts(self) -> tuple[int, int]:
        adapter = sum(
            getattr(self, f"{projection}_lora_{factor}")[0].numel()
            for projection in ("w1", "w2", "w3")
            for factor in ("A", "B")
        )
        base = sum(getattr(self.base_layer, projection).numel() for projection in ("w1", "w2", "w3"))
        return adapter, base

    def state_dict_for_adapter(self, index: int) -> dict[str, torch.Tensor]:
        values = {}
        names = {
            "w1": "gate_proj",
            "w2": "down_proj",
            "w3": "up_proj",
        }
        for projection, exported_name in names.items():
            lora_a = getattr(self, f"{projection}_lora_A")[index].detach()
            lora_b = getattr(self, f"{projection}_lora_B")[index].detach()
            if isinstance(lora_a, DTensor):
                lora_a = lora_a.full_tensor()
                lora_b = lora_b.full_tensor()
            for expert_index in range(self.num_experts):
                values[f"{expert_index}.{exported_name}.lora_A.weight"] = lora_a[
                    expert_index
                ].clone()
                values[f"{expert_index}.{exported_name}.lora_B.weight"] = lora_b[
                    expert_index
                ].clone()
        return values

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        received_input=None,
        received_scores=None,
    ) -> torch.Tensor:
        adapter_index = 0
        scaling = self.alpha / self.rank
        return fp8_lora_experts(
            self.base_layer.w1,
            self.base_layer.w2,
            self.base_layer.w3,
            self.base_layer.w1_scale_inv,
            self.base_layer.w2_scale_inv,
            self.base_layer.w3_scale_inv,
            self.w1_lora_A[adapter_index],
            self.w1_lora_B[adapter_index],
            self.w2_lora_A[adapter_index],
            self.w2_lora_B[adapter_index],
            self.w3_lora_A[adapter_index],
            self.w3_lora_B[adapter_index],
            scaling,
            x,
            num_tokens_per_expert,
            received_input,
            received_scores,
            owner=self,
        )
