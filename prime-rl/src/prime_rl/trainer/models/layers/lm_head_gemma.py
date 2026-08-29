from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.trainer.models.layers.lm_head import (
    PrimeLmOutput,
    _online_logsumexp_and_weighted_update,
    _patch_model_forward,
)
from prime_rl.utils.logger import get_logger


class GemmaFusedOutputLinear(nn.Linear):
    """Chunked softcap head, including model-specific pre-softcap scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        chunk_size: int,
        softcap: float,
        output_multiplier: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.softcap = softcap
        self.output_multiplier = output_multiplier

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        assert labels is not None, "GemmaFusedOutputLinear requires labels"
        assert temperature is not None, "GemmaFusedOutputLinear requires temperature"
        batch, sequence, hidden = hidden_states.shape
        flat_hidden = hidden_states.reshape(batch * sequence, hidden).contiguous()
        flat_labels = labels.reshape(batch * sequence).contiguous()
        inv_temperature = 1.0 / temperature.reshape(batch * sequence).contiguous()
        logprobs, entropy = _SoftcapLogProbEntropy.apply(
            flat_hidden,
            self.weight,
            flat_labels,
            inv_temperature,
            self.chunk_size,
            self.softcap,
            self.output_multiplier,
        )
        return PrimeLmOutput(
            logprobs=logprobs.reshape(batch, sequence),
            entropy=entropy.reshape(batch, sequence),
        )


class GemmaVanillaOutputLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        softcap: float,
        output_multiplier: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias=False)
        self.softcap = softcap
        self.output_multiplier = output_multiplier

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> PrimeLmOutput:
        logits = nn.functional.linear(hidden_states, self.weight)
        logits = logits.float() * self.output_multiplier
        logits = self.softcap * torch.tanh(logits / self.softcap)
        return PrimeLmOutput(logits=logits)


class _SoftcapLogProbEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        weight: Tensor,
        labels: Tensor,
        inv_temperature: Tensor,
        chunk_size: int,
        softcap: float,
        output_multiplier: float,
    ) -> tuple[Tensor, Tensor]:
        token_count = hidden.shape[0]
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        logprobs = torch.empty(token_count, device=hidden.device, dtype=torch.float32)
        entropy = torch.empty_like(logprobs)
        logz = torch.empty_like(logprobs)

        for start in range(0, token_count, chunk_size):
            end = min(start + chunk_size, token_count)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t = inv_temperature[start:end].unsqueeze(-1)
            count = end - start
            maximum = torch.full((count,), float("-inf"), device=hidden.device)
            exp_sum = torch.zeros(count, device=hidden.device)
            weighted_sum = torch.zeros(count, device=hidden.device)
            target_logits = torch.zeros(count, device=hidden.device)

            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                raw = (hidden_chunk @ weight[vocab_start:vocab_end].t()).float()
                capped = softcap * torch.tanh(raw * output_multiplier / softcap)
                scaled = capped * inv_t
                maximum, exp_sum, weighted_sum = _online_logsumexp_and_weighted_update(
                    maximum, exp_sum, weighted_sum, scaled
                )
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    indices = (labels_chunk[mask] - vocab_start).long()
                    target_logits[mask] = scaled[mask, indices]

            chunk_logz = maximum + torch.log(exp_sum)
            logz[start:end] = chunk_logz
            logprobs[start:end] = target_logits - chunk_logz
            entropy[start:end] = chunk_logz - weighted_sum / exp_sum

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz)
        ctx.chunk_size = chunk_size
        ctx.softcap = softcap
        ctx.output_multiplier = output_multiplier
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: Tensor, grad_entropy: Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0), (
            "Backward through entropy is not implemented"
        )
        hidden, weight, labels, inv_temperature, logz = ctx.saved_tensors
        token_count, _ = hidden.shape
        vocab_size = weight.shape[0]
        vocab_chunk_size = min(vocab_size, 8192)
        needs_hidden, needs_weight = ctx.needs_input_grad[:2]
        grad_hidden = torch.zeros_like(hidden) if needs_hidden else None
        grad_weight = torch.zeros_like(weight) if needs_weight else None

        for start in range(0, token_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, token_count)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].float()
            inv_t = inv_temperature[start:end].unsqueeze(-1)
            chunk_logz = logz[start:end]
            for vocab_start in range(0, vocab_size, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                weight_chunk = weight[vocab_start:vocab_end]
                raw = (hidden_chunk @ weight_chunk.t()).float()
                tanh_value = torch.tanh(raw * ctx.output_multiplier / ctx.softcap)
                scaled = ctx.softcap * tanh_value * inv_t
                probabilities = torch.exp(scaled - chunk_logz.unsqueeze(-1))
                grad_logits = -grad_chunk.unsqueeze(-1) * probabilities
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    indices = (labels_chunk[mask] - vocab_start).long()
                    grad_logits[mask, indices] += grad_chunk[mask]
                grad_logits *= inv_t * ctx.output_multiplier * (1 - tanh_value.square())
                if needs_hidden:
                    grad_hidden[start:end].add_(grad_logits.to(hidden.dtype) @ weight_chunk)
                if needs_weight:
                    grad_weight[vocab_start:vocab_end].add_(
                        grad_logits.to(weight.dtype).t() @ hidden_chunk
                    )

        return grad_hidden, grad_weight, None, None, None, None, None


def inject_gemma_lm_head(model: nn.Module, chunk_size: int | None, softcap: float) -> None:
    text_config = model.config.get_text_config()
    output_multiplier = float(getattr(text_config, "output_multiplier", 1.0))
    get_logger().info(
        f"Injecting softcap LM head with chunk size {chunk_size}, "
        f"softcap={softcap}, output_multiplier={output_multiplier}"
    )
    old_head = model.lm_head
    head_type = GemmaFusedOutputLinear if chunk_size is not None else GemmaVanillaOutputLinear
    kwargs = {
        "in_features": old_head.in_features,
        "out_features": old_head.out_features,
        "softcap": softcap,
        "output_multiplier": output_multiplier,
    }
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    model.lm_head = head_type(**kwargs)
    model.lm_head.weight = old_head.weight
    del old_head
    _patch_model_forward(model)
