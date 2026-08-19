#!/usr/bin/env python3
"""Compare the fused Muse head with the official logit formula."""

import torch

from prime_rl.trainer.models.layers.lm_head_gemma import GemmaFusedOutputLinear


torch.manual_seed(7)
hidden = torch.randn(2, 11, 7, requires_grad=True)
weight = torch.randn(31, 7, requires_grad=True)
labels = torch.randint(0, 31, (2, 11))
temperature = torch.rand(2, 11) * 0.7 + 0.6
grad = torch.randn(2, 11)
softcap = 20.0
multiplier = 0.19611613513818404

head = GemmaFusedOutputLinear(7, 31, 4, softcap, multiplier)
head.weight = torch.nn.Parameter(weight.detach().clone())
actual = head(hidden, labels, temperature)
(actual["logprobs"] * grad).sum().backward()
actual_hidden_grad = hidden.grad.detach().clone()
actual_weight_grad = head.weight.grad.detach().clone()

reference_hidden = hidden.detach().clone().requires_grad_(True)
reference_weight = weight.detach().clone().requires_grad_(True)
logits = reference_hidden @ reference_weight.t()
logits = softcap * torch.tanh(logits.float() * multiplier / softcap)
scaled = logits / temperature.unsqueeze(-1)
reference_logprobs = scaled.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
reference_entropy = -(scaled.softmax(-1) * scaled.log_softmax(-1)).sum(-1)
(reference_logprobs * grad).sum().backward()

torch.testing.assert_close(actual["logprobs"], reference_logprobs, atol=2e-5, rtol=2e-5)
torch.testing.assert_close(actual["entropy"], reference_entropy, atol=2e-5, rtol=2e-5)
torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad, atol=2e-5, rtol=2e-5)
torch.testing.assert_close(actual_weight_grad, reference_weight.grad, atol=2e-5, rtol=2e-5)
print("Muse fused LM-head forward and backward parity: PASS")
