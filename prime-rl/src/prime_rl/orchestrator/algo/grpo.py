from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


class GRPOAlgorithm(Algorithm):
    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[Rollout]) -> None:
        rewards = torch.tensor([rollout.reward for rollout in group], dtype=torch.float32)
        if self.length_penalty is None:
            shaped_rewards = rewards
        else:
            output = torch.tensor([rollout.num_output_tokens for rollout in group], dtype=rewards.dtype)
            total = torch.tensor([rollout.num_total_tokens for rollout in group], dtype=rewards.dtype)
            turns = torch.tensor([rollout.num_turns for rollout in group], dtype=rewards.dtype)
            penalty_frac = (
                self.length_penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                + self.length_penalty.num_input_tokens_weight * ((total - output) / (total - output).max().clamp(min=1))
                + self.length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
            )
            shaped_rewards = rewards - rewards.mean() * penalty_frac
        advantages = shaped_rewards - shaped_rewards.mean()
        advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)
        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
