from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from prime_rl.configs.algorithm import ReasoningFailureGRPOConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


def _decensor_result(rollout: Rollout) -> dict[str, Any]:
    value = rollout.info.get("decensor")
    return value if isinstance(value, dict) else {}


def _failed_only_for_reasoning_policy(rollout: Rollout) -> bool:
    result = _decensor_result(rollout)
    return (
        result.get("evaluation_status") == "scored"
        and rollout.reward == 0.0
        and float(result.get("quality_reward", 0.0)) > 0.0
        and result.get("response_rejected_because") is None
        and result.get("reasoning_has_safety_or_policy_discussion") == 1
        and result.get("reasoning_trace_is_clear") == 1
    )


class ReasoningFailureGRPOAlgorithm(Algorithm):
    def __init__(
        self,
        config: ReasoningFailureGRPOConfig,
        policy_pool: InferencePool,
    ) -> None:
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty
        self.failed_group_scale = config.failed_group_scale
        self.virtual_reward = config.virtual_reward

    def _shape_rewards(self, group: list[Rollout]) -> torch.Tensor:
        rewards = torch.tensor(
            [rollout.reward for rollout in group], dtype=torch.float32
        )
        if self.length_penalty is None:
            return rewards
        output = torch.tensor(
            [rollout.num_output_tokens for rollout in group], dtype=rewards.dtype
        )
        total = torch.tensor(
            [rollout.num_total_tokens for rollout in group], dtype=rewards.dtype
        )
        turns = torch.tensor(
            [rollout.num_turns for rollout in group], dtype=rewards.dtype
        )
        penalty_fraction = (
            self.length_penalty.num_output_tokens_weight
            * (output / output.max().clamp(min=1))
            + self.length_penalty.num_input_tokens_weight
            * ((total - output) / (total - output).max().clamp(min=1))
            + self.length_penalty.num_turns_weight * (turns / turns.max().clamp(min=1))
        )
        return rewards - rewards.mean() * penalty_fraction

    async def score_group(self, group: list[Rollout]) -> None:
        shaped_rewards = self._shape_rewards(group)
        all_failed_for_reasoning = all(
            _failed_only_for_reasoning_policy(rollout) for rollout in group
        )
        if all_failed_for_reasoning:
            values_for_normalization = torch.cat(
                [shaped_rewards, shaped_rewards.new_tensor([self.virtual_reward])]
            )
            mean = values_for_normalization.mean()
            standard_deviation = values_for_normalization.std(unbiased=False)
            advantages = (
                (shaped_rewards - mean)
                / standard_deviation.clamp_min(1e-6)
                * self.failed_group_scale
            )
        else:
            advantages = shaped_rewards - shaped_rewards.mean()
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1e-6)

        for rollout, advantage in zip(group, advantages.tolist(), strict=True):
            rollout.assign_advantages(advantage)
            rollout.info["reasoning_failure_training"] = {
                "all_failed_reasoning_group": all_failed_for_reasoning,
                "environment_reward": rollout.reward,
                "assigned_advantage": advantage,
                "failed_group_scale": self.failed_group_scale,
                "virtual_reward": self.virtual_reward
                if all_failed_for_reasoning
                else None,
            }
            rollout.record_metric(
                "all_failed_reasoning_group", float(all_failed_for_reasoning)
            )
            rollout.record_metric("assigned_advantage", float(advantage))
