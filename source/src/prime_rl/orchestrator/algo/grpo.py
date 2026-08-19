from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.utils.client import InferencePool


DECENSOR_QUALITY_ADVANTAGE_WEIGHT = 1.0


def calculate_decensor_advantages(
    reasoning_requirement_passes: list[int],
    quality_rewards: list[float],
    *,
    quality_weight: float = DECENSOR_QUALITY_ADVANTAGE_WEIGHT,
) -> tuple[list[float], list[float], list[float]]:
    if len(reasoning_requirement_passes) != len(quality_rewards):
        raise ValueError(
            "reasoning pass values and quality rewards must have equal length"
        )
    if not reasoning_requirement_passes:
        return [], [], []
    if any(value not in (0, 1) for value in reasoning_requirement_passes):
        raise ValueError("reasoning pass values must be 0 or 1")
    if any(not 0.0 <= float(value) <= 1.0 for value in quality_rewards):
        raise ValueError("quality rewards must be between 0 and 1")

    pass_rate = sum(reasoning_requirement_passes) / len(reasoning_requirement_passes)
    passing_quality = [
        float(quality)
        for passed, quality in zip(
            reasoning_requirement_passes,
            quality_rewards,
            strict=True,
        )
        if passed
    ]
    mean_passing_quality = (
        sum(passing_quality) / len(passing_quality) if passing_quality else 0.0
    )
    reasoning_advantages = [
        float(passed) - pass_rate for passed in reasoning_requirement_passes
    ]
    quality_advantages = [
        (float(quality) - mean_passing_quality) if passed else 0.0
        for passed, quality in zip(
            reasoning_requirement_passes,
            quality_rewards,
            strict=True,
        )
    ]
    combined = [
        reasoning + quality_weight * quality
        for reasoning, quality in zip(
            reasoning_advantages,
            quality_advantages,
            strict=True,
        )
    ]
    return combined, reasoning_advantages, quality_advantages


def _decensor_reward_components(
    group: list[Rollout],
) -> tuple[list[int], list[float]] | None:
    passes: list[int] = []
    quality_rewards: list[float] = []
    for rollout in group:
        details = rollout.info.get("decensor", {})
        if details.get("advantage_scheme") != "separate_reasoning_and_quality":
            return None
        passed = details.get("reasoning_requirement_pass")
        quality = details.get("quality_reward")
        if passed not in (0, 1) or not isinstance(quality, (int, float)):
            return None
        passes.append(int(passed))
        quality_rewards.append(float(quality))
    return passes, quality_rewards


class GRPOAlgorithm(Algorithm):
    def __init__(self, config: GRPOAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.length_penalty = config.length_penalty

    async def score_group(self, group: list[Rollout]) -> None:
        decensor_components = _decensor_reward_components(group)
        if decensor_components is not None:
            passes, quality_rewards = decensor_components
            advantages, reasoning_advantages, quality_advantages = (
                calculate_decensor_advantages(passes, quality_rewards)
            )
            pass_count = sum(passes)
            for rollout, advantage, reasoning_advantage, quality_advantage in zip(
                group,
                advantages,
                reasoning_advantages,
                quality_advantages,
                strict=True,
            ):
                rollout.info["decensor_training"] = {
                    "reasoning_pass_count": pass_count,
                    "group_size": len(group),
                    "reasoning_advantage": reasoning_advantage,
                    "quality_advantage": quality_advantage,
                    "combined_advantage": advantage,
                    "quality_advantage_weight": DECENSOR_QUALITY_ADVANTAGE_WEIGHT,
                }
                rollout.record_metric(
                    "decensor_reasoning_advantage",
                    reasoning_advantage,
                )
                rollout.record_metric(
                    "decensor_quality_advantage",
                    quality_advantage,
                )
                rollout.record_metric(
                    "decensor_combined_advantage",
                    advantage,
                )
                rollout.record_metric(
                    "decensor_reasoning_pass_count",
                    float(pass_count),
                )
                rollout.assign_advantages(advantage)
            return

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
