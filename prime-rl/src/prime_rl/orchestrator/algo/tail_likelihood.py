from __future__ import annotations

import math
from typing import TYPE_CHECKING

from prime_rl.orchestrator.algo.base import Algorithm

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout


def tail_likelihood_advantages(rewards: list[float]) -> list[float]:
    """Return canonical N-scaled, centered TailRL weights in input order."""
    count = len(rewards)
    if count < 2:
        raise ValueError("tail_likelihood requires at least two rollouts per group")
    for reward in rewards:
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not 0.0 <= reward <= 1.0
            or not math.isfinite(reward)
        ):
            raise ValueError("tail_likelihood requires finite numeric rewards in [0, 1]")
    order = sorted(range(count), key=rewards.__getitem__)
    weights = [0.0] * count
    previous = 0.0
    cumulative = 0.0
    for rank, index in enumerate(order):
        reward = rewards[index]
        cumulative += (reward - previous) / (count - rank)
        weights[index] = cumulative
        previous = reward
    mean = math.fsum(weights) / count
    return [count * (weight - mean) for weight in weights]


class TailRLAlgorithm(Algorithm):
    """Reward-tail credit with unchanged RL loss routing and token masks."""

    async def score_group(self, group: list[Rollout]) -> None:
        advantages = tail_likelihood_advantages([rollout.reward for rollout in group])
        for rollout, advantage in zip(group, advantages, strict=True):
            rollout.assign_advantages(advantage)
