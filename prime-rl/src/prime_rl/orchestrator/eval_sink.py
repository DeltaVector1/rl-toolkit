"""Collect eval episodes and explicit dispatcher results."""

from __future__ import annotations

import uuid
from collections import defaultdict

from prime_rl.orchestrator.dispatch_results import DispatchFailure, GroupCancellation
from prime_rl.orchestrator.envs import EvalEnvs
from prime_rl.orchestrator.metrics import EvalRollouts
from prime_rl.orchestrator.types import EvalBatch, Rollout
from prime_rl.utils.logger import get_logger


class EvalSink:
    def __init__(self, *, eval_envs: EvalEnvs) -> None:
        self.eval_envs = eval_envs
        self.pending_groups: dict[uuid.UUID, list[Rollout]] = defaultdict(list)
        self.pending_failures: dict[uuid.UUID, list[DispatchFailure]] = defaultdict(list)
        self.pending_cancellations: dict[uuid.UUID, GroupCancellation] = {}
        self.pending_batches: dict[tuple[str, int], list[Rollout]] = defaultdict(list)
        self.pending_batch_failures: dict[tuple[str, int], list[DispatchFailure]] = defaultdict(list)
        self.pending_batch_cancelled: dict[tuple[str, int], int] = defaultdict(int)

    def add(self, episode: list[Rollout]) -> EvalBatch | None:
        group_id = episode[0].group_id
        self.pending_groups[group_id].extend(episode)
        if self._group_count(group_id) >= self.group_size_for(episode[0].env_name):
            self.process_group(group_id)
        key = (episode[0].env_name, episode[0].eval_step)
        return self.process_batch(key) if self._batch_count(key) >= self.batch_size_for(key[0]) else None

    def fail(self, failure: DispatchFailure) -> EvalBatch | None:
        if failure.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {failure.kind} dispatch failure")
        self.pending_failures[failure.group_id].append(failure)
        if self._group_count(failure.group_id) >= self.group_size_for(failure.env_name):
            self.process_group(failure.group_id)
        key = (failure.env_name, failure.step)
        return self.process_batch(key) if self._batch_count(key) >= self.batch_size_for(key[0]) else None

    def cancel(self, cancellation: GroupCancellation) -> EvalBatch | None:
        if cancellation.kind != "eval":
            raise ValueError(f"EvalSink cannot process a {cancellation.kind} group cancellation")
        if cancellation.group_id in self.pending_cancellations:
            raise ValueError(f"duplicate cancellation for group {cancellation.group_id}")
        self.pending_cancellations[cancellation.group_id] = cancellation
        if self._group_count(cancellation.group_id) >= self.group_size_for(cancellation.env_name):
            self.process_group(cancellation.group_id)
        key = (cancellation.env_name, cancellation.step)
        return self.process_batch(key) if self._batch_count(key) >= self.batch_size_for(key[0]) else None

    def _group_count(self, group_id: uuid.UUID) -> int:
        cancellation = self.pending_cancellations.get(group_id)
        return (
            len(self.pending_groups[group_id])
            + sum(f.count for f in self.pending_failures[group_id])
            + (cancellation.count if cancellation is not None else 0)
        )

    def _batch_count(self, key: tuple[str, int | None]) -> int:
        return (
            len(self.pending_batches[key])
            + sum(f.count for f in self.pending_batch_failures[key])
            + self.pending_batch_cancelled[key]
        )

    def group_size_for(self, env_name: str) -> int:
        return self.eval_envs.get(env_name).config.group_size

    def batch_size_for(self, env_name: str) -> int:
        env = self.eval_envs.get(env_name)
        return len(env.examples) * env.config.group_size

    def batch_progress(self) -> list[tuple[str, int, int, int, int]]:
        keys = set(self.pending_batches) | set(self.pending_batch_failures) | set(self.pending_batch_cancelled)
        return [
            (env_name, step, self._batch_count((env_name, step)), self.batch_size_for(env_name), 0)
            for env_name, step in keys
        ]

    def process_group(self, group_id: uuid.UUID) -> None:
        group = self.pending_groups.pop(group_id, [])
        failures = self.pending_failures.pop(group_id, [])
        cancellation = self.pending_cancellations.pop(group_id, None)
        if group:
            env_name, step = group[0].env_name, group[0].eval_step
            task_idx = group[0].task.data.idx
        elif failures:
            env_name, step = failures[0].env_name, failures[0].step
            task_idx = failures[0].task_idx
        elif cancellation is not None:
            env_name, step = cancellation.env_name, cancellation.step
            task_idx = -1
        else:
            return
        key = (env_name, step)
        self.pending_batches[key].extend(group)
        self.pending_batch_failures[key].extend(failures)
        if cancellation is not None:
            self.pending_batch_cancelled[key] += cancellation.count
        if failures or cancellation is not None:
            get_logger().warning(
                f"Finished eval group | env={env_name} task_idx={task_idx} | "
                f"dispatch failures={sum(f.count for f in failures)} "
                f"cancelled={cancellation.count if cancellation is not None else 0}"
            )

    def process_batch(self, key: tuple[str, int | None]) -> EvalBatch:
        env_name, step = key
        self.pending_batch_failures.pop(key, None)
        self.pending_batch_cancelled.pop(key, None)
        return EvalBatch(
            env_name=env_name,
            step=step,
            rollouts=EvalRollouts(self.pending_batches.pop(key, [])),
        )
