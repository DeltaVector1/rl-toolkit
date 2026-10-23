"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). Optional train-level ``ratio_stages`` entries replace those weights
by training step.
A v1 env serves the tasks the orchestrator loaded client-side: a
finite one as a shuffled table (reshuffled on cursor exhaustion), an
infinite one (``num_tasks is None``) straight off its generator — every
pull is a fresh task and there are no epochs to shuffle. A legacy env's
dataset lives on its server, so it serves shuffled task *indices*."""

from __future__ import annotations

import random
from collections.abc import Iterator

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.utils.logger import get_logger


class TrainSource:
    """``next_example(available_permits)`` picks a weighted-RR env and
    returns its next example (or ``None`` when the env's per-call permit
    cost doesn't fit — the dispatch loop retries when permits free up).
    Returned dicts carry ``env_name`` + ``task_idx`` (+ ``task`` for v1 envs,
    whose data is shipped to the env server at dispatch)."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None, ratio_stages=None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        # A finite env's shuffled example table; ``None`` for an infinite env,
        # whose generator (``self.iters``) is pulled per example.
        self.examples: dict[str, list[dict] | None] = {}
        self.iters: dict[str, Iterator[vf.Task]] = {}
        self.cursors: dict[str, int] = {}
        # A fresh group needs room for all of its rollouts. Existing groups
        # still dispatch one rollout at a time as permits become available.
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            if env.tasks is None:  # legacy: sample over the index range from info()
                rows: list[dict] = [{"task_idx": i, "env_name": env.name} for i in range(env.num_tasks)]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            elif env.num_tasks is None:  # infinite: pull the generator per example
                self.examples[env.name] = None
                self.iters[env.name] = env.tasks
            else:
                rows = [{"task_idx": task.data.idx, "task": task, "env_name": env.name} for task in env.tasks]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size

        self.env_names = [e.name for e in self.envs]
        self.weights: list[float] = [float(e.config.ratio) for e in self.envs]
        self._last_logged_stage: int | None = None
        self.ratio_stages: dict[str, tuple[tuple[int, float], ...]] | None = None
        if ratio_stages is not None:
            if not ratio_stages:
                raise ValueError("ratio_stages must not be empty")
            self.ratio_stages = {}
            starts = [stage.start_step for stage in ratio_stages]
            if starts[0] != 1:
                raise ValueError("ratio_stages must start at step 1")
            if starts != sorted(starts) or len(starts) != len(set(starts)):
                raise ValueError("ratio_stages must use strictly increasing start_step values")
            env_names = set(self.env_names)
            for stage in ratio_stages:
                stage_names = set(stage.ratios)
                if stage_names != env_names:
                    missing = sorted(env_names - stage_names)
                    extra = sorted(stage_names - env_names)
                    raise ValueError(
                        "ratio_stages ratios must exactly match training environment names; "
                        f"missing: {missing}, extra: {extra}"
                    )
                if any(ratio <= 0 for ratio in stage.ratios.values()):
                    raise ValueError("ratio_stages ratios must be greater than zero")
            for env_name in self.env_names:
                self.ratio_stages[env_name] = tuple(
                    (int(stage.start_step), float(stage.ratios[env_name])) for stage in ratio_stages
                )

    def weights_for_step(self, step: int | None) -> list[float]:
        """Return the configured weights for the active training step."""
        if self.ratio_stages is None:
            return self.weights
        if not isinstance(step, int) or isinstance(step, bool) or step < 1:
            raise ValueError("step must be a positive integer when ratio_stages are configured")
        weights: list[float] = []
        for env_name in self.env_names:
            active_ratio = None
            for start_step, ratio in self.ratio_stages[env_name]:
                if start_step > step:
                    break
                active_ratio = ratio
            assert active_ratio is not None
            weights.append(active_ratio)
        stage_start = max(start for start, _ in self.ratio_stages[self.env_names[0]] if start <= step)
        if stage_start != self._last_logged_stage:
            get_logger().info(
                f"Train ratio stage at step {step}: "
                f"{dict(zip(self.env_names, weights, strict=True))}"
            )
            self._last_logged_stage = stage_start
        return weights

    def next_example(self, available_permits: int, *, step: int | None = None) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights_for_step(step), k=1)[0]
        if self.env_costs[env_name] > available_permits:
            return None
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if rows is None:  # infinite env: pull the next generated task
            task = next(self.iters[env_name])
            return {"task_idx": task.data.idx, "task": task, "env_name": env_name}
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
