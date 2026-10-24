"""Per-subtask blend control, shared by the four nemotron category tasksets.

Without this, `load()` concatenates every row of every subtask and the training mix is
dictated by raw source size: `nemotron-reasoning` ran 83% science / 2% math purely because
the science file is 40x larger, and `nemotron-tool-calling` ran 41% of an unlearnable
subtask against 4% of its best-signal one.

The design follows NeMo-RL's `use_multiple_dataloader` / `num_prompts_per_dataloader`
(one cycling iterator per source, a per-step quota, small sources repeat rather than
starve) and NeMo-Gym's integer `num_repeats`. The adaptation: our tasksets build a fixed
pool once in `load()` rather than sampling per step, so a quota over the pool produces the
same blend — the orchestrator draws from the pool uniformly.
"""

from __future__ import annotations

import random
from typing import Iterable, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")


class SubtaskMix(BaseModel):
    """Blend control for one subtask. All fields are optional; the default is a no-op."""

    weight: float | None = None
    """Share of the loaded pool, relative to the other weights. Mutually exclusive with
    `max_examples`. Subtasks with fewer rows than their quota are cycled, not truncated."""

    max_examples: int | None = Field(default=None, ge=0)
    """Hard cap on rows kept from this subtask, applied after the seeded shuffle so it is a
    uniform sample rather than a file-order prefix."""

    repeat: int = Field(default=1, ge=1)
    """Integer upsampling applied after the cap."""

    enabled: bool = True
    """False drops the subtask entirely — the config-level way to retire a dead slice."""

    @model_validator(mode="after")
    def _reject_weight_with_cap(self) -> SubtaskMix:
        if self.weight is not None and self.max_examples is not None:
            raise ValueError("set `weight` or `max_examples`, not both")
        if self.weight is not None and self.weight < 0:
            raise ValueError("`weight` must be non-negative")
        return self


def _quotas(weights: dict[str, float], total: int) -> dict[str, int]:
    """Split `total` across `weights` by largest remainder, so the parts sum exactly."""
    scale = sum(weights.values())
    if scale <= 0:
        return {name: 0 for name in weights}
    exact = {name: total * weight / scale for name, weight in weights.items()}
    counts = {name: int(value) for name, value in exact.items()}
    remainder = total - sum(counts.values())
    for name in sorted(exact, key=lambda n: exact[n] - counts[n], reverse=True)[:remainder]:
        counts[name] += 1
    return counts


def _cycle(rows: list[T], count: int) -> list[T]:
    """Take `count` rows, repeating the pool when it is smaller than the quota."""
    if not rows or count <= 0:
        return []
    out = list(rows) * (count // len(rows))
    return out + rows[: count - len(out)]


def apply_mix(
    rows_by_subtask: dict[str, list[T]],
    mix: dict[str, SubtaskMix],
    seed: int,
    total: int | None = None,
) -> list[T]:
    """Blend per-subtask row lists into one pool.

    With no `mix` entries this is plain concatenation — today's behavior. Entries carrying
    `weight` claim a share of `total` (defaulting to the number of rows available); entries
    without one keep their rows subject to `max_examples` / `repeat`.
    """
    kept = {
        name: rows
        for name, rows in rows_by_subtask.items()
        if rows and mix.get(name, SubtaskMix()).enabled
    }
    for name, rows in kept.items():
        random.Random(seed + hash(name) % 10_000).shuffle(rows)

    weighted = {name: m.weight for name in kept if (m := mix.get(name)) and m.weight is not None}
    pool: list[T] = []

    if weighted:
        budget = total if total is not None else sum(len(rows) for rows in kept.values())
        for name, count in _quotas(weighted, budget).items():
            pool.extend(_cycle(kept[name], count))

    for name, rows in kept.items():
        if name in weighted:
            continue
        settings = mix.get(name, SubtaskMix())
        capped = rows[: settings.max_examples] if settings.max_examples is not None else rows
        pool.extend(capped * settings.repeat)

    random.Random(seed).shuffle(pool)
    return pool


def mix_summary(rows_by_subtask: dict[str, list[T]], pool: Iterable[T]) -> dict[str, int]:
    """Realized per-subtask counts, for logging what the blend actually produced.

    Counts by which input list a row came from, by identity — not by any field on the row.
    A taskset's loader key and the `subtask` it stamps on the data need not match (the
    instruction-following env loads `structured_v2_direct` and stamps
    `structured_v2_direct_generation`), and keying off the field reports phantom zeros.
    """
    origin = {id(row): name for name, rows in rows_by_subtask.items() for row in rows}
    counts = {name: 0 for name in rows_by_subtask}
    for row in pool:
        name = origin.get(id(row))
        if name is not None:
            counts[name] += 1
    return counts
