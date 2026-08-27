"""Pipeline parallelism for the GLM MoE DSA trainer.

Why this exists: the expert all-to-all is ~88% of every micro batch, and on this
cluster it crosses a single 10 Gb/s link per node because the expert-parallel
group spans all 64 GPUs. Measured on the real topology, the same all-to-all
inside one node runs 80x faster (971 ms/layer -> 12 ms/layer) because NCCL uses
shared memory instead of the NIC.

Splitting the decoder into ``pp`` contiguous stages shrinks the EP group to the
ranks of a single stage. With 8 nodes and ``pp=8``, ``pp`` is the slowest-varying
mesh dimension, so stage ``s`` is exactly ranks ``[8s, 8s+8)`` -- one node -- and
the all-to-all stops touching the network.

Expert memory per GPU is unchanged, which is what makes this viable at all:
each rank holds ``layers/pp`` layers sharded ``world/pp`` ways rather than all
layers sharded ``world`` ways. For this model that is 75/8 layers over 8 ranks
versus 75 layers over 64 ranks: 11.3 GB either way.
"""

import torch.nn as nn

from prime_rl.utils.logger import get_logger


def _configured_boundaries(
    num_layers: int, pp_size: int, config
) -> list[int] | None:
    """Return a validated explicit plan for the requested stage topology."""
    configured = getattr(config, "pp_virtual_layer_boundaries", None)
    if configured is None:
        configured = getattr(config, "pp_layer_boundaries", None)
    if configured is None:
        return None
    bounds = [int(value) for value in configured]
    if len(bounds) != pp_size + 1:
        raise ValueError(
            f"pipeline boundaries need {pp_size + 1} values, got {len(bounds)}"
        )
    if bounds[0] != 0 or bounds[-1] != num_layers:
        raise ValueError(f"pipeline boundaries must cover [0, {num_layers})")
    if any(start >= end for start, end in zip(bounds, bounds[1:])):
        raise ValueError("pipeline boundaries must have non-empty ordered stages")
    return bounds



def _index_computing_layers(config) -> list[int] | None:
    """Layers that compute their own sparse-MLA top-k indices.

    Layers marked "shared" reuse the previous layer's indices through the
    decoder loop's ``cached_indices``. That chain cannot cross a pipeline stage
    boundary -- the previous layer is on another node -- and a stage starting on
    a "shared" layer receives ``None``, which surfaces deep in the attention
    kernel as ``'NoneType' object has no attribute 'is_contiguous'``. So stage
    boundaries are restricted to layers that compute indices themselves, which
    keeps the numerics identical to a non-pipelined run.
    """
    if getattr(config, "pp_transfer_sparse_indices", False):
        return list(range(getattr(config, "num_hidden_layers", 0)))
    types = getattr(config, "indexer_types", None)
    if types is not None:
        return [i for i, t in enumerate(types) if t != "shared"]
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None:
        return [i for i, t in enumerate(pattern) if t != "S"]
    freq = getattr(config, "index_topk_freq", 1)
    if freq and freq > 1:
        return list(range(0, len(range(getattr(config, "num_hidden_layers", 0))), freq))
    return None


def sparse_cache_stage_flags(
    model: nn.Module,
    pp_rank: int,
    pp_size: int,
    virtual_stage: int = 0,
    virtual_stages: int = 1,
) -> tuple[bool, bool]:
    from prime_rl.trainer.model import get_language_model

    lm = get_language_model(model)
    config = getattr(lm, "config", None) or getattr(model, "config", None)
    types = getattr(config, "indexer_types", None)
    if types is not None:
        computes = {index for index, kind in enumerate(types) if kind != "shared"}
    else:
        pattern = getattr(config, "index_topk_pattern", None)
        if pattern is not None:
            computes = {index for index, kind in enumerate(pattern) if kind != "S"}
        else:
            frequency = getattr(config, "index_topk_freq", None)
            computes = (
                set(range(0, config.num_hidden_layers, frequency))
                if frequency and frequency > 1
                else None
            )
    stage_index = pp_rank + virtual_stage * pp_size
    total_stages = pp_size * virtual_stages
    if computes is None:
        return stage_index > 0, stage_index < total_stages - 1
    start = stage_layer_range(config.num_hidden_layers, stage_index, total_stages, config)
    receive = stage_index > 0 and start[0] not in computes
    if stage_index == total_stages - 1:
        return receive, False
    next_start = stage_layer_range(
        config.num_hidden_layers, stage_index + 1, total_stages, config
    )[0]
    return receive, next_start not in computes


def balanced_boundaries(num_layers: int, pp_size: int, allowed: list[int] | None) -> list[int]:
    """Split layers across stages, keeping every boundary on an allowed layer.

    Minimises the largest stage first (that is what sets the pipeline's rate),
    then evens out the rest. Falls back to a contiguous split when there is no
    constraint.
    """
    if pp_size <= 1:
        return [0, num_layers]
    if not allowed:
        base, extra = divmod(num_layers, pp_size)
        out, at = [0], 0
        for s in range(pp_size):
            at += base + (1 if s < extra else 0)
            out.append(at)
        return out

    cand = sorted({b for b in allowed if 0 < b < num_layers})
    INF = float("inf")
    # best[(stage, start)] -> (max_cost, sum_sq_cost, boundary_list)
    from functools import lru_cache

    # Balance PEAK MEMORY, not layer count. Under 1F1B stage s keeps
    # (pp_size - s) micro-batches live, so its footprint is
    #     layers * (per-layer weights + per-layer activations * in-flight)
    # Minimising the largest stage ignores the in-flight term and loads the
    # early stages hardest, which is where the host mirror hits its ceiling
    # first (measured x16: 488 GB shared on stage 0 versus 112 GB on stage 7).
    #
    # Weighting by activations ALONE is worse, not better: it hands the last
    # stage 32 of 78 layers, and weights dominate at ~1.5 GB/layer, so that
    # stage needs 52 GB on a 48 GB card. Both terms have to be in the cost.
    #
    # Constants measured on x16 (10-layer stage 0, 8 micro-batches in flight):
    #   at rest 15018 MiB  -> 1.50 GB per layer of weights
    #   at peak 25938 MiB  -> (25938-15018)/(10*8) = 0.137 GB per layer per
    #                         in-flight micro-batch
    # Only their ratio matters to the optimiser, so drift in absolute values is
    # harmless as long as weights stay roughly an order of magnitude larger.
    _GB_WEIGHTS_PER_LAYER = 1.50
    _GB_ACTS_PER_LAYER_PER_MICROBATCH = 0.137

    def _stage_cost(size: int, stage: int) -> float:
        in_flight = pp_size - stage
        return size * (_GB_WEIGHTS_PER_LAYER + _GB_ACTS_PER_LAYER_PER_MICROBATCH * in_flight)

    @lru_cache(maxsize=None)
    def solve(stage: int, start: int):
        if stage == pp_size - 1:
            cost = _stage_cost(num_layers - start, stage)
            return (cost, cost * cost, (num_layers,))
        best = (INF, INF, ())
        for b in cand:
            if b <= start:
                continue
            if num_layers - b < pp_size - stage - 1:
                break
            cost = _stage_cost(b - start, stage)
            sub_max, sub_sq, tail = solve(stage + 1, b)
            got = (max(cost, sub_max), cost * cost + sub_sq, (b,) + tail)
            if got[:2] < best[:2]:
                best = got
        return best

    result = solve(0, 0)
    if not result[2]:
        raise ValueError(
            f"no stage split of {num_layers} layers over pp={pp_size} lands every "
            f"boundary on an index-computing layer (allowed: {sorted(cand)}); an "
            "even fallback would cut the sparse-MLA index-cache chain and crash "
            "in the attention kernel"
        )
    return [0] + list(result[2])


def compute_balanced_boundaries(num_layers: int, pp_size: int, config=None) -> list[int]:
    """Minimize the maximum layer count when stages can transfer index caches.

    The ordinary planner prioritizes activation memory and deliberately leaves
    12 layers on late stages. Once a sparse-index cache can cross boundaries,
    trainer rate becomes the first objective. Among equal-rate plans, keep
    fewer layers on early stages because 1F1B holds more activations there.
    """
    from functools import lru_cache

    hidden_size = getattr(config, "hidden_size", 0)
    vocab_size = getattr(config, "vocab_size", 0)
    embedding_gb = hidden_size * vocab_size * 2 / 1_000_000_000
    output_head_gb = 0.0 if getattr(config, "tie_word_embeddings", False) else embedding_gb

    @lru_cache(maxsize=None)
    def solve(stage: int, start: int):
        remaining_stages = pp_size - stage
        def stage_cost(size: int) -> float:
            cost = size * (1.50 + 0.137 * remaining_stages)
            if stage == 0:
                cost += embedding_gb
            if stage == pp_size - 1:
                cost += output_head_gb
            return cost

        if remaining_stages == 1:
            size = num_layers - start
            return (size, stage_cost(size), size * size, (num_layers,))
        best = (float("inf"), float("inf"), float("inf"), ())
        last = num_layers - (remaining_stages - 1)
        for boundary in range(start + 1, last + 1):
            size = boundary - start
            sub_max, sub_live, sub_sq, tail = solve(stage + 1, boundary)
            got = (
                max(size, sub_max),
                max(stage_cost(size), sub_live),
                size * size + sub_sq,
                (boundary,) + tail,
            )
            if got[:3] < best[:3]:
                best = got
        return best

    return [0] + list(solve(0, 0)[3])


def stage_layer_range(num_layers: int, pp_rank: int, pp_size: int, config=None) -> tuple[int, int]:
    """Contiguous [start, end) layer range for a stage.

    With a sparse-MLA index cache present, boundaries are pinned to layers that
    compute their own indices (see ``_index_computing_layers``); otherwise the
    split is even with the remainder going to the earliest stages.
    """
    boundary_plan = getattr(config, "pp_boundary_plan", "balanced")
    configured = (
        _configured_boundaries(num_layers, pp_size, config)
        if config is not None
        else None
    )
    is_interleaved = (
        config is not None and getattr(config, "pp_schedule", None) == "interleaved_1f1b"
    )
    if is_interleaved and getattr(config, "pp_virtual_layer_boundaries", None) is None:
        raise ValueError(
            "interleaved_1f1b requires explicit pp_virtual_layer_boundaries"
        )
    has_virtual_plan = (
        config is not None
        and getattr(config, "pp_virtual_layer_boundaries", None) is not None
    )
    if configured is not None and (has_virtual_plan or boundary_plan == "explicit"):
        bounds = configured
        allowed = _index_computing_layers(config)
        if not getattr(config, "pp_transfer_sparse_indices", False):
            if allowed is not None and any(value not in allowed for value in bounds[1:-1]):
                raise ValueError(
                    "explicit PP boundaries break the sparse index-cache chain"
                )
    elif config is not None and getattr(config, "pp_transfer_sparse_indices", False):
        bounds = compute_balanced_boundaries(num_layers, pp_size, config)
    else:
        allowed = _index_computing_layers(config) if config is not None else None
        bounds = balanced_boundaries(num_layers, pp_size, allowed)
    return bounds[pp_rank], bounds[pp_rank + 1]


def pp_layer_ranges(model: nn.Module, pp_rank: int, pp_size: int) -> list[tuple[int, int]]:
    lm = model
    ranges = getattr(lm, "pp_layer_ranges", None)
    if ranges is None:
        raise RuntimeError("pipeline layer ranges were not recorded on the decoder")
    if getattr(lm, "pp_physical_rank", pp_rank) != pp_rank:
        raise RuntimeError("pipeline layer ranges belong to a different PP rank")
    if getattr(lm, "pp_physical_size", pp_size) != pp_size:
        raise RuntimeError("pipeline layer range size does not match the PP mesh")
    return [tuple(pair) for pair in ranges]


def apply_pp_layers(
    model: nn.Module, pp_rank: int, pp_size: int, virtual_stages: int = 1
) -> tuple[int, int]:
    """Trim ``model`` in place to the decoder layers owned by ``pp_rank``.

    Returns the (start, end) global layer range kept, and records the offset on
    the decoder so router replay keeps indexing ``routed_experts`` globally.
    Call this as early as possible so quantization, LoRA, EP, AC, FSDP and the
    checkpoint loader only ever see this stage's layers.

    The output head is left alone here: ``inject_prime_lm_head`` runs later in
    ``setup_model`` and asserts a real ``nn.Linear``. Use ``apply_pp_boundaries``
    after that to drop it on stages that do not own it.
    """
    from prime_rl.trainer.model import get_language_model

    lm = get_language_model(model)
    num_layers = len(lm.layers)
    # The decoder config carries indexer_types, which constrains where a stage
    # may start; see stage_layer_range.
    cfg = getattr(lm, "config", None) or getattr(model, "config", None)
    if virtual_stages < 1:
        raise ValueError(f"virtual_stages must be positive, got {virtual_stages}")
    virtual_size = pp_size * virtual_stages
    ranges = [
        stage_layer_range(
            num_layers,
            pp_rank + local_index * pp_size,
            virtual_size,
            config=cfg,
        )
        for local_index in range(virtual_stages)
    ]
    if any(start >= end for start, end in ranges):
        raise RuntimeError(f"empty pipeline virtual stage range: {ranges}")
    start, end = ranges[0]

    # Key by GLOBAL layer index. A dense ModuleList would renumber layers 0..k
    # and the checkpoint keys (model.layers.69.*) would no longer resolve, so
    # DCP fails with "Missing key in checkpoint state_dict". A ModuleDict keeps
    # the original names and makes the key itself the global index, so router
    # replay needs no separate offset.
    layers = list(lm.layers)
    owned_layers = [index for pair in ranges for index in range(pair[0], pair[1])]
    lm.layers = nn.ModuleDict({str(i): layers[i] for i in owned_layers})
    lm.pp_layer_offset = start
    lm.pp_layer_ranges = tuple(ranges)
    lm.pp_physical_rank = pp_rank
    lm.pp_physical_size = pp_size
    lm.pp_virtual_stages = virtual_stages
    if pp_rank != 0:
        lm.embed_tokens = None

    get_logger().info(
        f"PP stage {pp_rank}/{pp_size}: virtual ranges {ranges}, layer_offset={start}"
    )
    return start, end


def apply_pp_boundaries(model: nn.Module, pp_rank: int, pp_size: int) -> None:
    """Drop the norm and output head on every stage but the last.

    Worth doing rather than leaving them resident: the head is
    ``hidden x vocab`` = 6144 x 154880, about 1.9 GB in bf16, on each of the
    seven stages that never use it.
    """
    from prime_rl.trainer.model import get_language_model

    if pp_rank == pp_size - 1:
        return
    get_language_model(model).norm = None
    for attr in ("lm_head", "output"):
        if getattr(model, attr, None) is not None:
            setattr(model, attr, None)
    get_logger().info(f"PP stage {pp_rank}/{pp_size}: dropped norm and output head")
