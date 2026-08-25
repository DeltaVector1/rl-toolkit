"""Pipeline-parallel execution of one optimizer step.

Design notes, because the obvious approaches do not fit this trainer:

* A pipeline stage only receives the previous stage's activations, but this
  model also needs ``position_ids``, ``seq_lens`` and ``routed_experts`` on every
  stage. Passing them down the pipe would ship ``routed_experts`` (tens of MB)
  across every boundary. Each rank keeps side inputs keyed by stage and
  micro-batch. Activation checkpointing recomputes inside layers, not at stage
  granularity, so it does not change those keys.

* ``torch.distributed.pipelining`` wants ``loss_fn(output, target)`` on the last
  stage and chunks ``target`` for you. This RL loss takes about eight per-token
  tensors plus variable-length ``sequence_lengths`` splits, which does not fit
  a single chunked target. So the last stage computes the loss inside its own
  forward from its side inputs and returns it, and ``loss_fn`` is the identity.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
import torch.nn as nn


def minimum_pipeline_microbatches(schedule_name: str, pp_size: int) -> int:
    """Return the minimum batch that the selected pipeline schedule accepts."""
    if schedule_name == "1f1b":
        # PyTorch Schedule1F1B rejects fewer micro-batches than stages. Check
        # this before schedule construction so a rollout batch with mostly
        # failed samples can be discarded without stopping all trainer ranks.
        return pp_size
    if schedule_name == "interleaved_1f1b":
        return pp_size
    if schedule_name == "gpipe":
        return 1
    raise ValueError(
        "pp_schedule must be one of ['1f1b', 'interleaved_1f1b', 'gpipe'], "
        f"got {schedule_name!r}"
    )


def validate_interleaved_microbatches(num_microbatches: int, pp_size: int) -> int:
    if num_microbatches < pp_size:
        raise ValueError(
            f"interleaved_1f1b needs at least {pp_size} micro-batches, "
            f"got {num_microbatches}"
        )
    rounds = max(1, num_microbatches // pp_size)
    if num_microbatches % rounds:
        raise ValueError(
            "interleaved_1f1b requires the number of micro-batches to be a "
            f"multiple of the number of rounds ({rounds}), got {num_microbatches}"
        )
    return rounds


def _flatten_schedule_action(action):
    sub_actions = getattr(action, "sub_actions", None)
    if sub_actions is None:
        return (action,)
    flattened = []
    for sub_action in sub_actions:
        flattened.extend(_flatten_schedule_action(sub_action))
    return tuple(flattened)


def _schedule_action_positions(actions_by_rank):
    positions = {}
    for rank, actions in actions_by_rank.items():
        position = 0
        for action in actions:
            if action is None:
                continue
            for flat_action in _flatten_schedule_action(action):
                kind = getattr(getattr(flat_action, "computation_type", None), "value", None)
                stage_index = getattr(flat_action, "stage_index", None)
                microbatch_index = getattr(flat_action, "microbatch_index", None)
                if kind not in {"F", "B", "I", "RECV_F", "RECV_B"}:
                    position += 1
                    continue
                if not isinstance(stage_index, int) or not isinstance(microbatch_index, int):
                    raise RuntimeError(
                        "interleaved transport action has an invalid stage or microbatch"
                    )
                key = (stage_index, microbatch_index, kind)
                if key in positions:
                    raise RuntimeError(f"duplicate interleaved transport action {key}")
                positions[key] = (rank, position)
                position += 1
    return positions


def _interval_capacity(intervals):
    if not intervals:
        return 0
    events = []
    for start, end, _microbatch in intervals:
        if start > end:
            raise RuntimeError("interleaved transport interval ends before it starts")
        events.append((start, 1))
        events.append((end + 1, -1))
    events.sort()
    active = 0
    capacity = 0
    for _position, change in events:
        active += change
        if active < 0:
            raise RuntimeError("interleaved transport interval accounting underflow")
        capacity = max(capacity, active)
    return capacity


def _validate_modulo_slots(intervals, capacity, label):
    if not intervals:
        if capacity != 0:
            raise RuntimeError(f"{label} has slots but no receive intervals")
        return
    if capacity < 1:
        raise RuntimeError(f"{label} has receive intervals but no slots")
    by_slot = {}
    for start, end, microbatch in intervals:
        by_slot.setdefault(microbatch % capacity, []).append((start, end, microbatch))
    for slot, slot_intervals in by_slot.items():
        slot_intervals.sort()
        for previous, current in zip(slot_intervals, slot_intervals[1:]):
            if current[0] <= previous[1]:
                raise RuntimeError(
                    f"{label} modulo slot {slot} overlaps microbatches "
                    f"{previous[2]} and {current[2]}"
                )


def derive_interleaved_transport_capacities(
    actions_by_rank,
    stage_to_rank,
    num_stages: int,
    num_microbatches: int,
):
    """Derive exact receive-ring sizes from Torch's lowered schedule.

    Hidden inputs remain owned through backward because they require gradients.
    Sparse cache inputs are read during forward only.
    """
    if not isinstance(actions_by_rank, dict) or not actions_by_rank:
        raise RuntimeError("Torch did not expose interleaved pipeline actions")
    if not isinstance(stage_to_rank, dict):
        raise RuntimeError("Torch did not expose interleaved stage ownership")
    if num_stages < 1 or num_microbatches < 1:
        raise ValueError("interleaved transport dimensions must be positive")
    positions = _schedule_action_positions(actions_by_rank)
    capacities = {}
    for stage_index in range(num_stages):
        if stage_index not in stage_to_rank:
            raise RuntimeError(f"missing rank for interleaved stage {stage_index}")
        rank = stage_to_rank[stage_index]
        if rank not in actions_by_rank:
            raise RuntimeError(f"missing action list for interleaved rank {rank}")
        expects_forward_receive = (
            stage_index > 0
            and stage_to_rank[stage_index - 1] != rank
        )
        expects_backward_receive = (
            stage_index < num_stages - 1
            and stage_to_rank[stage_index + 1] != rank
        )
        forward_intervals = []
        backward_intervals = []
        sparse_intervals = []
        for microbatch in range(num_microbatches):
            forward = positions.get((stage_index, microbatch, "F"))
            backward = positions.get((stage_index, microbatch, "B"))
            backward_input = positions.get((stage_index, microbatch, "I"))
            if backward is not None and backward_input is not None:
                raise RuntimeError(
                    f"interleaved stage {stage_index} has two backward actions "
                    f"for microbatch {microbatch}"
                )
            backward_end = backward or backward_input
            if forward is None or backward_end is None:
                raise RuntimeError(
                    f"incomplete compute actions for stage {stage_index}, "
                    f"microbatch {microbatch}"
                )
            if forward[0] != rank or backward_end[0] != rank:
                raise RuntimeError(
                    f"compute action for stage {stage_index} is owned by the wrong rank"
                )
            forward_receive = positions.get((stage_index, microbatch, "RECV_F"))
            backward_receive = positions.get((stage_index, microbatch, "RECV_B"))
            if forward_receive is not None:
                if not expects_forward_receive:
                    raise RuntimeError(
                        f"stage {stage_index} has an unexpected forward receive action"
                    )
                if forward_receive[0] != rank:
                    raise RuntimeError("forward receive is owned by the wrong rank")
                if forward is None or backward_end is None:
                    raise RuntimeError(
                        f"incomplete forward interval for stage {stage_index}, "
                        f"microbatch {microbatch}"
                    )
                forward_intervals.append(
                    (forward_receive[1], backward_end[1], microbatch)
                )
                sparse_intervals.append(
                    (forward_receive[1], forward[1], microbatch)
                )
            elif forward is not None and expects_forward_receive:
                raise RuntimeError(
                    f"stage {stage_index} has forward work without a receive action"
                )
            if backward_receive is not None:
                if not expects_backward_receive:
                    raise RuntimeError(
                        f"stage {stage_index} has an unexpected backward receive action"
                    )
                if backward_receive[0] != rank:
                    raise RuntimeError("backward receive is owned by the wrong rank")
                if backward_end is None:
                    raise RuntimeError(
                        f"incomplete backward interval for stage {stage_index}, "
                        f"microbatch {microbatch}"
                    )
                backward_intervals.append(
                    (backward_receive[1], backward_end[1], microbatch)
                )
            elif backward_end is not None and expects_backward_receive:
                raise RuntimeError(
                    f"stage {stage_index} has backward work without a receive action"
                )
        forward_slots = _interval_capacity(forward_intervals)
        backward_slots = _interval_capacity(backward_intervals)
        sparse_slots = _interval_capacity(sparse_intervals)
        _validate_modulo_slots(forward_intervals, forward_slots, "forward transport")
        _validate_modulo_slots(backward_intervals, backward_slots, "backward transport")
        _validate_modulo_slots(sparse_intervals, sparse_slots, "sparse-cache transport")
        capacities[stage_index] = {
            "forward": forward_slots,
            "backward": backward_slots,
            "sparse": sparse_slots,
        }
    return capacities


class PipelineStageModule(nn.Module):
    """One pipeline stage: hidden states in, hidden states (or loss) out.

    ``side_inputs`` are the per-micro-batch tensors this stage will need;
    each lookup uses the schedule micro-batch id.
    """

    def __init__(
        self,
        model: nn.Module,
        is_first: bool,
        is_last: bool,
        loss_closure: Callable[[Any, dict], torch.Tensor] | None = None,
        offload_cfg=None,
        transfer_sparse_indices: bool = False,
        receive_sparse_indices: bool | None = None,
        send_sparse_indices: bool | None = None,
        stage_identity: int = 0,
        layer_indices: Sequence[int] | None = None,
    ):
        super().__init__()
        self.model = model
        self.is_first = is_first
        self.is_last = is_last
        self.stage_identity = int(stage_identity)
        self.layer_indices = (
            tuple(int(index) for index in layer_indices)
            if layer_indices is not None
            else None
        )
        self.loss_closure = loss_closure
        # One OffloadActivations instance PER FORWARD, created in forward().
        # A single instance wrapped around the whole schedule asserts
        # "backward pass should have cleared tracker" the moment 1F1B runs a
        # second forward before the first backward; per-micro-batch instances
        # each get drained by their own backward, so interleaving is irrelevant.
        self.offload_cfg = offload_cfg
        self.transfer_sparse_indices = transfer_sparse_indices
        self.receive_sparse_indices = (
            transfer_sparse_indices
            if receive_sparse_indices is None
            else receive_sparse_indices
        )
        self.send_sparse_indices = (
            transfer_sparse_indices if send_sparse_indices is None else send_sparse_indices
        )
        self.profile_stage = os.environ.get("PRIME_PROFILE_PP_STAGE") == "1"
        self._forward_events = []
        self._backward_events = []
        if self.profile_stage:
            self.register_full_backward_pre_hook(self._record_backward_start)
            self.register_full_backward_hook(self._record_backward_end)
        self._side: dict[tuple[int, int], dict] = {}
        self._consumed_side: set[tuple[int, int]] = set()
        self._active_microbatch: int | None = None

    def _record_backward_start(self, _module, _grad_output) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._backward_events.append([event, None])

    def _record_backward_end(self, _module, _grad_input, _grad_output) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._backward_events[-1][1] = event

    def profile_times_ms(self, *, clear: bool = False) -> tuple[float, float]:
        if not self.profile_stage:
            return 0.0, 0.0
        torch.cuda.synchronize()
        forward = sum(start.elapsed_time(end) for start, end in self._forward_events)
        backward = sum(
            start.elapsed_time(end)
            for start, end in self._backward_events
            if end is not None
        )
        if clear:
            self._forward_events.clear()
            self._backward_events.clear()
        return forward, backward

    def set_side_inputs(self, side: Mapping | Sequence[dict]) -> None:
        if isinstance(side, Mapping):
            entries = {}
            for key, value in side.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    raise RuntimeError(
                        "pipeline side-input keys must be (stage_identity, microbatch_id)"
                    )
                normalized = (int(key[0]), int(key[1]))
                if normalized in entries:
                    raise RuntimeError(f"duplicate pipeline side-input key {normalized}")
                entries[normalized] = value
            self._side = entries
        else:
            self._side = {
                (self.stage_identity, microbatch_id): value
                for microbatch_id, value in enumerate(side)
            }
        self._consumed_side.clear()
        self._active_microbatch = None

    def reset(self) -> None:
        self._side.clear()
        self._consumed_side.clear()
        self._active_microbatch = None

    def set_active_microbatch(self, microbatch_id: int) -> None:
        if self._active_microbatch is not None:
            raise RuntimeError(
                f"pipeline stage {self.stage_identity} already has active microbatch "
                f"{self._active_microbatch}"
            )
        self._active_microbatch = int(microbatch_id)

    def clear_active_microbatch(self) -> None:
        self._active_microbatch = None

    def assert_side_inputs_consumed(self) -> None:
        if self._side:
            keys = sorted(self._side)
            raise RuntimeError(
                f"pipeline stage {self.stage_identity} did not consume side inputs {keys[:4]}"
            )

    def _next_side(self, microbatch_id: int | None = None) -> dict:
        # Side inputs are staged on CPU and moved to the GPU only when the
        # micro-batch's forward consumes them; the keyed entry is dropped so the
        # CPU copy frees as the schedule advances. Keeping the WHOLE batch's
        # side inputs GPU-resident cost ~7.6 GB/rank at 190 micro-batches
        # (routed_experts alone is ~40 MB each) and OOM'd the 12-layer stages.
        if microbatch_id is None:
            microbatch_id = self._active_microbatch
        if microbatch_id is None:
            raise RuntimeError(
                f"pipeline stage {self.stage_identity} requested side inputs without a microbatch id"
            )
        key = (self.stage_identity, int(microbatch_id))
        if key in self._consumed_side:
            raise RuntimeError(f"duplicate pipeline side-input consumption for {key}")
        if key not in self._side:
            raise RuntimeError(f"missing pipeline side inputs for {key}")
        cpu = self._side.pop(key)
        self._consumed_side.add(key)
        if not isinstance(cpu, Mapping):
            raise RuntimeError(f"pipeline side inputs for {key} are not a mapping")
        out = {}
        for k, v in cpu.items():
            if torch.is_tensor(v):
                v = v.to("cuda", non_blocking=True)
                if k == "routed_experts":
                    v = v.to(torch.int32)
            out[k] = v
        return out

    def forward(
        self,
        x: torch.Tensor,
        sparse_indices_cache: torch.Tensor | None = None,
    ):
        from prime_rl.utils.act_offloading import maybe_activation_offloading

        start = None
        if self.profile_stage:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        with maybe_activation_offloading(self.offload_cfg):
            output = self._forward_inner(x, sparse_indices_cache)
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._forward_events.append((start, end))
        return output

    def _forward_inner(
        self,
        x: torch.Tensor,
        sparse_indices_cache: torch.Tensor | None = None,
    ):
        if sparse_indices_cache is not None and self.receive_sparse_indices:
            # Send the unsigned 16-bit index bits through an FP16 tensor. NCCL
            # preserves these bits, including patterns which encode FP16 NaN.
            # SparseCachePipelineStage keeps this discrete input out of the
            # backward graph. A production micro-batch carries 16 MiB of index
            # bits; allocating its unused gradient for every pipeline slot
            # costs 1.5 GiB at 95 micro-batches and can OOM during schedule
            # setup before the first forward.
            sparse_indices_cache = (
                sparse_indices_cache.view(torch.int16)
                .to(torch.int32)
                .bitwise_and_(0xFFFF)
            )
        elif not self.receive_sparse_indices:
            sparse_indices_cache = None
        side = self._next_side()
        kwargs = dict(
            position_ids=side["position_ids"],
            routed_experts=side.get("routed_experts"),
            seq_lens=side.get("seq_lens"),
            seq_lens_are_pre_shard=side.get("seq_lens_are_pre_shard", False),
        )

        pipeline_kwargs = {}
        if self.layer_indices is not None:
            pipeline_kwargs = {
                "pipeline_layer_indices": self.layer_indices,
                "pipeline_apply_norm": self.is_last,
            }

        if self.is_last:
            # Go through the same helper the non-PP path uses so the fused output
            # head still receives labels and temperature and computes logprobs
            # itself; recomputing them from logits would diverge numerically.
            from prime_rl.trainer.model import forward as model_forward

            out = model_forward(
                self.model,
                x if self.is_first else None,
                side["position_ids"],
                seq_lens=side.get("seq_lens"),
                labels=side.get("labels"),
                temperature=side.get("temperatures"),
                routed_experts=side.get("routed_experts"),
                seq_lens_are_pre_shard=side.get("seq_lens_are_pre_shard", False),
                inputs_embeds=None if self.is_first else x,
                sparse_indices_cache=sparse_indices_cache,
                **pipeline_kwargs,
            )
            assert self.loss_closure is not None, "last stage needs a loss closure"
            return self.loss_closure(out, side)

        # First and middle stages run the decoder only and hand on hidden states.
        decoder = self.model.model
        out = decoder(
            input_ids=x if self.is_first else None,
            inputs_embeds=None if self.is_first else x,
            **kwargs,
            sparse_indices_cache=sparse_indices_cache,
            **pipeline_kwargs,
        )
        if not self.transfer_sparse_indices:
            return out.last_hidden_state
        if not self.send_sparse_indices:
            placeholder = torch.zeros(1, dtype=torch.float16, device=out.last_hidden_state.device)
            return out.last_hidden_state, placeholder
        if out.past_key_values is None:
            raise RuntimeError(
                "pipeline sparse-index transfer was enabled, but the stage did not produce a cache"
            )
        cache_transport = out.past_key_values.to(torch.int16).view(torch.float16)
        return out.last_hidden_state, cache_transport


def sparse_cache_transport_shape(
    seq_len: int, index_topk: int
) -> tuple[int, int, int, int]:
    """Return the exact cache shape emitted by sparse MLA on one CP rank.

    ``index_n_heads`` configures work inside the indexer. Its final cache is a
    single shared top-k list for each token, as fixed by
    ``Indexer.compute_sparse_indices``. Using ``index_n_heads`` here made each
    pipeline buffer 32 times too large for GLM-5.2.
    """
    return 1, seq_len, 1, index_topk


def _keyed_pipeline_stage_cls():
    from torch.distributed.pipelining import PipelineStage

    class KeyedPipelineStage(PipelineStage):
        def forward_one_chunk(
            self, fwd_chunk_id, args, kwargs=None, save_forward_output=True
        ):
            self.submod.set_active_microbatch(fwd_chunk_id)
            try:
                return super().forward_one_chunk(
                    fwd_chunk_id,
                    args,
                    kwargs,
                    save_forward_output=save_forward_output,
                )
            finally:
                self.submod.clear_active_microbatch()

    return KeyedPipelineStage


def _bounded_pipeline_stage_cls():
    from torch.distributed.pipelining.stage import (
        _RecvInfo,
        _RootArgPlaceholder,
        _make_tensor_from_meta,
    )

    class BoundedPipelineStage(_keyed_pipeline_stage_cls()):
        def require_exact_transport_capacities(self) -> None:
            self._require_exact_transport_capacities = True

        def set_transport_capacities(self, capacities) -> None:
            if not self._require_exact_transport_capacities:
                raise RuntimeError("exact transport capacities were not requested")
            if not isinstance(capacities, dict):
                raise RuntimeError("interleaved transport capacities are invalid")
            try:
                forward = int(capacities["forward"])
                backward = int(capacities["backward"])
                sparse = int(capacities["sparse"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("interleaved transport capacities are incomplete") from exc
            if min(forward, backward, sparse) < 0:
                raise RuntimeError("interleaved transport capacities cannot be negative")
            if self.is_first and forward != 0:
                raise RuntimeError("first interleaved stage cannot receive hidden states")
            if not self.is_first and forward == 0:
                raise RuntimeError("non-first interleaved stage needs hidden receive slots")
            if self.is_last and backward != 0:
                raise RuntimeError("last interleaved stage cannot receive gradients")
            if not self.is_last and backward == 0:
                raise RuntimeError("non-last interleaved stage needs gradient slots")
            if not self.is_first and sparse == 0:
                raise RuntimeError("non-first interleaved stage needs sparse-cache slots")
            self._forward_slot_count = forward
            self._backward_slot_count = backward
            self._sparse_cache_slot_count = sparse
            self._transport_slot_count = max(forward, backward, sparse)
            self._transport_capacities_set = True

        def _prepare_transport_rings(self, num_microbatches):
            if num_microbatches < 1:
                raise RuntimeError("pipeline needs at least one micro-batch")
            slot_count = min(num_microbatches, max(1, self.num_stages))
            if getattr(self, "_require_exact_transport_capacities", False):
                if not getattr(self, "_transport_capacities_set", False):
                    raise RuntimeError(
                        "interleaved transport capacities were not set before stage initialization"
                    )
                forward_slot_count = self._forward_slot_count
                backward_slot_count = self._backward_slot_count
            else:
                forward_slot_count = slot_count if not self.is_first else 0
                backward_slot_count = slot_count if not self.is_last else 0
                self._sparse_cache_slot_count = slot_count if not self.is_first else 0
                self._transport_slot_count = slot_count
            if self.is_first:
                self._forward_slots = []
                self._forward_slot_by_key = {}
                self._forward_slot_owner = []
                self._forward_events = []
                self._forward_receive_keys = set()
            else:
                self._forward_slots = [
                    _make_tensor_from_meta(self.inputs_meta[0], self.device)
                    for _ in range(forward_slot_count)
                ]
                self._forward_slot_by_key = {
                    (self.stage_index, chunk_id): chunk_id % forward_slot_count
                    for chunk_id in range(num_microbatches)
                }
                self._forward_slot_owner = [None] * forward_slot_count
                self._forward_events = [None] * forward_slot_count
                self._forward_receive_keys = set()
            if self.is_last:
                self._backward_slots = []
                self._backward_slot_by_key = {}
                self._backward_slot_owner = []
                self._backward_events = []
                self._backward_receive_keys = set()
            else:
                output_meta = self.get_outputs_meta()[0]
                self._backward_slots = [
                    _make_tensor_from_meta(output_meta, self.device)
                    for _ in range(backward_slot_count)
                ]
                self._backward_slot_by_key = {
                    (self.stage_index, chunk_id): chunk_id % backward_slot_count
                    for chunk_id in range(num_microbatches)
                }
                self._backward_slot_owner = [None] * backward_slot_count
                self._backward_events = [None] * backward_slot_count
                self._backward_receive_keys = set()

        def _begin_forward_receive(self, fwd_chunk_id):
            key = (self.stage_index, int(fwd_chunk_id))
            slot_by_key = self._forward_slot_by_key
            if key not in slot_by_key:
                raise RuntimeError(f"missing hidden-state receive key {key}")
            if key in self._forward_receive_keys:
                raise RuntimeError(f"duplicate hidden-state receive key {key}")
            slot = slot_by_key[key]
            owner = self._forward_slot_owner[slot]
            if owner == key:
                raise RuntimeError(f"duplicate hidden-state slot owner {key}")
            if owner is not None:
                event = self._forward_events[slot]
                if event is None:
                    raise RuntimeError(f"hidden-state buffer overwrite before {owner} completed")
                event.synchronize()
            self._forward_receive_keys.add(key)
            self._forward_slot_owner[slot] = key

        def _finish_forward_receive(self, fwd_chunk_id):
            key = (self.stage_index, int(fwd_chunk_id))
            if key not in self._forward_receive_keys:
                return
            slot = self._forward_slot_by_key[key]
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            self._forward_events[slot] = event

        def _begin_backward_receive(self, bwd_chunk_id):
            key = (self.stage_index, int(bwd_chunk_id))
            slot_by_key = self._backward_slot_by_key
            if key not in slot_by_key:
                raise RuntimeError(f"missing gradient receive key {key}")
            if key in self._backward_receive_keys:
                raise RuntimeError(f"duplicate gradient receive key {key}")
            slot = slot_by_key[key]
            owner = self._backward_slot_owner[slot]
            if owner == key:
                raise RuntimeError(f"duplicate gradient slot owner {key}")
            if owner is not None:
                event = self._backward_events[slot]
                if event is None:
                    raise RuntimeError(f"gradient buffer overwrite before {owner} completed")
                event.synchronize()
            self._backward_receive_keys.add(key)
            self._backward_slot_owner[slot] = key

        def _finish_backward_receive(self, bwd_chunk_id):
            key = (self.stage_index, int(bwd_chunk_id))
            if key not in self._backward_receive_keys:
                return
            slot = self._backward_slot_by_key[key]
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            self._backward_events[slot] = event

        def _prepare_forward_infra(self, num_microbatches, args, kwargs=None):
            outputs = ()
            if self.inputs_meta is None:
                outputs = self._shape_inference(args, kwargs)
            if self.inputs_meta is None:
                raise RuntimeError(
                    "pipeline shape inference did not configure input metadata"
                )
            if len(self.inputs_meta) != 1:
                raise RuntimeError(
                    "bounded pipeline stages require one hidden-state input"
                )
            self._prepare_transport_rings(num_microbatches)
            for chunk_id in range(num_microbatches):
                if self.is_first:
                    self.args_recv_info[chunk_id] = tuple(
                        _RootArgPlaceholder(i) for i in self.inputs_meta
                    )
                    continue
                hidden = _RecvInfo(
                    f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                    self.stage_index - 1,
                    self._forward_slots[
                        self._forward_slot_by_key[(self.stage_index, chunk_id)]
                    ],
                )
                hidden.buffer.requires_grad_(True)
                self.args_recv_info[chunk_id] = (hidden,)
            self.act_send_info = {
                idx: ([] if self.is_last else [self.stage_index + 1])
                for idx in range(len(self.get_outputs_meta()))
            }
            return outputs

        def _create_grad_recv_info(self, act_send_info):
            if self.is_last:
                return ()
            hidden_dst = act_send_info[0][0]
            return (
                _RecvInfo(
                    f"recv_grad_for_{self.stage_index}_from_{hidden_dst}",
                    hidden_dst,
                    self._backward_slots[0],
                ),
            )

        def _prepare_backward_infra(self, num_microbatches):
            self.chunks = num_microbatches
            for mb_index in range(num_microbatches):
                if self.is_last:
                    self.grad_recv_info[mb_index] = ()
                    continue
                hidden_dst = self.act_send_info[0][0]
                slot = self._backward_slot_by_key[(self.stage_index, mb_index)]
                self.grad_recv_info[mb_index] = (
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{hidden_dst}",
                        hidden_dst,
                        self._backward_slots[slot],
                    ),
                )

        def get_fwd_recv_ops(self, fwd_chunk_id):
            if not self.is_first:
                self._begin_forward_receive(fwd_chunk_id)
            return super().get_fwd_recv_ops(fwd_chunk_id)

        def forward_one_chunk(
            self, fwd_chunk_id, args, kwargs=None, save_forward_output=True
        ):
            output = super().forward_one_chunk(
                fwd_chunk_id,
                args,
                kwargs,
                save_forward_output=save_forward_output,
            )
            return output

        def get_bwd_recv_ops(self, bwd_chunk_id):
            if not self.is_last:
                self._begin_backward_receive(bwd_chunk_id)
            return super().get_bwd_recv_ops(bwd_chunk_id)

        def backward_one_chunk(
            self, bwd_chunk_id, loss=None, full_backward=True, last_backward=False
        ):
            output = super().backward_one_chunk(
                bwd_chunk_id,
                loss=loss,
                full_backward=full_backward,
                last_backward=last_backward,
            )
            if not self.is_first:
                self._finish_forward_receive(bwd_chunk_id)
            if not self.is_last:
                self._finish_backward_receive(bwd_chunk_id)
            return output

    return BoundedPipelineStage


def _sparse_cache_pipeline_stage_cls():
    """Return a PipelineStage which sends no gradient for its cache output.

    PyTorch 2.8 assumes every inter-stage tensor participates in backward. The
    sparse-index cache is integer control data carried in FP16 only because the
    pipeline transport accepts floating tensors. Override the four small parts
    of that assumption instead of retaining one full-size zero gradient per
    micro-batch.
    """
    from torch.distributed.pipelining.stage import (
        _RecvInfo,
        _RootArgPlaceholder,
        _make_tensor_from_meta,
    )

    class SparseCachePipelineStage(_bounded_pipeline_stage_cls()):
        def _prepare_forward_infra(self, num_microbatches, args, kwargs=None):
            outputs = ()
            if self.inputs_meta is None:
                outputs = self._shape_inference(args, kwargs)
            if self.inputs_meta is None:
                raise RuntimeError(
                    "pipeline shape inference did not configure input metadata"
                )
            self._prepare_transport_rings(num_microbatches)

            if not self.is_first:
                # The cache is consumed during forward and is not needed by
                # backward. Reuse a small receive ring instead of retaining a
                # 16 MiB buffer for every micro-batch. Record one CUDA event for
                # each slot so NCCL cannot overwrite it before the previous
                # default-stream forward has finished reading it.
                if len(self.inputs_meta) != 2:
                    raise RuntimeError(
                        "sparse-cache pipeline stages require hidden-state and cache inputs"
                    )
                slot_count = self._sparse_cache_slot_count
                slots = [
                    _make_tensor_from_meta(self.inputs_meta[1], self.device)
                    for _ in range(slot_count)
                ]
                self._sparse_cache_slots = slots
                self._sparse_cache_events = [None] * slot_count
                self._sparse_cache_slot_by_key = {
                    (self.stage_index, chunk_id): chunk_id % slot_count
                    for chunk_id in range(num_microbatches)
                }
                self._sparse_cache_slot_owner = [None] * slot_count
                self._sparse_cache_receive_keys = set()

            for chunk_id in range(num_microbatches):
                if self.is_first:
                    self.args_recv_info[chunk_id] = tuple(
                        _RootArgPlaceholder(i) for i in self.inputs_meta
                    )
                    continue
                hidden = _RecvInfo(
                    f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                    self.stage_index - 1,
                    self._forward_slots[
                        self._forward_slot_by_key[(self.stage_index, chunk_id)]
                    ],
                )
                hidden.buffer.requires_grad_(True)
                cache = _RecvInfo(
                    f"recv_cache_for_{self.stage_index}_from_{self.stage_index - 1}",
                    self.stage_index - 1,
                    slots[self._sparse_cache_slot_by_key[(self.stage_index, chunk_id)]],
                )
                self.args_recv_info[chunk_id] = (hidden, cache)

            self.act_send_info = {
                idx: ([] if self.is_last else [self.stage_index + 1])
                for idx in range(len(self.get_outputs_meta()))
            }
            return outputs

        def get_fwd_recv_ops(self, fwd_chunk_id):
            if not self.is_first:
                key = (self.stage_index, int(fwd_chunk_id))
                if key not in self._sparse_cache_slot_by_key:
                    raise RuntimeError(f"missing sparse-cache receive key {key}")
                if key in self._sparse_cache_receive_keys:
                    raise RuntimeError(f"duplicate sparse-cache receive key {key}")
                self._sparse_cache_receive_keys.add(key)
                slot = self._sparse_cache_slot_by_key[key]
                owner = self._sparse_cache_slot_owner[slot]
                if owner == key:
                    raise RuntimeError(f"duplicate sparse-cache slot owner {key}")
                event = self._sparse_cache_events[slot]
                if owner is not None:
                    if event is None:
                        raise RuntimeError(
                            f"sparse-cache buffer overwrite before {owner} completed"
                        )
                    event.synchronize()
            return super().get_fwd_recv_ops(fwd_chunk_id)

        def forward_one_chunk(
            self, fwd_chunk_id, args, kwargs=None, save_forward_output=True
        ):
            output = super().forward_one_chunk(
                fwd_chunk_id,
                args,
                kwargs,
                save_forward_output=save_forward_output,
            )
            if not self.is_first:
                key = (self.stage_index, int(fwd_chunk_id))
                slot = self._sparse_cache_slot_by_key[key]
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                self._sparse_cache_events[slot] = event
                self._sparse_cache_slot_owner[slot] = key
            return output

        def get_fwd_send_ops(self, fwd_chunk_id):
            ops = super().get_fwd_send_ops(fwd_chunk_id)
            stage_output, input_values = self.fwd_cache[fwd_chunk_id]
            if not self.is_last:
                # P2POp keeps the real cache tensor alive through the send.
                # Backward ignores this discrete output, so do not retain it
                # in the forward cache for the rest of the schedule.
                stage_output = (stage_output[0], None)
            if not self.is_first:
                input_values = list(input_values)
                input_values[1] = None
            self.fwd_cache[fwd_chunk_id] = (stage_output, input_values)
            return ops

        def _retrieve_recv_grads(self, bwd_chunk_id):
            hidden_grad = super()._retrieve_recv_grads(bwd_chunk_id)
            if len(hidden_grad) != 1:
                raise RuntimeError(
                    "sparse-cache pipeline expected one differentiable output"
                )
            return hidden_grad[0], None

        def _create_grad_send_info(self, args_recv_info):
            send_info = super()._create_grad_send_info(args_recv_info)
            if not self.is_first:
                if len(send_info) != 2:
                    raise RuntimeError(
                        "sparse-cache pipeline stages require hidden-state and cache inputs"
                    )
                send_info[1] = None
            return send_info

    return SparseCachePipelineStage


def force_fsdp_lazy_init(model: nn.Module) -> int:
    """Run FSDP2's lazy initialisation before the pipeline schedule starts.

    FSDP normally initialises its comm context on the first forward. A pipeline
    schedule can issue a backward prefetch (``_prefetch_unshard``) before that
    has happened for a given module, and the prefetch reads
    ``all_gather_copy_in_stream``, which only exists post-init -- surfacing as
    ``'FSDPCommContext' object has no attribute 'all_gather_copy_in_stream'``.
    Initialising up front avoids depending on schedule ordering.
    """
    from torch.distributed.fsdp import FSDPModule

    count = 0
    for module in model.modules():
        if isinstance(module, FSDPModule):
            state = module._get_fsdp_state()
            if state is not None:
                state._lazy_init()
                count += 1
    return count


def build_schedule(
    stage_module,
    pp_rank: int,
    pp_size: int,
    n_microbatches: int,
    pp_group,
    schedule_name: str,
    example_input=None,
    example_output=None,
):
    """Wrap ``stage_module`` in a PipelineStage and pick a schedule.

    1F1B is the default: it keeps only one stage-depth of activations live,
    which matters here because the 48 GB cards are already at 41 GB.

    ``example_input``/``example_output`` bypass runtime shape inference. Without
    them the stages exchange tensor metadata over P2P before the first forward,
    which is another distributed rendezvous that can wedge; every micro-batch
    here is padded to one shape, so the shapes are simply known.
    """
    from torch.distributed.pipelining import (
        Schedule1F1B,
        ScheduleGPipe,
        ScheduleInterleaved1F1B,
    )

    stage_modules = list(stage_module) if isinstance(stage_module, (list, tuple)) else [stage_module]
    if schedule_name == "interleaved_1f1b" and len(stage_modules) != 2:
        raise ValueError("interleaved_1f1b requires exactly two local pipeline stages")
    if schedule_name != "interleaved_1f1b" and len(stage_modules) != 1:
        raise ValueError(f"{schedule_name} requires exactly one local pipeline stage")

    stage_classes = {}
    stages = []
    example_inputs = (
        list(example_input) if schedule_name == "interleaved_1f1b" else [example_input]
    )
    example_outputs = (
        list(example_output) if schedule_name == "interleaved_1f1b" else [example_output]
    )
    local_stage_count = len(stage_modules)
    exact_transport_capacities = schedule_name == "interleaved_1f1b"
    for local_index, module in enumerate(stage_modules):
        if module.transfer_sparse_indices:
            if True not in stage_classes:
                stage_classes[True] = _sparse_cache_pipeline_stage_cls()
            stage_cls = stage_classes[True]
        else:
            if False not in stage_classes:
                stage_classes[False] = _bounded_pipeline_stage_cls()
            stage_cls = stage_classes[False]
        stage_index = pp_rank + local_index * pp_size
        if module.stage_identity != stage_index:
            raise ValueError(
                f"stage identity {module.stage_identity} does not match pipeline index {stage_index}"
            )
        stage = stage_cls(
                module,
                stage_index=stage_index,
                num_stages=pp_size * local_stage_count,
                device=torch.device("cuda"),
                group=pp_group,
                input_args=example_inputs[local_index],
                output_args=example_outputs[local_index],
            )
        if exact_transport_capacities:
            stage.require_exact_transport_capacities()
        stages.append(stage)
    schedules = {
        "1f1b": Schedule1F1B,
        "interleaved_1f1b": ScheduleInterleaved1F1B,
        "gpipe": ScheduleGPipe,
    }
    minimum_pipeline_microbatches(schedule_name, pp_size)
    if schedule_name == "interleaved_1f1b":
        validate_interleaved_microbatches(n_microbatches, pp_size)
    cls = schedules[schedule_name]
    # scale_grads=False is load-bearing. The schedule's default divides every
    # gradient by n_microbatches, which is only correct for a loss that AVERAGES
    # over micro-batches. compute_loss normalizes by the global token count and
    # the non-PP path accumulates loss.backward() per micro-batch unscaled, so
    # the default silently shrank gradients ~n_microbatches-fold (effective LR
    # of lr/N with N varying per batch -- the unexplained tiny grad norms).
    schedule = cls(
        stages if schedule_name == "interleaved_1f1b" else stages[0],
        n_microbatches=n_microbatches,
        loss_fn=lambda output, target: output,
        scale_grads=False,
    )
    if schedule_name == "interleaved_1f1b":
        capacities = derive_interleaved_transport_capacities(
            schedule.pipeline_order_with_comms,
            schedule.stage_index_to_group_rank,
            num_stages=pp_size * local_stage_count,
            num_microbatches=n_microbatches,
        )
        for stage in stages:
            stage.set_transport_capacities(capacities[stage.stage_index])
    return schedule


def pad_side_inputs_to_common_length(
    side_inputs: list[dict], cp_world_size: int = 1
) -> list[dict]:
    """Verify every micro-batch already shares one sequence length.

    Padding happens in the trainer BEFORE context-parallel sharding (see
    ``pad_micro_batch_to_length``), which is the only place it can be done
    correctly: padding a CP shard here would leave the pad tokens interleaved
    through the sequence after ``gather_for_cp`` while the full-length loss
    tensors carry theirs at the end, misaligning every per-token quantity
    against the logprobs. This function therefore only checks the invariant that
    the fixed-size PipelineStage buffers depend on.
    """
    lengths = {sd["input_ids"].shape[1] for sd in side_inputs}
    if len(lengths) > 1:
        raise RuntimeError(
            f"pipeline micro-batches must share one sequence length, got {sorted(lengths)}; "
            "they should have been padded before context-parallel sharding"
        )
    for sd in side_inputs:
        full = sum(sd["sequence_lengths"])
        shard = sd["input_ids"].shape[1]
        if full != shard * cp_world_size:
            raise RuntimeError(
                f"sequence_lengths sum {full} != shard {shard} x cp {cp_world_size}; "
                "loss splits would misalign against the CP-gathered logprobs"
            )
    return side_inputs
