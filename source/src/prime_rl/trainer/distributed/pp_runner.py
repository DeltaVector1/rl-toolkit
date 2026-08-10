"""Pipeline-parallel execution of one optimizer step.

Design notes, because the obvious approaches do not fit this trainer:

* A pipeline stage only receives the previous stage's activations, but this
  model also needs ``position_ids``, ``seq_lens`` and ``routed_experts`` on every
  stage. Passing them down the pipe would ship ``routed_experts`` (tens of MB)
  across every boundary. Instead each rank keeps the per-micro-batch side inputs
  in a list and consumes them in order: a schedule runs each stage's forwards in
  micro-batch order, so an index is enough. Activation checkpointing recomputes
  inside layers, not at stage granularity, so it does not disturb that order.

* ``torch.distributed.pipelining`` wants ``loss_fn(output, target)`` on the last
  stage and chunks ``target`` for you. This RL loss takes about eight per-token
  tensors plus variable-length ``sequence_lengths`` splits, which does not fit
  a single chunked target. So the last stage computes the loss inside its own
  forward from its side inputs and returns it, and ``loss_fn`` is the identity.
"""

from typing import Any, Callable

import torch
import torch.nn as nn


class PipelineStageModule(nn.Module):
    """One pipeline stage: hidden states in, hidden states (or loss) out.

    ``side_inputs`` is the ordered list of per-micro-batch tensors this rank will
    need; ``reset()`` must be called before each optimizer step.
    """

    def __init__(
        self,
        model: nn.Module,
        is_first: bool,
        is_last: bool,
        loss_closure: Callable[[Any, dict], torch.Tensor] | None = None,
        offload_cfg=None,
        transfer_sparse_indices: bool = False,
    ):
        super().__init__()
        self.model = model
        self.is_first = is_first
        self.is_last = is_last
        self.loss_closure = loss_closure
        # One OffloadActivations instance PER FORWARD, created in forward().
        # A single instance wrapped around the whole schedule asserts
        # "backward pass should have cleared tracker" the moment 1F1B runs a
        # second forward before the first backward; per-micro-batch instances
        # each get drained by their own backward, so interleaving is irrelevant.
        self.offload_cfg = offload_cfg
        self.transfer_sparse_indices = transfer_sparse_indices
        self._side: list[dict] = []
        self._idx = 0

    def set_side_inputs(self, side: list[dict]) -> None:
        self._side = side
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def _next_side(self) -> dict:
        # Side inputs are staged on CPU and moved to the GPU only when the
        # micro-batch's forward consumes them; the list slot is dropped so the
        # CPU copy frees as the schedule advances. Keeping the WHOLE batch's
        # side inputs GPU-resident cost ~7.6 GB/rank at 190 micro-batches
        # (routed_experts alone is ~40 MB each) and OOM'd the 12-layer stages.
        cpu = self._side[self._idx]
        self._side[self._idx] = None
        self._idx += 1
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

        with maybe_activation_offloading(self.offload_cfg):
            return self._forward_inner(x, sparse_indices_cache)

    def _forward_inner(
        self,
        x: torch.Tensor,
        sparse_indices_cache: torch.Tensor | None = None,
    ):
        if sparse_indices_cache is not None:
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
        side = self._next_side()
        kwargs = dict(
            position_ids=side["position_ids"],
            routed_experts=side.get("routed_experts"),
            seq_lens=side.get("seq_lens"),
            seq_lens_are_pre_shard=side.get("seq_lens_are_pre_shard", False),
        )

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
        )
        if not self.transfer_sparse_indices:
            return out.last_hidden_state
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


def _sparse_cache_pipeline_stage_cls():
    """Return a PipelineStage which sends no gradient for its cache output.

    PyTorch 2.8 assumes every inter-stage tensor participates in backward. The
    sparse-index cache is integer control data carried in FP16 only because the
    pipeline transport accepts floating tensors. Override the four small parts
    of that assumption instead of retaining one full-size zero gradient per
    micro-batch.
    """
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.stage import (
        _RecvInfo,
        _RootArgPlaceholder,
        _make_tensor_from_meta,
    )

    class SparseCachePipelineStage(PipelineStage):
        def _prepare_forward_infra(self, num_microbatches, args, kwargs=None):
            outputs = ()
            if self.inputs_meta is None:
                outputs = self._shape_inference(args, kwargs)
            if self.inputs_meta is None:
                raise RuntimeError(
                    "pipeline shape inference did not configure input metadata"
                )

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
                slot_count = min(num_microbatches, self.num_stages)
                slots = [
                    _make_tensor_from_meta(self.inputs_meta[1], self.device)
                    for _ in range(slot_count)
                ]
                self._sparse_cache_events = [None] * slot_count

            for chunk_id in range(num_microbatches):
                if self.is_first:
                    self.args_recv_info[chunk_id] = tuple(
                        _RootArgPlaceholder(i) for i in self.inputs_meta
                    )
                    continue
                hidden = _RecvInfo(
                    f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                    self.stage_index - 1,
                    _make_tensor_from_meta(self.inputs_meta[0], self.device),
                )
                hidden.buffer.requires_grad_(True)
                cache = _RecvInfo(
                    f"recv_cache_for_{self.stage_index}_from_{self.stage_index - 1}",
                    self.stage_index - 1,
                    slots[chunk_id % slot_count],
                )
                self.args_recv_info[chunk_id] = (hidden, cache)

            self.act_send_info = {
                idx: ([] if self.is_last else [self.stage_index + 1])
                for idx in range(len(self.get_outputs_meta()))
            }
            return outputs

        def get_fwd_recv_ops(self, fwd_chunk_id):
            if not self.is_first:
                slot = fwd_chunk_id % len(self._sparse_cache_events)
                event = self._sparse_cache_events[slot]
                if event is not None:
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
                slot = fwd_chunk_id % len(self._sparse_cache_events)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                self._sparse_cache_events[slot] = event
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

        def _create_grad_recv_info(self, act_send_info):
            if self.is_last:
                return ()
            hidden_dst = act_send_info[0][0]
            return (
                _RecvInfo(
                    f"recv_grad_for_{self.stage_index}_from_{hidden_dst}",
                    hidden_dst,
                    _make_tensor_from_meta(self.get_outputs_meta()[0], self.device),
                ),
            )

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
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleGPipe

    stage_cls = (
        _sparse_cache_pipeline_stage_cls()
        if stage_module.transfer_sparse_indices
        else PipelineStage
    )
    stage = stage_cls(
        stage_module,
        stage_index=pp_rank,
        num_stages=pp_size,
        device=torch.device("cuda"),
        group=pp_group,
        input_args=example_input,
        output_args=example_output,
    )
    # interleaved_1f1b is deliberately absent: it needs a LIST of stage chunks
    # and runs forwards out of micro-batch order, which breaks the side-input
    # FIFO. Unknown names raise instead of silently training on a different
    # schedule than the config asked for.
    schedules = {
        "1f1b": Schedule1F1B,
        "gpipe": ScheduleGPipe,
    }
    if schedule_name not in schedules:
        raise ValueError(
            f"pp_schedule must be one of {sorted(schedules)}, got {schedule_name!r}"
        )
    cls = schedules[schedule_name]
    # scale_grads=False is load-bearing. The schedule's default divides every
    # gradient by n_microbatches, which is only correct for a loss that AVERAGES
    # over micro-batches. compute_loss normalizes by the global token count and
    # the non-PP path accumulates loss.backward() per micro-batch unscaled, so
    # the default silently shrank gradients ~n_microbatches-fold (effective LR
    # of lr/N with N varying per batch -- the unexplained tiny grad norms).
    return cls(
        stage,
        n_microbatches=n_microbatches,
        loss_fn=lambda output, target: output,
        scale_grads=False,
    )


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
