from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from prime_rl.configs.trainer import FakeDataLoaderConfig
from prime_rl.trainer.rl.packer import BasePacker, setup_packer
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.world import get_world
from prime_rl.transport import (
    MicroBatch,
    MicroBatchReceiver,
    TransportConfig,
    setup_micro_batch_receiver,
)


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    # Token level
    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    ref_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperatures: Float[Tensor, "batch seq"]  # Per-token temperatures
    env_names: list[str]
    sequence_lengths: list[int]

    # Batch level
    lora_num_tokens: Int[Tensor, "n_loras"]
    seq_lens: Int[Tensor, "segments"]

    # MoE router replay
    routed_experts: Int[Tensor, "batch seq layers topk"] | None

    # Generic multimodal kwargs — flat dict matching the model's forward
    # signature (e.g. ``{"pixel_values": ..., "image_grid_thw": ...}`` for
    # Qwen3-VL; ``{"pixel_values": ...}`` for Gemma3-VL). The trainer
    # ``**`` -unpacks this into the forward call, so any HF VLM whose
    # processor and forward agree on kwarg names works out of the box.
    mm_kwargs: dict[str, Tensor] | None
    # mm_token_type_ids: token type per token [batch seq], int64 (0=text, 1=image, 2=video)
    mm_token_type_ids: Int[Tensor, "batch seq"] | None

    # Per-token component weight streams. ``None`` means absent: no ce/ref_kl
    # component, rl weight 1.0 on every loss-masked token.
    rl_weights: Float[Tensor, "batch seq"] | None
    ce_weights: Float[Tensor, "batch seq"] | None
    ref_kl_weights: Float[Tensor, "batch seq"] | None

    # Packer-derived metadata used for run-local debug exports.
    run_id: str | None
    run_step: int | None


class FakeDataLoader:
    def __init__(
        self,
        config: FakeDataLoaderConfig,
        seq_len: int,
        dp_world_size: int,
        dp_rank: int,
    ):
        self.world = get_world()
        self.dp_world_size = dp_world_size
        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_world_size={dp_world_size}")
        self.dp_rank = dp_rank

        self.batch_size = config.batch_size
        self.num_micro_batches = self.batch_size // self.dp_world_size
        self.seq_len = seq_len
        self.generate_samples = config.generate_samples
        self.ce_only = config.ce_only
        reference_payload = (
            json.loads(config.reference_path.read_text())
            if config.reference_path is not None
            else None
        )
        self.references = None
        if reference_payload is not None:
            self.references = reference_payload.get("microbatches", [reference_payload])
            if not self.references:
                raise ValueError("fake reference contains no microbatches")
            if len(self.references) not in (1, self.num_micro_batches):
                raise ValueError(
                    "fake reference must contain one microbatch or exactly "
                    f"{self.num_micro_batches}, got {len(self.references)}"
                )
        if self.references is not None and self.ce_only:
            raise ValueError("fake reference_path and ce_only are mutually exclusive")
        for reference in self.references or ():
            token_count = len(reference["input_ids"])
            if token_count != self.seq_len:
                raise ValueError(
                    f"fake reference has {token_count} tokens, expected seq_len={self.seq_len}"
                )
            if len(reference["inference_logprobs"]) != token_count:
                raise ValueError("fake reference log probabilities do not align with its tokens")
            if (
                reference.get("loss_mask") is not None
                and len(reference["loss_mask"]) != token_count
            ):
                raise ValueError("fake reference loss mask does not align with its tokens")
            if (
                reference.get("advantages") is not None
                and len(reference["advantages"]) != token_count
            ):
                raise ValueError("fake reference advantages do not align with its tokens")
            if sum(reference["sequence_lengths"]) != token_count:
                raise ValueError("fake reference sequence lengths do not cover its tokens")
        self.batch_counter = 0
        self.multi_run_manager = get_multi_run_manager()

    def wait_for_batch(self) -> None:
        return

    def get_batch(self) -> list[TensorMicroBatch]:
        if not self.generate_samples:
            get_micro_batch_fn = self._get_micro_batch
        else:
            get_micro_batch_fn = self._get_sample_micro_batch

        # This is a pretty ugly hack to ensure that all CP ranks in a data parallel group receive the same micro batch.
        micro_batches = []
        for micro_batch_idx in range(self.num_micro_batches):
            seed = self.dp_rank * 1000000 + self.batch_counter * 1000 + micro_batch_idx
            generator = torch.Generator().manual_seed(seed)
            if self.references is not None:
                reference = self.references[micro_batch_idx % len(self.references)]
                micro_batches.append(self._get_reference_micro_batch(reference))
            else:
                micro_batches.append(get_micro_batch_fn(generator))

        self.batch_counter += 1
        return micro_batches

    def _get_reference_micro_batch(self, reference: dict) -> TensorMicroBatch:
        input_ids = torch.tensor(reference["input_ids"], dtype=torch.long)
        inference_logprobs = torch.tensor(
            reference["inference_logprobs"], dtype=torch.float
        )
        sequence_lengths = list(reference["sequence_lengths"])
        position_ids = torch.cat([torch.arange(length) for length in sequence_lengths])
        reference_mask = reference.get("loss_mask")
        loss_mask = (
            torch.tensor(reference_mask, dtype=torch.bool)
            if reference_mask is not None
            else torch.ones(input_ids.shape[0], dtype=torch.bool)
        )
        offset = 0
        for length in sequence_lengths:
            # A sequence has no preceding token in its own attention block.
            # Keep this false even when a supplied rollout mask is malformed.
            loss_mask[offset] = False
            offset += length
        reference_advantages = reference.get("advantages")
        advantages = (
            torch.tensor(reference_advantages, dtype=torch.float)
            if reference_advantages is not None
            else torch.zeros_like(inference_logprobs)
        )
        lora_num_tokens = torch.zeros(
            self.multi_run_manager.max_runs, dtype=torch.int32
        )
        lora_num_tokens[0] = input_ids.shape[0]
        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones_like(inference_logprobs).unsqueeze(0),
            "env_names": ["direct_parity"] * input_ids.shape[0],
            "sequence_lengths": sequence_lengths,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "rl_weights": None,
            "ce_weights": None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }

    def _get_sample_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        total_seq_len = 0
        input_ids = []
        position_ids = []
        sequence_lengths = []

        while total_seq_len < self.seq_len:
            # Generate reasonably long documents
            seq_len_to_generate = torch.randint(1, self.seq_len // 8, (1,), generator=generator).item()
            if seq_len_to_generate + total_seq_len > self.seq_len:
                seq_len_to_generate = self.seq_len - total_seq_len
            total_seq_len += seq_len_to_generate
            sequence_lengths.append(seq_len_to_generate)
            tmp_input_ids = torch.randint(0, 120000, (seq_len_to_generate,), generator=generator).long()
            tmp_position_ids = torch.arange(seq_len_to_generate).long()

            input_ids.append(tmp_input_ids)
            position_ids.append(tmp_position_ids)

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        loss_mask = torch.ones(input_ids.shape[0], dtype=torch.bool)
        advantages = torch.randn(input_ids.shape[0], generator=generator)
        inference_logprobs = torch.randn(input_ids.shape[0], generator=generator)
        lora_num_tokens = torch.zeros(self.multi_run_manager.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = input_ids.shape[0]

        rl_weights = torch.zeros(input_ids.shape[0]) if self.ce_only else None
        ce_weights = torch.ones(input_ids.shape[0]) if self.ce_only else None

        return {
            "input_ids": input_ids.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "advantages": advantages.unsqueeze(0),
            "inference_logprobs": inference_logprobs.unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(input_ids.shape[0]).unsqueeze(0),
            "env_names": ["fake"] * input_ids.shape[0],
            "sequence_lengths": sequence_lengths,
            "loss_mask": loss_mask.unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": torch.tensor(sequence_lengths, dtype=torch.long),
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "rl_weights": rl_weights.unsqueeze(0) if rl_weights is not None else None,
            "ce_weights": ce_weights.unsqueeze(0) if ce_weights is not None else None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }

    def _get_micro_batch(self, generator: torch.Generator) -> TensorMicroBatch:
        lora_num_tokens = torch.zeros(self.multi_run_manager.max_runs, dtype=torch.int32)
        lora_num_tokens[0] = self.seq_len
        rl_weights = torch.zeros(self.seq_len) if self.ce_only else None
        ce_weights = torch.ones(self.seq_len) if self.ce_only else None
        return {
            "input_ids": torch.randint(
                0,
                100,
                (
                    1,
                    self.seq_len,
                ),
                generator=generator,
            ),
            "position_ids": torch.cat([torch.arange(self.seq_len)]).unsqueeze(0),
            "advantages": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "inference_logprobs": torch.randn(self.seq_len, generator=generator).unsqueeze(0),
            "ref_logprobs": None,
            "temperatures": torch.ones(self.seq_len).unsqueeze(0),
            "env_names": ["fake"] * self.seq_len,
            "sequence_lengths": [self.seq_len],
            "loss_mask": torch.ones(self.seq_len, dtype=torch.bool).unsqueeze(0),
            "lora_num_tokens": lora_num_tokens,
            "seq_lens": torch.tensor([self.seq_len], dtype=torch.long),
            "routed_experts": None,
            "mm_kwargs": None,
            "mm_token_type_ids": None,
            "rl_weights": rl_weights.unsqueeze(0) if rl_weights is not None else None,
            "ce_weights": ce_weights.unsqueeze(0) if ce_weights is not None else None,
            "ref_kl_weights": None,
            "run_id": None,
            "run_step": None,
        }


class DataLoader:
    """Loads serialized data from a data path written by the orchestrator."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        pad_to_multiple_of: int,
        bin_cost: Callable[[Sequence[int]], int],
        config: TransportConfig,
    ):
        self.world = get_world()

        if self.world.is_master:
            self.packer: BasePacker = setup_packer(
                dp_world_size=dp_world_size,
                seq_len=seq_len,
                transport_config=config,
                pad_to_multiple_of=pad_to_multiple_of,
                bin_cost=bin_cost,
                start_step=start_step,
            )

        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"invalid dp_rank={dp_rank} for dp_world_size={dp_world_size}")
        self.multi_run_manager = get_multi_run_manager()

        self.receiver: MicroBatchReceiver = setup_micro_batch_receiver(output_dir, dp_rank, start_step, config)

    def wait_for_batch(self) -> None:
        if self.world.is_master:
            self.packer._arm_watchdog()
            try:
                self.packer.pack()
            finally:
                self.packer._disarm_watchdog()
        self.receiver.wait()
        self.multi_run_manager.synchronize_state()

    def get_batch(self) -> list[TensorMicroBatch]:
        micro_batches = self.receiver.receive()
        return [self._micro_batch_to_tensor(mb) for mb in micro_batches]

    def _micro_batch_to_tensor(self, micro_batch: MicroBatch) -> TensorMicroBatch:
        """Convert a MicroBatch (msgspec struct with lists) to a TensorMicroBatch (dict with tensors)."""
        if micro_batch.lora_num_tokens is None:
            micro_batch.lora_num_tokens = [0] * self.multi_run_manager.max_runs
            micro_batch.lora_num_tokens[0] = len(micro_batch.input_ids)
        mm_kwargs: dict[str, Tensor] | None = None
        if micro_batch.mm_kwargs:
            # Each value is an EncodedTensor (dtype, shape, raw bytes).
            # No batch dim — the orchestrator concatenates per-image along
            # dim=0 generically, matching what each HF VLM's forward expects.
            mm_kwargs = {
                key: torch.frombuffer(bytearray(payload.data), dtype=_torch_dtype(payload.dtype)).reshape(payload.shape)
                for key, payload in micro_batch.mm_kwargs.items()
            }
        routed_experts = None
        packed_routed_experts = micro_batch.routed_experts
        if packed_routed_experts is not None:
            routed_experts = (
                torch.frombuffer(
                    packed_routed_experts.data,
                    dtype=_torch_dtype(packed_routed_experts.dtype),
                )
                .reshape(packed_routed_experts.shape)
                # Keep the wire dtype on CPU. The int32 upcast happens on GPU in
                # the train loop; doing it here quadrupled the CPU-resident batch
                # (per rank, whole batch) and at 16K contributed to host OOM.
                .unsqueeze(0)
            )
        return TensorMicroBatch(
            input_ids=torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
            position_ids=torch.tensor(micro_batch.position_ids, dtype=torch.long).unsqueeze(0),
            advantages=torch.tensor(micro_batch.advantages, dtype=torch.float).unsqueeze(0),
            inference_logprobs=torch.tensor(micro_batch.inference_logprobs, dtype=torch.float).unsqueeze(0),
            ref_logprobs=torch.tensor(micro_batch.ref_logprobs, dtype=torch.float).unsqueeze(0)
            if micro_batch.ref_logprobs is not None
            else None,
            loss_mask=torch.tensor(micro_batch.loss_mask, dtype=torch.bool).unsqueeze(0),
            temperatures=torch.tensor(micro_batch.temperatures, dtype=torch.float).unsqueeze(0),
            env_names=micro_batch.env_names,
            sequence_lengths=micro_batch.sequence_lengths,
            lora_num_tokens=torch.tensor(micro_batch.lora_num_tokens, dtype=torch.int32),
            seq_lens=torch.tensor(micro_batch.seq_lens, dtype=torch.long),
            mm_kwargs=mm_kwargs,
            mm_token_type_ids=torch.tensor(micro_batch.mm_token_type_ids, dtype=torch.long).unsqueeze(0)
            if micro_batch.mm_token_type_ids is not None
            else None,
            routed_experts=routed_experts,
            rl_weights=torch.tensor(micro_batch.rl_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.rl_weights is not None
            else None,
            ce_weights=torch.tensor(micro_batch.ce_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.ce_weights is not None
            else None,
            ref_kl_weights=torch.tensor(micro_batch.ref_kl_weights, dtype=torch.float).unsqueeze(0)
            if micro_batch.ref_kl_weights is not None
            else None,
            run_id=micro_batch.run_id,
            run_step=micro_batch.run_step,
        )


def _torch_dtype(name: str) -> torch.dtype:
    """Resolve a numpy/torch dtype name (e.g. ``"float32"``) to torch.dtype."""
    # Strip the ``numpy.`` prefix some dtype reprs carry.
    name = name.replace("numpy.", "")
    if hasattr(torch, name):
        return getattr(torch, name)
    # numpy ↔ torch alias mismatches (rare but possible) — fall back via numpy.
    import numpy as np

    return torch.from_numpy(np.zeros(1, dtype=np.dtype(name))).dtype
