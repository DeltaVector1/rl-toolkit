import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

from contextlib import nullcontext
import os
import time
from datetime import timedelta

# Import environment before any other imports
# ruff: noqa: I001

from prime_rl.trainer.models.layers.attn import substitute_ring_attn
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.utils.act_offloading import maybe_activation_offloading
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _set_pg_timeout
from torch.profiler import profile, ProfilerActivity, record_function
from prime_rl.trainer.ckpt import setup_ckpt_managers
from prime_rl.trainer.multi_ckpt import setup_multi_checkpoint_manager
from prime_rl.trainer.optim import setup_optimizer, setup_multi_optimizer
from prime_rl.trainer.scheduler import setup_scheduler, setup_multi_scheduler
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader
from prime_rl.utils.cp import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    setup_cp_params,
    shard_for_cp,
)
from prime_rl.utils.logger import format_time, setup_logger
from prime_rl.trainer.rl.loss import (
    compute_entropy,
    compute_loss,
    compute_importance_ratio_and_mismatch_kl,
    selective_log_softmax,
    setup_rl_loss_fn,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.trainer.rl.token_export import setup_token_exporter
from prime_rl.trainer.model import (
    DTYPE_MAP,
    forward,
    setup_tokenizer,
    setup_model,
    is_tt_moe_model,
    get_load_balance_stats,
)
from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
from prime_rl.trainer.perf import get_perf_counter
from prime_rl.trainer.utils import (
    build_bin_cost,
    GarbageCollection,
    MemoryProfiler,
    Tensors,
    export_benchmark_json,
    filter_rl_trainer_tensor_stats_for_wandb,
    get_zero_gradient_ratio,
    get_ckpt_disk_metrics,
    setup_torch_distributed,
    print_benchmark,
)
from prime_rl.trainer.world import get_world
from prime_rl.trainer.runs import setup_multi_run_manager, Progress, get_multi_run_manager
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.metrics_server import HealthServer, MetricsServer, RunStats
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit, resolve_latest_ckpt_step, to_col_format
from ring_flash_attn import substitute_hf_flash_attn
from torchtitan.distributed.utils import clip_grad_norm_



def move_micro_batch_to_cuda(micro_batch) -> None:
    """Move a micro-batch's tensors to the GPU in place, field by field.

    Under pipeline parallelism every micro-batch is padded up to the global
    16K target. Doing that padding on CPU meant materializing the whole padded
    batch in host RAM on EVERY rank (8 per node) -- measured at ~50 GB/rank at
    16K, which the kernel OOM-killer answered. Moving fields to the GPU first
    drops each CPU copy as it goes, and the subsequent pads then allocate on
    the GPU where the tensors were headed anyway.
    """
    for name in (
        "input_ids",
        "position_ids",
        "advantages",
        "inference_logprobs",
        "ref_logprobs",
        "loss_mask",
        "temperatures",
        "rl_weights",
        "ce_weights",
        "ref_kl_weights",
        "mm_token_type_ids",
        "routed_experts",
    ):
        t = micro_batch[name]
        if t is not None:
            micro_batch[name] = t.to("cuda", non_blocking=True)


def pad_micro_batch_to_distributed_max(micro_batch, group) -> None:
    length = micro_batch["input_ids"].shape[1]
    max_length = torch.tensor(length, device="cuda")
    dist.all_reduce(max_length, op=dist.ReduceOp.MAX, group=group)
    pad_micro_batch_to_length(micro_batch, int(max_length.item()))


def pad_micro_batch_to_length(micro_batch, target_length: int) -> None:
    """Pad one micro-batch to ``target_length`` tokens.

    This must run BEFORE context-parallel sharding. Padding a CP shard instead
    puts the pad tokens at the end of each shard, so after gather_for_cp they sit
    interleaved through the sequence while loss_mask/advantages/inference_logprobs
    still carry theirs at the end -- every per-token tensor then lines up against
    the wrong logprobs, silently corrupting both the loss and the metrics.
    """
    length = micro_batch["input_ids"].shape[1]
    padding = target_length - length
    if padding <= 0:
        return

    for name, value in (
        ("input_ids", 1),
        ("position_ids", 0),
        ("advantages", 0.0),
        ("inference_logprobs", 0.0),
        ("ref_logprobs", 0.0),
        ("loss_mask", False),
        ("temperatures", 1.0),
        ("rl_weights", 0.0),
        ("ce_weights", 0.0),
        ("ref_kl_weights", 0.0),
        ("mm_token_type_ids", 0),
    ):
        tensor = micro_batch[name]
        if tensor is not None:
            micro_batch[name] = torch.nn.functional.pad(tensor, (0, padding), value=value)

    if micro_batch["routed_experts"] is not None:
        routed = micro_batch["routed_experts"]
        micro_batch["routed_experts"] = torch.cat(
            (routed, routed.new_zeros((routed.shape[0], padding, *routed.shape[2:]))),
            dim=1,
        )
    micro_batch["env_names"].extend([""] * padding)
    micro_batch["sequence_lengths"][-1] += padding
    micro_batch["seq_lens"][-1] += padding
    micro_batch["lora_num_tokens"][-1] += padding


@clean_exit
def train(config: TrainerConfig):
    # Setup world and logger
    world = get_world()
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting RL trainer in {world} in {config.output_dir}")

    # Print warning if running in benchmark mode
    if config.bench is not None:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(f"Initializing monitor (wandb={config.wandb}, swanlab={config.swanlab})")
    monitor = setup_monitor(
        config.wandb,
        swanlab_config=config.swanlab,
        file_config=config.file_monitor,
        output_dir=config.output_dir,
        run_config=config,
    )

    # Setup heartbeat (only on rank 0)
    heart = None
    if config.heartbeat is not None and world.is_master:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat.url)

    # Setup metrics server (full on master, health-only on other nodes' local rank 0)
    metrics_server = None
    health_server = None
    if config.metrics_server is not None and world.local_rank == 0:
        if world.is_master:
            logger.info(f"Initializing metrics server on port {config.metrics_server.port}")
            metrics_server = MetricsServer(config.metrics_server)
            metrics_server.start()
        else:
            logger.info(f"Initializing health server on port {config.metrics_server.port}")
            health_server = HealthServer(config.metrics_server.port, config.metrics_server.host)
            health_server.start()

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds), enable_gloo=config.model.fsdp_cpu_offload
    )
    # Configurable to support ROCm/AMD GPUs where reduced precision
    # matmul corrupts softmax over large vocabularies. Override via config
    # (e.g. matmul_precision = "highest") on ROCm.
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Setup multi run manager and offsets (including LoRA validation/scaling hooks if applicable)
    multi_run_manager = setup_multi_run_manager(
        config.output_dir, config.max_concurrent_runs, torch.device("cuda", world.local_rank), config.model.lora
    )

    # Resolve ep="auto" to a concrete integer before creating parallel dims
    resolve_ep(config.model)

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)

    # For single-run, check for checkpoint to resume from
    checkpoint_step = None
    if config.max_concurrent_runs == 1:
        # Set up checkpoint manager for single-run
        logger.info(f"Initializing checkpoint managers ({config.ckpt})")
        ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(config.output_dir, config.ckpt, config.model.lora)

        if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
            if config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = config.ckpt.resume_step
    else:
        # Multi-run uses per-run checkpointing via MultiCheckpointManager
        ckpt_manager, weight_ckpt_manager = setup_multi_checkpoint_manager(config.output_dir)
        logger.info("Initialized multi-run checkpoint manager")

    # Initialize the model and tokenizer
    logger.info(f"Initializing model ({config.model})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
    group_timeout = timedelta(seconds=config.dist_timeout_seconds)
    process_groups = {}
    for mesh in (parallel_dims.world_mesh, *parallel_dims._submeshes.values()):
        for group in mesh.get_all_groups():
            process_groups[id(group)] = group
    for group in process_groups.values():
        _set_pg_timeout(group_timeout, group)
    logger.info(f"Set {len(process_groups)} distributed group timeouts to {config.dist_timeout_seconds} seconds")
    resume_adapter = os.environ.get("PRIME_RL_RESUME_ADAPTER") if checkpoint_step is None else None
    if resume_adapter:
        from safetensors.torch import load_file
        from torch.distributed.tensor import DTensor, distribute_tensor

        adapter = load_file(resume_adapter)
        loaded = set()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                key = (
                    name.replace("._checkpoint_wrapped_module", "")
                    .replace(".lora_A.0", ".lora_A.weight")
                    .replace(".lora_B.0", ".lora_B.weight")
                )
                if key not in adapter:
                    continue
                value = adapter[key].to("cuda", dtype=parameter.dtype)
                if isinstance(parameter, DTensor):
                    value = distribute_tensor(value, parameter.device_mesh, parameter.placements)
                parameter.copy_(value)
                loaded.add(key)
        if loaded != adapter.keys():
            raise ValueError(f"Adapter keys not loaded: {sorted(adapter.keys() - loaded)}")
        logger.info(f"Loaded {len(loaded)} adapter tensors from {resume_adapter}")

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    if config.model.vlm is not None and not getattr(model, "supports_packed_multimodal_training", False):
        raise ValueError("Packed multimodal training requires model support")

    # Set up the loss function for the RL loss type (ce / ref_kl are fixed)
    logger.info(f"Setting up loss function ({config.loss})")
    rl_loss_fn = setup_rl_loss_fn(config.loss)

    # Set up the optimizer
    logger.info(f"Initializing optimizer ({config.optim})")

    if config.max_concurrent_runs == 1:
        optimizer = setup_optimizer(
            config.optim,
            list(model.named_parameters()),
            parallel_dims,
            lora=config.model.lora is not None,
            cpu_offload=config.model.optim_cpu_offload,
        )
        scheduler = setup_scheduler(optimizer, config.scheduler, config.max_steps, config.optim.lr)
    else:
        optimizer = setup_multi_optimizer(config.optim, parallel_dims)
        scheduler = setup_multi_scheduler(optimizer, config.scheduler, config.max_steps)

        # Register checkpoint loading callback at index 1 (after scheduler creation at index 0)
        def load_run_checkpoint(_optimizer, idx: int) -> None:
            ckpt_manager.load_run(idx, optimizer, scheduler)

        optimizer.register_post_creation_callback(load_run_checkpoint, index=1)

    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    # Set up weight broadcast (skip when using fake data since there's no inference server)
    if config.data.fake:
        weight_broadcast = None
        logger.info("Skipping weight broadcast setup (fake data mode)")
    else:
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        weight_broadcast = setup_weight_broadcast(
            config.output_dir,
            config.weight_broadcast,
            parallel_dims,
            config.model.lora,
        )

    if parallel_dims.cp_enabled:
        cp_group = parallel_dims.world_mesh["cp"].get_group()
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()
        if config.model.cp_style == "ring":
            substitute_hf_flash_attn(cp_group, heads_k_stride=1)
            substitute_ring_attn(cp_group, heads_k_stride=1, attn_impl=config.model.attn)
        else:
            from prime_rl.trainer.models.layers.ulysses_attn import (
                substitute_hf_ulysses_attn,
                substitute_ulysses_attn,
            )

            substitute_hf_ulysses_attn(cp_group)
            substitute_ulysses_attn(cp_group, attn_impl=config.model.attn)
        from prime_rl.utils.cp import (
            assert_cp_style_supports_model,
            setup_model_cp,
            setup_sparse_mla_cp,
        )

        assert_cp_style_supports_model(config.model.cp_style, model)
        # sparse MLA is softmax (works with both ring and ulysses).
        setup_sparse_mla_cp(model, cp_group, cp_rank, parallel_dims.cp)
        # Under pipeline parallelism only the last stage owns the output head.
        if getattr(model, "lm_head", None) is not None:
            model.lm_head.cp_group = cp_group
            model.lm_head.cp_rank = cp_rank
            model.lm_head.cp_size = parallel_dims.cp
        # Linear-attn / Mamba layers are only configured under ulysses; with ring
        # we'd have already raised above.
        if config.model.cp_style == "ulysses":
            setup_model_cp(model, cp_group, cp_rank, parallel_dims.cp)

    # Pipeline parallelism: wrap this rank's stage once, before the loop.
    pp_stage_module = None
    pp_rank = 0
    if parallel_dims.pp_enabled:
        from prime_rl.trainer.distributed.pp_runner import PipelineStageModule
        from prime_rl.trainer.distributed.pp_runner import build_schedule as build_pp_schedule
        from prime_rl.trainer.distributed.pp_runner import pad_side_inputs_to_common_length as pad_pp_side_inputs
        from prime_rl.trainer.distributed.pp_runner import force_fsdp_lazy_init

        pp_rank = parallel_dims.get_mesh("pp").get_local_rank()
        # Weight broadcast: tell the gatherer to merge stage dicts over the pp
        # group. "Stage master" = the first rank within each stage, which is
        # exactly the membership of the global master's pp group.
        from prime_rl.trainer.weights import set_pp_gather_context

        # pp is the slowest-varying mesh dim, so stage s owns the contiguous
        # ranks [s*k, (s+1)*k) with k = world/pp; the stage master is the first.
        _ranks_per_stage = dist.get_world_size() // parallel_dims.pp
        set_pp_gather_context(
            parallel_dims.get_mesh("pp").get_group(),
            is_stage_master=(dist.get_rank() % _ranks_per_stage) == 0,
        )
        pp_stage_module = PipelineStageModule(
            model,
            is_first=pp_rank == 0,
            is_last=pp_rank == parallel_dims.pp - 1,
            offload_cfg=config.model.ac_offloading,
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if checkpoint_step is not None:
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        # The checkpoint finished step ``checkpoint_step``; resume training at the next step.
        progress.step += 1
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
    elif resume_adapter:
        progress.step = int(os.environ["PRIME_RL_RESUME_ADAPTER_STEP"]) + 1
        multi_run_manager.wait_for_run(0)
        for idx in multi_run_manager.used_idxs:
            multi_run_manager.progress[idx].step = progress.step

    logger.info(
        f"Starting from step {progress.step} (total_tokens={progress.total_tokens}, total_samples={progress.total_samples})"
    )

    # Set up the data loader (Optionally, use a fake data loader for debugging)
    logger.info(f"Initializing data loader ({config.data})")
    bin_cost = build_bin_cost(model.config)
    if config.data.fake:
        dataloader = FakeDataLoader(config.data.fake, config.model.seq_len, parallel_dims.get_mesh("dp").size())
    else:
        dataloader = DataLoader(
            config.output_dir,
            progress.step,
            parallel_dims.get_mesh("dp").size(),
            config.model.seq_len,
            config.model.cp,
            bin_cost,
            config.rollout_transport,
        )

    token_exporter = setup_token_exporter(config, parallel_dims, world, logger)

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True).__enter__()
        maybe_record_function = record_function
    start_step = progress.step
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = config.max_steps is not None and progress.step == config.max_steps

        logger.debug(f"Starting training step {progress.step}")
        step_start_time = time.perf_counter()

        # In-memory transfers broadcast the incoming policy (v{progress.step-1}) before waiting
        # for its rollouts so the trainer and inference pool join the same update lifecycle.
        if (
            progress.step == start_step
            and weight_broadcast is not None
            and config.weight_broadcast.type in ("nccl", "nixl")
        ):
            logger.info(f"Broadcasting startup policy weights (v{progress.step - 1}) to inference engines")
            multi_run_manager.wait_for_run(0)
            for idx in multi_run_manager.used_idxs:
                multi_run_manager.ready_to_update[idx] = True
            weight_broadcast.broadcast_weights(model, step=progress.step - 1)

        # Wait for the batch to be available
        logger.debug("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.perf_counter()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.perf_counter() - wait_for_batch_start_time
        logger.debug(f"Waited for batch to arrive for {wait_for_batch_time:.2f} seconds")
        load_batch_start_time = time.perf_counter()
        micro_batches = dataloader.get_batch()
        load_data_time = time.perf_counter() - load_batch_start_time
        logger.debug(f"Loaded batch in {load_data_time:.2f} seconds")

        # Reject a corrupt batch instead of dying on it. On 2026-08-06 10:13 a
        # micro-batch arrived with input_ids of shape [1, 9142729177965406720]
        # -- a sequence length just under 2^63, i.e. garbage deserialized as
        # tensor metadata, not a real length. torch raised "Storage size
        # calculation overflowed", rank 0 died, and the other ranks then hung
        # until the NCCL collective timeout. The run sat dead for six hours.
        #
        # Sequences are packed to at most seq_len (times cp for the pre-shard
        # case), so anything past a generous multiple of that is corruption.
        # Skipping costs one batch; crashing costs the whole run.
        _max_len = config.model.seq_len * max(config.model.cp, 1) * 4
        _bad = []
        for _i, _mb in enumerate(micro_batches):
            _ids = _mb.get("input_ids") if isinstance(_mb, dict) else None
            if _ids is None:
                _bad.append((_i, "missing input_ids"))
                continue
            if _ids.dim() != 2 or _ids.shape[0] != 1 or not (0 < _ids.shape[1] <= _max_len):
                _bad.append((_i, f"input_ids shape {tuple(_ids.shape)} (max len {_max_len})"))
        if _bad:
            for _i, _why in _bad[:5]:
                logger.error(f"Corrupt micro-batch {_i}: {_why}")
            logger.error(
                f"Discarding batch at step {progress.step}: {len(_bad)}/{len(micro_batches)} micro-batches corrupt. "
                "Waiting for the next batch rather than crashing the run."
            )
            continue

        batch_size = len(micro_batches)
        memory_profiler = None
        if config.memory_profiler_path is not None:
            memory_profiler = MemoryProfiler(progress.step, config.memory_profiler_path)

        forward_backward_start_time = time.perf_counter()
        seq_len = micro_batches[0]["input_ids"].shape[1]

        # Normalize each loss component by its own global (dp_cp) token count, so every rank
        # divides by the same denominator. With a per-rank denominator, ranks with fewer loss
        # tokens implicitly upweight their per-token gradient contribution after FSDP averaging.
        # FSDP's per-rank divide is undone after the microbatch loop via
        # fsdp_gradient_divide_factor. One batched collective keeps every rank issuing the same
        # op regardless of which components its samples carry.
        local_rl_scale = 0
        local_ce_scale = 0
        local_ref_kl_scale = 0
        for micro_batch in micro_batches:
            mask = micro_batch["loss_mask"]
            rl_w = micro_batch["rl_weights"]
            local_rl_scale += int((mask & (rl_w != 0)).sum()) if rl_w is not None else int(mask.sum())
            if micro_batch["ce_weights"] is not None:
                local_ce_scale += int((micro_batch["ce_weights"] != 0).sum())
            if micro_batch["ref_kl_weights"] is not None:
                local_ref_kl_scale += int((micro_batch["ref_kl_weights"] != 0).sum())
        global_scales = torch.tensor(
            [local_rl_scale, local_ce_scale, local_ref_kl_scale], dtype=torch.int64, device="cuda"
        )
        dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
        dist.all_reduce(global_scales, op=dist.ReduceOp.SUM, group=dp_cp_group)
        rl_scale, ce_scale, ref_kl_scale = (max(scale, 1) for scale in global_scales.tolist())

        logger.debug(f"Starting forward and backward pass ({batch_size=})")
        tensors = Tensors()  # Used to accumulate tensor statistics across micro-batches and ranks for logging
        cp_enabled = parallel_dims.cp_enabled
        cp_rank = parallel_dims.world_mesh["cp"].get_local_rank() if cp_enabled else 0
        cp_group = parallel_dims.world_mesh["cp"].get_group() if cp_enabled else None
        cp_size = parallel_dims.cp
        pp_enabled = parallel_dims.pp_enabled
        pp_side_inputs: list[dict] = []

        pp_target_length = 0
        if pp_enabled:
            # PipelineStage uses fixed-size inter-stage buffers, so every
            # micro-batch must be the same length. Agree on one length across the
            # WHOLE world (every stage builds the same buffers) and pad to it here,
            # before CP sharding, so each shard stays uniform and the pad tokens
            # land at the true end of the sequence.
            _local_max = max((mb["input_ids"].shape[1] for mb in micro_batches), default=0)
            _max_t = torch.tensor(_local_max, device="cuda")
            dist.all_reduce(_max_t, op=dist.ReduceOp.MAX)
            pp_target_length = int(_max_t.item())
            if cp_enabled and pp_target_length % cp_size:
                pp_target_length += cp_size - (pp_target_length % cp_size)

        for micro_step, micro_batch in enumerate(micro_batches):
            pad_micro_batch_to_distributed_max(micro_batch, dp_cp_group)
            if pp_enabled:
                pad_micro_batch_to_length(micro_batch, pp_target_length)
            _dev = "cpu" if pp_enabled else "cuda"
            input_ids = micro_batch["input_ids"].to(_dev)
            position_ids = micro_batch["position_ids"].to(_dev)
            advantages = micro_batch["advantages"].to(_dev)
            loss_mask = micro_batch["loss_mask"].to(_dev)
            inference_logprobs = micro_batch["inference_logprobs"].to(_dev)
            ref_logprobs = micro_batch["ref_logprobs"].to(_dev) if micro_batch["ref_logprobs"] is not None else None
            rl_weights = micro_batch["rl_weights"].to(_dev) if micro_batch["rl_weights"] is not None else None
            ce_weights = micro_batch["ce_weights"].to(_dev) if micro_batch["ce_weights"] is not None else None
            ref_kl_weights = (
                micro_batch["ref_kl_weights"].to(_dev) if micro_batch["ref_kl_weights"] is not None else None
            )
            routed_experts = (
                micro_batch["routed_experts"].to(_dev) if micro_batch["routed_experts"] is not None else None
            )
            if routed_experts is not None and not pp_enabled:
                # PP defers this cast to GPU-consumption time in _next_side.
                routed_experts = routed_experts.to(torch.int32)

            if routed_experts is None and config.enable_router_replay:
                raise ValueError(
                    "You must set `enable_return_routed_experts=True` in the inference config or pass `--enable-return-routed-experts` to vLLM server to use router replay."
                )

            if routed_experts is not None and not config.enable_router_replay:
                # we could've gotten routed experts from the inference server, but we didn't enable router replay
                routed_experts = None

            # Multimodal kwargs are an opaque per-model dict (e.g.
            # {"pixel_values": ..., "image_grid_thw": ...} for Qwen3-VL,
            # just {"pixel_values": ...} for Gemma3-VL) — we move every
            # tensor to CUDA and let the model's forward sort them.
            mm_kwargs_raw = micro_batch.get("mm_kwargs")
            mm_kwargs = {k: v.to("cuda") for k, v in mm_kwargs_raw.items()} if mm_kwargs_raw else None
            if mm_kwargs is not None and config.model.vlm is None:
                raise ValueError(
                    "Received multimodal samples but [model.vlm] is not set. "
                    "Set [model.vlm] to train on multimodal samples."
                )
            mm_token_type_ids = (
                micro_batch["mm_token_type_ids"].to("cuda")
                if micro_batch.get("mm_token_type_ids") is not None
                else None
            )

            seq_lens = micro_batch["seq_lens"].to("cuda")

            labels = shift_tensor_left(input_ids)
            logprob_mask = shift_tensor_left(loss_mask)
            if ce_weights is not None:
                logprob_mask |= shift_tensor_left(ce_weights != 0)
            if ref_kl_weights is not None:
                logprob_mask |= shift_tensor_left(ref_kl_weights != 0)

            seq_lens_are_pre_shard = False

            if cp_enabled:
                # MRoPE batches must merge image embeddings before sharding.
                defer_vlm_cp_to_model = mm_kwargs is not None and "image_grid_thw" in mm_kwargs
                if not defer_vlm_cp_to_model:
                    input_ids, position_ids = setup_cp_params(
                        input_ids,
                        position_ids,
                        cp_rank,
                        cp_size,
                        cp_group,
                        seq_lens=seq_lens,
                        cp_style=config.model.cp_style,
                    )
                seq_lens_are_pre_shard = True
                labels = shard_for_cp(labels, cp_rank=cp_rank, cp_world_size=cp_size)
                logprob_mask = shard_for_cp(logprob_mask, cp_rank=cp_rank, cp_world_size=cp_size)
                if routed_experts is not None and not defer_vlm_cp_to_model:
                    routed_experts = shard_for_cp(routed_experts, cp_rank=cp_rank, cp_world_size=cp_size)

            labels = labels.masked_fill(~logprob_mask, -100)

            if config.model.lora:
                lora_num_tokens = micro_batch["lora_num_tokens"].to("cuda")
                if cp_enabled:
                    chunk_size = labels.shape[1]
                    # Convert to cumsum, adjust for CP chunk, convert back to num_tokens
                    cu_offsets = lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
                    adjusted_cu = torch.clip(cu_offsets - chunk_size * cp_rank, min=0, max=chunk_size)
                    lora_num_tokens = torch.diff(
                        adjusted_cu, prepend=torch.tensor([0], device=adjusted_cu.device, dtype=adjusted_cu.dtype)
                    )
                set_lora_num_tokens(lora_num_tokens)

            temperatures = micro_batch["temperatures"].to(_dev)

            # Shard temperatures for context parallelism if enabled
            if cp_enabled:
                temperatures = shard_for_cp(temperatures, cp_rank=cp_rank, cp_world_size=cp_size)

            if pp_enabled:
                # Under pipeline parallelism the forward/backward is driven by
                # the schedule after this loop, not here. Stash everything the
                # stage will need, in micro-batch order.
                pp_side_inputs.append(
                    dict(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        labels=labels,
                        temperatures=temperatures,
                        seq_lens=seq_lens,
                        seq_lens_are_pre_shard=seq_lens_are_pre_shard,
                        routed_experts=routed_experts,
                        sequence_lengths=micro_batch["sequence_lengths"],
                        inference_logprobs=inference_logprobs,
                        ref_logprobs=ref_logprobs,
                        advantages=advantages,
                        loss_mask=loss_mask,
                        rl_weights=rl_weights,
                        ce_weights=ce_weights,
                        ref_kl_weights=ref_kl_weights,
                    )
                )
                continue

            # Forward pass with per-token temperatures
            with maybe_record_function("forward"), maybe_activation_offloading(config.model.ac_offloading):
                out = forward(
                    model,
                    input_ids,
                    position_ids,
                    labels=labels,
                    temperature=temperatures,
                    mm_kwargs=mm_kwargs,
                    mm_token_type_ids=mm_token_type_ids,
                    seq_lens=seq_lens,
                    seq_lens_are_pre_shard=seq_lens_are_pre_shard,
                    routed_experts=routed_experts,
                )

            if out.get("logprobs") is None:
                # VanillaOutputLinear was used - need to compute logprobs externally with per-token temps
                assert out.get("logits") is not None, "Logits must be provided to compute logprobs"
                logits = out["logits"]
                # Per-token temperature scaling: temperatures is [batch, seq], logits is [batch, seq, vocab]
                scaled_logits = logits / temperatures.unsqueeze(-1)
                out["logprobs"] = selective_log_softmax(scaled_logits, labels)
                out["entropy"] = compute_entropy(scaled_logits)
            # else: FusedOutputLinear was used - logprobs already computed with per-token temperatures

            if cp_enabled:
                out["logprobs"] = gather_for_cp(out["logprobs"], cp_group)
                out["entropy"] = gather_for_cp_wo_grad(out["entropy"], cp_size, cp_group)

            vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
            # This is not really necessary as the first token should be masked out, but we do it anyway to be sure
            out["logprobs"] = shift_tensor_right(
                out["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
            )
            out["entropy"] = shift_tensor_right(
                out["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
            )

            # Compute loss
            sequence_lengths = micro_batch["sequence_lengths"]
            loss, loss_tensors = compute_loss(
                trainer_logprobs=out["logprobs"].squeeze().split(sequence_lengths),
                inference_logprobs=inference_logprobs.squeeze().split(sequence_lengths),
                ref_logprobs=ref_logprobs.squeeze().split(sequence_lengths) if ref_logprobs is not None else None,
                advantages=advantages.squeeze().split(sequence_lengths),
                loss_mask=loss_mask.squeeze().split(sequence_lengths),
                rl_weights=rl_weights.squeeze().split(sequence_lengths) if rl_weights is not None else None,
                ce_weights=ce_weights.squeeze().split(sequence_lengths) if ce_weights is not None else None,
                ref_kl_weights=ref_kl_weights.squeeze().split(sequence_lengths) if ref_kl_weights is not None else None,
                rl_loss_fn=rl_loss_fn,
                rl_scale=rl_scale,
                ce_scale=ce_scale,
                ref_kl_scale=ref_kl_scale,
            )

            # Backward pass
            with maybe_record_function("backward"):
                loss.backward()

            # Add relevant tensors to tensor dict for logging purposes
            entropy = out["entropy"][loss_mask].detach().to("cpu")
            tensors["entropy/all"].append(entropy)
            tensors["loss"].append(loss.detach().to("cpu").unsqueeze(0))

            env_names = micro_batch["env_names"]
            masked_env_names = [env_name for env_name, keep in zip(env_names, loss_mask.flatten().tolist()) if keep]
            env_to_indices: dict[str, list[int]] = {}
            for idx, env_name in enumerate(masked_env_names):
                env_to_indices.setdefault(env_name, []).append(idx)

            for env_name, indices in env_to_indices.items():
                tensors[f"entropy/{env_name}"].append(entropy[indices])

            # Mismatch KL is only meaningful where sampling logprobs exist —
            # keep rl/ref_kl member tokens (policy-sampled), exclude tokens
            # whose action component is ce (frozen-model tokens).
            if rl_weights is None and ref_kl_weights is None:
                mismatch_mask = loss_mask
                has_mismatch_tokens = True
            else:
                sampled_mask = (rl_weights != 0) if rl_weights is not None else loss_mask
                if ref_kl_weights is not None:
                    sampled_mask = sampled_mask | (ref_kl_weights != 0)
                mismatch_mask = loss_mask & sampled_mask
                has_mismatch_tokens = bool(mismatch_mask.any())
            if has_mismatch_tokens:
                with torch.no_grad():
                    _, _, mismatch_kl = compute_importance_ratio_and_mismatch_kl(out["logprobs"], inference_logprobs)
                mismatch_kl = mismatch_kl[mismatch_mask].detach().to("cpu")
                tensors["mismatch_kl/all"].append(mismatch_kl)
                mismatch_env_names = [
                    env_name for env_name, keep in zip(env_names, mismatch_mask.flatten().tolist()) if keep
                ]
                mismatch_env_to_indices: dict[str, list[int]] = {}
                for idx, env_name in enumerate(mismatch_env_names):
                    mismatch_env_to_indices.setdefault(env_name, []).append(idx)
                for env_name, indices in mismatch_env_to_indices.items():
                    tensors[f"mismatch_kl/{env_name}"].append(mismatch_kl[indices])

            token_exporter.export(
                progress.step,
                micro_step,
                micro_batch,
                out,
                sequence_lengths,
                config.loss,
            )

            if is_tt_moe_model(model):
                load_balance_stats = get_load_balance_stats(model)
                for k, v in load_balance_stats.items():
                    if v is not None:
                        tensors[k].append(v)

            # Add loss tensors to tensor dict for logging purposes
            for key, loss_tensor in loss_tensors.items():
                tensors[key].append(loss_tensor.detach().to("cpu"))

            # Debug log with *local, micro step* stats
            micro_step_message = f"Micro Step {micro_step}/{len(micro_batches)} | Loss {tensors['loss'][-1].mean().item():.4f} | Entropy {tensors['entropy/all'][-1].mean().item():.4f}"
            if has_mismatch_tokens:
                micro_step_message += f" | Mismatch KL {tensors['mismatch_kl/all'][-1].mean().item():.4f}"
            if "max_vio" in tensors:
                micro_step_message += f" | Max Vio {tensors['max_vio'][-1].mean().item():.4f}"
            if "routing_confidence" in tensors:
                micro_step_message += f" | Routing Conf. {tensors['routing_confidence'][-1].mean().item():.4f}"
            logger.debug(micro_step_message)

        if pp_enabled:
            # Loss for the last stage. Mirrors the non-PP path: temperature-scaled
            # logprobs, CP gather, right-shift, then the RL loss. Returns a scalar
            # so the schedule only backprops one value.
            def _pp_loss(out, side):
                o = dict(out) if isinstance(out, dict) else {"logits": out}
                # The PP loss consumes head-computed logprobs directly. Without
                # the fused head this KeyErrors mid-schedule on the last stage
                # while 48 other ranks sit in P2P -- fail loudly instead.
                assert o.get("logprobs") is not None, (
                    "pipeline parallelism requires the fused output head "
                    "(logprobs computed in the head); got logits-only output"
                )
                if cp_enabled:
                    o["logprobs"] = gather_for_cp(o["logprobs"], cp_group)
                    o["entropy"] = gather_for_cp_wo_grad(o["entropy"], cp_size, cp_group)
                vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
                o["logprobs"] = shift_tensor_right(
                    o["logprobs"], pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item()
                )
                o["entropy"] = shift_tensor_right(
                    o["entropy"], pad_value=torch.log(torch.tensor(float(vocab_size))).item()
                )
                sl = side["sequence_lengths"]
                loss, loss_tensors = compute_loss(
                    trainer_logprobs=o["logprobs"].squeeze().split(sl),
                    inference_logprobs=side["inference_logprobs"].squeeze().split(sl),
                    ref_logprobs=side["ref_logprobs"].squeeze().split(sl) if side["ref_logprobs"] is not None else None,
                    advantages=side["advantages"].squeeze().split(sl),
                    loss_mask=side["loss_mask"].squeeze().split(sl),
                    rl_weights=side["rl_weights"].squeeze().split(sl) if side["rl_weights"] is not None else None,
                    ce_weights=side["ce_weights"].squeeze().split(sl) if side["ce_weights"] is not None else None,
                    ref_kl_weights=side["ref_kl_weights"].squeeze().split(sl)
                    if side["ref_kl_weights"] is not None
                    else None,
                    rl_loss_fn=rl_loss_fn,
                    rl_scale=rl_scale,
                    ce_scale=ce_scale,
                    ref_kl_scale=ref_kl_scale,
                )
                tensors["loss"].append(loss.detach().to("cpu").unsqueeze(0))
                tensors["entropy/all"].append(o["entropy"][side["loss_mask"]].detach().to("cpu"))
                for k, v in loss_tensors.items():
                    tensors[k].append(v.detach().to("cpu"))
                return loss

            # Stage buffers are fixed size, so every micro-batch must share one
            # sequence length; the pad is appended as a masked-out sequence.
            # PRIME_PP_MAX_MICRO truncates the micro-batch list so a debug step
            # takes about a minute instead of thirteen. Never set in production.
            _cap = int(os.environ.get("PRIME_PP_MAX_MICRO", "0") or 0)
            if _cap and len(pp_side_inputs) > _cap:
                pp_side_inputs = pp_side_inputs[:_cap]
                logger.warning(f"PRIME_PP_MAX_MICRO: using only {_cap} micro-batches (debug only)")
            _n = torch.tensor([len(pp_side_inputs), -len(pp_side_inputs)], device="cuda")
            dist.all_reduce(_n, op=dist.ReduceOp.MAX)
            if _n[0].item() != -_n[1].item() or _n[0].item() == 0:
                raise RuntimeError(
                    f"micro-batch count mismatch across ranks (max={_n[0].item()}, "
                    f"min={-_n[1].item()}); the pipeline schedule would deadlock"
                )
            pp_side_inputs = pad_pp_side_inputs(pp_side_inputs, cp_world_size=parallel_dims.cp)
            pp_stage_module.loss_closure = _pp_loss
            pp_stage_module.set_side_inputs(pp_side_inputs)
            # All micro-batches share one padded shape, so hand the stage its
            # exact tensors and skip the metadata-exchange rendezvous.
            _L = pp_side_inputs[0]["input_ids"].shape[1]
            _hs = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
            # The inter-stage buffers carry ACTIVATIONS, so their dtype is
            # optimization_dtype -- NOT the dtype of the first parameter. Those
            # coincided under bf16 weights, but with
            # `quantization.dequantize_checkpoint = false` the first parameter is
            # float8_e4m3fn, which declared every hidden-state buffer fp8 and
            # failed as
            #   PipeliningShapeError: Stage 1 forward outputs: value 0 has a
            #   dtype mismatch: expected torch.float8_e4m3fn actual torch.bfloat16
            # Scanning for the first "real float" parameter is no better: an fp32
            # weight_scale_inv can precede the bf16 weights and would declare the
            # buffers fp32. Read the configured value instead.
            _stage_dtype = DTYPE_MAP[config.model.optimization_dtype]
            _hidden_ex = torch.empty(1, _L, _hs, dtype=_stage_dtype, device="cuda")
            _is_last = pp_rank == parallel_dims.pp - 1
            pp_schedule = build_pp_schedule(
                pp_stage_module,
                pp_rank=pp_rank,
                pp_size=parallel_dims.pp,
                n_microbatches=len(pp_side_inputs),
                pp_group=parallel_dims.get_mesh("pp").get_group(),
                schedule_name=config.model.pp_schedule,
                example_input=torch.empty(1, _L, dtype=torch.int64, device="cuda") if pp_rank == 0 else _hidden_ex,
                example_output=torch.empty((), dtype=torch.float32, device="cuda") if _is_last else _hidden_ex,
            )
            # FSDP must be initialised before the schedule issues its first
            # prefetch, which happens before any forward on some stages.
            force_fsdp_lazy_init(model)
            # Side inputs are CPU-staged; the schedule input itself must be CUDA.
            pp_inputs = torch.cat([sd["input_ids"] for sd in pp_side_inputs], dim=0).to("cuda")
            if dist.get_rank() == 0:
                torch.cuda.synchronize()
                pp_start = time.perf_counter()
            # return_outputs=False: the schedule would otherwise concatenate the
            # last stage's per-micro-batch outputs, which are scalar losses --
            # "zero-dimensional tensor cannot be concatenated". The loss closure
            # already accumulates what the metrics need.
            if pp_rank == 0:
                pp_schedule.step(pp_inputs, return_outputs=False)
            elif pp_rank == parallel_dims.pp - 1:
                # The schedule indexes target_mbs unconditionally on the last
                # stage; the loss closure ignores the value, so a zero per
                # micro-batch is enough to satisfy it.
                pp_schedule.step(target=torch.zeros(len(pp_side_inputs), device="cuda"), return_outputs=False)
            else:
                pp_schedule.step(return_outputs=False)
            if dist.get_rank() == 0:
                torch.cuda.synchronize()
                logger.info(
                    f"PP step: {len(pp_side_inputs)} micro-batches in {time.perf_counter() - pp_start:.2f}s"
                )
            # Free this step's pipeline state NOW. The side inputs (including
            # routed_experts, tens of MB per micro-batch) and the stage's
            # per-micro-batch recv/grad buffers otherwise survive through the
            # optimizer step and overlap with the NEXT batch's allocations --
            # a multi-GB transient that ran the 47.4 GiB cards to the edge.
            pp_stage_module.set_side_inputs([])
            pp_stage_module.loss_closure = None
            del pp_schedule, pp_inputs, _hidden_ex, pp_side_inputs
            pp_side_inputs = []

            # F6: MoE router stat buffers accumulate via add_ every forward and
            # are only drained by this call; without it they grow for the whole
            # run under PP (the non-PP path drains them per micro-batch).
            if is_tt_moe_model(model):
                lb_stats = get_load_balance_stats(model)
                for k, v in lb_stats.items():
                    if v is not None:
                        tensors[k].append(v)

        if config.enable_token_export:
            dist.barrier()
            ready_run_ids = {
                multi_run_manager.idx_2_id[idx]
                for idx in multi_run_manager.ready_to_update_idxs
                if idx in multi_run_manager.idx_2_id
            }
            token_exporter.mark_stable(ready_run_ids)

        # compute_loss already divided by the global token count. Undo FSDP's per-rank averaging
        # across dp_cp so the final gradient is the true per-token mean over the global batch.
        found_nonfinite = torch.zeros((), device="cuda", dtype=torch.int32)
        nonfinite_grad_names: list[str] = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)
                grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
                is_nonfinite = (~torch.isfinite(grad).all()).to(dtype=torch.int32)
                found_nonfinite = torch.maximum(found_nonfinite, is_nonfinite)
                if is_nonfinite.item():
                    nonfinite_grad_names.append(name)
        if nonfinite_grad_names:
            logger.error(
                "Non-finite local gradients at step %s: %s",
                progress.step,
                nonfinite_grad_names,
            )

        # Do not silently replace NaN/Inf values with zero. That changes the
        # optimizer direction and made the later finite-norm guard ineffective.
        # All ranks must make the same decision before any rank enters the
        # collective gradient clipping path.
        dist.all_reduce(found_nonfinite, op=dist.ReduceOp.MAX)
        if found_nonfinite.item():
            raise RuntimeError(
                f"Non-finite gradient at step {progress.step}; refusing to update or broadcast."
            )

        # Optionally, clip the gradients
        grad_norm: torch.Tensor | None = None
        if config.optim.max_norm is not None:
            grad_norm = clip_grad_norm_(
                model.parameters(),
                max_norm=config.optim.max_norm,
                ep_enabled=parallel_dims.ep_enabled,
                # Each stage's DTensor meshes do not span pp, so without this the
                # norm is per-stage and clipping diverges from the non-PP runs.
                pp_mesh=parallel_dims.get_mesh("pp") if parallel_dims.pp_enabled else None,
            )
            if grad_norm.device.type == "cpu":
                grad_norm = grad_norm.to(torch.device("cuda"))
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {progress.step}; refusing to update or broadcast.")
            if dist.get_rank() == 0:
                logger.info(f"grad_norm full precision: {grad_norm.item():.6e}")

        if pp_enabled and dist.get_rank() % (dist.get_world_size() // parallel_dims.pp) == 0:
            # One line per stage master per step, full precision. A stage whose
            # backward silently detached shows l1=0 here; the SUCCESS line's
            # 4-decimal grad norm cannot distinguish that from a small gradient.
            import sys

            # Operate on LOCAL shards only. Reducing a DTensor here would issue
            # a collective on the stage mesh that the other 7 ranks of the stage
            # never join (only the stage master runs this block), deadlocking
            # the step -- a diagnostic must never alter collective ordering.
            from torch.distributed.tensor import DTensor as _DT

            _l1 = 0.0
            _n_with = 0
            for _p in model.parameters():
                if _p.grad is None:
                    continue
                _n_with += 1
                _g = _p.grad.to_local() if isinstance(_p.grad, _DT) else _p.grad
                _l1 += float(_g.abs().sum())
            print(
                f"[grad-check] step={progress.step} stage={pp_rank} "
                f"lora_grad_l1={_l1:.6e} params_with_grad={_n_with}",
                file=sys.stderr,
                flush=True,
            )

        zero_grad_ratio = get_zero_gradient_ratio(model.parameters(), parallel_dims.dp_replicate)

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        scheduler.step()

        if config.max_concurrent_runs == 1:
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.get_current_lr()
        forward_backward_time = time.perf_counter() - forward_backward_start_time

        # Broadcast the model just produced (policy v{progress.step}) so the orchestrator can
        # sample its next step from it. In-memory transports retain their two-step shutdown
        # window; filesystem broadcast still writes every version for resume.
        if weight_broadcast is None:
            broadcast_weights_time = 0
        else:
            broadcast_unused = (
                config.weight_broadcast.type in ("nccl", "nixl")
                and config.max_steps is not None
                and progress.step >= config.max_steps - 1
            )
            if not broadcast_unused:
                broadcast_weights_start_time = time.perf_counter()
                weight_broadcast.broadcast_weights(model, step=progress.step)
                broadcast_weights_time = time.perf_counter() - broadcast_weights_start_time
                # Clean up old broadcast directories (unless at ckpt interval if using filesystem weight broadcast)
                if config.weight_broadcast.type == "filesystem":
                    interval_to_keep = config.ckpt and config.ckpt.interval
                    weight_broadcast.maybe_clean(interval_to_keep)
            else:
                broadcast_weights_time = 0
                # Usually the broadcast will set this. If broadcast is skipped, we need to reset this here.
                for idx in multi_run_manager.used_idxs:
                    multi_run_manager.ready_to_update[idx] = False

        # Checkpoint the step we just finished (model = policy v{progress.step}).
        if config.max_concurrent_runs > 1:
            # Multi-run: save per-run checkpoints every step (interval-gated inside
            # MultiCheckpointManager); there is no after-loop final save for multi-run.
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(optimizer, scheduler)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time
            ckpt_manager.maybe_clean()
        elif (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            # the last step is written once after the loop (final ckpt), so skip it here
            and not is_last_step
            and progress.step % config.ckpt.interval == 0
        ):
            save_ckpt_time = 0

            if not config.ckpt.weights_only:
                # Single-run: Save full checkpoint
                logger.info(f"Saving checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
                save_ckpt_time += time.perf_counter() - save_ckpt_start_time

            ckpt_manager.maybe_clean()

            # Save weight checkpoint
            if weight_ckpt_manager is not None:
                logger.info(f"Saving weight checkpoint at step {progress.step}")
                save_ckpt_start_time = time.perf_counter()
                weight_ckpt_manager.save(progress.step, model, tokenizer)
                save_ckpt_time += time.perf_counter() - save_ckpt_start_time
                weight_ckpt_manager.maybe_clean()
        else:
            save_ckpt_time = 0

        # Optionally, dump memory snapshot
        if memory_profiler is not None:
            memory_profiler.step()

        # Synchronize the tensor metrics across all steps and ranks
        tensor_stats = tensors.compute_stats()

        # Compute step metrics
        # Under PP every micro-batch was padded up to pp_target_length after
        # seq_len was read; measure the padded length or throughput/MFU are
        # biased low and not comparable across runs.
        _eff_seq = pp_target_length if (pp_enabled and pp_target_length) else seq_len
        num_local_tokens = _eff_seq * batch_size
        num_tokens = parallel_dims.get_mesh("dp").size() * num_local_tokens
        progress.total_tokens += num_tokens
        progress.total_samples += batch_size
        perf_counter = get_perf_counter(model, seq_len)
        throughput = perf_counter.get_step_tokens_per_second(num_tokens, forward_backward_time)
        mfu = perf_counter.get_step_mfu(num_tokens, forward_backward_time)
        # Report the WORST rank, not rank 0. Under pipeline parallelism rank 0 is
        # stage 0, which owns the fewest layers; reporting it hid 12-layer stages
        # sitting at 46 of 47.4 GiB until they OOM'd.
        _peak = torch.tensor(torch.cuda.max_memory_reserved() / 1024**3, device="cuda")
        dist.all_reduce(_peak, op=dist.ReduceOp.MAX)
        peak_memory = _peak.item()  # GiB, max across ranks

        # Log step metrics
        step_time = time.perf_counter() - step_start_time
        step_message = f"Step {progress.step} | {format_time(step_time):>7} | Loss {tensor_stats['loss/mean']:.4f} | Entropy {tensor_stats['entropy/all/mean']:.4f}"
        if "mismatch_kl/all/mean" in tensor_stats:
            step_message += f" | Mismatch KL {tensor_stats['mismatch_kl/all/mean']:.4f}"
        if grad_norm is not None:
            step_message += f" | Grad. Norm {grad_norm:.4f}"
        step_message += f" | LR {current_lr:.2e} | Throughput {throughput:.0f} tokens/s | MFU {mfu:.1f}% | Peak Mem. {peak_memory:.1f} GiB"
        if "max_vio/mean" in tensor_stats:
            step_message += f" | Max Vio {tensor_stats['max_vio/mean']:.4f}"
        if "routing_confidence/mean" in tensor_stats:
            step_message += f" | Routing Conf. {tensor_stats['routing_confidence/mean']:.4f}"
        logger.success(step_message)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "perf/peak_memory": peak_memory,
            "step": progress.step,
        }
        monitor.log(perf_metrics, step=progress.step)

        # Log optimizer metrics
        optim_metrics = {
            "optim/lr": current_lr,
            "optim/zero_grad_ratio": zero_grad_ratio,
            "step": progress.step,
        }
        if grad_norm is not None:
            optim_metrics["optim/grad_norm"] = grad_norm.item()
        monitor.log(optim_metrics, step=progress.step)

        # Compute derived metrics
        entropy_mean = tensor_stats.get("entropy/all/mean", 0.0)
        mismatch_kl_mean = tensor_stats.get("mismatch_kl/all/mean")
        if mismatch_kl_mean is not None and entropy_mean > 0:
            tensor_stats["kl_ent_ratio/mean"] = mismatch_kl_mean / entropy_mean

        tensor_stats["step"] = progress.step
        monitor.log(filter_rl_trainer_tensor_stats_for_wandb(tensor_stats), step=progress.step)

        # Log time metrics
        time_metrics = {
            "time/step": step_time,
            "time/wait_for_batch": wait_for_batch_time,
            "time/load_data": load_data_time,
            "time/broadcast_weights": broadcast_weights_time,
            "time/save_ckpt": save_ckpt_time,
            "time/forward_backward": forward_backward_time,
            "step": progress.step,
        }
        monitor.log(time_metrics, step=progress.step)

        # Log disk metrics
        disk_metrics = get_ckpt_disk_metrics(config.output_dir)
        disk_metrics["step"] = progress.step
        monitor.log(disk_metrics, step=progress.step)

        # Update Prometheus metrics if configured
        if metrics_server is not None:
            metrics_server.update(
                step=progress.step,
                loss=tensor_stats["loss/mean"],
                throughput=throughput,
                grad_norm=grad_norm.item() if grad_norm is not None else None,
                peak_memory_gib=peak_memory,
                learning_rate=current_lr,
                mfu=mfu,
                entropy=tensor_stats.get("entropy/all/mean", 0.0),
                mismatch_kl=tensor_stats.get("mismatch_kl/all/mean", 0.0),
                zero_grad_ratio=zero_grad_ratio,
            )
            # Update run/LoRA metrics
            multi_run_manager = get_multi_run_manager()
            runs_discovered = len(list(config.output_dir.glob("run_*")))
            run_stats = []
            for idx in multi_run_manager.used_idxs:
                run_id = multi_run_manager.idx_2_id[idx]
                run_progress = multi_run_manager.progress[idx]
                if config.max_concurrent_runs == 1:
                    lr = optimizer.param_groups[0]["lr"]
                else:
                    lr = optimizer.get_current_lr(idx) if optimizer.optimizers[idx] else 0.0
                run_stats.append(
                    RunStats(
                        run_id=run_id,
                        step=run_progress.step,
                        total_tokens=run_progress.total_tokens,
                        learning_rate=lr,
                        ready=multi_run_manager.ready_to_update[idx],
                    )
                )
            metrics_server.update_runs(
                runs_discovered=runs_discovered,
                runs_max=multi_run_manager.max_runs,
                run_stats=run_stats,
            )

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

        if is_last_step:
            break
        progress.step += 1

    if config.trace_path:
        prof.__exit__(None, None, None)
        config.trace_path.mkdir(parents=True, exist_ok=True)
        trace_file = str(config.trace_path / f"trace_{dist.get_rank()}.json.gz")
        logger.info(f"Saving trace to {trace_file}")
        prof.export_chrome_trace(trace_file)
        logger.info(f"Saved trace to {trace_file}")

    token_exporter.close()

    # Write final checkpoint (only for single-run mode; multi-run checkpoints are managed by MultiCheckpointManager)
    if config.max_concurrent_runs == 1 and ckpt_manager is not None:
        if not (config.ckpt and config.ckpt.weights_only):
            logger.info("Writing final checkpoint")
            ckpt_manager.save(progress.step, model, [optimizer], scheduler, progress)
        ckpt_manager.maybe_clean()

    if config.max_concurrent_runs == 1 and weight_ckpt_manager is not None:
        logger.info("Writing final weight checkpoint")
        weight_ckpt_manager.save(progress.step, model, tokenizer)
        weight_ckpt_manager.maybe_clean()

    logger.info(f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB")
    logger.success("RL trainer finished!")

    # Stop metrics/health server if configured
    if metrics_server is not None:
        metrics_server.stop()
    if health_server is not None:
        health_server.stop()

    # Optionally, print benchmark table and export JSON
    if config.bench is not None and world.is_master:
        history = to_col_format(monitor.history)
        print_benchmark(history)
        if config.bench.output_json:
            export_benchmark_json(history, config.bench.output_json)
            logger.info(f"Benchmark results written to {config.bench.output_json}")


def main():
    """Main entry-point for RL trainer. Run using `uv run trainer`"""
    set_proc_title("Trainer")
    train(cli(TrainerConfig))


if __name__ == "__main__":
    main()
