"""Async-pipelined RL orchestrator.

``Orchestrator`` owns the shared state (policy, progress, ckpt, monitor)
and drives the pipeline. Components are single-purpose:

- ``RolloutDispatcher`` schedules rollouts; emits ``Rollout`` (train/eval
  discriminated by ``kind``) on its queue.
- ``TrainSink`` ingests train rollouts (tokenize → advantages → filters)
  and returns a ``TrainBatch`` when the threshold is met.
- ``EvalSink`` ingests eval rollouts and returns an ``EvalBatch`` (the full
  returned cohort) on epoch completion.
- ``TrainRollouts`` / ``EvalRollouts`` carry the rollouts and build the per-step W&B metrics
  (``batch.rollouts.metrics`` / ``.effective.metrics``).
- ``WeightWatcher`` advances ``Policy`` and notifies observers.
- ``PeriodicLogger`` polls the components on a shared interval for the
  ``_timestamp``-axis pipeline log.

Components don't reference the orchestrator. The orchestrator wires them
in ``setup()`` and drives them from ``main_loop()``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING
from urllib.request import urlopen

import tomli_w
import verifiers.v1 as vf
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

if TYPE_CHECKING:
    from renderers.base import Renderer
    from transformers.tokenization_utils import PreTrainedTokenizer

    from prime_rl.orchestrator.ckpt import CheckpointManager
    from prime_rl.transport.base import TrainingBatchSender
    from prime_rl.utils.client import InferencePool
    from prime_rl.utils.monitor.base import Monitor
import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.dispatcher import DispatcherMetrics, DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_sink import EvalSink
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.periodic_logger import PeriodicLogger
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    EvalBatch,
    Policy,
    Progress,
    Rollout,
    TrainBatch,
)
from prime_rl.orchestrator.utils import (
    get_weight_dir,
    intercept_vf_logging,
    save_rollouts,
    set_default_executor,
    setup_policy_inference_pool,
    trim_process_memory,
)
from prime_rl.orchestrator.watcher import (
    WeightWatcher,
    _adapter_file_records,
    _write_receivers_ready,
    inference_weight_path,
    validate_adapter_manifest,
)
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.transport import TrainingBatch, setup_training_batch_sender
from prime_rl.utils.async_utils import EventLoopLagMonitor, EventLoopLagStats, safe_cancel
from prime_rl.utils.client import (
    _delta_stage_timeout_s,
    init_nccl_broadcast,
    init_nixl_broadcast,
    stage_lora_adapter,
)
from prime_rl.utils.config import to_toml_dict
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import format_time, get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import (
    get_all_ckpt_steps,
    get_broadcast_dir,
    get_log_dir,
    get_rollout_dir,
    get_step_path,
    get_trace_path,
)
from prime_rl.utils.usage_reporter import UsageReporter
from prime_rl.utils.utils import (
    clean_exit,
    resolve_latest_ckpt_step,
)

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Wall-clock budget for post-training cleanup; force-exit if graceful
# shutdown wedges (env-server ZMQ recv, vLLM admin aclose, etc)
SHUTDOWN_TIMEOUT_S = 300

# Abort after this many consecutive train batches drop all rollouts to
# post-batch filters — usually a misconfigured filter or homogeneous-reward
# dataset; fail loudly instead of spinning
MAX_CONSECUTIVE_EMPTY_BATCHES = 10

# Keep complete GRPO groups available when the last active request is slow.
# Completed overflow carries into the next batch.
ROLLOUT_TAIL_SPARE_GROUPS = 32


def rollout_budget(
    *,
    batch_size: int,
    current: int,
    buffered: int,
    live: int,
    group_size: int,
    max_inflight: int,
    replica_count: int,
) -> int:
    group_size = max(1, group_size)
    spare = group_size * ROLLOUT_TAIL_SPARE_GROUPS
    budget = batch_size + spare - current - buffered - live
    if current < batch_size and current + buffered > 0:
        target_live = min(max_inflight, group_size * max(1, replica_count))
        refill = max(0, target_live - live)
        if refill:
            refill = ((refill + group_size - 1) // group_size) * group_size
            budget = max(budget, refill)
    return budget

# Let a valid old-policy batch finish before an adapter update.
POLICY_UPDATE_BATCH_WAIT_TIMEOUT_S = 1200

# Default wait for the trainer's startup weight broadcast when no ckpt block
# configures ``wait_for_weights_timeout`` (e.g. a from-scratch run). The
# broadcast is always coming, so wait rather than fail immediately.
STARTUP_WEIGHT_WAIT_TIMEOUT_S = 1200


def _verify_startup_adapter_delta(
    config: OrchestratorConfig,
    lora_name: str | None,
    step: int,
    target_path,
    target_url: str | None = None,
) -> None:
    """Reject a positive resume unless its published adjacent delta is exact."""
    if lora_name is None or step <= 0:
        return
    broadcast_dir = get_broadcast_dir(config.output_dir)
    previous_path = get_step_path(broadcast_dir, step - 1)
    previous_url = inference_weight_path(config, previous_path, lora_name)

    previous_previous_path = None
    previous_previous_url = None
    if step >= 2:
        previous_previous_path = get_step_path(broadcast_dir, step - 2)
        previous_previous_url = inference_weight_path(
            config, previous_previous_path, lora_name
        )
        if not isinstance(previous_previous_url, str):
            raise RuntimeError(
                f"cannot resume LoRA policy v{step}: second predecessor has no source URL"
            )
    def read_manifest(path, url: str | None) -> dict[str, object]:
        try:
            if url is None:
                with (path / "adapter_manifest.json").open("r", encoding="utf-8") as source:
                    value = json.load(source)
            else:
                with urlopen(f"{url.rstrip('/')}/adapter_manifest.json", timeout=_delta_stage_timeout_s()) as source:
                    value = json.load(source)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot resume LoRA policy v{step}: adapter manifest is unavailable"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"cannot resume LoRA policy v{step}: adapter manifest is not an object")
        return value

    manifest = read_manifest(target_path, target_url)
    try:
        validate_adapter_manifest(
            manifest,
            target_step=step,
            target_url=target_url,
            base_url=previous_url if isinstance(previous_url, str) else None,
            base2_url=previous_previous_url,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"cannot resume LoRA policy v{step}: {exc}"
        ) from exc
    if manifest.get("version") == 2:
        # Version-two manifests carry exact identities for both predecessor
        # adapters. The receiver checks the corresponding cached files.
        return

    if not (previous_path / "STABLE").is_file():
        raise RuntimeError(
            f"cannot resume LoRA policy v{step}: exact predecessor v{step - 1} is not STABLE"
        )
    previous_manifest = read_manifest(
        previous_path, previous_url if isinstance(previous_url, str) else None
    )
    if step == 1:
        valid_previous = (
            previous_manifest.get("version") in {1, 2}
            and previous_manifest.get("kind") == "lora-adapter"
            and previous_manifest.get("target_step") == 0
        )
    else:
        valid_previous = (
            previous_manifest.get("version") in {1, 2}
            and previous_manifest.get("kind") == "lora-adapter"
            and previous_manifest.get("mode") == "delta"
            and previous_manifest.get("target_step") == step - 1
            and (
                (
                    previous_manifest.get("version") == 1
                    and previous_manifest.get("transport") == "chunked-shuffled-xor-zstd"
                    and previous_manifest.get("base_step") == step - 2
                )
                or (
                    previous_manifest.get("version") == 2
                    and previous_manifest.get("transport") == "chunked-bf16-linear-zigzag-sparse-zstd"
                    and previous_manifest.get("base_step") == step - 2
                    and previous_manifest.get("base2_step") == step - 3
                )
            )
        )
    if not valid_previous:
        raise RuntimeError(
            f"cannot resume LoRA policy v{step}: predecessor manifest is not policy v{step - 1}"
        )
    try:
        target_files = _adapter_file_records(manifest)
        previous_files = _adapter_file_records(previous_manifest)
    except RuntimeError as exc:
        raise RuntimeError(f"cannot resume LoRA policy v{step}: {exc}") from exc
    target_weights = target_files["adapter_model.safetensors"]
    target_config = target_files["adapter_config.json"]
    previous_weights = previous_files["adapter_model.safetensors"]
    previous_config = previous_files["adapter_config.json"]
    if (
        manifest.get("target_size") != target_weights.get("size")
        or manifest.get("target_sha256") != target_weights.get("sha256")
        or manifest.get("config_size") != target_config.get("size")
        or manifest.get("config_sha256") != target_config.get("sha256")
        or manifest.get("base_size") != previous_weights.get("size")
        or manifest.get("base_sha256") != previous_weights.get("sha256")
        or manifest.get("base_config_sha256") != previous_config.get("sha256")
    ):
        raise RuntimeError(
            f"cannot resume LoRA policy v{step}: adjacent adapter base or SHA-256 identity differs"
        )


class Orchestrator:
    # Set in ``__init__``
    config: OrchestratorConfig
    progress: Progress
    policy: Policy
    stopped: asyncio.Event
    draining: bool
    last_batch_at: float | None
    consecutive_empty_batches: int
    eval_triggered_at: dict[tuple[str, int], float]
    ckpt_manager: CheckpointManager | None
    component_tasks: list[asyncio.Task]

    # Always set by ``setup()``
    tokenizer: PreTrainedTokenizer
    policy_inference: InferencePool
    monitor: Monitor
    sender: TrainingBatchSender
    train_envs: TrainEnvs
    train_source: TrainSource
    train_sink: TrainSink
    dispatcher: RolloutDispatcher
    watcher: WeightWatcher
    lag_monitor: EventLoopLagMonitor
    periodic_logger: PeriodicLogger

    # Set by ``setup()`` only when relevant config is present
    renderer: Renderer | None
    mm_token_type_ids_mapping: dict[int, int] | None
    heart: Heartbeat | None
    usage_reporter: UsageReporter | None
    inference_metrics: InferenceMetricsCollector | None
    eval_envs: EvalEnvs | None
    eval_sink: EvalSink | None
    eval_source: EvalSource | None
    lora_name: str | None
    resume_step: int | None
    resume_policy_step: int | None
    lag_task: asyncio.Task | None

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        setup_logger(config.log.level, json_logging=config.log.json_logging)
        # Route the in-process v1 library logging through our handler. The
        # env server runs in a child process, so its logging is separate.
        intercept_vf_logging(logger="verifiers.v1", level="WARN")
        algorithms = sorted({env.algo.type for env in config.train.env if env.algo is not None})
        get_logger().info(f"Starting orchestrator (algorithm: {', '.join(algorithms)})")

        if config.bench:
            get_logger().warning(f"Running in benchmark mode (max_steps={config.max_steps})")

        self.progress = Progress()
        self.ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
        self.policy = Policy(version=0, model_name="")
        self.stopped = asyncio.Event()
        # True after the final train step ships — pipeline winds down without
        # scheduling new train rollouts
        self.draining = False
        # Previous ``TrainBatch`` arrival timestamp; reset every ship so
        # ``step_time`` in the success log is real sink-to-sink cycle time
        self.last_batch_at = None
        # Trigger timestamps so eval success logs can report epoch duration
        self.eval_triggered_at = {}
        self.consecutive_empty_batches = 0
        self.dispatch_paused_at = None
        # Pulsed by the version hooks so a held ship can re-check ``policy.version``
        self.version_advanced = asyncio.Event()
        # Pulsed after a train batch advances ``progress.step``. The watcher
        # uses it for a bounded policy update at the train-batch boundary.
        self.batch_completed = asyncio.Event()
        self.wait_for_policy_time = 0.0
        self.component_tasks = []

        # Optional attributes — ``setup()`` populates them when the relevant
        # config is present
        self.renderer = None
        self.mm_token_type_ids_mapping = None
        self.heart = None
        self.usage_reporter = None
        self.inference_metrics = None
        self.eval_envs = None
        self.eval_sink = None
        self.eval_source = None
        self.lora_name = None
        self.resume_step = None
        self.resume_policy_step = None
        self.lag_task = None
        self.model_express = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def setup(self) -> None:
        """Install envs, load models/pools, resume from checkpoint, and
        construct the pipeline components."""
        config = self.config
        set_default_executor()

        # Persist the resolved config alongside the run
        config_dir = config.output_dir / "control"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "orch.toml", "wb") as f:
            tomli_w.dump(to_toml_dict(config), f)

        get_logger().info(f"Initializing tokenizer ({config.tokenizer})")
        self.tokenizer = setup_tokenizer(config.tokenizer)

        # The one model prime-rl hosts: the live policy. Frozen model
        # references are external endpoints — each env's Algorithm builds its
        # own pools in ``setup()`` below.
        get_logger().info(
            f"Initializing policy inference pool (base_url={', '.join(config.model.client.base_url)}, "
            f"model={config.model.name})"
        )
        self.renderer, self.policy_inference = await setup_policy_inference_pool(
            config=config, tokenizer=self.tokenizer
        )
        self.mm_token_type_ids_mapping = (
            getattr(self.renderer, "mm_token_type_id_map", None) if self.renderer is not None else None
        )
        if self.mm_token_type_ids_mapping == {}:
            self.mm_token_type_ids_mapping = None

        get_logger().info(
            f"Initializing monitor (wandb={config.wandb}, swanlab={config.swanlab}, "
            f"prime_monitor={config.prime_monitor})"
        )
        self.monitor = setup_monitor(
            wandb_config=config.wandb,
            swanlab_config=config.swanlab,
            prime_config=config.prime_monitor,
            file_config=config.file_monitor,
            output_dir=config.output_dir,
            tokenizer=self.tokenizer,
            run_config=config,
            keep_full_history=config.bench,
            train_env_names=[env.resolved_name for env in config.train.env],
            eval_env_names=[env.resolved_name for env in config.eval.env] if config.eval is not None else [],
        )

        if config.heartbeat is not None:
            self.heart = Heartbeat(config.heartbeat.url)

        usage_base_url = os.environ.get("PI_USAGE_BASE_URL")
        usage_api_key = os.environ.get("PI_USAGE_API_KEY")
        if usage_base_url and usage_api_key:
            self.usage_reporter = UsageReporter()

        # Filters apply to train rollouts only
        pre_filters = setup_filters(config.pre_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="pre-batch")
        post_filters = setup_filters(config.post_batch_filters, vocab_size=self.tokenizer.vocab_size, kind="post-batch")

        get_logger().info("Loading training environments")
        self.train_envs = TrainEnvs(
            config.train.env, policy_pool=self.policy_inference, renderer_config=config.renderer
        )
        get_logger().debug(
            f"Loaded {len(self.train_envs)} training environment(s) ({', '.join(self.train_envs.names)})"
        )
        await self.train_envs.start(
            log_dir=get_log_dir(config.output_dir.parent) / "envs" / "train",
            log_level=config.log.vf_level,
            json_logging=config.log.json_logging,
        )
        get_logger().success("Train environment(s) ready")

        if config.eval is not None:
            get_logger().info("Loading eval environment(s)")
            self.eval_envs = EvalEnvs(config.eval.env)
            get_logger().debug(f"Loaded {len(self.eval_envs)} eval environment(s) ({', '.join(self.eval_envs.names)})")
            await self.eval_envs.start(
                log_dir=get_log_dir(config.output_dir.parent) / "envs" / "eval",
                log_level=config.log.vf_level,
                json_logging=config.log.json_logging,
            )
            get_logger().success("Eval environment(s) ready")

        if config.ckpt is not None and config.ckpt.resume_step is not None and self.ckpt_manager is not None:
            if config.ckpt.resume_step == -1 and config.ckpt.skip_progress:
                self.resume_step = max(
                    (
                        step
                        for step in get_all_ckpt_steps(get_rollout_dir(config.output_dir))
                        if get_trace_path(config.output_dir, step, "train", "effective").exists()
                    ),
                    default=None,
                )
                self.resume_policy_step = max(
                    (
                        step
                        for step in get_all_ckpt_steps(get_broadcast_dir(config.output_dir))
                        if (get_step_path(get_broadcast_dir(config.output_dir), step) / "STABLE").exists()
                    ),
                    default=None,
                )
            elif config.ckpt.resume_step == -1:
                self.resume_step = resolve_latest_ckpt_step(self.ckpt_manager.ckpt_dir)
            else:
                self.resume_step = config.ckpt.resume_step
            if self.resume_policy_step is None and not (
                config.ckpt.resume_step == -1 and config.ckpt.skip_progress
            ):
                self.resume_policy_step = self.resume_step

        # Resume below may bump ``policy.version`` and the LoRA model name
        self.policy.model_name = self.policy_inference.model_name

        get_logger().info("Waiting for policy inference pool to be ready")
        await self.policy_inference.wait_for_ready(config.model.name)
        get_logger().success("Policy inference pool ready")
        # Build + ready pools for each env's frozen sampling source and the
        # algorithm's frozen reference model
        await asyncio.gather(
            *(env.sampler.setup() for env in self.train_envs),
            *(env.algorithm.setup() for env in self.train_envs),
        )

        if config.wandb is not None and config.collect_inference_metrics:
            self.inference_metrics = InferenceMetricsCollector(
                self.policy_inference.admin_clients,
                roles=config.inference_metrics_roles,
            )
            await self.inference_metrics.start()

        get_logger().info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                self.policy_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                inference_world_size=config.weight_broadcast.inference_world_size,
                quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
            )
        elif config.weight_broadcast.type == "nixl":
            await init_nixl_broadcast(
                self.policy_inference.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                config.weight_broadcast.inference_world_size,
                config.weight_broadcast.session_id,
            )
            self.model_express = ModelExpressSession(
                client=MxClient(server_url=f"{config.weight_broadcast.host}:{config.weight_broadcast.port}"),
                role="orchestrator",
                rank=0,
                session_id=config.weight_broadcast.session_id,
                worker_id="orchestrator",
            )
            self.model_express.publish()
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_INITIALIZING)

        get_logger().info(f"Initializing training batch sender ({config.rollout_transport})")
        self.sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

        self.lora_name = config.model.lora.name if config.model.lora else None

        if self.resume_step is not None and self.ckpt_manager is not None:
            if not config.ckpt.skip_progress:
                self.ckpt_manager.load(self.progress, step=self.resume_step)
            # The checkpoint finished step ``resume_step``; resume at the next step. Derive the step
            # from ``resume_step`` (not the loaded progress.step) so it stays coordinated with the
            # trainer even when ``ckpt.skip_progress`` left the counter unrestored.
            self.progress.step = self.resume_step + 1
            get_logger().info(
                f"Resuming orchestrator at rollout batch {self.progress.step} "
                f"from policy v{self.resume_policy_step or 0}"
            )
        else:
            get_logger().info("Training from scratch")

        # Sync inference to the incoming policy before the first step when resuming or when using
        # an in-memory transport, which rendezvouses with the trainer's startup broadcast.
        if self.resume_policy_step is not None or config.weight_broadcast.type in ("nccl", "nixl"):
            sync_version = self.resume_policy_step if self.resume_policy_step is not None else 0
            if config.weight_broadcast.type == "nixl":
                weights_path = None
                local_weights_path = None
            else:
                check_exists = config.weight_broadcast.type == "filesystem"
                # Without a ckpt block, fall back to a default timeout instead of not waiting at all.
                wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else STARTUP_WEIGHT_WAIT_TIMEOUT_S
                weights_path = get_weight_dir(
                    config.output_dir, sync_version, check_exists=check_exists, wait_timeout=wait_timeout
                )
                local_weights_path = weights_path
                source_weights_path = inference_weight_path(config, weights_path, self.lora_name)
                _verify_startup_adapter_delta(
                    config,
                    self.lora_name,
                    sync_version,
                    weights_path,
                    source_weights_path if isinstance(source_weights_path, str) else None,
                )
                weights_path = source_weights_path
            if self.model_express is not None:
                await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_READY)
            require_delta = self.lora_name is not None and sync_version > 0
            prefetched = False
            adapter_identity = None
            if self.lora_name is not None and isinstance(weights_path, str):
                stage_result = await stage_lora_adapter(
                    self.policy_inference.admin_clients,
                    weights_path,
                    require_delta=require_delta,
                    origin_base_urls=self.policy_inference.adapter_origin_base_urls,
                )
                adapter_identity = stage_result["adapter_identity"]
                prefetched = True
            receipts = await self.policy_inference.update_weights(
                weights_path,
                lora_name=self.lora_name,
                step=sync_version,
                require_delta=require_delta,
                prefetched=prefetched,
                expected_adapter_identity=adapter_identity,
            )
            if self.lora_name is not None and isinstance(weights_path, str):
                assert local_weights_path is not None
                _write_receivers_ready(
                    local_weights_path,
                    run_name=config.output_dir.name,
                    step=sync_version,
                    source_url=weights_path,
                    slot=self.lora_name,
                    adapter_identity=adapter_identity,
                    admin_clients=self.policy_inference.admin_clients,
                    receipts=receipts,
                )
            if self.model_express is not None:
                await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_INITIALIZING)
                # Complete the startup rendezvous before the watcher begins its next cycle.
                await asyncio.to_thread(
                    self.model_express.wait_for,
                    "trainer",
                    count=1,
                    status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                    timeout=config.weight_broadcast.timeout,
                )
            if self.lora_name is not None:
                self.policy_inference.update_model_name(self.lora_name)
                self.policy.model_name = self.lora_name
            self.policy.version = sync_version
            # Do not start rollout scheduling until every inference endpoint
            # exposes the exact policy that was loaded.
            await self.policy_inference.wait_for_ready(self.policy.model_name)

        self.train_source = TrainSource(self.train_envs, seed=42)
        self.eval_source: EvalSource | None = (
            EvalSource(
                self.eval_envs,
                config.eval,
                is_resumed=self.resume_step is not None,
            )
            if config.eval is not None and self.eval_envs is not None
            else None
        )

        assert config.max_inflight_rollouts is not None, "max_inflight_rollouts must be resolved before dispatcher init"
        log_interval = config.log.interval
        wandb_enabled = config.wandb is not None
        self.dispatcher = RolloutDispatcher(
            train_envs=self.train_envs,
            eval_envs=self.eval_envs,
            train_source=self.train_source,
            eval_source=self.eval_source,
            policy_pool=self.policy_inference,
            policy=self.policy,
            max_inflight_rollouts=config.max_inflight_rollouts,
            tasks_per_minute=config.tasks_per_minute,
            max_off_policy_steps=config.max_off_policy_steps,
        )
        self.train_sink = TrainSink(
            config,
            tokenizer=self.tokenizer,
            train_envs=self.train_envs,
            mm_token_type_ids_mapping=self.mm_token_type_ids_mapping,
            batch_size=config.batch_size,
            token_batch_size=config.token_batch_size,
            training_policy_version=self.progress.step - 1,
            pre_filters=pre_filters,
            post_filters=post_filters,
        )
        self.eval_sink = EvalSink(eval_envs=self.eval_envs) if self.eval_envs is not None else None
        if config.batch_size is not None:
            self.dispatcher.set_fresh_train_rollout_budget(config.batch_size)
        self.watcher = WeightWatcher(
            config,
            policy=self.policy,
            inference=self.policy_inference,
            # The orchestrator gets the boundary hook first. It leaves fresh
            # scheduling enabled while the current batch finishes. The
            # dispatcher then drains only the small overflow at the boundary
            # before the adapter load.
            observers=[self, self.dispatcher],
            lora_name=self.lora_name,
            ckpt_step=self.policy.version,
            model_express=self.model_express,
        )
        # Single periodic logger for the whole pipeline. It's the only
        # consumer of ``dispatcher.metrics.drained()`` (which clears on read)
        self.lag_monitor = EventLoopLagMonitor()
        self.periodic_logger = PeriodicLogger(
            name="Pipeline",
            collect=self.collect_pipeline_view,
            metric_keys=[
                *list(self.dispatcher.gauges().keys()),
                *DispatcherMetrics.drain_keys(
                    train_envs={e.name for e in self.train_envs},
                    eval_envs={e.name for e in self.eval_envs} if self.eval_envs is not None else set(),
                ),
                *list(self.watcher.gauges().keys()),
                "event_loop_lag/min",
                "event_loop_lag/mean",
                "event_loop_lag/median",
                "event_loop_lag/p90",
                "event_loop_lag/p99",
                "event_loop_lag/max",
                "event_loop_lag/n",
            ],
            interval=log_interval,
            wandb_enabled=wandb_enabled,
        )

    async def start(self) -> None:
        """Run the orchestrator until shutdown. Drives setup, spawns the
        background tasks, runs the main loop in this task, then cleans up."""
        await self.setup()
        self.update_dispatch_permission()
        config = self.config
        get_logger().info(f"Starting orchestrator loop (max_steps={config.max_steps or 'infinite'})")
        start_time = time.perf_counter()

        # Spawn background loops (dispatcher schedules, watcher polls). The
        # pipeline ``main_loop`` runs inline in this task; the single
        # ``PeriodicLogger`` polls dispatcher / watcher / sinks / lag
        # monitor each ``log.interval`` seconds for the pipeline-view log
        self.lag_task = asyncio.create_task(self.lag_monitor.run(), name="event_loop_lag")
        await self.periodic_logger.start()
        # On resume, an immutable trainer checkpoint can be newer than the
        # policy currently loaded by the engines. Close scheduling before the
        # dispatcher task exists, so it cannot fill 128 requests in the race
        # before WeightWatcher.on_version_pending starts the drain.
        if self.watcher.compute_next_ckpt_step() > self.watcher.ckpt_step:
            self.dispatcher.weight_update_pending = True
        self.component_tasks = [
            asyncio.create_task(self.dispatcher.start(), name="dispatcher"),
            asyncio.create_task(self.watcher.start(), name="watcher"),
        ]

        # Base-model eval (policy v0) — fires before any train rollouts, logged at the first
        # step, unless ``eval.skip_first_step=True`` (or this is a resume)
        self.maybe_trigger_eval(self.progress.step)

        # Anchor step-time clock so the first step measures startup → first batch
        self.last_batch_at = time.perf_counter()

        # ``clean_exit`` stays False if ``main_loop`` raises (signal-driven
        # CancelledError, KeyboardInterrupt, or a real error), so the teardown
        # logs a forced-cleanup warning instead of a clean-exit success.
        clean_exit = False
        try:
            await self.main_loop()
            clean_exit = True
        finally:
            elapsed = format_time(time.perf_counter() - start_time)
            if clean_exit:
                get_logger().success(f"Orchestrator step loop done in {elapsed}")
            else:
                get_logger().warning(f"Orchestrator interrupted after {elapsed} — forcing cleanup (not a clean exit)")
            self.monitor.save_final_summary()
            # ``progress.step`` points at the next (unshipped) step; the last finished step is
            # ``progress.step - 1``. Checkpoint it as ``step_{progress.step - 1}`` (no-op before the
            # first ship).
            if self.ckpt_manager is not None and self.progress.step > 1:
                self.progress.step -= 1
                get_logger().info("Writing final checkpoint")
                self.ckpt_manager.save(self.progress, step=self.progress.step)
            await self.stop()
            if clean_exit:
                get_logger().success("Orchestrator finished.")
            else:
                get_logger().warning("Orchestrator cleanup complete (forced).")
            trim_process_memory()

    async def main_loop(self) -> None:
        """Consume episodes (``list[Rollout]``) from the dispatcher and route them
        to the train / eval sink. Both sinks return a finalized batch (or
        ``None``) from ``add()``; we just dispatch on the result."""
        while not self.stopped.is_set():
            for task in self.component_tasks:
                if not task.done():
                    continue
                if task.cancelled():
                    raise RuntimeError(f"{task.get_name()} stopped unexpectedly")
                error = task.exception()
                if error is not None:
                    raise RuntimeError(f"{task.get_name()} failed") from error
                raise RuntimeError(f"{task.get_name()} exited unexpectedly")

            if self.draining and self.dispatcher.is_idle:
                get_logger().info("Pipeline drained, exiting main loop")
                self.stopped.set()
                break

            try:
                episode: list[Rollout] = await asyncio.wait_for(self.dispatcher.out_q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            # Every completed rollout — errored, filtered, or never batched — lands in the
            # ``all`` trace file the moment it arrives, so it survives crashes and drains.
            # Train rollouts belong to the batch window currently collecting (``progress.step``),
            # eval rollouts to the step whose eval triggered them.
            kind = episode[0].kind
            if self.dispatcher.is_abandoned_group(episode[0].group_id):
                self.dispatcher.mark_episode_consumed(kind)
                continue
            step = episode[0].eval_step if kind == "eval" else self.progress.step
            assert step is not None
            run: vf.RunInfo = (
                vf.EvalRunInfo(id=self.monitor.run_id, step=step)
                if kind == "eval"
                else vf.TrainRunInfo(id=self.monitor.run_id, step=step)
            )
            for rollout in episode:
                rollout.stamp(
                    run,
                    env_name=rollout.env_name,
                    group_id=str(rollout.group_id),
                    episode_id=rollout.episode_id,
                    policy_version=rollout.policy_version,
                )
            await asyncio.to_thread(
                save_rollouts,
                [rollout.to_record() for rollout in episode],
                get_trace_path(self.config.output_dir, step, kind, "all"),
            )

            if kind == "eval":
                assert self.eval_sink is not None  # eval rollouts only emitted when eval is configured
                eval_batch = self.eval_sink.add(episode)
                if eval_batch is not None:
                    await self.finalize_eval_batch(eval_batch)
                continue

            train_batch = await self.train_sink.add(episode)
            self.dispatcher.mark_episode_consumed(kind)
            if train_batch is None:
                self.update_rollout_budget()
            # In drain mode any late-arriving train batch is dropped — we
            # don't want to ship past ``max_steps``
            if train_batch is not None and not self.draining and not self.stopped.is_set():
                self.dispatcher.set_fresh_train_rollout_budget(0)
                await self.finalize_train_batch(train_batch)
                self.update_rollout_budget()

    async def finalize_train_batch(self, batch: TrainBatch) -> None:
        """Ship one ``TrainBatch`` out to the trainer and handle the I/O
        side-effects (ckpt, save_rollouts, reference scoring, sender.send,
        metrics, heartbeat, progress, eval trigger). The sink has already
        done all data-transformation work."""
        config = self.config
        step = self.progress.step

        # Sink-to-sink cycle time — the actual time between batches, not
        # including the orchestrator's ship I/O (overlapped with the
        # dispatcher producing the next batch)
        now = time.perf_counter()
        step_time = (now - self.last_batch_at) if self.last_batch_at is not None else 0.0
        self.last_batch_at = now

        if config.max_steps is not None and step > config.max_steps:
            self.draining = True
            self.dispatcher.disable_train_scheduling()
            n_cancelled = await self.dispatcher.cancel_inflight_train_rollouts()
            get_logger().info(
                f"Draining pipeline (cancelled {n_cancelled} in-flight train rollout(s); "
                f"any in-flight evals will complete)"
            )
            return

        if not batch.samples:
            self.consecutive_empty_batches += 1
            get_logger().warning(
                f"Step {step}: empty train batch (0 of {len(batch.rollouts)} generated rollouts shipped — "
                f"all errored or filtered out) "
                f"(consecutive empty batches: {self.consecutive_empty_batches}/{MAX_CONSECUTIVE_EMPTY_BATCHES})"
            )
            if self.consecutive_empty_batches >= MAX_CONSECUTIVE_EMPTY_BATCHES:
                raise RuntimeError(
                    f"{self.consecutive_empty_batches} consecutive empty train batches — "
                    "check filter config (pre_batch_filters / post_batch_filters) or task difficulty."
                )
            return
        self.consecutive_empty_batches = 0
        n_trainable = sum(1 for r in batch.rollouts if r.is_trainable)
        if n_trainable / len(batch.rollouts) <= 0.1:
            get_logger().warning(
                f"Only {n_trainable}/{len(batch.rollouts)} generated rollouts are trainable "
                f"({n_trainable / len(batch.rollouts):.1%}) — consider reviewing task difficulty / filter config"
            )

        # Ship batch ``step`` only once the trainer has published a policy within the configured lag.
        # Without this, fast envs fill batches from buffered rollouts, the
        # orchestrator finishes early, and its teardown strands the trainer inside
        # an in-memory broadcast handshake that needs the live weight watcher.
        # The final in-memory broadcasts are handled below. Bench runs have no
        # trainer, so they ship freely.
        required_version = step - 1 - config.max_off_policy_steps
        if not config.bench and self.policy.version < required_version:
            get_logger().info(
                f"Holding batch {step} until the trainer publishes policy v{required_version} "
                f"(currently v{self.policy.version})"
            )
            hold_start = time.perf_counter()
            while True:
                self.version_advanced.clear()
                if self.policy.version >= required_version:
                    break
                await self.version_advanced.wait()
            self.wait_for_policy_time += time.perf_counter() - hold_start

        # Stamp each rollout's true staleness: batch ``step`` trains on policy
        # v{step-1}, so a rollout generated from v{k} is (step-1)-k versions
        # off-policy — queue time included, unlike the dispatcher's in-flight
        # counter, which only sees weight updates during generation. Frozen-
        # sourced rollouts stay 0 (their sampler doesn't follow the policy).
        for r in batch.rollouts:
            if self.train_envs.get(r.env_name).sampler.samples_from_live_policy:
                r.off_policy_steps = (step - 1) - r.policy_version

        # The effective (clean, trained-on) subset lands in the per-step ``effective`` trace file
        # at ship time; the full arrival window already streamed into ``all`` on arrival.
        # to_record drops the per-node training tensors — they're for training, not the rollout
        # record, and can't round-trip json (raw numpy bytes).
        effective = batch.rollouts.effective
        records = [r.to_record() for r in effective]
        await asyncio.to_thread(save_rollouts, records, get_trace_path(config.output_dir, step, "train", "effective"))

        await self.sender.send(TrainingBatch(examples=batch.samples, step=step))
        self.progress.step += 1
        self.batch_completed.set()
        self.train_sink.advance_training_policy_version(self.progress.step - 1)
        self.update_dispatch_permission()
        # Checkpoint the step we just shipped (resume point: continue at step + 1).
        save_ckpt_time = await self.maybe_save_ckpt(step)
        trim_process_memory()

        # Rollout metrics over the {agg,<env>} × {all,effective} matrix. ``batch.rollouts`` is the
        # full arrival window (errored + filtered included); ``.effective`` is the clean subset.
        metrics: dict[str, float] = {}
        for subset, pool in (("all", batch.rollouts), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix="train/agg", subset=subset)
            for env_name, env_pool in pool.by_env().items():
                metrics |= env_pool.metrics.to_wandb(prefix=f"train/{env_name}", subset=subset)

        # Progress / timing / env-share / pre-filter accounting (assembled here, not in the metrics
        # objects). ``num_tokens`` is over the full arrival window; the input/output breakdown is over
        # the effective (shipped) subset, summing the same ``vf.Trace`` token properties the metric
        # matrix reports.
        num_tokens = sum(r.num_total_tokens for r in batch.rollouts)
        num_input = sum(r.num_input_tokens for r in effective)
        num_output = sum(r.num_output_tokens for r in effective)
        num_rollouts = len(batch.rollouts)
        num_unique_examples = len({r.group_id for r in batch.rollouts})
        metrics |= {
            "progress/tokens": num_tokens,
            "progress/input_tokens": num_input,
            "progress/output_tokens": num_output,
            "progress/rollouts": num_rollouts,
            "progress/tasks": num_unique_examples,
            "progress/total_tokens": self.progress.total_tokens,
            "progress/total_rollouts": self.progress.total_samples,
            "progress/total_tasks": self.progress.total_problems,
            "time/step": step_time,
            "time/save_ckpt": save_ckpt_time,
            "time/wait_for_policy": self.wait_for_policy_time,
            "step": step,
        }
        for env_name, env_pool in batch.rollouts.by_env().items():
            metrics[f"batch/{env_name}"] = len(env_pool) / len(batch.rollouts)
        if self.train_sink.pre_filter_seen > 0:
            metrics["pre_filters/all/dropped_rate"] = (
                self.train_sink.pre_filter_dropped / self.train_sink.pre_filter_seen
            )
            for name, count in self.train_sink.pre_filter_dropped_by_name.items():
                metrics[f"pre_filters/all/{name}/rate"] = count / self.train_sink.pre_filter_seen
        self.monitor.log(metrics, step=step)
        self.wait_for_policy_time = 0.0
        self.monitor.log_samples(effective.rollouts, step=step)
        self.monitor.log_distributions(
            distributions={
                "rewards": [r.reward for r in effective],
                "advantages": [a for r in effective if (a := r.scalar_advantage()) is not None],
            },
            step=step,
        )

        if self.usage_reporter is not None:
            run_id = os.getenv("RUN_ID", "")
            if run_id:
                self.usage_reporter.report_training_usage(
                    run_id=run_id,
                    step=step,
                    tokens=num_input + num_output,
                )
        if self.heart is not None:
            self.heart.beat()

        self.progress.total_tokens += num_tokens
        self.progress.total_samples += num_rollouts
        self.progress.total_problems += num_unique_examples

        self.log_train_batch(batch, step=step, step_time=step_time)

        self.train_sink.reset_pre_filter_stats()
        self.maybe_trigger_eval(self.progress.step)
        trim_process_memory()

    def maybe_trigger_eval(self, step: int) -> None:
        """Fire eligible eval epochs and flip to ``PREFER_EVAL`` if anything
        fires. No-op when eval is not configured."""
        if self.eval_source is None:
            return
        fired = self.eval_source.trigger(step)
        if not fired:
            return
        reason = f"eval was triggered at step {step}"
        self.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason=reason)
        now = time.perf_counter()
        for env_name in fired:
            self.eval_triggered_at[(env_name, step)] = now
        assert self.eval_envs is not None
        total_rollouts = sum(
            self.eval_envs.get(env_name).config.group_size * len(self.eval_envs.get(env_name).examples)
            for env_name in fired
        )
        get_logger().info(f"Starting evals in {', '.join(fired)} ({total_rollouts} total rollouts)")

    def collect_pipeline_view(self) -> tuple[str, dict[str, float]]:
        """Pipeline view for the orchestrator's ``PeriodicLogger``. Returns
        ``(console_body, wandb_payload)``. Per-env ``(env=N, …)``
        breakdowns inline only when there's more than one train / eval env;
        the eval halves drop entirely when nothing is accumulating."""
        # Re-derive the fresh-rollout budget on every tick. The event-driven
        # call sites only fire when an episode arrives, so once the pipeline has
        # fully drained nothing recomputes the budget and a batch left short by
        # dropped groups stays wedged forever. This tick is the only thing that
        # still runs in that state, so the recovery in update_rollout_budget
        # needs to hang off it. The call is idempotent.
        self.update_rollout_budget()
        disp_gauges = self.dispatcher.gauges()
        disp_drain = self.dispatcher.metrics.drained(
            train_envs={e.name for e in self.train_envs},
            eval_envs={e.name for e in self.eval_envs} if self.eval_envs is not None else set(),
        )
        watcher_gauges = self.watcher.gauges()
        lag_stats = EventLoopLagStats.from_monitor(self.lag_monitor)

        inflight_by_env = self.dispatcher.inflight_by_env
        inflight_train = self.dispatcher.inflight_train_count
        inflight_eval = self.dispatcher.inflight_eval_count
        queued_train = self.dispatcher.queued_train_count
        scheduling_train = self.dispatcher.pending_train_schedule_count
        train_batch, train_target, _train_unit = self.train_sink.batch_progress()
        train_buffered = self.train_sink.buffered_count()
        train_batch_by_env = self.train_sink.pending_batch_by_env()
        eval_batches = self.eval_sink.batch_progress() if self.eval_sink is not None else []
        multi_train = len(self.train_envs) > 1
        multi_eval = self.eval_envs is not None and len(self.eval_envs) > 1

        # Train batch: finalized-group survivors only (0→target). Partial-group
        # arrivals are surfaced as a separate ``(+N buffered)`` addendum
        train_pct = train_batch / train_target if train_target else 0.0
        train_batch_part = f"Train batch {train_batch}/{train_target} ({train_pct:.1%})"
        if multi_train:
            pairs = [(e.name, train_batch_by_env.get(e.name, 0)) for e in self.train_envs]
            train_batch_part += " (" + ", ".join(f"{n}={v}" for n, v in pairs) + ")"
        if train_buffered:
            train_batch_part += f" (+{train_buffered} buffered)"
        if queued_train:
            train_batch_part += f" (+{queued_train} queued)"
        if scheduling_train:
            train_batch_part += f" (+{scheduling_train} scheduling)"

        eval_batch_part = ""
        for env, _step, eb, exp, _ebuf in eval_batches:
            eval_pct = eb / exp if exp else 0.0
            eval_batch_part += f" | {env} {eb}/{exp} ({eval_pct:.1%})"

        # Unified inflight tail: total, then train/eval split, then per-env
        # (only when more than one env of a kind makes the split ambiguous)
        inflight_part = (
            f"{inflight_train + inflight_eval} inflight rollouts (train={inflight_train}, eval={inflight_eval}"
        )
        if multi_train or multi_eval:
            env_pairs = [(e.name, inflight_by_env.get(("train", e.name), 0)) for e in self.train_envs]
            if self.eval_envs is not None:
                env_pairs += [(e.name, inflight_by_env.get(("eval", e.name), 0)) for e in self.eval_envs]
            inflight_part += " | " + ", ".join(f"{n}={v}" for n, v in env_pairs)
        inflight_part += ")"

        body = train_batch_part + eval_batch_part + "; " + inflight_part

        payload: dict[str, float] = {**disp_gauges, **disp_drain, **watcher_gauges}
        if lag_stats.n > 0:
            payload["event_loop_lag/min"] = lag_stats.min
            payload["event_loop_lag/mean"] = lag_stats.mean
            payload["event_loop_lag/median"] = lag_stats.median
            payload["event_loop_lag/p90"] = lag_stats.p90
            payload["event_loop_lag/p99"] = lag_stats.p99
            payload["event_loop_lag/max"] = lag_stats.max
            payload["event_loop_lag/n"] = float(lag_stats.n)
        return body, payload

    def log_train_batch(self, batch: TrainBatch, *, step: int, step_time: float) -> None:
        """Per-step ``Step …`` success line. Multi-env runs append an indented ``╰─`` line per env.
        ``Error`` is the sink-level rate (errored arrivals / total arrivals, over the full window);
        the quality metrics are over the effective (clean, trained-on) subset; ``Trainable`` is
        relative to all generated rollouts."""
        rollouts = batch.rollouts
        effective = rollouts.effective
        eff = effective.metrics
        n_generated = len(rollouts)
        n_trainable = sum(1 for r in rollouts if r.is_trainable)
        trainable_rate = (n_trainable / n_generated) if n_generated else 0.0
        max_off_policy = max((r.off_policy_steps for r in effective), default=0)

        head = (
            f"Step {step} | {format_time(step_time):>7} | Reward {eff.reward.mean():.4f} | "
            f"Trainable {n_trainable}/{n_generated} ({trainable_rate:.1%}) | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {eff.num_branches.mean():.1f} | "
            f"Max Off-Policy {max_off_policy} | "
            f"Error {rollouts.metrics.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )
        if len(self.train_envs) <= 1:
            get_logger().success(head)
            return

        by_env = rollouts.by_env()
        name_width = max((len(n) for n in by_env), default=0)
        lines = [head]
        for env_name in sorted(by_env):
            pool = by_env[env_name]
            env_eff_pool = pool.effective
            env_eff = env_eff_pool.metrics
            ratio = (len(pool) / n_generated) if n_generated else 0.0
            lines.append(
                f"╰─ {env_name:<{name_width}} | Ratio {ratio:.1%} | Reward {env_eff.reward.mean():.4f} | "
                f"Turns {env_eff.num_turns.mean():.1f} | Branches {env_eff.num_branches.mean():.1f} | "
                f"Max Off-Policy {max((r.off_policy_steps for r in env_eff_pool), default=0)} | "
                f"Error {pool.metrics.has_error.mean():.1%} | Truncation {env_eff.is_truncated.mean():.1%}"
            )
        get_logger().success("\n\t\t ".join(lines))

    async def finalize_eval_batch(self, batch: EvalBatch) -> None:
        """Persist + log one completed eval epoch (save_rollouts,
        monitor.log_eval_samples, monitor.log)."""
        if not batch.rollouts:
            get_logger().warning(f"Eval @ step={batch.step} env={batch.env_name}: no rollouts returned, skipping log")
            return

        # The non-errored subset lands in the per-step ``effective`` trace file on epoch
        # completion (multiple eval envs share the step file — each epoch appends its cohort
        # once, and every record carries ``env_name``); the full returned cohort already
        # streamed into ``all`` on arrival.
        records = [r.to_record() for r in batch.rollouts.effective]
        await asyncio.to_thread(
            save_rollouts, records, get_trace_path(self.config.output_dir, batch.step, "eval", "effective")
        )
        self.monitor.log_eval_samples(batch.rollouts, env_name=batch.env_name, step=batch.step)
        policy_versions = {r.policy_version for r in batch.rollouts}
        policy_version = min(policy_versions)
        if len(policy_versions) > 1:
            get_logger().warning(
                f"Eval {batch.env_name} step {batch.step} had mixed policy versions: {sorted(policy_versions)}"
            )
        # Rollout metrics over {all,effective} (eval batches are per-env, so no `agg` axis).
        # ``effective`` = non-errored; pass@k / pass^k only over the effective set.
        rollouts = batch.rollouts
        effective = rollouts.effective
        metrics: dict[str, float] = {}
        for subset, pool in (("all", rollouts), ("effective", effective)):
            metrics |= pool.metrics.to_wandb(prefix=f"eval/{batch.env_name}", subset=subset)
        metrics[f"eval/{batch.env_name}/policy_version"] = float(policy_version)
        metrics["step"] = float(batch.step)
        self.monitor.log(metrics, step=batch.step)

        # Success line — reward / turns / truncation over the effective set, error rate + branches
        # over the full returned cohort. ``Stat.mean()`` is 0.0 for an empty set.
        eff, full = effective.metrics, rollouts.metrics
        triggered_at = self.eval_triggered_at.pop((batch.env_name, batch.step), None)
        elapsed = (time.perf_counter() - triggered_at) if triggered_at is not None else 0.0
        get_logger().success(
            f"Evaluated {batch.env_name} (Step {batch.step}) | "
            f"Policy v{policy_version} | {format_time(elapsed):>7} | Reward {eff.reward.mean():.4f} | "
            f"Turns {eff.num_turns.mean():.1f} | Branches {full.num_branches.mean():.1f} | "
            f"Error {full.has_error.mean():.1%} | Truncation {eff.is_truncated.mean():.1%}"
        )

    async def maybe_save_ckpt(self, step: int) -> float:
        """Checkpoint the step just shipped if it's an interval boundary. Returns
        elapsed time (0.0 when no save happened)."""
        if self.ckpt_manager is None or self.config.ckpt is None or not self.config.ckpt.interval:
            return 0.0
        # The final step's checkpoint is written once in ``start()``'s teardown; skip it here so
        # we don't double-save. This mirrors the trainer (its is_last_step skips the in-loop save).
        if self.config.max_steps is not None and step >= self.config.max_steps:
            return 0.0
        if step % self.config.ckpt.interval != 0:
            return 0.0
        get_logger().info(f"Saving checkpoint at step {step}")
        t = time.perf_counter()
        await asyncio.to_thread(self.ckpt_manager.save, self.progress, step)
        return time.perf_counter() - t

    def update_rollout_budget(self) -> None:
        if self.config.batch_size is None:
            return
        current, _, _ = self.train_sink.batch_progress()
        live = (
            self.dispatcher.queued_train_count
            + self.dispatcher.inflight_train_count
            + self.dispatcher.pending_train_schedule_count
        )
        buffered = self.train_sink.buffered_count()
        group = max(1, getattr(self.config, "group_size", 1))
        # Full-group overflow carries into the next step.
        budget = rollout_budget(
            batch_size=self.config.batch_size,
            current=current,
            buffered=buffered,
            live=live,
            group_size=group,
            max_inflight=self.config.max_inflight_rollouts,
            replica_count=len(self.policy_inference.train_clients),
        )
        # Rollouts buffered in a partial group cannot finish once nothing is in
        # flight to finish them: their group-mates were dropped by a filter
        # (zero-advantage groups are enforced) or lost. Counting them against
        # the budget then wedges the batch permanently — it shows up as
        # "Train batch N/1024 ... 0 inflight rollouts" with the trainer idle
        # indefinitely. When the pipeline has fully drained and the batch is
        # still short, ignore the stuck buffer and open fresh groups. Guarded on
        # current > 0 so the normal post-ship path (current == 0, buffer carried
        # into the next batch) keeps its existing behaviour.
        # Fires whenever the pipeline is drained and the budget cannot open a
        # whole group. Includes current == 0 with a buffer full of orphans:
        # over hundreds of steps dropped group-mates accumulate until
        # buffered >= batch_size at a post-ship boundary, and that state is
        # indistinguishable from a healthy idle orchestrator. The only state
        # excluded is a true fresh start (nothing collected, nothing buffered),
        # where budget == batch_size >= group anyway.
        if live == 0 and current < self.config.batch_size and (current + buffered) > 0 and budget < group:
            # Two wedge shapes end here, both showing "N/1024 ... 0 inflight":
            #   * budget <= 0: buffered partial-group rollouts absorbed it all
            #     (their group-mates were dropped by the zero-advantage filter);
            #   * 0 < budget < group_size: the tail of the batch needs fewer
            #     rollouts than one group, and groups only open whole. Seen live
            #     as 1017/1024 with budget 7 and group size 8, stuck for good.
            # Either way nothing is in flight, so nothing can ever finish the
            # batch. Round the shortfall UP to whole groups; the overshoot lands
            # in the buffer and carries into the next batch, which is the normal
            # post-ship path anyway.
            shortfall = self.config.batch_size - current
            budget = ((shortfall + group - 1) // group) * group
            get_logger().warning(
                f"Batch stuck at {current}/{self.config.batch_size} with nothing in flight "
                f"({buffered} buffered); opening {budget // group} fresh group(s) to finish it"
            )
        self.dispatcher.set_fresh_train_rollout_budget(budget)

    def update_dispatch_permission(self) -> None:
        """Pause/resume the dispatcher based on how far the in-flight batch runs
        ahead of ``policy.version``. ``progress.step`` is always the batch being
        collected — advanced right after shipping — so both call sites (ship time
        here, policy update in ``on_new_version``) share one lead formula. Steps
        are 1-indexed while policy versions stay 0-indexed, so the shipped-batch
        count is ``progress.step - 1``."""
        lead = (self.progress.step - 1) - self.policy.version
        # The trainer skips the final in-memory weight broadcasts, so policy.version never
        # reaches the last step. Let the final batch through instead of waiting for it.
        building_final_batch_without_update = (
            self.config.weight_broadcast.type in ("nccl", "nixl")
            and self.config.max_steps is not None
            and self.progress.step >= self.config.max_steps - 1
        )
        dispatch_permission = self.dispatcher.dispatch_allowed
        was_set = dispatch_permission.is_set()
        if lead > self.config.max_off_policy_steps and not building_final_batch_without_update:
            if was_set:
                get_logger().info(
                    "Pausing rollout scheduling because the orchestrator is too far "
                    "ahead of the trainer. Waiting for a new policy."
                )
                self.dispatch_paused_at = time.perf_counter()
            dispatch_permission.clear()
        else:
            if not was_set:
                get_logger().info("Resuming dispatcher")
                if self.dispatch_paused_at is not None:
                    self.wait_for_policy_time += (
                        time.perf_counter() - self.dispatch_paused_at
                    )
                    self.dispatch_paused_at = None
            dispatch_permission.set()

    async def on_version_pending(self, step: int) -> None:
        """``VersionObserver`` hook, fired at publish confirmation (pre-apply):
        ``policy.version`` already carries the new version, so wake a held ship."""
        if self.model_express is not None:
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_READY)
        restarted = self.dispatcher.restarted_group_ids(step)
        if restarted:
            discarded = self.train_sink.abandon_groups(restarted)
            get_logger().info(
                f"Policy v{step}: removed {discarded} buffered rollout episode(s) from "
                f"{len(restarted)} restarted GRPO group(s)"
            )
            self.update_rollout_budget()
        self.version_advanced.set()

    async def wait_for_policy_update_boundary(self, step: int) -> None:
        """Let the collecting train batch ship before invalidating groups.

        The adapter is already present on every inference host when this hook
        runs. Defer only when the current policy is recent enough to ship the
        batch under the configured off-policy limit. The dispatcher continues
        to schedule old-policy groups during this wait. Otherwise the batch
        could wait for the same update and deadlock.
        """
        if self.config.batch_size is None or self.config.max_off_policy_steps < 1:
            return
        current, target, _ = self.train_sink.batch_progress()
        # There is no accepted batch work to preserve at zero progress. Apply
        # the staged policy now. Waiting here can consume the full tail timeout
        # and then restart the same incomplete groups after the batch grows.
        if current == 0:
            return
        if (
            self.dispatcher.inflight_train_count
            + self.dispatcher.pending_train_schedule_count
            + self.dispatcher.queued_train_count
            == 0
        ):
            return

        collecting_step = self.progress.step
        required_version = collecting_step - 1 - self.config.max_off_policy_steps
        if self.policy.version < required_version:
            return

        get_logger().info(
            f"Policy v{step}: deferring the group boundary at {current}/{target} "
            f"until rollout batch {collecting_step} ships (maximum "
            f"{POLICY_UPDATE_BATCH_WAIT_TIMEOUT_S}s)"
        )
        self.batch_completed.clear()
        # Avoid losing a completion that raced the clear above.
        if self.progress.step != collecting_step or self.stopped.is_set():
            return
        try:
            await asyncio.wait_for(
                self.batch_completed.wait(), timeout=POLICY_UPDATE_BATCH_WAIT_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            get_logger().warning(
                f"Policy v{step}: rollout batch {collecting_step} did not finish during the "
                "bounded tail deferral; applying the policy update now"
            )
        else:
            self.dispatcher.mark_batch_boundary_complete(step)

    async def on_new_version(self, step: int) -> None:
        """``VersionObserver`` hook: the weight update completed;
        re-evaluate the dispatch permission (may resume if the trainer caught up)."""
        if self.model_express is not None:
            await asyncio.to_thread(self.model_express.set_status, p2p_pb2.SOURCE_STATUS_INITIALIZING)
        self.update_dispatch_permission()

    async def stop(self) -> None:
        """Bounded best-effort teardown of all components. Has a global
        timeout so a wedged peer can't keep the process alive forever —
        training artifacts are already persisted before this is reached."""

        self.batch_completed.set()

        async def teardown() -> None:
            if self.sender is not None:
                self.sender.close()
            if self.dispatcher is not None:
                await self.dispatcher.stop()
            if self.watcher is not None:
                await self.watcher.stop()
            if self.periodic_logger is not None:
                await self.periodic_logger.stop()
            if self.lag_task is not None:
                await safe_cancel(self.lag_task)
                self.lag_task = None
            for task in self.component_tasks:
                await safe_cancel(task)
            self.component_tasks.clear()
            if self.inference_metrics is not None:
                await self.inference_metrics.stop()
            if getattr(self, "policy_inference", None) is not None:
                await self.policy_inference.stop()
            if self.train_envs is not None:
                for env in self.train_envs:
                    for pool in (*env.sampler.connected_pools, *env.algorithm.connected_pools):
                        await pool.stop()
                self.train_envs.shutdown()
            if self.eval_envs is not None:
                self.eval_envs.shutdown()
            if self.usage_reporter is not None:
                self.usage_reporter.close()

        task = asyncio.create_task(teardown())
        _, pending = await asyncio.wait({task}, timeout=SHUTDOWN_TIMEOUT_S)
        if pending:
            get_logger().warning(
                f"Orchestrator shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
                "forcing process exit. Training artifacts are already persisted."
            )
            os._exit(0)
        await task


@clean_exit
async def run_orchestrator(config: OrchestratorConfig) -> None:
    """Top-level entrypoint. Wrapped in ``@clean_exit`` so wandb is flushed
    on exit (success or crash); keeps that out of the class.
    """
    await Orchestrator(config).start()


def main() -> None:
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("Orchestrator")
    import uvloop

    uvloop.install()
    asyncio.run(run_orchestrator(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
