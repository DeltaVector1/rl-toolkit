import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before import

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone

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
from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader, RolloutCacheDataLoader
from prime_rl.utils.cp import (
    gather_for_cp,
    gather_for_cp_wo_grad,
    setup_cp_params,
    shard_for_cp,
)
from prime_rl.utils.checkpoint_delta import sha256_file
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
from prime_rl.parity_router_trace import (
    prepare_trainer_bundle_communication,
    publish_trainer_bundle,
    trace_dir_env,
)
from prime_rl.trainer.model import (
    DTYPE_MAP,
    forward,
    setup_tokenizer,
    setup_model,
    is_tt_moe_model,
    get_load_balance_stats,
    preserve_glm_moe_bias_precision,
    cache_glm_moe_bias_values,
    restore_glm_moe_bias_values,
    validate_glm_moe_bias_values,
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
from prime_rl.trainer.runs import (
    setup_multi_run_manager,
    Progress,
    get_multi_run_manager,
)
from prime_rl.trainer.models.layers.lora import set_lora_num_tokens
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.metrics_server import HealthServer, MetricsServer, RunStats
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import (
    clean_exit,
    get_broadcast_dir,
    get_step_path,
    resolve_latest_ckpt_step,
    to_col_format,
)
from ring_flash_attn import substitute_hf_flash_attn
from torchtitan.distributed.utils import clip_grad_norm_


def _parse_cpu_list(value: str) -> set[int]:
    cpus = set()
    for part in value.strip().split(","):
        start, separator, end = part.partition("-")
        if not start:
            raise ValueError(f"invalid CPU list {value!r}")
        first = int(start)
        last = int(end) if separator else first
        if last < first:
            raise ValueError(f"invalid CPU range {part!r}")
        cpus.update(range(first, last + 1))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def bind_worker_to_gpu_numa_node() -> tuple[int, int] | None:
    """Bind one trainer worker and its communication threads near its GPU."""
    if os.environ.get("PRIME_BIND_GPU_NUMA", "0") != "1":
        return None
    local_rank = int(os.environ["LOCAL_RANK"])
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    bus_ids = [line.strip().lower() for line in result.stdout.splitlines() if line.strip()]
    if local_rank >= len(bus_ids):
        raise RuntimeError(
            f"local rank {local_rank} has no GPU PCI address in {bus_ids}"
        )
    pci_address = bus_ids[local_rank][-12:]
    node_path = Path("/sys/bus/pci/devices") / pci_address / "numa_node"
    try:
        numa_node = int(node_path.read_text().strip())
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            f"cannot read NUMA node for local GPU {local_rank} at {node_path}"
        ) from error
    if numa_node < 0:
        raise RuntimeError(f"local GPU {local_rank} has no NUMA node")
    cpu_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
    cpus = _parse_cpu_list(cpu_path.read_text())
    os.sched_setaffinity(0, cpus)
    return numa_node, len(cpus)


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
            micro_batch[name] = torch.nn.functional.pad(
                tensor, (0, padding), value=value
            )

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


def verify_pipeline_batch_identity(micro_batches, pp_group) -> None:
    """Fail before forward when stages in one pipeline loaded different data."""
    digests = []
    for micro_batch in micro_batches:
        digest = hashlib.sha256()
        for name in (
            "input_ids",
            "position_ids",
            "inference_logprobs",
            "loss_mask",
            "lora_num_tokens",
            "seq_lens",
        ):
            tensor = micro_batch[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(bytes(tensor.numpy().data))
        digest.update(json.dumps(micro_batch["sequence_lengths"]).encode())
        digests.append(
            int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)
        )
    local = torch.tensor(digests, dtype=torch.int64, device="cuda")
    minimum = local.clone()
    maximum = local.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=pp_group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=pp_group)
    mismatched = torch.nonzero(minimum != maximum).flatten().tolist()
    if mismatched:
        raise RuntimeError(
            "pipeline stages loaded different trainer data for microbatches "
            f"{mismatched[:10]}; check data-parallel rank mapping"
        )


def initialize_fresh_lora_policy(
    multi_run_manager, lora_config, device: torch.device | str
) -> tuple[int, int]:
    """Make policy zero exactly equal the base model and verify all ranks."""
    fresh_parameters = list(multi_run_manager.get_named_parameters_for_run(0))
    fresh_b = [
        (name, parameter)
        for name, parameter in fresh_parameters
        if "lora_B" in name
    ]
    if not fresh_b:
        raise RuntimeError("fresh LoRA policy has no B matrices")
    nonzero_b_shards = torch.tensor(
        sum(
            int(
                torch.count_nonzero(
                    parameter.to_local()
                    if hasattr(parameter, "to_local")
                    else parameter
                ).item()
                > 0
            )
            for _, parameter in fresh_b
        ),
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(nonzero_b_shards, op=dist.ReduceOp.SUM)
    with torch.no_grad():
        for _, parameter in fresh_b:
            parameter.zero_()

    local_bad = torch.zeros((), dtype=torch.int32, device=device)
    for _, parameter in fresh_parameters:
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        local_bad = torch.maximum(
            local_bad,
            (~torch.isfinite(local).all()).to(dtype=torch.int32),
        )
    for _, parameter in fresh_b:
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        local_bad = torch.maximum(
            local_bad,
            (torch.count_nonzero(local) != 0).to(dtype=torch.int32),
        )
    expected_scale = lora_config.alpha / lora_config.rank
    scale = torch.as_tensor(
        multi_run_manager.scaling_factors[0], device=device, dtype=torch.float32
    )
    local_bad = torch.maximum(
        local_bad,
        (~torch.isclose(scale, torch.tensor(expected_scale, device=device))).to(
            dtype=torch.int32
        ),
    )
    dist.all_reduce(local_bad, op=dist.ReduceOp.MAX)
    if local_bad.item():
        raise RuntimeError(
            "fresh LoRA policy initialization failed: parameters must be finite, "
            "all B matrices must be zero, and the scaling factor must match alpha/rank"
        )
    return len(fresh_b), int(nonzero_b_shards.item())


def parity_guard_failures(tensor_stats: dict, guard) -> list[str]:
    """Return parity violations. Missing or nonfinite evidence fails closed."""
    failures = []
    unmasked = tensor_stats.get("unmasked_mismatch_kl/mean")
    masked = tensor_stats.get("masked_mismatch_kl/mean")
    masked_fraction = tensor_stats.get("is_masked/mean")

    if unmasked is None or not math.isfinite(unmasked):
        failures.append("unmasked mismatch KL is missing or nonfinite")
    elif unmasked > guard.max_unmasked_mismatch_kl:
        failures.append(
            f"unmasked mismatch KL {unmasked:.6f} > {guard.max_unmasked_mismatch_kl:.6f}"
        )

    if masked_fraction is None or not math.isfinite(masked_fraction):
        failures.append("masked token fraction is missing or nonfinite")
    else:
        if masked_fraction > guard.max_masked_fraction:
            failures.append(
                f"masked token fraction {masked_fraction:.6f} > {guard.max_masked_fraction:.6f}"
            )
        if masked_fraction > 0:
            if masked is None or not math.isfinite(masked):
                failures.append("masked mismatch KL is missing or nonfinite")
            elif masked > guard.max_masked_mismatch_kl:
                failures.append(
                    f"masked mismatch KL {masked:.6f} > {guard.max_masked_mismatch_kl:.6f}"
                )
    return failures


_qualification_environment_names = (
    "PRIME_QUALIFICATION_SESSION_ID",
    "PRIME_QUALIFICATION_PROFILE",
    "PRIME_QUALIFICATION_CONFIG_SHA256",
    "PRIME_QUALIFICATION_SOURCE_SHA256",
    "PRIME_QUALIFICATION_SESSION_SHA256",
    "PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256",
    "PRIME_QUALIFICATION_LAUNCHER_SHA256",
    "PRIME_TRAINER_IMAGE_DIGEST",
    "PRIME_TRAINER_HOST",
    "PRIME_TRAINER_HEAD",
)


def _qualification_environment() -> dict[str, str] | None:
    values = {name: os.environ.get(name, "") for name in _qualification_environment_names}
    qualification_names = [
        name
        for name in _qualification_environment_names
        if name.startswith("PRIME_QUALIFICATION_")
    ]
    if not any(values[name] for name in qualification_names):
        return None
    if not all(values.values()):
        raise RuntimeError("qualification identity is incomplete")
    if not re.fullmatch(r"[0-9a-f]{32}", values["PRIME_QUALIFICATION_SESSION_ID"]):
        raise RuntimeError("qualification session ID is invalid")
    if not re.fullmatch(
        r"glm52-native-(?:b2048|stability20)",
        values["PRIME_QUALIFICATION_PROFILE"],
    ):
        raise RuntimeError("qualification profile is invalid")
    for name in (
        "PRIME_QUALIFICATION_CONFIG_SHA256",
        "PRIME_QUALIFICATION_SOURCE_SHA256",
        "PRIME_QUALIFICATION_SESSION_SHA256",
        "PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256",
        "PRIME_QUALIFICATION_LAUNCHER_SHA256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", values[name]):
            raise RuntimeError(f"{name} is invalid")
    if not re.fullmatch(
        r"sha256:[0-9a-f]{64}", values["PRIME_TRAINER_IMAGE_DIGEST"]
    ):
        raise RuntimeError("trainer image digest is invalid")
    for name in ("PRIME_TRAINER_HOST", "PRIME_TRAINER_HEAD"):
        if not re.fullmatch(r"n[0-9]+", values[name]):
            raise RuntimeError(f"{name} is invalid")
    return values


def _qualification_runtime_records(
    identity: dict[str, str], world, session: dict
) -> list[dict]:
    local = {
        "global_rank": world.rank,
        "local_rank": world.local_rank,
        "host": identity["PRIME_TRAINER_HOST"],
        "image_digest": identity["PRIME_TRAINER_IMAGE_DIGEST"],
        "source_sha256": identity["PRIME_QUALIFICATION_RUNTIME_SOURCE_SHA256"],
        "launcher_sha256": identity["PRIME_QUALIFICATION_LAUNCHER_SHA256"],
    }
    payload = json.dumps(local, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > 1024:
        raise RuntimeError("qualification runtime identity is too large")
    encoded = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    encoded[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8, device="cuda")
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)
    failed = torch.zeros((), dtype=torch.int32, device="cuda")
    records = []
    if world.is_master:
        try:
            for value in gathered:
                raw = bytes(value.cpu().tolist()).split(b"\0", 1)[0]
                records.append(json.loads(raw))
            hosts = session["trainer_hosts"]
            images = session["image_digests"]
            source_sha256 = session["source_sha256"]
            launcher_sha256 = session["source_files"]["scripts/china/cluster.sh"]
            if len(records) != 64 or len(hosts) != 8:
                raise RuntimeError("qualification runtime identity count differs")
            for rank, record in enumerate(records):
                expected_host = hosts[rank // 8]
                if (
                    record.get("global_rank") != rank
                    or record.get("local_rank") != rank % 8
                    or record.get("host") != expected_host
                    or record.get("image_digest") != images.get(expected_host)
                    or record.get("source_sha256") != source_sha256
                    or record.get("launcher_sha256") != launcher_sha256
                ):
                    raise RuntimeError(
                        f"qualification runtime identity differs at rank {rank}"
                    )
        except Exception:
            failed.fill_(1)
    dist.broadcast(failed, src=0)
    if failed.item():
        raise RuntimeError("qualification runtime identity validation failed")
    return records


def _qualification_batch_sha256(micro_batches: list[dict]) -> str:
    digest = hashlib.sha256()
    for index, micro_batch in enumerate(micro_batches):
        input_ids = micro_batch["input_ids"].detach().cpu().contiguous()
        lengths = micro_batch["sequence_lengths"]
        digest.update(index.to_bytes(4, "little"))
        digest.update(str(input_ids.dtype).encode())
        digest.update(json.dumps(list(input_ids.shape), separators=(",", ":")).encode())
        digest.update(input_ids.numpy().tobytes())
        digest.update(json.dumps(lengths, separators=(",", ":")).encode())
    return digest.hexdigest()


def _verify_qualification_batch_sha256(value: str) -> None:
    encoded = torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8, device="cuda")
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)
    if any(not torch.equal(item, encoded) for item in gathered):
        raise RuntimeError("qualification input differs between ranks")


def _qualification_step_records(
    history: list[dict], max_steps: int, input_sha256: dict[int, str]
) -> list[dict]:
    keys = (
        "data/rollout_rows",
        "loss/mean",
        "optim/grad_norm",
        "optim/lr",
        "perf/peak_memory",
        "time/forward_backward",
    )
    merged = {step: {"step": step} for step in range(1, max_steps + 1)}
    for record in history:
        step = record.get("step") if isinstance(record, dict) else None
        if step not in merged:
            continue
        for key in keys:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise RuntimeError(f"qualification metric {key} is not scalar")
                value = value.item()
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"qualification metric {key} is not numeric")
            value = float(value)
            previous = merged[step].get(key)
            if previous is not None and previous != value:
                raise RuntimeError(f"qualification metric {key} conflicts at step {step}")
            merged[step][key] = value
    for step, record in merged.items():
        missing = set(keys) - set(record)
        if missing:
            raise RuntimeError(
                f"qualification step {step} is missing metrics: {sorted(missing)}"
            )
        if not all(math.isfinite(record[key]) for key in keys):
            raise RuntimeError(f"qualification step {step} has a nonfinite metric")
        digest = input_sha256.get(step)
        if digest is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"qualification step {step} input checksum is missing")
        record["input_sha256"] = digest
        if step > 1 and digest == input_sha256.get(step - 1):
            raise RuntimeError(f"qualification step {step} reused the previous input")
    return list(merged.values())


def _write_atomic_json(path: Path, value: dict, session_id: str) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.parent / f".{path.name}-{session_id}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _validate_resume_adapter_inventory(
    adapter_shapes: dict[str, tuple[int, ...]],
    expected_by_rank: list[dict[str, tuple[int, ...]]],
) -> None:
    expected: dict[str, tuple[int, ...]] = {}
    conflicts: list[str] = []
    for rank_expected in expected_by_rank:
        for key, shape in rank_expected.items():
            previous = expected.setdefault(key, shape)
            if previous != shape:
                conflicts.append(key)
    missing = sorted(set(expected) - set(adapter_shapes))
    unexpected = sorted(set(adapter_shapes) - set(expected))
    mismatched = sorted(
        key
        for key in set(expected) & set(adapter_shapes)
        if expected[key] != adapter_shapes[key]
    )
    if conflicts or missing or unexpected or mismatched:
        raise ValueError(
            "Resume adapter inventory differs from the complete distributed LoRA model: "
            f"conflicts={conflicts[:5]} ({len(conflicts)} total), "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"unexpected={unexpected[:5]} ({len(unexpected)} total), "
            f"shape_mismatch={mismatched[:5]} ({len(mismatched)} total)"
        )


def _load_qualification_session(
    config: TrainerConfig, identity: dict[str, str]
) -> dict:
    session_path = config.output_dir / "qualification-session.json"
    session_bytes = session_path.read_bytes()
    if hashlib.sha256(session_bytes).hexdigest() != identity[
        "PRIME_QUALIFICATION_SESSION_SHA256"
    ]:
        raise RuntimeError("qualification session file checksum differs")
    session = json.loads(session_bytes)
    if (
        session.get("session_id") != identity["PRIME_QUALIFICATION_SESSION_ID"]
        or session.get("profile") != identity["PRIME_QUALIFICATION_PROFILE"]
        or session.get("config_sha256")
        != identity["PRIME_QUALIFICATION_CONFIG_SHA256"]
        or session.get("source_sha256")
        != identity["PRIME_QUALIFICATION_SOURCE_SHA256"]
    ):
        raise RuntimeError("qualification session file identity differs")
    return session


def _write_qualification_result(
    config: TrainerConfig,
    parallel_dims,
    monitor,
    progress: Progress,
    identity: dict[str, str],
    runtime_records: list[dict],
    input_sha256: dict[int, str],
    adapter_path: Path | None,
) -> None:
    _load_qualification_session(config, identity)
    if config.max_steps is None or progress.step != config.max_steps:
        raise RuntimeError("qualification did not finish its exact step count")
    records = _qualification_step_records(
        monitor.history, config.max_steps, input_sha256
    )
    metrics_path = config.output_dir / config.file_monitor.filename
    adapter = None
    if config.data.fake.export_final_adapter:
        expected_adapter_path = (
            config.output_dir
            / "run_trainer_only"
            / "broadcasts"
            / f"step_{progress.step}"
            / "adapter_model.safetensors"
        )
        if adapter_path != expected_adapter_path:
            raise RuntimeError("qualification final adapter path differs")
        if adapter_path is None or not adapter_path.is_file():
            raise RuntimeError("qualification final adapter is missing")
        stable = adapter_path.parent / "STABLE"
        adapter_config = adapter_path.parent / "adapter_config.json"
        digest_path = adapter_path.parent / "adapter_model.safetensors.sha256"
        if (
            not stable.is_file()
            or not adapter_config.is_file()
            or not digest_path.is_file()
        ):
            raise RuntimeError("qualification final adapter is incomplete")
        adapter_sha256 = sha256_file(adapter_path)
        checksum_value = digest_path.read_text()
        if checksum_value != f"{adapter_sha256}\n":
            raise RuntimeError("qualification final adapter checksum file differs")
        adapter_config_value = json.loads(adapter_config.read_bytes())
        expected_targets = {
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "shared_expert",
            "experts",
        }
        adapter_targets = adapter_config_value.get("target_modules")
        if (
            isinstance(adapter_config_value.get("r"), bool)
            or adapter_config_value.get("r") != 32
            or isinstance(adapter_config_value.get("lora_alpha"), bool)
            or adapter_config_value.get("lora_alpha") != 64
            or isinstance(adapter_config_value.get("lora_dropout"), bool)
            or adapter_config_value.get("lora_dropout") != 0.0
            or not isinstance(adapter_targets, list)
            or any(not isinstance(target, str) for target in adapter_targets)
            or len(adapter_targets) != len(expected_targets)
            or set(adapter_targets) != expected_targets
        ):
            raise RuntimeError("qualification final adapter configuration differs")
        adapter = {
            "path": str(adapter_path),
            "sha256": adapter_sha256,
            "size": adapter_path.stat().st_size,
            "step": progress.step,
            "config_path": str(adapter_config),
            "config_sha256": sha256_file(adapter_config),
            "stable_path": str(stable),
            "stable_sha256": sha256_file(stable),
            "checksum_path": str(digest_path),
            "checksum_sha256": sha256_file(digest_path),
            "checksum_value": checksum_value.rstrip("\n"),
            "config": adapter_config_value,
        }
    fake = config.data.fake
    result = {
        "schema_version": 1,
        "kind": "glm52_trainer_qualification",
        "session_id": identity["PRIME_QUALIFICATION_SESSION_ID"],
        "profile": identity["PRIME_QUALIFICATION_PROFILE"],
        "session_sha256": identity["PRIME_QUALIFICATION_SESSION_SHA256"],
        "config_sha256": identity["PRIME_QUALIFICATION_CONFIG_SHA256"],
        "source_sha256": identity["PRIME_QUALIFICATION_SOURCE_SHA256"],
        "trainer_head": identity["PRIME_TRAINER_HEAD"],
        "trainer_head_image_digest": identity["PRIME_TRAINER_IMAGE_DIGEST"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "topology": {
            "world_size": dist.get_world_size(),
            "pp": parallel_dims.pp,
            "cp": parallel_dims.cp,
            "ep": parallel_dims.ep,
            "dp_replicate": parallel_dims.dp_replicate,
            "dp_shard": parallel_dims.dp_shard,
        },
        "fake_data": {
            "batch_size": fake.batch_size,
            "samples_per_micro_batch": fake.samples_per_micro_batch,
            "group_size": fake.group_size,
            "sequence_length": config.model.seq_len,
            "rollout_cache": False,
        },
        "lora": {
            "rank": config.model.lora.rank,
            "alpha": config.model.lora.alpha,
            "targets": sorted(config.model.lora.target_modules),
        },
        "max_steps": config.max_steps,
        "steps": records,
        "runtime_ranks": runtime_records,
        "runtime_sha256": hashlib.sha256(
            (
                json.dumps(runtime_records, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode()
        ).hexdigest(),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "adapter": adapter,
    }
    _write_atomic_json(
        config.output_dir / "qualification-result.json",
        result,
        identity["PRIME_QUALIFICATION_SESSION_ID"],
    )


@clean_exit
def train(config: TrainerConfig):
    # Setup world and logger
    world = get_world()
    numa_binding = bind_worker_to_gpu_numa_node()
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    logger.info(f"Starting RL trainer in {world} in {config.output_dir}")
    qualification_identity = _qualification_environment()
    if qualification_identity is not None:
        fake = config.data.fake
        expected_steps = {
            "glm52-native-b2048": (2, False),
            "glm52-native-stability20": (20, True),
        }[qualification_identity["PRIME_QUALIFICATION_PROFILE"]]
        expected_targets = {
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "shared_expert",
            "experts",
        }
        if (
            fake is None
            or config.data.rollout_cache is not None
            or fake.batch_size != 64
            or fake.samples_per_micro_batch != 32
            or fake.group_size != 16
            or fake.export_final_adapter is not expected_steps[1]
            or config.max_steps != expected_steps[0]
            or config.model.seq_len != 32768
            or config.optim.lr != 8e-6
            or config.model.lora is None
            or config.model.lora.rank != 32
            or config.model.lora.alpha != 64
            or set(config.model.lora.target_modules) != expected_targets
        ):
            raise RuntimeError("qualification trainer configuration differs")
    if numa_binding is not None:
        logger.info(
            f"Bound local GPU {world.local_rank} worker to NUMA node "
            f"{numa_binding[0]} ({numa_binding[1]} CPUs)"
        )

    # Print warning if running in benchmark mode
    if config.bench is not None:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup the monitor
    logger.info(
        f"Initializing monitor (wandb={config.wandb}, swanlab={config.swanlab})"
    )
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
            logger.info(
                f"Initializing metrics server on port {config.metrics_server.port}"
            )
            metrics_server = MetricsServer(config.metrics_server)
            metrics_server.start()
        else:
            logger.info(
                f"Initializing health server on port {config.metrics_server.port}"
            )
            health_server = HealthServer(
                config.metrics_server.port, config.metrics_server.host
            )
            health_server.start()

    # Set precision
    setup_torch_distributed(
        timeout=timedelta(seconds=config.dist_timeout_seconds),
        enable_gloo=config.model.fsdp_cpu_offload,
    )
    # Configurable to support ROCm/AMD GPUs where reduced precision
    # matmul corrupts softmax over large vocabularies. Override via config
    # (e.g. matmul_precision = "highest") on ROCm.
    torch.set_float32_matmul_precision(config.matmul_precision)

    # Setup multi run manager and offsets (including LoRA validation/scaling hooks if applicable)
    multi_run_manager = setup_multi_run_manager(
        config.output_dir,
        config.max_concurrent_runs,
        torch.device("cuda", world.local_rank),
        config.model.lora,
    )
    replay_rollouts = (
        config.data.rollout_cache is not None
        and config.data.rollout_cache.mode == "replay"
    )
    trainer_only = config.data.fake is not None or replay_rollouts

    # Resolve ep="auto" to a concrete integer before creating parallel dims
    resolve_ep(config.model)

    # Initialize parallel dimensions
    parallel_dims = get_parallel_dims(config.model)
    if qualification_identity is not None and (
        world.world_size,
        parallel_dims.pp,
        parallel_dims.cp,
        parallel_dims.ep,
        parallel_dims.dp_replicate,
        parallel_dims.dp_shard,
    ) != (64, 8, 8, 8, 1, 1):
        raise RuntimeError("qualification trainer topology differs")
    qualification_runtime_records = []
    if qualification_identity is not None:
        qualification_session = _load_qualification_session(
            config, qualification_identity
        )
        qualification_runtime_records = _qualification_runtime_records(
            qualification_identity, world, qualification_session
        )

    # For single-run, check for checkpoint to resume from
    checkpoint_step = None
    if config.max_concurrent_runs == 1:
        # Set up checkpoint manager for single-run
        logger.info(f"Initializing checkpoint managers ({config.ckpt})")
        ckpt_manager, weight_ckpt_manager = setup_ckpt_managers(
            config.output_dir, config.ckpt, config.model.lora
        )

        if (
            config.ckpt
            and config.ckpt.resume_step is not None
            and ckpt_manager is not None
        ):
            if config.ckpt.resume_step == -1:
                checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
            else:
                checkpoint_step = config.ckpt.resume_step
    else:
        # Multi-run uses per-run checkpointing via MultiCheckpointManager
        ckpt_manager, weight_ckpt_manager = setup_multi_checkpoint_manager(
            config.output_dir
        )
        logger.info("Initialized multi-run checkpoint manager")

    # Initialize the model and tokenizer
    logger.info(f"Initializing model ({config.model})")
    loading_from_ckpt_later = config.ckpt and checkpoint_step is not None
    model = setup_model(config.model, parallel_dims, loading_from_ckpt_later)
    if trainer_only and config.model.lora is not None:
        multi_run_manager.create_trainer_only_run(config.model.lora)
    group_timeout = timedelta(seconds=config.dist_timeout_seconds)
    process_groups = {}
    for mesh in (parallel_dims.world_mesh, *parallel_dims._submeshes.values()):
        for group in mesh.get_all_groups():
            process_groups[id(group)] = group
    for group in process_groups.values():
        _set_pg_timeout(group_timeout, group)
    logger.info(
        f"Set {len(process_groups)} distributed group timeouts to {config.dist_timeout_seconds} seconds"
    )
    resume_adapter = (
        os.environ.get("PRIME_RL_RESUME_ADAPTER") if checkpoint_step is None else None
    )
    if resume_adapter:
        from safetensors.torch import load_file
        from torch.distributed.tensor import DTensor, distribute_tensor

        adapter = load_file(resume_adapter)
        expected: dict[str, tuple[int, ...]] = {}
        parameters = {}
        for name, parameter in model.named_parameters():
            if ".lora_A.0" not in name and ".lora_B.0" not in name:
                continue
            key = (
                name.replace("._checkpoint_wrapped_module", "")
                .replace(".lora_A.0", ".lora_A.weight")
                .replace(".lora_B.0", ".lora_B.weight")
            )
            expected[key] = tuple(parameter.shape)
            parameters[key] = parameter
        expected_by_rank: list[dict[str, tuple[int, ...]] | None] = [
            None
        ] * dist.get_world_size()
        dist.all_gather_object(expected_by_rank, expected)
        if any(value is None for value in expected_by_rank):
            raise RuntimeError("Resume adapter inventory gather is incomplete")
        _validate_resume_adapter_inventory(
            {key: tuple(value.shape) for key, value in adapter.items()},
            [value for value in expected_by_rank if value is not None],
        )
        with torch.no_grad():
            for key, parameter in parameters.items():
                value = adapter[key].to("cuda", dtype=parameter.dtype)
                if isinstance(parameter, DTensor):
                    value = distribute_tensor(
                        value, parameter.device_mesh, parameter.placements
                    )
                parameter.copy_(value)
        logger.info(
            f"Loaded {len(parameters)} local tensors from complete adapter {resume_adapter}"
        )

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    if config.model.vlm is not None and not getattr(
        model, "supports_packed_multimodal_training", False
    ):
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
        scheduler = setup_scheduler(
            optimizer, config.scheduler, config.max_steps, config.optim.lr
        )
    else:
        optimizer = setup_multi_optimizer(config.optim, parallel_dims)
        scheduler = setup_multi_scheduler(optimizer, config.scheduler, config.max_steps)

        # Register checkpoint loading callback at index 1 (after scheduler creation at index 0)
        def load_run_checkpoint(_optimizer, idx: int) -> None:
            ckpt_manager.load_run(idx, optimizer, scheduler)

        optimizer.register_post_creation_callback(load_run_checkpoint, index=1)

    logger.info(f"Using `{config.scheduler.type}` scheduler ({config.scheduler})")

    fresh_lora_policy = (
        config.model.lora is not None
        and not trainer_only
        and checkpoint_step is None
        and not resume_adapter
    )
    if fresh_lora_policy:
        multi_run_manager.wait_for_run(0)
        fresh_b_count, reset_b_shards = initialize_fresh_lora_policy(
            multi_run_manager, config.model.lora, torch.device("cuda", world.local_rank)
        )
        expected_scale = config.model.lora.alpha / config.model.lora.rank
        logger.info(
            f"Verified fresh LoRA policy: {fresh_b_count} local B matrices are zero; "
            f"scale={expected_scale:g}; reset {reset_b_shards} nonzero B shards"
        )

    export_final_adapter = bool(
        config.data.fake is not None and config.data.fake.export_final_adapter
    )
    if trainer_only and not export_final_adapter:
        weight_broadcast = None
        logger.info("Skipping weight broadcast setup for trainer-only qualification")
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
            substitute_ring_attn(
                cp_group, heads_k_stride=1, attn_impl=config.model.attn
            )
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
        from prime_rl.trainer.distributed.pp_runner import (
            build_schedule as build_pp_schedule,
        )
        from prime_rl.trainer.distributed.pp_runner import (
            pad_side_inputs_to_common_length as pad_pp_side_inputs,
        )
        from prime_rl.trainer.distributed.pp_runner import force_fsdp_lazy_init
        from prime_rl.trainer.distributed.pp_runner import minimum_pipeline_microbatches
        from prime_rl.trainer.distributed.pp_runner import sparse_cache_transport_shape
        from prime_rl.trainer.distributed.pipeline_parallel import sparse_cache_stage_flags

        pp_rank = parallel_dims.get_mesh("pp").get_local_rank()
        # Weight broadcast: tell the gatherer to merge stage dicts over the pp
        # group. "Stage master" = the first rank within each stage, which is
        # exactly the membership of the global master's pp group.
        from prime_rl.trainer.weights import set_pp_gather_context

        # pp is the slowest-varying mesh dim, so stage s owns the contiguous
        # ranks [s*k, (s+1)*k) with k = world/pp; the stage master is the first.
        _ranks_per_stage = dist.get_world_size() // parallel_dims.pp
        _stage_master_ranks = [
            stage * _ranks_per_stage for stage in range(parallel_dims.pp)
        ]
        set_pp_gather_context(
            parallel_dims.get_mesh("pp").get_group(),
            is_stage_master=(dist.get_rank() % _ranks_per_stage) == 0,
            stage_master_ranks=_stage_master_ranks,
        )
        receive_sparse_indices = False
        send_sparse_indices = False
        if config.model.pp_transfer_sparse_indices:
            receive_sparse_indices, send_sparse_indices = sparse_cache_stage_flags(
                model, pp_rank, parallel_dims.pp
            )
            logger.info(
                f"PP stage {pp_rank}: receive sparse cache={receive_sparse_indices}, "
                f"send sparse cache={send_sparse_indices}"
            )
        pp_stage_module = PipelineStageModule(
            model,
            is_first=pp_rank == 0,
            is_last=pp_rank == parallel_dims.pp - 1,
            offload_cfg=config.model.ac_offloading,
            transfer_sparse_indices=config.model.pp_transfer_sparse_indices,
            receive_sparse_indices=receive_sparse_indices,
            send_sparse_indices=send_sparse_indices,
        )

    # Optionally, resume training from a checkpoint
    progress = Progress()
    if checkpoint_step is not None:
        preserve_glm_moe_bias_precision(model)
        ckpt_manager.load(checkpoint_step, model, [optimizer], scheduler, progress)
        if config.model.freeze_moe_router:
            base_snapshot = Path(config.model.name)
            if not base_snapshot.is_dir():
                raise RuntimeError(
                    "Frozen GLM resume requires the local base model snapshot"
                )
            restore_glm_moe_bias_values(model, base_snapshot, parallel_dims)
        else:
            cache_glm_moe_bias_values(model)
        validate_glm_moe_bias_values(model, parallel_dims)
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
        dataloader = FakeDataLoader(
            config.data.fake,
            config.model.seq_len,
            parallel_dims.get_mesh("dp").size(),
            parallel_dims.get_mesh("dp").get_local_rank(),
        )
    else:
        real_dataloader = None
        if not replay_rollouts:
            real_dataloader = DataLoader(
                config.output_dir,
                progress.step,
                parallel_dims.get_mesh("dp").size(),
                parallel_dims.get_mesh("dp").get_local_rank(),
                config.model.seq_len,
                config.model.cp,
                bin_cost,
                config.rollout_transport,
            )
        if config.data.rollout_cache is not None:
            dataloader = RolloutCacheDataLoader(
                config.data.rollout_cache,
                real_dataloader,
                config.model.seq_len,
                parallel_dims.get_mesh("dp").size(),
                parallel_dims.get_mesh("dp").get_local_rank(),
                config.model.cp,
            )
        else:
            dataloader = real_dataloader

    token_exporter = setup_token_exporter(config, parallel_dims, world, logger, model)
    parity_bundle_enabled = trace_dir_env in config.env_vars
    if parity_bundle_enabled and not parallel_dims.pp_enabled:
        raise ValueError("trainer parity bundle requires pipeline parallel training")
    parity_bundle_communication = prepare_trainer_bundle_communication(
        parity_bundle_enabled, torch.device("cuda", world.local_rank)
    )
    parity_bundle_readiness = None
    qualification_input_sha256: dict[int, str] = {}

    gc_handler = GarbageCollection(config.gc.interval) if config.gc else None

    logger.info(f"Starting training loop (max_steps={config.max_steps or 'infinite'})")
    maybe_record_function = nullcontext
    if config.trace_path:
        logger.info(f"Tracing to {config.trace_path}")
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
        ).__enter__()
        maybe_record_function = record_function
    start_step = progress.step
    while True:
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        if gc_handler is not None:
            gc_handler.run(progress.step)
        is_last_step = (
            config.max_steps is not None and progress.step == config.max_steps
        )

        logger.debug(f"Starting training step {progress.step}")
        step_start_time = time.perf_counter()

        # In-memory transfers broadcast the incoming policy (v{progress.step-1}) before waiting
        # for its rollouts so the trainer and inference pool join the same update lifecycle.
        if (
            progress.step == start_step
            and weight_broadcast is not None
            and config.weight_broadcast.type in ("nccl", "nixl")
        ):
            logger.info(
                f"Broadcasting startup policy weights (v{progress.step - 1}) to inference engines"
            )
            multi_run_manager.wait_for_run(0)
            for idx in multi_run_manager.used_idxs:
                multi_run_manager.ready_to_update[idx] = True
            weight_broadcast.broadcast_weights(model, step=progress.step - 1)

        # Wait for the batch to be available
        logger.debug("Waiting for training batch to arrive")
        wait_for_batch_start_time = time.perf_counter()
        dataloader.wait_for_batch()
        wait_for_batch_time = time.perf_counter() - wait_for_batch_start_time
        logger.debug(
            f"Waited for batch to arrive for {wait_for_batch_time:.2f} seconds"
        )
        if fresh_lora_policy and progress.step == 1:
            fresh_b_count, reset_b_shards = initialize_fresh_lora_policy(
                multi_run_manager,
                config.model.lora,
                torch.device("cuda", world.local_rank),
            )
            logger.info(
                f"Reverified fresh LoRA policy after batch state synchronization: "
                f"{fresh_b_count} local B matrices are zero; "
                f"reset {reset_b_shards} nonzero B shards"
            )
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
            if (
                _ids.dim() != 2
                or _ids.shape[0] != 1
                or not (0 < _ids.shape[1] <= _max_len)
            ):
                _bad.append(
                    (_i, f"input_ids shape {tuple(_ids.shape)} (max len {_max_len})")
                )
        if _bad:
            for _i, _why in _bad[:5]:
                logger.error(f"Corrupt micro-batch {_i}: {_why}")
            logger.error(
                f"Discarding batch at step {progress.step}: {len(_bad)}/{len(micro_batches)} micro-batches corrupt. "
                "Waiting for the next batch rather than crashing the run."
            )
            continue

        if parallel_dims.pp_enabled:
            # Schedule1F1B requires at least one micro-batch for each pipeline
            # stage. An engine restart can make almost all rollouts fail and
            # leave a much smaller packed trainer batch. x60r received two
            # micro-batches under PP8 and PyTorch stopped the complete trainer.
            # Agree on the smallest rank-local count before any later
            # collective, then consume and discard the undersized rollout
            # batch. DataLoader advances to the next fresh batch while the
            # trainer policy step remains unchanged.
            _minimum_count = torch.tensor(
                len(micro_batches), dtype=torch.int64, device="cuda"
            )
            dist.all_reduce(_minimum_count, op=dist.ReduceOp.MIN)
            _minimum_required = minimum_pipeline_microbatches(
                config.model.pp_schedule, parallel_dims.pp
            )
            if _minimum_count.item() < _minimum_required:
                logger.error(
                    f"Discarding undersized rollout batch at step {progress.step}: "
                    f"minimum rank-local micro-batches={_minimum_count.item()}, "
                    f"pipeline schedule {config.model.pp_schedule!r} with "
                    f"{parallel_dims.pp} stages requires {_minimum_required}. "
                    "Waiting for the next fresh batch rather than stopping the trainer."
                )
                continue

            _count_range = torch.tensor(
                [len(micro_batches), -len(micro_batches)],
                dtype=torch.int64,
                device="cuda",
            )
            dist.all_reduce(_count_range, op=dist.ReduceOp.MAX)
            if _count_range[0].item() != -_count_range[1].item():
                raise RuntimeError(
                    "trainer ranks received different microbatch counts before "
                    "pipeline data verification"
                )
            verify_pipeline_batch_identity(
                micro_batches, parallel_dims.get_mesh("pp").get_group()
            )

        batch_size = len(micro_batches)
        if qualification_identity is not None:
            input_sha256 = _qualification_batch_sha256(micro_batches)
            _verify_qualification_batch_sha256(input_sha256)
            previous_sha256 = qualification_input_sha256.get(progress.step - 1)
            if previous_sha256 == input_sha256:
                raise RuntimeError(
                    f"qualification step {progress.step} reused the previous input"
                )
            qualification_input_sha256[progress.step] = input_sha256
        rollout_rows = sum(len(micro_batch["sequence_lengths"]) for micro_batch in micro_batches)
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
            local_rl_scale += (
                int((mask & (rl_w != 0)).sum()) if rl_w is not None else int(mask.sum())
            )
            if micro_batch["ce_weights"] is not None:
                local_ce_scale += int((micro_batch["ce_weights"] != 0).sum())
            if micro_batch["ref_kl_weights"] is not None:
                local_ref_kl_scale += int((micro_batch["ref_kl_weights"] != 0).sum())
        global_scales = torch.tensor(
            [local_rl_scale, local_ce_scale, local_ref_kl_scale],
            dtype=torch.int64,
            device="cuda",
        )
        dp_cp_group = parallel_dims.get_mesh("dp_cp").get_group()
        dist.all_reduce(global_scales, op=dist.ReduceOp.SUM, group=dp_cp_group)
        rl_scale, ce_scale, ref_kl_scale = (
            max(scale, 1) for scale in global_scales.tolist()
        )

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
            _local_max = max(
                (mb["input_ids"].shape[1] for mb in micro_batches), default=0
            )
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
            ref_logprobs = (
                micro_batch["ref_logprobs"].to(_dev)
                if micro_batch["ref_logprobs"] is not None
                else None
            )
            rl_weights = (
                micro_batch["rl_weights"].to(_dev)
                if micro_batch["rl_weights"] is not None
                else None
            )
            ce_weights = (
                micro_batch["ce_weights"].to(_dev)
                if micro_batch["ce_weights"] is not None
                else None
            )
            ref_kl_weights = (
                micro_batch["ref_kl_weights"].to(_dev)
                if micro_batch["ref_kl_weights"] is not None
                else None
            )
            routed_experts = (
                micro_batch["routed_experts"].to(_dev)
                if micro_batch["routed_experts"] is not None
                else None
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
            mm_kwargs = (
                {k: v.to("cuda") for k, v in mm_kwargs_raw.items()}
                if mm_kwargs_raw
                else None
            )
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
                defer_vlm_cp_to_model = (
                    mm_kwargs is not None and "image_grid_thw" in mm_kwargs
                )
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
                logprob_mask = shard_for_cp(
                    logprob_mask, cp_rank=cp_rank, cp_world_size=cp_size
                )
                if routed_experts is not None and not defer_vlm_cp_to_model:
                    routed_experts = shard_for_cp(
                        routed_experts, cp_rank=cp_rank, cp_world_size=cp_size
                    )

            labels = labels.masked_fill(~logprob_mask, -100)

            if config.model.lora:
                lora_num_tokens = micro_batch["lora_num_tokens"].to("cuda")
                if cp_enabled:
                    chunk_size = labels.shape[1]
                    # Convert to cumsum, adjust for CP chunk, convert back to num_tokens
                    cu_offsets = lora_num_tokens.cumsum(dim=0, dtype=torch.int32)
                    adjusted_cu = torch.clip(
                        cu_offsets - chunk_size * cp_rank, min=0, max=chunk_size
                    )
                    lora_num_tokens = torch.diff(
                        adjusted_cu,
                        prepend=torch.tensor(
                            [0], device=adjusted_cu.device, dtype=adjusted_cu.dtype
                        ),
                    )
                set_lora_num_tokens(lora_num_tokens)

            temperatures = micro_batch["temperatures"].to(_dev)

            # Shard temperatures for context parallelism if enabled
            if cp_enabled:
                temperatures = shard_for_cp(
                    temperatures, cp_rank=cp_rank, cp_world_size=cp_size
                )

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
                        export_micro_batch=micro_batch,
                        micro_step=micro_step,
                    )
                )
                continue

            # Forward pass with per-token temperatures
            with maybe_record_function("forward"), maybe_activation_offloading(
                config.model.ac_offloading
            ):
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
                assert (
                    out.get("logits") is not None
                ), "Logits must be provided to compute logprobs"
                logits = out["logits"]
                # Per-token temperature scaling: temperatures is [batch, seq], logits is [batch, seq, vocab]
                scaled_logits = logits / temperatures.unsqueeze(-1)
                out["logprobs"] = selective_log_softmax(scaled_logits, labels)
                out["entropy"] = compute_entropy(scaled_logits)
            # else: FusedOutputLinear was used - logprobs already computed with per-token temperatures

            if cp_enabled:
                out["logprobs"] = gather_for_cp(out["logprobs"], cp_group)
                out["entropy"] = gather_for_cp_wo_grad(
                    out["entropy"], cp_size, cp_group
                )

            vocab_size = (
                getattr(model.config, "vocab_size", None)
                or model.config.text_config.vocab_size
            )
            # This is not really necessary as the first token should be masked out, but we do it anyway to be sure
            out["logprobs"] = shift_tensor_right(
                out["logprobs"],
                pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item(),
            )
            out["entropy"] = shift_tensor_right(
                out["entropy"],
                pad_value=torch.log(torch.tensor(float(vocab_size))).item(),
            )

            # Compute loss
            sequence_lengths = micro_batch["sequence_lengths"]
            loss, loss_tensors = compute_loss(
                trainer_logprobs=out["logprobs"].squeeze().split(sequence_lengths),
                inference_logprobs=inference_logprobs.squeeze().split(sequence_lengths),
                ref_logprobs=ref_logprobs.squeeze().split(sequence_lengths)
                if ref_logprobs is not None
                else None,
                advantages=advantages.squeeze().split(sequence_lengths),
                loss_mask=loss_mask.squeeze().split(sequence_lengths),
                rl_weights=rl_weights.squeeze().split(sequence_lengths)
                if rl_weights is not None
                else None,
                ce_weights=ce_weights.squeeze().split(sequence_lengths)
                if ce_weights is not None
                else None,
                ref_kl_weights=ref_kl_weights.squeeze().split(sequence_lengths)
                if ref_kl_weights is not None
                else None,
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
            masked_env_names = [
                env_name
                for env_name, keep in zip(env_names, loss_mask.flatten().tolist())
                if keep
            ]
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
                sampled_mask = (
                    (rl_weights != 0) if rl_weights is not None else loss_mask
                )
                if ref_kl_weights is not None:
                    sampled_mask = sampled_mask | (ref_kl_weights != 0)
                mismatch_mask = loss_mask & sampled_mask
                has_mismatch_tokens = bool(mismatch_mask.any())
            if has_mismatch_tokens:
                with torch.no_grad():
                    _, _, mismatch_kl = compute_importance_ratio_and_mismatch_kl(
                        out["logprobs"], inference_logprobs
                    )
                mismatch_kl = mismatch_kl[mismatch_mask].detach().to("cpu")
                tensors["mismatch_kl/all"].append(mismatch_kl)
                mismatch_env_names = [
                    env_name
                    for env_name, keep in zip(
                        env_names, mismatch_mask.flatten().tolist()
                    )
                    if keep
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
                micro_step_message += (
                    f" | Mismatch KL {tensors['mismatch_kl/all'][-1].mean().item():.4f}"
                )
            if "max_vio" in tensors:
                micro_step_message += (
                    f" | Max Vio {tensors['max_vio'][-1].mean().item():.4f}"
                )
            if "routing_confidence" in tensors:
                micro_step_message += f" | Routing Conf. {tensors['routing_confidence'][-1].mean().item():.4f}"
            logger.debug(micro_step_message)

        if pp_enabled:
            # Loss for the last stage. Mirrors the non-PP path: temperature-scaled
            # logprobs, CP gather, right-shift, then the RL loss. Returns a scalar
            # so the schedule only backprops one value.
            direct_parity_dumped = False

            def _pp_loss(out, side):
                nonlocal direct_parity_dumped
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
                    o["entropy"] = gather_for_cp_wo_grad(
                        o["entropy"], cp_size, cp_group
                    )
                vocab_size = (
                    getattr(model.config, "vocab_size", None)
                    or model.config.text_config.vocab_size
                )
                o["logprobs"] = shift_tensor_right(
                    o["logprobs"],
                    pad_value=torch.log(torch.tensor(1.0 / vocab_size)).item(),
                )
                o["entropy"] = shift_tensor_right(
                    o["entropy"],
                    pad_value=torch.log(torch.tensor(float(vocab_size))).item(),
                )
                qualification_dump = (
                    config.data.fake
                    and config.data.fake.reference_path is not None
                )
                production_dump = (
                    not config.data.fake
                    and config.parity_guard is not None
                    and progress.step == 1
                )
                if (qualification_dump or production_dump) and not direct_parity_dumped:
                    full_input_ids = gather_for_cp_wo_grad(
                        side["input_ids"], cp_size, cp_group
                    )
                    full_position_ids = gather_for_cp_wo_grad(
                        side["position_ids"], cp_size, cp_group
                    )
                    dp_rank = parallel_dims.get_mesh("dp").get_local_rank()
                    if dp_rank == 0 and cp_rank == 0:
                        trainer_values = o["logprobs"].squeeze(0).detach().float().cpu()
                        reference_values = (
                            side["inference_logprobs"].squeeze(0).detach().float().cpu()
                        )
                        delta = trainer_values - reference_values
                        valid = side["loss_mask"].squeeze(0).detach().cpu()
                        if not valid.any():
                            raise RuntimeError("direct parity data has no retained tokens")
                        if not (
                            torch.isfinite(trainer_values).all()
                            and torch.isfinite(reference_values).all()
                            and torch.isfinite(delta[valid]).all()
                        ):
                            raise RuntimeError("direct parity data contains non-finite values")
                        payload = {
                            "input_ids": full_input_ids.squeeze(0).cpu().tolist(),
                            "position_ids": full_position_ids.squeeze(0).cpu().tolist(),
                            "sequence_lengths": side["sequence_lengths"],
                            "loss_mask": valid.tolist(),
                            "trainer_logprobs": trainer_values.tolist(),
                            "reference_logprobs": reference_values.tolist(),
                            "delta": delta.tolist(),
                            "valid_mean_abs_delta": float(delta[valid].abs().mean()),
                            "valid_max_abs_delta": float(delta[valid].abs().max()),
                        }
                        dump_path = config.output_dir / (
                            "direct-parity.json"
                            if qualification_dump
                            else "parity-step1.json"
                        )
                        dump_path.write_text(json.dumps(payload))
                        logger.info(
                            "Wrote direct trainer/reference parity data to %s "
                            "(mean abs delta %.6f, max %.6f)",
                            dump_path,
                            payload["valid_mean_abs_delta"],
                            payload["valid_max_abs_delta"],
                        )
                    direct_parity_dumped = True
                sl = side["sequence_lengths"]
                loss, loss_tensors = compute_loss(
                    trainer_logprobs=o["logprobs"].squeeze().split(sl),
                    inference_logprobs=side["inference_logprobs"].squeeze().split(sl),
                    ref_logprobs=side["ref_logprobs"].squeeze().split(sl)
                    if side["ref_logprobs"] is not None
                    else None,
                    advantages=side["advantages"].squeeze().split(sl),
                    loss_mask=side["loss_mask"].squeeze().split(sl),
                    rl_weights=side["rl_weights"].squeeze().split(sl)
                    if side["rl_weights"] is not None
                    else None,
                    ce_weights=side["ce_weights"].squeeze().split(sl)
                    if side["ce_weights"] is not None
                    else None,
                    ref_kl_weights=side["ref_kl_weights"].squeeze().split(sl)
                    if side["ref_kl_weights"] is not None
                    else None,
                    rl_loss_fn=rl_loss_fn,
                    rl_scale=rl_scale,
                    ce_scale=ce_scale,
                    ref_kl_scale=ref_kl_scale,
                )
                tensors["loss"].append(loss.detach().to("cpu").unsqueeze(0))
                tensors["entropy/all"].append(
                    o["entropy"][side["loss_mask"]].detach().to("cpu")
                )
                for k, v in loss_tensors.items():
                    tensors[k].append(v.detach().to("cpu"))
                token_exporter.export(
                    progress.step,
                    side["micro_step"],
                    side["export_micro_batch"],
                    o,
                    sl,
                    config.loss,
                )
                return loss

            # Stage buffers are fixed size, so every micro-batch must share one
            # sequence length; the pad is appended as a masked-out sequence.
            _n = torch.tensor([len(pp_side_inputs), -len(pp_side_inputs)], device="cuda")
            dist.all_reduce(_n, op=dist.ReduceOp.MAX)
            if _n[0].item() != -_n[1].item() or _n[0].item() == 0:
                raise RuntimeError(
                    f"micro-batch count mismatch across ranks (max={_n[0].item()}, "
                    f"min={-_n[1].item()}); the pipeline schedule would deadlock"
                )
            parity_bundle_readiness = _n
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
            _cache_ex = None
            if config.model.pp_transfer_sparse_indices:
                _cache_ex = torch.empty(
                    sparse_cache_transport_shape(_L, model.config.index_topk)
                    if pp_stage_module.receive_sparse_indices
                    else (1,),
                    # PipelineStage requires floating-point receive buffers.
                    # FP16 carries the lossless unsigned 16-bit index bits.
                    dtype=torch.float16,
                    device="cuda",
                )
            _is_last = pp_rank == parallel_dims.pp - 1
            _example_input = (
                torch.empty(1, _L, dtype=torch.int64, device="cuda")
                if pp_rank == 0
                else ((_hidden_ex, _cache_ex) if _cache_ex is not None else _hidden_ex)
            )
            _output_cache_ex = None
            if config.model.pp_transfer_sparse_indices and not _is_last:
                _output_cache_ex = torch.empty(
                    sparse_cache_transport_shape(_L, model.config.index_topk)
                    if pp_stage_module.send_sparse_indices
                    else (1,),
                    dtype=torch.float16,
                    device="cuda",
                )
            _example_output = (
                torch.empty((), dtype=torch.float32, device="cuda")
                if _is_last
                else (
                    (_hidden_ex, _output_cache_ex)
                    if _output_cache_ex is not None
                    else _hidden_ex
                )
            )
            pp_schedule = build_pp_schedule(
                pp_stage_module,
                pp_rank=pp_rank,
                pp_size=parallel_dims.pp,
                n_microbatches=len(pp_side_inputs),
                pp_group=parallel_dims.get_mesh("pp").get_group(),
                schedule_name=config.model.pp_schedule,
                example_input=_example_input,
                example_output=_example_output,
            )
            # FSDP must be initialised before the schedule issues its first
            # prefetch, which happens before any forward on some stages.
            force_fsdp_lazy_init(model)
            # Side inputs are CPU-staged; the schedule input itself must be CUDA.
            pp_inputs = torch.cat([sd["input_ids"] for sd in pp_side_inputs], dim=0).to(
                "cuda"
            )
            profile_pp_stage = os.environ.get("PRIME_PROFILE_PP_STAGE") == "1"
            if dist.get_rank() == 0 or profile_pp_stage:
                torch.cuda.synchronize()
                pp_start = time.perf_counter()
            # return_outputs=False: the schedule would otherwise concatenate the
            # last stage's per-micro-batch outputs, which are scalar losses --
            # "zero-dimensional tensor cannot be concatenated". The loss closure
            # already accumulates what the metrics need.
            profile_rank = int(os.environ.get("PRIME_PROFILE_PP_RANK", "-1"))
            schedule_profiler = (
                profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                if dist.get_rank() == profile_rank
                else nullcontext()
            )
            with schedule_profiler:
                if pp_rank == 0:
                    pp_schedule.step(pp_inputs, return_outputs=False)
                elif pp_rank == parallel_dims.pp - 1:
                    # The schedule indexes target_mbs unconditionally on the last
                    # stage; the loss closure ignores the value, so a zero per
                    # micro-batch is enough to satisfy it.
                    pp_schedule.step(
                        target=torch.zeros(len(pp_side_inputs), device="cuda"),
                        return_outputs=False,
                    )
                else:
                    pp_schedule.step(return_outputs=False)
            if dist.get_rank() == profile_rank:
                logger.info(
                    "PP CUDA profile:\n"
                    + schedule_profiler.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=40
                    )
                )
            local_schedule_ms = 0.0
            if dist.get_rank() == 0 or profile_pp_stage:
                torch.cuda.synchronize()
                local_schedule_ms = (time.perf_counter() - pp_start) * 1000.0
            if dist.get_rank() == 0:
                logger.info(
                    f"PP step: {len(pp_side_inputs)} micro-batches in {local_schedule_ms / 1000.0:.2f}s"
                )
            if profile_pp_stage:
                forward_ms, backward_ms = pp_stage_module.profile_times_ms(clear=True)
                local_profile = torch.tensor(
                    [float(pp_rank), local_schedule_ms, forward_ms, backward_ms],
                    dtype=torch.float64,
                    device="cuda",
                )
                gathered_profiles = [torch.empty_like(local_profile) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_profiles, local_profile)
                if dist.get_rank() == 0:
                    stage_profiles = []
                    for stage in range(parallel_dims.pp):
                        rows = [row.tolist() for row in gathered_profiles if int(row[0].item()) == stage]
                        stage_profiles.append(
                            (
                                stage,
                                max(row[1] for row in rows),
                                max(row[2] for row in rows),
                                max(row[3] for row in rows),
                            )
                        )
                    logger.info(
                        "PP stage profile ms (stage, schedule, forward, backward): "
                        f"{stage_profiles}"
                    )
            # Free this step's pipeline state NOW. The side inputs (including
            # routed_experts, tens of MB per micro-batch) and the stage's
            # per-micro-batch recv/grad buffers otherwise survive through the
            # optimizer step and overlap with the NEXT batch's allocations --
            # a multi-GB transient that ran the 47.4 GiB cards to the edge.
            pp_stage_module.set_side_inputs([])
            pp_stage_module.loss_closure = None
            del pp_schedule, pp_inputs, _hidden_ex, _cache_ex, _output_cache_ex, pp_side_inputs
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
            publish_trainer_bundle(
                token_exporter.parity_source(),
                parity_bundle_enabled,
                parity_bundle_communication,
                parity_bundle_readiness,
            )
            ready_run_ids = {
                multi_run_manager.idx_2_id[idx]
                for idx in multi_run_manager.ready_to_update_idxs
                if idx in multi_run_manager.idx_2_id
            }
            token_exporter.mark_stable(ready_run_ids)

        # Synchronize forward/loss diagnostics before validating gradients.
        # A rejected update must still preserve its measured parity evidence.
        tensor_stats = tensors.compute_stats()

        if config.parity_guard is not None:
            violations = parity_guard_failures(tensor_stats, config.parity_guard)
            if violations:
                monitor.log(
                    {
                        **tensor_stats,
                        "diagnostic/parity_guard_rejected": 1.0,
                        "step": progress.step,
                    },
                    step=progress.step,
                )
                optimizer.zero_grad()
                raise RuntimeError(
                    "Parity guard rejected step before optimizer update and weight broadcast: "
                    + "; ".join(violations)
                )

        # compute_loss already divided by the global token count. Undo FSDP's per-rank averaging
        # across dp_cp so the final gradient is the true per-token mean over the global batch.
        found_nonfinite = torch.zeros((), device="cuda", dtype=torch.int32)
        nonfinite_grad_names: list[str] = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad.mul_(parallel_dims.fsdp_gradient_divide_factor)
                grad = (
                    param.grad.to_local()
                    if hasattr(param.grad, "to_local")
                    else param.grad
                )
                is_nonfinite = (~torch.isfinite(grad).all()).to(dtype=torch.int32)
                found_nonfinite = torch.maximum(found_nonfinite, is_nonfinite)
                if is_nonfinite.item():
                    nonfinite_grad_names.append(name)
        if nonfinite_grad_names:
            logger.error(
                "Non-finite local gradients at step {}: {}",
                progress.step,
                nonfinite_grad_names,
            )

        # Do not silently replace NaN/Inf values with zero. That changes the
        # optimizer direction and made the later finite-norm guard ineffective.
        # All ranks must make the same decision before any rank enters the
        # collective gradient clipping path.
        dist.all_reduce(found_nonfinite, op=dist.ReduceOp.MAX)
        if found_nonfinite.item():
            rejected_metrics = {
                **tensor_stats,
                "diagnostic/nonfinite_gradient": 1.0,
                "step": progress.step,
            }
            monitor.log(rejected_metrics, step=progress.step)
            logger.error(
                "Rejected-step forward diagnostics at step {}: entropy={} masked_fraction={} "
                "masked_mismatch_kl={} unmasked_mismatch_kl={}",
                progress.step,
                tensor_stats.get("entropy/all/mean"),
                tensor_stats.get("is_masked/mean"),
                tensor_stats.get("masked_mismatch_kl/mean"),
                tensor_stats.get("unmasked_mismatch_kl/mean"),
            )
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

        if (
            pp_enabled
            and dist.get_rank() % (dist.get_world_size() // parallel_dims.pp) == 0
        ):
            import sys

            from torch.distributed.tensor import DTensor as _DT

            stage_gradient_l1 = 0.0
            parameters_with_gradient = 0
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                parameters_with_gradient += 1
                gradient = (
                    parameter.grad.to_local()
                    if isinstance(parameter.grad, _DT)
                    else parameter.grad
                )
                stage_gradient_l1 += float(gradient.abs().sum())
            print(
                f"[grad-check] step={progress.step} stage={pp_rank} "
                f"lora_grad_l1={stage_gradient_l1:.6e} "
                f"params_with_grad={parameters_with_gradient}",
                file=sys.stderr,
                flush=True,
            )

        zero_grad_ratio = get_zero_gradient_ratio(model.parameters(), parallel_dims.dp_replicate)

        # Update the model parameters
        optimizer.step()
        optimizer.zero_grad()

        # Update learning rate scheduler
        scheduler.step()

        if os.environ.get("PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD", "") == "1":
            torch.cuda.synchronize()
            host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
            if host_empty_cache is None:
                raise RuntimeError(
                    "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD requires torch._C._host_emptyCache "
                    "to bound pinned host memory across steps"
                )
            host_empty_cache()

        if config.max_concurrent_runs == 1:
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.get_current_lr()
        forward_backward_time = time.perf_counter() - forward_backward_start_time
        if qualification_identity is not None:
            maximum_forward_backward = torch.tensor(
                forward_backward_time, dtype=torch.float64, device="cuda"
            )
            dist.all_reduce(maximum_forward_backward, op=dist.ReduceOp.MAX)
            forward_backward_time = float(maximum_forward_backward.item())

        # Broadcast the model just produced (policy v{progress.step}) so the orchestrator can
        # sample its next step from it. In-memory transports retain their two-step shutdown
        # window; filesystem broadcast still writes every version for resume.
        if trainer_only and export_final_adapter and is_last_step:
            for idx in multi_run_manager.used_idxs:
                multi_run_manager.ready_to_update[idx] = True
                multi_run_manager.progress[idx].step = progress.step + 1
        if weight_broadcast is None or (trainer_only and not is_last_step):
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
                broadcast_weights_time = (
                    time.perf_counter() - broadcast_weights_start_time
                )
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
                ckpt_manager.save(
                    progress.step, model, [optimizer], scheduler, progress
                )
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

        # Forward/loss tensor metrics were synchronized before gradient
        # validation so rejected updates preserve the same parity evidence.

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
        throughput = perf_counter.get_step_tokens_per_second(
            num_tokens, forward_backward_time
        )
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
            step_message += (
                f" | Routing Conf. {tensor_stats['routing_confidence/mean']:.4f}"
            )
        logger.success(step_message)

        # Log performance metrics
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/throughput_per_gpu": throughput / world.world_size,
            "perf/mfu": mfu,
            "perf/peak_memory": peak_memory,
            "data/rollout_rows": rollout_rows,
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
        monitor.log(
            filter_rl_trainer_tensor_stats_for_wandb(tensor_stats), step=progress.step
        )

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
                    lr = (
                        optimizer.get_current_lr(idx)
                        if optimizer.optimizers[idx]
                        else 0.0
                    )
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

    logger.info(
        f"Peak memory: {max(to_col_format(monitor.history)['perf/peak_memory']):.1f} GiB"
    )
    if qualification_identity is not None:
        dist.barrier()
        qualification_failed = torch.zeros((), dtype=torch.int32, device="cuda")
        if world.is_master:
            try:
                adapter_path = None
                if export_final_adapter:
                    adapter_path = (
                        get_step_path(
                            get_broadcast_dir(multi_run_manager.get_run_dir(0)),
                            progress.step,
                        )
                        / "adapter_model.safetensors"
                    )
                _write_qualification_result(
                    config,
                    parallel_dims,
                    monitor,
                    progress,
                    qualification_identity,
                    qualification_runtime_records,
                    qualification_input_sha256,
                    adapter_path,
                )
            except Exception:
                logger.exception("qualification result write failed")
                qualification_failed.fill_(1)
        dist.broadcast(qualification_failed, src=0)
        if qualification_failed.item():
            raise RuntimeError("qualification result write failed")

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
