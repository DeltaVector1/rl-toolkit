"""WeightWatcher: polls the broadcast dir, advances ``Policy``, notifies
observers (dispatcher → off-policy cancel). Standalone async task; the
orchestrator's barrier bounds the in-flight lead."""

from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

import httpx
from modelexpress import p2p_pb2

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import (
    InferencePool,
    _validate_loaded_receipt,
    adapter_stage_limit_seconds,
    common_loaded_lora_receipt,
    load_lora_adapter,
    model_is_visible,
    stage_full_checkpoint,
    stage_lora_adapter,
    remove_lora_adapter,
)
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import get_latest_ckpt_step
from prime_rl.utils.checkpoint_delta import build_xor_delta_checkpoint


v1_adapter_transport = "chunked-shuffled-xor-zstd"
v2_adapter_transport = "chunked-bf16-linear-zigzag-sparse-zstd"
adapter_manifest_max_bytes = 2 * 1024 * 1024
adapter_deadline_header = "X-Adapter-Deadline-Seconds"


def _receiver_id(client) -> str:
    return str(client.base_url).rstrip("/").removesuffix("/v1")


def _ack_identity(value):
    if not isinstance(value, dict):
        return value
    identity = dict(value)
    identity.pop("slot", None)
    receivers = identity.get("receivers")
    if isinstance(receivers, list):
        identity["receivers"] = []
        for item in receivers:
            if not isinstance(item, dict):
                identity["receivers"].append(item)
                continue
            stable_item = dict(item)
            receipt = stable_item.get("receipt")
            if isinstance(receipt, dict):
                stable_receipt = dict(receipt)
                stable_receipt.pop("server_instance", None)
                stable_receipt.pop("lora_name", None)
                stable_item["receipt"] = stable_receipt
            identity["receivers"].append(stable_item)
    return identity


def _write_receivers_ready(
    step_path,
    *,
    run_name: str,
    step: int,
    source_url: str,
    slot: str,
    adapter_identity: dict[str, object],
    admin_clients,
    receipts,
) -> None:
    if not step_path.is_dir() or not (step_path / "STABLE").is_file():
        raise RuntimeError(f"cannot publish receiver ACK for unpublished adapter step {step}")
    if not admin_clients or not isinstance(receipts, list) or len(receipts) != len(admin_clients):
        raise RuntimeError("receiver ACK requires one successful receipt per configured endpoint")

    endpoint_ids = [_receiver_id(client) for client in admin_clients]
    if len(set(endpoint_ids)) != len(endpoint_ids):
        raise RuntimeError("receiver ACK has duplicate endpoint")

    records = []
    for endpoint, receipt in zip(endpoint_ids, receipts):
        validated = _validate_loaded_receipt(
            receipt,
            source_url=source_url,
            lora_name=slot,
            expected=adapter_identity,
        )
        records.append({"endpoint": endpoint, "receipt": validated})
    records.sort(key=lambda item: item["endpoint"])
    payload = {
        "version": 1,
        "run_name": run_name,
        "step": step,
        "source_url": source_url,
        "target_sha256": adapter_identity["target_sha256"],
        "target_size": adapter_identity["target_size"],
        "config_sha256": adapter_identity["config_sha256"],
        "config_size": adapter_identity["config_size"],
        "slot": slot,
        "receivers": records,
    }
    marker = step_path / "receivers-ready.json"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"receiver ACK is invalid at {marker}") from exc
        if _ack_identity(existing) != _ack_identity(payload):
            raise RuntimeError(f"receiver ACK is immutable and differs at {marker}")
        return
    temporary = None
    try:
        output = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=step_path,
            prefix=f".{marker.name}.",
            suffix=".tmp",
            delete=False,
        )
        temporary = output.name
        with output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, marker)
        directory = os.open(step_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            os.unlink(temporary) if os.path.exists(temporary) else None


def _read_receivers_ready(
    step_path,
    *,
    run_name: str,
    step: int,
    source_url: str,
    slot: str,
    admin_clients,
) -> dict[str, object]:
    marker = step_path / "receivers-ready.json"
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"receiver ACK is missing for adapter step {step}: {marker}")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"receiver ACK is invalid for adapter step {step}: {marker}") from exc
    required = {
        "version",
        "run_name",
        "step",
        "source_url",
        "target_sha256",
        "target_size",
        "config_sha256",
        "config_size",
        "slot",
        "receivers",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RuntimeError(f"receiver ACK fields are invalid for adapter step {step}: {marker}")
    if (
        type(payload["version"]) is not int
        or payload["version"] != 1
        or not isinstance(payload["run_name"], str)
        or payload["run_name"] != run_name
        or type(payload["step"]) is not int
        or payload["step"] != step
        or not isinstance(payload["source_url"], str)
        or payload["source_url"] != source_url
        or not _valid_digest(payload["target_sha256"])
        or type(payload["target_size"]) is not int
        or payload["target_size"] <= 0
        or not _valid_digest(payload["config_sha256"])
        or type(payload["config_size"]) is not int
        or payload["config_size"] < 0
        or not isinstance(payload["slot"], str)
        or payload["slot"] != slot
    ):
        raise RuntimeError(f"receiver ACK is stale for adapter step {step}: {marker}")
    receivers = payload["receivers"]
    expected_endpoints = sorted(_receiver_id(client) for client in admin_clients)
    if (
        len(expected_endpoints) != len(set(expected_endpoints))
        or not isinstance(receivers, list)
        or len(receivers) != len(expected_endpoints)
    ):
        raise RuntimeError(f"receiver ACK records are invalid for adapter step {step}: {marker}")
    identity = {
        name: payload[name]
        for name in ("source_url", "target_sha256", "target_size", "config_sha256", "config_size")
    }
    actual_endpoints = []
    for item in receivers:
        if not isinstance(item, dict) or set(item) != {"endpoint", "receipt"}:
            raise RuntimeError(f"receiver ACK records are invalid for adapter step {step}: {marker}")
        endpoint = item["endpoint"]
        if not isinstance(endpoint, str) or endpoint in actual_endpoints or endpoint not in expected_endpoints:
            raise RuntimeError(f"receiver ACK endpoint set is invalid for adapter step {step}: {marker}")
        if not isinstance(item["receipt"], dict):
            raise RuntimeError(f"receiver ACK receipt is invalid for adapter step {step}: {marker}")
        _validate_loaded_receipt(
            item["receipt"],
            source_url=source_url,
            lora_name=slot,
            expected=identity,
        )
        actual_endpoints.append(endpoint)
    if set(actual_endpoints) != set(expected_endpoints):
        raise RuntimeError(f"receiver ACK endpoint set differs for adapter step {step}: {marker}")
    return payload


def _validate_receipt_against_ack(
    receipt: dict[str, object] | None,
    ack: dict[str, object],
    *,
    step: int,
) -> None:
    if receipt is None:
        raise RuntimeError(f"cannot reconcile policy v{step}: no common loaded adapter receipt")
    if (
        not isinstance(receipt.get("lora_name"), str)
        or not isinstance(ack.get("slot"), str)
        or receipt["lora_name"] != ack["slot"]
    ):
        raise RuntimeError(f"cannot reconcile policy v{step}: loaded adapter slot differs from receiver ACK")
    for name in (
        "source_url",
        "target_sha256",
        "target_size",
        "config_sha256",
        "config_size",
    ):
        actual = receipt.get(name)
        expected = ack.get(name)
        if name.endswith("_size"):
            matches = (
                type(actual) is int
                and type(expected) is int
                and actual == expected
                and (actual > 0 if name == "target_size" else actual >= 0)
            )
        else:
            matches = isinstance(actual, str) and isinstance(expected, str) and actual == expected
        if not matches:
            raise RuntimeError(
                f"cannot reconcile policy v{step}: loaded adapter {name} differs from receiver ACK"
            )


async def _run_with_retries(
    name: str,
    operation: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 3,
) -> Any:
    if type(attempts) is not int or attempts <= 0:
        raise ValueError("retry attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except asyncio.CancelledError:
            raise
        except Exception:
            if attempt == attempts:
                get_logger().exception(
                    f"{name} failed on attempt {attempt}/{attempts}; stopping"
                )
                raise
            delay = 0.5 * 2 ** (attempt - 1)
            get_logger().exception(
                f"{name} failed on attempt {attempt}/{attempts}; retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"{name} retry loop ended unexpectedly after {attempts} attempts")


async def _stage_policy_adapter(
    admin_clients,
    weights_path: str,
    *,
    policy_step: int,
    delta_deadline: float,
    origin_base_urls,
) -> dict[str, object]:
    # The first deadline includes manifest preparation. Backoff and response
    # cleanup consume the same fixed 900s budget; each stage still has <=600s.
    total_deadline = delta_deadline + 300.0
    for attempt in range(1, 3):
        started = time.monotonic()
        attempt_deadline = min(
            delta_deadline if attempt == 1 else total_deadline,
            started + adapter_stage_limit_seconds(policy_step),
        )
        if attempt_deadline <= started:
            raise TimeoutError("adapter policy staging deadline expired")
        try:
            result = await stage_lora_adapter(
                admin_clients, weights_path, require_delta=True,
                policy_step=policy_step, delta_deadline=attempt_deadline,
                origin_base_urls=origin_base_urls,
            )
        except (TimeoutError, httpx.HTTPStatusError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    error = exc.response.json()
                except ValueError:
                    raise exc
                if (
                    exc.response.status_code != 504 or not isinstance(error, dict)
                    or error.get("error_type") != "TimeoutError"
                    or error.get("phase") != "staging"
                ):
                    raise
            remaining = max(0.0, total_deadline - time.monotonic())
            get_logger().warning(
                f"Adapter v{policy_step} stage attempt {attempt}/2 failed: {exc!r}; "
                f"elapsed={time.monotonic() - started:.3f}s, "
                "preserved_partial=receiver-verified-only, bytes=not-reported, "
                f"remaining_budget={remaining:.3f}s"
            )
            if attempt == 2 or remaining <= 0:
                raise
            # Reuse the exact URL and policy. Receivers revalidate the stored
            # transport identity and each preserved chunk before reuse.
            await asyncio.sleep(min(0.5, max(0.0, total_deadline - time.monotonic())))
        else:
            get_logger().info(
                f"Adapter v{policy_step} stage attempt {attempt}/2 succeeded; "
                f"elapsed={time.monotonic() - started:.3f}s, "
                f"preserved_partial={'receiver-verified-only' if attempt == 2 else 'not-used'}, "
                f"bytes={result.get('bytes_done', 'not-reported')}, "
                f"remaining_budget={max(0.0, total_deadline - time.monotonic()):.3f}s"
            )
            return result
    raise RuntimeError("adapter policy staging retry loop ended unexpectedly")


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _adapter_file_records(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("adapter manifest has no file records")
    files: dict[str, dict[str, object]] = {}
    for item in raw_files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError("adapter manifest has an invalid file record")
        name = item["name"]
        if name in files or name not in {"adapter_config.json", "adapter_model.safetensors"}:
            raise RuntimeError("adapter manifest has an invalid file set")
        size = item.get("size")
        digest = item.get("sha256")
        if type(size) is not int or size < 0 or not _valid_digest(digest):
            raise RuntimeError("adapter manifest has an invalid file identity")
        files[name] = item
    if set(files) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise RuntimeError("adapter manifest has an incomplete file set")
    return files


def validate_adapter_manifest(
    manifest: dict[str, object],
    *,
    target_step: int,
    target_url: str | None = None,
    base_url: str | None = None,
    base2_url: str | None = None,
) -> dict[str, object]:
    """Validate the exact adjacent transport selected by a policy update."""
    if manifest.get("kind") != "lora-adapter" or manifest.get("mode") != "delta":
        raise RuntimeError("adapter manifest is not a delta adapter")
    if type(manifest.get("target_step")) is not int or manifest["target_step"] != target_step:
        raise RuntimeError("adapter manifest target step is not the requested policy")
    version = manifest.get("version")
    if version == 1:
        if manifest.get("transport") != v1_adapter_transport:
            raise RuntimeError("adapter manifest selects an unsupported version-one transport")
        if manifest.get("base_step") != target_step - 1:
            raise RuntimeError("adapter manifest base step is not adjacent to its target")
    elif version == 2:
        if manifest.get("transport") != v2_adapter_transport:
            raise RuntimeError("adapter manifest selects an unsupported predictive transport")
        if target_step < 2:
            raise RuntimeError("predictive adapter transport requires two predecessor steps")
        if manifest.get("base_step") != target_step - 1 or manifest.get("base2_step") != target_step - 2:
            raise RuntimeError("predictive adapter manifest steps are not adjacent")
        for prefix in ("target", "base", "base2"):
            size = manifest.get(f"{prefix}_size")
            digest = manifest.get(f"{prefix}_sha256")
            if type(size) is not int or size <= 0 or not _valid_digest(digest):
                raise RuntimeError(f"predictive adapter manifest has an invalid {prefix} identity")
        for name in ("config_sha256", "base_config_sha256", "base2_config_sha256"):
            if not _valid_digest(manifest.get(name)):
                raise RuntimeError(f"predictive adapter manifest has an invalid {name}")
        for name in ("config_size", "base_config_size", "base2_config_size"):
            value = manifest.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"predictive adapter manifest has an invalid {name}")
        data_offset = manifest.get("data_offset")
        if type(data_offset) is not int or data_offset < 0 or data_offset % 2:
            raise RuntimeError("predictive adapter manifest has an invalid data offset")
    else:
        raise RuntimeError("adapter manifest version is unsupported")

    if version == 2:
        if not isinstance(target_url, str) or manifest.get("target_url") != target_url:
            raise RuntimeError("predictive adapter manifest target URL differs from the requested policy")
        if not isinstance(base_url, str) or manifest.get("base_url") != base_url:
            raise RuntimeError("predictive adapter manifest base URL differs from the adjacent policy")
        if not isinstance(base2_url, str) or manifest.get("base2_url") != base2_url:
            raise RuntimeError("predictive adapter manifest second base URL differs from the adjacent policy")
    if target_url is not None and manifest.get("target_url") not in (None, target_url):
        raise RuntimeError("adapter manifest target URL differs from the requested policy")
    if base_url is not None and manifest.get("base_url") not in (None, base_url):
        raise RuntimeError("adapter manifest base URL differs from the adjacent policy")
    if version == 2:
        files = _adapter_file_records(manifest)
        if manifest["target_size"] != files["adapter_model.safetensors"]["size"]:
            raise RuntimeError("predictive adapter target size differs from its file record")
        if manifest["target_sha256"] != files["adapter_model.safetensors"]["sha256"]:
            raise RuntimeError("predictive adapter target digest differs from its file record")
        if manifest["config_size"] != files["adapter_config.json"]["size"]:
            raise RuntimeError("predictive adapter config size differs from its file record")
        if manifest["config_sha256"] != files["adapter_config.json"]["sha256"]:
            raise RuntimeError("predictive adapter config digest differs from its file record")
        if manifest["base_size"] != manifest["target_size"] or manifest["base2_size"] != manifest["target_size"]:
            raise RuntimeError("predictive adapter predecessor sizes differ from the target")
    return manifest


def inference_weight_path(config: OrchestratorConfig, weights_path, lora_name: str | None):
    broadcast = config.weight_broadcast
    if not (broadcast.type == "filesystem" and broadcast.source_url):
        return weights_path
    if broadcast.source_dir is None:
        raise ValueError("weight_broadcast.source_dir is required with source_url")
    return (
        f"{broadcast.source_url.rstrip('/')}/"
        f"{quote(weights_path.relative_to(broadcast.source_dir).as_posix())}"
    )


def _read_adapter_manifest(
    path,
    source_url: str | None,
    timeout: float | int | None,
    *,
    prefer_remote: bool = False,
) -> tuple[dict[str, object], str]:
    """Read a published adapter manifest from local storage or its source URL."""
    manifest_path = path / "adapter_manifest.json"
    if source_url is None or (manifest_path.is_file() and not prefer_remote):
        try:
            with manifest_path.open("rb") as source:
                payload = source.read(adapter_manifest_max_bytes + 1)
            if len(payload) > adapter_manifest_max_bytes:
                raise RuntimeError("published adapter manifest exceeds the size limit")
            manifest = json.loads(payload)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError("published adapter manifest is unavailable") from exc
        location = manifest_path.as_posix()
    else:
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise TimeoutError("published adapter manifest deadline expired")
        return asyncio.run(
            _read_remote_adapter_manifest(
                source_url, time.monotonic() + min(float(timeout), 600.0)
            )
        )
    if not isinstance(manifest, dict):
        raise RuntimeError("published adapter manifest is not an object")
    return manifest, location


async def _read_remote_adapter_manifest(
    source_url: str,
    deadline: float,
) -> tuple[dict[str, object], str]:
    """Read the source manifest within one cancellable absolute deadline."""
    remaining = min(deadline - time.monotonic(), 600.0)
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("published adapter manifest deadline expired")
    location = f"{source_url.rstrip('/')}/adapter_manifest.json"
    headers = {adapter_deadline_header: f"{remaining:.6f}"}
    try:
        async with asyncio.timeout(remaining):
            async with httpx.AsyncClient(timeout=httpx.Timeout(remaining)) as client:
                async with client.stream("GET", location, headers=headers) as response:
                    response.raise_for_status()
                    declared_size = response.headers.get("Content-Length")
                    if declared_size is not None:
                        try:
                            parsed_size = int(declared_size)
                        except ValueError as exc:
                            raise RuntimeError(
                                "published adapter manifest has an invalid content length"
                            ) from exc
                        if parsed_size < 0 or parsed_size > adapter_manifest_max_bytes:
                            raise RuntimeError(
                                "published adapter manifest exceeds the size limit"
                            )
                    payload = bytearray()
                    async for chunk in response.aiter_raw():
                        payload.extend(chunk)
                        if len(payload) > adapter_manifest_max_bytes:
                            raise RuntimeError(
                                "published adapter manifest exceeds the size limit"
                            )
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise TimeoutError("published adapter manifest deadline expired") from exc
    except (httpx.HTTPError, OSError) as exc:
        raise RuntimeError(
            "published adapter manifest is unavailable from the adapter source"
        ) from exc
    try:
        manifest = json.loads(payload)
    except ValueError as exc:
        raise RuntimeError(
            "published adapter manifest is unavailable from the adapter source"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("published adapter manifest is not an object")
    return manifest, location


class WeightWatcher:
    """``await watcher.start()`` to drive the polling loop until ``stop()``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        policy: Policy,
        inference: InferencePool,
        observers: list[VersionObserver],
        lora_name: str | None,
        ckpt_step: int = 0,
        poll_interval: float = 1.0,
        reconcile_interval: float = 30.0,
        model_express: ModelExpressSession | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.inference = inference
        self.observers = observers
        self.lora_name = lora_name
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval
        self.reconcile_interval = reconcile_interval
        self.model_express = model_express

        self.last_update_weights_time: float = 0.0
        self.last_stage_weights_time: float = 0.0
        self.last_adapter_delta_build_time: float = -1.0
        self.last_adapter_source_prepare_time: float = 0.0
        self.last_adapter_wire_stage_time: float = 0.0
        self.last_adapter_reconstruct_time: float = -1.0
        self.last_adapter_load_time: float = 0.0
        self.last_adapter_pause_time: float = 0.0
        self.last_adapter_resume_time: float = 0.0
        self.last_adapter_receipt_time: float = 0.0
        self.last_adapter_policy_step: int = 0
        self.last_adapter_stage_limit_seconds: float = 0.0
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0
        self.reconcile_count: int = 0
        self.last_reconcile_time: float = 0.0
        self.last_reconcile_check_at: float = 0.0
        self.current_weights_path = None

        # A delta is safe only when the inference pool has confirmed the
        # immediately previous adapter. A restarted watcher may trust the
        # startup load, but it must never skip an adapter version.
        self.verified_adapter_step: int | None = ckpt_step if lora_name is not None else None
        self.verified_adapter_url: str | None = None
        self.bootstrap_staged = lora_name is None or ckpt_step != 0 or model_express is not None

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                operation_name = "Initial adapter staging"
                policy_update_started = False
                try:
                    if not self.bootstrap_staged:
                        await _run_with_retries(operation_name, self.stage_bootstrap_adapter)
                        if not self.bootstrap_staged:
                            await asyncio.sleep(self.poll_interval)
                            continue
                    next_step = self.compute_next_ckpt_step()
                    if next_step > self.ckpt_step:
                        operation_name = f"Policy update {next_step}"
                        policy_update_started = True
                        await self.apply_policy_update(next_step)
                    elif time.monotonic() - self.last_reconcile_check_at >= self.reconcile_interval:
                        operation_name = "Current policy reconciliation"
                        self.last_reconcile_check_at = time.monotonic()
                        reconcile_attempts = (
                            1 if self.lora_name is not None and self.ckpt_step > 0 else 3
                        )
                        await _run_with_retries(
                            operation_name,
                            self.reconcile_current_policy,
                            attempts=reconcile_attempts,
                        )
                except httpx.TransportError as exc:
                    if policy_update_started:
                        # Never turn a failed stage/load into another apply.
                        raise
                    # Wait for bounded engine health after an endpoint transport failure.
                    get_logger().warning(
                        f"{operation_name} saw an inference endpoint outage ({exc!r}); "
                        "waiting for configured inference health before retrying"
                    )
                    await self.inference.wait_for_ready(self.config.model.name)
                    self.last_reconcile_check_at = 0.0
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def stage_bootstrap_adapter(self) -> None:
        """Stage policy zero without loading it or advancing policy state."""
        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        local_path = get_step_path(broadcast_dir, 0)
        if not (local_path / "STABLE").is_file():
            return
        source_path = inference_weight_path(self.config, local_path, self.lora_name)
        if isinstance(source_path, str):
            started = time.perf_counter()
            result = await stage_lora_adapter(
                self.inference.admin_clients,
                source_path,
                require_delta=False,
                origin_base_urls=self.inference.adapter_origin_base_urls,
            )
            transfer_seconds = result.get("transfer_wall_seconds")
            self.last_adapter_wire_stage_time = (
                float(transfer_seconds)
                if isinstance(transfer_seconds, (int, float)) and transfer_seconds >= 0
                else -1.0
            )
            origin_stage = format_time(float(result.get("origin_stage_seconds", 0.0)))
            peer_stage = format_time(float(result.get("peer_stage_seconds", 0.0)))
            reconstruct_seconds = result.get(
                "reconstruction_wall_seconds", result.get("reconstruct_seconds")
            )
            self.last_adapter_reconstruct_time = (
                float(reconstruct_seconds) if reconstruct_seconds is not None else -1.0
            )
            self.last_stage_weights_time = time.perf_counter() - started
            get_logger().info(
                f"Staged initial adapter policy v0 on every inference host in "
                f"{format_time(self.last_stage_weights_time)} "
                f"(origin {origin_stage}, peer {peer_stage})"
            )
        self.current_weights_path = source_path
        self.verified_adapter_step = 0
        self.verified_adapter_url = str(source_path)
        self.bootstrap_staged = True

    async def stop(self) -> None:
        self.stopped.set()
        # Let an in-flight apply finish before dying: the trainer blocks inside
        # its in-memory broadcast until the apply completes, so cancelling
        # mid-apply would strand it. The orchestrator's global teardown budget
        # bounds this wait.
        async with self.update_lock:
            pass
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    def compute_next_ckpt_step(self) -> int:
        """Return the next policy version exposed by the configured transport."""
        if self.model_express is not None:
            if self.config.max_steps is not None and self.ckpt_step >= self.config.max_steps - 2:
                return self.ckpt_step
            # ModelExpress status changes are unversioned, so its rendezvous advances
            # exactly one policy version per READY/INITIALIZING cycle.
            return self.ckpt_step + 1

        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        latest_ckpt_step = get_latest_ckpt_step(broadcast_dir) or 0
        if self.lora_name is not None:
            # Delta manifests are adjacent-step objects. Do not jump to the
            # newest checkpoint: that would apply a delta against the wrong
            # base after a delayed or restarted watcher.
            if latest_ckpt_step > self.ckpt_step:
                return self.ckpt_step + 1
            return self.ckpt_step
        return max(self.policy.version, latest_ckpt_step)

    def _adapter_delta_base(
        self,
        next_step: int,
        local_weights_path,
        target_url: str | None = None,
        deadline: float | None = None,
        prepared_manifest: tuple[dict[str, object], str] | None = None,
    ) -> str:
        """Return the verified adjacent base URL and published delta manifest."""
        if self.lora_name is None or next_step <= 0:
            raise RuntimeError("an adapter delta requires a positive LoRA policy step")
        if self.verified_adapter_step != next_step - 1:
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: verified base is "
                f"v{self.verified_adapter_step}, expected v{next_step - 1}"
            )

        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        previous_path = get_step_path(broadcast_dir, next_step - 1)
        if not (previous_path / "STABLE").is_file():
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: previous policy v{next_step - 1} "
                "is not marked STABLE"
            )
        previous_url = inference_weight_path(self.config, previous_path, self.lora_name)
        if not isinstance(previous_url, str):
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: previous policy has no source URL"
            )
        if self.verified_adapter_url is not None and previous_url != self.verified_adapter_url:
            raise RuntimeError(
                f"cannot apply adapter policy v{next_step}: verified base URL changed"
            )

        previous_previous_url = None
        if next_step >= 2:
            previous_previous_path = get_step_path(broadcast_dir, next_step - 2)
            if not (previous_previous_path / "STABLE").is_file():
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: previous policy v{next_step - 2} "
                    "is not marked STABLE"
                )
            previous_previous_url = inference_weight_path(
                self.config, previous_previous_path, self.lora_name
            )
            if not isinstance(previous_previous_url, str):
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: second base has no source URL"
                )

        if target_url is None:
            target_url = inference_weight_path(self.config, local_weights_path, self.lora_name)
        manifest_timeout = adapter_stage_limit_seconds(next_step)
        if deadline is not None:
            manifest_timeout = deadline - time.monotonic()
            if manifest_timeout <= 0:
                raise TimeoutError(
                    f"cannot apply adapter policy v{next_step}: staging deadline expired"
                )
        if prepared_manifest is None:
            try:
                manifest, manifest_location = _read_adapter_manifest(
                    local_weights_path,
                    target_url if isinstance(target_url, str) else None,
                    manifest_timeout,
                    prefer_remote=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: {exc}"
                ) from exc
        else:
            manifest, manifest_location = prepared_manifest
        if manifest.get("version") != 1:
            try:
                validate_adapter_manifest(
                    manifest,
                    target_step=next_step,
                    target_url=target_url if isinstance(target_url, str) else None,
                    base_url=previous_url,
                    base2_url=previous_previous_url,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: {exc}"
                ) from exc
        else:
            if (
                manifest.get("kind") != "lora-adapter"
                or manifest.get("mode") != "delta"
                or manifest.get("transport") != "chunked-shuffled-xor-zstd"
                or manifest.get("target_step") != next_step
                or manifest.get("base_step") != next_step - 1
            ):
                raise RuntimeError(
                    f"cannot apply adapter policy v{next_step}: manifest is not the adjacent delta"
                )
        build_seconds = manifest.get("build_seconds")
        self.last_adapter_delta_build_time = (
            float(build_seconds) if isinstance(build_seconds, (int, float)) and build_seconds >= 0 else -1.0
        )
        return manifest_location

    async def _prewarm_adapter_manifest(
        self,
        next_step: int,
        local_weights_path,
        target_url: str,
        deadline: float | None = None,
    ) -> str:
        if deadline is None:
            deadline = time.monotonic() + adapter_stage_limit_seconds(next_step)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"cannot apply adapter policy v{next_step}: staging deadline expired"
            )
        manifest = await _read_remote_adapter_manifest(target_url, deadline)
        return self._adapter_delta_base(
            next_step,
            local_weights_path,
            target_url,
            deadline,
            manifest,
        )

    async def _remove_initial_lora(self) -> None:
        """Release only the bootstrap slot after the two rolling slots are in use."""
        if self.lora_name is None or self.ckpt_step < 2:
            return
        current_slot = f"{self.lora_name}-slot{self.ckpt_step % 2}"
        if (
            self.policy.model_name != current_slot
            or self.policy.version != self.ckpt_step
            or self.verified_adapter_step != self.ckpt_step
        ):
            raise RuntimeError("cannot remove initial LoRA before the current policy is verified and switched")
        clients = self.inference.admin_clients
        await remove_lora_adapter(clients, self.lora_name)
        if await common_loaded_lora_receipt(clients, self.lora_name) is not None:
            raise RuntimeError("initial LoRA is still loaded after removal")
        get_logger().info(f"Initial LoRA slot {self.lora_name} is absent after policy v{self.ckpt_step}")

    async def apply_policy_update(
        self,
        next_step: int,
        update_deadline: float | None = None,
    ) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying
                return

            if next_step >= 3:
                await self._remove_initial_lora()

            checkpoint_wait_started = time.perf_counter()
            weights_path = None
            local_weights_path = None
            require_delta = False
            if self.model_express is not None:
                await self.wait_for_model_express_status(p2p_pb2.SOURCE_STATUS_READY)
            else:
                broadcast_dir = get_broadcast_dir(self.config.output_dir)
                weights_path = get_step_path(broadcast_dir, next_step)
                stable_marker = weights_path / "STABLE"
                if not stable_marker.exists():
                    get_logger().info(
                        f"Orchestrator paused: waiting for trainer to broadcast checkpoint {next_step}. "
                        "Training is progressing normally."
                    )
                    await wait_for_path(stable_marker)
                self.last_wait_for_ckpt_time = time.perf_counter() - checkpoint_wait_started
                local_weights_path = weights_path
                weights_path = inference_weight_path(self.config, weights_path, self.lora_name)
                if self.lora_name is None and next_step > 1 and isinstance(weights_path, str):
                    previous_path = get_step_path(broadcast_dir, next_step - 1)
                    previous_url = inference_weight_path(self.config, previous_path, self.lora_name)
                    if isinstance(previous_url, str):
                        delta_path = await asyncio.to_thread(
                            build_xor_delta_checkpoint,
                            previous_path,
                            local_weights_path,
                            base_url=previous_url,
                            target_url=weights_path,
                        )
                        if delta_path is not None:
                            get_logger().info(
                                f"Built lossless checkpoint delta for policy v{next_step}: {delta_path}"
                            )
                elif self.lora_name is not None and next_step > 0 and isinstance(weights_path, str):
                    if local_weights_path is None:
                        raise RuntimeError("adapter delta target path is not local")
                    delta_deadline = time.monotonic() + adapter_stage_limit_seconds(next_step)
                    if update_deadline is not None:
                        delta_deadline = min(delta_deadline, update_deadline)
                    if delta_deadline <= time.monotonic():
                        raise TimeoutError(
                            f"cannot apply adapter policy v{next_step}: update time limit expired"
                        )
                    source_prepare_started = time.perf_counter()
                    try:
                        manifest_path = await self._prewarm_adapter_manifest(
                            next_step,
                            local_weights_path,
                            weights_path,
                            delta_deadline,
                        )
                    finally:
                        self.last_adapter_source_prepare_time = (
                            time.perf_counter() - source_prepare_started
                        )
                    require_delta = True
                    build_time = (
                        format_time(self.last_adapter_delta_build_time)
                        if self.last_adapter_delta_build_time >= 0
                        else "not reported by trainer"
                    )
                    get_logger().info(
                        f"Using lossless adapter delta for policy v{next_step}: {manifest_path} "
                        f"(build {build_time})"
                    )
            if self.model_express is not None:
                self.last_wait_for_ckpt_time = time.perf_counter() - checkpoint_wait_started

            # Remote LoRA files cross the high-loss China link. Stage them on
            # both inference hosts before stopping old-policy generation. This
            # keeps useful complete groups flowing during the transfer and
            # reduces the engine pause to the local in-place load.
            adapter_identity = None
            if isinstance(weights_path, str):
                stage_started = time.perf_counter()
                if self.lora_name is not None:
                    if require_delta:
                        stage_result = await _stage_policy_adapter(
                            self.inference.admin_clients, weights_path,
                            policy_step=next_step, delta_deadline=delta_deadline,
                            origin_base_urls=self.inference.adapter_origin_base_urls,
                        )
                    else:
                        stage_result = await stage_lora_adapter(
                            self.inference.admin_clients, weights_path,
                            origin_base_urls=self.inference.adapter_origin_base_urls,
                        )
                    self.last_adapter_policy_step = int(stage_result.get("policy_step", next_step))
                    self.last_adapter_stage_limit_seconds = float(
                        stage_result.get(
                            "stage_limit_seconds", adapter_stage_limit_seconds(next_step)
                        )
                    )
                    adapter_identity = stage_result["adapter_identity"]
                    transfer_seconds = stage_result.get("transfer_wall_seconds")
                    self.last_adapter_wire_stage_time = (
                        float(transfer_seconds)
                        if isinstance(transfer_seconds, (int, float)) and transfer_seconds >= 0
                        else -1.0
                    )
                    reconstruct_seconds = stage_result.get(
                        "reconstruction_wall_seconds", stage_result.get("reconstruct_seconds")
                    )
                    if reconstruct_seconds is not None:
                        self.last_adapter_reconstruct_time = float(reconstruct_seconds)
                    else:
                        self.last_adapter_reconstruct_time = -1.0
                    origin_stage = format_time(float(stage_result.get("origin_stage_seconds", 0.0)))
                    peer_stage = format_time(float(stage_result.get("peer_stage_seconds", 0.0)))
                else:
                    await stage_full_checkpoint(self.inference.admin_clients, weights_path)
                self.last_stage_weights_time = time.perf_counter() - stage_started
                get_logger().info(
                    f"Staged policy v{next_step} on every inference host in "
                    f"{format_time(self.last_stage_weights_time)}"
                )
                if self.lora_name is not None:
                    reconstruct = (
                        format_time(self.last_adapter_reconstruct_time)
                        if self.last_adapter_reconstruct_time >= 0
                        else "not reported by inference host"
                    )
                    transfer = (
                        format_time(self.last_adapter_wire_stage_time)
                        if self.last_adapter_wire_stage_time >= 0
                        else "not reported by inference host"
                    )
                    get_logger().info(
                        f"Adapter v{next_step} transfer: transfer "
                        f"{transfer}, "
                        f"origin {origin_stage}, peer {peer_stage}, reconstruct {reconstruct}"
                    )
                    get_logger().info(
                        f"Adapter v{next_step} staging limit "
                        f"{self.last_adapter_stage_limit_seconds:g}s "
                        "(300s acceptance target)"
                    )

            # An observer can request a bounded safe boundary before we stop
            # old-policy work. The production orchestrator lets the current
            # valid batch ship instead of regenerating its incomplete groups.
            for observer in self.observers:
                wait_for_boundary = getattr(observer, "wait_for_policy_update_boundary", None)
                if wait_for_boundary is None:
                    continue
                try:
                    await wait_for_boundary(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.wait_for_policy_update_boundary"
                        f"({next_step}) raised: {exc!r}; applying the update now"
                    )

            # Stop old-policy rollouts before changing the adapter.
            for observer in self.observers:
                try:
                    await observer.on_version_pending(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.on_version_pending({next_step}) raised: {exc!r}"
                    )

            get_logger().debug(f"Updating weights to step {next_step}")
            t1 = time.perf_counter()
            lora_name = f"{self.lora_name}-slot{next_step % 2}" if self.lora_name is not None else None
            timings: dict[str, float] = {}
            receipts = await self.inference.update_weights(
                weights_path,
                lora_name=lora_name,
                step=next_step,
                require_delta=require_delta,
                prefetched=True,
                expected_adapter_identity=adapter_identity,
                timings=timings,
            )
            self.last_adapter_pause_time = timings.get("pause", 0.0)
            self.last_adapter_load_time = timings.get("adapter_load", 0.0)
            self.last_adapter_resume_time = timings.get("resume", 0.0)
            self.last_adapter_receipt_time = 0.0
            if self.lora_name is not None and isinstance(weights_path, str):
                assert local_weights_path is not None
                receipt_started = time.perf_counter()
                _write_receivers_ready(
                    local_weights_path,
                    run_name=self.config.output_dir.name,
                    step=next_step,
                    source_url=weights_path,
                    slot=lora_name,
                    adapter_identity=adapter_identity,
                    admin_clients=self.inference.admin_clients,
                    receipts=receipts,
                )
                self.last_adapter_receipt_time = time.perf_counter() - receipt_started
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            get_logger().info(f"Updated weights to step {next_step} in {format_time(self.last_update_weights_time)}")
            self.ckpt_step = next_step
            self.policy.version = next_step
            self.current_weights_path = weights_path

            if self.lora_name is not None:
                self.verified_adapter_step = next_step
                self.verified_adapter_url = str(weights_path)

            if lora_name is not None:
                self.inference.update_model_name(lora_name)
                self.policy.model_name = lora_name

            # The staging call and weight update cover all endpoints, but a
            # rollout must not use the new policy until both inference hosts
            # report the adapter as visible.
            await self.inference.wait_for_ready(self.policy.model_name)

            if next_step == 2:
                await self._remove_initial_lora()

            for observer in self.observers:
                try:
                    await observer.on_new_version(next_step)
                except Exception as exc:
                    get_logger().warning(
                        f"Observer {type(observer).__name__}.on_new_version({next_step}) raised: {exc!r}"
                    )

            if self.model_express is not None:
                await self.wait_for_model_express_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)

    async def reconcile_current_policy(self) -> None:
        """Restore the current LoRA on endpoints that restarted after an update.

        A healthy base engine is not sufficient: every endpoint must expose the
        exact policy model before it receives more rollout requests.
        """
        if self.lora_name is None or self.policy.model_name == self.config.model.name:
            return
        async with self.update_lock:
            remote_filesystem = (
                self.config.weight_broadcast.type == "filesystem"
                and bool(getattr(self.config.weight_broadcast, "source_url", None))
            )
            weights_path = self.current_weights_path
            if weights_path is None and self.model_express is None and self.ckpt_step > 0:
                broadcast_dir = get_broadcast_dir(self.config.output_dir)
                candidate = get_step_path(broadcast_dir, self.ckpt_step)
                if (candidate / "STABLE").exists():
                    weights_path = inference_weight_path(self.config, candidate, self.lora_name)
                    self.current_weights_path = weights_path
                elif remote_filesystem:
                    raise RuntimeError(
                        f"cannot reconcile policy v{self.ckpt_step}: adapter step is not STABLE"
                    )
            if weights_path is None:
                return

            require_delta = self.ckpt_step > 0
            if require_delta and self.verified_adapter_step != self.ckpt_step:
                raise RuntimeError(
                    f"cannot reconcile policy v{self.ckpt_step}: exact loaded policy is not verified"
                )
            if require_delta and self.verified_adapter_url is not None:
                if str(weights_path) != self.verified_adapter_url:
                    raise RuntimeError(
                        f"cannot reconcile policy v{self.ckpt_step}: verified adapter URL changed"
                    )

            receiver_ack = None
            if remote_filesystem:
                broadcast_dir = get_broadcast_dir(self.config.output_dir)
                step_path = get_step_path(broadcast_dir, self.ckpt_step)
                if not (step_path / "STABLE").is_file():
                    raise RuntimeError(
                        f"cannot reconcile policy v{self.ckpt_step}: adapter step is not STABLE"
                    )
                receiver_ack = _read_receivers_ready(
                    step_path,
                    run_name=self.config.output_dir.name,
                    step=self.ckpt_step,
                    source_url=str(weights_path),
                    slot=self.policy.model_name,
                    admin_clients=self.inference.admin_clients,
                )
                loaded_receipt = await common_loaded_lora_receipt(
                    self.inference.admin_clients,
                    self.policy.model_name,
                )
                _validate_receipt_against_ack(loaded_receipt, receiver_ack, step=self.ckpt_step)

            checks = await asyncio.gather(
                *(model_is_visible(client, self.policy.model_name) for client in self.inference.admin_clients)
            )
            missing = [client for client, visible in zip(self.inference.admin_clients, checks) if not visible]
            if not missing:
                return

            get_logger().warning(
                f"Policy model {self.policy.model_name} is absent on {len(missing)}/"
                f"{len(self.inference.admin_clients)} inference endpoint(s); restoring it"
            )
            t0 = time.perf_counter()
            stage_result = await stage_lora_adapter(
                missing,
                weights_path,
                require_delta=require_delta,
                policy_step=self.ckpt_step if require_delta else None,
            )
            if require_delta:
                self.last_adapter_policy_step = int(stage_result["policy_step"])
                self.last_adapter_stage_limit_seconds = float(stage_result["stage_limit_seconds"])
            await load_lora_adapter(
                missing,
                self.policy.model_name,
                weights_path,
                require_delta=require_delta,
                prefetched=True,
                expected_identity=stage_result["adapter_identity"],
            )
            visible = await asyncio.gather(
                *(model_is_visible(client, self.policy.model_name) for client in self.inference.admin_clients)
            )
            if not all(visible):
                raise RuntimeError(
                    f"Policy model {self.policy.model_name} is still missing on "
                    f"{visible.count(False)} inference endpoint(s) after reload"
                )
            if remote_filesystem:
                loaded_receipt = await common_loaded_lora_receipt(
                    self.inference.admin_clients,
                    self.policy.model_name,
                )
                _validate_receipt_against_ack(loaded_receipt, receiver_ack, step=self.ckpt_step)
            self.last_reconcile_time = time.perf_counter() - t0
            self.reconcile_count += 1
            get_logger().info(
                f"Restored policy model {self.policy.model_name} on every inference endpoint "
                f"in {format_time(self.last_reconcile_time)}"
            )

    async def wait_for_model_express_status(self, status: int) -> None:
        while not self.stopped.is_set():
            found = await asyncio.to_thread(
                self.model_express.exists_role_with_status,
                "trainer",
                status,
            )
            if found:
                return
            await asyncio.sleep(self.poll_interval)
        raise asyncio.CancelledError

    def gauges(self) -> dict[str, float]:
        return {
            "watcher/policy_version": float(self.policy.version),
            "watcher/update_count": float(self.update_count),
            "watcher/reconcile_count": float(self.reconcile_count),
            "watcher/last_update_weights_time": self.last_update_weights_time,
            "watcher/last_stage_weights_time": self.last_stage_weights_time,
            "watcher/last_adapter_delta_build_time": self.last_adapter_delta_build_time,
            "watcher/last_adapter_source_prepare_time": self.last_adapter_source_prepare_time,
            "watcher/last_adapter_wire_stage_time": self.last_adapter_wire_stage_time,
            "watcher/last_adapter_reconstruct_time": self.last_adapter_reconstruct_time,
            "watcher/last_adapter_load_time": self.last_adapter_load_time,
            "watcher/last_adapter_pause_time": self.last_adapter_pause_time,
            "watcher/last_adapter_resume_time": self.last_adapter_resume_time,
            "watcher/last_adapter_receipt_time": self.last_adapter_receipt_time,
            "watcher/last_adapter_policy_step": float(self.last_adapter_policy_step),
            "watcher/last_adapter_stage_limit_seconds": self.last_adapter_stage_limit_seconds,
            "watcher/adapter_stage_acceptance_target_seconds": 300.0,
            "watcher/last_wait_for_ckpt_time": self.last_wait_for_ckpt_time,
            "watcher/last_reconcile_time": self.last_reconcile_time,
        }
