import asyncio
import json
import math
import os
import re
import signal
import sys
import threading
import time
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any

import uvloop
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.vllm.worker.filesystem import (
    _ADAPTER_MARKER_MAX_BYTES,
    _ADAPTER_METADATA_MAX_BYTES,
    _open_directory_nofollow,
    _read_text_nofollow,
    _stat_regular_entry,
    _delta_stage_timeout_s,
    _adapter_step_from_source_url,
    adapter_stage_limit_seconds,
    adapter_cache_path,
    adapter_reader_lock,
    adapter_stage_status,
    full_adapter_stage_timeout_s,
    stage_full_checkpoint_files,
)
from prime_rl.inference.patches import (
    monkey_patch_dp_coordinator_startup_timeout,
    monkey_patch_harmony_stop_token_propagation,
    monkey_patch_nano_v3_reasoning_parser,
    monkey_patch_strip_routed_experts_from_chat,
    monkey_patch_tokenize_params_validation,
)
from prime_rl.utils.logger import get_logger

logger = get_logger()

# NOTE: Fix harmony stop token propagation for GPT-OSS models
# Upstream issue still open: https://github.com/vllm-project/vllm/issues/22519
monkey_patch_harmony_stop_token_propagation()
# NOTE: Monkeypatch TokenizeParams to fix overly conservative validation
# Still needed in vLLM 0.20 — upstream rejects prompt_len > max_model_len - max_tokens
monkey_patch_tokenize_params_validation()
# NOTE: Register Nano V3 reasoning parser so configs can use
# `reasoning_parser = "nano_v3"` without a vLLM plugin file.
monkey_patch_nano_v3_reasoning_parser()
# NOTE: routed_experts are consumed only via the serialized /generate path (router
# replay). The chat-completions path encodes them as a base64 np.save string the PD
# router cannot merge, which fails eval rollouts (they use chat completions). Strip
# routed_experts from chat responses since the server-wide enable flag has no
# per-request toggle.
monkey_patch_strip_routed_experts_from_chat()
# NOTE: vLLM hard-codes a 120s DP coordinator startup timeout, which the rank-0
# API server blows through when all engine-core ranks on the node are loading
# weights concurrently (multi-node disaggregated deployments).
monkey_patch_dp_coordinator_startup_timeout()

logger = init_logger("vllm.entrypoints.openai.api_server")

# Create our own router for custom endpoints
router = APIRouter()
lora_load_lock = asyncio.Lock()
_loaded_lora_receipts: dict[str, dict[str, object]] = {}
_server_instance_id = uuid.uuid4().hex
_lora_stage_lock = threading.Lock()
_lora_stage_count = 0
_lora_stage_started_at: float | None = None
_lora_stage_process: tuple[str, int] | None = None
_engine_state_lock = threading.Lock()
_engine_state = "running"


def _set_engine_state(state: str) -> None:
    global _engine_state
    with _engine_state_lock:
        _engine_state = state


def _get_engine_state() -> str:
    with _engine_state_lock:
        return _engine_state


def _begin_lora_stage() -> None:
    global _lora_stage_count, _lora_stage_started_at
    with _lora_stage_lock:
        if _lora_stage_count == 0:
            _lora_stage_started_at = time.monotonic()
        _lora_stage_count += 1


def _end_lora_stage() -> None:
    global _lora_stage_count, _lora_stage_started_at
    with _lora_stage_lock:
        _lora_stage_count = max(0, _lora_stage_count - 1)
        if _lora_stage_count == 0:
            _lora_stage_started_at = None


def _lora_stage_status() -> dict[str, Any]:
    with _lora_stage_lock:
        if _lora_stage_count == 0 or _lora_stage_started_at is None:
            return {"status": "idle", "age_seconds": 0.0}
        state = {
            "status": "staging",
            "age_seconds": round(max(0.0, time.monotonic() - _lora_stage_started_at), 3),
        }
        if _lora_stage_process is not None:
            state["stage_id"], state["pid"] = _lora_stage_process
        return state


def _set_lora_stage_process(stage_id: str, pid: int | None) -> None:
    global _lora_stage_process
    with _lora_stage_lock:
        _lora_stage_process = (stage_id, pid) if pid is not None else None


async def _kill_stage_process(process, communication=None) -> tuple[bytes, bytes]:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait_for(asyncio.shield(process.wait()), timeout=5.0)
    if communication is None:
        return b"", b""
    if not communication.done():
        try:
            await asyncio.wait_for(asyncio.shield(communication), timeout=5.0)
        except asyncio.TimeoutError:
            communication.cancel()
            await asyncio.gather(communication, return_exceptions=True)
            return b"", b""
    stdout, stderr = communication.result()
    return stdout, stderr or b""


def _combined_lora_stage_status(
    active: dict[str, Any], durable: dict[str, object]
) -> dict[str, Any]:
    """Return durable transfer details without hiding an active stage."""
    state = dict(active)
    active_status = active.get("status")
    durable_status = durable.get("status")
    if active_status == "staging":
        if (
            active.get("stage_id") != durable.get("stage_id")
            or active.get("pid") != durable.get("pid")
        ):
            return state
        state.update(durable)
        state["status"] = "staging"
        if "age_seconds" in active:
            state["age_seconds"] = active["age_seconds"]
        for key in ("stage_id", "pid"):
            if key in active:
                state[key] = active[key]
        return state
    if durable_status in {"failed", "ready"}:
        state.update(durable)
    elif durable_status == "staging":
        state.update(durable)
        state["status"] = "failed"
        state["error"] = "RuntimeError: previous adapter stage did not finish"
    elif durable_status not in {None, "idle"}:
        state.update(durable)
        state["status"] = "unknown"
    return state


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _target_has_routed_experts(target: str) -> bool:
    return "experts" in target.split(".")


def _adapter_requires_routed_experts(adapter_path: str | Path) -> bool:
    try:
        config = json.loads(
            _read_text_nofollow(
                Path(adapter_path) / "adapter_config.json",
                encoding="utf-8",
                max_bytes=_ADAPTER_METADATA_MAX_BYTES,
            )
        )
    except (OSError, ValueError) as exc:
        raise ValueError("adapter configuration cannot be read") from exc
    if not isinstance(config, dict):
        raise ValueError("adapter configuration must be an object")
    targets = config.get("target_modules")
    if isinstance(targets, str):
        targets = [targets]
    if not isinstance(targets, list) or any(not isinstance(target, str) for target in targets):
        raise ValueError("adapter configuration target_modules is invalid")
    return any(_target_has_routed_experts(target) for target in targets)


def _validate_adapter_target_compatibility(adapter_path: str | Path, handler: OpenAIServingModels) -> None:
    if not _adapter_requires_routed_experts(adapter_path):
        return
    lora_config = handler.engine_client.vllm_config.lora_config
    if lora_config is None:
        raise ValueError("adapter requires routed experts, but server LoRA is disabled")
    if lora_config.target_modules is None:
        return
    configured_targets = lora_config.target_modules
    if not any(_target_has_routed_experts(target) for target in configured_targets):
        raise ValueError(
            "adapter target_modules requires routed experts, but the effective "
            "server lora_target_modules excludes 'experts'"
        )


def _set_moe_lora_mode(target_modules: list[str] | None) -> None:
    import os

    routed_moe_lora = any(
        _target_has_routed_experts(target) for target in (target_modules or [])
    )
    if target_modules and not routed_moe_lora:
        os.environ["PRIME_NO_MOE_LORA"] = "1"
    else:
        os.environ.pop("PRIME_NO_MOE_LORA", None)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def _read_verified_adapter_identity(
    source_url: str,
    destination,
    *,
    expected_stage_id: str | None = None,
    expected_pid: int | None = None,
) -> dict[str, object]:
    """Read the identity already verified when the cache reached READY."""
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        raise RuntimeError("adapter source URL is not remote")
    try:
        directory_fd = _open_directory_nofollow(
            Path(destination),
            create=False,
            label="adapter cache directory",
        )
    except OSError as exc:
        raise RuntimeError("adapter cache is not a safe directory") from exc
    try:
        if _read_text_nofollow(
            "READY",
            encoding="ascii",
            dir_fd=directory_fd,
            max_bytes=_ADAPTER_MARKER_MAX_BYTES,
        ) != "ready\n":
            raise RuntimeError("adapter cache is not ready")
        try:
            manifest = json.loads(
                _read_text_nofollow(
                    "manifest.json",
                    encoding="utf-8",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError("adapter cache manifest is invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise RuntimeError("adapter cache manifest has an unsupported version")
        if manifest.get("source_url") != source_url:
            raise RuntimeError("adapter cache source identity differs from the requested source")

        files = manifest.get("files")
        if not isinstance(files, dict):
            raise RuntimeError("adapter cache manifest has no file records")
        target = files.get("adapter_model.safetensors")
        config = files.get("adapter_config.json")
        if not isinstance(target, dict) or not isinstance(config, dict):
            raise RuntimeError("adapter cache manifest is missing adapter files")
        target_sha256 = target.get("sha256")
        target_size = target.get("size")
        config_sha256 = config.get("sha256")
        config_size = config.get("size")
        if (
            not isinstance(target_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", target_sha256) is None
            or type(target_size) is not int
            or target_size <= 0
            or not isinstance(config_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
            or type(config_size) is not int
            or config_size < 0
        ):
            raise RuntimeError("adapter cache file identity is invalid")
        status = adapter_stage_status()
        if (
            status.get("status") != "ready"
            or status.get("source_url") != source_url
            or status.get("path") != Path(destination).as_posix()
            or not isinstance(status.get("stage_id"), str)
            or not isinstance(status.get("pid"), int)
        ):
            raise RuntimeError("adapter cache has no same-stage verification proof")
        if expected_stage_id is not None and status.get("stage_id") != expected_stage_id:
            raise RuntimeError("adapter cache stage proof differs from the request")
        if expected_pid is not None and status.get("pid") != expected_pid:
            raise RuntimeError("adapter cache worker proof differs from the request")
        verified_files = status.get("verified_files")
        file_stats = status.get("file_stats")
        if not isinstance(verified_files, dict) or not isinstance(file_stats, dict):
            raise RuntimeError("adapter cache has incomplete verification proof")
        for name, record in (("adapter_model.safetensors", target), ("adapter_config.json", config)):
            size = record.get("size")
            digest = record.get("sha256")
            entry = _stat_regular_entry(directory_fd, name, "adapter cache file")
            proof = verified_files.get(name)
            stats = file_stats.get(name)
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or type(size) is not int
                or size < 0
                or entry is None
                or entry.st_size != size
                or not isinstance(proof, dict)
                or proof.get("size") != size
                or proof.get("sha256") != digest
                or not isinstance(stats, dict)
                or stats.get("dev") != entry.st_dev
                or stats.get("ino") != entry.st_ino
                or stats.get("size") != entry.st_size
                or stats.get("mtime_ns") != entry.st_mtime_ns
            ):
                raise RuntimeError(f"adapter cache {name} identity is invalid")
        try:
            sidecar = _read_text_nofollow(
                "adapter_model.safetensors.sha256",
                encoding="ascii",
                dir_fd=directory_fd,
                max_bytes=_ADAPTER_MARKER_MAX_BYTES,
            ).strip()
        except OSError as exc:
            raise RuntimeError("adapter cache target sidecar is invalid") from exc
        if sidecar != target_sha256:
            raise RuntimeError("adapter cache target sidecar differs from manifest")

        identity: dict[str, object] = {
            "source_url": source_url,
            "target_sha256": target_sha256,
            "target_size": target_size,
            "config_sha256": config_sha256,
            "config_size": config_size,
        }

        try:
            transport = json.loads(
                _read_text_nofollow(
                    "transport.json",
                    encoding="utf-8",
                    dir_fd=directory_fd,
                    max_bytes=_ADAPTER_METADATA_MAX_BYTES,
                )
            )
        except FileNotFoundError:
            transport = None
        except (OSError, ValueError) as exc:
            raise RuntimeError("adapter cache transport record is invalid") from exc
        if transport is not None:
            if not isinstance(transport, dict) or transport.get("source_url") != source_url:
                raise RuntimeError("adapter cache transport source differs from manifest")
            if transport.get("target_sha256") != target_sha256:
                raise RuntimeError("adapter cache transport target differs from manifest")
            if transport.get("config_sha256") not in (None, config_sha256):
                raise RuntimeError("adapter cache transport configuration differs from manifest")
        return identity
    except (FileNotFoundError, UnicodeError) as exc:
        raise RuntimeError("adapter cache metadata is missing or invalid") from exc
    finally:
        os.close(directory_fd)


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
}


@router.post("/pause")
async def pause(request: Request):
    mode = request.query_params.get("mode", "abort")
    if mode not in ("abort", "keep"):
        return JSONResponse({"error": f"unsupported pause mode: {mode}"}, status_code=400)
    clear_cache = request.query_params.get("clear_cache", "true").lower() == "true"
    logger.debug(f"Received /pause request (mode={mode}, clear_cache={clear_cache})")
    _set_engine_state("pausing")
    try:
        await engine_client(request).pause_generation(mode=mode, clear_cache=clear_cache)
    except BaseException:
        _set_engine_state("unknown")
        raise
    _set_engine_state("paused")
    return {"status": "paused"}


@router.get("/resume_status")
async def resume_status():
    return {"state": _get_engine_state()}


@router.post("/resume")
async def resume(request: Request):
    _set_engine_state("resuming")
    try:
        await engine_client(request).resume_generation()
    except BaseException:
        _set_engine_state("unknown")
        raise
    _set_engine_state("running")
    return {"status": "resumed"}


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    await engine_client(request).collective_rpc("update_weights_from_path", args=(data.get("weight_dir"),))
    return {"status": "ok"}


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request):
    """Stage remote adapters and serialize loads so retries cannot overlap."""
    async with lora_load_lock:
        handler = models(raw_request)
        remote_adapter = lora_request.lora_path.startswith(("http://", "https://"))
        request_data = await raw_request.json()
        require_delta = request_data.get("require_delta", False)
        if not isinstance(require_delta, bool):
            return JSONResponse(content={"error": "require_delta must be a boolean"}, status_code=400)
        prefetched = request_data.get("prefetched", False)
        if not isinstance(prefetched, bool):
            return JSONResponse(content={"error": "prefetched must be a boolean"}, status_code=400)
        if remote_adapter:
            _loaded_lora_receipts.pop(lora_request.lora_name, None)
        if remote_adapter:
            _begin_lora_stage()
        try:
            adapter_identity = None
            if remote_adapter:
                source_url = lora_request.lora_path
                policy_step = None
                if require_delta:
                    try:
                        policy_step = _adapter_step_from_source_url(source_url)
                    except OSError as exc:
                        return JSONResponse({"error": str(exc)}, status_code=400)
                if not prefetched:
                    class StageRequest:
                        async def json(self):
                            return {
                                "lora_path": source_url,
                                "require_delta": require_delta,
                                "policy_step": policy_step,
                            }

                        async def is_disconnected(self):
                            probe = getattr(raw_request, "is_disconnected", None)
                            return await probe() if probe is not None else False

                    staged = await stage_lora_adapter(StageRequest())
                    if not isinstance(staged, dict) or staged.get("status") != "ok":
                        return staged
                else:
                    staged = None
                destination = adapter_cache_path(source_url)
                lora_request.lora_path = destination.as_posix()
            lora_request.load_inplace = True
            async def load_engine_adapter():
                task = asyncio.create_task(handler.load_lora_adapter(lora_request))
                try:
                    done, _ = await asyncio.wait({task}, timeout=600.0)
                    if task in done:
                        return task.result()
                    task.cancel()
                    done, _ = await asyncio.wait({task}, timeout=5.0)
                    if task in done:
                        raise asyncio.TimeoutError
                    _set_engine_state("unknown")
                    os.kill(os.getpid(), signal.SIGTERM)
                    raise asyncio.TimeoutError
                except asyncio.CancelledError:
                    task.cancel()
                    done, _ = await asyncio.wait({task}, timeout=5.0)
                    if task not in done:
                        _set_engine_state("unknown")
                        os.kill(os.getpid(), signal.SIGTERM)
                    raise
            if remote_adapter:
                with adapter_reader_lock(destination):
                    adapter_identity = _read_verified_adapter_identity(
                        source_url,
                        destination,
                        expected_stage_id=(
                            staged.get("stage_id")
                            if isinstance(staged, dict)
                            else None
                        ),
                        expected_pid=(
                            staged.get("pid")
                            if isinstance(staged, dict) and isinstance(staged.get("pid"), int)
                            else None
                        ),
                    )
                    try:
                        _validate_adapter_target_compatibility(destination, handler)
                    except ValueError as exc:
                        return JSONResponse({"error": str(exc)}, status_code=400)
                    try:
                        response = await load_engine_adapter()
                    except asyncio.TimeoutError:
                        return JSONResponse({"error": "LoRA load timed out; server restart required"}, status_code=504)
            else:
                _loaded_lora_receipts.pop(lora_request.lora_name, None)
                try:
                    _validate_adapter_target_compatibility(lora_request.lora_path, handler)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
                try:
                    response = await load_engine_adapter()
                except asyncio.TimeoutError:
                    return JSONResponse({"error": "LoRA load timed out; server restart required"}, status_code=504)
            stored = handler.lora_requests.get(lora_request.lora_name)
            if stored is not None:
                stored.load_inplace = False
            if isinstance(response, ErrorResponse):
                return JSONResponse(content=response.model_dump(), status_code=response.error.code)
            if not remote_adapter:
                return {"status": "ok"}
            stored_name = getattr(stored, "lora_name", None) if stored is not None else None
            stored_path = getattr(stored, "lora_path", None) if stored is not None else None
            if (
                not isinstance(adapter_identity, dict)
                or stored_name != lora_request.lora_name
                or str(stored_path) != lora_request.lora_path
            ):
                return JSONResponse(
                    {"error": "loaded adapter identity is unavailable or mismatched"},
                    status_code=500,
                )
            receipt = {
                **adapter_identity,
                "lora_name": lora_request.lora_name,
                "server_instance": _server_instance_id,
            }
            _loaded_lora_receipts[lora_request.lora_name] = receipt
            return {"status": "ok", "loaded_receipt": receipt}
        finally:
            if remote_adapter:
                _end_lora_stage()


@router.post("/stage_lora_adapter")
async def stage_lora_adapter(raw_request: Request):
    """Prefetch a remote adapter without pausing or calling engine workers."""
    data = await raw_request.json()
    source_url = data.get("lora_path")
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        return JSONResponse({"error": "lora_path must be an HTTP URL"}, status_code=400)
    require_delta = data.get("require_delta", False)
    if not isinstance(require_delta, bool):
        return JSONResponse({"error": "require_delta must be a boolean"}, status_code=400)
    requested_step = data.get("policy_step")
    if requested_step is not None and (
        type(requested_step) is not int or requested_step <= 0
    ):
        return JSONResponse({"error": "policy_step must be a positive integer"}, status_code=400)
    if require_delta:
        if requested_step is None:
            return JSONResponse(
                {"error": "policy_step is required for delta adapter staging"},
                status_code=400,
            )
        try:
            policy_step = _adapter_step_from_source_url(source_url)
        except OSError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        if requested_step is not None and requested_step != policy_step:
            return JSONResponse(
                {"error": "policy_step differs from the validated adapter path"},
                status_code=400,
            )
        stage_limit = adapter_stage_limit_seconds(policy_step)
    else:
        if requested_step is not None:
            return JSONResponse(
                {"error": "policy_step is only valid for delta adapter staging"},
                status_code=400,
            )
        policy_step = None
        stage_limit = None
    requested_timeout = data.get("stage_timeout_seconds")
    configured_timeout = _delta_stage_timeout_s() if require_delta else None
    if requested_timeout is not None:
        if (
            not require_delta
            or isinstance(requested_timeout, bool)
            or not isinstance(requested_timeout, (int, float))
            or not _is_finite_number(requested_timeout)
            or requested_timeout <= 0
            or configured_timeout is None
            or requested_timeout > configured_timeout
            or stage_limit is None
            or requested_timeout > stage_limit
        ):
            return JSONResponse(
                {"error": "stage_timeout_seconds must be within the configured delta limit"},
                status_code=400,
            )
        stage_timeout = float(requested_timeout)
    else:
        stage_timeout = stage_limit if require_delta else configured_timeout
    if require_delta and (configured_timeout is None or configured_timeout < stage_limit):
        return JSONResponse(
            {"error": "configured adapter stage limit is below the policy limit"},
            status_code=500,
        )
    _begin_lora_stage()
    started = time.monotonic()
    operation_timeout = (
        stage_timeout
        if stage_timeout is not None
        else full_adapter_stage_timeout_s
    )
    deadline = started + operation_timeout
    stage_id = uuid.uuid4().hex
    process = None
    communication = None

    def success_payload(destination):
        if time.monotonic() > deadline:
            state = adapter_stage_status()
            ready_at = state.get("ready_at_monotonic")
            if state.get("status") != "ready" or not isinstance(ready_at, (int, float)) or ready_at > deadline:
                raise TimeoutError("adapter staging completed after its deadline")
        if destination != adapter_cache_path(source_url):
            raise RuntimeError("adapter stage returned an unexpected destination")
        identity = _read_verified_adapter_identity(
            source_url,
            destination,
            expected_stage_id=stage_id,
            expected_pid=process.pid,
        )
        stage_state = adapter_stage_status()
        if (
            stage_state.get("status") != "ready"
            or stage_state.get("stage_id") != stage_id
            or stage_state.get("pid") != process.pid
            or stage_state.get("source_url") != source_url
            or stage_state.get("path") != destination.as_posix()
            or stage_state.get("target_sha256") != identity.get("target_sha256")
        ):
            raise RuntimeError("adapter stage status identity does not match the worker")
        if time.monotonic() > deadline:
            ready_at = stage_state.get("ready_at_monotonic")
            if stage_state.get("status") != "ready" or not isinstance(ready_at, (int, float)) or ready_at > deadline:
                raise TimeoutError("adapter staging identity verification exceeded its deadline")
        payload = {
            "status": "ok",
            "path": destination.as_posix(),
            "stage_seconds": time.monotonic() - started,
            "total_stage_wall_seconds": time.monotonic() - started,
            "adapter_identity": identity,
            "stage_id": stage_id,
            "pid": process.pid,
        }
        if policy_step is not None:
            payload.update(
                {
                    "policy_step": policy_step,
                    "stage_limit_seconds": stage_limit,
                    "acceptance_target_seconds": 300.0,
                }
            )
        for name in (
            "transfer_wall_seconds",
            "reconstruction_wall_seconds",
            "reconstruction_worker_seconds",
            "final_verification_wall_seconds",
        ):
            value = stage_state.get(name)
            if isinstance(value, (int, float)) and value >= 0:
                payload[name] = float(value)
        payload["cache_hit"] = bool(stage_state.get("cache_hit", False))
        payload["bytes_done"] = int(stage_state.get("bytes_done", 0) or 0)
        payload["bytes_total"] = int(stage_state.get("bytes_total", 0) or 0)
        return payload

    def current_commit(destination) -> bool:
        try:
            identity = _read_verified_adapter_identity(
                source_url,
                destination,
                expected_stage_id=stage_id,
                expected_pid=process.pid,
            )
            state = adapter_stage_status()
        except (OSError, RuntimeError, ValueError):
            return False
        return (
            state.get("status") == "ready"
            and state.get("stage_id") == stage_id
            and state.get("pid") == process.pid
            and state.get("source_url") == source_url
            and state.get("path") == destination.as_posix()
            and state.get("target_sha256") == identity.get("target_sha256")
            and isinstance(state.get("ready_at_monotonic"), (int, float))
            and state["ready_at_monotonic"] <= deadline
        )

    async def monitor_disconnect() -> bool:
        while process is not None and process.returncode is None:
            if await raw_request.is_disconnected():
                return True
            await asyncio.sleep(0.25)
        return False

    def parse_result(stdout: bytes) -> dict[str, object]:
        if len(stdout) > 1_000_000:
            raise RuntimeError("adapter stage worker output is too large")
        try:
            text = stdout.decode("utf-8")
            result, end = json.JSONDecoder().raw_decode(text)
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("adapter stage worker returned invalid JSON") from exc
        if text[end:].strip():
            raise RuntimeError("adapter stage worker returned multiple JSON frames")
        if (
            not isinstance(result, dict)
            or result.get("stage_id") != stage_id
            or result.get("pid") != process.pid
        ):
            raise RuntimeError("adapter stage worker identity does not match the request")
        return result

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "prime_rl.inference.vllm.adapter_stage_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
            start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        _set_lora_stage_process(stage_id, process.pid)
        request = {
            "source_url": source_url,
            "require_delta": require_delta,
            "policy_step": policy_step,
            "stage_timeout_seconds": stage_timeout if require_delta else None,
            "stage_id": stage_id,
        }
        communication = asyncio.create_task(
            process.communicate(json.dumps(request, separators=(",", ":")).encode("utf-8"))
        )
        disconnect_task = asyncio.create_task(monitor_disconnect())
        try:
            remaining = max(0.0, deadline - time.monotonic())
            done, _ = await asyncio.wait(
                (communication, disconnect_task),
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            disconnected = disconnect_task in done and disconnect_task.result()
            timed_out = communication not in done and not disconnected
            if timed_out:
                stdout, _stderr = await _kill_stage_process(process, communication)
            elif disconnected:
                stdout, _stderr = await _kill_stage_process(process, communication)
                return JSONResponse({"error": "adapter stage request disconnected"}, status_code=499)
            else:
                stdout, _stderr = communication.result()
            if not stdout.strip():
                destination = adapter_cache_path(source_url)
                if current_commit(destination):
                    return success_payload(destination)
                raise TimeoutError("adapter staging worker stopped before returning a result")
            result = parse_result(stdout)
        finally:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
        if result.get("status") != "ok":
            error_type = result.get("error_type")
            status = 504 if error_type == "TimeoutError" else 500
            return JSONResponse({"error": result.get("error", "adapter staging failed")}, status_code=status)
        destination = Path(str(result.get("path", "")))
        try:
            return success_payload(destination)
        except TimeoutError:
            if current_commit(destination):
                return success_payload(destination)
            raise
    except asyncio.CancelledError:
        if process is not None and communication is not None:
            await _kill_stage_process(process, communication)
        raise
    except (TimeoutError, asyncio.TimeoutError):
        return JSONResponse(
            {
                "error": "adapter delta staging exceeded its time limit",
                "policy_step": policy_step,
                "stage_limit_seconds": stage_limit,
                "acceptance_target_seconds": 300.0,
            },
            status_code=504,
        )
    finally:
        if process is not None and communication is not None and process.returncode is None:
            await _kill_stage_process(process, communication)
        _set_lora_stage_process(stage_id, None)
        _end_lora_stage()


@router.get("/staging_status")
async def staging_status(raw_request: Request):
    state = _combined_lora_stage_status(_lora_stage_status(), adapter_stage_status())
    handler = models(raw_request)
    loaded = getattr(handler, "lora_requests", None)
    receipts = {}
    if isinstance(loaded, dict):
        for name, receipt in list(_loaded_lora_receipts.items()):
            source_url = receipt.get("source_url")
            stored = loaded.get(name)
            expected_path = (
                adapter_cache_path(source_url).as_posix()
                if isinstance(source_url, str)
                else None
            )
            if stored is None or str(getattr(stored, "lora_path", "")) != expected_path:
                _loaded_lora_receipts.pop(name, None)
                continue
            receipts[name] = dict(receipt)
    state["loaded_receipts"] = receipts
    return state


@router.post("/stage_full_checkpoint")
async def stage_full_checkpoint(raw_request: Request):
    """Prefetch a remote full checkpoint while generation continues."""
    data = await raw_request.json()
    source_url = data.get("checkpoint_url")
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        return JSONResponse({"error": "checkpoint_url must be an HTTP URL"}, status_code=400)
    destination = await asyncio.to_thread(stage_full_checkpoint_files, source_url)
    return {"status": "ok", "path": destination.as_posix()}


@router.get("/liveness")
async def liveness(raw_request: Request):
    """Check that the engine event loop can service a no-op worker RPC."""
    try:
        await asyncio.wait_for(
            engine_client(raw_request).collective_rpc("liveness_probe"),
            timeout=raw_request.app.state.liveness_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return JSONResponse({"status": "engine_unresponsive"}, status_code=503)
    return {"status": "ok"}


@router.post("/init_broadcaster")
async def init_broadcaster(request: Request):
    data = await request.json()
    host = data.get("host")
    port = data.get("port")
    timeout = data.get("timeout")
    rank_offset = data.get("rank_offset")
    inference_world_size = data.get("inference_world_size")
    quantize_in_weight_transfer = data.get("quantize_in_weight_transfer", False)
    session_id = data.get("session_id", "default")
    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, rank_offset, inference_world_size, timeout, quantize_in_weight_transfer, session_id),
    )
    return {"status": "ok"}


async def custom_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple,
):
    """
    Modifies init_app_state:
    1. Call the original init_app_state to set up standard state, including
       vLLM 0.20's ``serving_tokens`` for ``/inference/v1/generate``.
    2. Replace ``serving_tokens`` with ``PrimeRlServingTokens`` so DP-rank
       routing and ``routed_experts`` export survive the migration off the
       legacy ``/v1/generate`` endpoint.
    """
    await init_app_state(engine_client, state, args, supported_tasks)

    state.liveness_timeout_seconds = args.liveness_timeout_seconds

    # Swap in our ServingTokens subclass for /inference/v1/generate so the
    # X-data-parallel-rank header and routed_experts response field — both
    # used by prime-RL's renderer / router-replay paths — keep working.
    if "generate" in supported_tasks and state.serving_tokens is not None:
        from prime_rl.inference.vllm.serving_tokens import PrimeRlServingTokens

        upstream = state.serving_tokens
        prime_serving = object.__new__(PrimeRlServingTokens)
        prime_serving.__dict__.update(upstream.__dict__)
        state.serving_tokens = prime_serving


import vllm.entrypoints.openai.api_server  # noqa: E402
import vllm.v1.utils  # noqa: E402
from vllm.entrypoints.openai.api_server import build_app as _original_build_app  # noqa: E402
from vllm.v1.utils import run_api_server_worker_proc as _original_run_api_server_worker_proc  # noqa: E402


def custom_build_app(args: Namespace, supported_tasks: tuple, model_config=None):
    """
    Wrap build_app to include our custom router.
    """
    app = _original_build_app(args, supported_tasks, model_config)
    app.include_router(router)
    return app


def custom_run_api_server_worker_proc(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """
    Re-import our module in child processes so monkey patches (custom routes,
    custom init_app_state) are applied in multi-API-server mode.
    """
    import prime_rl.inference.vllm.server  # noqa: F401

    _original_run_api_server_worker_proc(listen_address, sock, args, client_config, **uvicorn_kwargs)


vllm.entrypoints.openai.api_server.init_app_state = custom_init_app_state
vllm.entrypoints.openai.api_server.build_app = custom_build_app
vllm.v1.utils.run_api_server_worker_proc = custom_run_api_server_worker_proc


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference we do some config translation (i.e. pass populated namespace
# to `parse_args`) and additional arg validation
def server(config: InferenceConfig, vllm_extra: dict[str, Any] | None = None):
    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.openai.api_server import run_server

    namespace = config.to_vllm()
    if vllm_extra:
        for key, value in vllm_extra.items():
            setattr(namespace, key, value)

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=namespace)
    assert args is not None
    validate_parsed_serve_args(args)
    _set_moe_lora_mode(args.lora_target_modules)

    # Set the worker extension class based on the broadcast backend
    args.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
