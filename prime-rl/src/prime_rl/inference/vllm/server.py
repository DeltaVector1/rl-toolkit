import asyncio
import hashlib
import json
import math
import re
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
    _delta_stage_timeout_s,
    _adapter_step_from_source_url,
    _adapter_stage_publish_lock,
    adapter_stage_limit_seconds,
    adapter_cache_path,
    adapter_reader_lock,
    adapter_stage_status,
    stage_full_checkpoint_files,
    stage_lora_adapter_files,
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
        return {
            "status": "staging",
            "age_seconds": round(max(0.0, time.monotonic() - _lora_stage_started_at), 3),
        }


def _combined_lora_stage_status(
    active: dict[str, Any], durable: dict[str, object]
) -> dict[str, Any]:
    """Return durable transfer details without hiding an active stage."""
    state = dict(active)
    active_status = active.get("status")
    durable_status = durable.get("status")
    if active_status == "staging":
        state.update(durable)
        state["status"] = "staging"
        if "age_seconds" in active:
            state["age_seconds"] = active["age_seconds"]
        return state
    if durable_status in {"failed", "ready"}:
        state.update(durable)
    elif durable_status == "staging":
        state.update(durable)
        state["status"] = "unknown"
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
    config_path = Path(adapter_path) / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
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


def _read_verified_adapter_identity(source_url: str, destination) -> dict[str, object]:
    """Read the identity already verified when the cache reached READY."""
    if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
        raise RuntimeError("adapter source URL is not remote")
    ready = destination / "READY"
    if ready.is_symlink() or not ready.is_file() or ready.read_text(encoding="ascii") != "ready\n":
        raise RuntimeError("adapter cache is not ready")

    manifest_path = destination / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("adapter cache manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    if (
        not isinstance(target_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", target_sha256) is None
        or type(target_size) is not int
        or target_size <= 0
    ):
        raise RuntimeError("adapter cache target identity is invalid")
    config_sha256 = config.get("sha256")
    config_size = config.get("size")
    if (
        not isinstance(config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
        or type(config_size) is not int
        or config_size < 0
    ):
        raise RuntimeError("adapter cache configuration identity is invalid")
    for name, record in (("adapter_model.safetensors", target), ("adapter_config.json", config)):
        size = record.get("size")
        digest = record.get("sha256")
        path = destination / name
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int
            or size < 0
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise RuntimeError(f"adapter cache {name} identity is invalid")
    config_path = destination / "adapter_config.json"
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("adapter cache configuration cannot be read") from exc
    if len(config_bytes) != config_size or hashlib.sha256(config_bytes).hexdigest() != config_sha256:
        raise RuntimeError("adapter cache configuration bytes differ from manifest")
    sidecar = destination / "adapter_model.safetensors.sha256"
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != target_sha256:
        raise RuntimeError("adapter cache target sidecar differs from manifest")

    identity: dict[str, object] = {
        "source_url": source_url,
        "target_sha256": target_sha256,
        "target_size": target_size,
        "config_sha256": config_sha256,
        "config_size": config_size,
    }

    transport_path = destination / "transport.json"
    if transport_path.exists():
        if transport_path.is_symlink() or not transport_path.is_file():
            raise RuntimeError("adapter cache transport record is invalid")
        try:
            transport = json.loads(transport_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("adapter cache transport record is invalid") from exc
        if not isinstance(transport, dict) or transport.get("source_url") != source_url:
            raise RuntimeError("adapter cache transport source differs from manifest")
        if transport.get("target_sha256") != target_sha256:
            raise RuntimeError("adapter cache transport target differs from manifest")
        if transport.get("config_sha256") not in (None, config_sha256):
            raise RuntimeError("adapter cache transport configuration differs from manifest")
    return identity


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
                    await asyncio.to_thread(
                        stage_lora_adapter_files,
                        source_url,
                        require_delta=require_delta,
                        policy_step=policy_step,
                        stage_timeout_seconds=(
                            adapter_stage_limit_seconds(policy_step)
                            if policy_step is not None
                            else None
                        ),
                    )
                destination = adapter_cache_path(source_url)
                lora_request.lora_path = destination.as_posix()
            lora_request.load_inplace = True
            if remote_adapter:
                with adapter_reader_lock(destination):
                    adapter_identity = _read_verified_adapter_identity(source_url, destination)
                    try:
                        _validate_adapter_target_compatibility(destination, handler)
                    except ValueError as exc:
                        return JSONResponse({"error": str(exc)}, status_code=400)
                    response = await handler.load_lora_adapter(lora_request)
            else:
                _loaded_lora_receipts.pop(lora_request.lora_name, None)
                try:
                    _validate_adapter_target_compatibility(lora_request.lora_path, handler)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
                response = await handler.load_lora_adapter(lora_request)
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
    started = time.perf_counter()
    stage_deadline = time.monotonic() + stage_timeout if stage_timeout is not None else None
    cancel_event = threading.Event()

    class CommitEvent(threading.Event):
        committed_at: float | None = None

        def set(self):
            if self.committed_at is None:
                self.committed_at = time.monotonic()
            super().set()

    committed_event = CommitEvent()
    stage_task = asyncio.create_task(
        asyncio.to_thread(
            stage_lora_adapter_files,
            source_url,
            require_delta=require_delta,
            stage_timeout_seconds=stage_timeout,
            policy_step=policy_step,
            cancel_event=cancel_event,
            committed_event=committed_event,
        )
    )

    async def monitor_disconnect() -> None:
        while not stage_task.done():
            if await raw_request.is_disconnected():
                with _adapter_stage_publish_lock:
                    if not committed_event.is_set():
                        cancel_event.set()
                return
            await asyncio.sleep(0.25)

    disconnect_task = asyncio.create_task(monitor_disconnect())

    def success_payload(destination):
        payload = {
            "status": "ok",
            "path": destination.as_posix(),
            "stage_seconds": time.perf_counter() - started,
        }
        if policy_step is not None:
            payload.update(
                {
                    "policy_step": policy_step,
                    "stage_limit_seconds": stage_limit,
                    "acceptance_target_seconds": 300.0,
                }
            )
        payload["adapter_identity"] = _read_verified_adapter_identity(source_url, destination)
        stage_state = adapter_stage_status()
        if isinstance(stage_state.get("reconstruct_seconds"), (int, float)):
            payload["reconstruct_seconds"] = float(stage_state["reconstruct_seconds"])
        return payload

    def timeout_response(*, cleanup_pending: bool = False):
        payload = {
            "error": "adapter delta staging exceeded its time limit",
            "policy_step": policy_step,
            "stage_limit_seconds": stage_limit,
            "acceptance_target_seconds": 300.0,
        }
        if committed_event.is_set():
            payload["committed"] = True
        if cleanup_pending:
            payload["cleanup_pending"] = True
        return JSONResponse(
            payload,
            status_code=504,
        )

    post_commit_cleanup_timeout_s = (
        min(30.0, max(1.0, stage_timeout)) if stage_timeout is not None else 30.0
    )

    def committed_within_deadline() -> bool:
        if stage_deadline is None:
            return True
        committed_at = committed_event.committed_at
        return committed_at is not None and committed_at < stage_deadline

    try:
        try:
            destination = await asyncio.wait_for(
                asyncio.shield(stage_task),
                timeout=stage_timeout,
            )
        except asyncio.TimeoutError:
            if stage_task.done():
                try:
                    destination = stage_task.result()
                except asyncio.TimeoutError:
                    return timeout_response()
                except BaseException:
                    raise
                if (
                    committed_event.is_set()
                    and committed_within_deadline()
                ):
                    return success_payload(destination)
                return timeout_response()
            with _adapter_stage_publish_lock:
                if not committed_event.is_set():
                    cancel_event.set()
            cleanup_pending = False
            try:
                destination = await asyncio.wait_for(
                    asyncio.shield(stage_task),
                    timeout=post_commit_cleanup_timeout_s,
                )
            except asyncio.TimeoutError:
                cleanup_pending = True
                cancel_event.set()
                try:
                    destination = await stage_task
                except BaseException:
                    destination = None
            except BaseException:
                destination = None
            if (
                committed_event.is_set()
                and destination is not None
                and committed_within_deadline()
            ):
                return success_payload(destination)
            if cleanup_pending:
                return timeout_response(cleanup_pending=True)
            return timeout_response()
        except asyncio.CancelledError:
            with _adapter_stage_publish_lock:
                if not committed_event.is_set():
                    cancel_event.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(stage_task),
                    timeout=post_commit_cleanup_timeout_s,
                )
            except asyncio.TimeoutError:
                cancel_event.set()
                try:
                    await stage_task
                except BaseException:
                    pass
            except BaseException:
                pass
            raise
        if not committed_within_deadline():
            return timeout_response()
        return success_payload(destination)
    finally:
        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass
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
