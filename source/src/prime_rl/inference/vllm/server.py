import asyncio
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import uvloop
from fastapi import APIRouter, HTTPException, Request
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
from prime_rl.inference.patches import (
    monkey_patch_dp_coordinator_startup_timeout,
    monkey_patch_harmony_stop_token_propagation,
    monkey_patch_nano_v3_reasoning_parser,
    monkey_patch_strip_routed_experts_from_chat,
    monkey_patch_tokenize_params_validation,
)
from prime_rl.inference.vllm.worker.filesystem import adapter_cache_path
from prime_rl.parity_router_trace import (
    TraceSession,
    assemble_chunks,
    clear_trace,
    encoded_array,
    read_h200_runtime_marker,
    read_session,
    trace_root,
    write_session,
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


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def parity_trace_root() -> Path:
    root = trace_root()
    if root is None:
        raise HTTPException(status_code=404, detail="parity router trace is disabled")
    return root


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
}


@router.post("/pause")
async def pause(request: Request):
    logger.debug("Received /pause request (mode=keep, clear_cache=False)")
    await engine_client(request).pause_generation(mode="keep", clear_cache=False)
    return {"status": "paused"}


@router.post("/resume")
async def resume(request: Request):
    await engine_client(request).resume_generation()
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
        if os.environ.get("PRIME_MOE_LORA_FORMAT") == "2d":
            if not hasattr(lora_request, "is_3d_lora_weight"):
                raise RuntimeError("this vLLM version cannot load per-expert LoRA weights")
            lora_request.is_3d_lora_weight = False
        handler = models(raw_request)
        if lora_request.lora_path.startswith(("http://", "https://")):
            source_url = lora_request.lora_path
            await engine_client(raw_request).collective_rpc("stage_lora_adapter", args=(source_url,))
            lora_request.lora_path = adapter_cache_path(source_url).as_posix()
        lora_request.load_inplace = True
        response = await handler.load_lora_adapter(lora_request)
        stored = handler.lora_requests.get(lora_request.lora_name)
        if stored is not None:
            stored.load_inplace = False
        if isinstance(response, ErrorResponse):
            return JSONResponse(content=response.model_dump(), status_code=response.error.code)
        return {"status": "ok"}


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


@router.post("/parity/router-trace/reset")
async def reset_parity_router_trace(request: Request):
    root = parity_trace_root()
    data = await request.json()
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="router trace session must be an object")
    try:
        session = TraceSession.from_dict(data)
        if session.side != "h200":
            raise ValueError("H200 router trace session side must be h200")
        clear_trace(root)
        write_session(root, session)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return session.as_dict()


@router.get("/parity/router-trace")
async def get_parity_router_trace():
    root = parity_trace_root()
    try:
        session = read_session(root)
        if session.side != "h200":
            raise ValueError("H200 router trace session side must be h200")
        runtime_marker = read_h200_runtime_marker(root, session)
        ids, weights = assemble_chunks(root, session)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {
        "session": session.as_dict(),
        "router_ids": encoded_array(ids),
        "router_weights": encoded_array(weights),
        "runtime_marker": runtime_marker,
    }


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

    routed_moe_lora = any(
        "experts" in target.split(".")
        for target in (config.lora_target_modules or [])
    )
    if config.lora_target_modules and not routed_moe_lora:
        os.environ["PRIME_NO_MOE_LORA"] = "1"
    else:
        os.environ.pop("PRIME_NO_MOE_LORA", None)
    if routed_moe_lora:
        os.environ["PRIME_MOE_LORA_FORMAT"] = "2d"
    else:
        os.environ.pop("PRIME_MOE_LORA_FORMAT", None)

    namespace = config.to_vllm()
    if routed_moe_lora:
        namespace.enable_mixed_moe_lora_format = True
    if vllm_extra:
        for key, value in vllm_extra.items():
            setattr(namespace, key, value)

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=namespace)
    assert args is not None
    validate_parsed_serve_args(args)

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
