"""Loopback-only Muse bootstrap and checkpoint-staging endpoints."""

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest

from prime_rl.inference.vllm.worker.filesystem import (
    adapter_cache_path,
    adapter_reader_lock,
    adapter_stage_status,
    stage_full_checkpoint_files,
    stage_lora_adapter_files,
)


_FILES = {
    ".gitattributes": Path("/data/muse-glimmer/model/.gitattributes"),
    "LICENSE": Path("/data/muse-glimmer/model/LICENSE"),
    "README.md": Path("/data/muse-glimmer/model/README.md"),
    "USAGE_POLICY.md": Path("/data/muse-glimmer/model/USAGE_POLICY.md"),
    "chat_template.jinja": Path("/data/muse-glimmer/model/chat_template.jinja"),
    "config.json": Path("/data/muse-glimmer/model/config.json"),
    "generation_config.json": Path("/data/muse-glimmer/model/generation_config.json"),
    "model-00001-of-00002.safetensors": Path("/data/muse-glimmer/model/model-00001-of-00002.safetensors"),
    "model-00002-of-00002.safetensors": Path("/data/muse-glimmer/model/model-00002-of-00002.safetensors"),
    "model.safetensors.index.json": Path("/data/muse-glimmer/model/model.safetensors.index.json"),
    "processor_config.json": Path("/data/muse-glimmer/model/processor_config.json"),
    "tokenizer.json": Path("/data/muse-glimmer/model/tokenizer.json"),
    "tokenizer_config.json": Path("/data/muse-glimmer/model/tokenizer_config.json"),
    "transformers-5.15.tar": Path("/data/muse-glimmer/transformers-5.15.tar"),
}
_LORA_LOAD_LOCK = asyncio.Lock()


class MuseFileEndpoint:
    name = "muse_file_server"
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        def require_loopback(request: Request) -> None:
            if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
                raise HTTPException(status_code=403, detail="loopback access only")

        @app.get("/muse-files/{filename}")
        async def get_file(filename: str, request: Request):
            require_loopback(request)
            path = _FILES.get(filename)
            if path is None or not path.is_file():
                raise HTTPException(status_code=404, detail="file not found")
            size = path.stat().st_size
            start, end = 0, size - 1
            value = request.headers.get("range")
            status = 200
            headers = {"Accept-Ranges": "bytes"}
            if value:
                try:
                    first, last = value.removeprefix("bytes=").split("-", 1)
                    start = int(first)
                    end = min(int(last), end) if last else end
                    if start < 0 or start > end:
                        raise ValueError
                except ValueError as exc:
                    raise HTTPException(status_code=416, detail="invalid byte range") from exc
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            headers["Content-Length"] = str(end - start + 1)

            def chunks():
                remaining = end - start + 1
                with path.open("rb") as source:
                    source.seek(start)
                    while remaining:
                        data = source.read(min(16 * 1024 * 1024, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data

            return StreamingResponse(chunks(), status_code=status, headers=headers)

        @app.post("/stage_full_checkpoint")
        async def stage_full_checkpoint(request: Request):
            """Download one checkpoint per host without blocking TP workers."""
            require_loopback(request)
            data = await request.json()
            source_url = data.get("checkpoint_url")
            if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail="checkpoint_url must be an HTTP URL")
            destination = await asyncio.to_thread(stage_full_checkpoint_files, source_url)
            return {"status": "ok", "path": destination.as_posix()}

        @app.post("/stage_lora_adapter")
        async def stage_lora_adapter(request: Request):
            """Download a remote adapter without stopping generation."""
            require_loopback(request)
            data = await request.json()
            source_url = data.get("lora_path")
            if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
                raise HTTPException(status_code=400, detail="lora_path must be an HTTP URL")
            destination = await asyncio.to_thread(stage_lora_adapter_files, source_url)
            return {"status": "ok", "path": destination.as_posix()}

        @app.get("/staging_status")
        async def staging_status(request: Request):
            """Expose durable transfer progress to the watchdog and operator."""
            require_loopback(request)
            return adapter_stage_status()

        @app.post("/pause")
        async def pause(request: Request):
            """Stop admission and abort registered decode requests for a policy load."""
            require_loopback(request)
            mode = request.query_params.get("mode", "abort")
            if mode not in {"abort", "keep"}:
                raise HTTPException(status_code=400, detail=f"unsupported pause mode: {mode}")
            clear_cache = request.query_params.get("clear_cache", "true").lower() == "true"
            await request.app.state.engine_client.pause_generation(mode=mode, clear_cache=clear_cache)
            return {"status": "paused"}

        @app.post("/resume")
        async def resume(request: Request):
            require_loopback(request)
            await request.app.state.engine_client.resume_generation()
            return {"status": "resumed"}

        @app.post("/load_lora_adapter")
        async def load_lora_adapter(request: Request):
            """Stage a remote adapter and load it through vLLM's native handler."""
            require_loopback(request)
            data = await request.json()
            lora_request = LoadLoRAAdapterRequest.model_validate(data)
            async with _LORA_LOAD_LOCK:
                source_url = None
                destination = None
                if lora_request.lora_path.startswith(("http://", "https://")):
                    source_url = lora_request.lora_path
                    destination = await asyncio.to_thread(stage_lora_adapter_files, source_url)
                    expected = adapter_cache_path(source_url)
                    if destination != expected:
                        raise RuntimeError("adapter cache path changed during staging")
                    lora_request.lora_path = destination.as_posix()
                lora_request.load_inplace = True
                handler = request.app.state.openai_serving_models
                if source_url is not None:
                    # Keep a shared lease until vLLM has finished opening the
                    # safetensors. A replacement stage may retire this cache
                    # directory, but must not delete it while it is read.
                    with adapter_reader_lock(destination):
                        response = await handler.load_lora_adapter(lora_request)
                else:
                    response = await handler.load_lora_adapter(lora_request)
                stored = handler.lora_requests.get(lora_request.lora_name)
                if stored is not None:
                    stored.load_inplace = False
                if isinstance(response, ErrorResponse):
                    return JSONResponse(content=response.model_dump(), status_code=response.error.code)
                return {"status": "ok"}

    async def init_state(self, engine_client, state, args) -> None:
        return None


def make_plugin() -> MuseFileEndpoint:
    return MuseFileEndpoint()
