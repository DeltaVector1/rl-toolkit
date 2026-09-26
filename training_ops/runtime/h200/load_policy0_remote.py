"""Run policy-0 adapter client calls inside the H200 inference container."""

from __future__ import annotations

import asyncio
import json
import os
import sys


def _category(error: BaseException) -> str:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if "transport" in name or "connect" in message or "connection" in message or "timeout" in message:
        return "tunnel"
    if "status" in name or "http" in name or "response" in message:
        return "http"
    if any(marker in message for marker in ("identity", "checksum", "receipt", "source_url", "adapter")):
        return "identity"
    return "config"


async def _run(request: dict[str, object]) -> dict[str, object]:
    try:
        from prime_rl.configs.shared import ClientConfig
        from prime_rl.utils.client import (
            load_lora_adapter,
            setup_admin_clients,
            stage_lora_adapter,
        )
    except Exception as error:
        return {"ok": False, "category": "config", "error": str(error)}
    try:
        admin_urls = request["admin_urls"]
        origin_urls = request["origin_urls"]
        source_url = request["source_url"]
        slot = request["qualification_slot"]
        if (
            not isinstance(admin_urls, list)
            or not admin_urls
            or not all(isinstance(value, str) for value in admin_urls)
            or not isinstance(origin_urls, list)
            or not origin_urls
            or not all(isinstance(value, str) for value in origin_urls)
            or not isinstance(source_url, str)
            or not isinstance(slot, str)
            or not slot
        ):
            raise ValueError("remote policy-0 request has invalid endpoint or slot fields")
        config = ClientConfig(
            base_url=list(admin_urls),
            admin_base_url=list(admin_urls),
            adapter_origin_base_urls=list(origin_urls),
        )
        clients = setup_admin_clients(config)
        try:
            operation = request.get("operation")
            if operation == "stage":
                result = await stage_lora_adapter(
                    clients,
                    source_url,
                    require_delta=False,
                    policy_step=None,
                    origin_base_urls=list(origin_urls),
                )
            elif operation == "load":
                expected = request.get("expected_identity")
                result = await load_lora_adapter(
                    clients,
                    slot,
                    source_url,
                    require_delta=False,
                    prefetched=True,
                    expected_identity=expected,
                )
            else:
                raise ValueError("remote policy-0 operation is invalid")
        finally:
            await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)
        return {"ok": True, "result": result}
    except Exception as error:
        return {"ok": False, "category": _category(error), "error": str(error)}


def main() -> int:
    try:
        raw_request = os.environ.get("POLICY0_REQUEST_JSON")
        request = json.loads(raw_request) if raw_request is not None else json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("remote policy-0 request is not an object")
        result = asyncio.run(_run(request))
    except Exception as error:
        result = {"ok": False, "category": "config", "error": str(error)}
    print(json.dumps(result, separators=(",", ":")))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
