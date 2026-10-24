"""Run policy-0 adapter requests inside the H200 inference container."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from urllib.parse import urlsplit

import httpx


stage_budget_seconds = 1800.0
load_budget_seconds = 600.0
request_connect_timeout_seconds = 10.0
request_write_timeout_seconds = 60.0


def _category(error: BaseException) -> str:
    name = type(error).__name__.lower()
    message = str(error).lower()
    if isinstance(error, httpx.HTTPStatusError) or "status" in name or "http" in name:
        return "http"
    if "transport" in name or "connect" in message or "connection" in message or "timeout" in name or "timeout" in message:
        return "tunnel"
    if any(marker in message for marker in ("identity", "checksum", "receipt", "source_url")):
        return "identity"
    return "config"


def _admin_root(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("admin endpoint is not a string")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("admin endpoint is not an HTTP URL")
    return value.rstrip("/").removesuffix("/v1")


def _request_timeout(seconds: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=min(request_connect_timeout_seconds, seconds),
        read=seconds,
        write=min(request_write_timeout_seconds, seconds),
        pool=min(request_connect_timeout_seconds, seconds),
    )


def _identity(value: object, source_url: str, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} identity is missing")
    result = dict(value)
    required = {
        "source_url": str,
        "target_sha256": str,
        "target_size": int,
        "config_sha256": str,
        "config_size": int,
    }
    for name, kind in required.items():
        item = result.get(name)
        if type(item) is not kind:
            raise ValueError(f"{label} identity has invalid {name}")
        if name == "source_url" and item != source_url:
            raise ValueError(f"{label} identity has a different source URL")
        if name.endswith("sha256") and (
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
        ):
            raise ValueError(f"{label} identity has invalid {name}")
        if name == "target_size" and item <= 0:
            raise ValueError(f"{label} identity has invalid {name}")
        if name == "config_size" and item < 0:
            raise ValueError(f"{label} identity has invalid {name}")
    return {name: result[name] for name in required}


def _receipt(
    value: object,
    source_url: str,
    slot: str,
    expected: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} receipt is missing")
    receipt = dict(value)
    identity = _identity(receipt, source_url, label)
    if receipt.get("lora_name") != slot:
        raise ValueError(f"{label} receipt has the wrong LoRA name")
    if not isinstance(receipt.get("server_instance"), str) or not receipt["server_instance"]:
        raise ValueError(f"{label} receipt has no server instance")
    for name, expected_value in expected.items():
        if name in identity and identity[name] != expected_value:
            raise ValueError(f"{label} receipt differs in {name}")
    return {
        **identity,
        "lora_name": slot,
        "server_instance": receipt["server_instance"],
    }


async def _json_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    body: dict[str, object] | None,
    timeout_seconds: float,
) -> dict[str, object]:
    response = await asyncio.wait_for(
        client.request(
            method,
            url,
            json=body,
            timeout=_request_timeout(timeout_seconds),
        ),
        timeout=timeout_seconds,
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        body = response.text[:2000]
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code} from {url}: {body}",
            request=error.request,
            response=error.response,
        ) from error
    try:
        payload = response.json()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid JSON response from {url}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON response from {url} is not an object")
    return payload


async def _request_all(
    client: httpx.AsyncClient,
    urls: list[str],
    method: str,
    path: str,
    *,
    body: dict[str, object] | None,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    tasks = [
        asyncio.create_task(
            _json_request(
                client,
                method,
                f"{url}{path}",
                body=body,
                timeout_seconds=timeout_seconds,
            )
        )
        for url in urls
    ]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _validate_request(request: Mapping[str, object]) -> tuple[list[str], str, str, str]:
    admin_values = request.get("admin_urls")
    origin_values = request.get("origin_urls")
    source_url = request.get("source_url")
    slot = request.get("qualification_slot")
    operation = request.get("operation")
    if (
        not isinstance(admin_values, list)
        or not admin_values
        or not all(isinstance(value, str) for value in admin_values)
        or not isinstance(origin_values, list)
        or not origin_values
        or not all(isinstance(value, str) for value in origin_values)
        or not isinstance(source_url, str)
        or not source_url.startswith(("http://", "https://"))
        or not isinstance(slot, str)
        or not slot
        or operation not in {"stage", "load"}
    ):
        raise ValueError("remote policy-0 request has invalid endpoint, source, slot, or operation fields")
    admin_urls = [_admin_root(value) for value in admin_values]
    [_admin_root(value) for value in origin_values]
    return admin_urls, source_url, slot, operation


async def _load_status(
    client: httpx.AsyncClient,
    admin_urls: list[str],
    source_url: str,
    slot: str,
    expected: Mapping[str, object],
    timeout_seconds: float,
) -> list[dict[str, object]]:
    statuses = await _request_all(
        client,
        admin_urls,
        "GET",
        "/staging_status",
        body=None,
        timeout_seconds=timeout_seconds,
    )
    receipts = []
    for index, payload in enumerate(statuses):
        loaded = payload.get("loaded_receipts")
        value = loaded.get(slot) if isinstance(loaded, dict) else None
        receipts.append(_receipt(value, source_url, slot, expected, f"endpoint {index} loaded"))
    return receipts


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError("policy-0 load phase exceeded its 600-second deadline")
    return remaining


async def _diagnostic_status(
    client: httpx.AsyncClient,
    admin_urls: list[str],
    deadline: float,
) -> str:
    try:
        statuses = await _request_all(
            client,
            admin_urls,
            "GET",
            "/staging_status",
            body=None,
            timeout_seconds=_remaining(deadline),
        )
        return json.dumps(statuses, sort_keys=True, separators=(",", ":"))[:4000]
    except Exception as error:
        return f"status read failed: {error}"


async def _run(request: dict[str, object]) -> dict[str, object]:
    try:
        admin_urls, source_url, slot, operation = _validate_request(request)
        expected_value = request.get("expected_identity")
        if not isinstance(expected_value, Mapping):
            raise ValueError(f"{operation} requires the expected adapter identity")
        expected = _identity(expected_value, source_url, "expected adapter")
        async with httpx.AsyncClient() as client:
            if operation == "stage":
                payload = {"lora_path": source_url, "require_delta": False}
                responses = await _request_all(
                    client,
                    admin_urls,
                    "POST",
                    "/stage_lora_adapter",
                    body=payload,
                    timeout_seconds=stage_budget_seconds,
                )
                identities = []
                for index, response in enumerate(responses):
                    if response.get("status") != "ok":
                        raise ValueError(f"endpoint {index} adapter stage failed")
                    identity = _identity(
                        response.get("adapter_identity"),
                        source_url,
                        f"endpoint {index} staged adapter",
                    )
                    if identity != expected:
                        raise ValueError(
                            f"endpoint {index} staged adapter differs from expected adapter identity"
                        )
                    identities.append(identity)
                if not identities or any(identity != identities[0] for identity in identities[1:]):
                    raise ValueError("inference endpoints reported different staged adapter identities")
                return {"ok": True, "result": {"status": "ok", "adapter_identity": identities[0]}}

            assert expected is not None
            payload = {
                "lora_name": slot,
                "lora_path": source_url,
                "require_delta": False,
                "prefetched": True,
            }
            load_deadline = asyncio.get_running_loop().time() + load_budget_seconds
            try:
                responses = await _request_all(
                    client,
                    admin_urls,
                    "POST",
                    "/load_lora_adapter",
                    body=payload,
                    timeout_seconds=_remaining(load_deadline),
                )
            except (httpx.TimeoutException, asyncio.TimeoutError) as error:
                diagnostic = await _diagnostic_status(client, admin_urls, load_deadline)
                raise TimeoutError(
                    f"load_lora_adapter timed out; observed staging status: {diagnostic}"
                ) from error
            loaded = []
            for index, response in enumerate(responses):
                if response.get("status") != "ok":
                    raise ValueError(f"endpoint {index} adapter load failed")
                loaded.append(_receipt(response.get("loaded_receipt"), source_url, slot, expected, f"endpoint {index} load"))
            status_receipts = await _load_status(
                client,
                admin_urls,
                source_url,
                slot,
                expected,
                _remaining(load_deadline),
            )
            if loaded != status_receipts:
                raise ValueError("staging status receipts differ from load responses")
            return {"ok": True, "result": loaded}
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
