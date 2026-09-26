import asyncio
import inspect
import json
from unittest.mock import patch

import pytest

httpx = pytest.importorskip("httpx")

from training_ops.runtime.h200 import load_policy0_remote as remote


SOURCE_URL = "http://127.0.0.1:18100/run/run_run/broadcasts/step_0"
IDENTITY = {
    "source_url": SOURCE_URL,
    "target_sha256": "a" * 64,
    "target_size": 123,
    "config_sha256": "b" * 64,
    "config_size": 456,
}


def request(operation, **extra):
    value = {
        "operation": operation,
        "admin_urls": ["http://127.0.0.1:8000/v1"],
        "origin_urls": ["http://127.0.0.1:8000/v1"],
        "source_url": SOURCE_URL,
        "qualification_slot": "qualification-policy-0",
    }
    value.update(extra)
    return value


def receipt(server="instance-a"):
    return {
        **IDENTITY,
        "lora_name": "qualification-policy-0",
        "server_instance": server,
    }


def run_remote(handler, value):
    transport = httpx.MockTransport(handler)
    real_client = remote.httpx.AsyncClient
    with patch.object(
        remote.httpx,
        "AsyncClient",
        lambda: real_client(transport=transport),
    ):
        return asyncio.run(remote._run(value))


def test_stage_uses_h200_admin_root_contract_and_exact_identity():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url == "http://127.0.0.1:8000/stage_lora_adapter"
        assert json.loads(request.content) == {
            "lora_path": SOURCE_URL,
            "require_delta": False,
        }
        return httpx.Response(
            200,
            json={"status": "ok", "adapter_identity": IDENTITY},
            request=request,
        )

    result = run_remote(handler, request("stage"))
    assert result == {"ok": True, "result": {"status": "ok", "adapter_identity": IDENTITY}}
    assert len(seen) == 1


def test_load_uses_exact_payload_and_checks_staging_status_receipt():
    seen = []
    loaded = receipt()

    def handler(request):
        seen.append(request)
        if request.url.path == "/load_lora_adapter":
            assert json.loads(request.content) == {
                "lora_name": "qualification-policy-0",
                "lora_path": SOURCE_URL,
                "require_delta": False,
                "prefetched": True,
            }
            return httpx.Response(200, json={"status": "ok", "loaded_receipt": loaded}, request=request)
        assert request.url.path == "/staging_status"
        return httpx.Response(200, json={"loaded_receipts": {"qualification-policy-0": loaded}}, request=request)

    result = run_remote(handler, request("load", expected_identity=IDENTITY))
    assert result == {"ok": True, "result": [loaded]}
    assert [item.url.path for item in seen] == ["/load_lora_adapter", "/staging_status"]


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (httpx.Response(503, text="unavailable"), "http"),
        (httpx.Response(200, text="not-json"), "config"),
    ],
)
def test_stage_http_and_json_failures_are_structured(response, category):
    def handler(request):
        response.request = request
        return response

    result = run_remote(handler, request("stage"))
    assert result["ok"] is False
    assert result["category"] == category


def test_http_status_preserves_status_and_response_body():
    def handler(request):
        return httpx.Response(504, text="server restart required", request=request)

    result = run_remote(handler, request("stage"))
    assert result["ok"] is False
    assert result["category"] == "http"
    assert "HTTP 504" in result["error"]
    assert "server restart required" in result["error"]


def test_timeout_is_reported_as_tunnel_failure():
    def handler(request):
        raise httpx.ReadTimeout("server did not answer", request=request)

    result = run_remote(handler, request("stage"))
    assert result["ok"] is False
    assert result["category"] == "tunnel"


def test_load_timeout_reports_exact_staging_status_without_retry():
    loaded = receipt()
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/load_lora_adapter":
            raise httpx.ReadTimeout("response did not finish", request=request)
        return httpx.Response(200, json={"loaded_receipts": {"qualification-policy-0": loaded}}, request=request)

    result = run_remote(handler, request("load", expected_identity=IDENTITY))
    assert result["ok"] is False
    assert result["category"] == "tunnel"
    assert "observed staging status" in result["error"]
    assert "instance-a" in result["error"]
    assert calls == ["/load_lora_adapter", "/staging_status"]


def test_load_post_and_diagnostic_status_share_one_absolute_deadline():
    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    seen = []
    deadlines = []

    async def fake_request_all(_client, _urls, method, _path, *, timeout_seconds, **_kwargs):
        seen.append((method, timeout_seconds))
        if method == "POST":
            raise httpx.ReadTimeout("response did not finish")
        return [{"status": "staging", "loaded_receipts": {}}]

    def fake_remaining(deadline):
        deadlines.append(deadline)
        return 600.0 if len(deadlines) == 1 else 599.0

    with patch.object(remote.httpx, "AsyncClient", lambda: DummyClient()), patch.object(
        remote, "_request_all", side_effect=fake_request_all
    ), patch.object(remote, "_remaining", side_effect=fake_remaining):
        result = asyncio.run(remote._run(request("load", expected_identity=IDENTITY)))
    assert result["ok"] is False
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert seen == [("POST", 600.0), ("GET", 599.0)]


def test_identity_and_receipt_mismatch_fail_closed():
    bad = dict(IDENTITY, source_url="http://127.0.0.1:18100/other/step_0")

    def stage_handler(request):
        return httpx.Response(200, json={"status": "ok", "adapter_identity": bad}, request=request)

    result = run_remote(stage_handler, request("stage"))
    assert result["ok"] is False
    assert result["category"] == "identity"

    def load_handler(request):
        return httpx.Response(200, json={"status": "ok", "loaded_receipt": receipt()}, request=request)

    mismatch = dict(IDENTITY, config_sha256="c" * 64)
    result = run_remote(load_handler, request("load", expected_identity=mismatch))
    assert result["ok"] is False
    assert result["category"] == "identity"


def test_helper_has_no_unavailable_prime_rl_imports():
    source = inspect.getsource(remote)
    assert "prime_rl" not in source
