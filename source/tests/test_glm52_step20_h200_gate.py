import importlib.util
import json
import os
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/h200/glm52_step20_gate.py"
SPEC = importlib.util.spec_from_file_location("glm52_step20_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def inventory() -> dict:
    return gate.load_inventory(ROOT / "configs/operations.toml")


def profile() -> dict:
    with (ROOT / "configs/qualification/glm52-step20-h200.toml").open("rb") as stream:
        value = tomllib.load(stream)
    value["profile_sha256"] = gate.sha256_file(
        ROOT / "configs/qualification/glm52-step20-h200.toml"
    )
    value["trainer_adapter"] = inventory()["trainer_adapter"]
    return value


def test_step20_h200_profile_is_production_shaped() -> None:
    value = profile()
    assert value["engine_config"] == (
        "configs/qualification/glm52-policy-parity-h200.toml"
    )
    assert value["adapter_path"] == inventory()["adapter"]
    assert value["requests"] == 1024
    assert value["group_size"] == 16
    assert value["max_inflight"] == 128
    assert value["max_output_tokens"] == 16384
    assert value["context_length"] == 32768
    assert value["tensor_parallel_size"] == 8
    assert value["cuda_graph"] is True
    assert value["cuda_graph_capture_size"] == 256
    assert value["prefix_cache"] is True
    assert value["kv_cache_dtype"] == "auto"
    assert value["max_num_seqs"] == 128
    assert value["max_num_batched_tokens"] == 32769
    assert value["batch_seconds_min"] == 480
    assert value["batch_seconds_max"] == 720
    assert value["min_output_tokens"] == 16384
    assert value["max_retries"] == 2
    assert value["retry_count_max"] == 32
    assert value["qualification_deadline_seconds"] == 720
    assert value["peak_vram_mib_max"] == 143000
    assert isinstance(value["seed"], int)


def test_step20_h200_profile_rejects_unpinned_fields(tmp_path: Path) -> None:
    source = ROOT / "configs/qualification/glm52-step20-h200.toml"
    path = tmp_path / source.name
    path.write_text(source.read_text() + "unbounded_option = true\n")
    try:
        gate.load_config(path, inventory())
    except ValueError as error:
        assert "fields" in str(error)
    else:
        raise AssertionError("an unpinned H200 option was accepted")


def test_step20_adapter_result_requires_exact_step_path_and_hash(tmp_path: Path) -> None:
    result_path = tmp_path / "qualification-result.json"
    adapter_sha = "a" * 64
    result_path.write_text(
        json.dumps(
            {
                "kind": "glm52_trainer_qualification",
                "profile": "glm52-native-stability20",
                "session_sha256": "b" * 64,
                "adapter": {
                    "step": 20,
                    "path": inventory()["trainer_adapter"],
                    "sha256": adapter_sha,
                    "checksum_value": adapter_sha,
                    "size": 300_000_000,
                    "config_sha256": "c" * 64,
                    "stable_sha256": "d" * 64,
                    "checksum_sha256": "e" * 64,
                },
            }
        )
    )
    assert gate.read_adapter(
        result_path, inventory()["trainer_adapter"]
    )["sha256"] == adapter_sha
    value = json.loads(result_path.read_text())
    value["adapter"]["path"] = "/data/x/broadcasts/step_19/adapter_model.safetensors"
    result_path.write_text(json.dumps(value))
    try:
        gate.read_adapter(result_path, inventory()["trainer_adapter"])
    except ValueError as error:
        assert "path or checksum" in str(error)
    else:
        raise AssertionError("step-19 adapter was accepted")


def valid_result() -> tuple[dict, dict]:
    config = profile()
    result = {
        "schema_version": 1,
        "kind": "glm52_step20_h200_gate",
        "profile_sha256": config["profile_sha256"],
        "engine": {
            "config": "configs/qualification/glm52-policy-parity-h200.toml",
            "image_digest": "sha256:" + "f" * 64,
            "config_sha256": "e" * 64,
            "tensor_parallel_size": 8,
            "cuda_graph": True,
            "cuda_graph_capture_size": 256,
            "prefix_cache": True,
            "kv_cache_dtype": "auto",
            "max_num_seqs": 128,
            "max_num_batched_tokens": 32769,
        },
        "adapter": {
            "name": config["model"],
            "trainer_path": inventory()["trainer_adapter"],
            "h200_path": inventory()["adapter"],
            "sha256": "a" * 64,
            "size": 300_000_000,
            "qualification_result_sha256": "b" * 64,
            "qualification_session_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "stable_sha256": "e" * 64,
            "checksum_sha256": "f" * 64,
        },
        "settings": {
            name: config[name]
            for name in gate.EXPECTED
            if name not in {"schema_version", "name"}
        },
        "parity": {
            "capture_manifest_sha256": "1" * 64,
            "adapter_sha256": "a" * 64,
            "graph_eager_guard": {"passed": True},
            "router_runtime_marker_sha256": "2" * 64,
            "array_sha256": {
                "full_logprobs": "3" * 64,
                "router_ids": "4" * 64,
                "router_weights": "5" * 64,
            },
        },
        "measurements": {
            "batch_seconds": 600.0,
            "completed_requests": 1024,
            "failed_requests": 0,
            "retries": 3,
            "retry_errors": ["H200RequestError: reset"] * 3,
            "output_tokens": 1024 * 16384,
            "output_tokens_per_second": (1024 * 16384) / 600.0,
            "max_prompt_tokens": 128,
            "peak_vram_mib": [120000] * 8,
            "latency_seconds": {
                "mean": 500.0,
                "p50": 490.0,
                "p95": 610.0,
                "p99": 650.0,
                "max": 680.0,
            },
            "first_errors": [],
        },
    }
    return result, config


def test_step20_h200_result_rejects_bounds_failures_and_mixed_settings() -> None:
    result, config = valid_result()
    assert gate.validate_result(result, config) == []
    result["measurements"]["batch_seconds"] = 721.0
    result["measurements"]["failed_requests"] = 1
    result["measurements"]["max_prompt_tokens"] = 16385
    result["settings"]["max_inflight"] = 127
    failures = gate.validate_result(result, config)
    assert any("480-720" in failure for failure in failures)
    assert any("failed requests" in failure for failure in failures)
    assert any("exceeds context" in failure for failure in failures)
    assert any("settings differ" in failure for failure in failures)


def test_step20_h200_result_rejects_oom_retry_and_vram_limit() -> None:
    result, config = valid_result()
    result["measurements"]["retry_errors"] = ["HTTP 500: CUDA out of memory"]
    result["measurements"]["retries"] = 1
    result["measurements"]["peak_vram_mib"][3] = 143001
    failures = gate.validate_result(result, config)
    assert any("OOM" in failure for failure in failures)
    assert any("VRAM" in failure for failure in failures)


def test_step20_gate_atomic_write_has_exact_content(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    value = {"kind": "test", "value": 1}
    gate.write_atomic(destination, value)
    assert json.loads(destination.read_text()) == value
    assert destination.stat().st_mode & 0o777 == 0o644
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]


def test_staged_adapter_requires_exact_files_hashes_and_size(tmp_path: Path) -> None:
    files = {
        "adapter_model.safetensors": b"weights",
        "adapter_config.json": b"{}\n",
        "STABLE": b"stable\n",
    }
    files["adapter_model.safetensors.sha256"] = (
        gate.hashlib.sha256(files["adapter_model.safetensors"]).hexdigest() + "\n"
    ).encode()
    for name, payload in files.items():
        (tmp_path / name).write_bytes(payload)
    identity = {
        "sha256": gate.sha256_file(tmp_path / "adapter_model.safetensors"),
        "size": len(files["adapter_model.safetensors"]),
        "config_sha256": gate.sha256_file(tmp_path / "adapter_config.json"),
        "stable_sha256": gate.sha256_file(tmp_path / "STABLE"),
        "checksum_sha256": gate.sha256_file(
            tmp_path / "adapter_model.safetensors.sha256"
        ),
    }
    gate.verify_staged_adapter(tmp_path, identity)
    (tmp_path / "adapter_config.json").write_bytes(b"changed\n")
    try:
        gate.verify_staged_adapter(tmp_path, identity)
    except ValueError as error:
        assert "adapter_config.json" in str(error)
    else:
        raise AssertionError("changed H200 adapter configuration was accepted")


def test_adapter_load_refuses_alias_and_proves_fresh_load(monkeypatch) -> None:
    calls = []

    def existing(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"data": [{"id": gate.EXPECTED_MODEL}]}

    monkeypatch.setattr(gate, "request_json", existing)
    try:
        gate.load_adapter(
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8100",
            gate.EXPECTED_MODEL,
            Path(inventory()["adapter"]),
        )
    except RuntimeError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("pre-existing adapter alias was accepted")
    assert len(calls) == 1

    replies = iter(
        [
            {"data": []},
            {"status": "ok"},
            {"data": [{"id": gate.EXPECTED_MODEL}]},
        ]
    )
    calls.clear()

    def fresh(url, payload, timeout):
        calls.append((url, payload, timeout))
        return next(replies)

    monkeypatch.setattr(gate, "request_json", fresh)
    gate.load_adapter(
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8100",
        gate.EXPECTED_MODEL,
        Path(inventory()["adapter"]),
    )
    assert calls[1][0] == "http://127.0.0.1:8100/load_lora_adapter"
    assert calls[1][1] == {
        "lora_name": gate.EXPECTED_MODEL,
        "lora_path": inventory()["adapter"],
    }


def test_parity_capture_binds_graph_routes_and_adapter(tmp_path: Path) -> None:
    adapter_sha = "a" * 64
    array_hashes = {}
    for name in ("full_logprobs", "router_ids", "router_weights"):
        array_path = tmp_path / f"{name}.npy"
        array_path.write_bytes(name.encode())
        array_hashes[name] = gate.sha256_file(array_path)
    capture = {
        "kind": "glm52_policy_parity_h200_capture",
        "adapter_sha256": adapter_sha,
        "phases": {
            "graph": {"adapter_sha256": adapter_sha, "mode": "normal_cuda_graph"},
            "eager_trace": {
                "adapter_sha256": adapter_sha,
                "mode": "force_eager_skip_compiled",
            },
        },
        "graph_eager_guard": {"passed": True},
        "sources": {
            "h200_trace_logprobs": "normal_cuda_graph_request",
            "router_ids": "isolated_eager_trace_request",
            "router_weights": "isolated_eager_trace_request",
        },
        "router_runtime_marker": {"layer_calls": {"3": 1}},
        "arrays": {
            name: {"file": f"{name}.npy", "sha256": digest}
            for name, digest in array_hashes.items()
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(capture))
    evidence = gate.read_parity_capture(path, adapter_sha)
    assert evidence["adapter_sha256"] == adapter_sha
    capture["phases"]["graph"]["adapter_sha256"] = "b" * 64
    path.write_text(json.dumps(capture))
    try:
        gate.read_parity_capture(path, adapter_sha)
    except ValueError as error:
        assert "not bound" in str(error)
    else:
        raise AssertionError("mixed-adapter graph evidence was accepted")
    capture["phases"]["graph"]["adapter_sha256"] = adapter_sha
    path.write_text(json.dumps(capture))
    (tmp_path / "router_ids.npy").write_bytes(b"changed")
    try:
        gate.read_parity_capture(path, adapter_sha)
    except ValueError as error:
        assert "router_ids" in str(error)
    else:
        raise AssertionError("changed H200 route evidence was accepted")


def test_step20_benchmark_records_retry_and_exact_request_fields(monkeypatch) -> None:
    config = profile()
    config.update(requests=2, group_size=1, max_inflight=2)
    payloads = []
    urls = []
    attempts = 0

    def request(url, payload, _timeout):
        nonlocal attempts
        attempts += 1
        urls.append(url)
        payloads.append(payload)
        if attempts == 1:
            raise TimeoutError("retry")
        return {"usage": {"prompt_tokens": 12, "completion_tokens": 16384}}

    monkeypatch.setattr(gate, "request_json", request)
    monkeypatch.setattr(gate, "gpu_memory_sample", lambda: [1000] * 8)
    result = gate.benchmark(config, "http://127.0.0.1:8000", config["model"])
    assert result["completed_requests"] == 2
    assert result["failed_requests"] == 0
    assert result["retries"] == 1
    assert result["retry_errors"] == ["TimeoutError: retry"]
    assert set(urls) == {"http://127.0.0.1:8000/v1/chat/completions"}
    assert all(payload["max_tokens"] == 16384 for payload in payloads)
    assert all(payload["min_tokens"] == 16384 for payload in payloads)
    assert all(payload["messages"][0]["role"] == "user" for payload in payloads)
    assert {payload["seed"] for payload in payloads} == {20260824, 20260825}


def test_step20_benchmark_uses_one_global_deadline(monkeypatch) -> None:
    config = profile()
    config.update(
        requests=1,
        group_size=1,
        max_inflight=1,
        qualification_deadline_seconds=0.01,
    )
    observed_timeouts = []

    def request(_url, _payload, timeout):
        observed_timeouts.append(timeout)
        gate.time.sleep(0.02)
        raise TimeoutError("slow transport")

    monkeypatch.setattr(gate, "request_json", request)
    monkeypatch.setattr(gate, "gpu_memory_sample", lambda: [1000] * 8)
    result = gate.benchmark(config, "http://127.0.0.1:8000", config["model"])
    assert result["completed_requests"] == 0
    assert result["failed_requests"] == 1
    assert observed_timeouts and observed_timeouts[0] <= 0.01
    assert result["batch_seconds"] < 0.2


def test_step20_gate_script_has_no_host_inventory() -> None:
    source = SCRIPT.read_text()
    assert "n79" not in source
    assert "10.92." not in source
    assert "152.236." not in source
    assert os.access(SCRIPT, os.R_OK)
