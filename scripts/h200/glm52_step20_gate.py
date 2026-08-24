#!/usr/bin/env python3
"""Measure the production-shaped GLM step-20 H200 policy."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import stat
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib

EXPECTED = {
    "schema_version": 1,
    "name": "glm52-step20-h200",
    "requests": 1024,
    "group_size": 16,
    "max_inflight": 128,
    "context_length": 32768,
    "max_output_tokens": 16384,
    "min_output_tokens": 16384,
    "temperature": 1.0,
    "top_p": 1.0,
    "seed": 20260824,
    "request_timeout_seconds": 700,
    "max_retries": 2,
    "retry_count_max": 32,
    "qualification_deadline_seconds": 720,
    "tensor_parallel_size": 8,
    "cuda_graph": True,
    "cuda_graph_capture_size": 256,
    "prefix_cache": True,
    "kv_cache_dtype": "auto",
    "max_num_seqs": 128,
    "max_num_batched_tokens": 32769,
    "batch_seconds_min": 480,
    "batch_seconds_max": 720,
    "peak_vram_mib_max": 143000,
    "evidence_max_age_seconds": 86400,
}
EXPECTED_MODEL = "glm52-step20-r32"
EXPECTED_ENGINE_CONFIG = "configs/qualification/glm52-policy-parity-h200.toml"
HEX64 = set("0123456789abcdef")


class H200RequestError(OSError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        operations = tomllib.load(stream)
    hosts = operations.get("hosts", {})
    inference = hosts.get("inference")
    gate = operations.get("gate", {}).get("glm52_step20_h200")
    ports = operations.get("ports", {})
    inference_inventory = operations.get("inference", {})
    if (
        not isinstance(inference, list)
        or len(inference) != 1
        or not isinstance(inference[0], str)
        or not isinstance(gate, dict)
    ):
        raise ValueError("step-20 H200 operations inventory is invalid")
    host = inference[0]
    model_ports = inference_inventory.get(host, {}).get("glm52", {}).get("ports")
    expected_gate_fields = {
        "profile",
        "result",
        "adapter",
        "qualification_result",
        "trainer_adapter",
    }
    if (
        set(gate) != expected_gate_fields
        or not isinstance(model_ports, list)
        or model_ports != [8000]
        or ports.get("h200_parity_backend") != 8100
        or gate.get("profile") != "configs/qualification/glm52-step20-h200.toml"
        or any(
            not isinstance(gate.get(name), str)
            or not gate[name].startswith("/data/glm52-rl/qualifications/")
            for name in (
                "result",
                "adapter",
                "qualification_result",
                "trainer_adapter",
            )
        )
        or gate.get("qualification_result")
        != gate.get("adapter", "") + "/.qualification-result.json"
    ):
        raise ValueError("step-20 H200 operations paths or ports differ")
    return {
        **gate,
        "host": host,
        "base_url": f"http://127.0.0.1:{model_ports[0]}",
        "control_url": f"http://127.0.0.1:{ports['h200_parity_backend']}",
    }


def load_config(path: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    failures = [name for name, value in EXPECTED.items() if config.get(name) != value]
    if config.get("output_path") != inventory["result"]:
        failures.append("output_path")
    if config.get("model") != EXPECTED_MODEL:
        failures.append("model")
    if config.get("engine_config") != EXPECTED_ENGINE_CONFIG:
        failures.append("engine_config")
    if config.get("adapter_path") != inventory["adapter"]:
        failures.append("adapter_path")
    if set(config) != {
        *EXPECTED,
        "output_path",
        "model",
        "engine_config",
        "adapter_path",
    }:
        failures.append("fields")
    if failures:
        raise ValueError(f"step-20 H200 profile differs: {sorted(set(failures))}")
    if config["max_output_tokens"] > config["context_length"]:
        raise ValueError("step-20 H200 output exceeds the context")
    if config["requests"] % config["group_size"]:
        raise ValueError("step-20 H200 requests do not contain complete groups")
    return config


def read_adapter(result_path: Path, expected_path: str) -> dict[str, Any]:
    result_bytes = result_path.read_bytes()
    if len(result_bytes) > 1_000_000:
        raise ValueError("step-20 trainer result exceeds its byte bound")
    result = json.loads(result_bytes)
    if not isinstance(result, dict):
        raise TypeError("step-20 trainer result is not an object")
    adapter = result.get("adapter")
    if (
        result.get("kind") != "glm52_trainer_qualification"
        or result.get("profile") != "glm52-native-stability20"
        or not isinstance(adapter, dict)
        or adapter.get("step") != 20
    ):
        raise ValueError("step-20 trainer result or adapter identity is invalid")
    digest = adapter.get("sha256")
    path = adapter.get("path")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in HEX64 for character in digest)
        or path != expected_path
        or adapter.get("checksum_value") != digest
        or not isinstance(adapter.get("size"), int)
        or not 0 < adapter["size"] <= 1_000_000_000
        or any(
            not isinstance(adapter.get(name), str)
            or len(adapter[name]) != 64
            or any(character not in HEX64 for character in adapter[name])
            for name in ("config_sha256", "stable_sha256", "checksum_sha256")
        )
        or not isinstance(result.get("session_sha256"), str)
        or len(result["session_sha256"]) != 64
        or any(character not in HEX64 for character in result["session_sha256"])
    ):
        raise ValueError("step-20 adapter path or checksum is invalid")
    return {
        "qualification_result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "qualification_session_sha256": result.get("session_sha256"),
        "trainer_path": path,
        "sha256": digest,
        "size": adapter.get("size"),
        "config_sha256": adapter.get("config_sha256"),
        "stable_sha256": adapter.get("stable_sha256"),
        "checksum_sha256": adapter.get("checksum_sha256"),
    }


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read(8192).decode(errors="replace").strip()
        raise H200RequestError(
            f"HTTP {error.code}: {detail[:1000]}"
        ) from error
    if not isinstance(value, dict):
        raise TypeError(f"H200 response is not an object: {url}")
    return value


def verify_staged_adapter(path: Path, identity: dict[str, Any]) -> None:
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError("H200 step-20 adapter directory is invalid")
    expected = {
        "adapter_model.safetensors": identity["sha256"],
        "adapter_config.json": identity["config_sha256"],
        "STABLE": identity["stable_sha256"],
        "adapter_model.safetensors.sha256": identity["checksum_sha256"],
    }
    for name, digest in expected.items():
        source = path / name
        metadata = source.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or sha256_file(source) != digest
        ):
            raise ValueError(f"H200 step-20 adapter file differs: {name}")
    adapter_file = path / "adapter_model.safetensors"
    if adapter_file.stat().st_size != identity["size"]:
        raise ValueError("H200 step-20 adapter size differs")
    checksum = (path / "adapter_model.safetensors.sha256").read_bytes()
    if checksum != f"{identity['sha256']}\n".encode():
        raise ValueError("H200 step-20 adapter checksum content differs")


def load_adapter(base_url: str, control_url: str, model: str, path: Path) -> None:
    models = request_json(base_url.rstrip("/") + "/v1/models", None, 60)
    if model in {
        item.get("id") for item in models.get("data", []) if isinstance(item, dict)
    }:
        raise RuntimeError("step-20 adapter alias already exists before proof")
    result = request_json(
        control_url.rstrip("/") + "/load_lora_adapter",
        {"lora_name": model, "lora_path": str(path)},
        600,
    )
    if result.get("status") != "ok":
        raise RuntimeError("step-20 adapter load did not return success")
    models = request_json(base_url.rstrip("/") + "/v1/models", None, 60)
    visible_models = {
        item.get("id") for item in models.get("data", []) if isinstance(item, dict)
    }
    if model not in visible_models:
        raise RuntimeError("step-20 adapter is not visible after load")


def read_parity_capture(path: Path, adapter_sha256: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) > 1_000_000:
        raise ValueError("H200 parity capture manifest exceeds its byte bound")
    capture = json.loads(payload)
    if not isinstance(capture, dict):
        raise TypeError("H200 parity capture is not an object")
    phases = capture.get("phases", {})
    arrays = capture.get("arrays", {})
    sources = capture.get("sources", {})
    graph_guard = capture.get("graph_eager_guard")
    required_arrays = ("full_logprobs", "router_ids", "router_weights")
    if (
        capture.get("kind") != "glm52_policy_parity_h200_capture"
        or capture.get("adapter_sha256") != adapter_sha256
        or not isinstance(phases, dict)
        or any(
            not isinstance(phases.get(name), dict)
            or phases[name].get("adapter_sha256") != adapter_sha256
            for name in ("graph", "eager_trace")
        )
        or phases["graph"].get("mode") != "normal_cuda_graph"
        or phases["eager_trace"].get("mode") != "force_eager_skip_compiled"
        or not isinstance(graph_guard, dict)
        or graph_guard.get("passed") is not True
        or sources.get("h200_trace_logprobs") != "normal_cuda_graph_request"
        or sources.get("router_ids") != "isolated_eager_trace_request"
        or sources.get("router_weights") != "isolated_eager_trace_request"
        or not isinstance(capture.get("router_runtime_marker"), dict)
        or any(
            not isinstance(arrays.get(name), dict)
            or not isinstance(arrays[name].get("sha256"), str)
            or len(arrays[name]["sha256"]) != 64
            or any(character not in HEX64 for character in arrays[name]["sha256"])
            for name in required_arrays
        )
    ):
        raise ValueError("H200 parity capture is not bound to the step-20 adapter")
    for name in required_arrays:
        entry = arrays[name]
        if entry.get("file") != f"{name}.npy":
            raise ValueError(f"H200 parity array path differs: {name}")
        array_path = path.parent / entry["file"]
        metadata = array_path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > 200_000_000
            or sha256_file(array_path) != entry["sha256"]
        ):
            raise ValueError(f"H200 parity array differs: {name}")
    return {
        "capture_manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "adapter_sha256": adapter_sha256,
        "graph_eager_guard": graph_guard,
        "router_runtime_marker_sha256": hashlib.sha256(
            (json.dumps(capture["router_runtime_marker"], sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "array_sha256": {name: arrays[name]["sha256"] for name in required_arrays},
    }


def gpu_memory_sample() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("cannot read H200 memory")
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 8 or any(value <= 0 for value in values):
        raise RuntimeError("H200 memory sample does not contain eight GPUs")
    return values


def benchmark(
    config: dict[str, Any],
    base_url: str,
    model: str,
) -> dict[str, Any]:
    stop = threading.Event()
    samples: list[list[int]] = []
    sample_errors: list[str] = []
    retry_lock = threading.Lock()
    retry_count = 0
    retry_errors: list[str] = []
    started = time.monotonic()
    deadline = started + config["qualification_deadline_seconds"]

    def sample_memory() -> None:
        while not stop.is_set():
            try:
                samples.append(gpu_memory_sample())
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                sample_errors.append(f"{type(error).__name__}: {error}"[:160])
                return
            stop.wait(1.0)

    def one(index: int) -> tuple[int, int, float]:
        nonlocal retry_count
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Give a precise technical analysis of deterministic "
                        f"distributed training group {index // config['group_size']}."
                    ),
                }
            ],
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "seed": config["seed"] + index,
            "max_tokens": config["max_output_tokens"],
            "min_tokens": config["min_output_tokens"],
        }
        retries = 0
        started = time.monotonic()
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("global H200 qualification deadline expired")
                response = request_json(
                    base_url.rstrip("/") + "/v1/chat/completions",
                    payload,
                    min(config["request_timeout_seconds"], remaining),
                )
                usage = response.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                output_tokens = int(usage.get("completion_tokens", 0))
                if (
                    prompt_tokens < 1
                    or output_tokens != config["max_output_tokens"]
                    or prompt_tokens + config["max_output_tokens"]
                    > config["context_length"]
                ):
                    raise ValueError("completion token counts are outside the profile")
                return prompt_tokens, output_tokens, time.monotonic() - started
            except (OSError, TimeoutError) as error:
                if retries >= config["max_retries"]:
                    raise
                retries += 1
                with retry_lock:
                    retry_count += 1
                    if len(retry_errors) < config["retry_count_max"] + 1:
                        retry_errors.append(f"{type(error).__name__}: {error}"[:1000])

    thread = threading.Thread(target=sample_memory, daemon=True)
    thread.start()
    results: list[tuple[int, int, float]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["max_inflight"]
    ) as pool:
        futures = [pool.submit(one, index) for index in range(config["requests"])]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}"[:160])
    elapsed = time.monotonic() - started
    stop.set()
    thread.join(timeout=5)
    if thread.is_alive() or sample_errors or not samples:
        errors.append((sample_errors or ["GPU memory sampler did not stop"])[0])
    latencies = sorted(row[2] for row in results)
    output_tokens = sum(row[1] for row in results)
    peak = [max(sample[index] for sample in samples) for index in range(8)] if samples else []

    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(math.ceil(len(latencies) * fraction) - 1, len(latencies) - 1)]

    return {
        "batch_seconds": elapsed,
        "completed_requests": len(results),
        "failed_requests": len(errors),
        "retries": retry_count,
        "retry_errors": retry_errors,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(latencies) if latencies else 0.0,
        },
        "max_prompt_tokens": max((row[0] for row in results), default=0),
        "peak_vram_mib": peak,
        "first_errors": errors[:3],
    }


def validate_result(result: dict[str, Any], config: dict[str, Any]) -> list[str]:
    failures = []
    if result.get("schema_version") != 1 or result.get("kind") != "glm52_step20_h200_gate":
        failures.append("H200 gate schema is invalid")
    if result.get("profile_sha256") != config.get("profile_sha256"):
        failures.append("H200 gate profile checksum differs")
    adapter_value = result.get("adapter")
    adapter = adapter_value if isinstance(adapter_value, dict) else {}
    adapter_hex_fields = (
        "sha256",
        "qualification_result_sha256",
        "qualification_session_sha256",
        "config_sha256",
        "stable_sha256",
        "checksum_sha256",
    )
    if (
        not isinstance(adapter_value, dict)
        or adapter.get("name") != config.get("model")
        or adapter.get("trainer_path") != config.get("trainer_adapter")
        or adapter.get("h200_path") != config.get("adapter_path")
        or isinstance(adapter.get("size"), bool)
        or not isinstance(adapter.get("size"), int)
        or not 0 < adapter["size"] <= 1_000_000_000
        or any(
            not isinstance(adapter.get(name), str)
            or len(adapter[name]) != 64
            or any(character not in HEX64 for character in adapter[name])
            for name in adapter_hex_fields
        )
    ):
        failures.append("H200 gate adapter identity is missing")
    engine = result.get("engine", {})
    if (
        not isinstance(engine, dict)
        or engine.get("config") != EXPECTED_ENGINE_CONFIG
        or engine.get("tensor_parallel_size") != 8
        or engine.get("cuda_graph") is not True
        or engine.get("cuda_graph_capture_size") != 256
        or engine.get("prefix_cache") is not True
        or engine.get("kv_cache_dtype") != "auto"
        or engine.get("max_num_seqs") != 128
        or engine.get("max_num_batched_tokens") != 32769
        or not isinstance(engine.get("image_digest"), str)
        or not engine["image_digest"].startswith("sha256:")
        or len(engine["image_digest"]) != 71
        or any(character not in HEX64 for character in engine["image_digest"][7:])
        or not isinstance(engine.get("config_sha256"), str)
        or len(engine["config_sha256"]) != 64
        or any(character not in HEX64 for character in engine["config_sha256"])
    ):
        failures.append("H200 gate engine identity is invalid")
    settings = result.get("settings")
    expected_settings = {name: config[name] for name in EXPECTED if name not in {"schema_version", "name"}}
    if settings != expected_settings:
        failures.append("H200 gate settings differ")
    parity_value = result.get("parity")
    parity = parity_value if isinstance(parity_value, dict) else {}
    array_hashes = parity.get("array_sha256")
    parity_hashes = (
        parity.get("capture_manifest_sha256"),
        parity.get("router_runtime_marker_sha256"),
        *(array_hashes.values() if isinstance(array_hashes, dict) else ()),
    )
    graph_guard = parity.get("graph_eager_guard")
    if (
        not isinstance(parity_value, dict)
        or parity.get("adapter_sha256") != adapter.get("sha256")
        or not isinstance(graph_guard, dict)
        or graph_guard.get("passed") is not True
        or not isinstance(array_hashes, dict)
        or set(array_hashes) != {"full_logprobs", "router_ids", "router_weights"}
        or len(parity_hashes) != 5
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in HEX64 for character in value)
            for value in parity_hashes
        )
    ):
        failures.append("H200 gate parity evidence identity is invalid")
    measurements = result.get("measurements", {})
    numeric = (
        "batch_seconds",
        "output_tokens_per_second",
    )
    if not isinstance(measurements, dict) or any(
        isinstance(measurements.get(name), bool)
        or not isinstance(measurements.get(name), (int, float))
        or not math.isfinite(measurements[name])
        for name in numeric
    ):
        failures.append("H200 gate measurements are missing or non-finite")
        return failures
    if not config["batch_seconds_min"] <= measurements["batch_seconds"] <= config["batch_seconds_max"]:
        failures.append("H200 gate batch time is outside 480-720 seconds")
    if measurements.get("completed_requests") != config["requests"]:
        failures.append("H200 gate request count differs")
    if measurements.get("failed_requests") != 0:
        failures.append("H200 gate has failed requests")
    retries = measurements.get("retries")
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 0 <= retries <= config["retry_count_max"]
    ):
        failures.append("H200 gate retry count is invalid")
    retry_errors = measurements.get("retry_errors")
    if (
        not isinstance(retry_errors, list)
        or len(retry_errors) != retries
        or any(not isinstance(value, str) or len(value) > 1000 for value in retry_errors)
        or any(
            marker in value.lower()
            for value in retry_errors
            for marker in ("out of memory", "oom", "cuda error")
        )
    ):
        failures.append("H200 gate retry or OOM evidence is invalid")
    if measurements["output_tokens_per_second"] <= 0:
        failures.append("H200 gate output throughput is not positive")
    output_tokens = measurements.get("output_tokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens != config["requests"] * config["max_output_tokens"]
    ):
        failures.append("H200 gate output token count is invalid")
    elif measurements["batch_seconds"] > 0 and not math.isclose(
        measurements["output_tokens_per_second"],
        output_tokens / measurements["batch_seconds"],
        rel_tol=1e-9,
    ):
        failures.append("H200 gate output throughput differs from token and time counts")
    if measurements.get("max_prompt_tokens", config["context_length"]) + config["max_output_tokens"] > config["context_length"]:
        failures.append("H200 gate prompt plus maximum output exceeds context")
    peak = measurements.get("peak_vram_mib")
    if not isinstance(peak, list) or len(peak) != 8 or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > config["peak_vram_mib_max"]
        for value in peak
    ):
        failures.append("H200 gate peak VRAM does not contain eight GPUs")
    latency = measurements.get("latency_seconds")
    if not isinstance(latency, dict) or any(
        isinstance(latency.get(name), bool)
        or not isinstance(latency.get(name), (int, float))
        or not math.isfinite(latency[name])
        or latency[name] <= 0
        for name in ("mean", "p50", "p95", "p99", "max")
    ):
        failures.append("H200 gate latency measurements are invalid")
    elif not (
        latency["p50"] <= latency["p95"] <= latency["p99"] <= latency["max"]
        and latency["mean"] <= latency["max"]
    ):
        failures.append("H200 gate latency order is invalid")
    if measurements.get("first_errors") != []:
        failures.append("H200 gate error evidence is not empty")
    return failures


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError("H200 gate output directory is invalid")
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{path.name}.{os.getpid()}"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=directory,
        )
        try:
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("H200 gate result write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification-result", type=Path, required=True)
    parser.add_argument("--parity-capture-manifest", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--control-url", required=True)
    parser.add_argument("--engine-image-digest", required=True)
    parser.add_argument("--engine-config-sha256", required=True)
    args = parser.parse_args()
    if (
        not args.engine_image_digest.startswith("sha256:")
        or len(args.engine_image_digest) != 71
        or any(character not in HEX64 for character in args.engine_image_digest[7:])
        or len(args.engine_config_sha256) != 64
        or any(character not in HEX64 for character in args.engine_config_sha256)
    ):
        raise ValueError("H200 engine identity is invalid")
    inventory = load_inventory(args.operations)
    config = load_config(args.config, inventory)
    config["trainer_adapter"] = inventory["trainer_adapter"]
    config["profile_sha256"] = sha256_file(args.config)
    if args.qualification_result != Path(inventory["qualification_result"]):
        raise ValueError("H200 step-20 qualification result path differs")
    if args.base_url != inventory["base_url"] or args.control_url != inventory["control_url"]:
        raise ValueError("H200 step-20 URLs differ from operations")
    adapter = read_adapter(args.qualification_result, inventory["trainer_adapter"])
    if args.adapter_path != Path(inventory["adapter"]):
        raise ValueError("H200 step-20 adapter path differs")
    verify_staged_adapter(args.adapter_path, adapter)
    parity = read_parity_capture(args.parity_capture_manifest, adapter["sha256"])
    load_adapter(
        args.base_url,
        args.control_url,
        config["model"],
        args.adapter_path,
    )
    measurements = benchmark(config, args.base_url, config["model"])
    result = {
        "schema_version": 1,
        "kind": "glm52_step20_h200_gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile_sha256": config["profile_sha256"],
        "engine": {
            "config": config["engine_config"],
            "image_digest": args.engine_image_digest,
            "config_sha256": args.engine_config_sha256,
            "tensor_parallel_size": config["tensor_parallel_size"],
            "cuda_graph": config["cuda_graph"],
            "cuda_graph_capture_size": config["cuda_graph_capture_size"],
            "prefix_cache": config["prefix_cache"],
            "kv_cache_dtype": config["kv_cache_dtype"],
            "max_num_seqs": config["max_num_seqs"],
            "max_num_batched_tokens": config["max_num_batched_tokens"],
        },
        "adapter": {
            **adapter,
            "name": config["model"],
            "h200_path": str(args.adapter_path),
        },
        "parity": parity,
        "settings": {
            name: config[name]
            for name in EXPECTED
            if name not in {"schema_version", "name"}
        },
        "measurements": measurements,
    }
    failures = validate_result(result, config)
    if failures:
        raise RuntimeError("step-20 H200 gate failed: " + "; ".join(failures))
    write_atomic(Path(inventory["result"]), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
