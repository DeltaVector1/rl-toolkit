"""Static and report-level guards for the isolated GLM MTP qualification."""

import importlib.util
import json
import pathlib
import tomllib


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase(*, elapsed: float, peak: float = 120000.0) -> dict:
    return {
        "output_tokens": 1000,
        "elapsed_seconds": elapsed,
        "peak_vram_mib": peak,
        "worker_restarts": 0,
        "short_decode_exact": True,
        "token_sha256": "a" * 64,
        "unmasked_mismatch_kl": 0.01,
        "masked_mismatch_kl": 0.02,
        "masked_fraction": 0.01,
        "route_ids_mismatch": 0,
        "route_weights_max_abs_error": 0.001,
        "route_trace_mode": "target-only",
        "target_request_flags": {
            "prime_router_trace": True,
            "prime_router_trace_session": "b" * 64,
            "prime_router_trace_request": "c" * 32,
        },
        "draft_request_flags": {},
        "target_route_trace_requests": 1,
        "draft_route_trace_requests": 0,
    }


def _report() -> dict:
    mtp = _phase(elapsed=90.0, peak=121000.0)
    mtp.update({"proposed_tokens": 1000, "accepted_tokens": 600})
    return {
        "schema_version": 1,
        "kind": "glm52_mtp_h200_benchmark",
        "profile": "glm52-mtp-h200",
        "status": "completed",
        "baseline": _phase(elapsed=100.0),
        "mtp": mtp,
    }


def test_profile_is_registered_without_changing_production_identity():
    operations = tomllib.loads((ROOT / "configs/operations.toml").read_text())
    mtp = operations["qualification"]["glm52_mtp_h200"]
    assert mtp == {
        "host": "h200a",
        "container": "glm52-optims-mtp-h200",
        "qualification_root": "/data/glm52-rl/qualifications/glm52-mtp-h200",
        "trace_dir": "/data/glm52-rl/qualifications/glm52-mtp-h200/run/router-trace",
        "profile": "configs/qualification/glm52-mtp-h200.toml",
        "launcher": "scripts/h200/start_mtp_qualification.sh",
        "benchmark": "scripts/h200/benchmark_glm52_mtp.py",
        "output": "benchmarks/glm52-mtp-h200.json",
        "deploy_files": mtp["deploy_files"],
    }
    assert "patches/vllm/0006-v026-mtp-slot-mapping-46908.patch" in mtp["deploy_files"]
    assert "patches/vllm/0007-v026-mtp-slot-mapping-47013.patch" in mtp["deploy_files"]
    for name in ("glm52-production3", "current"):
        identity = operations["profiles"][name].get("identity", {})
        if identity:
            assert identity["mtp_method"] == "disabled"
            assert identity["mtp_tokens"] == 0


def test_profile_has_the_required_safe_mtp_contract():
    config = tomllib.loads(
        (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
    )
    preflight = _load_module(
        ROOT / "scripts/glm52_h200_mtp_preflight.py", "glm52_mtp_preflight"
    )
    assert preflight.static_errors(config) == []
    assert config["parallel"] == {"tp": 8, "dp": 1}
    assert config["vllm_extra"]["speculative_config"]["draft_tensor_parallel_size"] == 8
    assert config["vllm_extra"]["speculative_config"]["num_speculative_tokens"] == 1
    assert config["model"]["max_model_len"] == 32768
    assert config["max_lora_rank"] == 32
    assert config["enable_return_routed_experts"] is True
    assert config["gpu_memory_utilization"] == 0.95
    assert config["enable_expert_parallel"] is True
    assert config["enable_eplb"] is False
    assert config["vllm_extra"]["moe_backend"] == "triton"


def test_mtp_preflight_rejects_non_triton_fp8_moe_backends():
    config = tomllib.loads(
        (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
    )
    preflight = _load_module(
        ROOT / "scripts/glm52_h200_mtp_preflight.py", "glm52_mtp_preflight_moe_backend"
    )
    assert preflight.static_errors(config) == []

    for backend in ("auto", "flashinfer_cutlass", "flashinfer_trtllm"):
        invalid = tomllib.loads(
            (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
        )
        invalid["vllm_extra"]["moe_backend"] = backend
        errors = preflight.static_errors(invalid)
        assert any("Triton MoE backend" in error for error in errors), backend


def test_mtp_preflight_requires_h200_headroom_and_ep_without_eplb():
    config = tomllib.loads(
        (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
    )
    preflight = _load_module(
        ROOT / "scripts/glm52_h200_mtp_preflight.py", "glm52_mtp_preflight_memory_ep"
    )
    assert config["gpu_memory_utilization"] == 0.95
    assert config["enable_expert_parallel"] is True
    assert config["enable_eplb"] is False
    assert preflight.static_errors(config) == []

    for field, value, message in (
        ("gpu_memory_utilization", 0.92, "gpu_memory_utilization"),
        ("enable_expert_parallel", False, "expert parallelism"),
        ("enable_eplb", True, "EPLB"),
    ):
        invalid = tomllib.loads(
            (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
        )
        invalid[field] = value
        errors = preflight.static_errors(invalid)
        assert any(message.lower() in error.lower() for error in errors), field


def test_mtp_block_size_is_supported_and_shared_by_target_and_draft():
    config = tomllib.loads(
        (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
    )
    preflight = _load_module(
        ROOT / "scripts/glm52_h200_mtp_preflight.py", "glm52_mtp_preflight_block_size"
    )
    assert config["vllm_extra"]["block_size"] == 64
    assert preflight.static_errors(config) == []

    for block_size in (32, 96):
        invalid = tomllib.loads(
            (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
        )
        invalid["vllm_extra"]["block_size"] = block_size
        errors = preflight.static_errors(invalid)
        assert any("block_size" in error for error in errors)

    mismatched = tomllib.loads(
        (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
    )
    mismatched["vllm_extra"]["speculative_config"]["block_size"] = 32
    errors = preflight.static_errors(mismatched)
    assert any("identical" in error for error in errors)


def test_both_launchers_mount_every_patched_vllm_file():
    overlay = _load_module(
        ROOT / "scripts/h200/flashmla_overlay_preflight.py", "overlay_preflight"
    )
    for launcher_path in (
        ROOT / "scripts/h200/start_inference.sh",
        ROOT / "scripts/h200/start_mtp_qualification.sh",
    ):
        source = launcher_path.read_text()
        for name, target in overlay.RUNTIME_MOUNTS.items():
            assert (
                f"source=$overlay_root/{overlay.FILES[name]},target={target},readonly"
                in source
            )
    assert "--restart=no" in (ROOT / "scripts/h200/start_mtp_qualification.sh").read_text()
    assert "PRIME_MTP_ROUTE_TRACE_MODE" not in (
        ROOT / "scripts/h200/start_mtp_qualification.sh"
    ).read_text()


def test_mtp_qualification_uses_the_profile_backend_and_overlay_patch_series():
    launcher = (ROOT / "scripts/h200/start_mtp_qualification.sh").read_text()
    profile = (ROOT / "configs/qualification/glm52-mtp-h200.toml").read_text()
    overlay = _load_module(
        ROOT / "scripts/h200/flashmla_overlay_preflight.py", "overlay_patch_series"
    )
    assert 'inference_config="$deployment/configs/qualification/glm52-mtp-h200.toml"' in launcher
    assert '"moe_backend":"triton"' not in launcher
    assert 'moe_backend = "triton"' in profile
    required = {
        "patches/vllm/0001-v026-flashmla-sparse-scratch-limit.patch",
        "patches/vllm/0002-v026-flashmla-sparse-mixed-batch-chunks.patch",
        "patches/vllm/0004-v026-request-exact-router-trace.patch",
        "patches/vllm/0006-v026-mtp-slot-mapping-46908.patch",
        "patches/vllm/0007-v026-mtp-slot-mapping-47013.patch",
    }
    assert required <= set(overlay.PATCHES)


def test_benchmark_accepts_only_complete_off_on_evidence():
    benchmark = _load_module(
        ROOT / "scripts/h200/benchmark_glm52_mtp.py", "benchmark_glm52_mtp"
    )
    failures, derived = benchmark.validate_report(_report())
    assert failures == []
    assert derived["acceptance_rate"] == 0.6
    assert derived["throughput_speedup"] > 1.05
    assert derived["peak_vram_delta_mib"] == 1000.0

    for field, value in (
        ("worker_restarts", 1),
        ("unmasked_mismatch_kl", 0.11),
        ("route_ids_mismatch", 1),
        ("route_trace_mode", "all"),
    ):
        report = _report()
        report["mtp"][field] = value
        failures, _ = benchmark.validate_report(report)
        assert failures, field

    report = _report()
    report["mtp"]["elapsed_seconds"] = 100.0
    failures, _ = benchmark.validate_report(report)
    assert failures

    pending = json.loads((ROOT / "benchmarks/glm52-mtp-h200.json").read_text())
    failures, _ = benchmark.validate_report(pending)
    assert failures
