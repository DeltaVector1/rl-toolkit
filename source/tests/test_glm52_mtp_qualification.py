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
    assert operations["qualification"]["glm52_mtp_h200"] == {
        "profile": "configs/qualification/glm52-mtp-h200.toml",
        "launcher": "scripts/h200/start_mtp_qualification.sh",
        "benchmark": "scripts/h200/benchmark_glm52_mtp.py",
        "output": "benchmarks/glm52-mtp-h200.json",
    }
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
