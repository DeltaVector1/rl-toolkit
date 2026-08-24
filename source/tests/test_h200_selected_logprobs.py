import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch


root = Path(__file__).resolve().parents[2]
script_path = root / "scripts/glm52_policy_parity.py"
spec = importlib.util.spec_from_file_location(
    "glm52_policy_parity_selected", script_path
)
parity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parity
spec.loader.exec_module(parity)
preflight_path = root / "scripts/glm52_policy_parity_fixture_preflight.py"
preflight_spec = importlib.util.spec_from_file_location(
    "glm52_policy_parity_fixture_preflight", preflight_path
)
preflight = importlib.util.module_from_spec(preflight_spec)
preflight_spec.loader.exec_module(preflight)
vllm_patch = (
    root / "patches/vllm/0003-v026-exact-target-prompt-logprobs.patch"
).read_text()


def distributed_logprobs(shards, valid_counts, starts, targets):
    valid = [
        shard[:, :count].astype(np.float32)
        for shard, count in zip(shards, valid_counts)
    ]
    if not all(np.isfinite(shard).all() for shard in valid):
        raise ValueError("nonfinite logit")
    local_max = np.stack([shard.max(axis=-1) for shard in valid], axis=-1)
    global_max = local_max.max(axis=-1)
    denominator = sum(
        np.exp(shard - global_max[:, None]).sum(axis=-1) for shard in valid
    )
    target_logits = np.zeros(targets.shape, dtype=np.float32)
    owners = np.zeros(targets.shape, dtype=np.int64)
    for shard, count, start in zip(valid, valid_counts, starts):
        owned = (targets >= start) & (targets < start + count)
        local_index = np.clip(targets - start, 0, max(count - 1, 0))
        target_logits += np.where(owned, shard[np.arange(targets.size), local_index], 0)
        owners += owned
    if not np.all(owners == 1):
        raise ValueError("invalid target ownership")
    return target_logits - global_max - np.log(denominator)


def distributed_torch_logprobs(shards, valid_counts, starts, targets):
    valid = [shard[:, :count].float() for shard, count in zip(shards, valid_counts)]
    global_max = (
        torch.stack([shard.max(dim=-1).values for shard in valid], dim=-1)
        .max(dim=-1)
        .values
    )
    denominator = sum(
        torch.exp(shard - global_max[:, None]).sum(dim=-1) for shard in valid
    )
    target_logits = torch.zeros(targets.shape, dtype=torch.float32)
    for shard, count, start in zip(valid, valid_counts, starts):
        owned = (targets >= start) & (targets < start + count)
        local_index = (targets - start).clamp(0, count - 1)
        target_logits += torch.where(
            owned, shard[torch.arange(targets.numel()), local_index], 0
        )
    return target_logits - global_max - torch.log(denominator)


def test_distributed_math_matches_full_fp32_softmax():
    generator = np.random.default_rng(19)
    full = generator.normal(size=(11, 13)).astype(np.float32)
    padded_first = np.pad(full[:, :7], ((0, 0), (0, 2)), constant_values=1000)
    padded_second = np.pad(full[:, 7:], ((0, 0), (0, 3)), constant_values=1000)
    targets = np.asarray([0, 6, 7, 12, 3, 9, 1, 11, 5, 8, 10])
    actual = distributed_logprobs(
        [padded_first, padded_second], [7, 6], [0, 7], targets
    )
    full_logprobs = full - np.logaddexp.reduce(full, axis=-1)[:, None]
    expected = full_logprobs[np.arange(targets.size), targets]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_uneven_shards_and_padded_vocab_do_not_change_result():
    full = np.asarray([[4.0, -2.0, 0.5, 3.0, 1.0]], dtype=np.float32)
    shards = [
        np.asarray([[4.0, -2.0, 900.0]], dtype=np.float32),
        np.asarray([[0.5, 3.0, 1.0, 800.0]], dtype=np.float32),
    ]
    targets = np.asarray([4])
    actual = distributed_logprobs(shards, [2, 3], [0, 2], targets)
    expected = full - np.logaddexp.reduce(full, axis=-1)[:, None]
    np.testing.assert_allclose(actual, expected[:, 4], rtol=1e-6, atol=1e-6)


def test_nonfinite_logit_is_rejected():
    shards = [
        np.asarray([[1.0, np.nan]], dtype=np.float32),
        np.asarray([[2.0, 3.0]], dtype=np.float32),
    ]
    with pytest.raises(ValueError, match="nonfinite"):
        distributed_logprobs(shards, [2, 2], [0, 2], np.asarray([0]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_shards_match_full_fp32_log_softmax(dtype):
    generator = torch.Generator().manual_seed(29)
    full = torch.randn((9, 17), generator=generator, dtype=torch.float32).to(dtype)
    first = torch.cat((full[:, :9], torch.full((9, 3), 1000, dtype=dtype)), dim=-1)
    second = torch.cat((full[:, 9:], torch.full((9, 4), 1000, dtype=dtype)), dim=-1)
    targets = torch.tensor([0, 8, 9, 16, 4, 13, 2, 15, 11])
    actual = distributed_torch_logprobs([first, second], [9, 8], [0, 9], targets)
    expected = torch.log_softmax(full, dim=-1, dtype=torch.float32)[
        torch.arange(targets.numel()), targets
    ]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_vllm_patch_uses_only_scalar_token_collectives():
    exact_method = vllm_patch.split("+    def get_selected_token_logprobs(", 1)[
        1
    ].split("+    def extra_repr", 1)[0]
    assert "tensor_model_parallel_all_gather" in exact_method
    assert "tensor_model_parallel_all_reduce" in exact_method
    assert "_gather_logits" not in exact_method
    assert "for start in range(0, flat_targets.numel(), 1024)" in exact_method
    assert "valid_logits = local_logits[..., : shard.num_org_elements]" in exact_method
    projection = exact_method.index("local_logits = self._apply_head")
    soft_cap = exact_method.index("if self.soft_cap is not None")
    scale = exact_method.index("if self.scale != 1.0")
    fp32_cast = exact_method.index("local_logits = local_logits.float()")
    assert projection < soft_cap < scale < fp32_cast
    assert "requires an FP32 head" not in exact_method


def test_graph_scoring_and_eager_trace_are_separate():
    trace_session = {
        "schema_version": 1,
        "side": "h200",
        "fixture_sha256": "1" * 64,
    }
    graph = parity.graph_sampling_params(123)
    eager = parity.trace_sampling_params(123, trace_session)
    assert graph["prompt_logprobs"] == 0
    assert graph["extra_args"] == {"prime_exact_target_prompt_logprobs": True}
    assert eager["extra_args"]["prime_exact_target_prompt_logprobs"] is True
    assert eager["extra_args"]["prime_router_trace"] is True
    assert len(eager["extra_args"]["prime_router_trace_session"]) == 64
    assert len(eager["extra_args"]["prime_router_trace_request"]) == 32
    assert graph["seed"] == eager["seed"] == 123
    assert (
        "prime_exact_target_prompt_logprobs"
        not in (root / "source/src/prime_rl/inference/vllm/server.py").read_text()
    )


def test_trace_flags_round_trip_through_tokens_request_payload():
    trace_session = {"schema_version": 1, "side": "h200", "fixture": "x"}
    sampling = parity.trace_sampling_params(41, trace_session)
    body = json.loads(
        json.dumps({"token_ids": [1, 2, 3], "sampling_params": sampling})
    )
    extra = body["sampling_params"]["extra_args"]
    assert extra["prime_router_trace"] is True
    assert len(extra["prime_router_trace_session"]) == 64
    assert len(extra["prime_router_trace_request"]) == 32
    patch_text = (
        root / "patches/vllm/0004-v026-request-exact-router-trace.patch"
    ).read_text()
    assert "sampling_params.extra_args" in patch_text
    assert "mark_h200_trace_flags(session, request)" in patch_text
    assert "class RouterTracePacket" in patch_text
    assert "scheduler_output.prime_router_trace_packet" in patch_text
    assert "for request_id in num_scheduled_tokens" in patch_text
    assert "broadcast" not in patch_text
    assert "all_reduce" not in patch_text


def test_graph_eager_metrics_use_existing_limits_and_fail_violations():
    profile = parity.Profile(
        Path("unused"),
        {
            "dppo_mask_low": 0.2,
            "dppo_mask_high": 0.2,
            "max_unmasked_mismatch_kl": 0.1,
            "max_masked_mismatch_kl": 0.5,
            "max_masked_fraction": 0.06,
        },
    )
    advantages = np.asarray([1.0, -1.0, 1.0, -1.0])
    graph = np.asarray([-2.0, -3.0, -4.0, -5.0])
    close = parity.inference_path_metrics(
        profile, advantages, graph, graph + 0.001
    )
    assert close["passed"] is True
    assert close["selected_tokens"] == 4
    assert close["max_abs_logprob_delta"] > 0
    failed = parity.inference_path_metrics(
        profile, advantages, np.full(4, -10.0), np.zeros(4)
    )
    assert failed["passed"] is False
    assert not all(failed["checks"].values())


def test_capture_sources_require_same_phase_identity_and_cannot_mix():
    identity = {
        "fixture_sha256": "1" * 64,
        "model": "glm",
        "model_revision": "2" * 40,
        "adapter_sha256": "3" * 64,
    }
    manifest = {
        "sources": {
            "trainer_reference_logprobs": "normal_cuda_graph_request",
            "h200_trace_logprobs": "normal_cuda_graph_request",
            "router_ids": "isolated_eager_trace_request",
            "router_weights": "isolated_eager_trace_request",
        },
        "graph_eager_guard": {"passed": True},
        "router_runtime_marker": {
            "layer_calls": {"3": 1},
            "id_sources": {"3": "modular_actual"},
            "route_output_scales": {"3": 2.5},
        },
        "route_output_scale": 2.5,
        "router_weight_provenance": "selected_weights_before_route_output_scale",
        "phases": {
            "graph": {**identity, "mode": "normal_cuda_graph"},
            "eager_trace": {
                **identity,
                "mode": "force_eager_skip_compiled",
            },
        },
    }
    parity.validate_h200_capture_sources(manifest)
    missing_marker = {**manifest}
    missing_marker.pop("router_runtime_marker")
    with pytest.raises(ValueError, match="runtime marker is missing"):
        parity.validate_h200_capture_sources(missing_marker)
    observation_only = {
        **manifest,
        "router_runtime_marker": {
            "layer_calls": {"3": 1},
            "id_sources": {"3": "observation_only"},
            "route_output_scales": {"3": 2.5},
        },
    }
    with pytest.raises(ValueError, match="not from the executed MoE path"):
        parity.validate_h200_capture_sources(observation_only)
    monolithic = {
        **manifest,
        "router_runtime_marker": {
            "layer_calls": {"3": 1},
            "id_sources": {"3": "monolithic_actual"},
            "route_output_scales": {"3": 2.5},
        },
    }
    with pytest.raises(ValueError, match="not from the executed MoE path"):
        parity.validate_h200_capture_sources(monolithic)
    mixed = {**manifest, "sources": {**manifest["sources"]}}
    mixed["sources"]["h200_trace_logprobs"] = "isolated_eager_trace_request"
    with pytest.raises(ValueError, match="sources are missing or mixed"):
        parity.validate_h200_capture_sources(mixed)
    wrong_model = {
        **manifest,
        "phases": {
            "graph": manifest["phases"]["graph"],
            "eager_trace": {**manifest["phases"]["eager_trace"], "model": "other"},
        },
    }
    with pytest.raises(ValueError, match="model values differ"):
        parity.validate_h200_capture_sources(wrong_model)


def test_trainer_and_h200_route_output_scales_must_match():
    profile = parity.Profile(
        Path("unused"),
        {"router_scale": 2.5, "max_router_weight_sum_error": 0.005},
    )
    provenance = {
        "router_weight_provenance": "selected_weights_before_route_output_scale"
    }
    trainer = {"route_output_scale": 2.5, **provenance}
    h200 = {"route_output_scale": 2.5, **provenance}
    assert parity.validate_route_output_scales(profile, trainer, h200) == 2.5
    with pytest.raises(ValueError, match="trainer and H200"):
        parity.validate_route_output_scales(
            profile, trainer, {"route_output_scale": 1.0, **provenance}
        )
    for invalid in (None, 0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="missing or invalid"):
            parity.validate_route_output_scales(
                profile,
                trainer,
                {"route_output_scale": invalid, **provenance},
            )
    with pytest.raises(ValueError, match="trainer router weight provenance"):
        parity.validate_route_output_scales(
            profile,
            {**trainer, "router_weight_provenance": "other"},
            h200,
        )


def test_route_output_scale_is_applied_after_selected_weights_on_both_paths():
    vllm_runner = (
        Path(__import__("os").environ["VLLM_V026_SOURCE_ROOT"])
        / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
    ).read_text()
    selected = vllm_runner.index("topk_weights, topk_ids = self.router.select_experts")
    experts = vllm_runner.index("self.routed_experts.forward_modular", selected)
    scale = vllm_runner.index("self._maybe_apply_routed_scale_to_output", experts)
    assert selected < experts < scale

    vllm_model = (
        Path(__import__("os").environ["VLLM_V026_SOURCE_ROOT"])
        / "vllm/model_executor/models/deepseek_v2.py"
    ).read_text()
    decoder = vllm_model.index("class DeepseekV2DecoderLayer")
    next_class = vllm_model.index("\nclass ", decoder + 1)
    decoder_body = vllm_model[decoder:next_class]
    assert (
        "apply_routed_scale_to_output=not rocm_aiter_ops.is_fused_moe_enabled()"
        in decoder_body
    )
    assert "class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)" in vllm_model

    trainer = (root / "source/src/prime_rl/trainer/models/layers/moe.py").read_text()
    selected = trainer.index("top_scores = top_scores / denominator")
    experts = trainer.index("routed_output = self._run_routed_experts(", selected)
    scale = trainer.index("routed_output = routed_output * self.router.route_scale", experts)
    combine = trainer.index("out = combine_expert_outputs(", scale)
    assert selected < experts < scale < combine


def test_trainer_stage_uses_validated_h200_capture(monkeypatch, tmp_path):
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text("router_scale = 2.5\n")
    profile = parity.Profile(profile_path, {"router_scale": 2.5})
    fixture = tmp_path / "fixture"
    capture = tmp_path / "h200-capture-final"
    fixture.mkdir()
    capture.mkdir()
    (capture / "manifest.json").write_text("{}\n")
    output = tmp_path / "stage"
    output.mkdir(mode=0o750)
    capture_manifest = {
        "fixture_sha256": "f" * 64,
        "profile_sha256": parity.file_sha256(profile_path),
        "route_output_scale": 2.5,
        "router_weight_provenance": (
            "selected_weights_before_route_output_scale"
        ),
    }

    def load_manifest(path, kind):
        if path == fixture:
            assert kind == "glm52_policy_parity_fixture"
            return {"created_utc": "2026-08-23T00:00:00+00:00"}
        assert path == capture
        assert kind == "glm52_policy_parity_h200_capture"
        return capture_manifest

    def make_reference(_profile, _fixture, _capture, path):
        path.write_text("{}\n")
        return {}

    def make_trace(_profile, _fixture, _capture, path):
        path.mkdir()
        (path / "session.json").write_text("{}\n")
        return {}

    monkeypatch.setattr(parity, "load_manifest", load_manifest)
    monkeypatch.setattr(parity, "check_fresh", lambda *_args: None)
    monkeypatch.setattr(parity, "validate_h200_capture_sources", lambda *_args: None)
    monkeypatch.setattr(parity, "fixture_digest", lambda _path: "f" * 64)
    monkeypatch.setattr(parity, "make_trainer_reference", make_reference)
    monkeypatch.setattr(parity, "prepare_trainer_router_trace", make_trace)

    manifest = parity.stage_trainer_artifacts(profile, fixture, capture, output)
    assert manifest["kind"] == "glm52_policy_parity_trainer_stage"
    assert manifest["route_output_scale"] == 2.5
    assert manifest["router_weight_provenance"] == (
        "selected_weights_before_route_output_scale"
    )
    assert set(manifest["files"]) == {
        "trainer-reference.json",
        "trainer-router-trace/session.json",
    }


def test_trainer_stage_requires_exact_operations_paths(tmp_path):
    operations = tmp_path / "operations.toml"
    operations.write_text(
        "[parity.glm52]\n"
        'fixture = "/data/parity/fixture"\n'
        'h200_capture = "/data/parity/h200-capture-final"\n'
        'trainer_stage = "/data/parity/trainer-stage-final"\n'
    )
    parity.validate_stage_paths(
        operations,
        Path("/data/parity/fixture"),
        Path("/data/parity/h200-capture-final"),
        Path("/data/parity/trainer-stage-final"),
    )
    with pytest.raises(ValueError, match="differ from operations"):
        parity.validate_stage_paths(
            operations,
            Path("/tmp/obsolete-fixture"),
            Path("/data/parity/h200-capture-final"),
            Path("/data/parity/trainer-stage-final"),
        )


def _refresh_fixture_inputs(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text("profile = true\n")
    profile = parity.Profile(
        profile_path,
        {
            "context_length": 8,
            "prompt_length": 4,
            "max_output_length": 4,
            "vocab_size": 512,
            "max_fixture_age_hours": 24,
        },
    )
    fixture = tmp_path / "fixture"
    parity.make_fixture(
        profile, fixture, 19, "2020-01-01T00:00:00+00:00"
    )
    manifest = parity.load_manifest(fixture, "glm52_policy_parity_fixture")
    token_ids = parity.load_array(fixture, manifest, "token_ids").astype(np.int64)
    positions = parity.load_array(fixture, manifest, "positions").astype(np.int64)
    selected = parity.load_array(fixture, manifest, "selected_indices").astype(np.int64)
    selected_advantages = parity.load_array(fixture, manifest, "advantages").astype(
        np.float32
    )
    loss_mask = np.zeros(8, dtype=np.bool_)
    loss_mask[selected] = True
    advantages = np.zeros(8, dtype=np.float32)
    advantages[selected] = selected_advantages
    session = {
        "schema_version": 1,
        "side": "trainer",
        "fixture_sha256": parity.fixture_digest(fixture),
        "token_ids_sha256": parity._array_bytes_sha256(token_ids),
        "model_revision": "a" * 40,
        "adapter_sha256": "b" * 64,
    }
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(session))
    token_export = tmp_path / "rank_7.jsonl"
    record = {
        "schema_version": 1,
        "rank": 7,
        "token_ids": token_ids.tolist(),
        "position_ids": positions.tolist(),
        "loss_mask": loss_mask.tolist(),
        "advantages": advantages.tolist(),
    }
    token_export.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    identity = hashlib.sha256(
        json.dumps(session, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    diagnostic = tmp_path / "diagnostic.npz"
    np.savez(
        diagnostic,
        values=np.zeros((1, 6144), dtype=np.float32),
        name=np.asarray("layer3_router_input-rank-0-pp-0-cp-0", dtype="<U128"),
        session_identity=np.asarray(identity, dtype="<U64"),
        fixture_sha256=np.asarray(session["fixture_sha256"], dtype="<U64"),
        token_ids_sha256=np.asarray(session["token_ids_sha256"], dtype="<U64"),
        model_revision=np.asarray(session["model_revision"], dtype="<U40"),
        adapter_sha256=np.asarray(session["adapter_sha256"], dtype="<U64"),
    )
    return profile, fixture, session_path, token_export, diagnostic


def test_refresh_fixture_rebuilds_stale_fixture_from_current_export(tmp_path):
    profile, fixture, session, export, diagnostic = _refresh_fixture_inputs(tmp_path)
    output = tmp_path / "current-fixture"
    diagnostic_output = tmp_path / "current-diagnostic.npz"
    result = parity.refresh_fixture_from_trainer_export(
        profile,
        fixture,
        session,
        export,
        diagnostic,
        output,
        diagnostic_output,
        7,
    )
    current_manifest = parity.load_manifest(output, "glm52_policy_parity_fixture")
    parity.check_fresh(profile, output, current_manifest)
    assert result["source_fixture_sha256"] == parity.fixture_digest(fixture)
    assert result["fixture_sha256"] == parity.fixture_digest(output)
    assert result["fixture_sha256"] != result["source_fixture_sha256"]
    assert result["records"] == 2
    assert current_manifest["source"]["token_export_sha256"] == parity.file_sha256(
        export
    )
    with np.load(diagnostic_output, allow_pickle=False) as values:
        assert str(values["fixture_sha256"].item()) == result["fixture_sha256"]
        assert str(values["source_fixture_sha256"].item()) == result[
            "source_fixture_sha256"
        ]


def test_refresh_fixture_rejects_session_or_semantic_hash_mismatch(tmp_path):
    profile, fixture, session, export, diagnostic = _refresh_fixture_inputs(tmp_path)
    bad_session = json.loads(session.read_text())
    bad_session["token_ids_sha256"] = "0" * 64
    session.write_text(json.dumps(bad_session))
    with pytest.raises(ValueError, match="token export hash"):
        parity.refresh_fixture_from_trainer_export(
            profile,
            fixture,
            session,
            export,
            diagnostic,
            tmp_path / "bad-session",
            tmp_path / "bad-session.npz",
            7,
        )
    assert not (tmp_path / "bad-session").exists()
    profile, fixture, session, export, diagnostic = _refresh_fixture_inputs(
        tmp_path / "semantic"
    )
    records = [json.loads(line) for line in export.read_text().splitlines()]
    for record in records:
        record["position_ids"][3] += 1
    export.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    with pytest.raises(ValueError, match="positions differs"):
        parity.refresh_fixture_from_trainer_export(
            profile,
            fixture,
            session,
            export,
            diagnostic,
            tmp_path / "bad-position",
            tmp_path / "bad-position.npz",
            7,
        )


def test_trainer_trace_collection_accepts_layer_owners_and_rejects_wrong_owner(tmp_path):
    profile = parity.Profile(
        tmp_path / "profile.toml",
        {"dense_layers": 1, "hidden_layers": 3},
    )
    trainer_config = tmp_path / "trainer.toml"
    trainer_config.write_text(
        "[model]\npp = 2\ncp = 2\n[data.fake]\nbatch_size = 2\n"
    )
    sources = []
    for pp in range(2):
        source = tmp_path / f"host{pp}"
        source.mkdir()
        (source / "session.json").write_text('{"side":"trainer"}\n')
        provenance = {
            "kind": "glm52_policy_parity_trainer_runtime",
            "host": f"host{pp}",
            "pp_rank": pp,
            "cp_rank": 0,
            "global_rank": pp * 2,
            "pp_size": 2,
            "cp_size": 2,
            "batch_size": 2,
            "image_digest": "sha256:" + ("a" if pp == 0 else "b") * 64,
            "owned_layers": [pp + 1],
            "trace_limits": {"max_calls": 4, "max_files": 4, "max_bytes": 1000000},
            "modules": {
                "mounted": {"location": "mounted", "sha256": "b" * 64},
                "unmounted": {"location": "unmounted", "sha256": "c" * 64},
            },
        }
        (source / f"trainer-runtime-pp-{pp}-host-host{pp}.json").write_text(
            json.dumps(provenance)
        )
        for cp in range(2):
            rank = pp * 2 + cp
            np.savez(
                source / f"trainer-rank-{rank}-pp-{pp}-cp-{cp}-layer-{pp + 1}-call-0.npz",
                ids=np.array([[0, 1]], dtype=np.int32),
                weights=np.array([[0.5, 0.5]], dtype=np.float32),
            )
        sources.append(source)
    result = parity.collect_trainer_router_trace(
        profile, trainer_config, sources, tmp_path / "merged"
    )
    assert result["chunks"] == 4
    host_bound = json.loads(
        (sources[1] / "trainer-runtime-pp-1-host-host1.json").read_text()
    )
    host_bound["host"] = "other-host"
    (sources[1] / "trainer-runtime-pp-1-host-host1.json").write_text(json.dumps(host_bound))
    with pytest.raises(ValueError, match="bound source"):
        parity.collect_trainer_router_trace(profile, trainer_config, sources, tmp_path / "host-bad")
    host_bound["host"] = "host1"
    (sources[1] / "trainer-runtime-pp-1-host-host1.json").write_text(json.dumps(host_bound))
    bad = json.loads(
        (sources[1] / "trainer-runtime-pp-1-host-host1.json").read_text()
    )
    bad["owned_layers"] = [1]
    (sources[1] / "trainer-runtime-pp-1-host-host1.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="multiple PP stages"):
        parity.collect_trainer_router_trace(profile, trainer_config, sources, tmp_path / "bad")


def test_trainer_trace_collection_requires_each_host_provenance(tmp_path):
    profile = parity.Profile(tmp_path / "profile.toml", {"dense_layers": 1, "hidden_layers": 3})
    trainer_config = tmp_path / "trainer.toml"
    trainer_config.write_text("[model]\npp = 2\ncp = 2\n[data.fake]\nbatch_size = 2\n")
    source = tmp_path / "host"
    source.mkdir()
    (source / "session.json").write_text('{"side":"trainer"}\n')
    with pytest.raises(ValueError, match="one host provenance"):
        parity.collect_trainer_router_trace(profile, trainer_config, [source], tmp_path / "merged")


def test_router_id_equality_is_exact_and_fails_on_one_disagreement():
    executed = np.asarray([[[0, 2], [1, 3]]], dtype=np.int32)
    observed = executed.copy()
    parity.require_exact_router_ids(executed, observed)
    observed[0, 1, 1] = 2
    with pytest.raises(ValueError, match="router IDs differ"):
        parity.require_exact_router_ids(executed, observed)
    with pytest.raises(ValueError, match="shapes differ"):
        parity.require_exact_router_ids(executed, observed[:, :, :1])


def test_zero_filled_builtin_router_payload_is_rejected():
    profile = parity.Profile(
        Path("unused"),
        {
            "context_length": 2,
            "hidden_layers": 2,
            "dense_layers": 1,
            "experts_per_token": 2,
            "experts": 4,
        },
    )
    with pytest.raises(ValueError, match="zero-filled"):
        parity.validate_router_ids(profile, np.zeros((2, 2, 2), dtype=np.uint8))
    duplicates = np.asarray(
        [[[0, 0], [0, 0]], [[0, 0], [2, 2]]], dtype=np.uint8
    )
    with pytest.raises(ValueError, match="same expert more than once"):
        parity.validate_router_ids(profile, duplicates)


def test_configured_glm_architecture_owns_the_scorer():
    with (root / "configs/qualification/glm52-policy-parity.toml").open("rb") as stream:
        profile = tomllib.load(stream)
    assert profile["model_architecture"] == "GlmMoeDsaForCausalLM"
    model_change = vllm_patch.split(
        "diff --git a/vllm/model_executor/models/deepseek_v2.py", 1
    )[1]
    assert "+class GlmMoeDsaForCausalLM" not in model_change
    assert " class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):" in model_change
    class_body = model_change.split(
        " class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):", 1
    )[1]
    assert "+    def compute_selected_token_logprobs(" in class_body
    assert "vllm/model_executor/models/glm4_moe.py" not in vllm_patch


def test_fixture_preflight_pins_regenerated_profile_and_array_hashes():
    profile_path = root / "configs/qualification/glm52-policy-parity.toml"
    assert hashlib.sha256(profile_path.read_bytes()).hexdigest() == (
        preflight.EXPECTED_PROFILE_SHA256
    )
    assert {
        name: values[2] for name, values in preflight.EXPECTED.items()
    } == {
        "advantages": "adc6a77593c349662403cf2d736afc9e27fbab6f3f6df580f4483bf3269a0d65",
        "positions": "aa1454e30b61a4e79ad1b4e8a8f142e11935b8133b8f2561b1d093c9d825d71e",
        "selected_indices": "364f56378cf2ff70ef90ed0227a016d701794db8930c94e7551d8edd85830087",
        "token_ids": "4a50cbca403ccef528d58bf84d1026288c1bd386eb703b055542b4f397a5823a",
    }
    assert preflight.EXPECTED_MANIFEST_SHA256 == (
        "8753dfd3cd62c7e3b849fc9bbfff1bac9cbd4dcf0a1cce4772d2aebf1612b65a"
    )
