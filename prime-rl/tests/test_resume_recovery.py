import ast
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"
ORCHESTRATOR_PATH = ROOT / "src/prime_rl/orchestrator/orchestrator.py"
SERVER_PATH = ROOT / "src/prime_rl/inference/vllm/server.py"


def load_startup_validator(urlopen, validate_manifest, file_records):
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_verify_startup_adapter_delta"
    )
    namespace = {
        "OrchestratorConfig": object,
        "json": json,
        "urlopen": urlopen,
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "inference_weight_path": lambda _config, path, _lora_name: (
            f"http://source/{path.name}"
        ),
        "_delta_stage_timeout_s": lambda: 30.0,
        "validate_adapter_manifest": validate_manifest,
        "_adapter_file_records": file_records,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORCHESTRATOR_PATH), "exec"), namespace)
    return namespace["_verify_startup_adapter_delta"]


class ResumeRecoveryTest(unittest.TestCase):
    def test_resume_uses_target_identity_when_public_predecessors_are_cleaned(self):
        target_url = "http://source/step_6"
        manifest = {
            "version": 2,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
            "target_step": 6,
            "base_step": 5,
            "base2_step": 4,
            "target_url": target_url,
            "base_url": "http://source/step_5",
            "base2_url": "http://source/step_4",
            "target_size": 8,
            "target_sha256": "a" * 64,
            "config_size": 2,
            "config_sha256": "b" * 64,
            "base_size": 8,
            "base_sha256": "c" * 64,
            "base_config_size": 2,
            "base_config_sha256": "d" * 64,
            "base2_size": 8,
            "base2_sha256": "e" * 64,
            "base2_config_size": 2,
            "base2_config_sha256": "f" * 64,
            "data_offset": 0,
            "files": [
                {"name": "adapter_config.json", "size": 2, "sha256": "b" * 64},
                {"name": "adapter_model.safetensors", "size": 8, "sha256": "a" * 64},
            ],
        }
        calls = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fetch(url, timeout):
            calls.append((url, timeout))
            return Response(json.dumps(manifest).encode("utf-8"))

        def validate(value, **kwargs):
            self.assertEqual(value, manifest)
            self.assertEqual(kwargs["target_step"], 6)
            self.assertEqual(kwargs["base_url"], "http://source/step_5")
            self.assertEqual(kwargs["base2_url"], "http://source/step_4")

        def records(value):
            return {item["name"]: item for item in value["files"]}

        validator = load_startup_validator(fetch, validate, records)
        with tempfile.TemporaryDirectory() as temporary:
            validator(SimpleNamespace(output_dir=Path(temporary)), "adapter", 6, Path(temporary) / "step_6", target_url)
        self.assertEqual(calls, [(target_url + "/adapter_manifest.json", 30.0)])

    def test_resume_has_a_longer_separate_budget(self):
        tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("RESUME_"):
                    values[target.id] = ast.literal_eval(node.value)

        self.assertEqual(values["RESUME_ATTEMPT_TIMEOUT_S"], 900.0)
        self.assertEqual(values["RESUME_TOTAL_TIMEOUT_S"], 1800.0)
        self.assertEqual(values["RESUME_CONFIRM_TIMEOUT_S"], 120.0)
        self.assertGreater(values["RESUME_ATTEMPT_TIMEOUT_S"], 300.0)

    def test_resume_timeout_is_confirmed_before_failure(self):
        source = CLIENT_PATH.read_text(encoding="utf-8")
        resume = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_resume_engines"
            ),
        )
        self.assertIn('"/resume_status"', resume)
        self.assertIn('"/liveness"', resume)
        self.assertIn('state == "running"', resume)
        self.assertIn("timed out, but", resume)
        self.assertIn("was not confirmed", resume)
        self.assertIn("route_missing and await _is_live(client)", resume)
        self.assertIn("except (TimeoutError, httpx.TimeoutException, httpx.TransportError)", resume)

    def test_server_exposes_explicit_resume_state(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('@router.get("/resume_status")', source)
        self.assertIn('_set_engine_state("resuming")', source)
        self.assertIn('_set_engine_state("running")', source)
        self.assertIn('_set_engine_state("unknown")', source)

    def test_staging_status_does_not_hide_an_active_stage(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("def _combined_lora_stage_status(", source)
        self.assertIn('if active_status == "staging":', source)
        self.assertIn('state["status"] = "staging"', source)
        self.assertIn('durable_status in {"failed", "ready"}', source)
        self.assertIn('elif durable_status == "staging":', source)
        self.assertIn('state["status"] = "unknown"', source)

    def test_active_stage_wins_over_terminal_transfer_record(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        function = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_combined_lora_stage_status"
        )
        namespace = {"Any": Any}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SERVER_PATH), "exec"), namespace)
        combine = namespace["_combined_lora_stage_status"]

        active = combine(
            {"status": "staging", "age_seconds": 3.5},
            {"status": "failed", "error": "old transfer"},
        )
        self.assertEqual(active["status"], "staging")
        self.assertEqual(active["age_seconds"], 3.5)
        self.assertEqual(active["error"], "old transfer")
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "ready"})["status"],
            "ready",
        )
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "failed"})["status"],
            "failed",
        )
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "staging"})["status"],
            "unknown",
        )
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "corrupt"})["status"],
            "unknown",
        )

    def test_startup_stages_once_per_host_before_loading(self):
        source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
        setup = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
            ),
        )
        stage_at = setup.index("await stage_lora_adapter(")
        update_at = setup.index("await self.policy_inference.update_weights(", stage_at)
        self.assertLess(stage_at, update_at)
        self.assertIn("require_delta=False", setup[stage_at:update_at])
        self.assertIn("prefetched=prefetched", setup[update_at:])

    def test_server_fallback_stages_in_the_host_process(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        load = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "load_lora_adapter"
            ),
        )
        self.assertIn("await asyncio.to_thread(", load)
        self.assertIn("stage_lora_adapter_files", load)
        self.assertNotIn('collective_rpc("stage_lora_adapter"', load)


if __name__ == "__main__":
    unittest.main()
