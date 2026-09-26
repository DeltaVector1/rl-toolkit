import copy
import hashlib
import json
import math
import tomllib
import unittest
from pathlib import Path

from training_ops.runtime.h200.post_update_qualification import (
    POST_UPDATE_LOAD_SLOT,
    POST_UPDATE_STEP,
    PostUpdateQualificationError,
    post_update_adapter_path,
    post_update_load_target,
    validate_forward_backward_seconds,
    validate_post_update_evidence,
    validate_post_update_reference,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/trinity-h200-post-update-qualification"


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


class H200PostUpdateQualificationTest(unittest.TestCase):
    def reference(self) -> dict:
        input_ids = [101, 102, 103, 104]
        prompt_ids = input_ids[:2]
        return {
            "policy": "policy-0",
            "request_mode": "serialized_generate_completion_logprobs",
            "source_kind": "operator_non_rollout",
            "input_ids": input_ids,
            "source_input_sha256": hashlib.sha256(
                json.dumps(prompt_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "inference_logprobs": [0.0, 0.0, -0.2, -0.3],
            "loss_mask": [False, False, True, True],
            "advantages": [0.0, 0.0, 0.75, -0.25],
            "sequence_lengths": [4],
            "routed_experts": {"shape": [3, 8, 1]},
            "policy_identity_before": {"target_sha256": "a"},
            "policy_identity_after": {"target_sha256": "a"},
            "cache_salt": "fresh-post-update",
            "generated_token_ids": [103, 104],
            "generated_token_count": 2,
        }

    def write_adapter(self, root: Path, weights: bytes = b"updated-adapter") -> tuple[Path, Path]:
        adapter_dir = post_update_adapter_path(root)
        adapter_dir.mkdir(parents=True)
        adapter = adapter_dir / "adapter_model.safetensors"
        adapter.write_bytes(weights)
        config = adapter_dir / "adapter_config.json"
        config_bytes = json.dumps(
            {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.0,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj", "shared_expert", "experts",
                ],
            },
            sort_keys=True,
        ).encode()
        config.write_bytes(config_bytes)
        adapter_sha = hashlib.sha256(weights).hexdigest()
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        (adapter_dir / "adapter_model.safetensors.sha256").write_text(
            adapter_sha + "\n"
        )
        (adapter_dir / "adapter_manifest.json").write_text(json.dumps({
            "version": 1,
            "kind": "lora-adapter",
            "mode": "full",
            "target_step": POST_UPDATE_STEP,
            "base_step": None,
            "full_size": len(weights),
            "full_sha256": adapter_sha,
            "target_size": len(weights),
            "target_sha256": adapter_sha,
            "config_size": len(config_bytes),
            "config_sha256": config_sha,
            "files": [
                {"name": "adapter_config.json", "size": len(config_bytes), "sha256": config_sha},
                {"name": "adapter_model.safetensors", "size": len(weights), "sha256": adapter_sha},
            ],
        }))
        (adapter_dir / "STABLE").write_text("")
        policy0 = root / "policy0" / "adapter_model.safetensors"
        policy0.parent.mkdir()
        policy0.write_bytes(b"policy-zero")
        identity = {
            "target_sha256": adapter_sha,
            "target_size": len(weights),
            "config_sha256": config_sha,
            "config_size": len(config_bytes),
        }
        evidence = {
            "save": {
                "path": str(adapter),
                "step": POST_UPDATE_STEP,
                **identity,
                "manifest_sha256": hashlib.sha256(
                    (adapter_dir / "adapter_manifest.json").read_bytes()
                ).hexdigest(),
            },
            "transfer": {"path": str(adapter_dir), **identity},
            "load": {
                "source_path": str(adapter_dir),
                "source_url": "http://h200b:18100/post-update/step_1",
                "lora_name": POST_UPDATE_LOAD_SLOT,
                "server_instance": "h200b-instance-post-update",
                **identity,
            },
            "post_update": {
                "step": POST_UPDATE_STEP,
                "source_url": "http://h200b:18100/post-update/step_1",
                "lora_name": POST_UPDATE_LOAD_SLOT,
                "server_instance": "h200b-instance-post-update",
                **identity,
            },
        }
        return adapter_dir, policy0, evidence

    def test_config_is_trainer_only_smoke_with_native_fp8_and_real_update(self):
        trainer = _load(CONFIG_DIR / "trainer.toml")
        inference = _load(CONFIG_DIR / "inference.toml")
        self.assertFalse((CONFIG_DIR / "orchestrator.toml").exists())
        self.assertEqual(trainer["output_dir"], "/data/glm52-rl/qualifications/trinity-h200-post-update-qualification")
        self.assertEqual(trainer["max_steps"], 1)
        self.assertEqual(trainer["data"]["fake"]["batch_size"], 1)
        self.assertEqual(
            trainer["data"]["fake"]["reference_path"],
            "/data/glm52-rl/qualifications/trinity-h200-post-update-qualification/reference.json",
        )
        self.assertTrue(trainer["data"]["fake"]["export_final_adapter"])
        self.assertEqual(trainer["optim"]["lr"], 8e-6)
        self.assertEqual(trainer["model"]["quantization"]["type"], "fp8")
        self.assertEqual(trainer["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertEqual(
            trainer["model"]["lora"]["initial_adapter_path"],
            "/data/glm52-rl/qualifications/trinity-h200-post-update-qualification/policy-0/adapter_model.safetensors",
        )
        self.assertEqual((trainer["model"]["pp"], trainer["model"]["cp"], trainer["model"]["ep"]), (8, 8, 8))
        self.assertEqual(inference["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertEqual(inference["parallel"], {"tp": 8, "dp": 1})
        self.assertEqual(inference, _load(ROOT / "configs/trinity-h200-parity-production/inference.toml"))
        target = post_update_load_target(trainer["output_dir"])
        self.assertEqual(target["step"], POST_UPDATE_STEP)
        self.assertEqual(target["slot"], POST_UPDATE_LOAD_SLOT)
        self.assertEqual(target["adapter_path"], str(post_update_adapter_path(trainer["output_dir"])))

    def test_forward_backward_gate_stays_600_seconds(self):
        self.assertEqual(validate_forward_backward_seconds(600.0), 600.0)
        with self.assertRaisesRegex(PostUpdateQualificationError, "600s"):
            validate_forward_backward_seconds(600.01)

    def test_reference_requires_aligned_finite_nonzero_advantages_and_rejects_rescore(self):
        value = self.reference()
        self.assertEqual(validate_post_update_reference(value), value)

        missing = copy.deepcopy(value)
        missing.pop("advantages")
        with self.assertRaisesRegex(PostUpdateQualificationError, "advantages"):
            validate_post_update_reference(missing)

        zero = copy.deepcopy(value)
        zero["advantages"] = [0.0] * 4
        with self.assertRaisesRegex(PostUpdateQualificationError, "all zero"):
            validate_post_update_reference(zero)

        nonfinite = copy.deepcopy(value)
        nonfinite["advantages"][2] = math.inf
        with self.assertRaisesRegex(PostUpdateQualificationError, "finite"):
            validate_post_update_reference(nonfinite)

        misaligned = copy.deepcopy(value)
        misaligned["advantages"] = [0.0, 0.0, 0.5]
        with self.assertRaisesRegex(PostUpdateQualificationError, "align"):
            validate_post_update_reference(misaligned)

        rescore = copy.deepcopy(value)
        rescore["request_mode"] = "serialized_generate_decode256_prefill_rescore"
        with self.assertRaisesRegex(PostUpdateQualificationError, "rescore"):
            validate_post_update_reference(rescore)

    def test_export_and_load_evidence_requires_fresh_changed_adapter(self):
        with self.subTest("missing evidence"):
            for field in ("save", "transfer", "load", "post_update"):
                with self.subTest(field):
                    missing_evidence = {name: {} for name in ("save", "transfer", "load", "post_update")}
                    missing_evidence.pop(field)
                    with self.assertRaisesRegex(PostUpdateQualificationError, f"{field} evidence"):
                        validate_post_update_evidence(Path("missing"), Path("missing-policy0"), missing_evidence)

        with self.subTest("valid evidence"):
            with self.subTest("fixture"):
                with __import__("tempfile").TemporaryDirectory() as directory:
                    adapter_dir, policy0, evidence = self.write_adapter(Path(directory))
                    result = validate_post_update_evidence(adapter_dir, policy0, evidence)
                    self.assertEqual(result["step"], POST_UPDATE_STEP)
                    self.assertEqual(result["slot"], POST_UPDATE_LOAD_SLOT)
                    self.assertNotEqual(result["target_sha256"], hashlib.sha256(policy0.read_bytes()).hexdigest())

        with self.subTest("same bytes rejected"):
            with __import__("tempfile").TemporaryDirectory() as directory:
                root = Path(directory)
                adapter_dir, policy0, evidence = self.write_adapter(root, weights=b"policy-zero")
                with self.assertRaisesRegex(PostUpdateQualificationError, "differs"):
                    validate_post_update_evidence(adapter_dir, policy0, evidence)

        with self.subTest("missing load receipt"):
            with __import__("tempfile").TemporaryDirectory() as directory:
                root = Path(directory)
                adapter_dir, policy0, evidence = self.write_adapter(root)
                evidence.pop("load")
                with self.assertRaisesRegex(PostUpdateQualificationError, "load evidence"):
                    validate_post_update_evidence(adapter_dir, policy0, evidence)


if __name__ == "__main__":
    unittest.main()
