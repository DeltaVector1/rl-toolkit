import base64
import json
import importlib.util
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import training_ops
from training_ops.runtime.h200.prepare_trinity_policy0_qualification import (
    QualificationError,
    validate_reference,
)
from training_ops.runtime.h200.post_update_qualification import (
    build_post_update_reference,
    validate_post_update_reference,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/trinity-h200-step1-parity"
CONFIG_NAME = "configs/trinity-h200-step1-parity"
QUALIFICATION_ROOT = "/data/glm52-rl/qualifications/trinity-h200-step1-parity"
STEP1_ADAPTER = f"{QUALIFICATION_ROOT}/policy-1/adapter_model.safetensors"
REFERENCE = f"{QUALIFICATION_ROOT}/reference.json"
REFERENCE_BUILDER = ROOT / "training_ops/runtime/h200/build_trinity_parity_reference.py"


_SPEC = importlib.util.spec_from_file_location("step1_reference_builder", REFERENCE_BUILDER)
assert _SPEC and _SPEC.loader
reference_builder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference_builder)


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


class H200Step1ParityTest(unittest.TestCase):
    def _step1_receipt(self):
        return {
            "target_sha256": "b" * 64,
            "target_size": 456,
            "config_sha256": "c" * 64,
            "config_size": 789,
            "source_url": "http://h200b:18100/qualified-adapters/step1",
            "lora_name": "trinity-h200-post-update-qualification-slot1",
            "server_instance": "h200-step1",
        }

    def _route(self, rows: int) -> dict:
        raw = bytearray()
        for index in range(rows):
            raw.extend((0,) * 12)
            raw.extend((1 + index % 2, 2 + index % 2, 2 + index % 2, 3 + index % 2))
        return {
            "data": base64.b64encode(raw).decode(),
            "dtype": "uint8",
            "shape": [rows, 8, 2],
            "start": 0,
        }

    def _full_route(self) -> dict:
        shape = [8191, 60, 4]
        raw = bytearray(shape[0] * shape[1] * shape[2])
        row_width = shape[1] * shape[2]
        for row in range(shape[0]):
            for layer in range(6, shape[1]):
                offset = row * row_width + layer * shape[2]
                raw[offset : offset + 4] = bytes(
                    (layer % 256, (layer + 1) % 256, (layer + 2) % 256, (layer + 3) % 256)
                )
        return {
            "data": base64.b64encode(raw).decode(),
            "dtype": "uint8",
            "shape": shape,
            "start": 0,
        }

    def test_policy1_reference_binds_loaded_slot_and_policy_label(self):
        receipt = self._step1_receipt()
        response = {
            "request_id": "step1-response",
            "choices": [{
                "token_ids": list(range(900, 932)),
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": -0.25}
                        for token in range(900, 932)
                    ]
                },
                "routed_experts": self._route(35),
            }],
        }
        record = {
            "source_kind": "operator_non_rollout",
            "input_ids": list(range(100, 104)),
            "sequence_lengths": [4],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = reference_builder.build_reference(
                endpoint="http://h200b:18100/inference/v1/generate",
                input_record=record,
                output_path=Path(directory) / "reference.json",
                base_model="/data/trinity-thinking-fp8",
                policy="policy-1",
                policy_model=receipt["lora_name"],
                expected_policy_identity=receipt,
                policy_receipt=receipt,
                cache_salt="step1-reference-0",
                readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                trainer_seq_len=36,
                minimum_tokens=1,
                post_json=lambda *_: response,
                get_json=lambda *_: receipt,
            )
        self.assertEqual(result["policy"], "policy-1")
        self.assertEqual(result["model"], receipt["lora_name"])
        self.assertEqual(result["policy_identity"]["lora_name"], receipt["lora_name"])
        self.assertEqual(validate_reference(result, expected_policy="policy-1")["policy"], "policy-1")
        changed = dict(result)
        changed["policy"] = "policy-0"
        with self.assertRaises(QualificationError):
            validate_reference(changed, expected_policy="policy-1")

    def test_strict_policy1_reference_validates_exact_loaded_slot(self):
        receipt = self._step1_receipt()
        response = {
            "request_id": "step1-response",
            "choices": [{
                "token_ids": list(range(900, 932)),
                "finish_reason": "length",
                "logprobs": {
                    "content": [
                        {"token": f"token_id:{token}", "logprob": -0.25}
                        for token in range(900, 932)
                    ]
                },
                "routed_experts": self._full_route(),
            }],
        }
        record = {
            "source_kind": "operator_non_rollout",
            "input_ids": list(range(8160)),
            "sequence_lengths": [8160],
        }
        with tempfile.TemporaryDirectory() as directory:
            result = build_post_update_reference(
                endpoint="http://h200b:18100/inference/v1/generate",
                input_record=record,
                output_path=Path(directory) / "reference.json",
                base_model="/data/trinity-thinking-fp8",
                policy_model=receipt["lora_name"],
                expected_policy_identity=receipt,
                policy_receipt=receipt,
                cache_salt="step1-reference-0",
                readiness_endpoint="http://h200b:18100/inference/v1/staging-status",
                policy="policy-1",
                post_json=lambda *_: response,
                get_json=lambda *_: receipt,
            )
        self.assertEqual(
            validate_post_update_reference(
                result,
                expected_seq_len=8192,
                expected_policy="policy-1",
                expected_policy_model=receipt["lora_name"],
            )["policy"],
            "policy-1",
        )

    def test_config_is_one_step_native_fp8_lr0_parity_with_step1_adapter(self):
        trainer_path = CONFIG_DIR / "trainer.toml"
        self.assertTrue(trainer_path.is_file())
        config = _load(trainer_path)
        self.assertFalse((CONFIG_DIR / "orchestrator.toml").exists())
        self.assertFalse((CONFIG_DIR / "inference.toml").exists())
        self.assertEqual(config["output_dir"], QUALIFICATION_ROOT)
        self.assertEqual(config["max_steps"], 1)
        self.assertEqual(config["data"]["fake"], {
            "batch_size": 8,
            "generate_samples": False,
            "reference_path": REFERENCE,
        })
        self.assertEqual(config["model"]["name"], "/data/trinity-thinking-fp8")
        self.assertEqual(config["model"]["seq_len"], 8192)
        self.assertEqual(
            (config["model"]["cp"], config["model"]["pp"], config["model"]["ep"]),
            (8, 8, 8),
        )
        self.assertEqual(
            config["model"]["lora"]["initial_adapter_path"], STEP1_ADAPTER
        )
        self.assertEqual(config["model"]["quantization"], {
            "type": "fp8",
            "dequantize_checkpoint": False,
            "expert_activation_transport": "bf16",
        })
        self.assertEqual(config["optim"]["lr"], 0.0)
        self.assertEqual(config["optim"]["weight_decay"], 0.0)
        self.assertEqual(config["optim"]["max_norm"], 1.0)

    def test_qualify_dispatches_step1_reference_to_strict_preflight(self):
        selected = {
            "kind": "trinity",
            "trainer": f"{CONFIG_NAME}/trainer.toml",
        }
        events = []

        def fake_ssh(host, script, timeout=120):
            if "docker ps" in script:
                return 0, ""
            events.append((host, script))
            return 0, "ready"

        with mock.patch.object(training_ops, "_config_dir", return_value=selected), \
            mock.patch.object(training_ops, "_validate_post_update_remote_reference", side_effect=lambda config, path, **kwargs: events.append(("reference", path, kwargs))), \
            mock.patch.object(training_ops, "_trinity_trainer_loss_preflight", side_effect=lambda nodes: events.append(("loss", nodes))), \
            mock.patch.object(training_ops, "_trainer_dependency_inventory", return_value=[]), \
            mock.patch.object(training_ops, "_deploy_dependencies"), \
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh), \
            mock.patch("builtins.print"):
            training_ops.qualify(
                "step1-parity-test",
                nodes=["n82"],
                config_dir=CONFIG_NAME,
            )

        self.assertEqual(
            events[0],
            (
                "reference",
                REFERENCE,
                {"expected_policy": "policy-1", "expected_policy_model": "trinity-h200-post-update-qualification-slot1"},
            ),
        )
        self.assertEqual(events[1], ("loss", ["n82"]))

    def test_step1_staging_plan_uses_pinned_post_update_reference_identity(self):
        identity = {
            "final_reference_sha256": "a" * 64,
            "final_reference_size": 123,
            "server_instance": "h200-step1",
            "adapter_sha256": "b" * 64,
            "adapter_size": 456,
            "config_sha256": "c" * 64,
            "config_size": 789,
        }
        with mock.patch.dict(
            os.environ,
            {"PRIME_RL_POST_UPDATE_REFERENCE_IDENTITY": json.dumps(identity)},
            clear=True,
        ):
            plan = training_ops.prepare_policy0_qualification(
                CONFIG_NAME,
                source_adapter_dir=(
                    "/data/glm52-rl/qualifications/"
                    "trinity-h200-post-update-qualification/"
                    "run_trainer_only/broadcasts/step_1"
                ),
                source_reference_path="/data/glm52-rl/h200-step1/reference.json",
                expected_old_reference_sha256="d" * 64,
                expected_old_reference_size=321,
            )
        self.assertEqual(plan["post_update_reference_identity"], identity)
        self.assertIn("QUAL_FILE reference.json", plan["source_reference_script"])
        self.assertNotIn("reference is not a policy-0 reference", plan["source_reference_script"])


if __name__ == "__main__":
    unittest.main()
