import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError:  # pragma: no cover - the runtime image supplies these packages.
    np = None
    save_file = None

import training_ops.runtime.h200.post_update_delta as delta
from training_ops.runtime.h200.post_update_delta import (
    Step1DeltaError,
    validate_step1_adapter_delta,
)


@unittest.skipUnless(np is not None and save_file is not None, "safetensors and numpy are required")
class H200PostUpdateDeltaTest(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "run_id": "qualification-run",
            "reference_sha256": "d" * 64,
            "trainer_config_sha256": "c" * 64,
            "trainer_config_size": 321,
            "code_sha256": "e" * 64,
            "code_size": 654,
            "loss_sha256": "f" * 64,
            "loss_size": 16446,
            "loss_mode": "644",
        }

    def _write_adapter(self, root, policy_arrays, step_arrays):
        policy = root / "policy0.safetensors"
        save_file(policy_arrays, str(policy))
        step = root / "run_trainer_only" / "broadcasts" / "step_1"
        step.mkdir(parents=True)
        model = step / "adapter_model.safetensors"
        save_file(step_arrays, str(model))
        config = step / "adapter_config.json"
        config_bytes = b'{"r":32,"lora_alpha":64}\n'
        config.write_bytes(config_bytes)
        target_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        config_sha = hashlib.sha256(config_bytes).hexdigest()
        (step / "adapter_model.safetensors.sha256").write_text(target_sha + "\n")
        manifest = {
            "version": 1,
            "kind": "lora-adapter",
            "mode": "full",
            "target_step": 1,
            "base_step": None,
            "target_sha256": target_sha,
            "target_size": model.stat().st_size,
            "full_sha256": target_sha,
            "full_size": model.stat().st_size,
            "config_sha256": config_sha,
            "config_size": len(config_bytes),
            "files": [
                {"name": "adapter_config.json", "sha256": config_sha, "size": len(config_bytes)},
                {"name": "adapter_model.safetensors", "sha256": target_sha, "size": model.stat().st_size},
            ],
        }
        (step / "adapter_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        (step / "STABLE").write_text("")
        return policy, step, target_sha, model.stat().st_size

    def _evidence(self, policy, step, target_sha, target_size):
        return {
            "run_id": self.expected["run_id"],
            "completed_optimizer_steps": 1,
            "learning_rate": 8e-6,
            "loss": 1.25,
            "grad_norm": 0.5,
            "forward_backward_seconds": 599.0,
            "export_seconds": 1.0,
            "export_completed": True,
            "initial_adapter_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "initial_adapter_size": policy.stat().st_size,
            "final_adapter_sha256": target_sha,
            "final_adapter_size": target_size,
            "reference_sha256": self.expected["reference_sha256"],
            "trainer_config_sha256": self.expected["trainer_config_sha256"],
            "trainer_config_size": self.expected["trainer_config_size"],
            "code_sha256": self.expected["code_sha256"],
            "code_size": self.expected["code_size"],
            "loss_identity": {
                "sha256": self.expected["loss_sha256"],
                "size": self.expected["loss_size"],
                "mode": self.expected["loss_mode"],
            },
        }

    def test_valid_delta_reports_bounded_metrics_and_atomic_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5, 3.0, 4.0], dtype=np.float32)},
            )
            output = root / "delta-receipt.json"
            result = validate_step1_adapter_delta(
                policy,
                step,
                self.expected,
                self._evidence(policy, step, target_sha, target_size),
                output,
                chunk_elements=2,
            )
            self.assertEqual(result["changed_element_count"], 1)
            self.assertEqual(result["changed_tensor_count"], 1)
            self.assertEqual(result["max_abs_delta"], 0.5)
            self.assertEqual(result["l2_delta"], 0.5)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["changed_element_count"], 1)
            self.assertEqual(
                receipt["identities_before"]["step1_config"],
                receipt["step1_config"],
            )
            self.assertEqual(
                receipt["identities_before"]["step1_manifest"],
                receipt["identities_after"]["step1_manifest"],
            )
            self.assertEqual(
                receipt["identities_before"]["policy0_adapter"],
                receipt["identities_after"]["policy0_adapter"],
            )

    def test_metadata_only_rewrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
            )
            with self.assertRaisesRegex(Step1DeltaError, "nonzero"):
                validate_step1_adapter_delta(
                    policy,
                    step,
                    self.expected,
                    self._evidence(policy, step, target_sha, target_size),
                    root / "receipt.json",
                )

    def test_nonfinite_corrupt_and_shape_mutations_are_rejected(self):
        mutations = (
            ("nan", lambda arrays: {"a": np.array([1.0, np.nan], dtype=np.float32)}),
            ("shape", lambda arrays: {"a": np.array([[1.0, 2.0]], dtype=np.float32)}),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                policy, step, target_sha, target_size = self._write_adapter(
                    root,
                    {"a": np.array([1.0, 2.0], dtype=np.float32)},
                    mutate(None),
                )
                with self.assertRaises(Step1DeltaError):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            model = step / "adapter_model.safetensors"
            model.write_bytes(model.read_bytes()[:-3])
            with self.assertRaises(Step1DeltaError):
                validate_step1_adapter_delta(
                    policy,
                    step,
                    self.expected,
                    self._evidence(policy, step, target_sha, target_size),
                    root / "receipt.json",
                )

    def test_evidence_gates_and_no_clobber_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            evidence = self._evidence(policy, step, target_sha, target_size)
            evidence["learning_rate"] = 0.0
            output = root / "receipt.json"
            failure = root / "failure.json"
            with self.assertRaisesRegex(Step1DeltaError, "learning rate"):
                validate_step1_adapter_delta(
                    policy, step, self.expected, evidence, output, failure_receipt_path=failure
                )
            first = failure.read_bytes()
            with self.assertRaises(Step1DeltaError):
                validate_step1_adapter_delta(
                    policy, step, self.expected, evidence, output, failure_receipt_path=failure
                )
            self.assertEqual(failure.read_bytes(), first)
            output.write_text("existing")
            evidence["learning_rate"] = 8e-6
            with self.assertRaisesRegex(Step1DeltaError, "clobber"):
                validate_step1_adapter_delta(policy, step, self.expected, evidence, output)

    def test_trainer_config_identity_is_not_exported_adapter_config_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            evidence = self._evidence(policy, step, target_sha, target_size)
            result = validate_step1_adapter_delta(
                policy, step, self.expected, evidence, root / "receipt.json"
            )
            self.assertEqual(
                result["expected_identity"]["trainer_config_sha256"],
                self.expected["trainer_config_sha256"],
            )
            self.assertNotEqual(
                result["step1_config"]["sha256"],
                result["expected_identity"]["trainer_config_sha256"],
            )
            wrong = dict(evidence)
            wrong["trainer_config_sha256"] = result["step1_config"]["sha256"]
            wrong["trainer_config_size"] = result["step1_config"]["size"]
            with self.assertRaisesRegex(Step1DeltaError, "trainer config"):
                validate_step1_adapter_delta(
                    policy, step, self.expected, wrong, root / "wrong.json"
                )

    def test_export_and_gradient_signals_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            for field in ("grad_norm", "export_seconds"):
                evidence = self._evidence(policy, step, target_sha, target_size)
                evidence[field] = 0.0
                with self.subTest(field=field), self.assertRaisesRegex(
                    Step1DeltaError, field.replace("_", " ")
                ):
                    validate_step1_adapter_delta(
                        policy, step, self.expected, evidence, root / f"{field}.json"
                    )

    def test_source_mutation_between_hashes_and_compare_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            original_compare = delta._compare_tensors

            def mutate_after_compare(*args, **kwargs):
                result = original_compare(*args, **kwargs)
                policy.write_bytes(policy.read_bytes() + b"mutation")
                return result

            with mock.patch.object(delta, "_compare_tensors", side_effect=mutate_after_compare):
                with self.assertRaisesRegex(Step1DeltaError, "changed during validation"):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )


if __name__ == "__main__":
    unittest.main()
