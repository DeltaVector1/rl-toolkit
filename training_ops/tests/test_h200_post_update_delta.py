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

try:
    import torch
    from safetensors.torch import save_file as save_torch_file
except ImportError:  # pragma: no cover - the runtime image supplies these packages.
    torch = None
    save_torch_file = None

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
        policy_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "/data/trinity-thinking-fp8",
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": [
                "down_proj", "experts", "gate_proj", "k_proj", "o_proj",
                "q_proj", "shared_expert", "up_proj", "v_proj",
            ],
            "modules_to_save": None,
        }
        (root / "adapter_config.json").write_text(json.dumps(policy_config, sort_keys=True))
        step = root / "run_trainer_only" / "broadcasts" / "step_1"
        step.mkdir(parents=True)
        model = step / "adapter_model.safetensors"
        save_file(step_arrays, str(model))
        config = step / "adapter_config.json"
        config_bytes = json.dumps(policy_config, sort_keys=True).encode()
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
            self.assertEqual(
                receipt["identities_before"]["policy0_config"],
                receipt["identities_after"]["policy0_config"],
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

    def test_duplicate_manifest_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            manifest_path = step / "adapter_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"].append(dict(manifest["files"][0]))
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(Step1DeltaError, "duplicate"):
                validate_step1_adapter_delta(
                    policy,
                    step,
                    self.expected,
                    self._evidence(policy, step, target_sha, target_size),
                    root / "receipt.json",
                )

    def test_adapter_config_semantics_and_policy0_base_identity_are_verified(self):
        mutations = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj"],
            "base_model_name_or_path": "/data/other-model",
            "bias": "lora_only",
        }
        for field, value in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                policy, step, target_sha, target_size = self._write_adapter(
                    root,
                    {"a": np.array([1.0, 2.0], dtype=np.float32)},
                    {"a": np.array([1.0, 2.5], dtype=np.float32)},
                )
                config_path = step / "adapter_config.json"
                config = json.loads(config_path.read_text())
                config[field] = value
                config_path.write_text(json.dumps(config, sort_keys=True))
                manifest_path = step / "adapter_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
                manifest["config_sha256"] = config_sha
                manifest["config_size"] = config_path.stat().st_size
                for record in manifest["files"]:
                    if record["name"] == "adapter_config.json":
                        record.update(sha256=config_sha, size=config_path.stat().st_size)
                manifest_path.write_text(json.dumps(manifest, sort_keys=True))
                with self.assertRaisesRegex(Step1DeltaError, "adapter config"):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )

    def test_step1_and_qualification_ancestors_reject_symlinks(self):
        for component in ("step1", "broadcasts", "run_trainer_only", "output"):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "qualification"
                root.mkdir()
                policy, step, target_sha, target_size = self._write_adapter(
                    root,
                    {"a": np.array([1.0, 2.0], dtype=np.float32)},
                    {"a": np.array([1.0, 2.5], dtype=np.float32)},
                )
                if component == "step1":
                    moved = root / "step_1-real"
                    step.rename(moved)
                    step.symlink_to(moved, target_is_directory=True)
                elif component == "broadcasts":
                    actual = root / "run_trainer_only" / "broadcasts"
                    moved = root / "broadcasts-real"
                    actual.rename(moved)
                    actual.symlink_to(moved, target_is_directory=True)
                elif component == "run_trainer_only":
                    actual = root / "run_trainer_only"
                    moved = root / "run_trainer_only-real"
                    actual.rename(moved)
                    actual.symlink_to(moved, target_is_directory=True)
                else:
                    moved = root.parent / "qualification-real"
                    root.rename(moved)
                    root.symlink_to(moved, target_is_directory=True)
                    policy = root / "policy0.safetensors"
                    step = root / "run_trainer_only" / "broadcasts" / "step_1"
                with self.assertRaisesRegex(Step1DeltaError, "directory|path"):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )

    def test_policy0_parent_must_be_regular_and_inside_qualification_root(self):
        for mode in ("symlink", "outside"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                parent = Path(directory)
                root = parent / "qualification"
                root.mkdir()
                policy, step, target_sha, target_size = self._write_adapter(
                    root,
                    {"a": np.array([1.0, 2.0], dtype=np.float32)},
                    {"a": np.array([1.0, 2.5], dtype=np.float32)},
                )
                old_config = root / "adapter_config.json"
                policy_dir = root / "policy-real"
                policy_dir.mkdir()
                policy.rename(policy_dir / "adapter_model.safetensors")
                old_config.rename(policy_dir / "adapter_config.json")
                if mode == "symlink":
                    link = root / "policy-link"
                    link.symlink_to(policy_dir, target_is_directory=True)
                    policy = link / "adapter_model.safetensors"
                else:
                    outside = parent / "outside-policy"
                    outside.mkdir()
                    policy = policy_dir / "adapter_model.safetensors"
                    policy_dir.rename(outside)
                    policy = outside / "adapter_model.safetensors"
                with self.assertRaisesRegex(Step1DeltaError, "policy-0 adapter"):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )

    def test_adapter_config_requires_exact_pinned_schema_and_sorted_targets(self):
        mutations = {
            "missing": lambda config: config.pop("bias"),
            "extra": lambda config: config.update(evil="same-on-policy-and-step"),
            "wrong_r_type": lambda config: config.update(r=True),
            "wrong_alpha_type": lambda config: config.update(lora_alpha="64"),
            "wrong_dropout_type": lambda config: config.update(lora_dropout=True),
            "duplicate_target": lambda config: config["target_modules"].append("q_proj"),
            "evil_base": lambda config: config.update(base_model_name_or_path="evil"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                policy, step, target_sha, target_size = self._write_adapter(
                    root,
                    {"a": np.array([1.0, 2.0], dtype=np.float32)},
                    {"a": np.array([1.0, 2.5], dtype=np.float32)},
                )
                policy_config_path = root / "adapter_config.json"
                step_config_path = step / "adapter_config.json"
                policy_config = json.loads(policy_config_path.read_text())
                step_config = json.loads(step_config_path.read_text())
                mutate(policy_config)
                mutate(step_config)
                policy_config_path.write_text(json.dumps(policy_config, sort_keys=True))
                step_config_path.write_text(json.dumps(step_config, sort_keys=True))
                manifest_path = step / "adapter_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                config_sha = hashlib.sha256(step_config_path.read_bytes()).hexdigest()
                manifest["config_sha256"] = config_sha
                manifest["config_size"] = step_config_path.stat().st_size
                for record in manifest["files"]:
                    if record["name"] == "adapter_config.json":
                        record.update(sha256=config_sha, size=step_config_path.stat().st_size)
                manifest_path.write_text(json.dumps(manifest, sort_keys=True))
                with self.assertRaisesRegex(Step1DeltaError, "adapter config"):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )

    def test_stable_marker_requires_empty_644_file(self):
        mutations = {
            "bytes": lambda stable: stable.write_bytes(b"not-empty"),
            "mode": lambda stable: stable.chmod(0o600),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                policy, step, target_sha, target_size = self._write_adapter(
                    root,
                    {"a": np.array([1.0, 2.0], dtype=np.float32)},
                    {"a": np.array([1.0, 2.5], dtype=np.float32)},
                )
                mutate(step / "STABLE")
                with self.assertRaisesRegex(Step1DeltaError, "STABLE"):
                    validate_step1_adapter_delta(
                        policy,
                        step,
                        self.expected,
                        self._evidence(policy, step, target_sha, target_size),
                        root / "receipt.json",
                    )

    def test_adapter_config_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            duplicate = (
                '{"peft_type":"LORA","task_type":"CAUSAL_LM",'
                '"base_model_name_or_path":"/data/trinity-thinking-fp8",'
                '"r":32,"r":32,"lora_alpha":64,"lora_dropout":0.0,'
                '"bias":"none","modules_to_save":null,"target_modules":['
                '"down_proj","experts","gate_proj","k_proj","o_proj",'
                '"q_proj","shared_expert","up_proj","v_proj"]}'
            )
            config_path = step / "adapter_config.json"
            config_path.write_text(duplicate)
            manifest_path = step / "adapter_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            manifest["config_sha256"] = config_sha
            manifest["config_size"] = config_path.stat().st_size
            for record in manifest["files"]:
                if record["name"] == "adapter_config.json":
                    record.update(sha256=config_sha, size=config_path.stat().st_size)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(Step1DeltaError, "adapter config"):
                validate_step1_adapter_delta(
                    policy,
                    step,
                    self.expected,
                    self._evidence(policy, step, target_sha, target_size),
                    root / "receipt.json",
                )

    def test_policy_and_step_adapter_config_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            config_path = step / "adapter_config.json"
            config = json.loads(config_path.read_text())
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True))
            manifest_path = step / "adapter_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            manifest["config_sha256"] = config_sha
            manifest["config_size"] = config_path.stat().st_size
            for record in manifest["files"]:
                if record["name"] == "adapter_config.json":
                    record.update(sha256=config_sha, size=config_path.stat().st_size)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True))
            with self.assertRaisesRegex(Step1DeltaError, "bytes differ"):
                validate_step1_adapter_delta(
                    policy,
                    step,
                    self.expected,
                    self._evidence(policy, step, target_sha, target_size),
                    root / "receipt.json",
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

    @unittest.skipUnless(torch is not None and save_torch_file is not None, "torch and safetensors are required")
    def test_bfloat16_delta_uses_cpu_torch_slices_and_bounded_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, step, target_sha, target_size = self._write_adapter(
                root,
                {"a": np.array([1.0, 2.0], dtype=np.float32)},
                {"a": np.array([1.0, 2.5], dtype=np.float32)},
            )
            save_torch_file({"a": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)}, str(policy))
            model = step / "adapter_model.safetensors"
            save_torch_file({"a": torch.tensor([1.0, 2.5, 3.0, 4.0], dtype=torch.bfloat16)}, str(model))
            target_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            target_size = model.stat().st_size
            sidecar = step / "adapter_model.safetensors.sha256"
            sidecar.write_text(target_sha + "\n")
            manifest = json.loads((step / "adapter_manifest.json").read_text())
            manifest["target_sha256"] = manifest["full_sha256"] = target_sha
            manifest["target_size"] = manifest["full_size"] = target_size
            for record in manifest["files"]:
                if record["name"] == "adapter_model.safetensors":
                    record.update(sha256=target_sha, size=target_size)
            (step / "adapter_manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            calls = {"tensor": 0, "slice": 0, "max": 0}
            delta._tensor_backend()
            original = delta.safe_open

            def instrumented(*args, **kwargs):
                handle = original(*args, **kwargs)
                # The production implementation must use the PT slice API for
                # non-scalars.  Proxy objects record the bounded calls below.
                class Handle:
                    def __enter__(self_inner):
                        opened = handle.__enter__()
                        class Reader:
                            def keys(self):
                                return opened.keys()
                            def get_tensor(self, key):
                                calls["tensor"] += 1
                                return opened.get_tensor(key)
                            def get_slice(self, key):
                                calls["slice"] += 1
                                sliced = opened.get_slice(key)
                                class Slice:
                                    def get_shape(self):
                                        return sliced.get_shape()
                                    def get_dtype(self):
                                        return sliced.get_dtype()
                                    def __getitem__(self, index):
                                        value = sliced[index]
                                        calls["max"] = max(calls["max"], int(value.numel()))
                                        return value
                                return Slice()
                        return Reader()
                    def __exit__(self_inner, *exit_args):
                        return handle.__exit__(*exit_args)
                return Handle()

            with mock.patch.object(delta, "safe_open", side_effect=instrumented):
                result = validate_step1_adapter_delta(
                    policy, step, self.expected, self._evidence(policy, step, target_sha, target_size),
                    root / "receipt.json", chunk_elements=2,
                )
            self.assertEqual(result["changed_element_count"], 1)
            self.assertEqual(calls["tensor"], 0)
            self.assertEqual(calls["slice"], 2)
            self.assertLessEqual(calls["max"], 2)


if __name__ == "__main__":
    unittest.main()
