import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/prime_rl/inference/vllm/adapter_validation.py"
sys.path.insert(0, str(ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location("glm_adapter_validation_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load adapter validation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_safetensors(path: Path, shapes: dict[str, tuple[int, int]], drop: str | None = None):
    header = {}
    payload = bytearray()
    for name, shape in shapes.items():
        if name == drop:
            continue
        size = 2 * shape[0] * shape[1]
        header[name] = {
            "dtype": "BF16",
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + size],
        }
        payload.extend(b"\0" * size)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class GlmAdapterValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def _config(self):
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "/data/ZiphuAI/GLM-5.2-FP8",
            "base_model_repo_id": "org/glm-base",
            "base_model_revision": "a" * 40,
            "base_model_variant": "native_fp8",
            "base_model_content_sha256": "b" * 64,
            "base_model_manifest_sha256": "c" * 64,
            "base_model_h200_name_or_path": "/models/glm52-fp8",
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": sorted(self.module._TARGET_MODULES),
            "modules_to_save": None,
        }

    def _adapter(
        self,
        root: Path,
        shapes: dict[str, tuple[int, int]],
        drop: str | None = None,
        **kwargs,
    ):
        config = self._config()
        config.update(kwargs)
        (root / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        model = root / "adapter_model.safetensors"
        _write_safetensors(model, shapes, drop=drop)
        (root / "adapter_model.safetensors.sha256").write_text(
            hashlib.sha256(model.read_bytes()).hexdigest() + "\n", encoding="ascii"
        )

    def test_expected_glm_inventory_has_exact_global_coverage(self):
        shapes = self.module.expected_tensor_shapes()
        self.assertEqual(len(shapes), 116448)
        self.assertEqual(shapes["model.layers.0.self_attn.q_a_proj.lora_A.weight"], (32, 6144))
        self.assertEqual(shapes["model.layers.0.self_attn.q_b_proj.lora_B.weight"], (24576, 32))
        self.assertEqual(shapes["model.layers.3.mlp.experts.255.down_proj.lora_A.weight"], (32, 2048))
        self.assertEqual(shapes["model.layers.77.mlp.shared_experts.up_proj.lora_B.weight"], (2048, 32))

    def test_valid_adapter_passes_all_checks(self):
        shapes = {
            "one.lora_A.weight": (32, 4),
            "one.lora_B.weight": (8, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._adapter(root, shapes)
            with patch.object(self.module, "_expected_shapes", return_value=shapes):
                result = self.module.validate_glm52_adapter(
                    root,
                    expected_repo_id="org/glm-base",
                    expected_revision="a" * 40,
                    expected_variant="native_fp8",
                    expected_content_sha256="b" * 64,
                    expected_manifest_sha256="c" * 64,
                    expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                    expected_h200_model_alias="/models/glm52-fp8",
                )
            self.assertEqual(result["tensor_count"], 2)
            self.assertFalse(result["is_3d_lora_weight"])

    def test_missing_tensor_is_rejected(self):
        shapes = {
            "one.lora_A.weight": (32, 4),
            "one.lora_B.weight": (8, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._adapter(root, shapes, drop="one.lora_B.weight")
            with patch.object(self.module, "_expected_shapes", return_value=shapes):
                with self.assertRaisesRegex(ValueError, "missing=1"):
                    self.module.validate_glm52_adapter(
                        root,
                        expected_repo_id="org/glm-base",
                        expected_revision="a" * 40,
                        expected_variant="native_fp8",
                        expected_content_sha256="b" * 64,
                        expected_manifest_sha256="c" * 64,
                        expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                        expected_h200_model_alias="/models/glm52-fp8",
                    )

    def test_wrong_dtype_shape_and_3d_request_are_rejected(self):
        shapes = {
            "one.lora_A.weight": (32, 4),
            "one.lora_B.weight": (8, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._adapter(root, shapes)
            model = root / "adapter_model.safetensors"
            header, header_size = self.module._read_header(model)
            header["one.lora_A.weight"]["dtype"] = "F16"
            encoded = json.dumps(header, separators=(",", ":")).encode()
            self.assertNotEqual(len(encoded), header_size)
            _write_safetensors(model, shapes)
            header["one.lora_A.weight"]["dtype"] = "F16"
            encoded = json.dumps(header, separators=(",", ":")).encode()
            payload = b"\0" * (2 * 32 * 4 + 2 * 8 * 32)
            model.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            (root / "adapter_model.safetensors.sha256").write_text(
                hashlib.sha256(model.read_bytes()).hexdigest() + "\n", encoding="ascii"
            )
            with patch.object(self.module, "_expected_shapes", return_value=shapes):
                with self.assertRaisesRegex(ValueError, "dtype is not BF16"):
                    self.module.validate_glm52_adapter(
                        root,
                        expected_repo_id="org/glm-base",
                        expected_revision="a" * 40,
                        expected_variant="native_fp8",
                        expected_content_sha256="b" * 64,
                        expected_manifest_sha256="c" * 64,
                        expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                        expected_h200_model_alias="/models/glm52-fp8",
                    )
                with self.assertRaisesRegex(ValueError, "2D"):
                    self.module.validate_glm52_adapter(
                        root,
                        expected_repo_id="org/glm-base",
                        expected_revision="a" * 40,
                        expected_variant="native_fp8",
                        expected_content_sha256="b" * 64,
                        expected_manifest_sha256="c" * 64,
                        expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                        expected_h200_model_alias="/models/glm52-fp8",
                        is_3d_lora_weight=True,
                    )

    def test_model_identity_mismatch_is_rejected(self):
        shapes = {
            "one.lora_A.weight": (32, 4),
            "one.lora_B.weight": (8, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._adapter(root, shapes)
            with patch.object(self.module, "_expected_shapes", return_value=shapes):
                with self.assertRaisesRegex(ValueError, "base model revision differs"):
                    self.module.validate_glm52_adapter(
                        root,
                        expected_repo_id="org/glm-base",
                        expected_revision="c" * 40,
                        expected_variant="native_fp8",
                        expected_content_sha256="b" * 64,
                        expected_manifest_sha256="c" * 64,
                        expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                        expected_h200_model_alias="/models/glm52-fp8",
                    )

    def test_model_content_identity_mismatch_is_rejected(self):
        shapes = {
            "one.lora_A.weight": (32, 4),
            "one.lora_B.weight": (8, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._adapter(root, shapes)
            with patch.object(self.module, "_expected_shapes", return_value=shapes):
                with self.assertRaisesRegex(ValueError, "content SHA-256 differs"):
                    self.module.validate_glm52_adapter(
                        root,
                        expected_repo_id="org/glm-base",
                        expected_revision="a" * 40,
                        expected_variant="native_fp8",
                        expected_content_sha256="d" * 64,
                        expected_manifest_sha256="c" * 64,
                        expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                        expected_h200_model_alias="/models/glm52-fp8",
                    )

    def test_model_path_is_not_a_repository_identity(self):
        with self.assertRaisesRegex(ValueError, "repository ID is invalid"):
            self.module.validate_glm52_adapter(
                Path("/tmp/missing"),
                expected_repo_id="/data/glm52-fp8",
                expected_revision="a" * 40,
                expected_variant="native_fp8",
                expected_content_sha256="b" * 64,
                expected_manifest_sha256="c" * 64,
                expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                expected_h200_model_alias="/models/glm52-fp8",
            )

    def test_wrong_shape_and_missing_identity_are_rejected(self):
        expected = {
            "one.lora_A.weight": (32, 4),
            "one.lora_B.weight": (8, 32),
        }
        actual = {
            "one.lora_A.weight": (16, 4),
            "one.lora_B.weight": (8, 32),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._adapter(root, actual)
            with patch.object(self.module, "_expected_shapes", return_value=expected):
                with self.assertRaisesRegex(ValueError, "shape differs"):
                    self.module.validate_glm52_adapter(
                        root,
                        expected_repo_id="org/glm-base",
                        expected_revision="a" * 40,
                        expected_variant="native_fp8",
                        expected_content_sha256="b" * 64,
                        expected_manifest_sha256="c" * 64,
                        expected_trainer_model_path="/data/ZiphuAI/GLM-5.2-FP8",
                        expected_h200_model_alias="/models/glm52-fp8",
                    )
        with patch.dict(self.module.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "canonical model identity"):
                self.module.expected_model_identity()

    def test_environment_identity_contains_revision_and_both_content_hashes(self):
        values = {
            "PRIME_LORA_BASE_MODEL_REPO_ID": "zai-org/GLM-5.2-FP8",
            "PRIME_LORA_BASE_MODEL_REVISION": "a" * 40,
            "PRIME_LORA_BASE_MODEL_VARIANT": "native_fp8",
            "PRIME_LORA_BASE_MODEL_CONTENT_SHA256": "b" * 64,
            "PRIME_LORA_BASE_MODEL_MANIFEST_SHA256": "c" * 64,
            "PRIME_LORA_BASE_MODEL_TRAINER_PATH": "/data/ZiphuAI/GLM-5.2-FP8",
            "PRIME_LORA_BASE_MODEL_H200_ALIAS": "/models/glm52-fp8",
        }
        with patch.dict(self.module.os.environ, values, clear=True):
            self.assertEqual(
                self.module.expected_model_identity(),
                {
                    "repo_id": "zai-org/GLM-5.2-FP8",
                    "revision": "a" * 40,
                    "variant": "native_fp8",
                    "content_sha256": "b" * 64,
                    "manifest_sha256": "c" * 64,
                    "trainer_model_path": "/data/ZiphuAI/GLM-5.2-FP8",
                    "h200_model_alias": "/models/glm52-fp8",
                },
            )


if __name__ == "__main__":
    unittest.main()
