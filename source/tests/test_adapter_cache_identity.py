import importlib.util
import hashlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_filesystem_module():
    torch = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.Module = type("Module", (), {})
    torch.nn = torch_nn

    loader = types.ModuleType("vllm.model_executor.model_loader")
    loader.DefaultModelLoader = type("DefaultModelLoader", (), {})
    loader.get_model_loader = lambda _config: None
    weight_transfer = types.ModuleType(
        "prime_rl.inference.vllm.worker.weight_transfer"
    )
    weight_transfer.load_weights_checkpoint_layerwise = lambda *_args, **_kwargs: None
    parity_trace = types.ModuleType("prime_rl.parity_router_trace")
    parity_trace.install_vllm_parity_trace = lambda: None
    stubs = {
        "torch": torch,
        "torch.nn": torch_nn,
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": loader,
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.inference": types.ModuleType("prime_rl.inference"),
        "prime_rl.inference.vllm": types.ModuleType("prime_rl.inference.vllm"),
        "prime_rl.inference.vllm.worker": types.ModuleType(
            "prime_rl.inference.vllm.worker"
        ),
        "prime_rl.inference.vllm.worker.weight_transfer": weight_transfer,
        "prime_rl.parity_router_trace": parity_trace,
    }
    path = (
        Path(__file__).parents[1]
        / "src/prime_rl/inference/vllm/worker/filesystem.py"
    )
    spec = importlib.util.spec_from_file_location("adapter_cache_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class AdapterCacheIdentityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_filesystem_module()

    def test_reused_url_reloads_when_config_or_model_identity_changes(self):
        module = self.module
        worker = module.FileSystemWeightUpdateWorker.__new__(
            module.FileSystemWeightUpdateWorker
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            cache = root / "cache" / "adapter"
            url = "http://127.0.0.1:8100/runs/reused"
            config = b'{"r":128}\n'
            model = b"old adapter bytes"
            (source / "adapter_config.json").write_bytes(config)
            (source / "adapter_model.safetensors").write_bytes(model)
            (source / "adapter_model.safetensors.sha256").write_text(
                hashlib.sha256(model).hexdigest() + "\n"
            )

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                self.assertTrue(create_only)
                filename = source_url.rsplit("/", 1)[-1]
                shutil.copyfile(source / filename, destination)

            with (
                patch.object(module, "adapter_cache_path", return_value=cache),
                patch.object(module, "download", side_effect=fake_download),
            ):
                worker.stage_lora_adapter(url)
                marker = json.loads((cache / "READY").read_text())
                self.assertEqual(
                    marker["adapter_config_sha256"],
                    hashlib.sha256(config).hexdigest(),
                )
                self.assertEqual((cache / "adapter_model.safetensors").read_bytes(), model)

                # The URL is unchanged, but both the model and its config now
                # describe a different adapter. A stale READY must not win.
                config = b'{"r":256}\n'
                model = b"new adapter bytes"
                (source / "adapter_config.json").write_bytes(config)
                (source / "adapter_model.safetensors").write_bytes(model)
                (source / "adapter_model.safetensors.sha256").write_text(
                    hashlib.sha256(model).hexdigest() + "\n"
                )
                worker.stage_lora_adapter(url)

            self.assertEqual((cache / "adapter_config.json").read_bytes(), config)
            self.assertEqual((cache / "adapter_model.safetensors").read_bytes(), model)
            self.assertEqual(
                json.loads((cache / "READY").read_text())["adapter_model_sha256"],
                hashlib.sha256(model).hexdigest(),
            )

    def test_legacy_empty_ready_is_not_accepted(self):
        module = self.module
        worker = module.FileSystemWeightUpdateWorker.__new__(
            module.FileSystemWeightUpdateWorker
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            cache = root / "cache" / "adapter"
            cache.mkdir(parents=True)
            (cache / "READY").touch()
            url = "http://127.0.0.1:8100/runs/reused"
            model = b"adapter bytes"
            (source / "adapter_config.json").write_text("{}\n")
            (source / "adapter_model.safetensors").write_bytes(model)
            (source / "adapter_model.safetensors.sha256").write_text(
                hashlib.sha256(model).hexdigest()
            )

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                self.assertTrue(create_only)
                shutil.copyfile(source / source_url.rsplit("/", 1)[-1], destination)

            with (
                patch.object(module, "adapter_cache_path", return_value=cache),
                patch.object(module, "download", side_effect=fake_download),
            ):
                worker.stage_lora_adapter(url)

            self.assertGreater((cache / "adapter_model.safetensors").stat().st_size, 0)
            self.assertTrue(json.loads((cache / "READY").read_text()))


if __name__ == "__main__":
    unittest.main()
