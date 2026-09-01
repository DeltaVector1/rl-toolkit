import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_filesystem():
    modules = {
        "torch": types.ModuleType("torch"),
        "torch.nn": types.ModuleType("torch.nn"),
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": types.ModuleType("vllm.model_executor.model_loader"),
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.utils": types.ModuleType("prime_rl.utils"),
        "prime_rl.utils.checkpoint_delta": types.ModuleType("prime_rl.utils.checkpoint_delta"),
    }
    modules["torch.nn"].Module = object
    modules["vllm.model_executor.model_loader"].DefaultModelLoader = object
    modules["vllm.model_executor.model_loader"].get_model_loader = lambda _config: None
    checkpoint_delta = modules["prime_rl.utils.checkpoint_delta"]
    checkpoint_delta.DELTA_MANIFEST = "checkpoint_manifest.json"
    checkpoint_delta.apply_xor_delta_file = lambda *args, **kwargs: None
    checkpoint_delta.sha256_file = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = ROOT / "src/prime_rl/inference/vllm/worker/filesystem.py"
        spec = importlib.util.spec_from_file_location("linear_receiver_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class LinearAdapterReceiverTest(unittest.TestCase):
    def test_manifest_requires_two_adjacent_bases_and_even_coverage(self):
        filesystem = load_filesystem()
        target = b"t" * 8
        digest = hashlib.sha256(target).hexdigest()
        config = b"{}\n"
        config_digest = hashlib.sha256(config).hexdigest()
        correction_digest = hashlib.sha256(b"correction").hexdigest()
        manifest = {
            "version": 2,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
            "target_step": 2,
            "base_step": 1,
            "base2_step": 0,
            "target_url": "http://source/step_2",
            "base_url": "http://source/step_1",
            "base2_url": "http://source/step_0",
            "target_size": len(target),
            "target_sha256": digest,
            "full_size": len(target),
            "full_sha256": digest,
            "config_size": len(config),
            "config_sha256": config_digest,
            "base_size": len(target),
            "base_sha256": "a" * 64,
            "base_config_size": len(config),
            "base_config_sha256": "b" * 64,
            "base2_size": len(target),
            "base2_sha256": "c" * 64,
            "base2_config_size": len(config),
            "base2_config_sha256": "d" * 64,
            "data_offset": 2,
            "chunk_alignment": 2,
            "chunk_coverage": {"offset": 0, "size": len(target)},
            "predictor": "bf16-float32-second-order-rne",
            "value_encoding": "sparse-zigzag-leb128",
            "mask_encoding": "zstd-bit-mask-little-endian",
            "chunk_count": 1,
            "chunks": [
                {
                    "file": "chunk-0",
                    "offset": 0,
                    "data_offset": 2,
                    "raw_size": len(target),
                    "wire_size": 4,
                    "wire_sha256": "e" * 64,
                    "raw_sha256": correction_digest,
                    "codec": "zstd",
                }
            ],
            "files": [
                {"name": "adapter_config.json", "size": len(config), "sha256": config_digest},
                {"name": "adapter_model.safetensors", "size": len(target), "sha256": digest},
            ],
        }
        old_reader = filesystem._read_source_text
        filesystem._read_source_text = lambda *_args, **_kwargs: json.dumps(manifest)
        try:
            parsed = filesystem._read_chunked_linear_manifest("http://source/step_2")
            self.assertEqual(parsed["base2_url"], "http://source/step_0")
            self.assertEqual(parsed["chunks"][0]["data_offset"], 2)
            manifest["chunks"][0]["raw_size"] = 7
            with self.assertRaisesRegex(OSError, "identity is invalid"):
                filesystem._read_chunked_linear_manifest("http://source/step_2")
        finally:
            filesystem._read_source_text = old_reader

    def test_receiver_reconstructs_exact_target_with_zero_correction(self):
        filesystem = load_filesystem()
        source_url = "http://source/step_2"
        target = b"target!!"
        base1 = b"base-one"
        base2 = b"base-two"
        correction = b""
        config = b"{}\n"
        wire = b"wire"
        digest = lambda value: hashlib.sha256(value).hexdigest()
        transport = {
            "version": 2,
            "mode": "delta",
            "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
            "source_manifest": {
                "version": 2,
                "kind": "lora-adapter",
                "mode": "delta",
                "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
                "files": [],
            },
            "base_url": "http://source/step_1",
            "base_sha256": digest(base1),
            "base_size": len(base1),
            "base_config_sha256": digest(config),
            "base_config_size": len(config),
            "base2_url": "http://source/step_0",
            "base2_sha256": digest(base2),
            "base2_size": len(base2),
            "base2_config_sha256": digest(config),
            "base2_config_size": len(config),
            "target_sha256": digest(target),
            "target_size": len(target),
            "config_sha256": digest(config),
            "config_size": len(config),
            "chunks": [
                {
                    "file": "chunk-0",
                    "offset": 0,
                    "data_offset": 2,
                    "raw_size": len(target),
                    "wire_size": len(wire),
                    "wire_sha256": digest(wire),
                    "raw_sha256": digest(correction),
                    "codec": "zstd",
                }
            ],
        }
        source_manifest = {
            "version": 1,
            "source_url": source_url,
            "files": {
                "adapter_config.json": {"size": len(config), "sha256": digest(config)},
                "adapter_model.safetensors": {"size": len(target), "sha256": digest(target)},
            },
        }
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_download = filesystem.download
        old_lanes = filesystem._adapter_source_urls
        old_streams = filesystem._configured_download_streams
        old_base = filesystem._verified_adapter_base
        old_source = filesystem._source_manifest
        old_decode = filesystem._decode_linear_transport_chunk
        old_timeout = filesystem._delta_stage_timeout_s
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                base_paths = {
                    "http://source/step_1": Path(temporary) / "base1",
                    "http://source/step_0": Path(temporary) / "base2",
                }
                for url, path, model in (
                    ("http://source/step_1", base_paths["http://source/step_1"], base1),
                    ("http://source/step_0", base_paths["http://source/step_0"], base2),
                ):
                    path.mkdir()
                    (path / "adapter_model.safetensors").write_bytes(model)
                    (path / "adapter_config.json").write_bytes(config)
                    (path / "READY").write_text("ready\n", encoding="utf-8")

                def fake_download(_source, destination, **kwargs):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(config if destination.name == "adapter_config.json" else wire)
                    if kwargs.get("progress"):
                        kwargs["progress"](0, destination.stat().st_size)

                def fake_base(url, *_args, **_kwargs):
                    return base_paths[url] / "adapter_model.safetensors", base_paths[url] / "adapter_config.json"

                filesystem.download = fake_download
                filesystem._adapter_source_urls = lambda _source: ["http://lane"]
                filesystem._configured_download_streams = lambda: 1
                filesystem._verified_adapter_base = fake_base
                filesystem._source_manifest = lambda *_args, **_kwargs: source_manifest
                filesystem._delta_stage_timeout_s = lambda: 30.0
                filesystem._decode_linear_transport_chunk = lambda *_args, **_kwargs: (target, digest(correction))
                result = filesystem._stage_chunked_linear_adapter(
                    source_url,
                    transport,
                    source_manifest,
                    deadline=None,
                )
                self.assertTrue((result / "READY").is_file())
                self.assertEqual((result / "adapter_model.safetensors").read_bytes(), target)
                status = filesystem.adapter_stage_status()
                self.assertEqual(status["status"], "ready")
                self.assertGreaterEqual(status["reconstruct_seconds"], 0.0)
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem.download = old_download
            filesystem._adapter_source_urls = old_lanes
            filesystem._configured_download_streams = old_streams
            filesystem._verified_adapter_base = old_base
            filesystem._source_manifest = old_source
            filesystem._decode_linear_transport_chunk = old_decode
            filesystem._delta_stage_timeout_s = old_timeout


if __name__ == "__main__":
    unittest.main()
