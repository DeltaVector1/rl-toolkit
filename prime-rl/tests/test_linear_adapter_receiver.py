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
    def test_verified_base_uses_published_identity_without_hashing_model(self):
        filesystem = load_filesystem()
        base_url = "http://source/step_1"
        model = b"cached model"
        config = b"{}\n"
        model_sha256 = hashlib.sha256(model).hexdigest()
        config_sha256 = hashlib.sha256(config).hexdigest()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_sha256_entry = filesystem._sha256_entry
        hashed = []
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                base_dir = filesystem.adapter_cache_path(base_url)
                base_dir.mkdir(parents=True)
                (base_dir / "READY").write_bytes(b"ready\n")
                (base_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "version": 1,
                            "source_url": base_url,
                            "files": {
                                "adapter_config.json": {
                                    "size": len(config),
                                    "sha256": config_sha256,
                                },
                                "adapter_model.safetensors": {
                                    "size": len(model),
                                    "sha256": model_sha256,
                                },
                            },
                        },
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                (base_dir / "adapter_config.json").write_bytes(config)
                (base_dir / "adapter_model.safetensors").write_bytes(model)
                (base_dir / "adapter_model.safetensors.sha256").write_text(
                    f"{model_sha256}\n",
                    encoding="ascii",
                )

                def hash_entry(parent_fd, name, label):
                    hashed.append(name)
                    if name == "adapter_model.safetensors":
                        raise AssertionError("published model identity must not be rehashed")
                    return old_sha256_entry(parent_fd, name, label)

                filesystem._sha256_entry = hash_entry
                result = filesystem._verified_adapter_base(
                    base_url,
                    model_sha256,
                    expected_size=len(model),
                    expected_config_sha256=config_sha256,
                    expected_config_size=len(config),
                )
                self.assertEqual(
                    result,
                    (
                        base_dir / "adapter_model.safetensors",
                        base_dir / "adapter_config.json",
                    ),
                )
                self.assertEqual(hashed, ["adapter_config.json"])
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem._sha256_entry = old_sha256_entry

    def test_v1_manifest_rejects_empty_and_dot_chunk_names(self):
        filesystem = load_filesystem()
        manifest = {
            "version": 1,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": "chunked-shuffled-xor-zstd",
            "target_step": 2,
            "base_step": 1,
            "base_sha256": "a" * 64,
            "target_sha256": "b" * 64,
            "full_sha256": "b" * 64,
            "target_size": 8,
            "full_size": 8,
            "config_sha256": "c" * 64,
            "config_size": 0,
            "base_config_sha256": "d" * 64,
            "chunks": [
                {
                    "file": "chunk-0",
                    "offset": 0,
                    "raw_size": 8,
                    "wire_size": 8,
                    "wire_sha256": "e" * 64,
                    "raw_sha256": "f" * 64,
                    "codec": "zstd",
                }
            ],
        }
        old_reader = filesystem._read_source_text
        filesystem._read_source_text = lambda *_args, **_kwargs: json.dumps(manifest)
        try:
            for invalid_name in ("", ".", ".."):
                manifest["chunks"][0]["file"] = invalid_name
                with self.assertRaisesRegex(OSError, "identity is invalid"):
                    filesystem._read_chunked_v1_manifest("http://source/step_2")
        finally:
            filesystem._read_source_text = old_reader

    def test_manifest_requires_two_adjacent_bases_and_even_coverage(self):
        filesystem = load_filesystem()
        self.assertEqual(filesystem._PREDICTIVE_RECONSTRUCT_WORKERS, 32)
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
            self.assertEqual(parsed["base_config_size"], len(config))
            self.assertEqual(parsed["base2_config_size"], len(config))
            self.assertEqual(parsed["chunks"][0]["data_offset"], 2)
            for invalid_name in ("", ".", ".."):
                manifest["chunks"][0]["file"] = invalid_name
                with self.assertRaisesRegex(OSError, "identity is invalid"):
                    filesystem._read_chunked_linear_manifest("http://source/step_2")
            manifest["chunks"][0]["file"] = "chunk-0"
            manifest["chunks"][0]["data_offset"] = 0
            with self.assertRaisesRegex(OSError, "identity is invalid"):
                filesystem._read_chunked_linear_manifest("http://source/step_2")
            manifest["chunks"][0]["data_offset"] = 2
            manifest["chunks"][0]["raw_size"] = 7
            with self.assertRaisesRegex(OSError, "identity is invalid"):
                filesystem._read_chunked_linear_manifest("http://source/step_2")
        finally:
            filesystem._read_source_text = old_reader

    def test_manifest_accepts_odd_safetensors_offset_and_file_size(self):
        filesystem = load_filesystem()
        target = b"t" * 9
        config = b"{}\n"

        def digest(value):
            return hashlib.sha256(value).hexdigest()

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
            "target_sha256": digest(target),
            "full_size": len(target),
            "full_sha256": digest(target),
            "config_size": len(config),
            "config_sha256": digest(config),
            "base_size": len(target),
            "base_sha256": "a" * 64,
            "base_config_size": len(config),
            "base_config_sha256": "b" * 64,
            "base2_size": len(target),
            "base2_sha256": "c" * 64,
            "base2_config_size": len(config),
            "base2_config_sha256": "d" * 64,
            "data_offset": 3,
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
                    "data_offset": 3,
                    "raw_size": len(target),
                    "wire_size": 4,
                    "wire_sha256": "e" * 64,
                    "raw_sha256": "f" * 64,
                    "codec": "zstd",
                }
            ],
            "files": [
                {"name": "adapter_config.json", "size": len(config), "sha256": digest(config)},
                {"name": "adapter_model.safetensors", "size": len(target), "sha256": digest(target)},
            ],
        }
        old_reader = filesystem._read_source_text
        filesystem._read_source_text = lambda *_args, **_kwargs: json.dumps(manifest)
        try:
            parsed = filesystem._read_chunked_linear_manifest("http://source/step_2")
            self.assertEqual(parsed["target_size"], len(target))
            self.assertEqual(parsed["data_offset"], 3)
            self.assertEqual(parsed["chunks"][0]["raw_size"], len(target))
            manifest["chunks"][0]["raw_size"] = 8
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
        def digest(value):
            return hashlib.sha256(value).hexdigest()
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
        relaying_seen_before_chunk = False
        wire_proof_seen_during_decode = False
        decode_calls = 0
        chunk_download_calls = 0
        base_calls = []
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
                    nonlocal relaying_seen_before_chunk, chunk_download_calls
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if "chunk-" in destination.name:
                        chunk_download_calls += 1
                        relaying_seen_before_chunk = (
                            destination.parent.parent / "RELAYING"
                        ).is_file()
                    destination.write_bytes(
                        config if "adapter_config.json" in destination.name else wire
                    )
                    if kwargs.get("progress"):
                        kwargs["progress"](0, destination.stat().st_size)

                def fake_base(url, *_args, **_kwargs):
                    base_calls.append((url, _kwargs))
                    return base_paths[url] / "adapter_model.safetensors", base_paths[url] / "adapter_config.json"

                filesystem.download = fake_download
                filesystem._adapter_source_urls = lambda _source: ["http://lane"]
                filesystem._configured_download_streams = lambda: 1
                filesystem._verified_adapter_base = fake_base
                filesystem._source_manifest = lambda *_args, **_kwargs: source_manifest
                filesystem._delta_stage_timeout_s = lambda: 30.0

                def fake_decode(*_args, **_kwargs):
                    nonlocal decode_calls, wire_proof_seen_during_decode
                    decode_calls += 1
                    chunk_path = (
                        filesystem.adapter_cache_path(source_url).with_name(
                            f"{filesystem.adapter_cache_path(source_url).name}.partial"
                        )
                        / "transport-chunks"
                        / "chunk-0"
                    )
                    wire_proof_seen_during_decode = chunk_path.with_name(
                        f"{chunk_path.name}.wire-ready"
                    ).is_file()
                    if decode_calls == 1:
                        raise OSError("test interruption after wire publication")
                    return target, digest(correction)

                filesystem._decode_linear_transport_chunk = fake_decode
                with self.assertRaisesRegex(OSError, "test interruption"):
                    filesystem._stage_chunked_linear_adapter(
                        source_url,
                        transport,
                        source_manifest,
                        deadline=None,
                        status_metadata={
                            "policy_step": 3,
                            "stage_timeout_seconds": 300.0,
                            "acceptance_target_seconds": 300.0,
                        },
                    )
                partial = filesystem.adapter_cache_path(source_url).with_name(
                    f"{filesystem.adapter_cache_path(source_url).name}.partial"
                )
                self.assertTrue(
                    (partial / "transport-chunks" / "chunk-0.wire-ready").is_file()
                )
                result = filesystem._stage_chunked_linear_adapter(
                    source_url,
                    transport,
                    source_manifest,
                    deadline=None,
                    status_metadata={
                        "policy_step": 3,
                        "stage_timeout_seconds": 300.0,
                        "acceptance_target_seconds": 300.0,
                    },
                )
                self.assertTrue((result / "READY").is_file())
                self.assertEqual((result / "adapter_model.safetensors").read_bytes(), target)
                self.assertTrue(wire_proof_seen_during_decode)
                self.assertEqual(chunk_download_calls, 1)
                self.assertFalse(
                    (result / "transport-chunks" / "chunk-0.wire-ready").exists()
                )
                self.assertTrue(
                    (result / "transport-chunks" / "chunk-0.done").is_file()
                )
                status = filesystem.adapter_stage_status()
                self.assertEqual(status["status"], "ready")
                self.assertGreaterEqual(status["reconstruct_seconds"], 0.0)
                self.assertEqual(status["policy_step"], 3)
                self.assertEqual(status["stage_timeout_seconds"], 300.0)
                self.assertEqual(status["acceptance_target_seconds"], 300.0)
                self.assertTrue(relaying_seen_before_chunk)
                self.assertEqual(
                    [call[1]["expected_config_size"] for call in base_calls],
                    [len(config), len(config), len(config), len(config)],
                )
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

    def test_receiver_rejects_symlinked_partial_before_external_write(self):
        filesystem = load_filesystem()
        source_url = "http://source/step_2"
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                destination = filesystem.adapter_cache_path(source_url)
                partial = destination.with_name(f"{destination.name}.partial")
                Path(external, "sentinel").write_text("do not change", encoding="utf-8")
                partial.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(OSError, "partial directory is a symbolic link"):
                    filesystem._stage_chunked_linear_adapter(source_url, {}, {})
                self.assertTrue(partial.is_symlink())
                self.assertEqual(
                    Path(external, "sentinel").read_text(encoding="utf-8"),
                    "do not change",
                )
                self.assertEqual(list(Path(external).iterdir()), [Path(external, "sentinel")])

                partial.unlink()
                destination.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(OSError, "destination is a symbolic link"):
                    filesystem._stage_chunked_linear_adapter(source_url, {}, {})
                self.assertTrue(destination.is_symlink())
                self.assertEqual(list(Path(external).iterdir()), [Path(external, "sentinel")])
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache

    def test_inner_adapter_files_reject_symlinks(self):
        filesystem = load_filesystem()
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
            root = Path(temporary)
            external_file = Path(external, "outside")
            external_file.write_bytes(b"outside")

            output = root / "output"
            output.symlink_to(external_file)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                filesystem._preallocate(output, 8)

            incoming = root / "incoming"
            incoming.symlink_to(external_file)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                filesystem.download("http://source/file", incoming, expected_size=8)

            proof = root / "chunk.done"
            proof.symlink_to(external_file)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                filesystem._proof_identity_matches(proof, {}, completed=True)

            manifest = root / "manifest.json"
            manifest.symlink_to(external_file)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                filesystem._write_json(manifest, {"safe": True})

            output_parent = root / "output-parent"
            output_parent.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                filesystem._preallocate(output_parent / "output", 8)

    def test_receiver_rejects_symlinked_chunk_directory(self):
        filesystem = load_filesystem()
        source_url = "http://source/step_2"
        transport = {
            "base_url": "http://source/step_1",
            "base_sha256": "a" * 64,
            "base_size": 8,
            "base_config_size": 3,
            "base_config_sha256": "b" * 64,
            "base2_url": "http://source/step_0",
            "base2_sha256": "c" * 64,
            "base2_size": 8,
            "base2_config_size": 3,
            "base2_config_sha256": "d" * 64,
            "target_sha256": "e" * 64,
            "target_size": 8,
            "config_sha256": "f" * 64,
            "config_size": 3,
            "chunks": [],
        }
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_base = filesystem._verified_adapter_base
        try:
            with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                destination = filesystem.adapter_cache_path(source_url)
                partial = destination.with_name(f"{destination.name}.partial")
                partial.mkdir(parents=True)
                (partial / "transport-manifest.json").write_text(
                    json.dumps(transport, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                (partial / "transport-chunks").symlink_to(external, target_is_directory=True)
                filesystem._verified_adapter_base = lambda *_args, **_kwargs: (
                    Path(temporary) / "base-model",
                    Path(temporary) / "base-config",
                )
                with self.assertRaisesRegex(OSError, "chunk directory is a symbolic link"):
                    filesystem._stage_chunked_linear_adapter(
                        source_url,
                        transport,
                        {},
                    )
                self.assertTrue((partial / "transport-chunks").is_symlink())
                self.assertEqual(list(Path(external).iterdir()), [])
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem._verified_adapter_base = old_base

    def test_partial_cleanup_does_not_remove_a_swapped_directory(self):
        filesystem = load_filesystem()
        original_read = filesystem._read_text_nofollow
        original_open = filesystem._open_directory_nofollow
        try:
            for version in ("v1", "v2"):
                identity_name = "transport-manifest.json"
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    cache = root / "cache"
                    cache.mkdir()
                    partial = cache / "adapter.partial"
                    partial.mkdir()
                    (partial / identity_name).write_text("old", encoding="utf-8")
                    (partial / "private-marker").write_text("private", encoding="utf-8")
                    external = root / "external"
                    external.mkdir()
                    marker = external / "must-survive"
                    marker.write_text("keep", encoding="utf-8")
                    swapped = False

                    def swap_after_identity_read(path, *args, **kwargs):
                        nonlocal swapped
                        value = original_read(path, *args, **kwargs)
                        if path == identity_name and not swapped:
                            swapped = True
                            original_partial = root / "original-partial"
                            partial.rename(original_partial)
                            external.rename(partial)
                        return value

                    filesystem._read_text_nofollow = swap_after_identity_read
                    with self.assertRaisesRegex(OSError, "replaced during cleanup"):
                        filesystem._reset_staging_directory_identity(
                            partial,
                            identity_name,
                            "new",
                            f"{version} adapter partial directory",
                        )
                    self.assertTrue(swapped)
                    self.assertEqual(
                        (partial / "must-survive").read_text(encoding="utf-8"),
                        "keep",
                    )
        finally:
            filesystem._read_text_nofollow = original_read
            filesystem._open_directory_nofollow = original_open

    def test_replace_paths_keeps_both_ends_in_the_held_directory(self):
        filesystem = load_filesystem()
        original_open = filesystem._open_directory_nofollow
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = root / "cache"
                cache.mkdir()
                incoming = cache / "incoming"
                incoming.write_bytes(b"private adapter")
                external = root / "external"
                external.mkdir()
                (external / "must-survive").write_text("keep", encoding="utf-8")
                swapped = False

                def swap_after_parent_open(path, *args, **kwargs):
                    nonlocal swapped
                    descriptor = original_open(path, *args, **kwargs)
                    if Path(path) == cache and not swapped:
                        swapped = True
                        cache.rename(root / "original-cache")
                        external.rename(cache)
                    return descriptor

                filesystem._open_directory_nofollow = swap_after_parent_open
                filesystem._replace_paths(
                    incoming,
                    cache / "published",
                    "adapter publish",
                )
                self.assertTrue(swapped)
                self.assertEqual(
                    (root / "original-cache" / "published").read_bytes(),
                    b"private adapter",
                )
                self.assertFalse((cache / "published").exists())
                self.assertEqual(
                    (cache / "must-survive").read_text(encoding="utf-8"),
                    "keep",
                )
        finally:
            filesystem._open_directory_nofollow = original_open

    def test_status_reports_filesystem_failures(self):
        filesystem = load_filesystem()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                status_path = Path(temporary) / ".adapter-stage-status.json"
                status_path.symlink_to(Path(external), target_is_directory=True)
                status = filesystem.adapter_stage_status()
                self.assertEqual(status["status"], "failed")
                self.assertIn("symbolic link", status["error"])
                status_path.unlink()
                status_path.write_text("not json", encoding="utf-8")
                self.assertEqual(filesystem.adapter_stage_status()["status"], "failed")
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache

    def test_cache_cleanup_rejects_a_symlinked_cache_root(self):
        filesystem = load_filesystem()
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
            cache_root = Path(temporary) / "cache"
            cache_root.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                filesystem._cleanup_previous_adapter_caches(
                    cache_root,
                    cache_root / "active",
                )

    def test_cache_cleanup_keeps_a_newer_adapter_during_recovery(self):
        filesystem = load_filesystem()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                base_url = "http://source/run/broadcasts/step_0"
                active_url = "http://source/run/broadcasts/step_1"
                future_url = "http://source/run/broadcasts/step_3"
                stale_url = "http://source/old/broadcasts/step_0"

                paths = {
                    url: filesystem.adapter_cache_path(url)
                    for url in (base_url, active_url, future_url, stale_url)
                }
                for url, path in paths.items():
                    path.mkdir(parents=True)
                    (path / "READY").write_text("ready\n", encoding="ascii")
                    transport = {"source_url": url}
                    if url == active_url:
                        transport["base_url"] = base_url
                    (path / "transport.json").write_text(
                        json.dumps(transport),
                        encoding="utf-8",
                    )

                filesystem._cleanup_previous_adapter_caches(
                    Path(temporary),
                    paths[active_url],
                )

                self.assertTrue(paths[base_url].is_dir())
                self.assertTrue(paths[active_url].is_dir())
                self.assertTrue(paths[future_url].is_dir())
                self.assertFalse(paths[stale_url].exists())
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache

    def test_delta_cleanup_rejects_a_symlinked_candidate(self):
        filesystem = load_filesystem()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                destination = filesystem.adapter_cache_path("http://source/step_1")
                destination.parent.mkdir(parents=True, exist_ok=True)
                candidate = destination.with_name(f"{destination.name}.partial")
                sentinel = Path(external) / "sentinel"
                sentinel.write_text("keep", encoding="utf-8")
                candidate.symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(OSError, "symbolic link"):
                    filesystem._cleanup_delta_stage_artifacts(destination)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache

    def test_delta_cleanup_removes_a_regular_assembling_file(self):
        filesystem = load_filesystem()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                destination = filesystem.adapter_cache_path("http://source/step_1")
                destination.parent.mkdir(parents=True, exist_ok=True)
                assembling = destination.with_name(f"{destination.name}.assembling")
                assembling.write_bytes(b"private partial output")
                self.assertTrue(filesystem._cleanup_delta_stage_artifacts(destination))
                self.assertFalse(assembling.exists())
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache

    def test_delta_cleanup_accepts_a_normal_chunk_directory(self):
        filesystem = load_filesystem()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                destination = filesystem.adapter_cache_path("http://source/step_1")
                chunk_root = destination.with_name(f"{destination.name}.partial") / "transport-chunks"
                chunk_root.mkdir(parents=True)
                for index in range(300):
                    (chunk_root / f"chunk-{index:04d}").touch()
                self.assertTrue(filesystem._cleanup_delta_stage_artifacts(destination))
                self.assertFalse(chunk_root.parent.exists())
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache

    def test_full_checkpoint_ready_requires_index_and_shard_hashes(self):
        filesystem = load_filesystem()
        source_url = "http://source/checkpoint"
        old_cache = os.environ.get("PRIME_FULL_CHECKPOINT_CACHE_DIR")
        try:
            with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
                os.environ["PRIME_FULL_CHECKPOINT_CACHE_DIR"] = temporary
                destination = filesystem.full_checkpoint_cache_path(source_url)
                destination.mkdir(parents=True)
                index = {"weight_map": {"x": "model-00001.safetensors"}}
                index_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")
                shard_bytes = b"checkpoint shard"
                (destination / "READY").write_bytes(b"ready\n")
                (destination / "model.safetensors.index.json").write_bytes(index_bytes)
                shard = destination / "model-00001.safetensors"
                shard.write_bytes(shard_bytes)
                parent_fd = filesystem._open_directory_nofollow(
                    destination.parent,
                    create=False,
                    label="test checkpoint parent",
                )
                try:
                    self.assertFalse(
                        filesystem._full_checkpoint_cache_is_valid(
                            parent_fd,
                            destination.name,
                            source_url,
                        )
                    )
                    manifest = {
                        "version": 1,
                        "source_url": source_url,
                        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                        "files": {
                            shard.name: {
                                "size": len(shard_bytes),
                                "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                            }
                        },
                    }
                    (destination / "checkpoint-cache.json").write_text(
                        json.dumps(manifest),
                        encoding="utf-8",
                    )
                    self.assertTrue(
                        filesystem._full_checkpoint_cache_is_valid(
                            parent_fd,
                            destination.name,
                            source_url,
                        )
                    )
                    shard.write_bytes(b"tampered")
                    self.assertFalse(
                        filesystem._full_checkpoint_cache_is_valid(
                            parent_fd,
                            destination.name,
                            source_url,
                        )
                    )
                    shard.unlink()
                    shard.symlink_to(Path(external) / "outside")
                    with self.assertRaisesRegex(OSError, "symbolic link"):
                        filesystem._full_checkpoint_cache_is_valid(
                            parent_fd,
                            destination.name,
                            source_url,
                        )
                finally:
                    os.close(parent_fd)
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_FULL_CHECKPOINT_CACHE_DIR", None)
            else:
                os.environ["PRIME_FULL_CHECKPOINT_CACHE_DIR"] = old_cache

    def test_full_checkpoint_staging_does_not_return_an_incomplete_ready_cache(self):
        filesystem = load_filesystem()
        source_url = "http://source/checkpoint"
        old_cache = os.environ.get("PRIME_FULL_CHECKPOINT_CACHE_DIR")
        old_urlopen = filesystem.urllib.request.urlopen
        old_download = filesystem.download
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_FULL_CHECKPOINT_CACHE_DIR"] = temporary
                destination = filesystem.full_checkpoint_cache_path(source_url)
                destination.mkdir(parents=True)
                (destination / "READY").write_bytes(b"ready\n")
                calls: list[str] = []
                index_bytes = json.dumps(
                    {"weight_map": {"x": "model-00001.safetensors"}},
                    separators=(",", ":"),
                ).encode("utf-8")
                shard_bytes = b"new checkpoint shard"

                def no_delta_manifest(url: str, **_kwargs):
                    raise filesystem.HTTPError(url, 404, "not found", {}, None)

                def fake_download(_url, path, **_kwargs):
                    calls.append(path.name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(index_bytes if path.name.endswith(".json") else shard_bytes)

                filesystem.urllib.request.urlopen = no_delta_manifest
                filesystem.download = fake_download
                result = filesystem.stage_full_checkpoint_files(
                    source_url,
                    download_fn=filesystem.download,
                )
                self.assertEqual(result, destination)
                self.assertIn("model.safetensors.index.json", calls)
                self.assertIn("model-00001.safetensors", calls)
                parent_fd = filesystem._open_directory_nofollow(
                    destination.parent,
                    create=False,
                    label="test checkpoint parent",
                )
                try:
                    self.assertTrue(
                        filesystem._full_checkpoint_cache_is_valid(
                            parent_fd,
                            destination.name,
                            source_url,
                        )
                    )
                finally:
                    os.close(parent_fd)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.download = old_download
            if old_cache is None:
                os.environ.pop("PRIME_FULL_CHECKPOINT_CACHE_DIR", None)
            else:
                os.environ["PRIME_FULL_CHECKPOINT_CACHE_DIR"] = old_cache


if __name__ == "__main__":
    unittest.main()
