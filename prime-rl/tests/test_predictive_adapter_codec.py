import importlib.util
import hashlib
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/prime_rl/utils/adapter_delta.py"
SPEC = importlib.util.spec_from_file_location("predictive_adapter_delta", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter_delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter_delta)

CHECKPOINT_DELTA_PATH = ROOT / "src/prime_rl/utils/checkpoint_delta.py"
CHECKPOINT_DELTA_SPEC = importlib.util.spec_from_file_location(
    "checkpoint_delta_safety",
    CHECKPOINT_DELTA_PATH,
)
assert CHECKPOINT_DELTA_SPEC is not None and CHECKPOINT_DELTA_SPEC.loader is not None
checkpoint_delta = importlib.util.module_from_spec(CHECKPOINT_DELTA_SPEC)
CHECKPOINT_DELTA_SPEC.loader.exec_module(checkpoint_delta)


def safetensor_bytes(words: list[int]) -> bytes:
    header = b'{"x":0} '
    return struct.pack("<Q", len(header)) + header + np.asarray(words, dtype="<u2").tobytes()


def valid_safetensor_bytes(words: list[int], dtype: str = "BF16") -> bytes:
    data = np.asarray(words, dtype="<u2").tobytes()
    element_size = 2 if dtype == "BF16" else 4
    header = json.dumps(
        {
            "x": {
                "dtype": dtype,
                "shape": [len(data) // element_size],
                "data_offsets": [0, len(data)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if (8 + len(header)) % 2:
        header += b" "
    return struct.pack("<Q", len(header)) + header + data


def odd_offset_safetensor_bytes(words: list[int]) -> bytes:
    data = np.asarray(words, dtype="<u2").tobytes()
    header = json.dumps(
        {
            "x": {
                "dtype": "BF16",
                "shape": [len(words)],
                "data_offsets": [0, len(data)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if (8 + len(header)) % 2 == 0:
        header += b" "
    assert (8 + len(header)) % 2 == 1
    return struct.pack("<Q", len(header)) + header + data


class PredictiveAdapterCodecTest(unittest.TestCase):
    def test_adapter_chunk_rejects_malicious_declared_zstd_size(self):
        payload = adapter_delta._compress(b"hello")
        malicious = payload[:4] + b"\xe0" + (1 << 40).to_bytes(8, "little") + payload[6:]
        with self.assertRaisesRegex(OSError, "size mismatch"):
            adapter_delta.decode_adapter_chunk(
                malicious,
                codec=adapter_delta.full_transport_name,
                max_output_size=5,
            )

    def test_build_checkpoint_delta_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            target = root / "target"
            base.mkdir()
            target.mkdir()
            index = {"weight_map": {"x": "shard.safetensors"}}
            (base / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            (target / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            base_bytes = bytes(range(32))
            target_bytes = bytes(range(32, 64))
            (base / "shard.safetensors").write_bytes(base_bytes)
            (target / "shard.safetensors").write_bytes(target_bytes)

            manifest_path = checkpoint_delta.build_xor_delta_checkpoint(
                base,
                target,
                base_url="http://source/base",
                target_url="http://source/target",
            )
            self.assertEqual(manifest_path, target / checkpoint_delta.DELTA_MANIFEST)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            restored = root / "restored"
            restored.mkdir()
            checkpoint_delta.apply_xor_delta_file(
                base / "shard.safetensors",
                target / manifest["files"][0]["delta"],
                restored / "shard.safetensors",
                expected_size=len(target_bytes),
                expected_sha256=hashlib.sha256(target_bytes).hexdigest(),
            )
            self.assertEqual((restored / "shard.safetensors").read_bytes(), target_bytes)

    def test_checkpoint_delta_round_trip_uses_the_target_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            delta = root / "delta"
            target = root / "target"
            base.mkdir()
            delta.mkdir()
            target.mkdir()
            base_bytes = bytes(range(16))
            target_bytes = bytes(range(16, 32))
            base_file = base / "shard.safetensors"
            delta_file = delta / "shard.xor.zlib"
            target_file = target / "shard.safetensors"
            base_file.write_bytes(base_bytes)
            delta_file.write_bytes(
                zlib.compress(checkpoint_delta._xor_bytes(base_bytes, target_bytes))
            )
            checkpoint_delta.apply_xor_delta_file(
                base_file,
                delta_file,
                target_file,
                expected_size=len(target_bytes),
                expected_sha256=hashlib.sha256(target_bytes).hexdigest(),
            )
            self.assertEqual(target_file.read_bytes(), target_bytes)

    def test_checkpoint_delta_keeps_output_in_the_held_cache_tree(self):
        original_open = checkpoint_delta._open_directory_nofollow
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "cache"
                root.mkdir()
                base_dir = root / "base"
                delta_dir = root / "delta"
                target_dir = root / "target"
                base_dir.mkdir()
                delta_dir.mkdir()
                target_dir.mkdir()
                base_bytes = b"base"
                target_bytes = b"out!"
                base_file = base_dir / "shard.safetensors"
                delta_file = delta_dir / "shard.xor.zlib"
                target_file = target_dir / "shard.safetensors"
                base_file.write_bytes(base_bytes)
                delta_file.write_bytes(
                    zlib.compress(checkpoint_delta._xor_bytes(base_bytes, target_bytes))
                )

                external_root = Path(temporary) / "external"
                external_root.mkdir()
                external_delta = external_root / "delta"
                external_target = external_root / "target"
                external_delta.mkdir()
                external_target.mkdir()
                external_target_file = external_target / "shard.safetensors"
                external_target_file.write_bytes(b"must-survive")
                swapped = False

                def swap_after_common_root_open(path, *args, **kwargs):
                    nonlocal swapped
                    descriptor = original_open(path, *args, **kwargs)
                    if Path(path) == root and not swapped:
                        swapped = True
                        root.rename(Path(temporary) / "original-cache")
                        external_root.rename(root)
                    return descriptor

                checkpoint_delta._open_directory_nofollow = swap_after_common_root_open
                checkpoint_delta.apply_xor_delta_file(
                    base_file,
                    delta_file,
                    target_file,
                    expected_size=len(target_bytes),
                    expected_sha256=hashlib.sha256(target_bytes).hexdigest(),
                )
                self.assertTrue(swapped)
                self.assertEqual(
                    (Path(temporary) / "original-cache" / "target" / "shard.safetensors").read_bytes(),
                    target_bytes,
                )
                self.assertEqual(
                    (root / "target" / "shard.safetensors").read_bytes(),
                    b"must-survive",
                )
        finally:
            checkpoint_delta._open_directory_nofollow = original_open

    def test_checkpoint_delta_holds_target_before_delta_parent_can_move(self):
        original_open = checkpoint_delta._open_directory_below
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "cache"
                root.mkdir()
                base_dir = root / "base"
                delta_dir = root / "delta"
                target_dir = root / "target"
                base_dir.mkdir()
                delta_dir.mkdir()
                target_dir.mkdir()
                base_bytes = b"base"
                target_bytes = b"out!"
                (base_dir / "shard.safetensors").write_bytes(base_bytes)
                (delta_dir / "shard.xor.zlib").write_bytes(
                    zlib.compress(checkpoint_delta._xor_bytes(base_bytes, target_bytes))
                )
                external = Path(temporary) / "external-target"
                external.mkdir()
                (external / "shard.safetensors").write_bytes(b"must-survive")
                swapped = False

                def swap_when_delta_parent_opens(root_fd, root_path, path, label):
                    nonlocal swapped
                    descriptor = original_open(root_fd, root_path, path, label)
                    if Path(path) == delta_dir and not swapped:
                        swapped = True
                        target_dir.rename(root / "original-target")
                        external.rename(target_dir)
                    return descriptor

                checkpoint_delta._open_directory_below = swap_when_delta_parent_opens
                checkpoint_delta.apply_xor_delta_file(
                    base_dir / "shard.safetensors",
                    delta_dir / "shard.xor.zlib",
                    target_dir / "shard.safetensors",
                    expected_size=len(target_bytes),
                    expected_sha256=hashlib.sha256(target_bytes).hexdigest(),
                )
                self.assertTrue(swapped)
                self.assertEqual(
                    (root / "original-target" / "shard.safetensors").read_bytes(),
                    target_bytes,
                )
                self.assertEqual(
                    (target_dir / "shard.safetensors").read_bytes(),
                    b"must-survive",
                )
        finally:
            checkpoint_delta._open_directory_below = original_open

    def test_checkpoint_delta_rejects_symlink_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
            root = Path(temporary)
            outside = Path(external, "outside")
            outside.write_bytes(b"outside")
            base = root / "base"
            delta = root / "delta"
            target = root / "target"
            base.write_bytes(b"base")
            delta.write_bytes(zlib.compress(checkpoint_delta._xor_bytes(b"base", b"out!")))

            base.unlink()
            base.symlink_to(outside)
            with self.assertRaisesRegex(OSError, "checkpoint base file is a symbolic link"):
                checkpoint_delta.apply_xor_delta_file(
                    base,
                    delta,
                    target,
                    expected_size=4,
                    expected_sha256=hashlib.sha256(b"out!").hexdigest(),
                )
            base.unlink()
            base.write_bytes(b"base")

            delta.unlink()
            delta.symlink_to(outside)
            with self.assertRaisesRegex(OSError, "checkpoint delta file is a symbolic link"):
                checkpoint_delta.apply_xor_delta_file(
                    base,
                    delta,
                    target,
                    expected_size=4,
                    expected_sha256=hashlib.sha256(b"out!").hexdigest(),
                )
            delta.unlink()
            delta.write_bytes(zlib.compress(checkpoint_delta._xor_bytes(b"base", b"out!")))

            target.symlink_to(outside)
            with self.assertRaisesRegex(OSError, "checkpoint target file is a symbolic link"):
                checkpoint_delta.apply_xor_delta_file(
                    base,
                    delta,
                    target,
                    expected_size=4,
                    expected_sha256=hashlib.sha256(b"out!").hexdigest(),
                )
            target.unlink()

            assembling = target.with_name("target.assembling")
            assembling.symlink_to(outside)
            with self.assertRaisesRegex(OSError, "temporary file is a symbolic link"):
                checkpoint_delta.apply_xor_delta_file(
                    base,
                    delta,
                    target,
                    expected_size=4,
                    expected_sha256="0" * 64,
                )
            assembling.unlink()
            self.assertEqual(outside.read_bytes(), b"outside")

            target_parent = root / "target-parent"
            target_parent.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(OSError, "symbolic link"):
                checkpoint_delta.apply_xor_delta_file(
                    base,
                    delta,
                    target_parent / "target",
                    expected_size=4,
                    expected_sha256="0" * 64,
                )
            self.assertFalse((Path(external) / "target").exists())

    def test_predictive_chunk_round_trip_includes_header_and_bf16_data(self):
        base2 = safetensor_bytes([0x3F80, 0x4000, 0x4040, 0x4080])
        base1 = safetensor_bytes([0x4000, 0x4040, 0x4080, 0x40A0])
        target = safetensor_bytes([0x4040, 0x4080, 0x40A0, 0x40C0])
        for offset, end in ((0, 16), (16, len(target))):
            payload = adapter_delta.encode_predictive_chunk(
                base2[offset:end], base1[offset:end], target[offset:end], offset, 16
            )
            restored = adapter_delta.decode_predictive_chunk(
                base2[offset:end], base1[offset:end], payload, offset, 16
            )
            self.assertEqual(restored, target[offset:end])
            self.assertEqual(len(payload[: adapter_delta._predictive_header_size]), 64)

    def test_linear_decoder_decompresses_mask_and_values_once(self):
        base2 = safetensor_bytes([0x3F80, 0x4000, 0x4040, 0x4080])
        base1 = safetensor_bytes([0x4000, 0x4040, 0x4080, 0x40A0])
        target = safetensor_bytes([0x4040, 0x4080, 0x40A0, 0x40C0])
        payload = adapter_delta.encode_predictive_chunk(base2, base1, target, 0, 16)
        calls = []
        original_decompress = adapter_delta._decompress_bounded

        def counted_decompress(value, expected_size, *, maximum):
            calls.append((len(value), expected_size, maximum))
            return original_decompress(value, expected_size, maximum=maximum)

        adapter_delta._decompress_bounded = counted_decompress
        try:
            restored, correction = adapter_delta.decode_linear_zigzag_sparse_chunk(
                base2,
                base1,
                payload,
                offset=0,
                data_offset=16,
                raw_size=len(target),
                codec=adapter_delta._codec_name(),
            )
        finally:
            adapter_delta._decompress_bounded = original_decompress

        self.assertEqual(restored, target)
        self.assertEqual(correction, adapter_delta.predictive_correction_bytes(payload))
        self.assertEqual(len(calls), 2)

    def test_predictive_chunk_accepts_an_all_zero_residual(self):
        base = safetensor_bytes([0x3F80, 0x4000])
        payload = adapter_delta.encode_predictive_chunk(base, base, base, 0, 16)
        header = list(adapter_delta._predictive_header.unpack(payload[: adapter_delta._predictive_header_size]))
        mask_size = header[-2]
        header[-1] = 0
        payload = (
            adapter_delta._predictive_header.pack(*header)
            + payload[adapter_delta._predictive_header_size : adapter_delta._predictive_header_size + mask_size]
        )
        restored = adapter_delta.decode_predictive_chunk(base, base, payload, 0, 16)
        self.assertEqual(restored, base)
        self.assertEqual(adapter_delta.predictive_raw_sha256(payload), adapter_delta.hashlib.sha256(b"").hexdigest())
        self.assertEqual(adapter_delta.predictive_correction_bytes(payload), b"")

    def test_predictive_chunk_round_trip_with_millions_of_values(self):
        word_count = 2_000_000
        header = b'{"x":0} '
        prefix = struct.pack("<Q", len(header)) + header
        base2 = prefix + np.full(word_count, 0x3F80, dtype="<u2").tobytes()
        base1 = prefix + np.full(word_count, 0x4000, dtype="<u2").tobytes()
        target = prefix + np.full(word_count, 0x4000, dtype="<u2").tobytes()
        payload = adapter_delta.encode_predictive_chunk(base2, base1, target, 0, len(prefix))
        restored = adapter_delta.decode_predictive_chunk(base2, base1, payload, 0, len(prefix))
        self.assertEqual(restored, target)
        self.assertEqual(len(adapter_delta._decode_leb128_values(b"\x01" * word_count, word_count)), word_count)

    def test_predictive_odd_header_and_file_round_trip_preserves_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / f"step_{step}" for step in (1, 2, 3)]
            values = ([0x3F80] * 1024, [0x4000] * 1024, [0x4080] * 1024)
            for directory, words in zip(directories, values):
                directory.mkdir()
                (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
                (directory / "adapter_model.safetensors").write_bytes(
                    odd_offset_safetensor_bytes(words)
                )
            target = directories[2] / "adapter_model.safetensors"
            target_bytes = target.read_bytes()
            data_offset = adapter_delta.safetensors_data_offset(target, require_bf16=True)
            self.assertEqual(data_offset % 2, 1)
            self.assertEqual(len(target_bytes) % 2, 1)

            manifest = adapter_delta.publish_adapter_manifest(
                directories[2],
                directories[1],
                base2_dir=directories[0],
                target_url="http://source/step_3",
                base_url="http://source/step_2",
                base2_url="http://source/step_1",
                target_step=3,
                base_step=2,
                base2_step=1,
                chunk_size=1024,
                max_workers=1,
            )
            self.assertEqual(manifest["data_offset"], data_offset)
            self.assertEqual(manifest["target_sha256"], hashlib.sha256(target_bytes).hexdigest())
            self.assertTrue(
                any(item["offset"] % 2 for item in manifest["chunks"] if item["offset"] >= data_offset)
            )
            base2_bytes = (directories[0] / "adapter_model.safetensors").read_bytes()
            base1_bytes = (directories[1] / "adapter_model.safetensors").read_bytes()
            restored = bytearray(len(target_bytes))
            for item in manifest["chunks"]:
                start = item["offset"]
                end = start + item["raw_size"]
                payload = (directories[2] / item["file"]).read_bytes()
                restored[start:end] = adapter_delta.decode_predictive_chunk(
                    base2_bytes[start:end],
                    base1_bytes[start:end],
                    payload,
                    start,
                    data_offset,
                )
            self.assertEqual(bytes(restored), target_bytes)
            self.assertEqual(
                hashlib.sha256(restored).hexdigest(), manifest["target_sha256"]
            )

    def test_predictive_chunk_rejects_noncanonical_values_and_trailing_data(self):
        base = safetensor_bytes([0x3F80, 0x4000])
        payload = bytearray(adapter_delta.encode_predictive_chunk(base, base, base, 0, 16))
        header = list(adapter_delta._predictive_header.unpack(payload[:64]))
        values_offset = 64 + header[-2]
        values_offset += 0
        payload.extend(b"x")
        with self.assertRaises(ValueError):
            adapter_delta.decode_predictive_chunk(base, base, bytes(payload), 0, 16)

    def test_predictive_manifest_requires_three_adjacent_steps_and_keeps_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / f"step_{step}" for step in (1, 2, 3)]
            for directory in directories:
                directory.mkdir()
                (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
            base2_words = [0x3F80] * 1024
            base1_words = [0x4000] * 1024
            target_words = [0x4000] * 1024
            target_words[700] = 0x4040
            (directories[0] / "adapter_model.safetensors").write_bytes(valid_safetensor_bytes(base2_words))
            (directories[1] / "adapter_model.safetensors").write_bytes(valid_safetensor_bytes(base1_words))
            (directories[2] / "adapter_model.safetensors").write_bytes(valid_safetensor_bytes(target_words))
            data_offset = int.from_bytes(
                (directories[2] / "adapter_model.safetensors").read_bytes()[:8], "little"
            ) + 8
            manifest = adapter_delta.publish_adapter_manifest(
                directories[2],
                directories[1],
                base2_dir=directories[0],
                target_step=3,
                base_step=2,
                base2_step=1,
                chunk_size=1024,
                max_workers=1,
            )
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["mode"], "delta")
            self.assertEqual(manifest["transport"], adapter_delta.predictive_transport_name)
            self.assertEqual(manifest["data_offset"], data_offset)
            self.assertEqual(manifest["chunk_alignment"], 2)
            self.assertEqual(
                manifest["chunk_coverage"],
                {"offset": 0, "size": (directories[2] / "adapter_model.safetensors").stat().st_size},
            )
            self.assertEqual(manifest["base2_step"], 1)
            self.assertTrue(all(item["offset"] % 2 == 0 and item["raw_size"] % 2 == 0 for item in manifest["chunks"]))
            with self.assertRaises(ValueError):
                adapter_delta.publish_adapter_manifest(
                    directories[2],
                    directories[1],
                    base2_dir=directories[0],
                    target_step=4,
                    base_step=2,
                    base2_step=1,
                    chunk_size=1024,
                    max_workers=1,
                )

    def test_manifest_hashes_match_single_read_streams_for_all_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def make_adapter(name, weights):
                directory = root / name
                directory.mkdir()
                (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
                (directory / "adapter_model.safetensors").write_bytes(weights)
                return directory

            full_weights = b"full adapter\n"
            base_words = [0x4000] * 2048
            target_words = base_words.copy()
            target_words[1300] = 0x4040
            base2_words = [0x3F80] * 2048
            base_weights = valid_safetensor_bytes(base_words)
            target_weights = valid_safetensor_bytes(target_words)
            base2_weights = valid_safetensor_bytes(base2_words)
            full = make_adapter("full", full_weights)
            base = make_adapter("base", base_weights)
            target = make_adapter("target", target_weights)
            base2 = make_adapter("base2", base2_weights)
            legacy_base = make_adapter("legacy-base", b"too short")

            original_sha256_file = adapter_delta.sha256_file
            weight_hash_calls = 0

            def counted_sha256_file(path):
                nonlocal weight_hash_calls
                if Path(path).name == adapter_delta.adapter_weights_name:
                    weight_hash_calls += 1
                return original_sha256_file(path)

            adapter_delta.sha256_file = counted_sha256_file
            try:
                full_manifest = adapter_delta.publish_adapter_manifest(full)
                delta_manifest = adapter_delta.publish_adapter_manifest(
                    target, base, chunk_size=1024, max_workers=3
                )
                predictive_manifest = adapter_delta.publish_adapter_manifest(
                    target, base, base2_dir=base2, target_step=3, base_step=2,
                    base2_step=1, chunk_size=1024, max_workers=3,
                )
                self.assertEqual(weight_hash_calls, 1)
                legacy_manifest = adapter_delta.publish_adapter_manifest(
                    target, legacy_base, base_url="http://source/base",
                    base_step=4, target_step=5,
                )
            finally:
                adapter_delta.sha256_file = original_sha256_file

            def digest(value):
                return hashlib.sha256(value).hexdigest()

            self.assertEqual(full_manifest["target_sha256"], digest(full_weights))
            self.assertEqual(delta_manifest["target_sha256"], digest(target_weights))
            self.assertEqual(delta_manifest["base_sha256"], digest(base_weights))
            self.assertEqual(predictive_manifest["target_sha256"], digest(target_weights))
            self.assertEqual(predictive_manifest["base_sha256"], digest(base_weights))
            self.assertEqual(predictive_manifest["base2_sha256"], digest(base2_weights))
            self.assertEqual(weight_hash_calls, 2)
            for manifest in (delta_manifest, predictive_manifest):
                self.assertEqual(
                    [item["offset"] for item in manifest["chunks"]],
                    sorted(item["offset"] for item in manifest["chunks"]),
                )
            self.assertEqual(legacy_manifest["mode"], "full")
            self.assertNotIn("base_url", legacy_manifest)
            self.assertIsNone(legacy_manifest["base_step"])

    def test_encode_error_removes_partial_transport_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            base = root / "base"
            target.mkdir()
            base.mkdir()
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")
            (base / "adapter_config.json").write_text("{}", encoding="utf-8")
            target_weights = target / "adapter_model.safetensors"
            target_weights.write_bytes(b"abcdefgh")
            (base / "adapter_model.safetensors").write_bytes(b"abcdefgz")
            original_open = Path.open

            class ShortReader:
                def __init__(self, source):
                    self.source = source
                    self.reads = 0

                def read(self, size=-1):
                    self.reads += 1
                    return b"" if self.reads == 2 else self.source.read(size)

                def __enter__(self):
                    self.source.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.source.__exit__(*args)

                def __getattr__(self, name):
                    return getattr(self.source, name)

            def flaky_open(path, *args, **kwargs):
                source = original_open(path, *args, **kwargs)
                return ShortReader(source) if path == target_weights else source

            Path.open = flaky_open
            try:
                with self.assertRaisesRegex(OSError, "adapter file ended early"):
                    adapter_delta.publish_adapter_manifest(
                        target, base, chunk_size=4, max_workers=1
                    )
            finally:
                Path.open = original_open
            self.assertFalse(list(target.glob("*.transport-*")))

    def test_atomic_writers_remove_only_their_partials_on_replace_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unrelated = root / ".unrelated.partial"
            unrelated.write_bytes(b"keep")
            original_replace = adapter_delta.os.replace
            seen = []

            def fail_replace(source, destination):
                seen.append(Path(source))
                raise OSError("replace failed")

            adapter_delta.os.replace = fail_replace
            try:
                for writer, path, value in (
                    (adapter_delta._atomic_write, root / "output", b"value"),
                    (adapter_delta._atomic_json, root / "manifest", {"value": "x"}),
                ):
                    with self.assertRaisesRegex(OSError, "replace failed"):
                        writer(path, value)
            finally:
                adapter_delta.os.replace = original_replace
            manifest = root / "manifest-ok"
            adapter_delta._atomic_json(manifest, {"value": "x"})
            self.assertEqual(manifest.read_bytes(), b'{"value":"x"}\n')
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(path.parent == root and not path.exists() for path in seen))
            self.assertTrue(unrelated.exists())

    def test_predictive_manifest_rejects_non_bf16_tensor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / f"step_{step}" for step in (1, 2, 3)]
            for directory in directories:
                directory.mkdir()
                (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
            words = [0x3F80] * 128
            (directories[0] / "adapter_model.safetensors").write_bytes(valid_safetensor_bytes(words))
            (directories[1] / "adapter_model.safetensors").write_bytes(valid_safetensor_bytes(words))
            invalid_dtype = valid_safetensor_bytes(words).replace(b'"BF16"', b'"FP16"')
            (directories[2] / "adapter_model.safetensors").write_bytes(invalid_dtype)
            with self.assertRaisesRegex(ValueError, "requires BF16"):
                adapter_delta.publish_adapter_manifest(
                    directories[2], directories[1], base2_dir=directories[0],
                    target_step=3, base_step=2, base2_step=1,
                    chunk_size=1024, max_workers=1,
                )


if __name__ == "__main__":
    unittest.main()
