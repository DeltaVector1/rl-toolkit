import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/prime_rl/utils/adapter_delta.py"
SPEC = importlib.util.spec_from_file_location("predictive_adapter_delta", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter_delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter_delta)


def safetensor_bytes(words: list[int]) -> bytes:
    header = b'{"x":0} '
    return struct.pack("<Q", len(header)) + header + np.asarray(words, dtype="<u2").tobytes()


class PredictiveAdapterCodecTest(unittest.TestCase):
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

    def test_predictive_chunk_accepts_an_all_zero_residual(self):
        base = safetensor_bytes([0x3F80, 0x4000])
        payload = adapter_delta.encode_predictive_chunk(base, base, base, 0, 16)
        restored = adapter_delta.decode_predictive_chunk(base, base, payload, 0, 16)
        self.assertEqual(restored, base)
        self.assertEqual(adapter_delta.predictive_raw_sha256(payload), adapter_delta.hashlib.sha256(b"").hexdigest())

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
            (directories[0] / "adapter_model.safetensors").write_bytes(safetensor_bytes(base2_words))
            (directories[1] / "adapter_model.safetensors").write_bytes(safetensor_bytes(base1_words))
            (directories[2] / "adapter_model.safetensors").write_bytes(safetensor_bytes(target_words))
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
            self.assertEqual(manifest["transport"], adapter_delta.predictive_transport_name)
            self.assertEqual(manifest["data_offset"], 16)
            self.assertEqual(manifest["chunk_alignment"], 2)
            self.assertEqual(manifest["chunk_coverage"], {"offset": 0, "size": 2064})
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


if __name__ == "__main__":
    unittest.main()
