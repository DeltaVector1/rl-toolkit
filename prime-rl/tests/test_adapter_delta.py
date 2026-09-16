import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from prime_rl.utils.checkpoint_delta import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_FILE,
    DELTA_MANIFEST,
    apply_xor_delta_file,
    build_xor_delta_adapter,
    sha256_file,
)


class AdapterDeltaTest(unittest.TestCase):
    def test_consecutive_adapter_delta_reconstructs_exact_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            base = root_path / "step_4"
            target = root_path / "step_5"
            base.mkdir()
            target.mkdir()
            base_bytes = bytes((index * 17 + 3) % 256 for index in range(131_071))
            target_bytes = bytearray(base_bytes)
            for index in range(0, len(target_bytes), 97):
                target_bytes[index] ^= (index * 11 + 5) % 256
            (base / ADAPTER_FILE).write_bytes(base_bytes)
            (target / ADAPTER_FILE).write_bytes(target_bytes)
            config = b'{"r":32,"lora_alpha":64}\n'
            (base / ADAPTER_CONFIG_FILE).write_bytes(config)
            (target / ADAPTER_CONFIG_FILE).write_bytes(config)

            manifest_path = build_xor_delta_adapter(
                base,
                target,
                base_url="http://example/step_4",
                target_url="http://example/step_5",
            )
            self.assertEqual(manifest_path, target / DELTA_MANIFEST)
            manifest = json.loads(manifest_path.read_text())
            spec = manifest["files"][0]
            self.assertEqual(spec["size"], len(target_bytes))
            self.assertEqual(spec["sha256"], sha256_file(target / ADAPTER_FILE))
            self.assertEqual(spec["base_sha256"], sha256_file(base / ADAPTER_FILE))

            reconstructed = root_path / "reconstructed.safetensors"
            apply_xor_delta_file(
                base / ADAPTER_FILE,
                target / spec["delta"],
                reconstructed,
                expected_size=spec["size"],
                expected_sha256=spec["sha256"],
                expected_base_sha256=spec["base_sha256"],
            )
            self.assertEqual(reconstructed.read_bytes(), target_bytes)

    def test_delta_rejects_wrong_base_checksum(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            base = root_path / "base"
            target = root_path / "target"
            base.mkdir()
            target.mkdir()
            (base / ADAPTER_FILE).write_bytes(b"base-data")
            delta = target / "delta.zlib"
            import zlib

            replacement = b"new-data!"
            delta.write_bytes(zlib.compress(bytes(a ^ b for a, b in zip(b"base-data", replacement))))
            expected_target = hashlib.sha256(replacement).hexdigest()
            with self.assertRaisesRegex(OSError, "base checksum mismatch"):
                apply_xor_delta_file(
                    base / ADAPTER_FILE,
                    delta,
                    root_path / "output",
                    expected_size=9,
                    expected_sha256=expected_target,
                    expected_base_sha256="0" * 64,
                )


if __name__ == "__main__":
    unittest.main()
