import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from prime_rl.utils.adapter_references import (
    ADAPTER_REFERENCE_HEADER,
    ADAPTER_REFERENCE_MANIFEST,
    ADAPTER_REFERENCE_PAYLOAD,
    AdapterReferenceError,
    apply_adapter_references,
    build_adapter_references,
    load_adapter_reference_manifest,
)


def _write_adapter(path: Path) -> bytes:
    first = b"same tensor bytes" * 64
    second = b"same tensor bytes" * 64
    third = bytes(range(256)) * 32
    payload = first + second + third
    header = json.dumps(
        {
            "first": {"dtype": "BF16", "shape": [len(first) // 2], "data_offsets": [0, len(first)]},
            "second": {"dtype": "BF16", "shape": [len(second) // 2], "data_offsets": [len(first), len(first) + len(second)]},
            "third": {"dtype": "BF16", "shape": [len(third) // 2], "data_offsets": [len(first) + len(second), len(payload)]},
            "__metadata__": {"format": "pt"},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return payload


class AdapterReferencesTest(unittest.TestCase):
    def test_deduplicates_and_reconstructs_exact_bytes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "step_1"
            source.mkdir()
            model = source / "adapter_model.safetensors"
            _write_adapter(model)
            config = source / "adapter_config.json"
            config.write_text('{"r":128}\n', encoding="utf-8")
            original = model.read_bytes()

            manifest_path = build_adapter_references(source)
            self.assertEqual(manifest_path, source / ADAPTER_REFERENCE_MANIFEST)
            manifest = load_adapter_reference_manifest(manifest_path)
            self.assertLess(manifest["unique_payload_size"], manifest["payload_size"])
            self.assertEqual((source / ADAPTER_REFERENCE_HEADER).read_bytes(), original[: manifest["header_size"]])

            restored = root / "restored.safetensors"
            apply_adapter_references(
                source / ADAPTER_REFERENCE_HEADER,
                source / ADAPTER_REFERENCE_PAYLOAD,
                restored,
                manifest,
            )
            self.assertEqual(restored.read_bytes(), original)

            payload = source / ADAPTER_REFERENCE_PAYLOAD
            altered = bytearray(payload.read_bytes())
            altered[-1] ^= 1
            payload.write_bytes(altered)
            with self.assertRaises(AdapterReferenceError):
                apply_adapter_references(
                    source / ADAPTER_REFERENCE_HEADER,
                    payload,
                    root / "rejected.safetensors",
                    manifest,
                )
            self.assertFalse((root / "rejected.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
