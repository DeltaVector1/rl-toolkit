import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prime_rl.utils.adapter_references import (
    ADAPTER_REFERENCE_HEADER,
    ADAPTER_REFERENCE_MANIFEST,
    ADAPTER_REFERENCE_PAYLOAD,
    AdapterReferenceError,
    apply_adapter_references,
    build_adapter_references,
    load_adapter_reference_manifest,
)
from source.tests.test_adapter_cache_identity import _load_filesystem_module


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


def _reference_export(root: Path) -> Path:
    source = root / "step_1"
    source.mkdir()
    model = source / "adapter_model.safetensors"
    _write_adapter(model)
    (source / "adapter_config.json").write_text('{"r":128}\n', encoding="utf-8")
    (source / "adapter_model.safetensors.sha256").write_text(
        hashlib.sha256(model.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    build_adapter_references(source)
    return source


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

    def test_staging_reconstructs_reference_without_full_transfer(self):
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _reference_export(root)
            calls: list[str] = []

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                self.assertTrue(create_only)
                calls.append(source_url)
                shutil.copyfile(source / source_url.rsplit("/", 1)[-1], destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_REFERENCES": "1",
                    "PRIME_ADAPTER_DELTA": "0",
                    "PRIME_GLM_ADAPTER_CODEC": "identity",
                    "PRIME_ADAPTER_PORTS": "",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                destination = filesystem.stage_lora_adapter_files(
                    "http://adapter/step_1"
                )

            self.assertEqual(
                (destination / "adapter_model.safetensors").read_bytes(),
                (source / "adapter_model.safetensors").read_bytes(),
            )
            self.assertTrue((destination / "READY").is_file())
            self.assertFalse(
                any(url.endswith("/adapter_model.safetensors") for url in calls)
            )

    def test_corrupt_advertised_reference_is_terminal_before_full_transfer(self):
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _reference_export(root)
            payload = source / ADAPTER_REFERENCE_PAYLOAD
            corrupted = bytearray(payload.read_bytes())
            corrupted[-1] ^= 1
            payload.write_bytes(corrupted)
            calls: list[str] = []

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                calls.append(source_url)
                shutil.copyfile(source / source_url.rsplit("/", 1)[-1], destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_REFERENCES": "1",
                    "PRIME_ADAPTER_DELTA": "0",
                    "PRIME_GLM_ADAPTER_CODEC": "identity",
                    "PRIME_ADAPTER_PORTS": "",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                with self.assertRaisesRegex(
                    filesystem.AdapterReferenceError, "payload identity differs"
                ):
                    filesystem.stage_lora_adapter_files("http://adapter/step_1")

            self.assertFalse(
                any(url.endswith("/adapter_model.safetensors") for url in calls)
            )

    def test_dead_reference_payload_tunnel_is_terminal_before_full_transfer(self):
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _reference_export(root)
            calls: list[str] = []

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                calls.append(source_url)
                if source_url.endswith(f"/{ADAPTER_REFERENCE_PAYLOAD}"):
                    raise ConnectionRefusedError("reference tunnel is down")
                shutil.copyfile(source / source_url.rsplit("/", 1)[-1], destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_REFERENCES": "1",
                    "PRIME_ADAPTER_DELTA": "0",
                    "PRIME_GLM_ADAPTER_CODEC": "identity",
                    "PRIME_ADAPTER_PORTS": "",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                with self.assertRaisesRegex(OSError, "artifact transfer failed"):
                    filesystem.stage_lora_adapter_files("http://adapter/step_1")

            self.assertFalse(
                any(url.endswith("/adapter_model.safetensors") for url in calls)
            )


if __name__ == "__main__":
    unittest.main()
