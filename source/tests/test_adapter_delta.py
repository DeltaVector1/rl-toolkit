"""Synthetic tests for the opt-in adjacent adapter delta protocol."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prime_rl.utils.adapter_delta import (
    ADAPTER_DELTA_FILE,
    AdapterDeltaError,
    apply_adapter_delta,
    build_adapter_delta,
    load_adapter_delta_manifest,
    load_adapter_delta_manifest_from_mapping,
    adapter_cache_identity,
    safetensors_metadata_sha256,
    select_adapter_transfer,
    wire_size_break_even,
)
from source.tests.test_adapter_cache_identity import _load_filesystem_module


def _write_safetensors(path: Path, payload: bytes) -> None:
    header = json.dumps(
        {
            "weight": {
                "dtype": "BF16",
                "shape": [len(payload) // 2],
                "data_offsets": [0, len(payload)],
            },
            "__metadata__": {"format": "pt"},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _export(root: Path, name: str, payload: bytes) -> Path:
    directory = root / name
    directory.mkdir()
    model = directory / "adapter_model.safetensors"
    _write_safetensors(model, payload)
    (directory / "adapter_config.json").write_text('{"r":128}\n', encoding="utf-8")
    (directory / "adapter_model.safetensors.sha256").write_text(
        hashlib.sha256(model.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    return directory


class AdapterDeltaTest(unittest.TestCase):
    def test_adjacent_bf16_reconstructs_byte_for_byte(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = bytes((index * 17) % 256 for index in range(16 * 1024))
            changed = bytearray(payload)
            for index in range(0, len(changed), 257):
                changed[index] ^= 0x5A
            base = _export(root, "step_1", payload)
            target = _export(root, "step_2", bytes(changed))
            manifest_path = build_adapter_delta(base, target, base_url="base", target_url="target")
            self.assertIsNotNone(manifest_path)
            manifest = load_adapter_delta_manifest(manifest_path)
            self.assertEqual(manifest["base_sha256"], hashlib.sha256((base / "adapter_model.safetensors").read_bytes()).hexdigest())
            self.assertEqual(manifest["target_sha256"], hashlib.sha256((target / "adapter_model.safetensors").read_bytes()).hexdigest())
            self.assertEqual(manifest["adapter_config_sha256"], hashlib.sha256((target / "adapter_config.json").read_bytes()).hexdigest())
            self.assertEqual(manifest["tensor_metadata_sha256"], safetensors_metadata_sha256(target / "adapter_model.safetensors"))
            restored = root / "restored.safetensors"
            apply_adapter_delta(base / "adapter_model.safetensors", target / ADAPTER_DELTA_FILE, restored, manifest)
            self.assertEqual(restored.read_bytes(), (target / "adapter_model.safetensors").read_bytes())

    def test_corrupt_frame_does_not_publish_a_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _export(root, "step_1", b"\0" * 8192)
            target = _export(root, "step_2", b"\0" * 8191 + b"x")
            build_adapter_delta(base, target, max_ratio=0.99)
            delta = target / ADAPTER_DELTA_FILE
            corrupt = bytearray(delta.read_bytes())
            corrupt[-1] ^= 1
            delta.write_bytes(corrupt)
            manifest = load_adapter_delta_manifest(target / "adapter_delta.json")
            manifest["delta_sha256"] = hashlib.sha256(corrupt).hexdigest()
            active = root / "active.safetensors"
            active.write_bytes(b"active adapter")
            with self.assertRaises(AdapterDeltaError):
                apply_adapter_delta(
                    base / "adapter_model.safetensors",
                    delta,
                    active,
                    manifest,
                )
            self.assertEqual(active.read_bytes(), b"active adapter")

    def test_wrong_base_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _export(root, "step_1", b"a" * 8192)
            target = _export(root, "step_2", b"b" * 8192)
            wrong = _export(root, "wrong", b"c" * 8192)
            build_adapter_delta(base, target, max_ratio=0.99)
            output = root / "wrong.safetensors"
            with self.assertRaisesRegex(AdapterDeltaError, "base SHA"):
                apply_adapter_delta(
                    wrong / "adapter_model.safetensors",
                    target / ADAPTER_DELTA_FILE,
                    output,
                    load_adapter_delta_manifest(target / "adapter_delta.json"),
                )
            self.assertFalse(output.exists())

    def test_interrupted_delta_is_rejected_and_temp_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _export(root, "step_1", b"\0" * 8192)
            target = _export(root, "step_2", b"\0" * 8191 + b"x")
            build_adapter_delta(base, target, max_ratio=0.99)
            delta = target / ADAPTER_DELTA_FILE
            delta.write_bytes(delta.read_bytes()[:-3])
            output = root / "interrupted.safetensors"
            with self.assertRaises(AdapterDeltaError):
                apply_adapter_delta(
                    base / "adapter_model.safetensors",
                    delta,
                    output,
                    load_adapter_delta_manifest(target / "adapter_delta.json"),
                )
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".*interrupted.safetensors.*assembling")))

    def test_same_url_cache_identity_changes_with_target_digest(self) -> None:
        url = "http://127.0.0.1:8100/runs/adjacent"
        old = "0" * 64
        new = "1" * 64
        self.assertNotEqual(adapter_cache_identity(url, old), adapter_cache_identity(url, new))
        self.assertEqual(
            adapter_cache_identity(url, old),
            adapter_cache_identity("http://127.0.0.1:8100/runs/other-step", old),
        )

    def test_manifest_requires_base_identity_and_frame_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _export(root, "step_1", b"a" * 8192)
            target = _export(root, "step_2", b"b" * 8192)
            build_adapter_delta(base, target, max_ratio=0.99)
            manifest = load_adapter_delta_manifest(target / "adapter_delta.json")
            for field in ("base_size", "base_adapter_config_sha256", "base_tensor_metadata_sha256", "frame_bytes"):
                invalid = dict(manifest)
                invalid.pop(field)
                with self.assertRaises(AdapterDeltaError):
                    load_adapter_delta_manifest_from_mapping(invalid)

    def test_oversize_frame_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _export(root, "step_1", b"a" * 8192)
            target = _export(root, "step_2", b"b" * 8192)
            build_adapter_delta(base, target, max_ratio=0.99)
            manifest = load_adapter_delta_manifest(target / "adapter_delta.json")
            manifest["frame_bytes"] = 1
            output = root / "oversize.safetensors"
            with self.assertRaisesRegex(AdapterDeltaError, "frame header"):
                apply_adapter_delta(
                    base / "adapter_model.safetensors",
                    target / ADAPTER_DELTA_FILE,
                    output,
                    manifest,
                )
            self.assertFalse(output.exists())

    def test_two_step_different_urls_reuses_base_without_full_target_download(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "step1": _export(root, "step1", b"a" * 8192),
                "step2": _export(root, "step2", b"a" * 8191 + b"b"),
            }
            build_adapter_delta(sources["step1"], sources["step2"], max_ratio=0.99)
            calls: list[str] = []

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                self.assertTrue(create_only)
                calls.append(source_url)
                parts = source_url.rstrip("/").split("/")
                step, filename = parts[-2], parts[-1]
                source = sources[step] / filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                shutil.copyfile(source, destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_DELTA": "1",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                first = filesystem.stage_lora_adapter_files("http://adapter/step1")
                calls.clear()
                second = filesystem.stage_lora_adapter_files("http://adapter/step2")

            self.assertEqual(
                first.joinpath("adapter_model.safetensors").read_bytes(),
                sources["step1"].joinpath("adapter_model.safetensors").read_bytes(),
            )
            self.assertEqual(
                second.joinpath("adapter_model.safetensors").read_bytes(),
                sources["step2"].joinpath("adapter_model.safetensors").read_bytes(),
            )
            self.assertTrue(any(url.endswith("/adapter_model.safetensors.delta") for url in calls))
            self.assertFalse(any(url.endswith("/adapter_model.safetensors") for url in calls))

    def test_cache_prune_caps_stale_entries_and_keeps_active_base(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            keep_active = cache / "active"
            keep_base = cache / "base"
            stale_old = cache / "stale-old"
            stale_new = cache / "stale-new"
            for entry in (keep_active, keep_base, stale_old, stale_new):
                entry.mkdir(parents=True)
                (entry / "READY").write_text("{}")
                (entry / "payload").write_bytes(b"x" * 10)
            os.utime(stale_old, (1, 1))
            os.utime(stale_new, (2, 2))
            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_MAX_BYTES": "10000",
                    "PRIME_LORA_CACHE_MAX_STALE_ENTRIES": "1",
                },
                clear=False,
            ):
                filesystem._prune_adapter_cache(cache, {keep_active, keep_base})
            self.assertTrue(keep_active.exists())
            self.assertTrue(keep_base.exists())
            self.assertFalse(stale_old.exists())
            self.assertTrue(stale_new.exists())

    def test_fallback_selection_is_explicit(self) -> None:
        manifest = {
            "base_sha256": "a" * 64,
            "target_sha256": "b" * 64,
            "uncompressed_size": 1000,
            "delta_size": 100,
        }
        self.assertEqual(
            select_adapter_transfer(
                manifest,
                base_available=True,
                base_sha256="a" * 64,
                target_sha256="b" * 64,
                target_size=1000,
                enabled=True,
                full_wire_size=1000,
                manifest_wire_size=20,
            ),
            "delta",
        )
        self.assertEqual(
            select_adapter_transfer(
                manifest,
                base_available=True,
                base_sha256="a" * 64,
                target_sha256="b" * 64,
                target_size=1000,
                enabled=True,
            ),
            "full",
        )
        self.assertEqual(
            select_adapter_transfer(
                manifest,
                base_available=False,
                base_sha256=None,
                target_sha256="b" * 64,
                target_size=1000,
                enabled=True,
            ),
            "full",
        )
        self.assertEqual(wire_size_break_even(1000, 20), 979)
        self.assertEqual(
            select_adapter_transfer(
                {**manifest, "delta_size": 950},
                base_available=True,
                base_sha256="a" * 64,
                target_sha256="b" * 64,
                target_size=1000,
                enabled=True,
                full_wire_size=1000,
                manifest_wire_size=20,
            ),
            "full",
        )
        self.assertEqual(
            select_adapter_transfer(
                manifest,
                base_available=True,
                base_sha256="wrong",
                target_sha256="b" * 64,
                target_size=1000,
                enabled=True,
            ),
            "full",
        )


if __name__ == "__main__":
    unittest.main()
