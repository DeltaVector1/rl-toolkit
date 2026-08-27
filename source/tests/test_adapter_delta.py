"""Synthetic tests for the opt-in adjacent adapter delta protocol."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import shutil
import struct
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import patch

import prime_rl.utils.adapter_delta as adapter_delta_module
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
    def test_xor_uses_the_compiled_numpy_helper(self) -> None:
        expected = bytes([0xFF, 0x00, 0xA5])
        with patch.object(
            adapter_delta_module.np,
            "bitwise_xor",
            wraps=adapter_delta_module.np.bitwise_xor,
        ) as helper:
            actual = adapter_delta_module._xor(
                bytes([0xF0, 0x0F, 0xAA]),
                bytes([0x0F, 0x0F, 0x0F]),
            )
        self.assertEqual(actual, expected)
        helper.assert_called_once()

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
                    raise filesystem.AdapterArtifactNotFound(source)
                shutil.copyfile(source, destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_DELTA": "1",
                    "PRIME_GLM_ADAPTER_CODEC": "gzip",
                    "PRIME_ADAPTER_FULL_WIRE_SIZE": "100000",
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

    def test_compressed_break_even_requires_current_wire_size(self) -> None:
        filesystem = _load_filesystem_module()
        with patch.dict(
            os.environ,
            {
                "PRIME_GLM_ADAPTER_CODEC": "gzip",
                "PRIME_ADAPTER_FULL_WIRE_SIZE": "80",
            },
            clear=False,
        ):
            with patch.object(filesystem, "_content_length", return_value=80):
                self.assertEqual(
                    filesystem._full_wire_size_estimate("http://adapter/model", 100),
                    80,
                )
            with patch.dict(os.environ, {"PRIME_ADAPTER_FULL_WIRE_SIZE": ""}), patch.object(
                filesystem.urllib.request, "urlopen", side_effect=OSError("no HEAD")
            ):
                with self.assertRaisesRegex(OSError, "comparison HEAD failed"):
                    filesystem._full_wire_size_estimate("http://adapter/model", 100)
            class Response:
                headers = {
                    "X-Prime-Wire-Length": "80",
                    "X-Prime-Raw-Length": "99",
                    "X-Prime-Wire-Codec": "gzip",
                }

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            with patch.dict(os.environ, {"PRIME_ADAPTER_FULL_WIRE_SIZE": ""}), patch.object(
                filesystem.urllib.request, "urlopen", return_value=Response()
            ):
                with self.assertRaisesRegex(OSError, "raw length differs"):
                    filesystem._full_wire_size_estimate("http://adapter/model", 100)

            with patch.dict(os.environ, {"PRIME_ADAPTER_FULL_WIRE_SIZE": ""}), patch.object(
                filesystem.urllib.request,
                "urlopen",
                side_effect=HTTPError(
                    "http://adapter/model", 404, "not found", {}, None
                ),
            ):
                self.assertIsNone(
                    filesystem._full_wire_size_estimate("http://adapter/model", 100)
                )

    def test_delta_runtime_gate_allows_gzip_only(self) -> None:
        filesystem = _load_filesystem_module()
        self.assertTrue(filesystem._adapter_delta_codec_supported("gzip"))
        self.assertFalse(filesystem._adapter_delta_codec_supported("zstd"))
        self.assertFalse(filesystem._adapter_delta_codec_supported("identity"))
    def test_range_server_generates_full_gzip_wire_metadata(self) -> None:
        from scripts.ops.glm_adapter_range_server import _FULL_WIRE_CHUNK_BYTES, _full_wire_length

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            payload = b"adapter bytes" * 100
            path.write_bytes(payload)
            expected = len(gzip.compress(payload, compresslevel=1, mtime=0))
            self.assertEqual(_full_wire_length(str(path), "gzip"), expected)
            self.assertLess(_FULL_WIRE_CHUNK_BYTES, 1024 * 1024 * 1024)

    def test_interrupted_partial_cache_is_removed_and_recent_partial_is_counted(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            old_partial = cache / "source.partial-interrupted"
            recent_partial = cache / "source.partial-active"
            old_partial.mkdir(parents=True)
            recent_partial.mkdir(parents=True)
            (old_partial / "model").write_bytes(b"old")
            (recent_partial / "model").write_bytes(b"recent")
            os.utime(old_partial, (1, 1))
            with patch.dict(
                os.environ,
                {"PRIME_LORA_CACHE_PARTIAL_MAX_AGE_SECONDS": "1"},
                clear=False,
            ):
                filesystem._prune_adapter_cache(cache, set())
            self.assertFalse(old_partial.exists())
            self.assertTrue(recent_partial.exists())
            self.assertGreaterEqual(filesystem._cache_adapter_bytes(cache), len(b"recent"))

    def test_disk_preflight_requires_active_base_and_assembling_space(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            active = cache / "active"
            active.mkdir(parents=True)
            (active / "adapter_model.safetensors").write_bytes(b"x" * 20)
            with patch.object(
                filesystem.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=169),
            ):
                with self.assertRaises(filesystem.AdapterDiskSpaceError):
                    filesystem._check_adapter_disk_space(cache, 100, 50, guard_bytes=0)
            with patch.object(
                filesystem.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=170),
            ):
                filesystem._check_adapter_disk_space(cache, 100, 50, guard_bytes=0)

    def test_disk_preflight_30_77gb_boundary_counts_delta_and_guard_once(self) -> None:
        filesystem = _load_filesystem_module()
        adapter = 30_770_000_000
        delta = 10_000_000
        manifest = 2_000
        guard = 256 * 1024 * 1024
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            active = cache / "active"
            base = cache / "base"
            active.mkdir(parents=True)
            base.mkdir(parents=True)
            active_model = active / "adapter_model.safetensors"
            base_model = base / "adapter_model.safetensors"
            with active_model.open("wb") as output:
                output.truncate(adapter)
            with base_model.open("wb") as output:
                output.truncate(adapter)
            required = adapter + adapter + delta + manifest + adapter + guard
            with patch.object(
                filesystem.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=required),
            ):
                filesystem._check_adapter_disk_space(
                    cache,
                    adapter,
                    adapter,
                    delta,
                    manifest,
                    active_path=active_model,
                    base_path=base_model,
                    guard_bytes=guard,
                )
            with patch.object(
                filesystem.shutil,
                "disk_usage",
                return_value=SimpleNamespace(free=required - 1),
            ):
                with self.assertRaises(filesystem.AdapterDiskSpaceError):
                    filesystem._check_adapter_disk_space(
                        cache,
                        adapter,
                        adapter,
                        delta,
                        manifest,
                        active_path=active_model,
                        base_path=base_model,
                        guard_bytes=guard,
                    )
            with patch.object(
                filesystem.shutil,
                "disk_usage",
                return_value=SimpleNamespace(
                    free=adapter + adapter + delta + manifest + guard
                ),
            ):
                # Active and base are the same generation: do not double count.
                filesystem._check_adapter_disk_space(
                    cache,
                    adapter,
                    adapter,
                    delta,
                    manifest,
                    active_path=active_model,
                    base_path=active_model,
                    guard_bytes=guard,
                )

    def test_cache_global_lock_serializes_different_source_urls(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "one": _export(root, "one", b"a" * 8192),
                "two": _export(root, "two", b"b" * 8192),
            }
            active = 0
            maximum = 0
            state_lock = threading.Lock()

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                nonlocal active, maximum
                step, filename = source_url.rstrip("/").split("/")[-2:]
                source = sources[step] / filename
                with state_lock:
                    active += 1
                    maximum = max(maximum, active)
                try:
                    time.sleep(0.02)
                    shutil.copyfile(source, destination)
                finally:
                    with state_lock:
                        active -= 1

            with patch.dict(
                os.environ,
                {"PRIME_LORA_CACHE_DIR": str(root / "cache")},
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(
                        pool.map(
                            filesystem.stage_lora_adapter_files,
                            ("http://adapter/one", "http://adapter/two"),
                        )
                    )
            self.assertEqual(maximum, 1)

    def test_prune_failure_does_not_invalidate_verified_ready_target(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _export(root, "source", b"a" * 8192)

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                shutil.copyfile(source / source_url.rsplit("/", 1)[-1], destination)

            with patch.dict(
                os.environ,
                {"PRIME_LORA_CACHE_DIR": str(root / "cache")},
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download), patch.object(
                filesystem,
                "_prune_adapter_cache_locked",
                side_effect=OSError("cleanup unavailable"),
            ):
                destination = filesystem.stage_lora_adapter_files("http://adapter/source")
            self.assertTrue(destination.joinpath("READY").is_file())

    def test_failed_delta_is_terminal_before_full_transfer(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "one": _export(root, "one", b"a" * 8192),
                "two": _export(root, "two", b"a" * 8191 + b"b"),
            }
            build_adapter_delta(sources["one"], sources["two"], max_ratio=0.99)
            calls: list[str] = []
            delta_present_during_full = None

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                nonlocal delta_present_during_full
                calls.append(source_url)
                step, filename = source_url.rstrip("/").split("/")[-2:]
                source = sources[step] / filename
                if not source.is_file():
                    raise filesystem.AdapterArtifactNotFound(source)
                if filename == "adapter_model.safetensors" and step == "two":
                    delta_present_during_full = destination.parent.joinpath(
                        ADAPTER_DELTA_FILE
                    ).exists()
                shutil.copyfile(source, destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_DELTA": "1",
                    "PRIME_GLM_ADAPTER_CODEC": "gzip",
                    "PRIME_ADAPTER_FULL_WIRE_SIZE": "100000",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                filesystem.stage_lora_adapter_files("http://adapter/one")
                with patch.object(
                    filesystem,
                    "apply_adapter_delta",
                    side_effect=filesystem.AdapterDeltaError("corrupt delta"),
                ):
                    with self.assertRaisesRegex(filesystem.AdapterDeltaError, "corrupt delta"):
                        filesystem.stage_lora_adapter_files("http://adapter/two")
            self.assertFalse(delta_present_during_full)
            self.assertFalse(
                any(
                    url.endswith("/adapter_model.safetensors")
                    and "/two/" in url
                    for url in calls
                )
            )

    def test_failed_full_wire_comparison_is_terminal_before_full_transfer(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "one": _export(root, "one", b"a" * 8192),
                "two": _export(root, "two", b"a" * 8191 + b"b"),
            }
            build_adapter_delta(sources["one"], sources["two"], max_ratio=0.99)
            calls: list[str] = []

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                calls.append(source_url)
                step, filename = source_url.rstrip("/").split("/")[-2:]
                source = sources[step] / filename
                if not source.is_file():
                    raise filesystem.AdapterArtifactNotFound(source)
                shutil.copyfile(source, destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_DELTA": "1",
                    "PRIME_GLM_ADAPTER_CODEC": "gzip",
                    "PRIME_ADAPTER_FULL_WIRE_SIZE": "",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download), patch.object(
                filesystem,
                "_full_wire_size_estimate",
                side_effect=OSError("comparison HEAD failed"),
            ):
                filesystem.stage_lora_adapter_files("http://adapter/one")
                with self.assertRaisesRegex(OSError, "comparison HEAD failed"):
                    filesystem.stage_lora_adapter_files("http://adapter/two")
            self.assertFalse(
                any(
                    url.endswith("/adapter_model.safetensors")
                    and "/two/" in url
                    for url in calls
                )
            )

    def test_cache_hit_keeps_valid_adjacent_base(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {
                "one": _export(root, "one", b"a" * 8192),
                "two": _export(root, "two", b"a" * 8191 + b"b"),
            }
            build_adapter_delta(sources["one"], sources["two"], max_ratio=0.99)

            def fake_download(source_url, destination, create_only=False, **_kwargs):
                step, filename = source_url.rstrip("/").split("/")[-2:]
                source = sources[step] / filename
                if not source.is_file():
                    raise filesystem.AdapterArtifactNotFound(source)
                shutil.copyfile(source, destination)

            with patch.dict(
                os.environ,
                {
                    "PRIME_LORA_CACHE_DIR": str(root / "cache"),
                    "PRIME_ADAPTER_DELTA": "1",
                    "PRIME_GLM_ADAPTER_CODEC": "gzip",
                    "PRIME_ADAPTER_FULL_WIRE_SIZE": "100000",
                    "PRIME_LORA_CACHE_MAX_STALE_ENTRIES": "0",
                },
                clear=False,
            ), patch.object(filesystem, "download", side_effect=fake_download):
                filesystem.stage_lora_adapter_files("http://adapter/one")
                target = filesystem.stage_lora_adapter_files("http://adapter/two")
                base_sha256 = hashlib.sha256(
                    sources["one"].joinpath("adapter_model.safetensors").read_bytes()
                ).hexdigest()
                base = filesystem.adapter_cache_path(
                    "http://adapter/other-url", base_sha256
                )
                stale = target.parent / "stale"
                stale.mkdir()
                (stale / "READY").write_text("{}")
                filesystem.stage_lora_adapter_files("http://adapter/two")
            self.assertTrue(target.exists())
            self.assertTrue(base.exists())
            self.assertFalse(stale.exists())

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
                codec="gzip",
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
                base_available=True,
                base_sha256="a" * 64,
                target_sha256="b" * 64,
                target_size=1000,
                enabled=True,
                full_wire_size=1000,
                manifest_wire_size=20,
                codec="zstd",
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

    def test_optional_transport_failure_is_not_treated_as_missing(self) -> None:
        filesystem = _load_filesystem_module()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifact"
            with patch.object(
                filesystem,
                "download",
                side_effect=ConnectionRefusedError("tunnel is down"),
            ):
                with self.assertRaisesRegex(
                    OSError, "adapter artifact transfer failed"
                ):
                    filesystem._download_optional(
                        "http://127.0.0.1:39219/step/adapter_delta.json",
                        destination,
                    )
            self.assertFalse(destination.exists())

            with patch.object(
                filesystem,
                "download",
                side_effect=FileNotFoundError("local destination failed"),
            ):
                with self.assertRaisesRegex(
                    OSError, "adapter artifact transfer failed"
                ):
                    filesystem._download_optional(
                        "http://127.0.0.1:39219/step/adapter_delta.json",
                        destination,
                    )


if __name__ == "__main__":
    unittest.main()
