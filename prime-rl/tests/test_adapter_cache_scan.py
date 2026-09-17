import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_linear_adapter_receiver import load_filesystem


class AdapterCacheScanTest(unittest.TestCase):
    def add_metadata(self, cache, count=300):
        paths = [cache / f".{index:064x}.readers.lock" for index in range(count)]
        paths.extend(cache / name for name in (
            ".adapter-stage-status.json", ".adapter-stage.lock", ".adapter-stage-status.lock"
        ))
        for path in paths:
            path.write_text("{}")
        return {path: path.stat().st_ino for path in paths}

    def assert_metadata_unchanged(self, metadata):
        for path, inode in metadata.items():
            self.assertEqual(path.stat().st_ino, inode)
            self.assertEqual(path.read_text(), "{}")

    def add_ready(self, filesystem, path, source_url):
        path.mkdir()
        files = {}
        for name, content in zip(filesystem._ADAPTER_FILES, (b"{}\n", b"adapter")):
            (path / name).write_bytes(content)
            files[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        manifest = {"version": 1, "source_url": source_url, "files": files}
        (path / "manifest.json").write_text(json.dumps(manifest))
        (path / "READY").write_text("ready\n")
        digest = files[filesystem._ADAPTER_FILES[1]]["sha256"]
        (path / filesystem._ADAPTER_DIGEST).write_text(digest + "\n")
        (path / "transport.json").write_text(json.dumps({
            "mode": "full", "source_url": source_url, "target_sha256": digest,
        }))
        return manifest

    def test_recovery_ignores_persistent_reader_metadata(self):
        filesystem = load_filesystem()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            metadata = self.add_metadata(cache)
            destination = cache / ("a" * 64)
            source_url = "http://source/step_21"
            for index in range(3):
                self.add_ready(filesystem, cache / f"{index:064x}", f"http://source/step_{index}")
            filesystem.reconcile_retired_adapter(destination, source_url)
            self.assertFalse(destination.exists())
            retired = destination.with_name(destination.name + ".retired-1")
            self.add_ready(filesystem, retired, source_url)
            invalid = destination.with_name(destination.name + ".retired-2")
            self.add_ready(filesystem, invalid, source_url)
            (invalid / filesystem._ADAPTER_FILES[1]).write_bytes(b"corrupt")
            filesystem.reconcile_retired_adapter(destination, source_url)
            self.assertEqual((destination / filesystem._ADAPTER_FILES[1]).read_bytes(), b"adapter")
            self.assertFalse(retired.exists())
            # Recovery holds a shared reader lease, so retired cleanup waits.
            self.assertTrue(invalid.exists())
            filesystem.reconcile_retired_adapter(destination, source_url)
            filesystem._cleanup_retired_adapters(cache, destination.name)
            self.assertFalse(invalid.exists())
            self.assert_metadata_unchanged(metadata)

    def test_cleanup_ignores_persistent_reader_metadata_and_preserves_leases(self):
        filesystem = load_filesystem()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {"PRIME_LORA_CACHE_DIR": temporary}
        ):
            cache = Path(temporary)
            metadata = self.add_metadata(cache)
            active, old, leased = (cache / (letter * 64) for letter in "abc")
            for path in (active, old, leased):
                self.add_ready(filesystem, path, "http://source/adapter")
            retired = active.with_name(active.name + ".retired-1")
            self.add_ready(filesystem, retired, "http://source/adapter")
            partial = old.with_name(old.name + ".partial")
            partial.mkdir()
            preserved = active.with_name(active.name + ".partial")
            preserved.mkdir()
            with filesystem.adapter_reader_lock(leased):
                leased_lock = filesystem._adapter_reader_lock(leased)
                leased_lock_inode = leased_lock.stat().st_ino
                filesystem._cleanup_adapter_caches(cache, active)
                filesystem._cleanup_obsolete_adapter_partials(cache, {active.name})
                self.assertEqual(leased_lock.stat().st_ino, leased_lock_inode)
            self.assertTrue(active.is_dir())
            self.assertTrue(leased.is_dir())
            self.assertTrue(preserved.is_dir())
            self.assertFalse(old.exists())
            self.assertFalse(retired.exists())
            self.assertFalse(partial.exists())
            self.assert_metadata_unchanged(metadata)

    def test_full_reuse_ignores_persistent_reader_metadata_and_checks_digest(self):
        filesystem = load_filesystem()
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                cache = Path(temporary)
                metadata = self.add_metadata(cache)
                candidate = cache / ("a" * 64)
                destination = cache / ("b" * 64)
                source_url = "http://new/step_0"
                manifest = self.add_ready(filesystem, candidate, "http://old/step_0")
                manifest["source_url"] = source_url
                if corrupt:
                    (candidate / filesystem._ADAPTER_FILES[1]).write_bytes(b"corrupt")
                reused = filesystem._reuse_full_adapter_cache(
                    source_url, destination, manifest,
                    {"mode": "full", "source_manifest": {"target_step": 0}},
                )
                self.assertEqual(reused, not corrupt)
                self.assertEqual(destination.exists(), not corrupt)
                if reused:
                    self.assertEqual((destination / "READY").read_text(), "ready\n")
                    self.assertEqual((destination / filesystem._ADAPTER_FILES[1]).read_bytes(), b"adapter")
                self.assert_metadata_unchanged(metadata)

    def scans(self, filesystem, cache, destination):
        return {
            "recovery": lambda: filesystem.reconcile_retired_adapter(destination),
            "retired": lambda: filesystem._cleanup_retired_adapters(cache, destination.name),
            "previous": lambda: filesystem._cleanup_previous_adapter_caches(cache, destination),
            "partial": lambda: filesystem._cleanup_obsolete_adapter_partials(cache, set()),
            "reuse": lambda: filesystem._reuse_full_adapter_cache(
                "http://source/step_0", destination, {"files": {}},
                {"mode": "full", "source_manifest": {"target_step": 0}},
            ),
        }

    def test_actionable_candidate_boundaries(self):
        filesystem = load_filesystem()
        limits = {"recovery": 64, "retired": 256, "previous": 64, "partial": 256, "reuse": 64}
        for kind, limit in limits.items():
            for count in (limit, limit + 1):
                with self.subTest(kind=kind, count=count), tempfile.TemporaryDirectory() as temporary:
                    cache = Path(temporary)
                    destination = cache / ("f" * 64)
                    metadata = self.add_metadata(cache)
                    for index in range(count):
                        name = f"{index:064x}"
                        if kind in {"recovery", "retired"}:
                            name = f"{destination.name}.retired-{index}"
                        elif kind == "partial":
                            name += ".partial"
                        candidate = cache / name
                        candidate.mkdir()
                        (candidate / "READY").write_text("ready\n")
                    scan = self.scans(filesystem, cache, destination)[kind]
                    if count > limit:
                        with self.assertRaisesRegex(OSError, "too many .*candidates") as raised:
                            scan()
                        self.assertFalse(filesystem._is_retryable_metadata_error(raised.exception))
                    else:
                        scan()
                    remaining = 0 if count == limit and kind in {"retired", "previous", "partial"} else count
                    self.assertEqual(sum(path.is_dir() for path in cache.iterdir()), remaining)
                    self.assert_metadata_unchanged(metadata)

    def test_outer_directory_entry_boundaries_for_every_scan(self):
        filesystem = load_filesystem()
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            destination = cache / ("f" * 64)
            with filesystem.adapter_reader_lock(destination):
                pass
            limit = filesystem._ADAPTER_DIRECTORY_ENTRY_LIMIT
            self.add_metadata(cache, count=limit - 4)  # Three fixed files and the recovery lock.
            for count in (limit, limit + 1):
                self.assertEqual(len(list(cache.iterdir())), count)
                for kind, scan in self.scans(filesystem, cache, destination).items():
                    with self.subTest(kind=kind, count=count):
                        if count > limit:
                            with self.assertRaisesRegex(OSError, "too many .*entries") as raised:
                                scan()
                            self.assertFalse(filesystem._is_retryable_metadata_error(raised.exception))
                        else:
                            scan()
                        self.assertEqual(len(list(cache.iterdir())), count)
                if count == limit:
                    (cache / f".{limit:064x}.readers.lock").touch()

    def test_cleanup_and_reuse_symlink_failures_remain_terminal(self):
        filesystem = load_filesystem()
        for kind in ("retired", "previous", "partial", "reuse"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                cache = Path(temporary) / "cache"
                cache.mkdir()
                self.add_metadata(cache)
                destination = cache / ("f" * 64)
                external = Path(temporary) / "external"
                external.mkdir()
                sentinel = external / "keep"
                sentinel.write_text("keep")
                name = "a" * 64
                if kind == "retired":
                    name = destination.name + ".retired-1"
                elif kind == "partial":
                    name += ".partial"
                (cache / name).symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(OSError, "symbolic link") as raised:
                    self.scans(filesystem, cache, destination)[kind]()
                self.assertFalse(filesystem._is_retryable_metadata_error(raised.exception))
                self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
