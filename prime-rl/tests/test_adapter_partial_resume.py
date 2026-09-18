import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_linear_adapter_receiver import load_filesystem


def fixture(filesystem, build_seconds=1):
    digest = lambda data: hashlib.sha256(data).hexdigest()
    target, config, wire = b"target!!", b"{}\n", b"wire"
    value = {
        "version": 2, "kind": "lora-adapter", "mode": "delta",
        "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
        "target_step": 2, "base_step": 1, "base2_step": 0,
        "target_url": "http://source/step_2", "base_url": "http://source/step_1",
        "base2_url": "http://source/step_0", "target_size": len(target),
        "target_sha256": digest(target), "full_size": len(target),
        "full_sha256": digest(target), "config_size": len(config),
        "config_sha256": digest(config), "base_size": 8, "base_sha256": digest(b"base-one"),
        "base_config_size": len(config), "base_config_sha256": digest(config),
        "base2_size": 8, "base2_sha256": digest(b"base-two"),
        "base2_config_size": len(config), "base2_config_sha256": digest(config),
        "data_offset": 2, "chunk_alignment": 2,
        "chunk_coverage": {"offset": 0, "size": len(target)},
        "predictor": "bf16-float32-second-order-rne",
        "value_encoding": "sparse-zigzag-leb128", "mask_encoding": "zstd-bit-mask-little-endian",
        "build_seconds": build_seconds, "stage_id": str(build_seconds),
        "chunks": [{"file": "chunk-0", "offset": 0, "data_offset": 2,
                    "raw_size": len(target), "wire_size": len(wire),
                    "wire_sha256": digest(wire), "raw_sha256": digest(b""), "codec": "zstd"}],
        "files": [{"name": "adapter_config.json", "size": len(config), "sha256": digest(config)},
                  {"name": "adapter_model.safetensors", "size": len(target), "sha256": digest(target)}],
    }
    with mock.patch.object(filesystem, "_read_source_text", return_value=json.dumps(value)):
        transport = filesystem._read_chunked_linear_manifest(value["target_url"])
    manifest = {"version": 1, "source_url": value["target_url"],
                "files": {item["name"]: {"size": item["size"], "sha256": item["sha256"]}
                          for item in value["files"]}}
    return transport, manifest, target, config, wire


def run_receiver(cache, fail=False):
    """Use a fresh real receiver; replace only network and BF16 test payload decoding."""
    os.environ["PRIME_LORA_CACHE_DIR"] = cache
    filesystem = load_filesystem()
    transport, manifest, target, config, wire = fixture(filesystem, 1 if fail else 99)
    source = manifest["source_url"]
    bases = {}
    for step, model in ((1, b"base-one"), (0, b"base-two")):
        root = Path(cache) / f"fixture-base-{step}"
        root.mkdir(exist_ok=True)
        (root / "model").write_bytes(model)
        (root / "config").write_bytes(config)
        bases[f"http://source/step_{step}"] = (root / "model", root / "config")
    downloaded, decoded = [], []

    def download(_urls, destination, **kwargs):
        downloaded.append(destination.name)
        destination.write_bytes(config if "config" in destination.name else wire)
        if kwargs.get("progress"):
            kwargs["progress"](0, destination.stat().st_size)

    class Pool:
        def __init__(self, *_args, **_kwargs):
            pass

        def urls_for_lane(self, *_args):
            return ["http://fixture"]

        def download(self, urls, lane, destination, parent_fd, **kwargs):
            download(urls, destination, **kwargs)

        def close(self):
            pass

    def decode(*_args, **_kwargs):
        decoded.append(True)
        return target, hashlib.sha256(b"").hexdigest()

    original_publish = filesystem._publish_relay_manifest

    def publish(*args, **kwargs):
        if fail and (args[0] / "transport-chunks/chunk-0.done").is_file():
            raise OSError("terminal local failure after verified chunks")
        return original_publish(*args, **kwargs)

    with mock.patch.multiple(
        filesystem,
        _read_delta_manifest_with_visibility_wait=lambda *_a, **_k: transport,
        _source_manifest=lambda *_a, **_k: manifest,
        _verified_adapter_base=lambda url, *_a, **_k: bases[url],
        _adapter_source_urls=lambda *_a: ["http://fixture"],
        _configured_download_streams=lambda: 1,
        _PersistentAdapterLanePool=Pool, _download_persistent_file=download,
        _decode_linear_transport_chunk=decode, _publish_relay_manifest=publish,
    ):
        try:
            filesystem.stage_lora_adapter_files(source, require_delta=True)
        except OSError as exc:
            if not fail or "terminal local failure" not in str(exc):
                raise
        else:
            assert not fail
    destination = filesystem.adapter_cache_path(source)
    partial = destination.with_name(destination.name + ".partial")
    return {"downloads": downloaded, "decoded": len(decoded),
            "status": filesystem.adapter_stage_status()["status"],
            "ready": (destination / "READY").is_file(), "partial": partial.exists(),
            "proof": (partial / "transport-chunks/chunk-0.done").is_file()}


class AdapterPartialResumeTest(unittest.TestCase):
    def child(self, cache, fail=False):
        result = subprocess.run(
            [sys.executable, __file__, cache, str(int(fail))],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_fresh_process_terminal_failure_keeps_verified_chunks(self):
        with tempfile.TemporaryDirectory() as cache:
            first = self.child(cache, True)
            self.assertEqual(first["status"], "failed")
            self.assertTrue(first["proof"])
            second = self.child(cache)
            self.assertEqual(second["status"], "ready")
            self.assertTrue(second["ready"])
            self.assertFalse(second["partial"])
            self.assertFalse(any("chunk-0" in name for name in second["downloads"]))
            self.assertEqual(second["decoded"], 0)

    def test_corrupt_chunk_proof_and_output_are_revalidated(self):
        for name, data, downloads, decodes in (
            ("transport-chunks/chunk-0", b"bad!", True, 1),
            ("transport-chunks/chunk-0.done", b"{}", False, 1),
            ("adapter_model.safetensors", b"damaged!", False, 1),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as cache:
                self.child(cache, True)
                partial = next(Path(cache).glob("*.partial"))
                (partial / name).write_bytes(data)
                result = self.child(cache)
                self.assertTrue(result["ready"])
                self.assertEqual(any("chunk-0" in value for value in result["downloads"]), downloads)
                self.assertEqual(result["decoded"], decodes)

    def test_fresh_process_legacy_identity_ignores_changed_build_metadata(self):
        with tempfile.TemporaryDirectory() as cache:
            self.assertTrue(self.child(cache, True)["proof"])
            partial = next(Path(cache).glob("*.partial"))
            legacy, *_ = fixture(load_filesystem(), 1)
            (partial / "transport-manifest.json").write_text(json.dumps(legacy))
            result = self.child(cache)
            self.assertTrue(result["ready"])
            self.assertFalse(result["partial"])
            self.assertFalse(any("chunk-0" in value for value in result["downloads"]))
            self.assertEqual(result["decoded"], 0)
            published = next(Path(cache).glob("*/transport-manifest.json"))
            identity = json.loads(published.read_text())
            self.assertEqual(identity["resume_identity_version"], 1)
            self.assertNotIn("source_manifest", identity)

    def test_missing_identity_discards_untrusted_partial_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as cache:
            self.child(cache, True)
            partial = next(Path(cache).glob("*.partial"))
            (partial / "transport-manifest.json").unlink()
            result = self.child(cache)
            self.assertTrue(result["ready"])
            self.assertTrue(any("chunk-0" in value for value in result["downloads"]))
            self.assertEqual(result["decoded"], 1)

    def test_terminal_preidentity_failure_discards_untrusted_partial(self):
        filesystem = load_filesystem()
        source = "http://source/step_2"
        with tempfile.TemporaryDirectory() as cache, mock.patch.dict(
            os.environ, {"PRIME_LORA_CACHE_DIR": cache},
        ):
            destination = filesystem.adapter_cache_path(source)
            partial = destination.with_name(destination.name + ".partial")
            partial.mkdir()
            (partial / "untrusted").write_bytes(b"not verified")
            with mock.patch.object(
                filesystem, "_read_delta_manifest_with_visibility_wait",
                side_effect=OSError("terminal preidentity failure"),
            ), self.assertRaisesRegex(OSError, "terminal preidentity failure"):
                filesystem.stage_lora_adapter_files(source, require_delta=True)
            self.assertFalse(partial.exists())
            self.assertEqual(filesystem.adapter_stage_status()["status"], "failed")

    def test_identity_covers_all_content_fields_and_ignores_metadata(self):
        filesystem = load_filesystem()
        transport, manifest, *_ = fixture(filesystem)
        source = manifest["source_url"]
        identity = filesystem._adapter_resume_identity(source, transport)
        changed, *_ = fixture(filesystem, 99)
        self.assertEqual(identity, filesystem._adapter_resume_identity(source, changed))
        parsed = json.loads(identity)
        for field in parsed:
            if field in {"resume_identity_version", "chunks", "source_url"}:
                continue
            with self.subTest(field=field):
                changed = copy.deepcopy(transport)
                owner = changed if field in changed else changed["source_manifest"]
                owner[field] = "changed"
                self.assertNotEqual(identity, filesystem._adapter_resume_identity(source, changed))
        for field in parsed["chunks"][0]:
            changed = copy.deepcopy(transport)
            changed["chunks"][0][field] = "changed"
            self.assertNotEqual(identity, filesystem._adapter_resume_identity(source, changed), field)
        self.assertNotEqual(identity, filesystem._adapter_resume_identity(source + "?changed", transport))

    def test_old_identity_migrates_only_with_exact_content(self):
        filesystem = load_filesystem()
        old, manifest, *_ = fixture(filesystem)
        new, *_ = fixture(filesystem, 99)
        source = manifest["source_url"]
        malformed = copy.deepcopy(old)
        malformed["source_manifest"]["target_url"] = source + "?changed"
        self.assertIsNone(filesystem._legacy_adapter_resume_identity(json.dumps(malformed), source))
        for value in ("not json", "{}", json.dumps({"source_manifest": {}})):
            self.assertIsNone(filesystem._legacy_adapter_resume_identity(value, source))
        for matches in (True, False):
            with self.subTest(matches=matches), tempfile.TemporaryDirectory() as cache:
                partial = Path(cache) / "partial"
                partial.mkdir()
                previous = copy.deepcopy(old)
                if not matches:
                    previous["target_sha256"] = "a" * 64
                (partial / "transport-manifest.json").write_text(json.dumps(previous))
                (partial / "verified-chunk").write_bytes(b"chunk")
                filesystem._reset_staging_directory_identity(
                    partial, "transport-manifest.json", filesystem._adapter_resume_identity(source, new),
                    "adapter partial directory", resume_source_url=source,
                )
                self.assertEqual(partial.exists(), matches)

    def test_missing_or_symlink_identity_is_not_preserved(self):
        filesystem = load_filesystem()
        transport, manifest, *_ = fixture(filesystem)
        source = manifest["source_url"]
        with tempfile.TemporaryDirectory() as cache:
            destination = Path(cache) / "adapter"
            partial = Path(cache) / "adapter.partial"
            partial.mkdir()
            self.assertFalse(filesystem._has_matching_adapter_resume_identity(destination, source, transport))
            external = Path(cache) / "external"
            external.write_text(filesystem._adapter_resume_identity(source, transport))
            (partial / "transport-manifest.json").symlink_to(external)
            self.assertFalse(filesystem._has_matching_adapter_resume_identity(destination, source, transport))
            self.assertFalse(filesystem._has_matching_adapter_resume_identity(destination, source, None))


if __name__ == "__main__":
    print(json.dumps(run_receiver(sys.argv[1], bool(int(sys.argv[2])))))
