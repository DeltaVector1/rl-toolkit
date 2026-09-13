import functools
import hashlib
import http.server
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT.parent / "training_ops/runtime/china/adapter_server.py"


def load_server():
    spec = importlib.util.spec_from_file_location("adapter_server_test", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter_server = load_server()


class QuietAdapterHandler(adapter_server.RangeHandler):
    def log_message(self, *_args):
        return


class AdapterServerTest(unittest.TestCase):
    def test_virtual_bundle_reads_existing_chunks_with_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = root / "step_1"
            step.mkdir()
            (step / "STABLE").write_text("stable\n", encoding="ascii")
            chunks = [b"first-wire", b"second-wire"]
            chunk_records = []
            for index, payload in enumerate(chunks):
                name = f"adapter_model.safetensors.transport-delta-{index:06d}"
                (step / name).write_bytes(payload)
                chunk_records.append(
                    {
                        "file": name,
                        "offset": sum(len(item) for item in chunks[:index]),
                        "raw_size": len(payload),
                        "wire_size": len(payload),
                        "wire_sha256": hashlib.sha256(payload).hexdigest(),
                        "raw_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            bundle = b"".join(chunks)
            manifest = {
                "version": 1,
                "kind": "lora-adapter",
                "mode": "delta",
                "target_step": 1,
                "transport": "chunked-shuffled-xor-zstd",
                "bundle": {
                    "path": ".adapter-transport-bundle",
                    "size": len(bundle),
                    "sha256": hashlib.sha256(bundle).hexdigest(),
                },
                "chunks": chunk_records,
            }
            (step / "adapter_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(QuietAdapterHandler, directory=str(root)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = (
                f"http://127.0.0.1:{server.server_port}/step_1/"
                ".adapter-transport-bundle"
            )
            try:
                request = urllib.request.Request(url, headers={"Range": "bytes=3-14"})
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Range"], "bytes 3-14/21")
                    self.assertEqual(response.headers["ETag"], f'"{hashlib.sha256(bundle).hexdigest()}"')
                    self.assertEqual(response.read(), bundle[3:15])
                with urllib.request.urlopen(url, timeout=5) as response:
                    self.assertEqual(response.read(), bundle)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_chunk_requests_cache_identity_until_same_size_weight_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active"
            step = active / "step_0"
            step.mkdir(parents=True)
            assets = root / "assets" / "step_5"
            assets.mkdir(parents=True)
            (assets / "readme.txt").write_bytes(b"ordinary static content")
            config = b"{}\n"
            weights = b"weights"
            chunk = b"chunk"
            (step / "STABLE").write_text("stable\n", encoding="utf-8")
            (step / "adapter_config.json").write_bytes(config)
            (step / "adapter_model.safetensors").write_bytes(weights)
            weights_digest = hashlib.sha256(weights).hexdigest()
            (step / "adapter_model.safetensors.sha256").write_text(
                weights_digest + "\n", encoding="ascii"
            )
            (step / "adapter_model.safetensors.transport-predictive-00000.bin").write_bytes(chunk)
            (step / "adapter_manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "lora-adapter",
                        "target_step": 0,
                        "files": [
                            {
                                "name": "adapter_config.json",
                                "size": len(config),
                                "sha256": hashlib.sha256(config).hexdigest(),
                            },
                            {
                                "name": "adapter_model.safetensors",
                                "size": len(weights),
                                "sha256": weights_digest,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            calls = []
            original_sha256 = adapter_server._sha256

            def counted_sha256(path):
                calls.append(path.name)
                if path.name == "adapter_model.safetensors":
                    time.sleep(0.05)
                return original_sha256(path)

            adapter_server._identity_cache.clear()
            adapter_server._sha256 = counted_sha256
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(
                    QuietAdapterHandler,
                    directory=str(root),
                    upgrade_path=("active",),
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/active/step_0/adapter_model.safetensors.transport-predictive-00000.bin"
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/assets/step_5/readme.txt",
                    timeout=5,
                ) as response:
                    self.assertEqual(response.read(), b"ordinary static content")
                errors = []

                def fetch_chunk():
                    try:
                        with urllib.request.urlopen(url, timeout=5) as response:
                            self.assertEqual(response.read(), chunk)
                    except Exception as exc:
                        errors.append(exc)

                requests = [threading.Thread(target=fetch_chunk) for _ in range(16)]
                for request in requests:
                    request.start()
                for request in requests:
                    request.join(timeout=5)
                self.assertFalse(errors)
                self.assertTrue(all(not request.is_alive() for request in requests))
                self.assertEqual(calls.count("adapter_config.json"), 1)

                original_identity = adapter_server._identity_from_published_directory
                replaced = False

                def validate_then_replace(
                    directory,
                    step_number,
                    *,
                    cache,
                    weight_fd=None,
                    verify_weights=True,
                ):
                    nonlocal replaced
                    result = original_identity(
                        directory,
                        step_number,
                        cache=cache,
                        weight_fd=weight_fd,
                        verify_weights=verify_weights,
                    )
                    if weight_fd is not None and not replaced:
                        replacement = step / ".replacement"
                        replacement.write_bytes(b"corrupt")
                        os.replace(replacement, step / "adapter_model.safetensors")
                        replaced = True
                    return result

                adapter_server._identity_from_published_directory = validate_then_replace
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/active/step_0/adapter_model.safetensors",
                    timeout=5,
                ) as response:
                    self.assertEqual(response.read(), weights)
                self.assertTrue(replaced)
                adapter_server._identity_from_published_directory = original_identity
                with urllib.request.urlopen(url, timeout=5) as response:
                    self.assertEqual(response.read(), chunk)
                self.assertEqual(calls.count("adapter_config.json"), 3)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()
                if "original_identity" in locals():
                    adapter_server._identity_from_published_directory = original_identity
                adapter_server._sha256 = original_sha256
                adapter_server._identity_cache.clear()

    def test_old_stable_manifest_is_not_upgraded_after_newer_step_exists(self):
        run_path = "run_glm-1/broadcasts"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_run = root / run_path
            old_step = active_run / "step_5"
            new_step = active_run / "step_6"
            old_step.mkdir(parents=True)
            new_step.mkdir()
            (old_step / "STABLE").write_text("stable\n", encoding="utf-8")
            (new_step / "STABLE").write_text("stable\n", encoding="utf-8")
            config = b"{}\n"
            weights = b"weights"
            (old_step / "adapter_config.json").write_bytes(config)
            (old_step / "adapter_model.safetensors").write_bytes(weights)
            (old_step / "adapter_model.safetensors.sha256").write_text(
                hashlib.sha256(weights).hexdigest() + "\n", encoding="ascii"
            )
            manifest = {
                "version": 1,
                "kind": "lora-adapter",
                "mode": "delta",
                "transport": "chunked-shuffled-xor-zstd",
                "target_step": 5,
                "base_step": 4,
                "target_size": len(weights),
                "target_sha256": hashlib.sha256(weights).hexdigest(),
                "config_size": len(config),
                "config_sha256": hashlib.sha256(config).hexdigest(),
                "base_size": len(weights),
                "base_sha256": "a" * 64,
                "base_config_sha256": "b" * 64,
                "files": [
                    {"name": "adapter_config.json", "size": len(config), "sha256": hashlib.sha256(config).hexdigest()},
                    {"name": "adapter_model.safetensors", "size": len(weights), "sha256": hashlib.sha256(weights).hexdigest()},
                ],
            }
            expected = json.dumps(manifest).encode("utf-8")
            (old_step / "adapter_manifest.json").write_bytes(expected)

            handler = functools.partial(
                QuietAdapterHandler,
                directory=str(root),
                upgrade_path=tuple(run_path.split("/")),
                source_base_url="http://127.0.0.1:8100/" + run_path,
            )
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/{run_path}/step_5/adapter_manifest.json"
                with urllib.request.urlopen(url, timeout=5) as response:
                    self.assertEqual(response.read(), expected)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
