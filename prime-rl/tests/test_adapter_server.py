import functools
import hashlib
import http.server
import importlib.util
import json
import os
import tempfile
import threading
import unittest
import urllib.error
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
                for _ in range(2):
                    with urllib.request.urlopen(url, timeout=5) as response:
                        self.assertEqual(response.read(), chunk)
                self.assertEqual(calls.count("adapter_model.safetensors"), 1)

                original_identity = adapter_server._identity_from_published_directory
                replaced = False

                def validate_then_replace(directory, step_number, *, cache, weight_fd=None):
                    nonlocal replaced
                    result = original_identity(
                        directory,
                        step_number,
                        cache=cache,
                        weight_fd=weight_fd,
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
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(url, timeout=5)
                self.assertEqual(error.exception.code, 500)
                self.assertEqual(calls.count("adapter_model.safetensors"), 2)
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
