import functools
import hashlib
import http.client
import http.server
import importlib.util
import json
import os
import socket
import tempfile
import threading
import time
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
    def test_http11_range_framing_and_disconnect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"framed adapter chunk"
            served = root / "chunk.bin"
            served.write_bytes(payload)
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(QuietAdapterHandler, directory=str(root)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/chunk.bin",
                    headers={"Range": "bytes=2-8"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    self.assertEqual(response.version, 11)
                    self.assertEqual(response.headers["Content-Length"], "7")
                    self.assertEqual(response.headers["Content-Range"], "bytes 2-8/20")
                    self.assertEqual(response.read(), payload[2:9])

                connection = socket.create_connection(
                    ("127.0.0.1", server.server_port), timeout=5
                )
                try:
                    connection.sendall(
                        b"GET /chunk.bin HTTP/1.1\r\n"
                        b"Host: localhost\r\n"
                        b"Range: bytes=0-19\r\n\r\n"
                    )
                    connection.recv(128)
                finally:
                    connection.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_persistent_socket_head_then_directory_body_is_framed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "directory"
            directory.mkdir()
            (directory / "entry.txt").write_text("directory entry\n", encoding="utf-8")
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(QuietAdapterHandler, directory=str(root)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            try:
                connection.request("HEAD", "/directory/")
                head = connection.getresponse()
                declared = int(head.getheader("Content-Length"))
                self.assertEqual(head.read(), b"")
                connection.request("GET", "/directory/")
                response = connection.getresponse()
                body = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(len(body), declared)
            finally:
                connection.close()
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_persistent_socket_file_then_directory_body_is_framed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.txt").write_text("file body\n", encoding="utf-8")
            directory = root / "directory"
            directory.mkdir()
            (directory / "entry.txt").write_text("directory entry\n", encoding="utf-8")
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(QuietAdapterHandler, directory=str(root)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            try:
                connection.request("GET", "/file.txt")
                file_response = connection.getresponse()
                self.assertEqual(file_response.read(), b"file body\n")
                connection.request("GET", "/directory/")
                response = connection.getresponse()
                body = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(len(body), int(response.getheader("Content-Length")))
            finally:
                connection.close()
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

    def test_symlink_into_private_predictive_cache_is_not_served(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public = root / "public.txt"
            public.write_text("public\n", encoding="utf-8")
            private = root / adapter_server.cache_name
            private.mkdir()
            (private / "secret.txt").write_text("private\n", encoding="utf-8")
            (root / "safe-link").symlink_to(public.name)
            (root / "private-link").symlink_to(private.name, target_is_directory=True)
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0),
                functools.partial(QuietAdapterHandler, directory=str(root)),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/safe-link",
                    timeout=5,
                ) as response:
                    self.assertEqual(response.read(), b"public\n")
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_port}/private-link/secret.txt",
                        timeout=5,
                    )
                self.assertEqual(raised.exception.code, 500)
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
