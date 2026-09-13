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

                for range_value, expected in (("bytes=-7", payload[-7:]), ("bytes=9-", payload[9:])):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/chunk.bin",
                        headers={"Range": range_value},
                    )
                    with urllib.request.urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 206)
                        self.assertEqual(response.read(), expected)

                for range_value in ("bytes=20-", "bytes=-0", "bytes=4-3", "bytes=0-1,2-3"):
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/chunk.bin",
                        headers={"Range": range_value},
                    )
                    with self.assertRaises(urllib.error.HTTPError) as raised:
                        urllib.request.urlopen(request, timeout=5)
                    self.assertEqual(raised.exception.code, 416)
                    self.assertEqual(raised.exception.headers["Content-Range"], "bytes */20")

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

    def test_send_head_closes_source_when_seek_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            served = root / "file.bin"
            served.write_bytes(b"payload")
            opened = []
            original_fdopen = adapter_server.os.fdopen

            class SeekFailure:
                def __init__(self, wrapped):
                    self.wrapped = wrapped
                    self.closed = False

                def fileno(self):
                    return self.wrapped.fileno()

                def seek(self, *_args):
                    raise OSError("seek failed")

                def close(self):
                    self.closed = True
                    self.wrapped.close()

            def fdopen(fd, mode):
                source = SeekFailure(original_fdopen(fd, mode))
                opened.append(source)
                return source

            handler = QuietAdapterHandler.__new__(QuietAdapterHandler)
            handler.path = "/file.bin"
            handler.directory = str(root)
            handler.upgrade_path = None
            handler.headers = {}
            handler.command = "GET"
            handler.translate_path = lambda _path: str(served)
            handler._maybe_upgrade = lambda: served
            handler.send_response = lambda *_args: None
            handler.send_header = lambda *_args: None
            handler.end_headers = lambda: None
            before = len(os.listdir("/proc/self/fd"))
            adapter_server.os.fdopen = fdopen
            try:
                with self.assertRaises(OSError):
                    handler.send_head()
                self.assertTrue(opened[0].closed)
            finally:
                adapter_server.os.fdopen = original_fdopen
                if opened and not opened[0].closed:
                    opened[0].close()
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_copyfile_closes_source_when_read_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source"
            source_path.write_bytes(b"payload")

            class ReadFailure:
                def __init__(self):
                    self.wrapped = source_path.open("rb")
                    self.closed = False

                def read(self, _size):
                    raise OSError("read failed")

                def close(self):
                    self.closed = True
                    self.wrapped.close()

            before = len(os.listdir("/proc/self/fd"))
            source = ReadFailure()
            handler = QuietAdapterHandler.__new__(QuietAdapterHandler)
            handler.remaining = 7
            handler.close_connection = False
            try:
                with self.assertRaises(OSError):
                    handler.copyfile(source, object())
                self.assertTrue(source.closed)
                self.assertTrue(handler.close_connection)
            finally:
                if not source.closed:
                    source.close()
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_copyfile_closes_source_when_write_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "source"
            source_path.write_bytes(b"payload")

            class Source:
                def __init__(self):
                    self.wrapped = source_path.open("rb")
                    self.closed = False

                def read(self, _size):
                    return b"payload"

                def close(self):
                    self.closed = True
                    self.wrapped.close()

            class WriteFailure:
                def write(self, _chunk):
                    raise OSError("write failed")

            before = len(os.listdir("/proc/self/fd"))
            source = Source()
            handler = QuietAdapterHandler.__new__(QuietAdapterHandler)
            handler.remaining = 7
            handler.close_connection = False
            try:
                handler.copyfile(source, WriteFailure())
                self.assertTrue(source.closed)
                self.assertTrue(handler.close_connection)
            finally:
                if not source.closed:
                    source.close()
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)

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
                        "chunks": [
                            {
                                "file": "adapter_model.safetensors.transport-predictive-00000.bin",
                                "wire_size": len(chunk),
                                "wire_sha256": hashlib.sha256(chunk).hexdigest(),
                            }
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

                chunk_path = step / "adapter_model.safetensors.transport-predictive-00000.bin"
                original_chunk = chunk_path.read_bytes()
                replaced_chunk = step / ".replaced-chunk"
                replaced_chunk.write_bytes(b"X" * len(chunk))
                os.replace(replaced_chunk, chunk_path)
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(url, timeout=5)
                self.assertEqual(raised.exception.code, 500)
                restored_chunk = step / ".restored-chunk"
                restored_chunk.write_bytes(original_chunk)
                os.replace(restored_chunk, chunk_path)

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
                stale = step / "adapter_model.safetensors.transport-predictive-stale.bin"
                stale.write_bytes(chunk)
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_port}/active/step_0/{stale.name}",
                        timeout=5,
                    )
                self.assertEqual(raised.exception.code, 404)
                self.assertEqual(calls.count("adapter_config.json"), 3)
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()
                if "original_identity" in locals():
                    adapter_server._identity_from_published_directory = original_identity
                adapter_server._sha256 = original_sha256
                adapter_server._identity_cache.clear()

    def test_predictive_install_rejects_false_chunk_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "step_2"
            build = root / "build"
            target.mkdir()
            build.mkdir()
            (target / adapter_server.adapter_manifest_name).write_text("{}", encoding="utf-8")
            (target / adapter_server.adapter_digest_name).write_text("0" * 64, encoding="ascii")
            chunk_name = f"{adapter_server.adapter_weights_name}.transport-predictive-00000.bin"
            chunk = b"chunk"
            (build / chunk_name).write_bytes(chunk)
            (build / adapter_server.adapter_manifest_name).write_text("{}", encoding="utf-8")
            (build / adapter_server.adapter_digest_name).write_text("0" * 64, encoding="ascii")
            manifest = {
                "version": 2,
                "mode": "delta",
                "target_step": 2,
                "base_step": 1,
                "base2_step": 0,
                "chunks": [
                    {
                        "file": chunk_name,
                        "wire_size": len(chunk),
                        "wire_sha256": "0" * 64,
                    }
                ],
            }
            with self.assertRaises(adapter_server.UpgradeError):
                adapter_server._install_predictive_manifest(target, build, manifest, 2)

    def test_upgrade_scavenges_exact_leftovers_and_keeps_private_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active"
            active.mkdir()
            cache = active / adapter_server.cache_name
            cache.mkdir(mode=0o700)
            (active / "step_4").mkdir()
            (active / ".predictive-build-11-22").mkdir()
            (cache / ".pending-11-22").mkdir()
            (active / ".predictive-build-not-a-leftover").mkdir()
            adapter_server._scavenge_upgrade_leftovers(active)
            self.assertFalse((active / ".predictive-build-11-22").exists())
            self.assertFalse((cache / ".pending-11-22").exists())
            self.assertTrue((active / "step_4").exists())
            self.assertTrue((active / ".predictive-build-not-a-leftover").exists())
            source = root / "source"
            source.mkdir()
            (source / adapter_server.adapter_config_name).write_bytes(b"{}")
            (source / adapter_server.adapter_weights_name).write_bytes(b"weights")
            identity = {
                "step": 4,
                "config_size": 2,
                "config_sha256": hashlib.sha256(b"{}").hexdigest(),
                "weights_size": 7,
                "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
            }
            destination = cache / "step_4"
            adapter_server._link_cache_entry(source, identity, destination)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            for name in (
                adapter_server.adapter_config_name,
                adapter_server.adapter_weights_name,
                adapter_server.adapter_digest_name,
                ".identity.json",
            ):
                self.assertEqual((destination / name).stat().st_mode & 0o777, 0o600)

    def test_private_cache_links_without_reading_adapter_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "cache" / "step_1"
            source.mkdir()
            (root / "cache").mkdir(mode=0o700)
            config = source / adapter_server.adapter_config_name
            weights = source / adapter_server.adapter_weights_name
            config.write_bytes(b"{}")
            weights.write_bytes(b"weights")
            identity = {
                "step": 1,
                "config_size": 2,
                "config_sha256": hashlib.sha256(b"{}").hexdigest(),
                "weights_size": 7,
                "weights_sha256": hashlib.sha256(b"weights").hexdigest(),
            }
            original_read = adapter_server.os.read
            adapter_server.os.read = lambda *_args: (_ for _ in ()).throw(
                AssertionError("private cache staging must not read payload bytes")
            )
            try:
                adapter_server._link_cache_entry(source, identity, destination)
            finally:
                adapter_server.os.read = original_read
            self.assertEqual(config.stat().st_ino, (destination / config.name).stat().st_ino)
            self.assertEqual(weights.stat().st_ino, (destination / weights.name).stat().st_ino)
            self.assertEqual(weights.stat().st_mode & 0o777, 0o600)

    def test_atomic_replace_rejects_source_identity_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "build" / "chunk"
            destination = root / "target" / "chunk"
            source.parent.mkdir()
            destination.parent.mkdir()
            source.write_bytes(b"before")
            destination.write_bytes(b"old-target")
            manifest = destination.parent / adapter_server.adapter_manifest_name
            manifest.write_text("old-manifest\n", encoding="utf-8")
            old_manifest = manifest.read_bytes()
            identity = adapter_server._stat_identity(source)
            digest = hashlib.sha256(b"before").hexdigest()
            original_link = adapter_server.os.link
            link_calls = 0

            def link_after_source_swap(*args, **kwargs):
                nonlocal link_calls
                link_calls += 1
                result = original_link(*args, **kwargs)
                if link_calls == 2:
                    source.write_bytes(b"raced!")
                return result

            adapter_server.os.link = link_after_source_swap
            try:
                with self.assertRaises(adapter_server.UpgradeError):
                    adapter_server._replace_from_dirs(source, destination, identity, digest)
            finally:
                adapter_server.os.link = original_link
            self.assertEqual(destination.read_bytes(), b"old-target")
            self.assertEqual(manifest.read_bytes(), old_manifest)

    def test_directory_candidate_race_preserves_old_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pending" / "step_1"
            destination = root / "cache" / "step_1"
            source.mkdir(parents=True)
            destination.mkdir(parents=True)
            (source / "value").write_bytes(b"new-tree")
            (destination / "value").write_bytes(b"old-tree")
            manifest = destination / adapter_server.adapter_manifest_name
            manifest.write_text("old-manifest\n", encoding="utf-8")
            old_manifest = manifest.read_bytes()
            identity = adapter_server._directory_identity(source)
            calls = 0

            def validate(candidate):
                nonlocal calls
                calls += 1
                if (candidate / "value").read_bytes() != b"new-tree":
                    raise OSError("candidate changed")
                if calls == 1:
                    (candidate / "value").write_bytes(b"raced-tree")

            with self.assertRaises(adapter_server.UpgradeError):
                adapter_server._replace_from_dirs(
                    source,
                    destination,
                    identity,
                    directory_validator=validate,
                )
            self.assertEqual((destination / "value").read_bytes(), b"old-tree")
            self.assertEqual(manifest.read_bytes(), old_manifest)

    def test_directory_candidate_rename_ignores_ctime_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pending" / "step_1"
            destination = root / "cache" / "step_1"
            source.mkdir(parents=True, mode=0o700)
            destination.parent.mkdir()
            identity = adapter_server._directory_identity(source)
            changed = False

            def validate(candidate):
                nonlocal changed
                if not changed:
                    os.chmod(candidate, 0o700)
                    changed = True

            adapter_server._replace_from_dirs(
                source,
                destination,
                identity,
                directory_validator=validate,
            )
            self.assertTrue(destination.is_dir())
            self.assertEqual(destination.stat().st_ino, identity[1])

    def test_regular_file_rename_ignores_ctime_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pending" / "chunk"
            destination = root / "cache" / "chunk"
            source.parent.mkdir(parents=True)
            destination.parent.mkdir()
            source.write_bytes(b"new-data")
            identity = adapter_server._stat_identity(source)
            digest = hashlib.sha256(b"new-data").hexdigest()
            original_replace = adapter_server.os.replace

            def replace_and_change_ctime(*args, **kwargs):
                result = original_replace(*args, **kwargs)
                link = destination.with_name(".ctime-link")
                os.link(destination, link)
                link.unlink()
                return result

            adapter_server.os.replace = replace_and_change_ctime
            try:
                adapter_server._replace_from_dirs(
                    source,
                    destination,
                    identity,
                    digest,
                )
            finally:
                adapter_server.os.replace = original_replace
            self.assertEqual(destination.read_bytes(), b"new-data")
            self.assertEqual(destination.stat().st_ino, identity[1])

    def test_regular_file_rename_rejects_published_content_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pending" / "chunk"
            destination = root / "cache" / "chunk"
            source.parent.mkdir(parents=True)
            destination.parent.mkdir()
            source.write_bytes(b"new-data")
            destination.write_bytes(b"old-data")
            identity = adapter_server._stat_identity(source)
            digest = hashlib.sha256(b"new-data").hexdigest()
            original_replace = adapter_server.os.replace
            replacements = 0

            def replace_then_mutate(*args, **kwargs):
                nonlocal replacements
                result = original_replace(*args, **kwargs)
                replacements += 1
                if replacements == 1:
                    destination.write_bytes(b"raced!!!")
                return result

            adapter_server.os.replace = replace_then_mutate
            try:
                with self.assertRaises(adapter_server.UpgradeError):
                    adapter_server._replace_from_dirs(
                        source,
                        destination,
                        identity,
                        digest,
                    )
            finally:
                adapter_server.os.replace = original_replace
            self.assertEqual(destination.read_bytes(), b"old-data")

    def test_existing_v2_manifest_with_missing_chunk_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "step_2"
            step.mkdir()
            config = b"{}\n"
            weights = b"weights"
            (step / adapter_server.stable_name).write_text("stable\n", encoding="utf-8")
            (step / adapter_server.adapter_config_name).write_bytes(config)
            (step / adapter_server.adapter_weights_name).write_bytes(weights)
            digest = hashlib.sha256(weights).hexdigest()
            (step / adapter_server.adapter_digest_name).write_text(digest + "\n", encoding="ascii")
            (step / adapter_server.adapter_manifest_name).write_text(
                json.dumps(
                    {
                        "version": 2,
                        "kind": "lora-adapter",
                        "target_step": 2,
                        "files": [
                            {
                                "name": adapter_server.adapter_config_name,
                                "size": len(config),
                                "sha256": hashlib.sha256(config).hexdigest(),
                            },
                            {
                                "name": adapter_server.adapter_weights_name,
                                "size": len(weights),
                                "sha256": digest,
                            },
                        ],
                        "chunks": [
                            {
                                "file": f"{adapter_server.adapter_weights_name}.transport-predictive-00000.bin",
                                "wire_size": 5,
                                "wire_sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(adapter_server.UpgradeError):
                adapter_server._identity_from_published_directory(step, 2, cache=False)

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
