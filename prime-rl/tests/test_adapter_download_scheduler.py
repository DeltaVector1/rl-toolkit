import errno
import hashlib
import http.server
import inspect
import importlib.util
import io
import json
import os
import socketserver
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_filesystem_module():
    modules = {
        "torch": types.ModuleType("torch"),
        "torch.nn": types.ModuleType("torch.nn"),
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": types.ModuleType(
            "vllm.model_executor.model_loader"
        ),
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.utils": types.ModuleType("prime_rl.utils"),
        "prime_rl.utils.checkpoint_delta": types.ModuleType(
            "prime_rl.utils.checkpoint_delta"
        ),
    }
    modules["torch.nn"].Module = object
    modules["vllm.model_executor.model_loader"].DefaultModelLoader = object
    modules["vllm.model_executor.model_loader"].get_model_loader = lambda _config: None
    checkpoint_delta = modules["prime_rl.utils.checkpoint_delta"]
    checkpoint_delta.DELTA_MANIFEST = "checkpoint_manifest.json"
    checkpoint_delta.apply_xor_delta_file = lambda *args, **kwargs: None
    checkpoint_delta.sha256_file = hashlib.sha256
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "filesystem_scheduler_test",
            ROOT / "src/prime_rl/inference/vllm/worker/filesystem.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class DownloadSchedulerTest(unittest.TestCase):
    def _persistent_server(self, filesystem, payload, action=None):
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            connections = 0
            active_connections = 0
            maximum_active_connections = 0
            requests = 0
            paths = []
            ranges = []

            def setup(self):
                super().setup()
                type(self).connections += 1
                type(self).active_connections += 1
                type(self).maximum_active_connections = max(
                    type(self).maximum_active_connections,
                    type(self).active_connections,
                )

            def finish(self):
                type(self).active_connections -= 1
                super().finish()

            def do_GET(self):
                type(self).requests += 1
                type(self).paths.append(self.path)
                range_header = self.headers.get("Range")
                type(self).ranges.append(range_header)
                if range_header is None:
                    start, end = 0, len(payload) - 1
                else:
                    start_text, end_text = range_header.removeprefix(
                        "bytes="
                    ).split("-", 1)
                    start = int(start_text)
                    end = len(payload) - 1 if not end_text else int(end_text)
                if action is not None and action(self, type(self).requests):
                    return
                body = payload[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header(
                    "Content-Range", f"bytes {start}-{end}/{len(payload)}"
                )
                self.send_header("ETag", f'"{hashlib.sha256(payload).hexdigest()}"')
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, Handler

    def _persistent_download(
        self,
        filesystem,
        server,
        root,
        name,
        payload,
        cancel_event=None,
        per_request_timeout_seconds=2,
    ):
        chunks = root / "chunks"
        chunks.mkdir(exist_ok=True)
        parent_fd = filesystem._open_directory_nofollow(
            chunks,
            create=False,
            label="test chunks",
        )
        pool = filesystem._PersistentAdapterLanePool(1)
        try:
            pool.download(
                [f"http://127.0.0.1:{server.server_address[1]}/{name}"],
                0,
                chunks / name,
                parent_fd,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                per_request_timeout_seconds=per_request_timeout_seconds,
                cancel_event=cancel_event,
            )
        finally:
            pool.close()
            os.close(parent_fd)

    def _legacy_full_download(
        self, filesystem, server, root, name, payload, *, stream_cap=1,
        source_urls=None, per_request_timeout_seconds=1, cancel_event=None,
        deadline_seconds=10,
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        parent_fd = filesystem._open_directory_nofollow(
            destination.parent,
            create=False,
            label="test legacy download parent",
        )
        url = f"http://127.0.0.1:{server.server_address[1]}/{name}"
        old_streams = filesystem._configured_download_streams
        filesystem._configured_download_streams = lambda: stream_cap
        try:
            filesystem._download_with_parent(
                url,
                destination,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
                source_urls=source_urls or [url],
                stream_cap=stream_cap,
                parallel_ranges=True,
                deadline=time.monotonic() + deadline_seconds,
                per_request_timeout_seconds=per_request_timeout_seconds,
                cancel_event=cancel_event,
                typed_transport=True,
                parent_fd=parent_fd,
            )
        finally:
            filesystem._configured_download_streams = old_streams
            os.close(parent_fd)
        return destination

    def _inject_enospc(self, filesystem):
        old_fdopen = filesystem.os.fdopen

        class FailingOutput:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.stream.close()

            def write(self, _value):
                raise OSError(errno.ENOSPC, "No space left on device")

        def failing_fdopen(fd, mode, *args, **kwargs):
            stream = old_fdopen(fd, mode, *args, **kwargs)
            if mode in {"ab", "wb"}:
                return FailingOutput(stream)
            return stream

        filesystem.os.fdopen = failing_fdopen
        return old_fdopen

    def test_persistent_chunks_reuse_one_connection(self):
        filesystem = load_filesystem_module()
        payload = b"persistent-range"
        server, thread, handler = self._persistent_server(filesystem, payload)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                try:
                    for name in ("one", "two"):
                        pool.download(
                            [f"http://127.0.0.1:{server.server_address[1]}/{name}"],
                            0,
                            chunks / name,
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            per_request_timeout_seconds=2,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual(handler.connections, 1)
                self.assertEqual(handler.requests, 2)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_pool_width_binds_primaries_and_ordered_failover(self):
        filesystem = load_filesystem_module()
        payload = b"fixed-primary"
        failed = False

        def fail_primary_once(request, _number):
            nonlocal failed
            if request.path == "/lane0/chunk" and not failed:
                failed = True
                request.close_connection = True
                return True
            return False

        server, thread, handler = self._persistent_server(
            filesystem, payload, fail_primary_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                urls = [
                    f"http://127.0.0.1:{server.server_address[1]}/lane{index}"
                    for index in range(8)
                ]
                pool = filesystem._PersistentAdapterLanePool(urls, width=3)
                try:
                    self.assertEqual(pool.slot_count, 3)
                    for lane, name in ((0, "chunk"), (0, "next"), (1, "one"), (2, "two")):
                        pool.download(
                            pool.urls_for_lane(lane, name),
                            lane,
                            chunks / name,
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            per_request_timeout_seconds=1,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual(
                    handler.paths[:3],
                    ["/lane0/chunk", "/lane1/chunk", "/lane0/next"],
                )
                self.assertLessEqual(handler.maximum_active_connections, 3)
                self.assertFalse(any(path.startswith("/lane3") for path in handler.paths))
                self.assertTrue(all((chunks / name).read_bytes() == payload for name in ("chunk", "next", "one", "two")))
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_connection_reopens_after_idle_threshold(self):
        filesystem = load_filesystem_module()
        payload = b"idle-reopen"
        server, thread, handler = self._persistent_server(filesystem, payload)
        old_idle = filesystem.persistent_connection_idle_seconds
        filesystem.persistent_connection_idle_seconds = 0.05
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                try:
                    for name in ("one", "two"):
                        pool.download(
                            [f"http://127.0.0.1:{server.server_address[1]}/{name}"],
                            0,
                            chunks / name,
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            per_request_timeout_seconds=1,
                        )
                        if name == "one":
                            time.sleep(0.08)
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual(handler.connections, 2)
        finally:
            filesystem.persistent_connection_idle_seconds = old_idle
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_recreates_after_short_body_and_resumes(self):
        filesystem = load_filesystem_module()
        payload = b"short-body-retry"

        def short_once(request, number):
            if number != 1:
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(payload)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}"
            )
            request.send_header("ETag", f'"{hashlib.sha256(payload).hexdigest()}"')
            request.end_headers()
            request.wfile.write(payload[:3])
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, short_once
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._persistent_download(filesystem, server, root, "chunk", payload)
                self.assertEqual((root / "chunks" / "chunk").read_bytes(), payload)
                self.assertGreaterEqual(handler.requests, 2)
                self.assertLessEqual(handler.requests, filesystem._ADAPTER_LANE_RETRIES)
                self.assertEqual(handler.ranges[0], "bytes=0-15")
                self.assertTrue(all(value == "bytes=3-15" for value in handler.ranges[1:]))
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_allows_continuous_trickle(self):
        filesystem = load_filesystem_module()
        payload = b"trickle-body"

        def trickle_once(request, number):
            if number != 1:
                return False
            start, end = map(
                int,
                request.headers["Range"].removeprefix("bytes=").split("-", 1),
            )
            body = payload[start : end + 1]
            request.send_response(206)
            request.send_header("Content-Length", str(len(body)))
            request.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(payload)}"
            )
            request.end_headers()
            for value in body:
                try:
                    request.wfile.write(bytes((value,)))
                    request.wfile.flush()
                except OSError:
                    return True
                time.sleep(0.05)
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, trickle_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                started = time.monotonic()
                try:
                    pool.download(
                        [
                            f"http://127.0.0.1:{server.server_address[1]}/slow",
                            f"http://127.0.0.1:{server.server_address[1]}/fast",
                        ],
                        0,
                        chunks / "trickle",
                        parent_fd,
                        expected_size=len(payload),
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        per_request_timeout_seconds=0.2,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
                elapsed = time.monotonic() - started
                self.assertEqual((chunks / "trickle").read_bytes(), payload)
                self.assertEqual(handler.requests, 1)
                self.assertGreaterEqual(elapsed, 0.55)
                self.assertLess(elapsed, 1.2)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_retries_temporary_http_status(self):
        filesystem = load_filesystem_module()
        payload = b"status-retry"

        def unavailable_once(request, number):
            if number != 1:
                return False
            request.send_response(404)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, unavailable_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                self._persistent_download(
                    filesystem, server, Path(temporary), "status", payload
                )
                self.assertEqual(handler.requests, 2)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_caps_idle_timeout_at_lane_timeout(self):
        filesystem = load_filesystem_module()
        payload = b"timeout-source"
        seen_timeouts = []
        old_read_range = filesystem._PersistentAdapterConnection.read_range

        def fake_read_range(connection, source_url, **kwargs):
            seen_timeouts.append(kwargs["timeout"])
            kwargs["consume"](payload)

        filesystem._PersistentAdapterConnection.read_range = fake_read_range
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                try:
                    pool.download(
                        ["http://source/chunk"],
                        0,
                        chunks / "chunk",
                        parent_fd,
                        expected_size=len(payload),
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        per_request_timeout_seconds=60,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
        finally:
            filesystem._PersistentAdapterConnection.read_range = old_read_range
        self.assertEqual(seen_timeouts, [filesystem._ADAPTER_LANE_TIMEOUT_S])

    def test_persistent_config_retries_404_then_succeeds(self):
        filesystem = load_filesystem_module()
        payload = b"config-404"

        def unavailable_once(request, number):
            if number != 1:
                return False
            request.send_response(404)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, unavailable_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "adapter_config.json"
                filesystem._download_persistent_file(
                    [f"http://127.0.0.1:{server.server_address[1]}/adapter_config.json"],
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    per_request_timeout_seconds=1,
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.requests, 2)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_config_retries_503_then_succeeds(self):
        filesystem = load_filesystem_module()
        payload = b"config-503"

        def unavailable_once(request, number):
            if number != 1:
                return False
            request.send_response(503)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, unavailable_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "adapter_config.json"
                filesystem._download_persistent_file(
                    [f"http://127.0.0.1:{server.server_address[1]}/adapter_config.json"],
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    per_request_timeout_seconds=1,
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.requests, 2)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_config_retries_http500(self):
        filesystem = load_filesystem_module()
        payload = b"config-500"

        def server_error(request, number):
            if number != 1:
                return False
            request.send_response(500)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, server_error
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "adapter_config.json"
                filesystem._download_persistent_file(
                    [f"http://127.0.0.1:{server.server_address[1]}/adapter_config.json"],
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    per_request_timeout_seconds=1,
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.requests, 2)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_config_local_write_failure_is_terminal(self):
        filesystem = load_filesystem_module()
        payload = b"config-disk"
        server, thread, handler = self._persistent_server(filesystem, payload)
        old_fdopen = self._inject_enospc(filesystem)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(OSError, "No space left"):
                    filesystem._download_persistent_file(
                        [f"http://127.0.0.1:{server.server_address[1]}/adapter_config.json"],
                        Path(temporary) / "adapter_config.json",
                        expected_size=len(payload),
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        per_request_timeout_seconds=1,
                    )
                self.assertEqual(handler.requests, 1)
        finally:
            filesystem.os.fdopen = old_fdopen
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_legacy_full_single_enospc_is_terminal(self):
        filesystem = load_filesystem_module()
        payload = b"legacy-single"
        server, thread, handler = self._persistent_server(filesystem, payload)
        old_fdopen = self._inject_enospc(filesystem)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(OSError, "No space left"):
                    self._legacy_full_download(
                        filesystem,
                        server,
                        Path(temporary),
                        "adapter_model.safetensors",
                        payload,
                    )
                self.assertEqual(handler.requests, 1)
        finally:
            filesystem.os.fdopen = old_fdopen
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_legacy_full_parallel_enospc_is_terminal(self):
        filesystem = load_filesystem_module()
        payload = b"p" * (filesystem._PARALLEL_DOWNLOAD_THRESHOLD + 1)
        server, thread, handler = self._persistent_server(filesystem, payload)
        old_fdopen = self._inject_enospc(filesystem)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                url = f"http://127.0.0.1:{server.server_address[1]}/adapter_model.safetensors"
                with self.assertRaisesRegex(OSError, "No space left"):
                    self._legacy_full_download(
                        filesystem,
                        server,
                        Path(temporary),
                        "adapter_model.safetensors",
                        payload,
                        stream_cap=2,
                        source_urls=[url, url],
                    )
                self.assertLessEqual(handler.requests, 2)
        finally:
            filesystem.os.fdopen = old_fdopen
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_legacy_full_network_reset_resumes_verified_bytes(self):
        filesystem = load_filesystem_module()
        payload = b"legacy-reset-resume"
        ranges = []

        def reset_once(request, number):
            ranges.append(request.headers.get("Range"))
            if number != 1:
                return False
            request.send_response(200)
            request.send_header("Content-Length", str(len(payload)))
            request.end_headers()
            request.wfile.write(payload[:5])
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, reset_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                url = f"http://127.0.0.1:{server.server_address[1]}/adapter_model.safetensors"
                destination = self._legacy_full_download(
                    filesystem,
                    server,
                    Path(temporary),
                    "adapter_model.safetensors",
                    payload,
                    source_urls=[url, url],
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.requests, 2)
                self.assertEqual(
                    ranges,
                    [None, f"bytes=5-"],
                )
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_legacy_full_trickle_body_hits_request_timeout(self):
        filesystem = load_filesystem_module()
        payload = b"x" * 128

        def trickle_body(request, _number):
            range_header = request.headers.get("Range")
            start = 0 if range_header is None else int(
                range_header.removeprefix("bytes=").split("-", 1)[0]
            )
            body = payload[start:]
            request.send_response(206)
            request.send_header("Content-Length", str(len(body)))
            request.send_header(
                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
            )
            request.end_headers()
            request.wfile.write(body[:1])
            request.wfile.flush()
            time.sleep(0.5)
            return True

        server, thread, _handler = self._persistent_server(
            filesystem, payload, trickle_body
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        old_threshold = filesystem._PARALLEL_DOWNLOAD_THRESHOLD
        filesystem.lane_rotation_delay_seconds = 0
        filesystem._PARALLEL_DOWNLOAD_THRESHOLD = 0
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                url = f"http://127.0.0.1:{server.server_address[1]}/trickle"
                with self.assertRaises(OSError):
                    self._legacy_full_download(
                        filesystem,
                        server,
                        Path(temporary),
                        "trickle",
                        payload,
                        stream_cap=2,
                        source_urls=[url, url],
                        per_request_timeout_seconds=0.05,
                        deadline_seconds=0.4,
                    )
            self.assertLess(time.monotonic() - started, 0.8)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            filesystem._PARALLEL_DOWNLOAD_THRESHOLD = old_threshold
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_legacy_full_cancellation_is_bounded_during_body_read(self):
        filesystem = load_filesystem_module()
        payload = b"cancel-legacy-body"

        def stalled_body(request, _number):
            request.send_response(206)
            request.send_header("Content-Length", str(len(payload)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}"
            )
            request.end_headers()
            request.wfile.write(payload[:1])
            request.wfile.flush()
            time.sleep(0.5)
            return True

        server, thread, _handler = self._persistent_server(
            filesystem, payload, stalled_body
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        old_threshold = filesystem._PARALLEL_DOWNLOAD_THRESHOLD
        filesystem.lane_rotation_delay_seconds = 0
        filesystem._PARALLEL_DOWNLOAD_THRESHOLD = 0
        cancel_event = threading.Event()
        errors = []
        started = time.monotonic()
        def run_download():
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    url = f"http://127.0.0.1:{server.server_address[1]}/cancel"
                    self._legacy_full_download(
                        filesystem,
                        server,
                        Path(temporary),
                        "cancel",
                        payload,
                        stream_cap=2,
                        source_urls=[url, url],
                        per_request_timeout_seconds=0.1,
                        cancel_event=cancel_event,
                        deadline_seconds=2,
                    )
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_download)
        worker.start()
        time.sleep(0.05)
        cancel_event.set()
        worker.join(timeout=1)
        elapsed = time.monotonic() - started
        try:
            self.assertFalse(worker.is_alive())
            self.assertLess(elapsed, 0.7)
            self.assertTrue(errors)
            self.assertIsInstance(errors[0], TimeoutError)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            filesystem._PARALLEL_DOWNLOAD_THRESHOLD = old_threshold
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_delta_chunk_network_reset_resumes_verified_bytes(self):
        filesystem = load_filesystem_module()
        payload = b"delta-reset-resume"
        ranges = []

        def reset_once(request, number):
            ranges.append(request.headers["Range"])
            if number != 1:
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(payload)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}"
            )
            request.send_header("ETag", f'"{hashlib.sha256(payload).hexdigest()}"')
            request.end_headers()
            request.wfile.write(payload[:4])
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, reset_once
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "chunk-0"
                filesystem._download_persistent_file(
                    [f"http://127.0.0.1:{server.server_address[1]}/chunk-0"],
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    per_request_timeout_seconds=1,
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.requests, 2)
                self.assertEqual(
                    ranges,
                    [
                        f"bytes=0-{len(payload) - 1}",
                        f"bytes=4-{len(payload) - 1}",
                    ],
                )
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_rejects_changed_etag_without_retry(self):
        filesystem = load_filesystem_module()
        payload = b"identity-check"

        def bad_identity(request, number):
            if number != 1:
                return False
            start, end = map(
                int,
                request.headers["Range"].removeprefix("bytes=").split("-", 1),
            )
            body = payload[start : end + 1]
            request.send_response(206)
            request.send_header("Content-Length", str(len(body)))
            request.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(payload)}"
            )
            request.send_header("ETag", '"changed"')
            request.end_headers()
            request.wfile.write(body)
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, bad_identity
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                (chunks / "identity").write_bytes(payload[:2])
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                try:
                    with self.assertRaisesRegex(OSError, "ETag"):
                        pool.download(
                            [f"http://127.0.0.1:{server.server_address[1]}/identity"],
                            0,
                            chunks / "identity",
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            per_request_timeout_seconds=2,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual((chunks / "identity").read_bytes(), payload[:2])
                self.assertEqual(handler.requests, 1)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_retries_http500(self):
        filesystem = load_filesystem_module()
        payload = b"server-error"

        def server_error(request, number):
            if number != 1:
                return False
            request.send_response(500)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, server_error
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                try:
                    pool.download(
                        [f"http://127.0.0.1:{server.server_address[1]}/error"],
                        0,
                        chunks / "error",
                        parent_fd,
                        expected_size=len(payload),
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        per_request_timeout_seconds=2,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual((chunks / "error").read_bytes(), payload)
                self.assertEqual(handler.requests, 2)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_read_timeout_observes_cancellation(self):
        filesystem = load_filesystem_module()
        payload = b"cancel-read"

        def stall_body(request, number):
            if number != 1:
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(payload)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}"
            )
            request.end_headers()
            time.sleep(0.5)
            return True

        server, thread, _handler = self._persistent_server(
            filesystem, payload, stall_body
        )
        cancel_event = threading.Event()
        errors = []
        started = time.monotonic()

        def run_download():
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    self._persistent_download(
                        filesystem,
                        server,
                        Path(temporary),
                        "cancel",
                        payload,
                        cancel_event=cancel_event,
                        per_request_timeout_seconds=0.05,
                    )
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_download)
        worker.start()
        time.sleep(0.05)
        cancel_event.set()
        worker.join(timeout=1)
        elapsed = time.monotonic() - started
        try:
            self.assertFalse(worker.is_alive())
            self.assertLess(elapsed, 0.8)
            self.assertTrue(errors)
            self.assertIsInstance(errors[0], TimeoutError)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_bounded_full_scheduler_refills_before_slow_outlier(self):
        filesystem = load_filesystem_module()
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        started = {}
        pending = 0
        maximum_pending = 0

        def worker(index, _item):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
                started[index] = time.monotonic()
            time.sleep(0.25 if index == 0 else 0.03)
            with lock:
                active -= 1

        real_executor = filesystem.ThreadPoolExecutor

        class RecordingExecutor:
            def __init__(self, max_workers):
                self.executor = real_executor(max_workers=max_workers)

            def submit(self, callback, *args, **kwargs):
                nonlocal pending, maximum_pending
                with lock:
                    pending += 1
                    maximum_pending = max(maximum_pending, pending)

                def run():
                    nonlocal pending
                    try:
                        return callback(*args, **kwargs)
                    finally:
                        with lock:
                            pending -= 1

                return self.executor.submit(run)

            def shutdown(self, *args, **kwargs):
                return self.executor.shutdown(*args, **kwargs)

        filesystem.ThreadPoolExecutor = RecordingExecutor
        try:
            filesystem._run_bounded_futures(
                list(range(5)),
                worker,
                2,
                time.monotonic() + 2,
                threading.Event(),
            )
        finally:
            filesystem.ThreadPoolExecutor = real_executor
        self.assertLess(started[2], started[0] + 0.20)
        self.assertLessEqual(maximum_active, 2)
        self.assertLessEqual(maximum_pending, 2)

    def test_bounded_full_scheduler_caps_configured_streams_to_lanes(self):
        filesystem = load_filesystem_module()
        lock = threading.Lock()
        pending = 0
        maximum_pending = 0

        def worker(_index, _item):
            time.sleep(0.005)

        real_executor = filesystem.ThreadPoolExecutor

        class RecordingExecutor:
            def __init__(self, max_workers):
                self.executor = real_executor(max_workers=max_workers)

            def submit(self, callback, *args, **kwargs):
                nonlocal pending, maximum_pending
                with lock:
                    pending += 1
                    maximum_pending = max(maximum_pending, pending)

                def run():
                    nonlocal pending
                    try:
                        return callback(*args, **kwargs)
                    finally:
                        with lock:
                            pending -= 1

                return self.executor.submit(run)

            def shutdown(self, *args, **kwargs):
                return self.executor.shutdown(*args, **kwargs)

        filesystem.ThreadPoolExecutor = RecordingExecutor
        try:
            configured_streams = 96
            responsive_lanes = 24
            filesystem._run_bounded_futures(
                list(range(48)),
                worker,
                min(configured_streams, responsive_lanes),
                time.monotonic() + 5,
                threading.Event(),
            )
        finally:
            filesystem.ThreadPoolExecutor = real_executor
        self.assertLessEqual(maximum_pending, responsive_lanes)

    def test_persistent_chunk_header_stall_observes_cancellation(self):
        filesystem = load_filesystem_module()
        payload = b"header-stall"

        def stall_headers(_request, number):
            if number == 1:
                time.sleep(0.5)
                return True
            return False

        server, thread, _handler = self._persistent_server(
            filesystem, payload, stall_headers
        )
        cancel_event = threading.Event()
        errors = []
        started = time.monotonic()

        def run_download():
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    self._persistent_download(
                        filesystem,
                        server,
                        Path(temporary),
                        "header",
                        payload,
                        cancel_event=cancel_event,
                        per_request_timeout_seconds=0.05,
                    )
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_download)
        worker.start()
        time.sleep(0.05)
        cancel_event.set()
        worker.join(timeout=5.5)
        elapsed = time.monotonic() - started
        try:
            self.assertFalse(worker.is_alive())
            self.assertLess(elapsed, 5.5)
            self.assertTrue(errors)
            self.assertIsInstance(errors[0], TimeoutError)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_download_keeps_public_signature(self):
        filesystem = load_filesystem_module()
        signature = str(inspect.signature(filesystem.download)).replace("pathlib._local.Path", "pathlib.Path")
        self.assertEqual(
            signature,
            "(source_url: str, destination: pathlib.Path, *, expected_size: int | None = None, expected_sha256: str | None = None, progress=None, source_urls: list[str] | None = None, parallel_ranges: bool = True, deadline: float | None = None, cancel_event: threading.Event | None = None, per_request_timeout_seconds: float | None = None, retry_http_statuses: set[int] | None = None) -> None",
        )
        self.assertFalse(hasattr(filesystem.download, "__wrapped__"))

    def test_sequential_download_survives_parent_swap_at_output_open(self):
        filesystem = load_filesystem_module()
        payload = b"safe"

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(_request, timeout):
            self.assertGreater(timeout, 0)
            return Response(payload)

        old_urlopen = filesystem.urllib.request.urlopen
        old_open = filesystem.os.open
        filesystem.urllib.request.urlopen = fake_urlopen
        triggered = False
        try:
            with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
                root = Path(temporary)
                cache = root / "cache"
                cache.mkdir()
                cache_original = root / "cache-original"
                outside = Path(external)

                def swap_before_output_open(path, flags, mode=0o777, *, dir_fd=None):
                    nonlocal triggered
                    if path == "output" and dir_fd is not None and not triggered:
                        cache.rename(cache_original)
                        cache.symlink_to(outside, target_is_directory=True)
                        triggered = True
                    return old_open(path, flags, mode, dir_fd=dir_fd)

                filesystem.os.open = swap_before_output_open
                filesystem.download(
                    "http://lane/output",
                    cache / "output",
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    parallel_ranges=False,
                )
                self.assertTrue(triggered)
                self.assertEqual((cache_original / "output").read_bytes(), payload)
                self.assertFalse((outside / "output").exists())
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.os.open = old_open

    def test_sequential_download_resumes_existing_output(self):
        filesystem = load_filesystem_module()
        payload = b"safe"
        calls = []

        class Response(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, **_kwargs):
            calls.append(request.headers.get("Range"))
            return Response(payload[2:])

        old_urlopen = filesystem.urllib.request.urlopen
        filesystem.urllib.request.urlopen = fake_urlopen
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "output"
                destination.write_bytes(payload[:2])
                filesystem.download(
                    "http://lane/output",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    parallel_ranges=False,
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
        self.assertEqual(calls, ["bytes=2-"])

    def test_glm_stream_count_accepts_24_lanes(self):
        filesystem = load_filesystem_module()
        old_value = filesystem.os.environ.get(filesystem._DOWNLOAD_STREAMS_ENV)
        filesystem.os.environ[filesystem._DOWNLOAD_STREAMS_ENV] = "24"
        try:
            self.assertEqual(filesystem._configured_download_streams(), 24)
        finally:
            if old_value is None:
                filesystem.os.environ.pop(filesystem._DOWNLOAD_STREAMS_ENV, None)
            else:
                filesystem.os.environ[filesystem._DOWNLOAD_STREAMS_ENV] = old_value

    def test_chunk_timeout_never_exceeds_stage_deadline(self):
        filesystem = load_filesystem_module()
        remaining = 0.05
        timeout = filesystem._chunk_request_timeout(
            filesystem.time.monotonic() + remaining,
            attempt=0,
            configured_timeout=15,
        )
        self.assertGreater(timeout, 0)
        self.assertLessEqual(timeout, remaining)

    def test_request_timeout_reports_a_deadline_crossing(self):
        filesystem = load_filesystem_module()
        old_monotonic = filesystem.time.monotonic
        try:
            for function, arguments in (
                (filesystem._request_timeout, (101.0,)),
                (filesystem._chunk_request_timeout, (101.0, 0, 15.0)),
            ):
                clock = iter((100.0, 102.0))
                filesystem.time.monotonic = lambda: next(clock)
                with self.assertRaisesRegex(TimeoutError, "exceeded its time limit"):
                    function(*arguments)
        finally:
            filesystem.time.monotonic = old_monotonic

    def test_metadata_read_moves_past_a_stalled_lane(self):
        filesystem = load_filesystem_module()
        calls = []

        class Response(io.BytesIO):
            headers = {"Content-Length": "4"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            url = request.full_url if hasattr(request, "full_url") else request
            calls.append((url, timeout))
            if url.startswith("http://lane-1/"):
                raise TimeoutError("stalled")
            return Response(b"safe")

        old_urlopen = filesystem.urllib.request.urlopen
        old_lanes = filesystem._adapter_source_urls
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem._adapter_source_urls = lambda _source: [
            "http://lane-1/manifest",
            "http://lane-2/manifest",
        ]
        try:
            self.assertEqual(
                filesystem._read_source_bytes(
                    "http://lane-1/manifest",
                    max_bytes=16,
                ),
                b"safe",
            )
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem._adapter_source_urls = old_lanes

        self.assertEqual([call[0] for call in calls], [
            "http://lane-1/manifest",
            "http://lane-2/manifest",
        ])
        self.assertTrue(all(call[1] <= filesystem._configured_chunk_timeout() for call in calls))

    def test_metadata_read_tries_all_lanes_after_temporary_failures(self):
        filesystem = load_filesystem_module()
        source_urls = [f"http://lane-{index}/manifest" for index in range(8)]
        calls = []

        class Response(io.BytesIO):
            headers = {"Content-Length": "4"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, **_kwargs):
            calls.append(request.full_url if hasattr(request, "full_url") else request)
            index = len(calls) - 1
            if index == 0:
                raise filesystem.HTTPError(calls[-1], 404, "missing", None, None)
            if index == 1:
                raise filesystem.HTTPError(calls[-1], 503, "starting", None, None)
            if index < len(source_urls) - 1:
                raise ConnectionResetError("connection reset by peer")
            return Response(b"safe")

        old_urlopen = filesystem.urllib.request.urlopen
        old_lanes = filesystem._adapter_source_urls
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem._adapter_source_urls = lambda _source: source_urls
        try:
            self.assertEqual(
                filesystem._read_source_bytes(
                    source_urls[0],
                    deadline=filesystem.time.monotonic() + 5,
                    max_bytes=16,
                ),
                b"safe",
            )
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem._adapter_source_urls = old_lanes

        self.assertEqual(calls, source_urls)

    def test_metadata_read_does_not_retry_invalid_content_length(self):
        filesystem = load_filesystem_module()
        calls = []

        class Response(io.BytesIO):
            headers = {"Content-Length": "not-a-size"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, **_kwargs):
            calls.append(request.full_url if hasattr(request, "full_url") else request)
            return Response()

        old_urlopen = filesystem.urllib.request.urlopen
        old_lanes = filesystem._adapter_source_urls
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem._adapter_source_urls = lambda _source: [
            "http://lane-1/manifest",
            "http://lane-2/manifest",
        ]
        try:
            with self.assertRaisesRegex(OSError, "invalid Content-Length"):
                filesystem._read_source_bytes(
                    "http://lane-1/manifest",
                    max_bytes=16,
                )
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem._adapter_source_urls = old_lanes

        self.assertEqual(calls, ["http://lane-1/manifest"])

    def test_metadata_read_does_not_retry_identity_error(self):
        filesystem = load_filesystem_module()
        calls = []

        def fake_urlopen(request, **_kwargs):
            calls.append(request.full_url if hasattr(request, "full_url") else request)
            raise OSError("adapter identity differs")

        old_urlopen = filesystem.urllib.request.urlopen
        old_lanes = filesystem._adapter_source_urls
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem._adapter_source_urls = lambda _source: [
            "http://lane-1/manifest",
            "http://lane-2/manifest",
        ]
        try:
            with self.assertRaisesRegex(OSError, "identity differs"):
                filesystem._read_source_bytes("http://lane-1/manifest")
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem._adapter_source_urls = old_lanes

        self.assertEqual(calls, ["http://lane-1/manifest"])

    def test_metadata_retry_classifier_is_fail_closed(self):
        filesystem = load_filesystem_module()
        retryable = (
            TimeoutError("timed out"),
            ConnectionResetError("connection reset by peer"),
            ConnectionRefusedError("connection refused"),
            filesystem.URLError(ConnectionResetError("connection reset by peer")),
            filesystem.HTTPError("http://lane/manifest", 404, "missing", None, None),
            filesystem.HTTPError("http://lane/manifest", 429, "busy", None, None),
            filesystem.HTTPError("http://lane/manifest", 500, "server error", None, None),
            filesystem.HTTPError("http://lane/manifest", 502, "bad gateway", None, None),
            filesystem.HTTPError("http://lane/manifest", 503, "starting", None, None),
            filesystem.HTTPError("http://lane/manifest", 504, "timeout", None, None),
        )
        terminal = (
            OSError("generic metadata failure"),
            OSError("connection reset by peer"),
            filesystem.URLError("connection reset by peer"),
            OSError("adapter checksum mismatch"),
            OSError("adapter identity differs"),
            OSError("adapter protocol error"),
            OSError("disk full"),
            TimeoutError("checksum verification timed out"),
            filesystem.HTTPError("http://lane/manifest", 503, "checksum mismatch", None, None),
        )
        for exception in retryable:
            with self.subTest(retryable=repr(exception)):
                self.assertTrue(filesystem._is_retryable_metadata_error(exception))
        for exception in terminal:
            with self.subTest(terminal=repr(exception)):
                self.assertFalse(filesystem._is_retryable_metadata_error(exception))

    def test_source_check_removes_stalled_connections(self):
        filesystem = load_filesystem_module()
        requests = []

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        def fake_urlopen(request, **_kwargs):
            requests.append(request)
            if "lane-2" in request.full_url:
                raise TimeoutError("stalled")
            return Response()

        old_urlopen = filesystem.urllib.request.urlopen
        old_lanes = filesystem._adapter_source_urls
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem._adapter_source_urls = lambda _source: [
            "http://lane-1/adapter",
            "http://lane-2/adapter",
            "http://lane-3/adapter",
        ]
        try:
            responsive, total = filesystem._responsive_adapter_source_urls(
                "http://lane-1/adapter",
                filesystem.time.monotonic() + 5,
                None,
            )
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem._adapter_source_urls = old_lanes

        self.assertEqual(total, 3)
        self.assertEqual(responsive, [
            "http://lane-1/adapter",
            "http://lane-3/adapter",
        ])
        self.assertTrue(
            all(request.full_url.endswith("/adapter_config.json") for request in requests)
        )

    def test_parallel_ranges_use_configured_stream_cap(self):
        filesystem = load_filesystem_module()
        payload = b"x" * (20 * 1024 * 1024)
        calls = []
        workers = []

        class Response(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            value = request.headers["Range"]
            start, end = (int(part) for part in value.removeprefix("bytes=").split("-"))
            calls.append((request.full_url, start, end, timeout))
            return Response(payload[start : end + 1])

        real_executor = filesystem.ThreadPoolExecutor

        class RecordingExecutor:
            def __init__(self, max_workers):
                workers.append(max_workers)
                self.executor = real_executor(max_workers=max_workers)

            def submit(self, *args, **kwargs):
                return self.executor.submit(*args, **kwargs)

            def shutdown(self, *args, **kwargs):
                return self.executor.shutdown(*args, **kwargs)

        old_urlopen = filesystem.urllib.request.urlopen
        old_executor = filesystem.ThreadPoolExecutor
        old_streams = filesystem._configured_download_streams
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.ThreadPoolExecutor = RecordingExecutor
        filesystem._configured_download_streams = lambda: 2
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "adapter"
                filesystem.download(
                    "http://lane-1/adapter",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=[
                        "http://lane-1/adapter",
                        "http://lane-2/adapter",
                        "http://lane-3/adapter",
                        "http://lane-4/adapter",
                    ],
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.ThreadPoolExecutor = old_executor
            filesystem._configured_download_streams = old_streams

        self.assertEqual(workers, [2])
        self.assertEqual(len(calls), 2)

    def test_full_and_global_download_worker_caps(self):
        filesystem = load_filesystem_module()
        payload = b"x" * (16 * 1024 * 1024)
        source_urls = [f"http://lane-{index}/adapter" for index in range(96)]
        calls = []
        workers = []

        class Response(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            value = request.headers.get("Range")
            if value is None:
                return Response(payload)
            start, end = (int(part) for part in value.removeprefix("bytes=").split("-"))
            calls.append(request.full_url)
            self.assertGreater(timeout, 0)
            return Response(payload[start : end + 1])

        real_executor = filesystem.ThreadPoolExecutor

        class RecordingExecutor:
            def __init__(self, max_workers):
                workers.append(max_workers)
                self.executor = real_executor(max_workers=max_workers)

            def submit(self, *args, **kwargs):
                return self.executor.submit(*args, **kwargs)

            def shutdown(self, *args, **kwargs):
                return self.executor.shutdown(*args, **kwargs)

        old_urlopen = filesystem.urllib.request.urlopen
        old_executor = filesystem.ThreadPoolExecutor
        old_threshold = filesystem._PARALLEL_DOWNLOAD_THRESHOLD
        old_lanes = filesystem._adapter_source_urls
        old_streams = filesystem._configured_download_streams
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.ThreadPoolExecutor = RecordingExecutor
        filesystem._PARALLEL_DOWNLOAD_THRESHOLD = 0
        filesystem._adapter_source_urls = lambda _source: source_urls
        filesystem._configured_download_streams = lambda: 48

        def run_download(source_url, destination, expected, stream_cap=None):
            parent_fd = filesystem._open_directory_nofollow(
                destination.parent,
                create=False,
                label="test download parent",
            )
            try:
                filesystem._download_with_parent(
                    source_url,
                    destination,
                    expected_size=len(expected),
                    expected_sha256=hashlib.sha256(expected).hexdigest(),
                    stream_cap=stream_cap,
                    parent_fd=parent_fd,
                )
            finally:
                filesystem.os.close(parent_fd)

        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                full = root / "full"
                run_download(
                    source_urls[0],
                    full,
                    payload,
                    stream_cap=filesystem.full_download_streams,
                )
                self.assertEqual(full.read_bytes(), payload)
                self.assertEqual(workers, [48])
                self.assertEqual(len(calls), 48)
                self.assertEqual(set(calls), set(source_urls[:48]))

                calls.clear()
                global_copy = root / "global"
                run_download(source_urls[0], global_copy, payload)
                self.assertEqual(global_copy.read_bytes(), payload)
                self.assertEqual(workers, [48, 48])
                self.assertEqual(len(calls), 48)

                source_urls = source_urls[:3]
                payload = b"xxxxx"
                calls.clear()
                small = root / "small"
                run_download(
                    source_urls[0],
                    small,
                    payload,
                    stream_cap=filesystem.full_download_streams,
                )
                self.assertEqual(small.read_bytes(), payload)
                self.assertEqual(workers, [48, 48, 3])
                self.assertEqual(len(calls), 3)

                empty = root / "empty"
                run_download(
                    source_urls[0],
                    empty,
                    b"",
                    stream_cap=filesystem.full_download_streams,
                )
                self.assertEqual(empty.read_bytes(), b"")
                self.assertEqual(workers, [48, 48, 3])
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.ThreadPoolExecutor = old_executor
            filesystem._PARALLEL_DOWNLOAD_THRESHOLD = old_threshold
            filesystem._adapter_source_urls = old_lanes
            filesystem._configured_download_streams = old_streams

    def test_parallel_download_resumes_parts_and_removes_private_files(self):
        filesystem = load_filesystem_module()
        payload = b"abcdefgh"
        calls = []

        class Response(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, **_kwargs):
            range_value = request.headers["Range"]
            start, end = (
                int(value)
                for value in range_value.removeprefix("bytes=").split("-")
            )
            calls.append((start, end))
            return Response(payload[start : end + 1])

        old_urlopen = filesystem.urllib.request.urlopen
        old_threshold = filesystem._PARALLEL_DOWNLOAD_THRESHOLD
        old_streams = filesystem._configured_download_streams
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem._PARALLEL_DOWNLOAD_THRESHOLD = 0
        filesystem._configured_download_streams = lambda: 2
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "output"
                parts = destination.with_name("output.parts")
                parts.mkdir()
                (parts / "manifest.json").write_text(
                    json.dumps(
                        {
                            "source_url": "http://lane/output",
                            "size": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "streams": 2,
                        }
                    ),
                    encoding="utf-8",
                )
                (parts / "part-000").write_bytes(payload[:2])
                filesystem.download(
                    "http://lane/output",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=["http://lane/output", "http://lane-2/output"],
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertFalse(parts.exists())
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem._PARALLEL_DOWNLOAD_THRESHOLD = old_threshold
            filesystem._configured_download_streams = old_streams
        self.assertEqual(sorted(calls), [(2, 3), (4, 7)])

    def test_chunk_retry_uses_short_timeout_and_next_lane(self):
        filesystem = load_filesystem_module()
        payload = b"chunk"
        calls = []
        sleeps = []

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, timeout))
            if len(calls) == 1:
                raise TimeoutError("lane read timed out")
            return Response(payload)

        old_urlopen = filesystem.urllib.request.urlopen
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.sleep = sleeps.append
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "chunk"
                filesystem.download(
                    "http://lane-1/chunk",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=["http://lane-1/chunk", "http://lane-2/chunk"],
                    parallel_ranges=False,
                    deadline=filesystem.time.monotonic() + 100,
                    per_request_timeout_seconds=15,
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.sleep = old_sleep

        self.assertEqual([call[0] for call in calls], [
            "http://lane-1/chunk",
            "http://lane-2/chunk",
        ])
        self.assertTrue(all(0 < call[1] <= 15 for call in calls))
        self.assertEqual(sleeps, [filesystem.lane_rotation_delay_seconds])

    def test_active_chunk_can_run_longer_than_idle_timeout(self):
        filesystem = load_filesystem_module()
        payload = b"six-bytes"
        calls = []
        clock = [0.0]

        class SlowResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read1(self, _size):
                clock[0] += 6
                return payload[int(clock[0] // 6) - 1 : int(clock[0] // 6)]

        class FastResponse(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, request.headers.get("Range"), timeout))
            if len(calls) == 1:
                return SlowResponse()
            return FastResponse(payload[3:])

        old_urlopen = filesystem.urllib.request.urlopen
        old_monotonic = filesystem.time.monotonic
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.monotonic = lambda: clock[0]
        filesystem.time.sleep = lambda _seconds: None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "chunk"
                filesystem.download(
                    "http://slow/chunk",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=["http://slow/chunk", "http://fast/chunk"],
                    parallel_ranges=False,
                    deadline=100,
                    per_request_timeout_seconds=60,
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.monotonic = old_monotonic
            filesystem.time.sleep = old_sleep

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "http://slow/chunk")
        self.assertIsNone(calls[0][1])
        self.assertLessEqual(calls[0][2], filesystem._ADAPTER_LANE_TIMEOUT_S)

    def test_trickling_chunk_rotates_after_its_total_timeout(self):
        filesystem = load_filesystem_module()
        payload = b"six-bytes"
        calls = []
        clock = [0.0]

        class SlowResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read1(self, _size):
                clock[0] += 6
                return payload[int(clock[0] // 6) - 1 : int(clock[0] // 6)]

        class FastResponse(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, request.headers.get("Range"), timeout))
            if len(calls) == 1:
                return SlowResponse()
            return FastResponse(payload[3:])

        old_urlopen = filesystem.urllib.request.urlopen
        old_monotonic = filesystem.time.monotonic
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.monotonic = lambda: clock[0]
        filesystem.time.sleep = lambda _seconds: None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "chunk"
                filesystem.download(
                    "http://slow/chunk",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=["http://slow/chunk", "http://fast/chunk"],
                    parallel_ranges=False,
                    deadline=100,
                    per_request_timeout_seconds=15,
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.monotonic = old_monotonic
            filesystem.time.sleep = old_sleep

        self.assertEqual(calls[0][0], "http://slow/chunk")
        self.assertEqual(calls[1][0], "http://fast/chunk")
        self.assertEqual(calls[1][1], "bytes=3-")

    def test_chunk_retry_can_try_every_lane_before_deadline(self):
        filesystem = load_filesystem_module()
        payload = b"chunk"
        source_urls = [f"http://lane-{index}/chunk" for index in range(8)]
        calls = []

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, timeout))
            if len(calls) < len(source_urls):
                raise TimeoutError("lane read timed out")
            return Response(payload)

        old_urlopen = filesystem.urllib.request.urlopen
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.sleep = lambda _seconds: None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "chunk"
                filesystem.download(
                    source_urls[0],
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=source_urls,
                    parallel_ranges=False,
                    deadline=filesystem.time.monotonic() + 100,
                    per_request_timeout_seconds=15,
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.sleep = old_sleep

        self.assertEqual([call[0] for call in calls], source_urls)

    def test_relay_chunk_waits_past_one_lane_cycle(self):
        filesystem = load_filesystem_module()
        payload = b"chunk"
        calls = []
        clock = [0.0]

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, timeout))
            if len(calls) <= 70:
                raise filesystem.HTTPError(request.full_url, 503, "not ready", None, None)
            return Response(payload)

        old_urlopen = filesystem.urllib.request.urlopen
        old_monotonic = filesystem.time.monotonic
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.monotonic = lambda: clock[0]
        filesystem.time.sleep = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "chunk"
                filesystem.download(
                    "http://lane-1/chunk",
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    source_urls=["http://lane-1/chunk", "http://lane-2/chunk"],
                    parallel_ranges=False,
                    deadline=100.0,
                    per_request_timeout_seconds=15.0,
                    retry_http_statuses=filesystem._RETRYABLE_ADAPTER_HTTP_STATUSES,
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.monotonic = old_monotonic
            filesystem.time.sleep = old_sleep

        self.assertEqual(len(calls), 71)
        self.assertEqual(clock[0], 70.0)

    def test_range_retry_rotates_to_a_distinct_tunnel_supervisor(self):
        filesystem = load_filesystem_module()
        payload = b"chunk"
        source_urls = [
            f"http://127.0.0.1:{port}/chunk"
            for offset in range(2)
            for port in range(18100 + offset, 18196, 2)
        ]
        self.assertEqual(len(source_urls), 96)
        self.assertEqual(len(set(source_urls)), 96)
        calls = []

        class Response(io.BytesIO):
            status = 206

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, **_kwargs):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise OSError("first tunnel is unavailable")
            return Response(payload)

        old_urlopen = filesystem.urllib.request.urlopen
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.sleep = lambda _seconds: None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary)
                parent_fd = filesystem.os.open(parent, filesystem.os.O_RDONLY)
                try:
                    filesystem._download_range(
                        source_urls,
                        23,
                        parent / "chunk",
                        parent_fd,
                        0,
                        len(payload) - 1,
                    )
                finally:
                    filesystem.os.close(parent_fd)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.sleep = old_sleep

        self.assertEqual(calls, [source_urls[23], source_urls[24]])
        self.assertEqual(
            [int(url.split(":")[2].split("/", 1)[0]) for url in calls],
            [18146, 18148],
        )


if __name__ == "__main__":
    unittest.main()
