import errno
import hashlib
import http.client
import http.server
import importlib.util
import io
import json
import os
import ssl
import socketserver
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
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
    checkpoint_delta.adapter_stage_timeout_seconds = 600.0
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
    def test_fallback_download_headers_share_deadline_and_resume(self):
        for ranged in (False, True):
            with self.subTest(ranged=ranged), tempfile.TemporaryDirectory() as temporary:
                filesystem = load_filesystem_module()
                filesystem.lane_rotation_delay_seconds = 0.001
                payload = b"bounded-fallback-body"
                finished = threading.Event()

                def slow_header(request, number):
                    if number != 1:
                        return False
                    try:
                        request.connection.sendall(b"HTTP/1.1 206 Partial Content\r\nX-Slow: ")
                        for _ in range(30):
                            request.connection.sendall(b"x")
                            time.sleep(0.02)
                    except OSError:
                        pass
                    finally:
                        request.close_connection = True
                        finished.set()
                    return True

                server, thread, handler = self._persistent_server(filesystem, payload, slow_header)
                root = Path(temporary)
                destination = root / "part"
                destination.write_bytes(payload[:3])
                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    urls = [f"http://127.0.0.1:{server.server_address[1]}/{lane}" for lane in ("slow", "fast")]
                    started = time.monotonic()
                    kwargs = dict(deadline=started + 0.3, per_request_timeout_seconds=0.06, typed_transport=True)
                    if ranged:
                        filesystem._download_range(urls, 0, destination, parent_fd, 0, len(payload) - 1, **kwargs)
                    else:
                        filesystem._download_with_parent(
                            urls[0], destination, parent_fd=parent_fd, source_urls=urls,
                            expected_size=len(payload), expected_sha256=hashlib.sha256(payload).hexdigest(),
                            parallel_ranges=False, **kwargs)
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, 0.18)
                    self.assertEqual(destination.read_bytes(), payload)
                    self.assertEqual(handler.paths, ["/slow", "/fast"])
                    self.assertTrue(all(value.startswith("bytes=3-") for value in handler.ranges))
                    self.assert_no_socket_guards()
                    print(f"fallback headers: {elapsed:.3f}s; range={ranged}; 3-byte prefix resumed")
                finally:
                    os.close(parent_fd)
                    finished.wait(1)
                    server.shutdown()
                    thread.join(1)
                    server.server_close()

    def test_metadata_https_redirects_and_completed_guard_reuse(self):
        filesystem = load_filesystem_module()
        payload = b"verified-https-metadata"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certificate, key = root / "certificate.pem", root / "key.pem"
            filesystem.subprocess.run(
                ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                 "-keyout", str(key), "-out", str(certificate), "-subj", "/CN=127.0.0.1",
                 "-addext", "subjectAltName=IP:127.0.0.1"],
                check=True, stdout=filesystem.subprocess.DEVNULL, stderr=filesystem.subprocess.DEVNULL,
            )

            def redirect(request, _number):
                if request.path != "/redirect":
                    return False
                request.send_response(302)
                request.send_header("Location", "/metadata")
                request.send_header("Content-Length", "0")
                request.end_headers()
                return True

            server, thread, handler = self._persistent_server(filesystem, payload, redirect)
            server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_context.load_cert_chain(certificate, key)
            server.socket = server_context.wrap_socket(server.socket, server_side=True)
            client_context = ssl.create_default_context(cafile=str(certificate))
            try:
                url = f"https://127.0.0.1:{server.server_address[1]}/redirect"
                with mock.patch.object(filesystem.http.client, "_create_https_context", return_value=client_context):
                    for attempt in range(2):
                        self.assertEqual(filesystem._read_source_bytes(url, deadline=time.monotonic() + 0.2), payload)
                        self.assert_no_socket_guards()
                        if attempt == 0:
                            time.sleep(0.22)
                self.assertEqual(handler.paths, ["/redirect", "/metadata"] * 2)
            finally:
                server.shutdown()
                thread.join(1)
                server.server_close()

    def test_metadata_rejects_unguarded_schemes(self):
        filesystem = load_filesystem_module()
        for source in ("file:///no-metadata", "ftp://source/metadata", "data:text/plain,metadata"):
            with self.subTest(source=source), self.assertRaisesRegex(filesystem._PersistentProtocolError, "HTTP or HTTPS"):
                with filesystem._adapter_http_open(source, timeout=1):
                    self.fail("unsupported metadata source was opened")

    def test_metadata_connect_is_registered_before_cancellation(self):
        filesystem = load_filesystem_module()
        cancelled = threading.Event()
        entered, aborted = threading.Event(), threading.Event()

        class Socket:
            def settimeout(self, _timeout):
                pass

            def connect(self, _address):
                entered.set()
                aborted.wait(1)
                raise ConnectionResetError("cancelled connect")

            def shutdown(self, _how):
                aborted.set()

            def close(self):
                pass

        def cancel_connect():
            if entered.wait(1):
                cancelled.set()

        cancelling = threading.Thread(target=cancel_connect)
        cancelling.start()
        started = time.monotonic()
        try:
            with mock.patch.object(filesystem.socket, "socket", return_value=Socket()):
                with self.assertRaises(filesystem._PersistentTransportError):
                    filesystem._read_source_bytes(
                        "http://127.0.0.1:8100/metadata", deadline=started + 1, cancel_event=cancelled,
                    )
            self.assertLess(time.monotonic() - started, 0.15)
            self.assertTrue(aborted.is_set())
            self.assert_no_socket_guards()
        finally:
            aborted.set()
            cancelling.join(1)

    def test_metadata_header_trickle_has_one_attempt_deadline_and_cancellation(self):
        payload = b"complete-metadata"
        for mode in ("retry", "stage", "cancel"):
            with self.subTest(mode=mode):
                filesystem = load_filesystem_module()
                entered, finished, cancelled = threading.Event(), threading.Event(), threading.Event()

                def trickle(request, number):
                    if number != 1:
                        return False
                    try:
                        request.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                        entered.set()
                        for _ in range(50):
                            request.connection.sendall(b"x")
                            time.sleep(0.02)
                    except OSError:
                        pass
                    finally:
                        request.close_connection = True
                        finished.set()
                    return True

                server, thread, handler = self._persistent_server(filesystem, payload, trickle)
                cancelling = None
                if mode == "cancel":
                    def cancel_headers():
                        if entered.wait(1):
                            cancelled.set()
                    cancelling = threading.Thread(target=cancel_headers)
                    cancelling.start()
                try:
                    urls = [f"http://127.0.0.1:{server.server_address[1]}/{lane}" for lane in ("slow", "fast")]
                    started = time.monotonic()
                    limit = 0.06 if mode == "stage" else 1
                    with mock.patch.object(filesystem, "_metadata_source_urls", return_value=urls), mock.patch.object(
                        filesystem, "_configured_chunk_timeout", return_value=0.06 if mode == "retry" else 1
                    ):
                        if mode == "retry":
                            self.assertEqual(filesystem._read_source_bytes(urls[0], deadline=started + limit), payload)
                        else:
                            with self.assertRaises((TimeoutError, filesystem._PersistentTransportError)):
                                filesystem._read_source_bytes(urls[0], deadline=started + limit, cancel_event=cancelled)
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, 0.18)
                    self.assertEqual(handler.paths, ["/slow", "/fast"] if mode == "retry" else ["/slow"])
                    self.assert_no_socket_guards()
                    print(f"metadata headers: {elapsed:.3f}s, mode={mode}, lanes={len(handler.paths)}")
                finally:
                    completed = finished.wait(1)
                    if cancelling is not None:
                        cancelling.join(1)
                    server.shutdown()
                    thread.join(1)
                    server.server_close()
                    self.assertTrue(completed)

    def test_metadata_dns_stall_is_killed_and_reaped(self):
        filesystem = load_filesystem_module()
        real_popen = filesystem.subprocess.Popen
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                children = []
                cancelled = threading.Event()

                def stalled_lookup(_args, **kwargs):
                    child = real_popen([sys.executable, "-c", "import time; time.sleep(5)"], **kwargs)
                    children.append(child)
                    if cancel:
                        cancelled.set()
                    return child

                started = time.monotonic()
                with mock.patch.object(filesystem.subprocess, "Popen", side_effect=stalled_lookup):
                    with self.assertRaises(TimeoutError):
                        filesystem._adapter_address_info(("resolver.invalid", 80), started + 0.06, cancelled)
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 0.15)
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].returncode)
                with self.assertRaises(ProcessLookupError):
                    os.kill(children[0].pid, 0)
                self.assert_no_socket_guards()
                print(f"metadata DNS: {elapsed:.3f}s, cancellation={cancel}, child reaped")

    def test_metadata_head_and_lane_probes_abort_header_trickle(self):
        for probe in (False, True):
            with self.subTest(probe=probe):
                filesystem = load_filesystem_module()
                finished = threading.Event()

                class Handler(http.server.BaseHTTPRequestHandler):
                    def do_HEAD(self):
                        try:
                            self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                            for _ in range(50):
                                self.connection.sendall(b"x")
                                time.sleep(0.02)
                        except OSError:
                            pass
                        finally:
                            self.close_connection = True
                            finished.set()

                    def log_message(self, *_args):
                        pass

                server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                try:
                    url = f"http://127.0.0.1:{server.server_address[1]}/step_1"
                    started = time.monotonic()
                    with mock.patch.object(filesystem, "_adapter_source_urls", return_value=[url, url]):
                        with self.assertRaises((TimeoutError, filesystem._PersistentTransportError)):
                            if probe:
                                filesystem._responsive_adapter_source_urls(url, started + 0.06, None)
                            else:
                                filesystem._content_length(url, started + 0.06)
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, 0.18)
                    self.assert_no_socket_guards()
                    print(f"metadata HEAD: {elapsed:.3f}s, lane_probe={probe}")
                finally:
                    finished.wait(1)
                    server.shutdown()
                    thread.join(1)
                    server.server_close()

    def test_visibility_wait_uses_deadline_instead_of_attempt_limit(self):
        filesystem = load_filesystem_module()
        clock = [0.0]
        attempts = []

        def manifest(*_args, **_kwargs):
            attempts.append(clock[0])
            if len(attempts) <= 301:
                raise filesystem._PersistentTransportError("adapter manifest is missing")
            return {"version": 2}

        with mock.patch.object(filesystem.time, "monotonic", side_effect=lambda: clock[0]), mock.patch.object(
            filesystem.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        ), mock.patch.object(filesystem, "_read_chunked_xor_manifest", side_effect=manifest):
            self.assertEqual(filesystem._read_delta_manifest_with_visibility_wait(
                "http://source/step_2", deadline=600, cancel_event=None,
            ), {"version": 2})
        self.assertEqual(len(attempts), 302)
        self.assertEqual(clock[0], 301)
        clock[0] = 0
        with mock.patch.object(filesystem.time, "monotonic", side_effect=lambda: clock[0]), mock.patch.object(
            filesystem.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        ), mock.patch.object(filesystem, "_read_chunked_xor_manifest", side_effect=filesystem._PersistentTransportError(
            "adapter manifest is missing"
        )) as read_manifest:
            with self.assertRaises(TimeoutError):
                filesystem._read_delta_manifest_with_visibility_wait(
                    "http://source/step_2", deadline=5, cancel_event=None,
                )
        self.assertEqual(clock[0], 5)
        self.assertEqual(read_manifest.call_count, 5)

    def assert_no_socket_guards(self):
        self.assertFalse([thread for thread in threading.enumerate()
                          if thread.name == "adapter-socket-deadline"])

    def test_header_trickle_deadline_rotates_and_resumes_retained_prefix(self):
        filesystem = load_filesystem_module()
        filesystem.lane_rotation_delay_seconds = 0.001
        payload = b"header-resume-body"
        slow_finished = threading.Event()

        def slow_header(request, number):
            if number != 1:
                return False
            try:
                request.connection.sendall(b"HTTP/1.1 206 Partial Content\r\nX-Slow: ")
                for _ in range(30):
                    request.connection.sendall(b"x")
                    time.sleep(0.02)
            except OSError:
                pass
            finally:
                request.close_connection = True
                slow_finished.set()
            return True

        server, thread, handler = self._persistent_server(filesystem, payload, slow_header)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                destination = root / "chunk"
                destination.write_bytes(payload[:3])
                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                pool = filesystem._PersistentAdapterLanePool(1)
                started = time.monotonic()
                try:
                    pool.download(
                        [f"http://127.0.0.1:{server.server_address[1]}/{lane}" for lane in ("slow", "fast")],
                        0, destination, parent_fd, expected_size=len(payload),
                        expected_sha256=hashlib.sha256(payload).hexdigest(),
                        per_request_timeout_seconds=0.06, deadline=started + 0.12,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
                elapsed = time.monotonic() - started
                self.assertGreaterEqual(elapsed, 0.05)
                self.assertLess(elapsed, 0.18)
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.paths, ["/slow", "/fast"])
                self.assertEqual(handler.ranges, [f"bytes=3-{len(payload) - 1}"] * 2)
                self.assert_no_socket_guards()
                print(f"header trickle: {elapsed:.3f}s, 2 lanes, retained prefix 3 bytes")
        finally:
            completed = slow_finished.wait(1)
            server.shutdown()
            thread.join(1)
            server.server_close()
            self.assertTrue(completed)

    def test_metadata_trickle_retries_incrementally_and_bounds_stage(self):
        payload = b"metadata-body" * 4
        for chunked, stage_only in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(chunked=chunked, stage_only=stage_only):
                filesystem = load_filesystem_module()
                slow_finished = threading.Event()

                def slow_body(request, number):
                    if number != 1:
                        return False
                    request.send_response(200)
                    if chunked:
                        request.send_header("Transfer-Encoding", "chunked")
                        body = b"0" * 30 + f"{len(payload):x}\r\n".encode() + payload + b"\r\n0\r\n\r\n"
                    else:
                        request.send_header("Content-Length", str(len(payload)))
                        body = payload
                    request.end_headers()
                    try:
                        for value in body:
                            request.wfile.write(bytes((value,)))
                            request.wfile.flush()
                            time.sleep(0.02)
                    except OSError:
                        pass
                    finally:
                        request.close_connection = True
                        slow_finished.set()
                    return True

                server, thread, handler = self._persistent_server(filesystem, payload, slow_body)
                try:
                    urls = [f"http://127.0.0.1:{server.server_address[1]}/{lane}" for lane in ("slow", "fast")]
                    started = time.monotonic()
                    with mock.patch.object(filesystem, "_metadata_source_urls", return_value=urls), mock.patch.object(
                        filesystem, "_configured_chunk_timeout", return_value=1 if stage_only else 0.06
                    ):
                        if stage_only:
                            with self.assertRaises((TimeoutError, filesystem._PersistentTransportError)):
                                filesystem._read_source_bytes(urls[0], deadline=started + 0.12, max_bytes=1024)
                        else:
                            self.assertEqual(filesystem._read_source_bytes(
                                urls[0], deadline=started + 0.3, max_bytes=1024,
                            ), payload)
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, 0.2 if stage_only else 0.18)
                    self.assertEqual(handler.paths, ["/slow"] if stage_only else ["/slow", "/fast"])
                    self.assertEqual(handler.ranges, [None] * len(handler.paths))
                    self.assert_no_socket_guards()
                    print(f"metadata trickle: {elapsed:.3f}s, chunked={chunked}, stage_only={stage_only}")
                finally:
                    completed = slow_finished.wait(1)
                    server.shutdown()
                    thread.join(1)
                    server.server_close()
                    self.assertTrue(completed)

    def test_cancellation_interrupts_header_and_metadata_reads(self):
        for metadata in (False, True):
            with self.subTest(metadata=metadata):
                filesystem = load_filesystem_module()
                cancel = threading.Event()
                entered = threading.Event()
                finished = threading.Event()

                def blocked(request, _number):
                    try:
                        if metadata:
                            request.connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nx")
                        else:
                            request.connection.sendall(b"HTTP/1.1 206 Partial Content\r\nX-Slow: ")
                        entered.set()
                        finished.wait(1)
                    finally:
                        request.close_connection = True
                    return True

                server, thread, handler = self._persistent_server(filesystem, b"hello", blocked)
                def cancel_after_entry():
                    if entered.wait(1):
                        cancel.set()
                cancelling = threading.Thread(target=cancel_after_entry)
                cancelling.start()
                connection = filesystem._PersistentAdapterConnection()
                try:
                    url = f"http://127.0.0.1:{server.server_address[1]}/blocked"
                    started = time.monotonic()
                    with self.assertRaises((TimeoutError, filesystem._PersistentTransportError)):
                        if metadata:
                            filesystem._read_source_bytes(url, deadline=started + 5, cancel_event=cancel)
                        else:
                            connection.read_range(url, start=0, end=4, total=5, expected_sha256=None,
                                                  timeout=5, stage_deadline=started + 5,
                                                  cancel_event=cancel, consume=lambda _chunk: None)
                    elapsed = time.monotonic() - started
                    self.assertLess(elapsed, 0.15)
                    self.assertEqual(handler.requests, 1)
                    self.assert_no_socket_guards()
                    print(f"request cancellation: {elapsed:.3f}s, metadata={metadata}")
                finally:
                    finished.set()
                    connection.close()
                    cancelling.join(1)
                    server.shutdown()
                    thread.join(1)
                    server.server_close()

    def test_finished_deadline_guard_cannot_close_reused_connection(self):
        filesystem = load_filesystem_module()
        payload = b"reuse"
        server, thread, handler = self._persistent_server(filesystem, payload)
        connection = filesystem._PersistentAdapterConnection()
        try:
            received = []
            for index in range(2):
                connection.read_range(
                    f"http://127.0.0.1:{server.server_address[1]}/chunk", start=0, end=4,
                    total=5, expected_sha256=hashlib.sha256(payload).hexdigest(), timeout=1,
                    attempt_deadline=time.monotonic() + 0.06, consume=received.append,
                )
                self.assert_no_socket_guards()
                if index == 0:
                    time.sleep(0.08)
            self.assertEqual(b"".join(received), payload * 2)
            self.assertEqual(handler.connections, 1)
        finally:
            connection.close()
            server.shutdown()
            thread.join(1)
            server.server_close()

    def test_socket_guard_joins_an_abort_that_won_the_completion_race(self):
        filesystem = load_filesystem_module()
        for local_error in (None, filesystem._PersistentLocalError("disk failed"),
                            OSError(errno.ENOSPC, "No space left on device"),
                            filesystem._PersistentProtocolError("checksum differs"),
                            filesystem._AdapterMetadataError("invalid manifest")):
            with self.subTest(local_error=local_error):
                abort_started = threading.Event()
                release_abort = threading.Event()
                returned = threading.Event()
                errors = []

                class Socket:
                    def shutdown(self, _how):
                        abort_started.set()
                        release_abort.wait(1)

                def operation():
                    try:
                        with filesystem._adapter_socket_deadline(Socket(), time.monotonic() + 0.02, None):
                            abort_started.wait(1)
                            if local_error is not None:
                                raise local_error
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        returned.set()

                caller = threading.Thread(target=operation)
                caller.start()
                try:
                    self.assertTrue(abort_started.wait(1))
                    self.assertFalse(returned.wait(0.02))
                finally:
                    release_abort.set()
                    caller.join(1)
                self.assertTrue(returned.is_set())
                self.assertEqual(len(errors), 1)
                if local_error is None:
                    self.assertIsInstance(errors[0], TimeoutError)
                else:
                    self.assertIs(errors[0], local_error)
                self.assert_no_socket_guards()

    def test_retained_prefix_progress_is_counted_once_and_rolled_back(self):
        payload = b"resume-body"
        for retained in (payload[:3], b"bad", payload, b"x" * len(payload), payload + b"extra"):
            with self.subTest(retained=retained), tempfile.TemporaryDirectory() as temporary:
                filesystem = load_filesystem_module()
                filesystem.lane_rotation_delay_seconds = 0
                root = Path(temporary)
                destination = root / "chunk"
                destination.write_bytes(retained)
                changes, starts = [], []

                def read_range(_connection, _source, **kwargs):
                    start = kwargs["start"]
                    starts.append(start)
                    if retained == payload[:3] and len(starts) == 1:
                        kwargs["consume"](payload[start:start + 2])
                        raise filesystem._PersistentTransportError("connection lost")
                    kwargs["consume"](payload[start:])

                pool = filesystem._PersistentAdapterLanePool(1)
                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    with mock.patch.object(filesystem._PersistentAdapterConnection, "read_range", read_range):
                        pool.download(
                            ["http://lane0/chunk", "http://lane1/chunk"], 0,
                            destination, parent_fd, expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            progress=lambda _lane, amount: changes.append(amount),
                            per_request_timeout_seconds=1,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(sum(changes), len(payload))
                self.assertTrue(all(0 <= sum(changes[:index]) <= len(payload)
                                    for index in range(1, len(changes) + 1)))
                if retained == payload[:3]:
                    self.assertEqual(starts, [3, 5])
                    self.assertEqual(changes, [3, 2, len(payload) - 5])
                if retained == b"bad":
                    self.assertEqual(starts, [3, 0])
                    self.assertIn(-len(payload), changes)
                if retained == payload:
                    self.assertEqual(starts, [])

    def test_persistent_attempt_deadline_is_bounded_by_chunk_and_stage(self):
        for remaining in (None, 0.5):
            with self.subTest(remaining=remaining), tempfile.TemporaryDirectory() as temporary:
                filesystem = load_filesystem_module()
                root = Path(temporary)
                started = time.monotonic()
                deadline = None if remaining is None else started + remaining
                calls = []

                def read_range(_connection, _source, **kwargs):
                    calls.append(kwargs)
                    kwargs["consume"](b"x")

                pool = filesystem._PersistentAdapterLanePool(1)
                parent_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    with mock.patch.object(filesystem, "_configured_chunk_timeout", return_value=1), mock.patch.object(
                        filesystem._PersistentAdapterConnection, "read_range", read_range
                    ):
                        pool.download(
                            ["http://lane/chunk"], 0, root / "chunk", parent_fd,
                            expected_size=1, expected_sha256=hashlib.sha256(b"x").hexdigest(),
                            per_request_timeout_seconds=5, deadline=deadline,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual(calls[0]["stage_deadline"], deadline)
                if deadline is not None:
                    self.assertEqual(calls[0]["attempt_deadline"], deadline)
                else:
                    self.assertGreaterEqual(calls[0]["attempt_deadline"], started + 1)
                    self.assertLessEqual(calls[0]["attempt_deadline"], time.monotonic() + 1)

    @staticmethod
    def _metadata_response(body, *, declared=None, chunked=False):
        class Socket:
            def makefile(self, *_args):
                if chunked:
                    header = b"Transfer-Encoding: chunked\r\n"
                    wire_body = b"d\r\n" + body
                else:
                    size = len(body) if declared is None else declared
                    header = f"Content-Length: {size}\r\n".encode()
                    wire_body = body
                return io.BytesIO(b"HTTP/1.1 200 OK\r\n" + header + b"\r\n" + wire_body)

        response = http.client.HTTPResponse(Socket())
        response.begin()
        return response

    def test_short_metadata_retries_the_next_lane(self):
        filesystem = load_filesystem_module()
        payload = b'{"version":2}'
        urls = ["http://lane0/metadata", "http://lane1/metadata"]
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                responses = [self._metadata_response(payload[:5], declared=len(payload), chunked=chunked),
                             self._metadata_response(payload)]
                with mock.patch.object(filesystem, "_metadata_source_urls", return_value=urls), mock.patch.object(
                    filesystem, "_adapter_http_open", side_effect=responses
                ) as request:
                    self.assertEqual(filesystem._read_source_bytes(urls[0], max_bytes=1024), payload)
                self.assertEqual([call.args[0] for call in request.call_args_list], urls)

    def test_metadata_failures_do_not_preserve_a_partial_without_identity(self):
        for failure in ("short", "chunked", "reset", "json", "manifest"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                filesystem = load_filesystem_module()
                source = "http://source/run/step_1"
                calls = []

                def request(url, timeout, **_kwargs):
                    calls.append(url)
                    if failure == "reset":
                        raise ConnectionResetError("connection reset by peer")
                    body = b'{"version":9}' if failure == "manifest" else b'{"ver'
                    return self._metadata_response(
                        body, declared=13 if failure == "short" else len(body),
                        chunked=failure == "chunked",
                    )

                with mock.patch.dict(os.environ, {"PRIME_LORA_CACHE_DIR": temporary}), mock.patch.object(
                    filesystem, "_metadata_source_urls", return_value=["http://lane0/meta", "http://lane1/meta"]
                ), mock.patch.object(filesystem, "_adapter_http_open", side_effect=request):
                    destination = filesystem.adapter_cache_path(source)
                    partial = destination.with_name(destination.name + ".partial")
                    partial.mkdir()
                    sentinel = partial / "retained-chunk"
                    sentinel.write_bytes(b"retained")
                    network_failure = failure in {"short", "chunked", "reset"}
                    exception = filesystem._PersistentTransportError if network_failure else (ValueError, OSError)
                    with self.assertRaises(exception):
                        filesystem.stage_lora_adapter_files(source, require_delta=True)
                    self.assertEqual(len(calls), 2 if network_failure else 1)
                    self.assertFalse(partial.exists())

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

    def test_persistent_pool_closes_reused_connection_after_success(self):
        filesystem = load_filesystem_module()
        payload = b"close-after-success"
        server, thread, _handler = self._persistent_server(filesystem, payload)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                chunks = Path(temporary) / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                connection = pool._lanes[0][1]
                try:
                    url = f"http://127.0.0.1:{server.server_address[1]}"
                    for name in ("one", "two"):
                        pool.download(
                            [f"{url}/{name}"],
                            0,
                            chunks / name,
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            per_request_timeout_seconds=2,
                        )
                    self.assertIsNotNone(connection._connection)
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertIsNone(connection._connection)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_pool_closes_connection_after_error(self):
        filesystem = load_filesystem_module()
        payload = b"close-after-error"

        def bad_identity(request, number):
            if number != 1:
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(payload)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(payload) - 1}/{len(payload)}"
            )
            request.send_header("ETag", '"wrong"')
            request.end_headers()
            request.wfile.write(payload)
            request.close_connection = True
            return True

        server, thread, _handler = self._persistent_server(
            filesystem, payload, bad_identity
        )
        try:
            with tempfile.TemporaryDirectory() as temporary:
                chunks = Path(temporary) / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                connection = pool._lanes[0][1]
                try:
                    with self.assertRaisesRegex(OSError, "ETag"):
                        pool.download(
                            [f"http://127.0.0.1:{server.server_address[1]}/bad"],
                            0,
                            chunks / "bad",
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=hashlib.sha256(payload).hexdigest(),
                            per_request_timeout_seconds=2,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertIsNone(connection._connection)
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

    def test_persistent_chunk_rotates_and_resumes_continuous_trickle(self):
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
                self.assertEqual(handler.requests, 2)
                self.assertEqual(handler.paths, ["/slow", "/fast"])
                resumed = int(handler.ranges[1].split("=")[1].split("-")[0])
                self.assertGreater(resumed, 0)
                self.assertLess(resumed, len(payload))
                self.assertGreaterEqual(elapsed, 0.2)
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

    def test_persistent_chunk_exhausted_http_retry_is_typed_transport_failure(self):
        filesystem = load_filesystem_module()
        payload = b"status-exhausted"

        def always_unavailable(request, _number):
            request.send_response(404)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, always_unavailable
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem.lane_rotation_delay_seconds = 0
        filesystem._ADAPTER_LANE_RETRIES = 3
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(
                    filesystem._PersistentTransportError,
                    "adapter persistent download retries exhausted",
                ) as raised:
                    self._persistent_download(
                        filesystem, server, Path(temporary), "status", payload
                    )
                self.assertIsInstance(raised.exception.__cause__, filesystem.HTTPError)
                self.assertEqual(raised.exception.__cause__.code, 404)
                self.assertEqual(handler.requests, 3)
        finally:
            filesystem._ADAPTER_LANE_RETRIES = old_retries
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_typed_range_retry_exhaustion_wraps_only_retryable_http(self):
        filesystem = load_filesystem_module()
        payload = b"range-status"
        old_sleep = filesystem.time.sleep
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem.time.sleep = lambda _seconds: None
        filesystem._ADAPTER_LANE_RETRIES = 3
        try:
            for status in (503, 400):
                def always_status(request, _number, status=status):
                    request.send_response(status)
                    request.send_header("Content-Length", "0")
                    request.send_header("Connection", "close")
                    request.end_headers()
                    request.close_connection = True
                    return True

                server, thread, handler = self._persistent_server(
                    filesystem, payload, always_status
                )
                try:
                    with tempfile.TemporaryDirectory() as temporary:
                        chunks = Path(temporary) / "chunks"
                        chunks.mkdir()
                        parent_fd = filesystem._open_directory_nofollow(
                            chunks, create=False, label="test chunks"
                        )
                        try:
                            expected = (
                                filesystem._PersistentTransportError
                                if status == 503
                                else filesystem.HTTPError
                            )
                            with self.subTest(status=status), self.assertRaises(expected) as raised:
                                filesystem._download_range(
                                    [
                                        f"http://127.0.0.1:{server.server_address[1]}/range"
                                    ],
                                    0,
                                    chunks / "part",
                                    parent_fd,
                                    0,
                                    len(payload) - 1,
                                    typed_transport=True,
                                    per_request_timeout_seconds=1,
                                )
                            if status == 503:
                                self.assertEqual(
                                    str(raised.exception),
                                    "adapter persistent download retries exhausted",
                                )
                                self.assertIsInstance(
                                    raised.exception.__cause__, filesystem.HTTPError
                                )
                                self.assertEqual(raised.exception.__cause__.code, status)
                            else:
                                self.assertEqual(raised.exception.code, status)
                            self.assertEqual(handler.requests, 3 if status == 503 else 1)
                        finally:
                            os.close(parent_fd)
                finally:
                    server.shutdown()
                    thread.join(timeout=5)
                    server.server_close()
        finally:
            filesystem.time.sleep = old_sleep
            filesystem._ADAPTER_LANE_RETRIES = old_retries

    def test_typed_sequential_retry_exhaustion_wraps_only_retryable_http(self):
        filesystem = load_filesystem_module()
        old_urlopen = filesystem._adapter_http_open
        old_sleep = filesystem.time.sleep
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem.time.sleep = lambda _seconds: None
        filesystem._ADAPTER_LANE_RETRIES = 3
        calls = []

        def always_status(request, timeout, status):
            calls.append((request.full_url, timeout))
            raise filesystem.HTTPError(request.full_url, status, "test", None, None)

        try:
            for status in (503, 400):
                calls.clear()
                filesystem._adapter_http_open = lambda request, timeout, status=status, **_kwargs: always_status(
                    request, timeout, status
                )
                with tempfile.TemporaryDirectory() as temporary:
                    destination = Path(temporary) / "adapter"
                    parent_fd = filesystem._open_directory_nofollow(
                        Path(temporary), create=False, label="test download parent"
                    )
                    try:
                        expected = (
                            filesystem._PersistentTransportError
                            if status == 503
                            else filesystem.HTTPError
                        )
                        with self.subTest(status=status), self.assertRaises(expected) as raised:
                            filesystem._download_with_parent(
                                "http://lane/adapter",
                                destination,
                                expected_size=len(b"data"),
                                expected_sha256=hashlib.sha256(b"data").hexdigest(),
                                source_urls=["http://lane/adapter"],
                                parallel_ranges=False,
                                per_request_timeout_seconds=1,
                                typed_transport=True,
                                parent_fd=parent_fd,
                            )
                        if status == 503:
                            self.assertEqual(
                                str(raised.exception),
                                "adapter persistent download retries exhausted",
                            )
                            self.assertIsInstance(
                                raised.exception.__cause__, filesystem.HTTPError
                            )
                            self.assertEqual(raised.exception.__cause__.code, status)
                        else:
                            self.assertEqual(raised.exception.code, status)
                        self.assertEqual(len(calls), 3 if status == 503 else 1)
                    finally:
                        os.close(parent_fd)
        finally:
            filesystem._adapter_http_open = old_urlopen
            filesystem.time.sleep = old_sleep
            filesystem._ADAPTER_LANE_RETRIES = old_retries

    def test_typed_sequential_transport_exhaustion_binds_original_error(self):
        filesystem = load_filesystem_module()
        old_urlopen = filesystem._adapter_http_open
        old_sleep = filesystem.time.sleep
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem.time.sleep = lambda _seconds: None
        filesystem._ADAPTER_LANE_RETRIES = 3
        calls = []

        def always_transport_failure(_request, **_kwargs):
            calls.append(True)
            raise filesystem._PersistentTransportError(
                "adapter network request failed"
            )

        filesystem._adapter_http_open = always_transport_failure
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "adapter"
                parent_fd = filesystem._open_directory_nofollow(
                    Path(temporary), create=False, label="test download parent"
                )
                try:
                    with self.assertRaisesRegex(
                        filesystem._PersistentTransportError,
                        "adapter persistent download retries exhausted",
                    ) as raised:
                        filesystem._download_with_parent(
                            "http://lane/adapter",
                            destination,
                            expected_size=4,
                            expected_sha256=hashlib.sha256(b"data").hexdigest(),
                            source_urls=["http://lane/adapter"],
                            parallel_ranges=False,
                            per_request_timeout_seconds=1,
                            typed_transport=True,
                            parent_fd=parent_fd,
                        )
                    self.assertIsInstance(
                        raised.exception.__cause__,
                        filesystem._PersistentTransportError,
                    )
                    self.assertEqual(len(calls), 3)
                finally:
                    os.close(parent_fd)
        finally:
            filesystem._adapter_http_open = old_urlopen
            filesystem.time.sleep = old_sleep
            filesystem._ADAPTER_LANE_RETRIES = old_retries

    def test_typed_range_transport_exhaustion_binds_original_error(self):
        filesystem = load_filesystem_module()
        old_urlopen = filesystem._adapter_http_open
        old_sleep = filesystem.time.sleep
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem.time.sleep = lambda _seconds: None
        filesystem._ADAPTER_LANE_RETRIES = 3
        calls = []

        def always_transport_failure(_request, **_kwargs):
            calls.append(True)
            raise OSError("socket failure")

        filesystem._adapter_http_open = always_transport_failure
        try:
            with tempfile.TemporaryDirectory() as temporary:
                chunks = Path(temporary) / "chunks"
                chunks.mkdir()
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                try:
                    with self.assertRaisesRegex(
                        filesystem._PersistentTransportError,
                        "adapter persistent download retries exhausted",
                    ) as raised:
                        filesystem._download_range(
                            ["http://lane/adapter"],
                            0,
                            chunks / "part",
                            parent_fd,
                            0,
                            3,
                            typed_transport=True,
                            per_request_timeout_seconds=1,
                        )
                    self.assertIsInstance(
                        raised.exception.__cause__,
                        filesystem._PersistentTransportError,
                    )
                    self.assertEqual(len(calls), 3)
                finally:
                    os.close(parent_fd)
        finally:
            filesystem._adapter_http_open = old_urlopen
            filesystem.time.sleep = old_sleep
            filesystem._ADAPTER_LANE_RETRIES = old_retries

    def test_persistent_chunk_exhausted_timeout_is_typed_transport_failure(self):
        filesystem = load_filesystem_module()
        old_read_range = filesystem._PersistentAdapterConnection.read_range
        old_sleep = filesystem.time.sleep
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem._ADAPTER_LANE_RETRIES = 3
        filesystem.time.sleep = lambda _seconds: None

        def always_timeout(_connection, _source_url, **_kwargs):
            raise TimeoutError("socket timed out")

        filesystem._PersistentAdapterConnection.read_range = always_timeout
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
                    with self.assertRaisesRegex(
                        filesystem._PersistentTransportError,
                        "adapter persistent download retries exhausted",
                    ) as raised:
                        pool.download(
                            ["http://source/chunk"],
                            0,
                            chunks / "chunk",
                            parent_fd,
                            expected_size=4,
                            expected_sha256=hashlib.sha256(b"data").hexdigest(),
                            per_request_timeout_seconds=1,
                        )
                    self.assertIsInstance(raised.exception.__cause__, TimeoutError)
                finally:
                    pool.close()
                    os.close(parent_fd)
        finally:
            filesystem._PersistentAdapterConnection.read_range = old_read_range
            filesystem.time.sleep = old_sleep
            filesystem._ADAPTER_LANE_RETRIES = old_retries

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
                self.assertGreaterEqual(handler.requests, 2)
                self.assertLessEqual(handler.requests, filesystem._ADAPTER_LANE_RETRIES)
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_config_waits_for_relay_until_stage_deadline(self):
        filesystem = load_filesystem_module()
        payload = b"late-relay-config"

        def unavailable_five_times(request, number):
            if number > 5:
                return False
            request.send_response(404)
            request.send_header("Content-Length", "0")
            request.send_header("Connection", "close")
            request.end_headers()
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, unavailable_five_times
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem.lane_rotation_delay_seconds = 0.01
        filesystem._ADAPTER_LANE_RETRIES = 1
        try:
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "adapter_config.json"
                filesystem._download_persistent_file(
                    [
                        f"http://127.0.0.1:{server.server_address[1]}/adapter_config.json",
                        f"http://127.0.0.1:{server.server_address[1]}/adapter_config.json",
                    ],
                    destination,
                    expected_size=len(payload),
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                    deadline=time.monotonic() + 1,
                    per_request_timeout_seconds=0.1,
                )
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(handler.requests, 6)
        finally:
            filesystem._ADAPTER_LANE_RETRIES = old_retries
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
                self.assertGreaterEqual(handler.requests, 2)
                self.assertLessEqual(handler.requests, filesystem._ADAPTER_LANE_RETRIES)
                self.assertEqual(ranges[0], f"bytes=0-{len(payload) - 1}")
                self.assertTrue(
                    all(value == f"bytes=4-{len(payload) - 1}" for value in ranges[1:])
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

    def test_persistent_chunk_rotates_after_checksum_mismatch(self):
        filesystem = load_filesystem_module()
        payload = b"checksum-good"
        corrupt = b"checksum-bad!"
        digest = hashlib.sha256(payload).hexdigest()
        progress_changes = []

        def corrupt_primary(request, _number):
            if request.path != "/lane0/chunk":
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(corrupt)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(corrupt) - 1}/{len(payload)}"
            )
            request.send_header("ETag", f'"{digest}"')
            request.end_headers()
            request.wfile.write(corrupt)
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, corrupt_primary
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
                    for index in range(2)
                ]
                pool = filesystem._PersistentAdapterLanePool(urls, width=1)
                try:
                    pool.download(
                        pool.urls_for_lane(0, "chunk"),
                        0,
                        chunks / "chunk",
                        parent_fd,
                        expected_size=len(payload),
                        expected_sha256=digest,
                        progress=lambda _lane, amount: progress_changes.append(amount),
                        per_request_timeout_seconds=1,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual((chunks / "chunk").read_bytes(), payload)
                self.assertEqual(sum(progress_changes), len(payload))
                self.assertIn(-len(payload), progress_changes)
                self.assertTrue(all(
                    0 <= sum(progress_changes[:index]) <= len(payload)
                    for index in range(1, len(progress_changes) + 1)
                ))
                self.assertEqual(
                    handler.paths[:2], ["/lane0/chunk", "/lane1/chunk"]
                )
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_checks_each_lane_before_checksum_retries(self):
        filesystem = load_filesystem_module()
        payload = b"checksum-good"
        corrupt = b"checksum-bad!"
        digest = hashlib.sha256(payload).hexdigest()

        def only_lane_one_is_good(request, _number):
            if request.path == "/lane1/chunk":
                return False
            if not request.path.startswith("/lane"):
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(corrupt)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(corrupt) - 1}/{len(payload)}"
            )
            request.send_header("ETag", f'"{digest}"')
            request.end_headers()
            request.wfile.write(corrupt)
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, only_lane_one_is_good
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
                    for index in range(12)
                ]
                pool = filesystem._PersistentAdapterLanePool(urls, width=1)
                try:
                    pool.download(
                        pool.urls_for_lane(0, "chunk"),
                        0,
                        chunks / "chunk",
                        parent_fd,
                        expected_size=len(payload),
                        expected_sha256=digest,
                        per_request_timeout_seconds=1,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual((chunks / "chunk").read_bytes(), payload)
                self.assertEqual(
                    handler.paths[:2], ["/lane0/chunk", "/lane1/chunk"]
                )
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_resets_corrupt_local_prefix(self):
        filesystem = load_filesystem_module()
        payload = b"prefix-recovery"
        digest = hashlib.sha256(payload).hexdigest()
        ranges = []

        def record_range(request, _number):
            ranges.append(request.headers.get("Range"))
            return False

        server, thread, _handler = self._persistent_server(
            filesystem, payload, record_range
        )
        old_delay = filesystem.lane_rotation_delay_seconds
        filesystem.lane_rotation_delay_seconds = 0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                chunks = root / "chunks"
                chunks.mkdir()
                destination = chunks / "chunk"
                destination.write_bytes(b"bad-")
                parent_fd = filesystem._open_directory_nofollow(
                    chunks, create=False, label="test chunks"
                )
                pool = filesystem._PersistentAdapterLanePool(1)
                try:
                    pool.download(
                        [f"http://127.0.0.1:{server.server_address[1]}/chunk"],
                        0,
                        destination,
                        parent_fd,
                        expected_size=len(payload),
                        expected_sha256=digest,
                        per_request_timeout_seconds=1,
                    )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertEqual(destination.read_bytes(), payload)
                self.assertEqual(
                    ranges[:2],
                    [f"bytes=4-{len(payload) - 1}", f"bytes=0-{len(payload) - 1}"],
                )
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_persistent_chunk_keeps_checksum_error_when_all_lanes_corrupt(self):
        filesystem = load_filesystem_module()
        payload = b"checksum-target"
        corrupt = b"checksum-source"
        digest = hashlib.sha256(payload).hexdigest()

        def corrupt_every_lane(request, _number):
            if not request.path.startswith(("/lane0/", "/lane1/")):
                return False
            request.send_response(206)
            request.send_header("Content-Length", str(len(corrupt)))
            request.send_header(
                "Content-Range", f"bytes 0-{len(corrupt) - 1}/{len(payload)}"
            )
            request.send_header("ETag", f'"{digest}"')
            request.end_headers()
            request.wfile.write(corrupt)
            request.close_connection = True
            return True

        server, thread, handler = self._persistent_server(
            filesystem, payload, corrupt_every_lane
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
                    for index in range(2)
                ]
                pool = filesystem._PersistentAdapterLanePool(urls, width=1)
                try:
                    with self.assertRaisesRegex(OSError, "checksum"):
                        pool.download(
                            pool.urls_for_lane(0, "chunk"),
                            0,
                            chunks / "chunk",
                            parent_fd,
                            expected_size=len(payload),
                            expected_sha256=digest,
                            per_request_timeout_seconds=1,
                        )
                finally:
                    pool.close()
                    os.close(parent_fd)
                self.assertFalse((chunks / "chunk").exists())
                self.assertEqual(handler.requests, 2)
                self.assertEqual(handler.paths, ["/lane0/chunk", "/lane1/chunk"])
        finally:
            filesystem.lane_rotation_delay_seconds = old_delay
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

        def worker(index, _item, _lane):
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

        def worker(_index, _item, _lane):
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

    def test_sequential_download_survives_parent_swap_at_output_open(self):
        filesystem = load_filesystem_module()
        payload = b"safe"

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(_request, timeout, **_kwargs):
            self.assertGreater(timeout, 0)
            return Response(payload)

        old_urlopen = filesystem._adapter_http_open
        old_open = filesystem.os.open
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        old_urlopen = filesystem._adapter_http_open
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
        self.assertEqual(calls, ["bytes=2-"])

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

        def fake_urlopen(request, timeout, **_kwargs):
            url = request.full_url if hasattr(request, "full_url") else request
            calls.append((url, timeout))
            if url.startswith("http://lane-1/"):
                raise TimeoutError("stalled")
            return Response(b"safe")

        old_urlopen = filesystem._adapter_http_open
        old_lanes = filesystem._adapter_source_urls
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        old_urlopen = filesystem._adapter_http_open
        old_lanes = filesystem._adapter_source_urls
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        old_urlopen = filesystem._adapter_http_open
        old_lanes = filesystem._adapter_source_urls
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
            filesystem._adapter_source_urls = old_lanes

        self.assertEqual(calls, ["http://lane-1/manifest"])

    def test_metadata_read_does_not_retry_identity_error(self):
        filesystem = load_filesystem_module()
        calls = []

        def fake_urlopen(request, **_kwargs):
            calls.append(request.full_url if hasattr(request, "full_url") else request)
            raise OSError("adapter identity differs")

        old_urlopen = filesystem._adapter_http_open
        old_lanes = filesystem._adapter_source_urls
        filesystem._adapter_http_open = fake_urlopen
        filesystem._adapter_source_urls = lambda _source: [
            "http://lane-1/manifest",
            "http://lane-2/manifest",
        ]
        try:
            with self.assertRaisesRegex(OSError, "identity differs"):
                filesystem._read_source_bytes("http://lane-1/manifest")
        finally:
            filesystem._adapter_http_open = old_urlopen
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

        old_urlopen = filesystem._adapter_http_open
        old_lanes = filesystem._adapter_source_urls
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
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

        old_urlopen = filesystem._adapter_http_open
        old_executor = filesystem.ThreadPoolExecutor
        old_streams = filesystem._configured_download_streams
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
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

        old_urlopen = filesystem._adapter_http_open
        old_executor = filesystem.ThreadPoolExecutor
        old_threshold = filesystem._PARALLEL_DOWNLOAD_THRESHOLD
        old_lanes = filesystem._adapter_source_urls
        old_streams = filesystem._configured_download_streams
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        old_urlopen = filesystem._adapter_http_open
        old_threshold = filesystem._PARALLEL_DOWNLOAD_THRESHOLD
        old_streams = filesystem._configured_download_streams
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
            calls.append((request.full_url, timeout))
            if len(calls) == 1:
                raise TimeoutError("lane read timed out")
            return Response(payload)

        old_urlopen = filesystem._adapter_http_open
        old_sleep = filesystem.time.sleep
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
            calls.append((request.full_url, request.headers.get("Range"), timeout))
            if len(calls) == 1:
                return SlowResponse()
            return FastResponse(payload[3:])

        old_urlopen = filesystem._adapter_http_open
        old_monotonic = filesystem.time.monotonic
        old_sleep = filesystem.time.sleep
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
            calls.append((request.full_url, request.headers.get("Range"), timeout))
            if len(calls) == 1:
                return SlowResponse()
            return FastResponse(payload[3:])

        old_urlopen = filesystem._adapter_http_open
        old_monotonic = filesystem.time.monotonic
        old_sleep = filesystem.time.sleep
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
            calls.append((request.full_url, timeout))
            if len(calls) < len(source_urls):
                raise TimeoutError("lane read timed out")
            return Response(payload)

        old_urlopen = filesystem._adapter_http_open
        old_sleep = filesystem.time.sleep
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        def fake_urlopen(request, timeout, **_kwargs):
            calls.append((request.full_url, timeout))
            if len(calls) <= 70:
                raise filesystem.HTTPError(request.full_url, 503, "not ready", None, None)
            return Response(payload)

        old_urlopen = filesystem._adapter_http_open
        old_monotonic = filesystem.time.monotonic
        old_sleep = filesystem.time.sleep
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
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

        old_urlopen = filesystem._adapter_http_open
        old_sleep = filesystem.time.sleep
        filesystem._adapter_http_open = fake_urlopen
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
            filesystem._adapter_http_open = old_urlopen
            filesystem.time.sleep = old_sleep

        self.assertEqual(calls, [source_urls[23], source_urls[24]])
        self.assertEqual(
            [int(url.split(":")[2].split("/", 1)[0]) for url in calls],
            [18146, 18148],
        )


if __name__ == "__main__":
    unittest.main()
