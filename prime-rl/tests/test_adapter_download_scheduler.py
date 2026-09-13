import hashlib
import inspect
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from urllib.parse import urlsplit


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
    def test_bundle_identity_is_explicit(self):
        filesystem = load_filesystem_module()
        with self.assertRaisesRegex(OSError, "not published"):
            filesystem._bundle_identity(
                "http://source/step_1",
                {"source_manifest": {}, "chunks": []},
            )

    def test_bundle_identity_preserves_query(self):
        filesystem = load_filesystem_module()
        payload = b"bundle-wire"
        item = {"wire_size": len(payload)}
        transport = {
            "source_manifest": {
                "bundle": {
                    "path": ".adapter-transport-bundle",
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            },
            "chunks": [item],
        }
        bundle = filesystem._bundle_identity(
            "http://source/step_1?token=abc",
            transport,
        )
        self.assertEqual(
            bundle["url"],
            "http://source/step_1/.adapter-transport-bundle?token=abc",
        )

    def test_bundle_lane_plan_uses_first_64_sources_and_exact_ranges(self):
        filesystem = load_filesystem_module()
        ports = list(range(18100, 18164, 2)) + list(range(18101, 18165, 2))
        source_urls = [f"http://source:{port}/step_1?token=abc" for port in ports]
        chunks = []
        bundle_offset = 0
        for index in range(1834):
            wire_size = 100 + (index % 17)
            chunks.append(
                {
                    "file": f"chunk-{index:04d}",
                    "offset": index * 4096,
                    "bundle_offset": bundle_offset,
                    "wire_size": wire_size,
                }
            )
            bundle_offset += wire_size
        plan = filesystem._bundle_lane_plan(
            {"size": bundle_offset},
            chunks,
            source_urls,
        )
        self.assertEqual(plan["lane_count"], 64)
        self.assertEqual(len(plan["lanes"]), 64)
        self.assertEqual(
            [urlsplit(url).port for url in (lane["url"] for lane in plan["lanes"])],
            ports[:64],
        )
        self.assertTrue(all("?token=abc" in lane["url"] for lane in plan["lanes"]))
        planned_files = [name for lane in plan["lanes"] for name in lane["chunks"]]
        self.assertEqual(planned_files, [item["file"] for item in chunks])
        expected_start = 0
        for lane in plan["lanes"]:
            self.assertEqual(lane["start"], expected_start)
            self.assertEqual(lane["end"] - lane["start"] + 1, lane["wire_bytes"])
            expected_start = lane["end"] + 1
        self.assertEqual(expected_start, bundle_offset)

    def test_bundle_lane_reset_resumes_and_preserves_hashes(self):
        filesystem = load_filesystem_module()
        pieces = [b"wire-a", b"wire-bb", b"wire-ccc"]
        payload = b"".join(pieces)
        chunks = []
        offset = 0
        raw_offset = 0
        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "file": f"chunk-{index}",
                    "offset": raw_offset,
                    "bundle_offset": offset,
                    "raw_size": len(piece),
                    "wire_size": len(piece),
                    "wire_sha256": hashlib.sha256(piece).hexdigest(),
                    "raw_sha256": hashlib.sha256(piece).hexdigest(),
                }
            )
            offset += len(piece)
            raw_offset += len(piece)
        bundle = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        lane = {
            "url": "http://source:18100/.adapter-transport-bundle?token=abc",
            "start": 0,
            "end": len(payload) - 1,
            "wire_bytes": len(payload),
            "chunks": [item["file"] for item in chunks],
        }
        calls = []

        class Response(io.BytesIO):
            status = 206

            def __init__(self, value, start, end, reset=False):
                super().__init__(value)
                self._reset = reset
                self._returned = False
                self.headers = {
                    "Content-Length": str(end - start + 1),
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "ETag": f'"{hashlib.sha256(payload).hexdigest()}"',
                }

            def read(self, amount=-1):
                if self._reset and not self._returned:
                    self._returned = True
                    return super().read(3)
                if self._reset:
                    raise filesystem.URLError("connection reset")
                return super().read(amount)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            del timeout
            start, end = (
                int(part)
                for part in request.headers["Range"].removeprefix("bytes=").split("-")
            )
            calls.append((request.full_url, start, end))
            return Response(payload[start : end + 1], start, end, reset=len(calls) == 1)

        def is_completed(_index, item, path):
            return path.with_name(f"{path.name}.done").exists()

        def process_chunk(_index, item):
            path = root / item["file"]
            encoded = path.read_bytes()
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), item["wire_sha256"])
            path.with_name(f"{path.name}.done").write_text(
                json.dumps({"output_sha256": hashlib.sha256(encoded).hexdigest()})
            )

        old_urlopen = filesystem.urllib.request.urlopen
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.time.sleep = lambda _seconds: None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                filesystem._download_bundle_lane(
                    bundle,
                    lane,
                    chunks,
                    root,
                    is_completed=is_completed,
                    process_chunk=process_chunk,
                )
                self.assertEqual(
                    b"".join((root / item["file"]).read_bytes() for item in chunks),
                    payload,
                )
                filesystem._verify_bundle_chunks(bundle, chunks, root)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.time.sleep = old_sleep
        self.assertEqual(calls[0][1:], (0, len(payload) - 1))
        self.assertEqual(calls[1][1:], (3, len(payload) - 1))
        self.assertEqual(calls[0][0], calls[1][0])

    def test_bundle_lanes_open_one_distinct_range_each(self):
        filesystem = load_filesystem_module()
        payload = bytes(range(64))
        source_urls = [f"http://source:{18100 + index}/step_1?token=abc" for index in range(64)]
        chunks = [
            {
                "file": f"chunk-{index}",
                "offset": index,
                "bundle_offset": index,
                "wire_size": 1,
                "wire_sha256": hashlib.sha256(bytes([index])).hexdigest(),
                "raw_sha256": hashlib.sha256(bytes([index])).hexdigest(),
            }
            for index in range(64)
        ]
        bundle = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        plan = filesystem._bundle_lane_plan(bundle, chunks, source_urls)
        calls = []

        class Response(io.BytesIO):
            status = 206

            def __init__(self, value, start, end):
                super().__init__(value)
                self.headers = {
                    "Content-Length": str(end - start + 1),
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "ETag": f'"{hashlib.sha256(payload).hexdigest()}"',
                }

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            del timeout
            start, end = (
                int(part)
                for part in request.headers["Range"].removeprefix("bytes=").split("-")
            )
            calls.append((request.full_url, start, end))
            return Response(payload[start : end + 1], start, end)

        def is_completed(_index, _item, path):
            return path.with_name(f"{path.name}.done").exists()

        def process_chunk(_index, item):
            path = root / item["file"]
            path.with_name(f"{path.name}.done").write_text(
                json.dumps({"output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            )

        old_urlopen = filesystem.urllib.request.urlopen
        filesystem.urllib.request.urlopen = fake_urlopen
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for lane_index, lane in enumerate(plan["lanes"]):
                    filesystem._download_bundle_lane(
                        bundle,
                        lane,
                        chunks,
                        root,
                        is_completed=is_completed,
                        process_chunk=process_chunk,
                        lane_index=lane_index,
                    )
                self.assertEqual([call[0] for call in calls], [lane["url"] for lane in plan["lanes"]])
                self.assertEqual([(call[1], call[2]) for call in calls], [(lane["start"], lane["end"]) for lane in plan["lanes"]])
                self.assertEqual(b"".join((root / item["file"]).read_bytes() for item in chunks), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen

    def test_bundle_lane_rejects_non_range_response(self):
        filesystem = load_filesystem_module()
        piece = b"wire"
        item = {
            "file": "chunk",
            "offset": 0,
            "bundle_offset": 0,
            "wire_size": len(piece),
            "wire_sha256": hashlib.sha256(piece).hexdigest(),
        }
        lane = {
            "url": "http://source:18100/.adapter-transport-bundle",
            "start": 0,
            "end": len(piece) - 1,
            "wire_bytes": len(piece),
            "chunks": ["chunk"],
        }

        class Response(io.BytesIO):
            status = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        calls = []
        old_urlopen = filesystem.urllib.request.urlopen
        filesystem.urllib.request.urlopen = lambda request, timeout: (calls.append(request), Response())[-1]
        try:
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(OSError, "did not honor"):
                    filesystem._download_bundle_lane(
                        {"size": len(piece), "sha256": hashlib.sha256(piece).hexdigest()},
                        lane,
                        [item],
                        Path(temporary),
                        is_completed=lambda *_args: False,
                        process_chunk=lambda *_args: None,
                    )
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
        self.assertEqual(len(calls), 1)

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
            filesystem.HTTPError("http://lane/manifest", 503, "starting", None, None),
        )
        terminal = (
            OSError("generic metadata failure"),
            OSError("connection reset by peer"),
            filesystem.URLError("adapter checksum mismatch"),
            filesystem.HTTPError("http://lane/manifest", 500, "server error", None, None),
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
                    retry_http_statuses={404, 503},
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
