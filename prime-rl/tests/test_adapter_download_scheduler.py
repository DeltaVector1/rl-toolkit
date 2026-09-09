import hashlib
import inspect
import importlib.util
import io
import json
import sys
import tempfile
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
    def test_download_keeps_public_signature(self):
        filesystem = load_filesystem_module()
        self.assertEqual(
            str(inspect.signature(filesystem.download)),
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
        filesystem.time.sleep = lambda _seconds: None
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
