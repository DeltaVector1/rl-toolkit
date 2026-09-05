import asyncio
import ast
import functools
import hashlib
import http.server
import importlib.util
import io
import json
import math
import os
import socketserver
import sys
import tempfile
import threading
import time
import types
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
MODULE_PATH = ROOT / "src/prime_rl/utils/adapter_delta.py"
SPEC = importlib.util.spec_from_file_location("adapter_delta", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter_delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter_delta)


def function_arguments(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    return [argument.arg for argument in function.args.args]


def load_watcher_delta_method(urlopen):
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"_read_adapter_manifest", "_adapter_delta_base"}
    ]
    namespace = {
        "json": json,
        "urlopen": urlopen,
        "_delta_stage_timeout_s": lambda: 30,
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "inference_weight_path": lambda _config, path, _lora_name: f"http://source/{path.name}",
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_adapter_delta_base"]


def load_manifest_reader(urlopen):
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    source = path.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_read_adapter_manifest"
    )
    namespace = {"json": json, "urlopen": urlopen}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_read_adapter_manifest"]


def load_manifest_prewarm_method():
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    source = path.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_prewarm_adapter_manifest"
    )
    namespace = {"asyncio": asyncio}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_prewarm_adapter_manifest"]


def load_broadcast_cleanup_method():
    path = ROOT / "src/prime_rl/trainer/rl/broadcast/filesystem.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "maybe_clean"
    )
    namespace = {
        "shutil": __import__("shutil"),
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "maybe_clean": lambda *_args: None,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["maybe_clean"]


def load_filesystem_module():
    path = ROOT / "src/prime_rl/inference/vllm/worker/filesystem.py"
    module_name = "filesystem_sidecar_test"
    modules = {
        "torch": types.ModuleType("torch"),
        "torch.nn": types.ModuleType("torch.nn"),
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": types.ModuleType("vllm.model_executor.model_loader"),
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.utils": types.ModuleType("prime_rl.utils"),
        "prime_rl.utils.checkpoint_delta": types.ModuleType("prime_rl.utils.checkpoint_delta"),
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
        spec = importlib.util.spec_from_file_location(module_name, path)
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


def load_server_stage_method(stage_worker, publish_lock=None):
    path = ROOT / "src/prime_rl/inference/vllm/server.py"
    source = path.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage_lora_adapter"
    )

    class FakeJSONResponse:
        def __init__(self, content, status_code):
            self.content = content
            self.status_code = status_code

    class FakeRouter:
        def post(self, *_args, **_kwargs):
            return lambda function: function

    namespace = {
        "asyncio": asyncio,
        "threading": threading,
        "time": time,
        "router": FakeRouter(),
        "Request": object,
        "JSONResponse": FakeJSONResponse,
        "_delta_stage_timeout_s": lambda: 0.01,
        "_adapter_stage_publish_lock": publish_lock or threading.Lock(),
        "_is_finite_number": lambda value: (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ),
        "_begin_lora_stage": lambda: None,
        "_end_lora_stage": lambda: None,
        "stage_lora_adapter_files": stage_worker,
        "adapter_stage_status": lambda: {},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["stage_lora_adapter"], FakeJSONResponse


class QuietDirectoryHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        return


class AdapterTransportStateTest(unittest.TestCase):
    def test_lora_broadcast_cleanup_does_not_keep_checkpoint_intervals(self):
        cleanup = load_broadcast_cleanup_method()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in range(7):
                (root / f"step_{step}").mkdir()
            manager = SimpleNamespace(
                used_idxs=[0],
                progress={0: SimpleNamespace(step=7)},
                get_run_dir=lambda _idx: root,
            )
            broadcast = SimpleNamespace(
                world=SimpleNamespace(is_master=True),
                multi_run_manager=manager,
                lora_config=object(),
                logger=SimpleNamespace(debug=lambda *_args: None),
            )
            cleanup(broadcast, 3)
            self.assertFalse((root / "step_3").exists())
            self.assertTrue((root / "step_4").exists())

    def test_exact_adjacent_delta_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "step_1"
            target = root / "step_2"
            base.mkdir()
            target.mkdir()
            target_size = 2 * adapter_delta.default_chunk_size + 2
            original = (bytes(range(251)) * ((target_size + 250) // 251))[:target_size]
            changed = bytearray(original)
            changed[0] = 17
            changed[4096] = 29
            for directory, payload in ((base, original), (target, changed)):
                (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
                (directory / "adapter_model.safetensors").write_bytes(payload)

            manifest = adapter_delta.publish_adapter_manifest(
                target,
                base,
                target_url="http://source/step_2",
                base_url="http://source/step_1",
                target_step=2,
                base_step=1,
                max_workers=2,
            )

            rebuilt = bytearray()
            for chunk in manifest["chunks"]:
                start = chunk["offset"]
                wire = (target / chunk["file"]).read_bytes()
                rebuilt.extend(
                    adapter_delta.decode_delta_chunk(
                        original[start : start + chunk["raw_size"]], wire, chunk["codec"]
                    )
                )
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["mode"], "delta")
            self.assertEqual(manifest["base_step"], 1)
            self.assertEqual(manifest["chunk_size"], adapter_delta.default_chunk_size)
            self.assertEqual(manifest["chunk_count"], 3)
            self.assertEqual(bytes(rebuilt), bytes(changed))
            self.assertEqual(manifest["target_sha256"], hashlib.sha256(changed).hexdigest())

    def test_watcher_reads_delta_manifest_from_source_when_local_sync_has_marker_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step_0").mkdir()
            (root / "step_1").mkdir()
            (root / "step_0" / "STABLE").touch()
            (root / "step_1" / "STABLE").touch()
            target_url = "http://source/step_1"
            manifest = {
                "version": 1,
                "kind": "lora-adapter",
                "mode": "delta",
                "transport": "chunked-shuffled-xor-zstd",
                "target_step": 1,
                "base_step": 0,
                "build_seconds": 2.5,
            }
            calls = []

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    self.close()

            def fake_urlopen(url, timeout):
                calls.append((url, timeout))
                return Response(json.dumps(manifest).encode("utf-8"))

            method = load_watcher_delta_method(fake_urlopen)
            watcher = SimpleNamespace(
                config=SimpleNamespace(
                    output_dir=root,
                    weight_broadcast=SimpleNamespace(),
                ),
                lora_name="adapter",
                verified_adapter_step=0,
                verified_adapter_url=None,
                last_adapter_delta_build_time=-1.0,
            )
            result = method(watcher, 1, root / "step_1")

            self.assertEqual(result, f"{target_url}/adapter_manifest.json")
            self.assertEqual(watcher.last_adapter_delta_build_time, 2.5)
            self.assertEqual(calls, [(f"{target_url}/adapter_manifest.json", 30)])

    def test_remote_manifest_is_used_when_local_manifest_is_old_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_manifest = {
                "version": 1,
                "transport": "chunked-shuffled-xor-zstd",
            }
            remote_manifest = {
                "version": 2,
                "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
            }
            (root / "adapter_manifest.json").write_text(
                json.dumps(local_manifest), encoding="utf-8"
            )
            calls = []

            class Response(io.BytesIO):
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    self.close()

            def fake_urlopen(url, timeout):
                calls.append((url, timeout))
                return Response(json.dumps(remote_manifest).encode("utf-8"))

            reader = load_manifest_reader(fake_urlopen)
            manifest, location = reader(
                root,
                "http://source/step_2",
                30,
                prefer_remote=True,
            )

            self.assertEqual(manifest, remote_manifest)
            self.assertEqual(location, "http://source/step_2/adapter_manifest.json")
            self.assertEqual(
                calls,
                [("http://source/step_2/adapter_manifest.json", 30)],
            )

    def test_manifest_prewarm_does_not_block_the_event_loop(self):
        prewarm = load_manifest_prewarm_method()
        started = threading.Event()
        release = threading.Event()

        def blocking_helper(*_args):
            started.set()
            release.wait(2)
            return "manifest-location"

        watcher = SimpleNamespace(_adapter_delta_base=blocking_helper)

        async def exercise():
            task = asyncio.create_task(prewarm(watcher, 2, Path("step_2"), "http://source/step_2"))
            await asyncio.to_thread(started.wait, 1)
            tick = asyncio.Event()
            asyncio.get_running_loop().call_later(0.01, tick.set)
            await asyncio.wait_for(tick.wait(), 0.5)
            release.set()
            return await task

        self.assertEqual(asyncio.run(exercise()), "manifest-location")

    def test_bootstrap_does_not_duplicate_the_full_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "step_0"
            target.mkdir()
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")
            (target / "adapter_model.safetensors").write_bytes(bytes(range(251)) * 33)
            manifest = adapter_delta.publish_adapter_manifest(target, target_step=0)
            self.assertEqual(manifest["mode"], "full")
            self.assertEqual(manifest["chunk_count"], 0)
            self.assertEqual(list(target.glob("*.transport-*")), [])

    def test_full_stage_publishes_model_checksum_sidecar(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "step_0"
            source.mkdir()
            config = b"{}\n"
            model = bytes(range(251)) * 33
            (source / "adapter_config.json").write_bytes(config)
            (source / "adapter_model.safetensors").write_bytes(model)
            manifest = {
                "version": 1,
                "kind": "lora-adapter",
                "mode": "full",
                "target_step": 0,
                "base_step": None,
                "files": [
                    {"name": "adapter_config.json", "size": len(config), "sha256": hashlib.sha256(config).hexdigest()},
                    {"name": "adapter_model.safetensors", "size": len(model), "sha256": hashlib.sha256(model).hexdigest()},
                ],
            }
            (source / "adapter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            handler = functools.partial(QuietDirectoryHandler, directory=str(root))
            with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as server:
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                source_url = f"http://127.0.0.1:{server.server_address[1]}/step_0"
                old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
                old_streams = os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS")
                os.environ["PRIME_LORA_CACHE_DIR"] = str(root / "cache")
                os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = "24"
                try:
                    destination = filesystem.stage_lora_adapter_files(source_url)
                finally:
                    if old_cache is None:
                        os.environ.pop("PRIME_LORA_CACHE_DIR", None)
                    else:
                        os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
                    if old_streams is None:
                        os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
                    else:
                        os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = old_streams
                    server.shutdown()
                    server_thread.join()
            sidecar = destination / "adapter_model.safetensors.sha256"
            self.assertEqual(sidecar.read_text(encoding="ascii").strip(), hashlib.sha256(model).hexdigest())
            self.assertTrue((destination / "adapter_manifest.json").is_file())
            relay_manifest = json.loads((destination / "relay_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(relay_manifest["transport"], "full")
            self.assertIn("adapter_model.safetensors", relay_manifest["files"])

    def test_missing_delta_base_uses_the_published_full_adapter(self):
        filesystem = load_filesystem_module()
        config = b"{}\n"
        model = b"full adapter"
        files = {
            "adapter_config.json": {
                "size": len(config),
                "sha256": hashlib.sha256(config).hexdigest(),
            },
            "adapter_model.safetensors": {
                "size": len(model),
                "sha256": hashlib.sha256(model).hexdigest(),
            },
        }
        source_manifest = {
            "version": 2,
            "kind": "lora-adapter",
            "mode": "delta",
            "files": [{"name": name, **record} for name, record in files.items()],
        }
        transport = {
            "mode": "delta",
            "transport": filesystem._ADAPTER_LINEAR_TRANSPORT,
            "source_manifest": source_manifest,
            "target_sha256": files["adapter_model.safetensors"]["sha256"],
            "target_size": len(model),
            "config_sha256": files["adapter_config.json"]["sha256"],
            "config_size": len(config),
        }
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_transport = filesystem._read_chunked_xor_manifest
        old_manifest = filesystem._source_manifest
        old_stage = filesystem._stage_chunked_linear_adapter
        old_download = filesystem.download
        old_timeout = filesystem._delta_stage_timeout_s

        def download(_source, destination, **_kwargs):
            destination.write_bytes(config if destination.name.endswith(".json") else model)

        filesystem._read_chunked_xor_manifest = lambda *_args, **_kwargs: transport
        filesystem._source_manifest = lambda *_args, **_kwargs: {
            "version": 1,
            "source_url": "http://source/step_4",
            "files": files,
        }
        filesystem._stage_chunked_linear_adapter = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("base is not cached")
        )
        filesystem.download = download
        filesystem._delta_stage_timeout_s = lambda: 30.0
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                destination = filesystem.stage_lora_adapter_files("http://source/step_4")
                self.assertEqual((destination / "adapter_model.safetensors").read_bytes(), model)
                relay = json.loads((destination / "relay_manifest.json").read_text())
                self.assertEqual(relay["transport"], "full")
                with self.assertRaises(FileNotFoundError):
                    filesystem.stage_lora_adapter_files(
                        "http://source/step_5", require_delta=True
                    )
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem._read_chunked_xor_manifest = old_transport
            filesystem._source_manifest = old_manifest
            filesystem._stage_chunked_linear_adapter = old_stage
            filesystem.download = old_download
            filesystem._delta_stage_timeout_s = old_timeout

    def test_protocol_has_one_strict_delta_flag(self):
        worker = ROOT / "src/prime_rl/inference/vllm/worker/filesystem.py"
        client = ROOT / "src/prime_rl/utils/client.py"
        server = ROOT / "src/prime_rl/inference/vllm/server.py"
        watcher = ROOT / "src/prime_rl/orchestrator/watcher.py"
        self.assertEqual(
            function_arguments(worker, "stage_lora_adapter_files"),
            ["source_url", "require_delta"],
        )
        self.assertEqual(
            function_arguments(worker, "stage_lora_adapter"),
            ["self", "source_url", "require_delta"],
        )
        for path in (client, server, watcher):
            self.assertIn("require_delta", path.read_text(encoding="utf-8"))
        worker_source = worker.read_text(encoding="utf-8")
        self.assertIn("delta adapter transport is required", worker_source)
        self.assertIn('chunk_urls = [f"{lane_url.rstrip(\'/\')}/{name}" for lane_url in lane_urls]', worker_source)
        self.assertIn("chunk_urls = chunk_urls[chunk_lane:] + chunk_urls[:chunk_lane]", worker_source)
        self.assertIn("source_urls=chunk_urls", worker_source)
        self.assertIn("parallel_ranges=False", worker_source)
        self.assertIn("executor.shutdown(wait=True, cancel_futures=True)", worker_source)
        self.assertIn("_cleanup_delta_stage_artifacts(", worker_source)
        self.assertIn("preserve_partial=isinstance(exc, TimeoutError)", worker_source)
        self.assertIn("deadline=delta_deadline", worker_source)
        self.assertIn("asyncio.wait_for(", server.read_text(encoding="utf-8"))
        watcher_source = watcher.read_text(encoding="utf-8")
        self.assertIn('manifest.get("version") != 1', watcher_source)
        self.assertIn("next_step > 0", watcher_source)

    def test_delta_cleanup_waits_for_relay_reader(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "adapter"
            partial = root / "adapter.partial"
            partial.mkdir()
            (partial / "RELAYING").write_text("relaying\n", encoding="utf-8")
            lock_path = filesystem._adapter_reader_lock(destination)
            lock_path.touch()
            with lock_path.open("a+") as reader:
                filesystem.fcntl.flock(reader, filesystem.fcntl.LOCK_SH)
                result = []
                cleanup = threading.Thread(
                    target=lambda: result.append(
                        filesystem._cleanup_delta_stage_artifacts(
                            destination,
                            deadline=time.monotonic() + 1.0,
                        )
                    )
                )
                cleanup.start()
                time.sleep(0.05)
                self.assertTrue(partial.is_dir())
                filesystem.fcntl.flock(reader, filesystem.fcntl.LOCK_UN)
            cleanup.join(timeout=2.0)
            self.assertFalse(cleanup.is_alive())
            self.assertEqual(result, [True])
            self.assertFalse(partial.exists())

    def test_download_stream_count_is_required_and_bounded(self):
        filesystem = load_filesystem_module()
        old_value = os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS")
        try:
            for value in (None, "", "0", "129", "not-an-integer"):
                if value is None:
                    os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
                else:
                    os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = value
                with self.assertRaises(OSError):
                    filesystem._configured_download_streams()
            os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = "96"
            self.assertEqual(filesystem._configured_download_streams(), 96)
        finally:
            if old_value is None:
                os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
            else:
                os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = old_value

    def test_explicit_delta_stage_timeout_is_positive_and_bounded(self):
        filesystem = load_filesystem_module()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_timeout = filesystem._delta_stage_timeout_s
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                filesystem._delta_stage_timeout_s = lambda: 30.0
                invalid_values = (
                    (False, 1.0),
                    (True, 0),
                    (True, -1.0),
                    (True, 31.0),
                    (True, True),
                    (True, "1"),
                    (True, float("nan")),
                    (True, float("inf")),
                    (True, float("-inf")),
                )
                for require_delta, timeout in invalid_values:
                    with self.assertRaisesRegex(OSError, "timeout is invalid"):
                        filesystem.stage_lora_adapter_files(
                            "http://source/step_1",
                            require_delta=require_delta,
                            stage_timeout_seconds=timeout,
                        )
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem._delta_stage_timeout_s = old_timeout

    def test_stage_timeout_returns_only_after_worker_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "READY"
            started = threading.Event()

            def stage_worker(
                _source_url,
                *,
                require_delta,
                stage_timeout_seconds,
                cancel_event,
                committed_event,
            ):
                self.assertTrue(require_delta)
                self.assertEqual(stage_timeout_seconds, 0.01)
                self.assertFalse(committed_event.is_set())
                started.set()
                time.sleep(0.04)
                if cancel_event.is_set():
                    raise TimeoutError("cancelled")
                ready.write_text("ready\n", encoding="utf-8")
                return Path(temporary)

            stage_method, response_type = load_server_stage_method(stage_worker)

            class Request:
                async def json(self):
                    return {
                        "lora_path": "http://source/step_1",
                        "require_delta": True,
                        "stage_timeout_seconds": 0.01,
                    }

                async def is_disconnected(self):
                    return False

            started_at = time.monotonic()
            response = asyncio.run(stage_method(Request()))
            elapsed = time.monotonic() - started_at

            self.assertIsInstance(response, response_type)
            self.assertEqual(response.status_code, 504)
            self.assertTrue(started.is_set())
            self.assertGreaterEqual(elapsed, 0.035)
            self.assertFalse(ready.exists())
            time.sleep(0.02)
            self.assertFalse(ready.exists())

    def test_publish_wins_timeout_race_returns_success(self):
        for _ in range(10):
            with tempfile.TemporaryDirectory() as temporary:
                ready = Path(temporary) / "READY"
                started = threading.Event()
                publish_lock = threading.Lock()

                def stage_worker(
                    _source_url,
                    *,
                    require_delta,
                    stage_timeout_seconds,
                    cancel_event,
                    committed_event,
                ):
                    self.assertTrue(require_delta)
                    self.assertEqual(stage_timeout_seconds, 0.005)
                    with publish_lock:
                        started.set()
                        time.sleep(0.02)
                        if cancel_event.is_set():
                            raise TimeoutError("cancelled")
                        ready.write_text("ready\n", encoding="utf-8")
                        committed_event.set()
                    return Path(temporary)

                stage_method, _ = load_server_stage_method(stage_worker, publish_lock)

                class Request:
                    async def json(self):
                        return {
                            "lora_path": "http://source/step_1",
                            "require_delta": True,
                            "stage_timeout_seconds": 0.005,
                        }

                    async def is_disconnected(self):
                        return False

                async def run_stage():
                    task = asyncio.create_task(stage_method(Request()))
                    await asyncio.to_thread(started.wait)
                    return await task

                response = asyncio.run(run_stage())
                self.assertIsInstance(response, dict)
                self.assertEqual(response["status"], "ok")
                self.assertTrue(ready.exists())

    def test_server_rejects_non_finite_stage_timeout(self):
        def stage_worker(**_kwargs):
            raise AssertionError("invalid timeout reached the worker")

        stage_method, response_type = load_server_stage_method(stage_worker)

        class Request:
            def __init__(self, timeout):
                self.timeout = timeout

            async def json(self):
                return {
                    "lora_path": "http://source/step_1",
                    "require_delta": True,
                    "stage_timeout_seconds": self.timeout,
                }

        for timeout in (float("nan"), float("inf"), float("-inf")):
            response = asyncio.run(stage_method(Request(timeout)))
            self.assertIsInstance(response, response_type)
            self.assertEqual(response.status_code, 400)

    def test_request_cancellation_waits_for_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "READY"
            started = threading.Event()
            stopped = threading.Event()

            def stage_worker(
                _source_url,
                *,
                require_delta,
                stage_timeout_seconds,
                cancel_event,
                committed_event,
            ):
                self.assertFalse(require_delta)
                self.assertIsNone(stage_timeout_seconds)
                self.assertFalse(committed_event.is_set())
                started.set()
                time.sleep(0.04)
                if cancel_event.is_set():
                    stopped.set()
                    raise TimeoutError("cancelled")
                ready.write_text("ready\n", encoding="utf-8")
                return Path(temporary)

            stage_method, _ = load_server_stage_method(stage_worker)

            class Request:
                async def json(self):
                    return {
                        "lora_path": "http://source/step_1",
                        "require_delta": False,
                    }

                async def is_disconnected(self):
                    return False

            async def cancel_request():
                task = asyncio.create_task(stage_method(Request()))
                await asyncio.to_thread(started.wait)
                started_at = time.monotonic()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                return time.monotonic() - started_at

            elapsed = asyncio.run(cancel_request())
            self.assertTrue(stopped.is_set())
            self.assertGreaterEqual(elapsed, 0.035)
            self.assertFalse(ready.exists())

    def test_request_disconnect_stops_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "READY"
            started = threading.Event()
            stopped = threading.Event()

            def stage_worker(
                _source_url,
                *,
                require_delta,
                stage_timeout_seconds,
                cancel_event,
                committed_event,
            ):
                self.assertFalse(require_delta)
                self.assertIsNone(stage_timeout_seconds)
                self.assertFalse(committed_event.is_set())
                started.set()
                time.sleep(0.04)
                if cancel_event.is_set():
                    stopped.set()
                    raise TimeoutError("cancelled")
                ready.write_text("ready\n", encoding="utf-8")
                return Path(temporary)

            stage_method, response_type = load_server_stage_method(stage_worker)

            class Request:
                async def json(self):
                    return {
                        "lora_path": "http://source/step_1",
                        "require_delta": False,
                    }

                async def is_disconnected(self):
                    return True

            response = asyncio.run(stage_method(Request()))
            self.assertIsInstance(response, response_type)
            self.assertEqual(response.status_code, 504)
            self.assertTrue(started.is_set())
            self.assertTrue(stopped.is_set())
            self.assertFalse(ready.exists())

    def test_inventory_defines_bounded_96_lane_transfer(self):
        operations = REPOSITORY / "configs/operations.toml"
        with operations.open("rb") as stream:
            config = tomllib.load(stream)
        ports = config["ports"]
        self.assertEqual(ports["lora_source_first"], 18100)
        self.assertEqual(ports["lora_source_last"], 18195)
        self.assertEqual(ports["adapter_relay"], 18196)
        self.assertEqual(ports["refusal_classifier"], 18250)
        self.assertEqual(ports["lora_source_last"] - ports["lora_source_first"] + 1, 96)
        self.assertEqual(config["limits"]["adapter_delta_stage_timeout_seconds"], 300)
        source_ports = set(range(ports["lora_source_first"], ports["lora_source_last"] + 1))
        self.assertNotIn(ports["adapter_relay"], source_ports)
        self.assertNotIn(ports["refusal_classifier"], source_ports)
        self.assertNotEqual(ports["adapter_relay"], ports["refusal_classifier"])
        reserved_ports = [
            value
            for name, value in ports.items()
            if name not in {"lora_source_first", "lora_source_last"}
        ]
        self.assertEqual(len(reserved_ports), len(set(reserved_ports)))
        config_path = REPOSITORY / "configs/current/orchestrator.toml"
        with config_path.open("rb") as stream:
            edge_config = tomllib.load(stream)
        source_url = edge_config["weight_broadcast"]["source_url"]
        self.assertEqual(urlsplit(source_url).port, ports["lora_source_first"])
        runtime = REPOSITORY / ".agents/skills/training-ops/runtime/h200"
        glm_source = (runtime / "start_inference.sh").read_text(encoding="utf-8")
        self.assertIn("PRIME_LORA_DOWNLOAD_STREAMS=96", glm_source)
        self.assertIn('PRIME_LORA_SOURCE_PORTS="$lora_source_port_list"', glm_source)
        orchestrator_source = (
            REPOSITORY / ".agents/skills/training-ops/runtime/h200/start_orchestrator.sh"
        ).read_text(encoding="utf-8")
        for legacy_port in ("18101", "18111", "18112", "18113", "18114"):
            self.assertNotIn(legacy_port, orchestrator_source)
        self.assertIn("parsed.port != source_first", orchestrator_source)
        worker_source = (REPOSITORY / "prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py").read_text(encoding="utf-8")
        self.assertIn("download_streams = _configured_download_streams()", worker_source)
        self.assertNotIn("_PARALLEL_DOWNLOAD_STREAMS", worker_source)

    def test_chunk_retries_rotate_lanes_without_nested_ranges(self):
        filesystem = load_filesystem_module()
        payload = b"chunk-payload"
        calls = []

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fake_urlopen(request, timeout):
            calls.append((request.full_url, request.headers.get("Range"), timeout))
            if len(calls) == 1:
                raise OSError("first lane is unavailable")
            return Response(payload)

        class UnexpectedRangeExecutor:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("one chunk must not start nested range workers")

        old_urlopen = filesystem.urllib.request.urlopen
        old_executor = filesystem.ThreadPoolExecutor
        old_sleep = filesystem.time.sleep
        filesystem.urllib.request.urlopen = fake_urlopen
        filesystem.ThreadPoolExecutor = UnexpectedRangeExecutor
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
                )
                self.assertEqual(destination.read_bytes(), payload)
        finally:
            filesystem.urllib.request.urlopen = old_urlopen
            filesystem.ThreadPoolExecutor = old_executor
            filesystem.time.sleep = old_sleep

        self.assertEqual([call[0] for call in calls], ["http://lane-1/chunk", "http://lane-2/chunk"])
        self.assertEqual([call[1] for call in calls], [None, None])

    def test_delta_timeout_keeps_only_matching_partial_progress(self):
        filesystem = load_filesystem_module()
        first_payload = b"first"
        second_payload = b"second"
        config = b"{}\n"
        source_url = "http://source/step_2"
        target = first_payload + second_payload
        target_sha256 = hashlib.sha256(target).hexdigest()
        manifest = {
            "version": 1,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": "chunked-shuffled-xor-zstd",
            "target_step": 2,
            "base_step": 1,
            "files": {
                "adapter_config.json": {
                    "size": len(config),
                    "sha256": hashlib.sha256(config).hexdigest(),
                },
                "adapter_model.safetensors": {
                    "size": len(target),
                    "sha256": target_sha256,
                },
            },
        }
        transport = {
            "version": 1,
            "mode": "delta",
            "transport": "chunked-shuffled-xor-zstd",
            "source_manifest": {
                "version": 1,
                "kind": "lora-adapter",
                "mode": "delta",
                "transport": "chunked-shuffled-xor-zstd",
                "target_step": 2,
                "base_step": 1,
                "files": [],
            },
            "base_url": "http://source/step_1",
            "base_sha256": "a" * 64,
            "base_config_sha256": "b" * 64,
            "target_sha256": target_sha256,
            "target_size": len(target),
            "config_sha256": hashlib.sha256(config).hexdigest(),
            "config_size": len(config),
            "chunks": [
                {
                    "file": "chunk-0",
                    "offset": 0,
                    "raw_size": len(first_payload),
                    "wire_size": len(first_payload),
                    "wire_sha256": hashlib.sha256(first_payload).hexdigest(),
                    "raw_sha256": hashlib.sha256(first_payload).hexdigest(),
                },
                {
                    "file": "chunk-1",
                    "offset": len(first_payload),
                    "raw_size": len(second_payload),
                    "wire_size": len(second_payload),
                    "wire_sha256": hashlib.sha256(second_payload).hexdigest(),
                    "raw_sha256": hashlib.sha256(second_payload).hexdigest(),
                },
            ],
        }
        base_directory = tempfile.TemporaryDirectory()
        base_model = Path(base_directory.name) / "base"
        base_model.write_bytes(b"\0" * len(target))
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_timeout = filesystem._delta_stage_timeout_s
        old_manifest = filesystem._read_chunked_xor_manifest
        old_source_manifest = filesystem._source_manifest
        old_base = filesystem._verified_adapter_base
        old_lanes = filesystem._adapter_source_urls
        old_streams = filesystem._configured_download_streams
        old_decode = filesystem._decode_transport_chunk
        old_raw_digest = filesystem._transport_raw_sha256
        old_xor = filesystem._xor_bytes
        old_download = filesystem.download
        old_cleanup = filesystem._cleanup_delta_stage_artifacts
        downloaded = []
        fail_second_chunk = True
        cleanup_contexts = []

        def record_cleanup(*args, **kwargs):
            cleanup_contexts.append(kwargs)
            return old_cleanup(*args, **kwargs)

        def fake_download(_source, destination, **kwargs):
            nonlocal fail_second_chunk
            if "chunk-" in destination.name:
                self.assertFalse(kwargs.get("parallel_ranges", True))
            downloaded.append(destination.name)
            payload = config
            if "chunk-0" in destination.name:
                payload = first_payload
            elif "chunk-1" in destination.name:
                payload = second_payload
            destination.write_bytes(payload)
            if "chunk-1" in destination.name and fail_second_chunk:
                fail_second_chunk = False
                raise TimeoutError("test deadline")

        filesystem._delta_stage_timeout_s = lambda: 30.0
        filesystem._read_chunked_xor_manifest = lambda *_args, **_kwargs: transport
        filesystem._source_manifest = lambda *_args, **_kwargs: manifest
        filesystem._verified_adapter_base = lambda *_args, **_kwargs: (base_model, base_model)
        filesystem._adapter_source_urls = lambda _source: ["http://lane-1", "http://lane-2"]
        filesystem._configured_download_streams = lambda: 1
        filesystem._decode_transport_chunk = lambda payload, **_kwargs: payload
        filesystem._transport_raw_sha256 = lambda payload: hashlib.sha256(payload).hexdigest()
        filesystem._xor_bytes = lambda _base, payload: payload
        filesystem.download = fake_download
        filesystem._cleanup_delta_stage_artifacts = record_cleanup
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                with self.assertRaises(TimeoutError):
                    filesystem.stage_lora_adapter_files(source_url, require_delta=True)

                destination = filesystem.adapter_cache_path(source_url)
                partial = destination.with_name(f"{destination.name}.partial")
                self.assertFalse((destination / "READY").exists())
                completed = partial / "transport-chunks" / "chunk-0.done"
                self.assertTrue(completed.is_file())
                self.assertTrue((partial / "adapter_model.safetensors").is_file())
                self.assertEqual(len(cleanup_contexts), 1)
                self.assertIsNotNone(cleanup_contexts[0]["cancel_event"])
                self.assertFalse(cleanup_contexts[0]["cancel_event"].is_set())
                self.assertGreater(cleanup_contexts[0]["deadline"], time.monotonic())
                downloaded_after_failure = list(downloaded)

                filesystem.stage_lora_adapter_files(source_url, require_delta=True)
                self.assertTrue((destination / "READY").is_file())
                self.assertTrue((destination / "adapter_manifest.json").is_file())
                self.assertTrue((destination / "relay_manifest.json").is_file())
                self.assertTrue((destination / "transport-chunks" / "chunk-0").is_file())
                self.assertTrue((destination / "transport-chunks" / "chunk-1").is_file())
                resumed = downloaded[len(downloaded_after_failure) :]
                self.assertFalse(any("chunk-0" in name for name in resumed))
                self.assertTrue(any("chunk-1" in name for name in resumed))
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem._delta_stage_timeout_s = old_timeout
            filesystem._read_chunked_xor_manifest = old_manifest
            filesystem._source_manifest = old_source_manifest
            filesystem._verified_adapter_base = old_base
            filesystem._adapter_source_urls = old_lanes
            filesystem._configured_download_streams = old_streams
            filesystem._decode_transport_chunk = old_decode
            filesystem._transport_raw_sha256 = old_raw_digest
            filesystem._xor_bytes = old_xor
            filesystem.download = old_download
            filesystem._cleanup_delta_stage_artifacts = old_cleanup
            base_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
