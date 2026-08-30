import ast
import functools
import hashlib
import http.server
import importlib.util
import io
import json
import os
import socketserver
import sys
import tempfile
import threading
import types
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace


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


class QuietDirectoryHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        return


class AdapterTransportStateTest(unittest.TestCase):
    def test_exact_adjacent_delta_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "step_1"
            target = root / "step_2"
            base.mkdir()
            target.mkdir()
            original = bytes(range(251)) * 33
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
                chunk_size=1024,
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
        self.assertIn("source_urls=chunk_urls", worker_source)
        self.assertIn("parallel_ranges=False", worker_source)
        self.assertIn("executor.shutdown(wait=True, cancel_futures=True)", worker_source)
        self.assertIn("_cleanup_delta_stage_artifacts(destination)", worker_source)
        self.assertIn("deadline=delta_deadline", worker_source)
        self.assertIn("asyncio.wait_for(", server.read_text(encoding="utf-8"))
        watcher_source = watcher.read_text(encoding="utf-8")
        self.assertIn('manifest.get("version") != 1', watcher_source)
        self.assertIn("next_step > 0", watcher_source)

    def test_download_stream_count_is_required_and_bounded(self):
        filesystem = load_filesystem_module()
        old_value = os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS")
        try:
            for value in (None, "", "0", "65", "not-an-integer"):
                if value is None:
                    os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
                else:
                    os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = value
                with self.assertRaises(OSError):
                    filesystem._configured_download_streams()
            os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = "24"
            self.assertEqual(filesystem._configured_download_streams(), 24)
        finally:
            if old_value is None:
                os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
            else:
                os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = old_value

    def test_inventory_defines_bounded_24_lane_transfer(self):
        operations = REPOSITORY / "configs/operations.toml"
        with operations.open("rb") as stream:
            config = tomllib.load(stream)
        ports = config["ports"]
        self.assertEqual(ports["lora_source_last"] - ports["lora_source_first"] + 1, 24)
        self.assertEqual(config["limits"]["adapter_delta_stage_timeout_seconds"], 300)
        runtime = REPOSITORY / ".agents/skills/training-ops/runtime/h200"
        for name in ("start_inference.sh", "start_muse_inference.sh"):
            source = (runtime / name).read_text(encoding="utf-8")
            self.assertIn("PRIME_LORA_DOWNLOAD_STREAMS=24", source)
            self.assertIn('PRIME_LORA_SOURCE_PORTS="$lora_source_port_list"', source)
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


if __name__ == "__main__":
    unittest.main()
