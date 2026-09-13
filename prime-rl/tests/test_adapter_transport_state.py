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
import re
import socketserver
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import tomllib
import unittest
import uuid
from unittest import mock
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
SERVER_PATH = ROOT.parent / "training_ops/runtime/china/adapter_server.py"
SERVER_SPEC = importlib.util.spec_from_file_location("adapter_server", SERVER_PATH)
assert SERVER_SPEC is not None and SERVER_SPEC.loader is not None
adapter_server = importlib.util.module_from_spec(SERVER_SPEC)
SERVER_SPEC.loader.exec_module(adapter_server)


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
        "adapter_stage_limit_seconds": lambda step: {
            1: 600.0,
            2: 600.0,
        }.get(step, 300.0),
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


def load_receiver_ack_writer():
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_receiver_id", "_ack_identity", "_write_receivers_ready"}
    ]
    namespace = {
        "json": json,
        "os": os,
        "tempfile": tempfile,
        "_validate_loaded_receipt": lambda receipt, **_kwargs: dict(receipt),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_write_receivers_ready"]


def load_receiver_ack_reader():
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_receiver_id", "_read_receivers_ready", "_valid_digest"}
    ]

    def validate_loaded_receipt(receipt, *, source_url, lora_name, expected):
        if not isinstance(receipt, dict):
            raise RuntimeError("loaded receipt is invalid")
        if receipt.get("source_url") != source_url or receipt.get("lora_name") != lora_name:
            raise RuntimeError("loaded receipt differs from marker")
        for name, value in expected.items():
            actual = receipt.get(name)
            if name.endswith("_size"):
                if type(actual) is not int or actual != value:
                    raise RuntimeError(f"loaded receipt differs in {name}")
            elif actual != value:
                raise RuntimeError(f"loaded receipt differs in {name}")
        return dict(receipt)

    namespace = {
        "json": json,
        "_validate_loaded_receipt": validate_loaded_receipt,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_read_receivers_ready"]


def load_receipt_ack_validator():
    path = ROOT / "src/prime_rl/orchestrator/watcher.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_receipt_against_ack"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_validate_receipt_against_ack"]


def load_broadcast_cleanup_method():
    path = ROOT / "src/prime_rl/trainer/rl/broadcast/filesystem.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_valid_digest",
            "_published_adapter_identity",
            "_receiver_ids",
            "_receiver_ack_identity",
            "_validate_receiver_ack",
            "maybe_clean",
        }
    ]
    namespace = {
        "Path": Path,
        "json": json,
        "re": __import__("re"),
        "shutil": __import__("shutil"),
        "quote": lambda value: value,
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "maybe_clean": lambda *_args: None,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["maybe_clean"]


def write_published_step(root: Path, step: int) -> Path:
    path = root / f"step_{step}"
    path.mkdir()
    config = b"{}"
    weights = b"weights"
    (path / "STABLE").touch()
    (path / "adapter_config.json").write_bytes(config)
    (path / "adapter_model.safetensors").write_bytes(weights)
    (path / "adapter_model.safetensors.sha256").write_text(
        hashlib.sha256(weights).hexdigest() + "\n", encoding="ascii"
    )
    (path / "adapter_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "files": [
                    {"name": "adapter_config.json", "size": len(config), "sha256": hashlib.sha256(config).hexdigest()},
                    {"name": "adapter_model.safetensors", "size": len(weights), "sha256": hashlib.sha256(weights).hexdigest()},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_receiver_ack(step_path: Path, root: Path, step: int, *, run_name="run_test", endpoints=None, **overrides):
    target = hashlib.sha256(b"weights").hexdigest()
    config_digest = hashlib.sha256(b"{}").hexdigest()
    source_url = f"http://source/broadcasts/step_{step}"
    identity = {
        "version": 1,
        "run_name": run_name,
        "step": step,
        "source_url": source_url,
        "target_sha256": target,
        "target_size": 7,
        "config_sha256": config_digest,
        "config_size": 2,
        "slot": "adapter" if step == 0 else f"adapter-slot{step % 2}",
    }
    identity.update(overrides)
    endpoints = endpoints or ["http://h200-a", "http://h200-b"]
    identity["receivers"] = [
        {
            "endpoint": endpoint,
            "receipt": {
                "source_url": identity["source_url"],
                "target_sha256": identity["target_sha256"],
                "target_size": identity["target_size"],
                "config_sha256": identity["config_sha256"],
                "config_size": identity["config_size"],
                "lora_name": identity["slot"],
                "server_instance": f"server-{index}",
            },
        }
        for index, endpoint in enumerate(endpoints)
    ]
    (step_path / "receivers-ready.json").write_text(json.dumps(identity), encoding="utf-8")


def load_broadcast_weights_method():
    path = ROOT / "src/prime_rl/trainer/rl/broadcast/filesystem.py"
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "broadcast_weights"
    )
    fake_model = type(
        "FakeModel",
        (),
        {
            "is_prime_state_dict": lambda _self, _state_dict: True,
            "convert_to_hf": lambda _self, _state_dict: None,
        },
    )
    namespace = {
        "nn": types.SimpleNamespace(Module=object),
        "time": time,
        "shutil": __import__("shutil"),
        "DTensor": object,
        "PreTrainedModelPrimeRL": fake_model,
        "gather_weights_on_master": lambda *_args, **_kwargs: {},
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "save_state_dict": lambda *_args, **_kwargs: None,
        "save_lora_config": lambda *_args, **_kwargs: None,
        "publish_adapter_manifest": lambda *_args, **_kwargs: {
            "mode": "linear",
            "encoded_size": 0,
            "full_size": 0,
            "build_seconds": 0.0,
            "chunk_count": 0,
        },
        "quote": lambda value: value,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["broadcast_weights"], fake_model


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
        "prime_rl.utils.adapter_delta": adapter_delta,
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

        def decode_transport_chunk(payload, *, codec, raw_size):
            return adapter_delta.decode_adapter_chunk(
                payload,
                codec=codec,
                max_output_size=raw_size,
            )

        module._decode_transport_chunk = decode_transport_chunk
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

    stage_path = {}
    stage_state = {}
    processes = {}

    class FakeProcess:
        _next_pid = 41000

        def __init__(self):
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.returncode = None
            self.stopped = threading.Event()
            self.finished = threading.Event()

        async def communicate(self, payload):
            request = json.loads(payload.decode("utf-8"))

            def run_stage():
                try:
                    destination = stage_worker(
                        request["source_url"],
                        require_delta=request["require_delta"],
                        stage_timeout_seconds=request.get("stage_timeout_seconds"),
                        policy_step=request.get("policy_step"),
                        cancel_event=self.stopped,
                        committed_event=threading.Event(),
                    )
                    stage_path["value"] = Path(destination)
                    stage_state.update(
                        {
                            "status": "ready",
                            "stage_id": request["stage_id"],
                            "pid": self.pid,
                            "source_url": request["source_url"],
                            "path": Path(destination).as_posix(),
                            "target_sha256": "a" * 64,
                            "ready_at_monotonic": time.monotonic(),
                        }
                    )
                    result = {
                        "stage_id": request["stage_id"],
                        "pid": self.pid,
                        "status": "ok",
                        "path": Path(destination).as_posix(),
                    }
                except BaseException as exc:
                    result = {
                        "stage_id": request.get("stage_id"),
                        "pid": self.pid,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                return json.dumps(result, separators=(",", ":")).encode("utf-8")

            try:
                stdout = await asyncio.to_thread(run_stage)
            finally:
                self.returncode = 0
                self.finished.set()
            return stdout, None

        async def wait(self):
            await asyncio.to_thread(self.finished.wait)
            return self.returncode

    async def create_subprocess_exec(*_args, **_kwargs):
        process = FakeProcess()
        processes[process.pid] = process
        return process

    def killpg(pid, _signal):
        process = processes[pid]
        process.stopped.set()

    async def reap(process, communication=None):
        if process.returncode is None:
            process.stopped.set()
            await process.wait()
        return await communication if communication is not None else (b"", b"")

    class AsyncioProxy:
        subprocess = types.SimpleNamespace(PIPE=object())

        def __getattr__(self, name):
            return getattr(asyncio, name)

    asyncio_proxy = AsyncioProxy()
    asyncio_proxy.create_subprocess_exec = create_subprocess_exec
    process_os = types.SimpleNamespace(
        **{name: getattr(os, name) for name in dir(os) if name != "killpg"}
    )
    process_os.killpg = killpg

    namespace = {
        "asyncio": asyncio_proxy,
        "json": json,
        "uuid": uuid,
        "sys": sys,
        "Path": Path,
        "os": process_os,
        "signal": signal,
        "threading": threading,
        "time": time,
        "router": FakeRouter(),
        "Request": object,
        "JSONResponse": FakeJSONResponse,
        "_delta_stage_timeout_s": lambda: 600.0,
        "full_adapter_stage_timeout_s": 1800.0,
        "_adapter_step_from_source_url": lambda source_url: int(
            re.fullmatch(r"step_(\d+)", Path(source_url.rstrip("/")).name).group(1)
        ),
        "adapter_stage_limit_seconds": lambda _step: 600.0,
        "_adapter_stage_publish_lock": publish_lock or threading.Lock(),
        "_is_finite_number": lambda value: (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ),
        "_begin_lora_stage": lambda: None,
        "_end_lora_stage": lambda: None,
        "_set_lora_stage_process": lambda *_args: None,
        "_kill_stage_process": reap,
        "adapter_cache_path": lambda source_url: stage_path.get("value", Path(source_url)),
        "stage_lora_adapter_files": stage_worker,
        "adapter_stage_status": lambda: dict(stage_state),
        "_read_verified_adapter_identity": lambda source_url, _destination, **_kwargs: {
            "source_url": source_url,
            "target_sha256": "a" * 64,
            "target_size": 8,
        },
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["stage_lora_adapter"], FakeJSONResponse


class QuietDirectoryHandler(adapter_server.RangeHandler):
    def log_message(self, *_args):
        return


class AdapterTransportStateTest(unittest.TestCase):
    def test_default_chunk_sizes_keep_v1_and_v2_transport_contracts(self):
        target_size = 30_767_549_784
        self.assertEqual(
            len(adapter_delta._chunk_ranges(target_size, adapter_delta.default_chunk_size)),
            1834,
        )
        self.assertEqual(
            len(adapter_delta._chunk_ranges(target_size, adapter_delta.predictive_chunk_size)),
            1834,
        )

    def test_broadcast_file_not_found_only_skips_deleted_run(self):
        broadcast_weights, fake_model = load_broadcast_weights_method()
        cases = (
            ("trainer-only run", "run_trainer_only", None, False, True),
            ("missing config, existing directory", "run_missing_config", None, False, True),
            ("existing run", "run_existing", object(), False, True),
            ("truly deleted run", "run_deleted", None, True, False),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, run_id, config, delete_run, should_raise in cases:
                with self.subTest(name):
                    run_dir = root / run_id
                    run_dir.mkdir()
                    message = f"missing publication for {run_id}"
                    warnings = []
                    manager = SimpleNamespace(
                        ready_to_update_idxs=[0],
                        ready_to_update={0: True},
                        idx_2_id={0: run_id},
                        progress={0: SimpleNamespace(step=1)},
                        config={
                            0: SimpleNamespace(
                                model=SimpleNamespace(lora=SimpleNamespace(rank=1, alpha=1.0))
                            )
                        },
                        get_state_dict_for_run=lambda _idx: {},
                        get_run_dir=lambda _idx: run_dir,
                        get_orchestrator_config=lambda _run_id: config,
                    )

                    def notify(_save_dir):
                        if delete_run:
                            __import__("shutil").rmtree(run_dir)
                        raise FileNotFoundError(message)

                    broadcast = SimpleNamespace(
                        lora_config=None,
                        save_format="safetensors",
                        save_sharded=False,
                        world=SimpleNamespace(is_master=True),
                        multi_run_manager=manager,
                        logger=SimpleNamespace(
                            debug=lambda *_args: None,
                            info=lambda *_args: None,
                            warning=warnings.append,
                            error=lambda *_args: None,
                        ),
                        _notify_orchestrator=notify,
                    )
                    if should_raise:
                        try:
                            broadcast_weights(broadcast, fake_model(), 0)
                        except FileNotFoundError as caught:
                            self.assertEqual(str(caught), message)
                            self.assertEqual(caught.args, (message,))
                            self.assertIsNotNone(caught.__traceback__)
                        else:
                            self.fail(f"expected FileNotFoundError: {message}")
                    else:
                        broadcast_weights(broadcast, fake_model(), 0)
                        self.assertFalse(run_dir.exists())
                        self.assertEqual(len(warnings), 1)
                    self.assertFalse(manager.ready_to_update[0])

    def test_lora_broadcast_cleanup_retains_without_receiver_ack(self):
        cleanup = load_broadcast_cleanup_method()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "broadcasts"
            root.mkdir()
            for step in range(7):
                (root / f"step_{step}").mkdir()
            manager = SimpleNamespace(
                used_idxs=[0],
                progress={0: SimpleNamespace(step=7)},
                get_run_dir=lambda _idx: root,
                idx_2_id={0: "run_test"},
                get_orchestrator_config=lambda _run_id: SimpleNamespace(),
            )
            broadcast = SimpleNamespace(
                world=SimpleNamespace(is_master=True),
                multi_run_manager=manager,
                lora_config=object(),
                logger=SimpleNamespace(debug=lambda *_args: None),
            )
            cleanup(broadcast, 3)
            self.assertTrue((root / "step_3").exists())
            self.assertTrue((root / "step_4").exists())

    def test_lora_cleanup_uses_delayed_ack_and_preserves_ack_bases(self):
        cleanup = load_broadcast_cleanup_method()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "broadcasts"
            root.mkdir()
            for step in range(10):
                write_published_step(root, step)
            (root / "unknown_dir").mkdir()
            (root / "unknown.txt").write_text("keep", encoding="utf-8")

            config = SimpleNamespace(
                model=SimpleNamespace(
                    lora=SimpleNamespace(name="adapter"),
                    client=SimpleNamespace(
                        base_url=["http://h200-a/v1", "http://h200-b/v1"],
                        admin_base_url=None,
                    ),
                ),
                weight_broadcast=SimpleNamespace(source_url="http://source", source_dir=root.parent),
            )
            manager = SimpleNamespace(
                used_idxs=[0],
                progress={0: SimpleNamespace(step=11)},
                get_run_dir=lambda _idx: root,
                idx_2_id={0: "run_test"},
                get_orchestrator_config=lambda _run_id: config,
            )
            broadcast = SimpleNamespace(
                world=SimpleNamespace(is_master=True),
                multi_run_manager=manager,
                lora_config=object(),
                logger=SimpleNamespace(debug=lambda *_args: None, warning=lambda *_args: None),
            )
            cleanup(broadcast, None)
            self.assertTrue((root / "step_0").exists())
            self.assertTrue((root / "step_5").exists())

            (root / "step_1" / "receivers-ready.json").write_text("stale", encoding="utf-8")
            for path in (root / "step_1").iterdir():
                if path.name != "receivers-ready.json":
                    path.unlink()
            write_receiver_ack(root / "step_8", root, 8)
            cleanup(broadcast, None)
            for step in range(6):
                self.assertFalse((root / f"step_{step}").exists())
            for step in (6, 7, 8, 9):
                self.assertTrue((root / f"step_{step}").exists())
            self.assertTrue((root / "unknown_dir").exists())
            self.assertTrue((root / "unknown.txt").exists())

    def test_lora_cleanup_rejects_invalid_highest_ack(self):
        cleanup = load_broadcast_cleanup_method()
        for field, value in (
            ("run_name", "wrong_run"),
            ("source_url", "http://wrong"),
            ("target_sha256", "f" * 64),
            ("config_sha256", "e" * 64),
            ("version", True),
            ("slot", []),
            ("slot", {}),
            ("target_size", True),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "broadcasts"
                root.mkdir()
                write_published_step(root, 3)
                write_receiver_ack(root / "step_3", root, 3, **{field: value})
                config = SimpleNamespace(
                    model=SimpleNamespace(
                        lora=SimpleNamespace(name="adapter"),
                        client=SimpleNamespace(
                            base_url=["http://h200-a/v1", "http://h200-b/v1"],
                            admin_base_url=None,
                        ),
                    ),
                    weight_broadcast=SimpleNamespace(source_url="http://source", source_dir=root.parent),
                )
                manager = SimpleNamespace(
                    used_idxs=[0],
                    progress={0: SimpleNamespace(step=4)},
                    get_run_dir=lambda _idx: root,
                    idx_2_id={0: "run_test"},
                    get_orchestrator_config=lambda _run_id: config,
                )
                broadcast = SimpleNamespace(
                    world=SimpleNamespace(is_master=True),
                    multi_run_manager=manager,
                    lora_config=object(),
                    logger=SimpleNamespace(debug=lambda *_args: None, warning=lambda *_args: None),
                )
                with self.assertRaises(RuntimeError):
                    cleanup(broadcast, None)

    def test_lora_cleanup_rejects_partial_receiver_ack(self):
        cleanup = load_broadcast_cleanup_method()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "broadcasts"
            root.mkdir()
            write_published_step(root, 3)
            write_receiver_ack(root / "step_3", root, 3, endpoints=["http://h200-a"])
            config = SimpleNamespace(
                model=SimpleNamespace(
                    lora=SimpleNamespace(name="adapter"),
                    client=SimpleNamespace(
                        base_url=["http://h200-a/v1", "http://h200-b/v1"],
                        admin_base_url=None,
                    ),
                ),
                weight_broadcast=SimpleNamespace(source_url="http://source", source_dir=root.parent),
            )
            manager = SimpleNamespace(
                used_idxs=[0],
                progress={0: SimpleNamespace(step=4)},
                get_run_dir=lambda _idx: root,
                idx_2_id={0: "run_test"},
                get_orchestrator_config=lambda _run_id: config,
            )
            broadcast = SimpleNamespace(
                world=SimpleNamespace(is_master=True),
                multi_run_manager=manager,
                lora_config=object(),
                logger=SimpleNamespace(debug=lambda *_args: None, warning=lambda *_args: None),
            )
            with self.assertRaises(RuntimeError):
                cleanup(broadcast, None)

    def test_receiver_ack_writer_publishes_startup_and_reuses_identity(self):
        write_ack = load_receiver_ack_writer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clients = [
                SimpleNamespace(base_url="http://h200-b/v1"),
                SimpleNamespace(base_url="http://h200-a/v1"),
            ]

            def publish(step, slot, server_suffix):
                step_path = root / f"step_{step}"
                step_path.mkdir()
                (step_path / "STABLE").touch()
                identity = {
                    "source_url": f"http://source/step_{step}",
                    "target_sha256": "a" * 64,
                    "target_size": 8,
                    "config_sha256": "b" * 64,
                    "config_size": 2,
                }
                receipts = [
                    {
                        **identity,
                        "lora_name": slot,
                        "server_instance": f"{server_suffix}-b",
                    },
                    {
                        **identity,
                        "lora_name": slot,
                        "server_instance": f"{server_suffix}-a",
                    },
                ]
                write_ack(
                    step_path,
                    run_name="run_test",
                    step=step,
                    source_url=identity["source_url"],
                    slot=slot,
                    adapter_identity=identity,
                    admin_clients=clients,
                    receipts=receipts,
                )
                return step_path, identity, receipts

            startup, identity, receipts = publish(0, "adapter", "old")
            startup_ack = json.loads((startup / "receivers-ready.json").read_text())
            self.assertEqual(
                [item["endpoint"] for item in startup_ack["receivers"]],
                ["http://h200-a", "http://h200-b"],
            )
            self.assertEqual((startup / "receivers-ready.json").stat().st_mode & 0o777, 0o644)
            original = (startup / "receivers-ready.json").read_text()
            write_ack(
                startup,
                run_name="run_test",
                step=0,
                source_url=identity["source_url"],
                slot="adapter-slot0",
                adapter_identity=identity,
                admin_clients=clients,
                receipts=[
                    {**item, "lora_name": "adapter-slot0", "server_instance": "new"}
                    for item in receipts
                ],
            )
            self.assertEqual((startup / "receivers-ready.json").read_text(), original)

            normal, identity, _ = publish(1, "adapter-slot1", "current")
            self.assertTrue((normal / "receivers-ready.json").is_file())
            with self.assertRaisesRegex(RuntimeError, "duplicate endpoint"):
                write_ack(
                    normal,
                    run_name="run_test",
                    step=1,
                    source_url=identity["source_url"],
                    slot="adapter-slot1",
                    adapter_identity=identity,
                    admin_clients=[clients[0], clients[0]],
                    receipts=[receipts[0], receipts[1]],
                )

    def test_receiver_ack_reader_requires_current_marker_identity(self):
        read_ack = load_receiver_ack_reader()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step_path = root / "step_3"
            step_path.mkdir()
            clients = [
                SimpleNamespace(base_url="http://h200-a/v1"),
                SimpleNamespace(base_url="http://h200-b/v1"),
            ]
            write_receiver_ack(step_path, root, 3)
            source_url = "http://source/broadcasts/step_3"

            def read_current():
                return read_ack(
                    step_path,
                    run_name="run_test",
                    step=3,
                    source_url=source_url,
                    slot="adapter-slot1",
                    admin_clients=clients,
                )

            self.assertEqual(
                read_current()["step"],
                3,
            )

            marker = step_path / "receivers-ready.json"
            payload = json.loads(marker.read_text())
            payload["version"] = True
            marker.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "stale"):
                read_current()

            for endpoints in (
                ["http://h200-a", "http://h200-a"],
                ["http://h200-a", "http://other"],
                ["http://h200-a"],
            ):
                write_receiver_ack(step_path, root, 3, endpoints=endpoints)
                with self.assertRaisesRegex(RuntimeError, "endpoint|records"):
                    read_current()

            write_receiver_ack(step_path, root, 3)
            payload = json.loads(marker.read_text())
            payload["receivers"][0]["receipt"] = None
            marker.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "receipt"):
                read_current()

            write_receiver_ack(step_path, root, 3)
            payload = json.loads(marker.read_text())
            payload["target_sha256"] = "b" * 64
            marker.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "differs"):
                read_current()

            write_receiver_ack(step_path, root, 3, run_name="old_run")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                read_current()

            (step_path / "receivers-ready.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                read_current()

    def test_receipt_ack_validator_rejects_boolean_sizes(self):
        validate = load_receipt_ack_validator()
        receipt = {
            "source_url": "http://source/broadcasts/step_3",
            "target_sha256": "a" * 64,
            "target_size": 7,
            "config_sha256": "b" * 64,
            "config_size": 2,
            "lora_name": "adapter-slot1",
        }
        ack = dict(receipt)
        ack["slot"] = ack.pop("lora_name")
        validate(receipt, ack, step=3)
        for name in ("target_size", "config_size"):
            with self.subTest(name=name):
                invalid = dict(receipt)
                invalid[name] = True
                with self.assertRaisesRegex(RuntimeError, name):
                    validate(invalid, ack, step=3)

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
            self.assertEqual(calls, [(f"{target_url}/adapter_manifest.json", 600.0)])

    def test_watcher_manifest_probe_uses_exact_policy_step_limit(self):
        for target_step, expected_limit in ((1, 600.0), (2, 600.0), (3, 300.0)):
            with self.subTest(target_step=target_step), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for step in range(target_step + 1):
                    (root / f"step_{step}").mkdir()
                    (root / f"step_{step}" / "STABLE").touch()
                target_url = f"http://source/step_{target_step}"
                manifest = {
                    "version": 1,
                    "kind": "lora-adapter",
                    "mode": "delta",
                    "transport": "chunked-shuffled-xor-zstd",
                    "target_step": target_step,
                    "base_step": target_step - 1,
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
                    verified_adapter_step=target_step - 1,
                    verified_adapter_url=None,
                    last_adapter_delta_build_time=-1.0,
                )
                method(watcher, target_step, root / f"step_{target_step}")

                self.assertEqual(
                    calls,
                    [(f"{target_url}/adapter_manifest.json", expected_limit)],
                )

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
            self.assertEqual(manifest["transport"], "chunked-full-zstd")
            self.assertEqual(manifest["chunk_count"], 1)
            self.assertEqual(len(list(target.glob("*.transport-*"))), 1)

    def test_full_stage_publishes_model_checksum_sidecar(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "step_0"
            source.mkdir()
            config = b"{}\n"
            model = (b"\x01\x02" * ((adapter_delta.default_chunk_size + 18) // 2))[
                : adapter_delta.default_chunk_size + 17
            ]
            (source / "adapter_config.json").write_bytes(config)
            (source / "adapter_model.safetensors").write_bytes(model)
            manifest = adapter_delta.publish_adapter_manifest(source, target_step=0, max_workers=2)
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
            self.assertNotIn(
                "transport",
                json.loads((destination / "adapter_manifest.json").read_text(encoding="utf-8")),
            )
            relay_manifest = json.loads((destination / "relay_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(relay_manifest["transport"], "full")
            self.assertIn("adapter_model.safetensors", relay_manifest["files"])

    def test_full_stage_corrupt_exact_size_progress_stays_bounded(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "step_0"
            source.mkdir()
            config = b"{}\n"
            model = (b"\x01\x02" * ((adapter_delta.default_chunk_size + 18) // 2))[
                : adapter_delta.default_chunk_size + 17
            ]
            (source / "adapter_config.json").write_bytes(config)
            (source / "adapter_model.safetensors").write_bytes(model)
            manifest = adapter_delta.publish_adapter_manifest(source, target_step=0, max_workers=2)
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
                    partial = destination.with_name(f"{destination.name}.partial")
                    destination.rename(partial)
                    (partial / "READY").unlink()
                    chunk = partial / "transport-chunks" / manifest["chunks"][0]["file"]
                    chunk.write_bytes(b"x" * chunk.stat().st_size)
                    statuses = []
                    old_status = filesystem._write_stage_status

                    def record_status(cache_root, value, **kwargs):
                        statuses.append(dict(value))
                        return old_status(cache_root, value, **kwargs)

                    filesystem._write_stage_status = record_status
                    try:
                        resumed = filesystem.stage_lora_adapter_files(source_url)
                    finally:
                        filesystem._write_stage_status = old_status
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
            self.assertEqual(resumed, destination)
            for status in statuses:
                self.assertGreaterEqual(status["bytes_done"], 0)
                self.assertLessEqual(status["bytes_done"], status["bytes_total"])
            self.assertEqual((resumed / "adapter_model.safetensors").read_bytes(), model)

    def test_full_stage_rejects_corrupt_wire_chunk(self):
        filesystem = load_filesystem_module()
        old_retries = filesystem._ADAPTER_LANE_RETRIES
        filesystem._ADAPTER_LANE_RETRIES = 1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "step_0"
            source.mkdir()
            config = b"{}\n"
            model = b"\x01\x02" * 32
            (source / "adapter_config.json").write_bytes(config)
            (source / "adapter_model.safetensors").write_bytes(model)
            manifest = adapter_delta.publish_adapter_manifest(source, target_step=0)
            self.assertEqual(manifest["transport"], adapter_delta.full_transport_name)
            chunk = source / manifest["chunks"][0]["file"]
            payload = bytearray(chunk.read_bytes())
            payload[-1] ^= 1
            chunk.write_bytes(payload)
            (source / "adapter_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            handler = functools.partial(QuietDirectoryHandler, directory=str(root))
            with socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler) as server:
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                source_url = f"http://127.0.0.1:{server.server_address[1]}/step_0"
                old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
                old_streams = os.environ.get("PRIME_LORA_DOWNLOAD_STREAMS")
                os.environ["PRIME_LORA_CACHE_DIR"] = str(root / "cache")
                os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = "4"
                try:
                    with self.assertRaisesRegex(OSError, "checksum"):
                        filesystem.stage_lora_adapter_files(source_url)
                finally:
                    if old_cache is None:
                        os.environ.pop("PRIME_LORA_CACHE_DIR", None)
                    else:
                        os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
                    if old_streams is None:
                        os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
                    else:
                        os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = old_streams
                    filesystem._ADAPTER_LANE_RETRIES = old_retries
                    server.shutdown()
                    server_thread.join()

    def test_policy_zero_full_zstd_exact_sha_and_raw_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compressed = root / "compressed"
            compressed.mkdir()
            (compressed / "adapter_config.json").write_text("{}", encoding="utf-8")
            compressed_bytes = b"a" * (adapter_delta.default_chunk_size + 17)
            (compressed / "adapter_model.safetensors").write_bytes(compressed_bytes)
            manifest = adapter_delta.publish_adapter_manifest(compressed, target_step=0, max_workers=2)
            self.assertEqual(manifest["transport"], adapter_delta.full_transport_name)
            restored = bytearray(len(compressed_bytes))
            for item in manifest["chunks"]:
                raw = adapter_delta.decode_adapter_chunk(
                    (compressed / item["file"]).read_bytes(),
                    codec=manifest["transport"],
                    max_output_size=item["raw_size"],
                )
                start = item["offset"]
                restored[start : start + item["raw_size"]] = raw
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item["raw_sha256"])
            self.assertEqual(bytes(restored), compressed_bytes)

            incompressible = root / "incompressible"
            incompressible.mkdir()
            (incompressible / "adapter_config.json").write_text("{}", encoding="utf-8")
            incompressible_bytes = b"".join(
                hashlib.sha256(str(index).encode("ascii")).digest() for index in range(260)
            )
            (incompressible / "adapter_model.safetensors").write_bytes(incompressible_bytes)
            fallback = adapter_delta.publish_adapter_manifest(incompressible, target_step=0)
            self.assertNotIn("transport", fallback)
            self.assertEqual(fallback["chunk_count"], 0)

    def test_full_zstd_rejects_policy_zero_manifest_at_later_url(self):
        filesystem = load_filesystem_module()
        digest = "a" * 64
        manifest = {
            "version": 1,
            "kind": "lora-adapter",
            "mode": "full",
            "transport": "chunked-full-zstd",
            "target_step": 0,
            "full_size": 1,
            "full_sha256": digest,
            "target_size": 1,
            "target_sha256": digest,
            "config_size": 0,
            "config_sha256": digest,
            "chunks": [],
        }
        old_reader = filesystem._read_source_text
        filesystem._read_source_text = lambda *_args, **_kwargs: json.dumps(manifest)
        try:
            with self.assertRaisesRegex(OSError, "policy zero"):
                filesystem._read_chunked_v1_manifest("http://source/step_9")
        finally:
            filesystem._read_source_text = old_reader

    def test_retryable_stage_errors_keep_matching_partial_data(self):
        filesystem = load_filesystem_module()
        self.assertTrue(filesystem._preserve_adapter_stage_partial(OSError("connection reset")))
        self.assertTrue(filesystem._preserve_adapter_stage_partial(TimeoutError("request timed out")))
        self.assertTrue(
            filesystem._preserve_adapter_stage_partial(
                OSError("adapter manifest is missing; transfer mode is unknown")
            )
        )
        self.assertTrue(
            filesystem._preserve_adapter_stage_partial(
                filesystem.HTTPError("http://relay", 404, "not ready", None, None)
            )
        )
        self.assertFalse(filesystem._preserve_adapter_stage_partial(OSError("checksum mismatch")))
        self.assertFalse(filesystem._preserve_adapter_stage_partial(OSError("path is a symbolic link")))

    def test_relay_unavailable_is_temporary_for_both_delta_versions(self):
        filesystem = load_filesystem_module()
        old_reader = filesystem._read_source_text

        def unavailable(*_args, **_kwargs):
            raise filesystem.HTTPError("http://relay", 503, "starting", None, None)

        filesystem._read_source_text = unavailable
        try:
            for reader in (
                filesystem._read_chunked_v1_manifest,
                filesystem._read_chunked_linear_manifest,
            ):
                with self.subTest(reader=reader.__name__):
                    with self.assertRaisesRegex(OSError, "manifest is missing"):
                        reader("http://relay/step_2")
        finally:
            filesystem._read_source_text = old_reader

    def test_full_cache_reuse_rebinds_url_without_model_download(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_url = "http://source/old/run_old/broadcasts/step_0"
            new_url = "http://source/new/run_new/broadcasts/step_0"
            old_cache_dir = os.environ.get("PRIME_LORA_CACHE_DIR")
            os.environ["PRIME_LORA_CACHE_DIR"] = str(root)
            old_cache = filesystem.adapter_cache_path(old_url)
            new_cache = filesystem.adapter_cache_path(new_url)
            old_cache.mkdir(parents=True)
            config = b"{}\n"
            model = b"model-bytes"
            config_spec = {"size": len(config), "sha256": hashlib.sha256(config).hexdigest()}
            model_spec = {"size": len(model), "sha256": hashlib.sha256(model).hexdigest()}
            files = {"adapter_config.json": config_spec, "adapter_model.safetensors": model_spec}
            (old_cache / "adapter_config.json").write_bytes(config)
            (old_cache / "adapter_model.safetensors").write_bytes(model)
            (old_cache / "manifest.json").write_text(
                json.dumps({"version": 1, "source_url": old_url, "files": files}),
                encoding="utf-8",
            )
            (old_cache / "adapter_model.safetensors.sha256").write_text(
                model_spec["sha256"] + "\n", encoding="ascii"
            )
            (old_cache / "READY").write_text("ready\n", encoding="ascii")
            new_manifest = {"version": 1, "source_url": new_url, "files": files}
            transport_manifest = {
                "version": 1,
                "kind": "lora-adapter",
                "mode": "full",
                "target_step": 0,
                "base_step": None,
                "files": [
                    {"name": name, **spec} for name, spec in files.items()
                ],
            }
            committed = threading.Event()
            try:
                with mock.patch.object(
                    filesystem,
                    "_read_chunked_xor_manifest",
                    return_value={"mode": "full", "source_manifest": transport_manifest},
                ), mock.patch.object(
                    filesystem, "_source_manifest", return_value=new_manifest
                ), mock.patch.object(
                    filesystem, "download", side_effect=AssertionError("WAN download")
                ), mock.patch.object(filesystem, "_cleanup_previous_adapter_caches"):
                    destination = filesystem.stage_lora_adapter_files(
                        new_url,
                        committed_event=committed,
                    )
            finally:
                if old_cache_dir is None:
                    os.environ.pop("PRIME_LORA_CACHE_DIR", None)
                else:
                    os.environ["PRIME_LORA_CACHE_DIR"] = old_cache_dir
            self.assertEqual(destination, new_cache)
            self.assertTrue(committed.is_set())
            self.assertEqual(json.loads((new_cache / "manifest.json").read_text())["source_url"], new_url)
            self.assertEqual(
                json.loads((new_cache / "relay_manifest.json").read_text())["source_url"],
                new_url,
            )
            self.assertTrue((new_cache / "READY").is_file())
            self.assertEqual((new_cache / "adapter_model.safetensors").read_bytes(), model)
            self.assertNotEqual(
                (new_cache / "adapter_model.safetensors").stat().st_ino,
                (old_cache / "adapter_model.safetensors").stat().st_ino,
            )
            (new_cache / "adapter_model.safetensors").write_bytes(b"new-model")
            self.assertEqual((old_cache / "adapter_model.safetensors").read_bytes(), model)

    def test_full_cache_reuse_rejects_changed_identity_and_symlink(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_url = "http://source/old/run_old/broadcasts/step_0"
            new_url = "http://source/new/run_new/broadcasts/step_0"
            old_cache_dir = os.environ.get("PRIME_LORA_CACHE_DIR")
            os.environ["PRIME_LORA_CACHE_DIR"] = str(root)
            old_cache = filesystem.adapter_cache_path(old_url)
            new_cache = filesystem.adapter_cache_path(new_url)
            old_cache.mkdir(parents=True)
            config = b"{}\n"
            model = b"old"
            files = {
                "adapter_config.json": {"size": len(config), "sha256": hashlib.sha256(config).hexdigest()},
                "adapter_model.safetensors": {"size": len(model), "sha256": hashlib.sha256(model).hexdigest()},
            }
            (old_cache / "adapter_config.json").write_bytes(config)
            (old_cache / "adapter_model.safetensors").write_bytes(model)
            (old_cache / "manifest.json").write_text(json.dumps({"version": 1, "source_url": old_url, "files": files}))
            (old_cache / "adapter_model.safetensors.sha256").write_text(files["adapter_model.safetensors"]["sha256"] + "\n")
            (old_cache / "READY").write_text("ready\n")
            transport = {"mode": "full", "source_manifest": {"version": 1, "mode": "full", "target_step": 0, "files": [{"name": n, **s} for n, s in files.items()]}}
            try:
                changed = dict(files)
                changed["adapter_model.safetensors"] = {"size": 3, "sha256": "a" * 64}
                self.assertFalse(
                    filesystem._reuse_full_adapter_cache(
                        new_url,
                        new_cache,
                        {"version": 1, "source_url": new_url, "files": changed},
                        transport,
                    )
                )
                changed_config = dict(files)
                changed_config["adapter_config.json"] = {"size": 3, "sha256": "b" * 64}
                self.assertFalse(
                    filesystem._reuse_full_adapter_cache(
                        new_url,
                        new_cache,
                        {"version": 1, "source_url": new_url, "files": changed_config},
                        transport,
                    )
                )
                (old_cache / "adapter_model.safetensors").unlink()
                (old_cache / "adapter_model.safetensors").symlink_to(root / "outside")
                (root / "outside").write_bytes(model)
                with self.assertRaises(OSError):
                    filesystem._reuse_full_adapter_cache(
                        new_url,
                        new_cache,
                        {"version": 1, "source_url": new_url, "files": files},
                        transport,
                    )
            finally:
                if old_cache_dir is None:
                    os.environ.pop("PRIME_LORA_CACHE_DIR", None)
                else:
                    os.environ["PRIME_LORA_CACHE_DIR"] = old_cache_dir

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
            ["self", "source_url", "require_delta", "policy_step"],
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
        self.assertIn("preserve_partial=_preserve_adapter_stage_partial(exc)", worker_source)
        self.assertIn("deadline=operation_deadline", worker_source)
        self.assertIn("asyncio.wait_for(", server.read_text(encoding="utf-8"))
        watcher_source = watcher.read_text(encoding="utf-8")
        self.assertIn('manifest.get("version") != 1', watcher_source)
        policy_update = watcher_source[watcher_source.index('f"Policy update'):]
        self.assertIn("attempts=update_attempts", policy_update[:300])
        self.assertIn("2 * adapter_stage_limit_seconds", watcher_source)
        self.assertIn("delta_deadline = min(delta_deadline, update_deadline)", watcher_source)
        self.assertIn("for attempt in range(1, attempts + 1)", watcher_source)
        self.assertIn("failed on attempt {attempt}/{attempts}", watcher_source)
        self.assertIn("delta_deadline = time.monotonic()", watcher_source)
        self.assertIn("delta_deadline=delta_deadline if require_delta else None", watcher_source)
        self.assertIn("next_step > 0", watcher_source)

    def test_stage_failure_preserves_transport_but_removes_invalid_partial(self):
        filesystem = load_filesystem_module()
        self.assertTrue(
            filesystem._preserve_adapter_stage_partial(OSError("connection reset by peer"))
        )
        self.assertTrue(
            filesystem._preserve_adapter_stage_partial(TimeoutError("request timed out"))
        )
        for message in (
            "adapter range ended early",
            "adapter file ended early",
            "incomplete adapter range",
            "incomplete adapter file",
        ):
            with self.subTest(message=message):
                self.assertTrue(filesystem._preserve_adapter_stage_partial(OSError(message)))
        for message in (
            "adapter checksum mismatch",
            "adapter digest mismatch",
            "adapter identity is invalid",
            "adapter protocol error",
            "adapter path is a symbolic link",
            "adapter payload is corrupt",
        ):
            with self.subTest(message=message):
                self.assertFalse(filesystem._preserve_adapter_stage_partial(OSError(message)))

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

    def test_reader_lock_deadline_does_not_wait_behind_writer(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "adapter"
            lock_path = filesystem._adapter_reader_lock(destination)
            lock_path.touch()
            with lock_path.open("a+") as writer:
                filesystem.fcntl.flock(writer, filesystem.fcntl.LOCK_EX)
                with self.assertRaisesRegex(TimeoutError, "exceeded its time limit"):
                    with filesystem.adapter_reader_lock(
                        destination,
                        deadline=time.monotonic() + 0.01,
                    ):
                        pass
                filesystem.fcntl.flock(writer, filesystem.fcntl.LOCK_UN)

    def test_publish_lock_deadline_does_not_wait_behind_reader(self):
        filesystem = load_filesystem_module()
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "adapter.lock"
            with lock_path.open("a+") as reader:
                filesystem.fcntl.flock(reader, filesystem.fcntl.LOCK_SH)
                with lock_path.open("a+") as writer:
                    with self.assertRaisesRegex(TimeoutError, "exceeded its time limit"):
                        filesystem._acquire_stage_lock(
                            writer,
                            time.monotonic() + 0.01,
                        )
                filesystem.fcntl.flock(reader, filesystem.fcntl.LOCK_UN)

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
            os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = "24"
            self.assertEqual(filesystem._configured_download_streams(), 24)
        finally:
            if old_value is None:
                os.environ.pop("PRIME_LORA_DOWNLOAD_STREAMS", None)
            else:
                os.environ["PRIME_LORA_DOWNLOAD_STREAMS"] = old_value

    def test_delta_stage_timeout_defaults_without_operations_config(self):
        filesystem = load_filesystem_module()
        old_timeout = os.environ.pop("PRIME_LORA_DELTA_STAGE_TIMEOUT_S", None)
        old_operations = os.environ.get("PRIME_RL_OPERATIONS_CONFIG")
        try:
            os.environ["PRIME_RL_OPERATIONS_CONFIG"] = "/missing/operations.toml"
            self.assertEqual(filesystem._delta_stage_timeout_s(), 600.0)
        finally:
            if old_timeout is not None:
                os.environ["PRIME_LORA_DELTA_STAGE_TIMEOUT_S"] = old_timeout
            if old_operations is None:
                os.environ.pop("PRIME_RL_OPERATIONS_CONFIG", None)
            else:
                os.environ["PRIME_RL_OPERATIONS_CONFIG"] = old_operations

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

    def test_delta_default_deadline_and_status_use_policy_limit(self):
        filesystem = load_filesystem_module()
        old_cache = os.environ.get("PRIME_LORA_CACHE_DIR")
        old_timeout = filesystem._delta_stage_timeout_s
        old_manifest = filesystem._read_delta_manifest_with_visibility_wait
        old_cleanup = filesystem._cleanup_delta_stage_artifacts
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.environ["PRIME_LORA_CACHE_DIR"] = temporary
                filesystem._delta_stage_timeout_s = lambda: 900.0
                filesystem._cleanup_delta_stage_artifacts = lambda *_args, **_kwargs: True
                for step, limit in ((2, 600.0), (3, 300.0)):
                    deadlines = []

                    def fail_manifest(*_args, **kwargs):
                        deadlines.append(kwargs["deadline"])
                        raise OSError("manifest unavailable")

                    filesystem._read_delta_manifest_with_visibility_wait = fail_manifest
                    with self.assertRaisesRegex(OSError, "manifest unavailable"):
                        filesystem.stage_lora_adapter_files(
                            f"http://source/step_{step}", require_delta=True
                        )
                    self.assertEqual(len(deadlines), 1)
                    self.assertLessEqual(deadlines[0] - time.monotonic(), limit)
                    status = filesystem.adapter_stage_status()
                    self.assertEqual(status["policy_step"], step)
                    self.assertEqual(status["stage_timeout_seconds"], limit)
                    self.assertEqual(status["acceptance_target_seconds"], 300.0)
        finally:
            if old_cache is None:
                os.environ.pop("PRIME_LORA_CACHE_DIR", None)
            else:
                os.environ["PRIME_LORA_CACHE_DIR"] = old_cache
            filesystem._delta_stage_timeout_s = old_timeout
            filesystem._read_delta_manifest_with_visibility_wait = old_manifest
            filesystem._cleanup_delta_stage_artifacts = old_cleanup

    def test_stage_timeout_returns_only_after_worker_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "READY"
            started = threading.Event()

            def stage_worker(
                _source_url,
                *,
                require_delta,
                stage_timeout_seconds,
                policy_step,
                cancel_event,
                committed_event,
            ):
                self.assertTrue(require_delta)
                self.assertEqual(policy_step, 1)
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
                        "policy_step": 1,
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
                    policy_step,
                    cancel_event,
                    committed_event,
                ):
                    self.assertTrue(require_delta)
                    self.assertEqual(policy_step, 1)
                    self.assertEqual(stage_timeout_seconds, 0.1)
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
                            "policy_step": 1,
                            "stage_timeout_seconds": 0.1,
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
                    "policy_step": 1,
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
                policy_step,
                cancel_event,
                committed_event,
            ):
                self.assertFalse(require_delta)
                self.assertIsNone(policy_step)
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
                policy_step,
                cancel_event,
                committed_event,
            ):
                self.assertFalse(require_delta)
                self.assertIsNone(policy_step)
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
            self.assertEqual(response.status_code, 499)
            self.assertTrue(started.is_set())
            self.assertTrue(stopped.is_set())
            self.assertFalse(ready.exists())

    def test_inventory_defines_bounded_transfer(self):
        nodes = REPOSITORY / "configs/nodes.toml"
        with nodes.open("rb") as stream:
            config = tomllib.load(stream)
        ports = config["ports"]
        self.assertEqual(ports["lora_source_first"], 18100)
        self.assertEqual(ports["lora_source_last"], 18227)
        self.assertEqual(ports["adapter_relay"], 18228)
        self.assertEqual(ports["refusal_classifier"], 18250)
        self.assertEqual(ports["lora_source_last"] - ports["lora_source_first"] + 1, 128)
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
        config_path = REPOSITORY / "configs/general-agent/orchestrator.toml"
        with config_path.open("rb") as stream:
            edge_config = tomllib.load(stream)
        source_url = edge_config["weight_broadcast"]["source_url"]
        self.assertEqual(urlsplit(source_url).port, ports["lora_source_first"])
        runtime = REPOSITORY / "training_ops/runtime/h200"
        glm_source = (runtime / "start_inference.sh").read_text(encoding="utf-8")
        self.assertIn("download_streams=${PRIME_LORA_DOWNLOAD_STREAMS-96}", glm_source)
        self.assertIn('-e PRIME_LORA_DOWNLOAD_STREAMS="$download_streams"', glm_source)
        self.assertIn('PRIME_LORA_SOURCE_PORTS="$lora_source_port_list"', glm_source)
        source_generator = glm_source.split(
            'runtime_output=$(python3 - "$deployment/configs/nodes.toml" "$engine_host" <<\'PY\'\n',
            1,
        )[1].split("\nPY\n", 1)[0]
        source_result = subprocess.run(
            [sys.executable, "-", str(nodes), "h200a"],
            input=source_generator,
            text=True,
            capture_output=True,
            check=True,
        )
        generated_ports = [
            int(value) for value in source_result.stdout.splitlines()[0].split(",")
        ]
        self.assertEqual(
            generated_ports,
            [
                port
                for offset in range(2)
                for port in range(18100 + offset, 18228, 2)
            ],
        )
        orchestrator_source = (
            REPOSITORY / "training_ops/runtime/h200/start_orchestrator.sh"
        ).read_text(encoding="utf-8")
        for legacy_port in ("18101", "18111", "18112", "18113", "18114"):
            self.assertNotIn(legacy_port, orchestrator_source)
        self.assertIn("parsed.port != source_first", orchestrator_source)
        worker_source = (REPOSITORY / "prime-rl/src/prime_rl/inference/vllm/worker/filesystem.py").read_text(encoding="utf-8")
        self.assertIn("full_download_streams = 48", worker_source)
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
        old_responsive_lanes = filesystem._responsive_adapter_source_urls
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
        filesystem._responsive_adapter_source_urls = lambda *_args: (
            ["http://lane-1", "http://lane-2"],
            2,
        )
        filesystem._configured_download_streams = lambda: 1
        filesystem._decode_transport_chunk = lambda payload, **_kwargs: payload
        filesystem._transport_raw_sha256 = lambda payload, **_kwargs: hashlib.sha256(payload).hexdigest()
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
                self.assertGreaterEqual(filesystem.adapter_stage_status()["elapsed_seconds"], 0.0)
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
            filesystem._responsive_adapter_source_urls = old_responsive_lanes
            filesystem._configured_download_streams = old_streams
            filesystem._decode_transport_chunk = old_decode
            filesystem._transport_raw_sha256 = old_raw_digest
            filesystem._xor_bytes = old_xor
            filesystem.download = old_download
            filesystem._cleanup_delta_stage_artifacts = old_cleanup
            base_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
