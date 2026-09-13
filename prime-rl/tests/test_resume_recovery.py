import ast
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"
ORCHESTRATOR_PATH = ROOT / "src/prime_rl/orchestrator/orchestrator.py"
SERVER_PATH = ROOT / "src/prime_rl/inference/vllm/server.py"
WATCHER_PATH = ROOT / "src/prime_rl/orchestrator/watcher.py"
DISPATCHER_PATH = ROOT / "src/prime_rl/orchestrator/dispatcher.py"


class _StubValue:
    def __call__(self, *_args, **_kwargs):
        return self

    def __getattr__(self, _name):
        return self

    def __iter__(self):
        return iter(())


class _StubModule(types.ModuleType):
    def __getattr__(self, _name):
        return _StubValue()


def _load_stubbed_source(path, name, names):
    names = names.split()
    modules = {module_name: _StubModule(module_name) for module_name in names}
    for module_name in list(modules):
        parts = module_name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            modules.setdefault(parent, _StubModule(parent))
            modules[parent].__path__ = []
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


async def _safe_cancel(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _safe_cancel_all(tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _client_identity(client):
    return (client.base_url, client.headers.get("X-data-parallel-rank"))


@dataclass
class _RuntimeInflightRollout:
    kind: str
    env_name: str
    group_id: object
    policy_version: int
    rollout_count: int
    client_config: object = None
    off_policy_steps: int = 0
    eval_step: int | None = None


@dataclass
class _RuntimeGroupState:
    kind: str
    env_name: str
    task_idx: int
    rollouts_to_schedule: int
    target_rollouts: int
    task: object = None
    emitted: int = 0
    eval_step: int | None = None
    pinned_client: object = None
    policy_version_at_start: int = 0


def load_weight_watcher_runtime():
    names = """
        modelexpress modelexpress.p2p_pb2 prime_rl.configs.orchestrator
        prime_rl.orchestrator.types prime_rl.trainer.rl.broadcast.nixl.model_express
        prime_rl.utils.async_utils prime_rl.utils.checkpoint_delta prime_rl.utils.client
        prime_rl.utils.logger prime_rl.utils.pathing prime_rl.utils.utils
    """
    return _load_stubbed_source(WATCHER_PATH, "_resume_test_watcher", names)


def load_dispatcher_runtime():
    names = """
        aiolimiter prime_rl.orchestrator.envs prime_rl.orchestrator.eval_source
        prime_rl.orchestrator.train_source prime_rl.orchestrator.types
        prime_rl.utils.async_utils prime_rl.utils.client prime_rl.utils.logger verifiers.v1
    """
    module = _load_stubbed_source(DISPATCHER_PATH, "_resume_test_dispatcher", names)
    module.GroupState = _RuntimeGroupState
    module.InflightRollout = _RuntimeInflightRollout
    module.safe_cancel = _safe_cancel
    module.safe_cancel_all = _safe_cancel_all
    module.client_identity = _client_identity
    module.get_logger = lambda: SimpleNamespace(
        debug=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    return module


def load_orchestrator_runtime():
    names = """
        modelexpress modelexpress.client prime_rl._compat prime_rl.configs.orchestrator
        prime_rl.orchestrator.ckpt prime_rl.orchestrator.dispatcher prime_rl.orchestrator.envs
        prime_rl.orchestrator.eval_sink prime_rl.orchestrator.eval_source prime_rl.orchestrator.filters
        prime_rl.orchestrator.inference_metrics prime_rl.orchestrator.patches
        prime_rl.orchestrator.periodic_logger prime_rl.orchestrator.train_sink
        prime_rl.orchestrator.train_source prime_rl.orchestrator.types
        prime_rl.orchestrator.utils prime_rl.orchestrator.watcher prime_rl.trainer.model
        prime_rl.trainer.rl.broadcast.nixl.model_express prime_rl.transport
        prime_rl.utils.async_utils prime_rl.utils.client prime_rl.utils.config
        prime_rl.utils.heartbeat prime_rl.utils.logger prime_rl.utils.monitor
        prime_rl.utils.pathing prime_rl.utils.usage_reporter prime_rl.utils.utils
        tomli_w verifiers.v1
    """
    return _load_stubbed_source(ORCHESTRATOR_PATH, "_resume_test_orchestrator", names)


def load_startup_validator(urlopen, validate_manifest, file_records):
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_verify_startup_adapter_delta"
    )
    namespace = {
        "OrchestratorConfig": object,
        "json": json,
        "time": time,
        "urlopen": urlopen,
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "inference_weight_path": lambda _config, path, _lora_name: (
            f"http://source/{path.name}"
        ),
        "_delta_stage_timeout_s": lambda: 30.0,
        "adapter_stage_limit_seconds": lambda step: {
            1: 600.0,
            2: 600.0,
        }.get(step, 300.0),
        "validate_adapter_manifest": validate_manifest,
        "_adapter_file_records": file_records,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORCHESTRATOR_PATH), "exec"), namespace)
    return namespace["_verify_startup_adapter_delta"]


def load_server_load_endpoints(destination, handler, reader_lock=None):
    source = SERVER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "_adapter_requires_routed_experts",
        "_is_finite_number",
        "_read_verified_adapter_identity",
        "_set_moe_lora_mode",
        "_target_has_routed_experts",
        "_validate_adapter_target_compatibility",
        "load_lora_adapter",
        "staging_status",
    }
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]

    class FakeRouter:
        def get(self, *_args, **_kwargs):
            return lambda function: function

        def post(self, *_args, **_kwargs):
            return lambda function: function

    class FakeJSONResponse:
        def __init__(self, content, status_code):
            self.content = content
            self.status_code = status_code

    class FakeErrorResponse:
        def __init__(self, code=422):
            self.error = SimpleNamespace(code=code)

        def model_dump(self):
            return {"error": "backend load failed"}

    def read_text_nofollow(path, *, encoding, dir_fd=None, max_bytes=None):
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd), "rb") as source:
            value = source.read((max_bytes + 1) if max_bytes is not None else -1)
        if max_bytes is not None and len(value) > max_bytes:
            raise OSError("fixture metadata is too large")
        return value.decode(encoding)

    def open_directory_nofollow(path, *, create, label):
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def stat_regular_entry(parent_fd, name, label):
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(entry.st_mode):
            raise OSError(f"fixture {label} is not regular")
        return entry

    def adapter_stage_status():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        return {
            "status": "ready",
            "source_url": manifest["source_url"],
            "path": destination.as_posix(),
            "stage_id": "fixture-stage",
            "pid": os.getpid(),
            "target_sha256": manifest["files"]["adapter_model.safetensors"]["sha256"],
            "verified_files": manifest["files"],
            "file_stats": {
                name: {
                    "dev": (destination / name).stat().st_dev,
                    "ino": (destination / name).stat().st_ino,
                    "size": (destination / name).stat().st_size,
                    "mtime_ns": (destination / name).stat().st_mtime_ns,
                }
                for name in ("adapter_config.json", "adapter_model.safetensors")
            },
        }

    namespace = {
        "Any": Any,
        "OpenAIServingModels": object,
        "Path": Path,
        "os": os,
        "stat": __import__("stat"),
        "_ADAPTER_MARKER_MAX_BYTES": 4 * 1024,
        "_ADAPTER_METADATA_MAX_BYTES": 2 * 1024 * 1024,
        "_open_directory_nofollow": open_directory_nofollow,
        "_read_text_nofollow": read_text_nofollow,
        "_stat_regular_entry": stat_regular_entry,
        "LoadLoRAAdapterRequest": object,
        "Request": object,
        "adapter_cache_path": lambda _source_url: destination,
        "adapter_reader_lock": reader_lock or contextlib.nullcontext,
        "adapter_stage_status": adapter_stage_status,
        "_adapter_step_from_source_url": lambda source_url: int(
            re.fullmatch(r"step_(\d+)", Path(source_url.rstrip("/")).name).group(1)
        ),
        "adapter_stage_limit_seconds": lambda step: {
            1: 600.0,
            2: 600.0,
        }.get(step, 300.0),
        "asyncio": asyncio,
        "ErrorResponse": FakeErrorResponse,
        "JSONResponse": FakeJSONResponse,
        "hashlib": hashlib,
        "json": json,
        "models": lambda _request: handler,
        "re": re,
        "router": FakeRouter(),
        "stage_lora_adapter_files": lambda *_args, **_kwargs: destination,
        "_begin_lora_stage": lambda: None,
        "_combined_lora_stage_status": lambda active, durable: {**active, **durable},
        "_end_lora_stage": lambda: None,
        "_lora_stage_status": lambda: {"status": "idle"},
        "_loaded_lora_receipts": {},
        "_server_instance_id": "server-a",
        "lora_load_lock": asyncio.Lock(),
    }
    async def stage_lora_adapter(request):
        data = await request.json()
        destination = namespace["stage_lora_adapter_files"](
            data["lora_path"],
            require_delta=data.get("require_delta", False),
            policy_step=data.get("policy_step"),
            stage_timeout_seconds=(
                namespace["adapter_stage_limit_seconds"](data["policy_step"])
                if data.get("policy_step") is not None
                else None
            ),
        )
        return {"status": "ok", "path": str(destination)}
    namespace["stage_lora_adapter"] = stage_lora_adapter
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SERVER_PATH), "exec"), namespace)
    return (
        namespace["load_lora_adapter"],
        namespace["staging_status"],
        namespace,
        FakeJSONResponse,
        FakeErrorResponse,
    )


def write_cached_adapter(
    destination,
    source_url,
    target_sha256="a" * 64,
    transport=None,
    target_modules=("o_proj",),
):
    destination.mkdir(parents=True)
    config = json.dumps({"target_modules": list(target_modules)}).encode("utf-8")
    weights = b"weights"
    (destination / "READY").write_text("ready\n", encoding="ascii")
    (destination / "adapter_config.json").write_bytes(config)
    (destination / "adapter_model.safetensors").write_bytes(weights)
    (destination / "adapter_model.safetensors.sha256").write_text(target_sha256 + "\n", encoding="ascii")
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_url": source_url,
                "files": {
                    "adapter_config.json": {
                        "size": len(config),
                        "sha256": hashlib.sha256(config).hexdigest(),
                    },
                    "adapter_model.safetensors": {"size": len(weights), "sha256": target_sha256},
                },
            }
        ),
        encoding="utf-8",
    )
    if transport is not None:
        (destination / "transport.json").write_text(json.dumps(transport), encoding="utf-8")


class ResumeRecoveryTest(unittest.TestCase):
    def test_server_load_receipt_requires_backend_success_and_exact_slot(self):
        source_url = "http://source/step_2"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "cache"
            write_cached_adapter(
                destination,
                source_url,
                transport={
                    "mode": "delta",
                    "source_url": source_url,
                    "base_url": "http://source/step_1",
                    "base_sha256": "c" * 64,
                    "base2_url": "http://source/step_0",
                    "base2_sha256": "d" * 64,
                    "target_sha256": "a" * 64,
                },
            )

            class Request:
                async def json(self):
                    return {"require_delta": False, "prefetched": True}

            class Handler:
                def __init__(self):
                    self.lora_requests = {}
                    self.engine_client = SimpleNamespace(
                        vllm_config=SimpleNamespace(
                            lora_config=SimpleNamespace(target_modules=None)
                        )
                    )
                    self.response = object()
                    self.path_override = None
                    self.active = 0
                    self.max_active = 0

                async def load_lora_adapter(self, request):
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    try:
                        await asyncio.sleep(0.005)
                        if isinstance(self.response, Exception):
                            return self.response
                        self.lora_requests[request.lora_name] = SimpleNamespace(
                            lora_name=request.lora_name,
                            lora_path=self.path_override or request.lora_path,
                            load_inplace=request.load_inplace,
                        )
                        return self.response
                    finally:
                        self.active -= 1

            handler = Handler()
            load, status, namespace, response_type, error_type = load_server_load_endpoints(
                destination, handler
            )
            stage_calls = []
            namespace["stage_lora_adapter_files"] = lambda *args, **kwargs: (
                stage_calls.append((args, kwargs)) or destination
            )

            class LoadRequest:
                def __init__(self, name="adapter-slot1", path=source_url):
                    self.lora_name = name
                    self.lora_path = path
                    self.load_inplace = False

            class UnprefetchedDeltaRequest(Request):
                async def json(self):
                    return {"require_delta": True, "prefetched": False}

            success = asyncio.run(load(LoadRequest(), Request()))
            self.assertEqual(success["status"], "ok")
            self.assertEqual(success["loaded_receipt"]["source_url"], source_url)
            self.assertEqual(
                success["loaded_receipt"]["config_sha256"],
                hashlib.sha256(b'{"target_modules": ["o_proj"]}').hexdigest(),
            )
            self.assertEqual(namespace["_loaded_lora_receipts"]["adapter-slot1"], success["loaded_receipt"])
            self.assertEqual(asyncio.run(status(Request()))["loaded_receipts"]["adapter-slot1"], success["loaded_receipt"])

            delta_success = asyncio.run(load(LoadRequest(), UnprefetchedDeltaRequest()))
            self.assertEqual(delta_success["status"], "ok")
            self.assertEqual(len(stage_calls), 1)
            self.assertEqual(stage_calls[0][1]["policy_step"], 2)
            self.assertEqual(stage_calls[0][1]["stage_timeout_seconds"], 600.0)

            async def status_during_staging():
                stage_started = threading.Event()

                def slow_stage(*_args, **_kwargs):
                    stage_started.set()
                    time.sleep(0.05)
                    return destination

                namespace["stage_lora_adapter_files"] = slow_stage

                class UnprefetchedRequest(Request):
                    async def json(self):
                        return {"require_delta": False, "prefetched": False}

                task = asyncio.create_task(load(LoadRequest(), UnprefetchedRequest()))
                if not await asyncio.to_thread(stage_started.wait, 1.0):
                    raise AssertionError("staging did not start")
                state = await status(Request())
                await task
                return state

            in_progress = asyncio.run(status_during_staging())
            self.assertNotIn("adapter-slot1", in_progress["loaded_receipts"])

            (destination / "adapter_config.json").write_bytes(b"[]")
            with self.assertRaisesRegex(RuntimeError, "adapter_config.json identity is invalid"):
                asyncio.run(load(LoadRequest(), Request()))
            (destination / "adapter_config.json").write_bytes(
                b'{"target_modules": ["o_proj"]}'
            )

            handler.response = error_type()
            failed = asyncio.run(load(LoadRequest(), Request()))
            self.assertIsInstance(failed, response_type)
            self.assertEqual(failed.status_code, 422)
            self.assertNotIn("adapter-slot1", namespace["_loaded_lora_receipts"])

            handler.response = object()
            local = asyncio.run(load(LoadRequest(path=str(destination)), Request()))
            self.assertEqual(local["status"], "ok")
            self.assertNotIn("adapter-slot1", namespace["_loaded_lora_receipts"])

            remote = asyncio.run(load(LoadRequest(), Request()))
            self.assertEqual(remote["status"], "ok")
            handler.lora_requests.pop("adapter-slot1")
            self.assertNotIn("adapter-slot1", asyncio.run(status(Request()))["loaded_receipts"])
            self.assertNotIn("adapter-slot1", namespace["_loaded_lora_receipts"])

            handler.response = object()
            handler.path_override = "/wrong/path"
            mismatched = asyncio.run(load(LoadRequest("adapter-slot-mismatch"), Request()))
            self.assertIsInstance(mismatched, response_type)
            self.assertEqual(mismatched.status_code, 500)
            self.assertNotIn("adapter-slot-mismatch", namespace["_loaded_lora_receipts"])
            handler.path_override = None

            async def concurrent_loads():
                return await asyncio.gather(
                    load(LoadRequest("adapter-slot-concurrent"), Request()),
                    load(LoadRequest("adapter-slot-concurrent"), Request()),
                )

            concurrent = asyncio.run(concurrent_loads())
            self.assertEqual([item["status"] for item in concurrent], ["ok", "ok"])
            self.assertEqual(handler.max_active, 1)

            restarted_handler = Handler()
            _, restarted_status, restarted_namespace, _, _ = load_server_load_endpoints(
                destination, restarted_handler
            )
            self.assertEqual(asyncio.run(restarted_status(Request()))["loaded_receipts"], {})
            self.assertEqual(restarted_namespace["_loaded_lora_receipts"], {})

    def test_server_rejects_routed_expert_adapter_when_targets_exclude_experts(self):
        source_url = "http://source/expert"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "cache"
            write_cached_adapter(destination, source_url, target_modules=("experts", "o_proj"))

            class Handler:
                def __init__(self):
                    self.lora_requests = {}
                    self.loaded = False
                    self.engine_client = SimpleNamespace(
                        vllm_config=SimpleNamespace(
                            lora_config=SimpleNamespace(target_modules=["o_proj"])
                        )
                    )

                async def load_lora_adapter(self, _request):
                    self.loaded = True
                    return object()

            class Request:
                async def json(self):
                    return {"require_delta": False, "prefetched": True}

            class LoadRequest:
                lora_name = "expert-adapter"
                lora_path = source_url
                load_inplace = False

            handler = Handler()
            load, _, _, response_type, _ = load_server_load_endpoints(destination, handler)
            result = asyncio.run(load(LoadRequest(), Request()))
            self.assertIsInstance(result, response_type)
            self.assertEqual(result.status_code, 400)
            self.assertIn("routed experts", result.content["error"])
            self.assertFalse(handler.loaded)
            local_request = LoadRequest()
            local_request.lora_path = str(destination)
            local_result = asyncio.run(load(local_request, Request()))
            self.assertIsInstance(local_result, response_type)
            self.assertEqual(local_result.status_code, 400)
            self.assertFalse(handler.loaded)
            handler.engine_client.vllm_config.lora_config = None
            disabled = asyncio.run(load(LoadRequest(), Request()))
            self.assertIsInstance(disabled, response_type)
            self.assertEqual(disabled.status_code, 400)
            self.assertIn("LoRA is disabled", disabled.content["error"])

    def test_server_allows_expert_attention_and_unrestricted_targets(self):
        cases = [
            (("experts", "o_proj"), ["experts", "o_proj"]),
            (("o_proj",), ["o_proj"]),
            (("experts",), None),
        ]
        for index, (adapter_targets, configured_targets) in enumerate(cases):
            with self.subTest(index=index):
                source_url = f"http://source/case-{index}"
                with tempfile.TemporaryDirectory() as temporary:
                    destination = Path(temporary) / "cache"
                    write_cached_adapter(
                        destination, source_url, target_modules=adapter_targets
                    )

                    class Handler:
                        def __init__(self):
                            self.lora_requests = {}
                            self.engine_client = SimpleNamespace(
                                vllm_config=SimpleNamespace(
                                    lora_config=SimpleNamespace(
                                        target_modules=configured_targets
                                    )
                                )
                            )

                        async def load_lora_adapter(self, request):
                            self.lora_requests[request.lora_name] = SimpleNamespace(
                                lora_name=request.lora_name,
                                lora_path=request.lora_path,
                            )
                            return object()

                    class Request:
                        async def json(self):
                            return {"require_delta": False, "prefetched": True}

                    class LoadRequest:
                        lora_name = "adapter"
                        lora_path = source_url
                        load_inplace = False

                    handler = Handler()
                    load, _, _, _, _ = load_server_load_endpoints(destination, handler)
                    result = asyncio.run(load(LoadRequest(), Request()))
                    self.assertEqual(result["status"], "ok")

    def test_server_validates_remote_adapter_inside_reader_lock(self):
        source_url = "http://source/locked"
        lock_active = False
        validator_checked_in_lock = False

        @contextlib.contextmanager
        def reader_lock(_destination):
            nonlocal lock_active
            lock_active = True
            try:
                yield
            finally:
                lock_active = False

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "cache"
            write_cached_adapter(destination, source_url, target_modules=("o_proj",))

            class Handler:
                def __init__(self):
                    self.lora_requests = {}
                    self.engine_client = SimpleNamespace(
                        vllm_config=SimpleNamespace(
                            lora_config=SimpleNamespace(target_modules=["o_proj"])
                        )
                    )

                async def load_lora_adapter(self, request):
                    self.lora_requests[request.lora_name] = SimpleNamespace(
                        lora_name=request.lora_name,
                        lora_path=request.lora_path,
                    )
                    return object()

            class Request:
                async def json(self):
                    return {"require_delta": False, "prefetched": True}

            class LoadRequest:
                lora_name = "locked-adapter"
                lora_path = source_url
                load_inplace = False

            handler = Handler()
            load, _, namespace, _, _ = load_server_load_endpoints(
                destination, handler, reader_lock
            )
            original_validator = namespace["_validate_adapter_target_compatibility"]

            def checked_validator(path, target_handler):
                nonlocal validator_checked_in_lock
                validator_checked_in_lock = lock_active
                return original_validator(path, target_handler)

            namespace["_validate_adapter_target_compatibility"] = checked_validator
            result = asyncio.run(load(LoadRequest(), Request()))
            self.assertEqual(result["status"], "ok")
            self.assertTrue(validator_checked_in_lock)

    def test_server_uses_effective_lora_targets_for_moe_mode(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        helpers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_set_moe_lora_mode", "_target_has_routed_experts", "_is_finite_number"}
        ]
        namespace = {"os": os, "math": math}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(SERVER_PATH), "exec"), namespace)
        set_mode = namespace["_set_moe_lora_mode"]

        self.assertTrue(namespace["_is_finite_number"](1.0))
        self.assertFalse(namespace["_is_finite_number"](float("nan")))
        self.assertFalse(namespace["_is_finite_number"]("1"))
        os.environ.pop("PRIME_NO_MOE_LORA", None)
        set_mode(["o_proj"])
        self.assertEqual(os.environ.get("PRIME_NO_MOE_LORA"), "1")
        set_mode(["experts"])
        self.assertNotIn("PRIME_NO_MOE_LORA", os.environ)
        set_mode(None)
        self.assertNotIn("PRIME_NO_MOE_LORA", os.environ)
        self.assertGreater(
            source.index("_set_moe_lora_mode(args.lora_target_modules)"),
            source.index("parse_args(args=[], namespace=namespace)"),
        )

    def test_resume_uses_target_identity_when_public_predecessors_are_cleaned(self):
        target_url = "http://source/step_6"
        manifest = {
            "version": 2,
            "kind": "lora-adapter",
            "mode": "delta",
            "transport": "chunked-bf16-linear-zigzag-sparse-zstd",
            "target_step": 6,
            "base_step": 5,
            "base2_step": 4,
            "target_url": target_url,
            "base_url": "http://source/step_5",
            "base2_url": "http://source/step_4",
            "target_size": 8,
            "target_sha256": "a" * 64,
            "config_size": 2,
            "config_sha256": "b" * 64,
            "base_size": 8,
            "base_sha256": "c" * 64,
            "base_config_size": 2,
            "base_config_sha256": "d" * 64,
            "base2_size": 8,
            "base2_sha256": "e" * 64,
            "base2_config_size": 2,
            "base2_config_sha256": "f" * 64,
            "data_offset": 0,
            "files": [
                {"name": "adapter_config.json", "size": 2, "sha256": "b" * 64},
                {"name": "adapter_model.safetensors", "size": 8, "sha256": "a" * 64},
            ],
        }
        calls = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fetch(url, timeout):
            calls.append((url, timeout))
            return Response(json.dumps(manifest).encode("utf-8"))

        def validate(value, **kwargs):
            self.assertEqual(value, manifest)
            self.assertEqual(kwargs["target_step"], 6)
            self.assertEqual(kwargs["base_url"], "http://source/step_5")
            self.assertEqual(kwargs["base2_url"], "http://source/step_4")

        def records(value):
            return {item["name"]: item for item in value["files"]}

        validator = load_startup_validator(fetch, validate, records)
        with tempfile.TemporaryDirectory() as temporary:
            validator(SimpleNamespace(output_dir=Path(temporary)), "adapter", 6, Path(temporary) / "step_6", target_url)
        self.assertEqual(calls, [(target_url + "/adapter_manifest.json", 300.0)])

    def test_resume_manifest_probe_uses_exact_policy_step_limit(self):
        for step, expected_limit in ((1, 600.0), (2, 600.0), (3, 300.0)):
            with self.subTest(step=step):
                calls = []

                class Response(io.BytesIO):
                    def __enter__(self):
                        return self

                    def __exit__(self, *_args):
                        self.close()

                def fetch(url, timeout):
                    calls.append((url, timeout))
                    return Response(json.dumps({"version": 2}).encode("utf-8"))

                validator = load_startup_validator(fetch, lambda *_args, **_kwargs: None, lambda _value: {})
                with tempfile.TemporaryDirectory() as temporary:
                    target_url = f"http://source/step_{step}"
                    validator(
                        SimpleNamespace(output_dir=Path(temporary)),
                        "adapter",
                        step,
                        Path(temporary) / f"step_{step}",
                        target_url,
                    )
                self.assertEqual(
                    calls,
                    [(target_url + "/adapter_manifest.json", expected_limit)],
                )

    def test_resume_uses_the_per_step_phase_budget(self):
        tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("RESUME_"):
                    values[target.id] = ast.literal_eval(node.value)

        self.assertEqual(values["RESUME_ATTEMPT_TIMEOUT_S"], 600.0)
        self.assertEqual(values["RESUME_TOTAL_TIMEOUT_S"], 600.0)
        self.assertEqual(values["RESUME_CONFIRM_TIMEOUT_S"], 600.0)

    def test_resume_timeout_is_confirmed_before_failure(self):
        source = CLIENT_PATH.read_text(encoding="utf-8")
        resume = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_resume_engines"
            ),
        )
        self.assertIn('"/resume_status"', resume)
        self.assertIn('"/liveness"', resume)
        self.assertIn('state == "running"', resume)
        self.assertIn("timed out, but", resume)
        self.assertIn("was not confirmed", resume)
        self.assertIn("route_missing and await _is_live(client, confirm_deadline)", resume)
        self.assertIn("except (TimeoutError, httpx.TimeoutException, httpx.TransportError)", resume)

    def test_server_exposes_explicit_resume_state(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('@router.get("/resume_status")', source)
        self.assertIn('_set_engine_state("resuming")', source)
        self.assertIn('_set_engine_state("running")', source)
        self.assertIn('_set_engine_state("unknown")', source)

    def test_staging_status_does_not_hide_an_active_stage(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("def _combined_lora_stage_status(", source)
        self.assertIn('if active_status == "staging":', source)
        self.assertIn('state["status"] = "staging"', source)
        self.assertIn('durable_status in {"failed", "ready"}', source)
        self.assertIn('elif durable_status == "staging":', source)
        self.assertIn('state["status"] = "failed"', source)

    def test_active_stage_wins_over_terminal_transfer_record(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        function = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
            and node.name == "_combined_lora_stage_status"
        )
        namespace = {"Any": Any}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SERVER_PATH), "exec"), namespace)
        combine = namespace["_combined_lora_stage_status"]

        active = combine(
            {"status": "staging", "age_seconds": 3.5, "stage_id": "new", "pid": 20},
            {"status": "failed", "stage_id": "new", "pid": 20, "error": "current transfer"},
        )
        self.assertEqual(active["status"], "staging")
        self.assertEqual(active["age_seconds"], 3.5)
        self.assertEqual(active["error"], "current transfer")
        stale = combine(
            {"status": "staging", "age_seconds": 1.0, "stage_id": "new", "pid": 20},
            {"status": "failed", "stage_id": "old", "pid": 10, "error": "old transfer"},
        )
        self.assertEqual(stale, {"status": "staging", "age_seconds": 1.0, "stage_id": "new", "pid": 20})
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "ready"})["status"],
            "ready",
        )
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "failed"})["status"],
            "failed",
        )
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "staging"})["status"],
            "failed",
        )
        self.assertEqual(
            combine({"status": "idle", "age_seconds": 0.0}, {"status": "corrupt"})["status"],
            "unknown",
        )

    async def _run_policy_update_ready_case(self, *, fail_ready):
        watcher_module = load_weight_watcher_runtime()
        dispatcher_module = load_dispatcher_runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (0, 1):
                (root / f"step_{step}").mkdir()
                (root / f"step_{step}" / "STABLE").touch()
            config = SimpleNamespace(output_dir=root, max_steps=None)
            policy = SimpleNamespace(version=0, model_name="base-model")
            client = SimpleNamespace(base_url="http://inference", headers={})
            ready_started, ready_release = asyncio.Event(), asyncio.Event()
            updates = []
            identity = {
                "source_url": "http://source/step_1",
                "target_sha256": "c" * 64,
                "target_size": 8,
                "config_sha256": "d" * 64,
                "config_size": 2,
            }

            async def stage(*_args, **_kwargs):
                return {
                    "transfer_wall_seconds": 0.0,
                    "origin_stage_seconds": 0.0,
                    "peer_stage_seconds": 0.0,
                    "reconstruction_wall_seconds": 0.0,
                    "adapter_identity": identity,
                }

            async def update_weights(weights_path, **kwargs):
                updates.append((weights_path, kwargs))
                return []

            async def wait_for_ready(_name):
                ready_started.set()
                await ready_release.wait()
                if fail_ready:
                    raise RuntimeError("policy v1 is not visible")

            inference = SimpleNamespace(
                admin_clients=[client], adapter_origin_base_urls=None,
                model_name="base-model", update_weights=update_weights,
                update_model_name=lambda name: setattr(inference, "model_name", name),
                wait_for_ready=wait_for_ready,
            )
            watcher_module.get_broadcast_dir = lambda path: Path(path)
            watcher_module.get_step_path = lambda path, step: Path(path) / f"step_{step}"
            watcher_module.inference_weight_path = lambda _config, path, _lora: f"http://source/{Path(path).name}"
            watcher_module.adapter_stage_limit_seconds = lambda _step: 1.0
            watcher_module.stage_lora_adapter = stage
            watcher_module._write_receivers_ready = lambda *_args, **_kwargs: None
            watcher_module.urlopen = lambda *_args, **_kwargs: io.BytesIO(
                json.dumps({"version": 1, "kind": "lora-adapter", "mode": "delta",
                            "transport": "chunked-shuffled-xor-zstd", "target_step": 1,
                            "base_step": 0}).encode("utf-8")
            )
            watcher_module.format_time = lambda value: f"{value:.3f}s"
            watcher = watcher_module.WeightWatcher(
                config, policy=policy, inference=inference, observers=[], lora_name="adapter",
                ckpt_step=0, poll_interval=0.001, reconcile_interval=100.0,
            )
            watcher.bootstrap_staged = True
            watcher.verified_adapter_step = 0
            watcher.verified_adapter_url = "http://source/step_0"
            pool = SimpleNamespace(train_clients=[])
            envs = SimpleNamespace(names=set(), get=lambda _name: None)
            source = SimpleNamespace(next_example=lambda *_args, **_kwargs: None)
            dispatcher = dispatcher_module.RolloutDispatcher(
                train_envs=envs, eval_envs=None, train_source=source, eval_source=None,
                policy_pool=pool, policy=policy, max_inflight_rollouts=1,
                tasks_per_minute=None, max_off_policy_steps=1,
            )
            watcher.observers = [dispatcher]
            update_task = asyncio.create_task(watcher.apply_policy_update(1))
            await asyncio.wait_for(ready_started.wait(), 1.0)
            self.assertEqual(len(updates), 1)
            self.assertTrue(dispatcher.weight_update_pending)
            self.assertEqual(dispatcher.policy_group_drain_pending, False)
            ready_release.set()
            if fail_ready:
                with self.assertRaisesRegex(RuntimeError, "policy v1 is not visible"):
                    await asyncio.wait_for(update_task, 1.0)
                self.assertTrue(dispatcher.weight_update_pending)
            else:
                await asyncio.wait_for(update_task, 1.0)
                self.assertFalse(dispatcher.weight_update_pending)
                self.assertEqual(policy.version, 1)
                self.assertEqual(policy.model_name, "adapter-slot1")

    def test_policy_one_waits_for_ready_before_reopening_rollouts(self):
        asyncio.run(self._run_policy_update_ready_case(fail_ready=False))

    def test_policy_one_wait_failure_stays_fail_closed(self):
        asyncio.run(self._run_policy_update_ready_case(fail_ready=True))

    def test_background_dispatch_stays_base_until_verified_bootstrap(self):
        watcher_module = load_weight_watcher_runtime()
        dispatcher_module = load_dispatcher_runtime()

        async def run_test():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for step in (0, 1):
                    (root / f"step_{step}").mkdir()
                    (root / f"step_{step}" / "STABLE").touch()
                config = SimpleNamespace(output_dir=root, max_steps=None)
                policy = SimpleNamespace(version=0, model_name="base-model")
                client = SimpleNamespace(base_url="http://inference", headers={})
                stage_started, stage_release = asyncio.Event(), asyncio.Event()
                stage_returned = asyncio.Event()
                rollout_started, rollout_release = asyncio.Event(), asyncio.Event()
                update_called = asyncio.Event()
                stage_calls, stage_results, rollout_models, updates = [], [], [], []
                identities = [
                    {"source_url": f"http://source/step_{step}", "target_sha256": digest * 64,
                     "target_size": 7 + step, "config_sha256": config_digest * 64, "config_size": 2}
                    for step, digest, config_digest in ((0, "a", "b"), (1, "c", "d"))
                ]
                watcher_holder = {}

                async def update_weights(weights_path, **kwargs):
                    self.assertTrue(stage_returned.is_set())
                    updates.append((weights_path, kwargs))
                    update_called.set()
                    watcher_holder["watcher"].stopped.set()
                    return []

                inference = SimpleNamespace(
                    admin_clients=[client], adapter_origin_base_urls=None,
                    model_name="base-model", update_weights=update_weights,
                    update_model_name=lambda name: setattr(inference, "model_name", name),
                    wait_for_ready=lambda _name: asyncio.sleep(0),
                )

                async def stage(_clients, path, **kwargs):
                    stage_calls.append((path, kwargs))
                    if len(stage_calls) == 1:
                        self.assertFalse(kwargs["require_delta"])
                        stage_started.set()
                        await stage_release.wait()
                    else:
                        self.assertTrue(kwargs["require_delta"])
                    result = {
                        "transfer_wall_seconds": 0.0,
                        "origin_stage_seconds": 0.0,
                        "peer_stage_seconds": 0.0,
                        "reconstruction_wall_seconds": 0.0,
                        "adapter_identity": identities[len(stage_calls) - 1],
                    }
                    if len(stage_calls) == 1:
                        stage_returned.set()
                    stage_results.append(result)
                    return result

                watcher_module.get_broadcast_dir = lambda path: Path(path)
                watcher_module.get_step_path = lambda path, step: Path(path) / f"step_{step}"
                watcher_module.inference_weight_path = lambda _config, path, _lora: f"http://source/{Path(path).name}"
                watcher_module.get_latest_ckpt_step = lambda _path: 1
                watcher_module.adapter_stage_limit_seconds = lambda _step: 1.0
                watcher_module.stage_lora_adapter = stage
                watcher_module._write_receivers_ready = lambda *_args, **_kwargs: None
                watcher_module.urlopen = lambda *_args, **_kwargs: io.BytesIO(
                    json.dumps({"version": 1, "kind": "lora-adapter", "mode": "delta",
                                "transport": "chunked-shuffled-xor-zstd", "target_step": 1,
                                "base_step": 0}).encode("utf-8")
                )
                watcher_module.format_time = lambda value: f"{value:.3f}s"
                watcher = watcher_module.WeightWatcher(
                    config,
                    policy=policy,
                    inference=inference,
                    observers=[],
                    lora_name="adapter",
                    ckpt_step=0,
                    poll_interval=0.001,
                    reconcile_interval=100.0,
                )
                watcher_holder["watcher"] = watcher

                pool = SimpleNamespace(train_clients=[client])
                pool.select_train_client = lambda _load: asyncio.sleep(0, result=client)
                async def rollout(*, model_name, **_kwargs):
                    rollout_models.append(model_name)
                    rollout_started.set()
                    await rollout_release.wait()
                    return []
                env = SimpleNamespace(
                    config=SimpleNamespace(group_size=1), requires_group_scoring=False,
                    sampler=SimpleNamespace(samples_from_live_policy=True, pool=pool), run=rollout,
                )
                envs = SimpleNamespace(names={"train"}, get=lambda _name: env)
                source = SimpleNamespace(used=False)
                source.next_example = lambda *_args, **_kwargs: (
                    None if source.used else (setattr(source, "used", True) or {"env_name": "train", "task_idx": 1})
                )

                dispatcher = dispatcher_module.RolloutDispatcher(
                    train_envs=envs,
                    eval_envs=None,
                    train_source=source,
                    eval_source=None,
                    policy_pool=pool,
                    policy=policy,
                    max_inflight_rollouts=1,
                    tasks_per_minute=None,
                    max_off_policy_steps=1,
                )
                watcher_task = asyncio.create_task(watcher.start())
                dispatcher_task = asyncio.create_task(dispatcher.start())
                try:
                    await asyncio.wait_for(stage_started.wait(), 1.0)
                    await asyncio.wait_for(rollout_started.wait(), 1.0)
                    self.assertEqual(rollout_models, ["base-model"])
                    self.assertEqual(policy.version, 0)
                    self.assertEqual(policy.model_name, "base-model")
                    self.assertEqual(updates, [])
                    self.assertEqual(len(stage_calls), 1)
                    self.assertEqual(stage_calls[0][0], "http://source/step_0")
                    stage_release.set()
                    await asyncio.wait_for(update_called.wait(), 1.0)
                    self.assertEqual(len(stage_calls), 2)
                    self.assertEqual(stage_calls[1][0], "http://source/step_1")
                    self.assertEqual(stage_calls[0][1]["require_delta"], False)
                    self.assertEqual(stage_calls[0][1].get("policy_step"), None)
                    self.assertEqual(stage_calls[1][1]["policy_step"], 1)
                    self.assertEqual(stage_results[0]["adapter_identity"], identities[0])
                    self.assertEqual(updates[0][1]["step"], 1)
                    self.assertEqual(updates[0][1]["expected_adapter_identity"], identities[1])
                    self.assertEqual(policy.version, 1)
                    self.assertEqual(policy.model_name, "adapter-slot1")
                    self.assertEqual(watcher.verified_adapter_step, 1)
                    self.assertEqual(watcher.verified_adapter_url, "http://source/step_1")
                finally:
                    rollout_release.set()
                    await dispatcher.stop()
                    await asyncio.wait_for(watcher_task, 1.0)
                    await asyncio.wait_for(dispatcher_task, 1.0)

        asyncio.run(run_test())

    def test_bootstrap_stage_boundary_rejects_invalid_external_stage(self):
        watcher_module = load_weight_watcher_runtime()

        async def run_test():
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "step_0").mkdir()
                (root / "step_0" / "STABLE").touch()
                config = SimpleNamespace(output_dir=root, max_steps=None)
                policy = SimpleNamespace(version=0, model_name="base-model")
                inference = SimpleNamespace(admin_clients=[object()], adapter_origin_base_urls=None)
                watcher_module.get_broadcast_dir = lambda path: Path(path)
                watcher_module.get_step_path = lambda path, step: Path(path) / f"step_{step}"
                watcher_module.inference_weight_path = (
                    lambda _config, path, _lora: f"http://source/{Path(path).name}"
                )
                watcher = watcher_module.WeightWatcher(
                    config,
                    policy=policy,
                    inference=inference,
                    observers=[],
                    lora_name="adapter",
                    ckpt_step=0,
                )
                for message in ("adapter identity mismatch", "adapter stage is not ready"):
                    async def rejected_stage(*_args, _message=message, **_kwargs):
                        raise RuntimeError(_message)

                    watcher_module.stage_lora_adapter = rejected_stage
                    with self.assertRaisesRegex(RuntimeError, message):
                        await watcher.stage_bootstrap_adapter()
                    self.assertFalse(watcher.bootstrap_staged)

        asyncio.run(run_test())

    def test_resume_setup_missing_policy_zero_fails_before_update(self):
        orchestrator_module = load_orchestrator_runtime()
        target_url = "http://source/step_1"
        manifest = {
            "files": [
                {"name": "adapter_config.json", "size": 2, "sha256": "b" * 64},
                {"name": "adapter_model.safetensors", "size": 8, "sha256": "a" * 64},
            ]
        }
        calls = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fetch(url, timeout):
            calls.append((url, timeout))
            return Response(json.dumps(manifest).encode("utf-8"))

        inference = SimpleNamespace(
            model_name="base-model", admin_clients=[object()], adapter_origin_base_urls=None,
            updates=[], wait_for_ready=lambda _name: asyncio.sleep(0),
        )
        async def update_weights(*_args, **_kwargs):
            inference.updates.append(True)
            return []
        inference.update_weights = update_weights

        class TrainEnvs(list):
            names = []

            def __init__(self, *_args, **_kwargs):
                super().__init__()

            async def start(self, **_kwargs):
                return None

        async def setup_policy_inference_pool(**_kwargs):
            return None, inference

        async def loaded_receipt(*_args):
            return {
                "source_url": target_url,
                "target_sha256": "a" * 64,
                "target_size": 8,
                "config_sha256": "b" * 64,
                "config_size": 2,
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step_1").mkdir()
            (root / "step_1" / "STABLE").touch()
            config = SimpleNamespace(
                output_dir=root,
                tokenizer="tokenizer",
                model=SimpleNamespace(
                    name="base-model",
                    client=SimpleNamespace(base_url=["http://inference"]),
                    lora=SimpleNamespace(name="adapter"),
                ),
                log=SimpleNamespace(level="INFO", json_logging=False, vf_level="WARN", interval=1.0),
                train=SimpleNamespace(env=[], ratio_stages=None),
                eval=None,
                ckpt=SimpleNamespace(resume_step=1, skip_progress=True, wait_for_weights_timeout=1.0),
                weight_broadcast=SimpleNamespace(
                    type="filesystem",
                    source_url="http://source",
                    source_dir=root,
                    timeout=1.0,
                ),
                renderer=None,
                output_dir_name=root.name,
                wandb=None,
                swanlab=None,
                prime_monitor=None,
                file_monitor=None,
                heartbeat=None,
                collect_inference_metrics=False,
                inference_metrics_roles=[],
                pre_batch_filters=[],
                post_batch_filters=[],
                rollout_transport="memory",
                max_inflight_rollouts=1,
                tasks_per_minute=None,
                max_off_policy_steps=1,
                batch_size=1,
                token_batch_size=1,
                max_steps=None,
                bench=False,
            )
            instance = orchestrator_module.Orchestrator.__new__(orchestrator_module.Orchestrator)
            instance.__dict__.update(
                config=config, progress=SimpleNamespace(step=0), ckpt_manager=object(),
                policy=SimpleNamespace(version=0, model_name=""), stopped=asyncio.Event(),
                model_express=None, resume_step=None, resume_policy_step=None, lora_name=None,
                eval_envs=None,
            )
            orchestrator_module.set_default_executor = lambda: None
            orchestrator_module.to_toml_dict = lambda _config: {}
            orchestrator_module.setup_tokenizer = lambda _tokenizer: SimpleNamespace(vocab_size=1)
            orchestrator_module.setup_policy_inference_pool = setup_policy_inference_pool
            orchestrator_module.setup_monitor = lambda **_kwargs: object()
            orchestrator_module.setup_filters = lambda *_args, **_kwargs: []
            orchestrator_module.TrainEnvs = TrainEnvs
            orchestrator_module.get_log_dir = lambda path: path
            orchestrator_module.setup_training_batch_sender = lambda *_args: object()
            orchestrator_module.get_broadcast_dir = lambda path: root
            orchestrator_module.get_step_path = lambda _path, step: root / f"step_{step}"
            orchestrator_module.get_all_ckpt_steps = lambda _path: [1]
            orchestrator_module.get_weight_dir = lambda *_args, **_kwargs: root / "step_1"
            orchestrator_module.inference_weight_path = (
                lambda _config, path, _lora: target_url if Path(path).name == "step_1" else str(path)
            )
            orchestrator_module.common_loaded_lora_receipt = loaded_receipt
            orchestrator_module.adapter_stage_limit_seconds = lambda _step: 600.0
            orchestrator_module.urlopen = fetch
            orchestrator_module._adapter_file_records = (
                lambda value: {item["name"]: item for item in value["files"]}
            )
            orchestrator_module.validate_adapter_manifest = lambda *_args, **_kwargs: None
            orchestrator_module.get_logger = lambda: SimpleNamespace(
                debug=lambda *_args, **_kwargs: None,
                info=lambda *_args, **_kwargs: None,
                success=lambda *_args, **_kwargs: None,
                warning=lambda *_args, **_kwargs: None,
            )
            with self.assertRaisesRegex(RuntimeError, "exact predecessor v0 is not STABLE"):
                asyncio.run(instance.setup())
        self.assertEqual(inference.updates, [])
        self.assertEqual(
            calls,
            [
                (target_url + "/adapter_manifest.json", 30),
                (target_url + "/adapter_manifest.json", 600.0),
            ],
        )

    def test_resume_setup_stages_chain_and_updates_final_policy(self):
        module = load_orchestrator_runtime()
        target_url = "http://source/step_1"
        files = [
            {"name": "adapter_config.json", "size": 2, "sha256": "b" * 64},
            {"name": "adapter_model.safetensors", "size": 8, "sha256": "a" * 64},
        ]
        manifests = {
            "step_1": {
                "version": 1, "kind": "lora-adapter", "mode": "delta",
                "transport": "chunked-shuffled-xor-zstd", "target_step": 1,
                "base_step": 0, "target_size": 8, "target_sha256": "a" * 64,
                "config_size": 2, "config_sha256": "b" * 64,
                "base_size": 8, "base_sha256": "a" * 64,
                "base_config_sha256": "b" * 64, "files": files,
            },
            "step_0": {"version": 1, "kind": "lora-adapter", "mode": "delta",
                       "transport": "chunked-shuffled-xor-zstd", "target_step": 0,
                       "base_step": -1, "files": files},
        }
        staged, updates, calls = [], [], []

        class Response(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                self.close()

        def fetch(url, timeout):
            calls.append((url, timeout))
            return Response(json.dumps(manifests[Path(url).parent.name]).encode("utf-8"))

        inference = SimpleNamespace(
            model_name="base-model", admin_clients=[object()], adapter_origin_base_urls=None,
            updates=[], wait_for_ready=lambda _name: asyncio.sleep(0),
            update_model_name=lambda name: setattr(inference, "model_name", name),
        )
        async def update_weights(weights_path, **kwargs):
            updates.append((weights_path, kwargs))
            return []
        inference.update_weights = update_weights

        class TrainEnvs(list):
            names = []
            def __init__(self, *_args, **_kwargs):
                super().__init__()
            async def start(self, **_kwargs):
                return None

        async def setup_pool(**_kwargs):
            return None, inference
        async def stage(*_args, **kwargs):
            staged.append((str(_args[1]), kwargs))
            return {"adapter_identity": {"source_url": str(_args[1])}}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in (0, 1):
                (root / f"step_{step}").mkdir()
                (root / f"step_{step}" / "STABLE").touch()
            config = SimpleNamespace(
                output_dir=root, tokenizer="tokenizer",
                model=SimpleNamespace(name="base-model", client=SimpleNamespace(base_url=["http://inference"]), lora=SimpleNamespace(name="adapter")),
                log=SimpleNamespace(level="INFO", json_logging=False, vf_level="WARN", interval=1.0),
                train=SimpleNamespace(env=[], ratio_stages=None), eval=None,
                ckpt=SimpleNamespace(resume_step=1, skip_progress=True, wait_for_weights_timeout=1.0),
                weight_broadcast=SimpleNamespace(type="filesystem", source_url="http://source", source_dir=root, timeout=1.0),
                renderer=None, wandb=None, swanlab=None, prime_monitor=None, file_monitor=None,
                heartbeat=None, collect_inference_metrics=False, inference_metrics_roles=[],
                pre_batch_filters=[], post_batch_filters=[], rollout_transport="memory",
                max_inflight_rollouts=1, tasks_per_minute=None, max_off_policy_steps=1,
                batch_size=1, token_batch_size=1, max_steps=None, bench=False,
            )
            instance = module.Orchestrator.__new__(module.Orchestrator)
            instance.__dict__.update(
                config=config, progress=SimpleNamespace(step=0), ckpt_manager=object(),
                policy=SimpleNamespace(version=0, model_name=""), stopped=asyncio.Event(),
                model_express=None, resume_step=None, resume_policy_step=None, lora_name=None,
                eval_envs=None,
            )
            module.set_default_executor = lambda: None
            module.to_toml_dict = lambda _config: {}
            module.setup_tokenizer = lambda _name: SimpleNamespace(vocab_size=1)
            module.setup_policy_inference_pool = setup_pool
            module.setup_monitor = lambda **_kwargs: object()
            module.setup_filters = lambda *_args, **_kwargs: []
            module.TrainEnvs = TrainEnvs
            module.get_log_dir = lambda path: path
            module.setup_training_batch_sender = lambda *_args: object()
            module.get_broadcast_dir = lambda _path: root
            module.get_step_path = lambda _path, step: root / f"step_{step}"
            module.get_all_ckpt_steps = lambda _path: [0, 1]
            module.get_weight_dir = lambda _config, step, **_kwargs: root / f"step_{step}"
            module.inference_weight_path = lambda _config, path, _lora: f"http://source/{Path(path).name}"
            module.common_loaded_lora_receipt = lambda *_args: asyncio.sleep(0, result=None)
            module.adapter_stage_limit_seconds = lambda _step: 600.0
            module.urlopen = fetch
            module.stage_lora_adapter = stage
            module._adapter_file_records = lambda value: {item["name"]: item for item in value["files"]}
            module.validate_adapter_manifest = lambda *_args, **_kwargs: None
            module.get_logger = lambda: SimpleNamespace(
                debug=lambda *_args, **_kwargs: None, info=lambda *_args, **_kwargs: None,
                success=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None,
            )
            asyncio.run(instance.setup())
        self.assertEqual([item[0] for item in staged], ["http://source/step_0", target_url])
        self.assertEqual([item[1]["policy_step"] for item in staged], [None, 1])
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0][1]["step"], 1)
        self.assertEqual(updates[0][1]["expected_adapter_identity"]["source_url"], target_url)
        self.assertEqual(instance.policy.version, 1)
        self.assertEqual(instance.policy.model_name, "adapter")
        self.assertEqual(inference.model_name, "adapter")
        self.assertEqual(len(calls), 2)

    def test_server_fallback_stages_in_the_host_process(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        load = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "load_lora_adapter"
            ),
        )
        self.assertIn("await stage_lora_adapter", load)
        self.assertNotIn("asyncio.to_thread", load)
        self.assertNotIn('collective_rpc("stage_lora_adapter"', load)


if __name__ == "__main__":
    unittest.main()
