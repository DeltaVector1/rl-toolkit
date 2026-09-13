import ast
import asyncio
import contextlib
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"
ORCHESTRATOR_PATH = ROOT / "src/prime_rl/orchestrator/orchestrator.py"
SERVER_PATH = ROOT / "src/prime_rl/inference/vllm/server.py"
WATCHER_PATH = ROOT / "src/prime_rl/orchestrator/watcher.py"


def load_weight_watcher_class():
    source = WATCHER_PATH.read_text(encoding="utf-8")
    class_node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "WeightWatcher"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            class_node,
        ],
        type_ignores=[],
    )
    namespace = {"asyncio": asyncio, "time": time}
    exec(compile(ast.fix_missing_locations(module), str(WATCHER_PATH), "exec"), namespace)
    return namespace["WeightWatcher"], namespace


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

    def test_resume_nonzero_without_exact_base_ready_fails_closed(self):
        target_url = "http://source/step_1"
        calls = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fetch(url, timeout):
            calls.append((url, timeout))
            return Response(json.dumps({"version": 1}).encode("utf-8"))

        validator = load_startup_validator(
            fetch,
            lambda *_args, **_kwargs: None,
            lambda _value: {},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "step_1").mkdir()
            with self.assertRaisesRegex(RuntimeError, "exact predecessor v0 is not STABLE"):
                validator(
                    SimpleNamespace(output_dir=root),
                    "adapter",
                    1,
                    root / "step_1",
                    target_url,
                )
        self.assertEqual(calls, [(target_url + "/adapter_manifest.json", 600.0)])

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

    def test_startup_stages_once_per_host_before_loading(self):
        source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
        setup = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
            ),
        )
        stage_at = setup.index("await stage_lora_adapter(")
        update_at = setup.index("await self.policy_inference.update_weights(", stage_at)
        self.assertLess(stage_at, update_at)
        self.assertIn("prefetched=prefetched", setup[update_at:])

    def test_policy_update_waits_for_verified_policy_zero_stage(self):
        watcher_type, namespace = load_weight_watcher_class()
        source_url = "http://source/step_0"
        identity = {
            "source_url": source_url,
            "target_sha256": "a" * 64,
            "target_size": 7,
            "config_sha256": "b" * 64,
            "config_size": 2,
        }
        updates = []
        stage_result = {
            "status": "ready",
            "adapter_identity": identity,
        }

        async def run_test():
            with tempfile.TemporaryDirectory() as temporary:
                output_dir = Path(temporary)
                (output_dir / "step_0").mkdir()
                (output_dir / "step_0" / "STABLE").touch()
                config = SimpleNamespace(output_dir=output_dir, max_steps=None)
                policy = SimpleNamespace(version=0, model_name="base-model")
                inference = SimpleNamespace(
                    admin_clients=[object()],
                    adapter_origin_base_urls=None,
                )

                namespace["get_broadcast_dir"] = lambda path: Path(path)
                namespace["get_step_path"] = lambda path, step: Path(path) / f"step_{step}"
                namespace["inference_weight_path"] = (
                    lambda _config, path, _lora: f"http://source/{Path(path).name}"
                )
                namespace["get_latest_ckpt_step"] = lambda _path: 1
                namespace["adapter_stage_limit_seconds"] = lambda _step: 1.0
                namespace["format_time"] = lambda value: f"{value:.3f}s"
                namespace["get_logger"] = lambda: SimpleNamespace(info=lambda *_args: None)

                async def run_once(_name, operation, *, attempts=3):
                    return await operation()

                namespace["_run_with_retries"] = run_once
                stage_started = asyncio.Event()
                stage_release = asyncio.Event()
                stage_returned = False

                async def delayed_stage(_clients, path, **kwargs):
                    nonlocal stage_returned
                    self.assertEqual(path, source_url)
                    self.assertFalse(kwargs["require_delta"])
                    stage_started.set()
                    await stage_release.wait()
                    stage_returned = True
                    return stage_result

                namespace["stage_lora_adapter"] = delayed_stage
                watcher = watcher_type(
                    config,
                    policy=policy,
                    inference=inference,
                    observers=[],
                    lora_name="adapter",
                    ckpt_step=0,
                    poll_interval=0.001,
                    reconcile_interval=100.0,
                )

                async def fake_apply(next_step, _deadline=None):
                    self.assertTrue(stage_returned)
                    self.assertTrue(watcher.bootstrap_staged)
                    self.assertEqual(watcher.verified_adapter_step, 0)
                    self.assertEqual(watcher.verified_adapter_url, source_url)
                    self.assertEqual(policy.version, 0)
                    self.assertEqual(policy.model_name, "base-model")
                    self.assertEqual(stage_result["status"], "ready")
                    self.assertEqual(stage_result["adapter_identity"], identity)
                    updates.append(next_step)
                    watcher.stopped.set()

                watcher.apply_policy_update = fake_apply
                watcher_task = asyncio.create_task(watcher.start())
                await asyncio.wait_for(stage_started.wait(), 1.0)

                dispatcher_progress = asyncio.Event()

                async def dispatcher_work():
                    dispatcher_progress.set()

                await asyncio.create_task(dispatcher_work())
                self.assertTrue(dispatcher_progress.is_set())
                self.assertEqual(policy.version, 0)
                self.assertEqual(policy.model_name, "base-model")
                self.assertFalse(watcher.bootstrap_staged)
                self.assertEqual(updates, [])

                stage_release.set()
                await asyncio.wait_for(watcher_task, 1.0)
                self.assertEqual(updates, [1])

        asyncio.run(run_test())

    def test_filesystem_lora_resume_stages_saved_policies_and_loads_only_the_last(self):
        source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
        setup = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
            ),
        )
        self.assertIn("for step in get_all_ckpt_steps(broadcast_dir)", setup)
        self.assertIn('get_step_path(broadcast_dir, step) / "STABLE"', setup)
        self.assertIn("saved adapter chain is incomplete", setup)
        self.assertIn("for sync_version in sync_versions", setup)
        self.assertIn("if sync_version != final_sync_version", setup)
        self.assertLess(
            setup.index("if sync_version != final_sync_version"),
            setup.index("await self.policy_inference.update_weights("),
        )

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
