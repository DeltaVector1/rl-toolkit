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
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"
ORCHESTRATOR_PATH = ROOT / "src/prime_rl/orchestrator/orchestrator.py"
SERVER_PATH = ROOT / "src/prime_rl/inference/vllm/server.py"


def load_rollout_budget_functions():
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    functions = {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name in {"rollout_budget", "token_batch_rollout_budget"}
    }
    namespace = {"ROLLOUT_TAIL_SPARE_GROUPS": 32}
    exec(
        compile(
            ast.Module(body=list(functions.values()), type_ignores=[]),
            str(ORCHESTRATOR_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["rollout_budget"], namespace["token_batch_rollout_budget"]


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
        "re": re,
        "time": time,
        "urlopen": urlopen,
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "inference_weight_path": lambda _config, path, _lora_name: (
            f"http://source/{path.name}"
        ),
        "_delta_stage_timeout_s": lambda: 30.0,
        "adapter_stage_limit_seconds": lambda step: {
            1: 960.0,
            2: 960.0,
        }.get(step, 960.0),
        "validate_adapter_manifest": validate_manifest,
        "_adapter_file_records": file_records,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORCHESTRATOR_PATH), "exec"), namespace)
    return namespace["_verify_startup_adapter_delta"]


def load_pruned_resume_selector(urlopen, verify, validate_manifest=None):
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_select_pruned_resume_tail"
    )
    namespace = {
        "OrchestratorConfig": object,
        "HTTPError": HTTPError,
        "json": json,
        "urlopen": urlopen,
        "get_broadcast_dir": lambda output_dir: output_dir,
        "get_step_path": lambda broadcast_dir, step: broadcast_dir / f"step_{step}",
        "inference_weight_path": lambda _config, path, _lora_name: (
            f"http://source/{path.name}"
        ),
        "_verify_startup_adapter_delta": verify,
        "validate_adapter_manifest": validate_manifest or (lambda *_args, **_kwargs: None),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORCHESTRATOR_PATH), "exec"), namespace)
    return namespace["_select_pruned_resume_tail"]


def load_resume_slot_finder(common_receipt):
    source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    function = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_find_loaded_resume_policy"
    )
    namespace = {"common_loaded_lora_receipt": common_receipt}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORCHESTRATOR_PATH), "exec"), namespace)
    return namespace["_find_loaded_resume_policy"]


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
            1: 960.0,
            2: 960.0,
        }.get(step, 960.0),
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


def startup_identity(step, target_sha256, config_sha256):
    return {
        "source_url": f"http://source/step_{step}",
        "target_size": 8,
        "target_sha256": target_sha256 * 64,
        "config_size": 2,
        "config_sha256": config_sha256 * 64,
    }


def startup_manifest(
    step,
    *,
    version=1,
    target_sha256="a",
    config_sha256="b",
    base_sha256="c",
    base_config_sha256="d",
    base2_sha256="e",
    base2_config_sha256="f",
):
    manifest = {
        "version": version,
        "kind": "lora-adapter",
        "mode": "delta",
        "transport": (
            "chunked-shuffled-xor-zstd"
            if version == 1
            else "chunked-bf16-linear-zigzag-sparse-zstd"
        ),
        "target_step": step,
        "base_step": step - 1,
        "target_url": f"http://source/step_{step}",
        "base_url": f"http://source/step_{step - 1}",
        "target_size": 8,
        "target_sha256": target_sha256 * 64,
        "config_size": 2,
        "config_sha256": config_sha256 * 64,
        "base_size": 8,
        "base_sha256": base_sha256 * 64,
        "base_config_sha256": base_config_sha256 * 64,
        "files": [
            {"name": "adapter_config.json", "size": 2, "sha256": config_sha256 * 64},
            {"name": "adapter_model.safetensors", "size": 8, "sha256": target_sha256 * 64},
        ],
    }
    if version == 2:
        manifest.update(
            {
                "base2_step": step - 2,
                "base2_url": f"http://source/step_{step - 2}",
                "base_config_size": 2,
                "base2_size": 8,
                "base2_sha256": base2_sha256 * 64,
                "base2_config_size": 2,
                "base2_config_sha256": base2_config_sha256 * 64,
                "data_offset": 0,
            }
        )
    return manifest


def retained_manifest_chain(first_step=32):
    targets = {
        first_step: ("2", "a"),
        first_step + 1: ("3", "b"),
        first_step + 2: ("4", "c"),
    }
    manifests = {}
    for step, (target_sha256, target_config_sha256) in targets.items():
        previous_sha256, previous_config_sha256 = targets.get(
            step - 1, ("1", "z")
        )
        previous_previous_sha256, previous_previous_config_sha256 = targets.get(
            step - 2, ("0", "y")
        )
        manifests[step] = startup_manifest(
            step,
            version=2,
            target_sha256=target_sha256,
            config_sha256=target_config_sha256,
            base_sha256=previous_sha256,
            base_config_sha256=previous_config_sha256,
            base2_sha256=previous_previous_sha256,
            base2_config_sha256=previous_previous_config_sha256,
        )
    return manifests


class JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def run_startup_validator(
    step,
    manifest,
    *,
    previous_identity=None,
    previous_previous_identity=None,
):
    target_url = f"http://source/step_{step}"
    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def fetch(url, timeout):
        calls.append((url, timeout))
        if url == target_url + "/adapter_manifest.json":
            return Response(json.dumps(manifest).encode("utf-8"))
        raise OSError("pruned predecessor manifest")

    validator = load_startup_validator(
        fetch,
        lambda *_args, **_kwargs: None,
        lambda value: {item["name"]: item for item in value["files"]},
    )
    with tempfile.TemporaryDirectory() as temporary:
        if step == 1:
            step_zero = Path(temporary) / "step_0"
            step_zero.mkdir()
            (step_zero / "STABLE").write_text("stable")
        validator(
            SimpleNamespace(output_dir=Path(temporary)),
            "adapter",
            step,
            Path(temporary) / f"step_{step}",
            target_url,
            previous_identity=previous_identity,
            previous_previous_identity=previous_previous_identity,
        )
    return calls


class ResumeRecoveryTest(unittest.TestCase):
    def test_token_batch_budget_refills_a_drained_pipeline(self):
        _, budget = load_rollout_budget_functions()
        self.assertEqual(
            budget(
                current_tokens=699_296,
                target_tokens=1_500_000,
                live=0,
                group_size=16,
                max_inflight=256,
            ),
            256,
        )

    def test_token_batch_budget_refills_whole_groups(self):
        _, budget = load_rollout_budget_functions()
        self.assertEqual(
            budget(
                current_tokens=699_296,
                target_tokens=1_500_000,
                live=249,
                group_size=16,
                max_inflight=256,
            ),
            16,
        )

    def test_token_batch_budget_stops_at_target(self):
        _, budget = load_rollout_budget_functions()
        self.assertEqual(
            budget(
                current_tokens=1_500_000,
                target_tokens=1_500_000,
                live=0,
                group_size=16,
                max_inflight=256,
            ),
            0,
        )

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
            self.assertEqual(stage_calls[0][1]["stage_timeout_seconds"], 960.0)

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
            if url != target_url + "/adapter_manifest.json":
                raise OSError("pruned source manifest")
            return Response(json.dumps(manifest).encode("utf-8"))

        def validate(value, **kwargs):
            self.assertEqual(value, manifest)
            self.assertEqual(kwargs["target_step"], 6)
            self.assertEqual(kwargs["base_url"], "http://source/step_5")
            self.assertEqual(kwargs["base2_url"], "http://source/step_4")

        def records(value):
            return {item["name"]: item for item in value["files"]}

        validator = load_startup_validator(fetch, validate, records)
        previous_identity = {
            "source_url": "http://source/step_5",
            "target_size": 8,
            "target_sha256": "c" * 64,
            "config_size": 2,
            "config_sha256": "d" * 64,
        }
        previous_previous_identity = {
            "source_url": "http://source/step_4",
            "target_size": 8,
            "target_sha256": "e" * 64,
            "config_size": 2,
            "config_sha256": "f" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            validator(
                SimpleNamespace(output_dir=Path(temporary)),
                "adapter",
                6,
                Path(temporary) / "step_6",
                target_url,
                previous_identity=previous_identity,
                previous_previous_identity=previous_previous_identity,
            )
        self.assertEqual(
            calls,
            [(target_url + "/adapter_manifest.json", 960.0)],
        )

    def test_resume_manifest_probe_uses_exact_policy_step_limit(self):
        for step, expected_limit in ((1, 960.0), (2, 960.0), (3, 960.0)):
            with self.subTest(step=step):
                manifest = startup_manifest(step, version=1 if step == 1 else 2)
                calls = run_startup_validator(
                    step,
                    manifest,
                    previous_identity=startup_identity(step - 1, "c", "d"),
                    previous_previous_identity=(
                        startup_identity(step - 2, "e", "f") if step >= 2 else None
                    ),
                )
                self.assertEqual(calls, [(f"http://source/step_{step}/adapter_manifest.json", expected_limit)])

    def test_resume_step_one_uses_staged_predecessor_identity_when_manifest_is_missing(self):
        calls = run_startup_validator(
            1,
            startup_manifest(1),
            previous_identity=startup_identity(0, "c", "d"),
        )
        self.assertEqual(calls, [("http://source/step_1/adapter_manifest.json", 960.0)])

    def test_resume_step_one_rejects_staged_predecessor_identity_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "identity differs"):
            run_startup_validator(
                1,
                startup_manifest(1),
                previous_identity=startup_identity(0, "e", "d"),
            )

    def test_resume_step_one_rejects_missing_predecessor_identity(self):
        with self.assertRaisesRegex(RuntimeError, "manifest is unavailable"):
            run_startup_validator(1, startup_manifest(1))

    def test_resume_v2_uses_both_staged_predecessor_identities(self):
        calls = run_startup_validator(
            2,
            startup_manifest(2, version=2),
            previous_identity=startup_identity(1, "c", "d"),
            previous_previous_identity=startup_identity(0, "e", "f"),
        )
        self.assertEqual(calls, [("http://source/step_2/adapter_manifest.json", 960.0)])

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

    def test_fresh_filesystem_lora_loads_policy_zero_before_rollouts(self):
        source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
        setup = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
            ),
        )
        initial_sync = setup.index("initial_filesystem_lora =")
        stage = setup.index("await stage_lora_adapter(", initial_sync)
        dispatcher = setup.index("self.dispatcher = RolloutDispatcher(")
        self.assertIn('config.weight_broadcast.type == "filesystem"', setup[initial_sync:stage])
        self.assertIn("configured_wait or STARTUP_WEIGHT_WAIT_TIMEOUT_S", setup[initial_sync:stage])
        self.assertLess(stage, dispatcher)

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

    def test_pruned_resume_requires_cache_hits_and_seeds_verified_loaded_identity(self):
        source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
        setup = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
            ),
        )
        self.assertIn("resume_seed_identities", setup)
        self.assertIn("sync_versions = (retained_tail[1], final_sync_version)", setup)
        self.assertIn("sync_versions = (retained_tail[0], final_sync_version)", setup)
        self.assertIn('stage_result.get("cache_hit") is not True', setup)
        self.assertIn("using_pruned_resume_tail and sync_version < final_sync_version", setup)

    def test_pruned_policy_zero_selects_three_step_v2_tail(self):
        final_sync_version = 34
        first_tail_step = final_sync_version - 2
        manifests = retained_manifest_chain(first_tail_step)
        calls = []

        def fetch(url, timeout):
            calls.append(url)
            if url.endswith("step_0/adapter_manifest.json"):
                raise HTTPError(url, 404, "gone", {}, None)
            step = int(re.search(r"step_(\d+)/adapter_manifest", url).group(1))
            if step not in manifests:
                raise HTTPError(url, 404, "gone", {}, None)
            return JsonResponse(json.dumps(manifests[step]).encode("utf-8"))

        def verify(*_args, **_kwargs):
            raise AssertionError("retained-tail validation must not fetch pruned predecessors")

        def validate(manifest, *, target_step, target_url, base_url, base2_url):
            if (
                manifest.get("version") != 2
                or manifest.get("target_step") != target_step
                or manifest.get("target_url") != target_url
                or manifest.get("base_url") != base_url
                or manifest.get("base2_url") != base2_url
                or manifest.get("transport") != "chunked-bf16-linear-zigzag-sparse-zstd"
            ):
                raise RuntimeError("invalid predecessor")

        selector = load_pruned_resume_selector(fetch, verify, validate)
        selected = selector(
            SimpleNamespace(output_dir=Path("/tmp/output")),
            "adapter",
            final_sync_version,
            tuple(range(1, final_sync_version + 1)),
            None,
        )
        self.assertEqual(selected, (32, 33, 34))
        self.assertEqual(selected, tuple(range(first_tail_step, final_sync_version + 1)))
        self.assertTrue(any("step_32/adapter_manifest.json" in url for url in calls))
        self.assertTrue(any("step_33/adapter_manifest.json" in url for url in calls))
        self.assertTrue(any("step_34/adapter_manifest.json" in url for url in calls))
        self.assertFalse(any("step_30/adapter_manifest.json" in url for url in calls))
        self.assertFalse(any("step_31/adapter_manifest.json" in url for url in calls))

    def test_pruned_resume_rejects_too_short_tail_before_manifest_probe(self):
        def fetch(_url, timeout):
            raise HTTPError("http://source/step_0/adapter_manifest.json", 404, "gone", {}, None)

        def verify(*_args, **_kwargs):
            raise AssertionError("too-short tails must fail before manifest validation")

        selector = load_pruned_resume_selector(fetch, verify)
        with self.assertRaisesRegex(RuntimeError, "too short"):
            selector(
                SimpleNamespace(output_dir=Path("/tmp/output")),
                "adapter",
                3,
                (1, 2, 3),
                None,
            )

    def test_pruned_tail_rejects_non_v2_or_malformed_predecessors(self):
        final_sync_version = 6
        first_tail_step = final_sync_version - 2

        def validate(manifest, *, target_step, target_url, base_url, base2_url):
            if (
                manifest.get("version") != 2
                or manifest.get("target_step") != target_step
                or manifest.get("target_url") != target_url
                or manifest.get("base_url") != base_url
                or manifest.get("base2_url") != base2_url
                or manifest.get("transport") != "chunked-bf16-linear-zigzag-sparse-zstd"
            ):
                raise RuntimeError("invalid predecessor")

        base = retained_manifest_chain(first_tail_step)
        cases = [
            (first_tail_step + 2, dict(base[first_tail_step + 2], version=1)),
            (
                first_tail_step + 1,
                dict(base[first_tail_step + 1], target_url="http://source/bad"),
            ),
            (
                first_tail_step,
                dict(base[first_tail_step], base2_url="http://source/bad"),
            ),
            (
                first_tail_step,
                dict(base[first_tail_step], transport="bad"),
            ),
        ]
        for bad_step, bad_manifest in cases:
            with self.subTest(bad_step=bad_step):
                manifests = dict(base)
                manifests[bad_step] = bad_manifest

                def fetch(url, timeout):
                    if url.endswith("step_0/adapter_manifest.json"):
                        raise HTTPError(url, 404, "gone", {}, None)
                    step = int(re.search(r"step_(\d+)/adapter_manifest", url).group(1))
                    return JsonResponse(json.dumps(manifests[step]).encode("utf-8"))

                selector = load_pruned_resume_selector(fetch, lambda *_args, **_kwargs: None, validate)
                with self.assertRaisesRegex(RuntimeError, "invalid|version-two"):
                    selector(
                        SimpleNamespace(output_dir=Path("/tmp/output")),
                        "adapter", final_sync_version, tuple(range(1, final_sync_version + 1)),
                        None,
                    )

    def test_pruned_tail_rejects_retained_identity_mismatch(self):
        manifests = retained_manifest_chain()
        manifests[34]["base_sha256"] = "f" * 64

        def fetch(url, timeout):
            if url.endswith("step_0/adapter_manifest.json"):
                raise HTTPError(url, 404, "gone", {}, None)
            step = int(re.search(r"step_(\d+)/adapter_manifest", url).group(1))
            return JsonResponse(json.dumps(manifests[step]).encode("utf-8"))

        selector = load_pruned_resume_selector(fetch, lambda *_args, **_kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "identities differ"):
            selector(
                SimpleNamespace(output_dir=Path("/tmp/output")),
                "adapter",
                34,
                tuple(range(1, 35)),
                None,
            )

    def test_pruned_tail_rejects_missing_retained_manifest(self):
        manifests = retained_manifest_chain()

        def fetch(url, timeout):
            if url.endswith("step_0/adapter_manifest.json") or "step_33/" in url:
                raise HTTPError(url, 404, "gone", {}, None)
            step = int(re.search(r"step_(\d+)/adapter_manifest", url).group(1))
            return JsonResponse(json.dumps(manifests[step]).encode("utf-8"))

        selector = load_pruned_resume_selector(fetch, lambda *_args, **_kwargs: None)
        with self.assertRaisesRegex(RuntimeError, "manifest is unavailable"):
            selector(
                SimpleNamespace(output_dir=Path("/tmp/output")),
                "adapter",
                34,
                tuple(range(1, 35)),
                None,
            )

    def test_loaded_resume_finds_saved_rotating_slot(self):
        receipt = {
            "source_url": "http://source/step_33",
            "target_size": 8,
            "target_sha256": "a" * 64,
            "config_size": 2,
            "config_sha256": "b" * 64,
        }
        calls = []

        async def common(_clients, slot):
            calls.append(slot)
            return receipt if slot.endswith("slot1") else None

        finder = load_resume_slot_finder(common)
        self.assertEqual(
            asyncio.run(finder([], "glm52-decensor-r32-node-local", 34)),
            ("glm52-decensor-r32-node-local-slot1", receipt),
        )
        self.assertEqual(
            calls,
            [
                "glm52-decensor-r32-node-local-slot0",
                "glm52-decensor-r32-node-local-slot1",
            ],
        )

    def test_loaded_resume_rejects_mixed_or_missing_slot_receipts(self):
        async def mixed(_clients, _slot):
            raise RuntimeError("loaded policy receipt is missing on one endpoint")

        finder = load_resume_slot_finder(mixed)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            asyncio.run(finder([], "adapter", 34))

        async def missing(_clients, _slot):
            return None

        finder = load_resume_slot_finder(missing)
        self.assertEqual(asyncio.run(finder([], "adapter", 34)), (None, None))

    def test_pruned_resume_rejects_noncontiguous_saved_tail(self):
        def fetch(_url, timeout):
            raise HTTPError("http://source/step_0/adapter_manifest.json", 404, "gone", {}, None)

        selector = load_pruned_resume_selector(fetch, lambda *_args, **_kwargs: {"version": 2})
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            selector(
                SimpleNamespace(output_dir=Path("/tmp/output")),
                "adapter",
                6,
                (1, 2, 3, 5, 6),
                None,
            )

    def test_available_policy_zero_keeps_full_replay(self):
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def fetch(_url, timeout):
            return Response(b"manifest")

        def verify(*_args):
            raise AssertionError("full replay must not preflight a tail")

        selector = load_pruned_resume_selector(fetch, verify)
        versions = tuple(range(7))
        self.assertEqual(
            selector(SimpleNamespace(output_dir=Path("/tmp/output")), "adapter", 6, versions, None),
            versions,
        )

    def test_pruned_policy_zero_does_not_skip_timeout_or_server_error(self):
        for failure in (
            TimeoutError("timed out"),
            HTTPError("http://source/step_0/adapter_manifest.json", 500, "error", {}, None),
        ):
            with self.subTest(failure=type(failure).__name__):
                def fetch(_url, timeout, failure=failure):
                    raise failure

                selector = load_pruned_resume_selector(fetch, lambda *_args, **_kwargs: {"version": 2})
                with self.assertRaises((TimeoutError, RuntimeError)):
                    selector(
                        SimpleNamespace(output_dir=Path("/tmp/output")),
                        "adapter",
                        6,
                        tuple(range(7)),
                        None,
                    )

    def test_pruned_resume_rejects_non_v2_or_missing_predecessors(self):
        def fetch(_url, timeout):
            raise HTTPError("http://source/step_0/adapter_manifest.json", 404, "gone", {}, None)

        for result in (
            {"version": 1},
            RuntimeError("adapter manifest is unavailable"),
        ):
            with self.subTest(result=result):
                def verify(*_args, result=result, **_kwargs):
                    if isinstance(result, Exception):
                        raise result
                    return result

                selector = load_pruned_resume_selector(fetch, verify)
                with self.assertRaises(RuntimeError):
                    selector(
                        SimpleNamespace(output_dir=Path("/tmp/output")),
                        "adapter",
                        6,
                        tuple(range(7)),
                        None,
                    )

    def test_common_loaded_receipt_keeps_existing_resume_path(self):
        def fetch(_url, _timeout):
            raise AssertionError("common loaded receipt must skip policy-zero probe")

        selector = load_pruned_resume_selector(fetch, lambda *_args: None)
        versions = tuple(range(7))
        self.assertEqual(
            selector(
                SimpleNamespace(output_dir=Path("/tmp/output")),
                "adapter",
                6,
                versions,
                {"source_url": "http://source/step_3"},
            ),
            versions,
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
