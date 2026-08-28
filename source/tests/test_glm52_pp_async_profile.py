"""Checks for the isolated full-shape asynchronous PP qualification."""

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "configs/qualification/glm52-native-pp-async.toml"
TARGETS = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
}


def _load(relative: str) -> dict:
    with (ROOT / relative).open("rb") as stream:
        return tomllib.load(stream)


def _load_training_ops(name: str):
    module_path = ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_worker_diagnostic(training_ops, session: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            training_ops._pp_async_cuda_diagnostic_program(str(session)),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


def test_async_profile_is_isolated_and_full_shape() -> None:
    operations = _load("configs/operations.toml")
    assert operations["profiles"]["glm52-native-pp-async"] == {
        "kind": "glm52",
        "trainer_container": "glm52_qualification_trainer",
        "trainer": PROFILE,
    }

    config = _load(PROFILE)
    assert config["output_dir"] == "/data/glm52-rl/qualifications/glm52-native-pp-async"
    assert config["max_steps"] == 2
    assert config["data"]["fake"] == {
        "batch_size": 64,
        "generate_samples": True,
        "samples_per_micro_batch": 32,
        "group_size": 16,
    }
    assert (
        config["data"]["fake"]["batch_size"]
        * config["data"]["fake"]["samples_per_micro_batch"]
        == 2048
    )

    model = config["model"]
    assert (model["pp"], model["cp"], model["ep"]) == (8, 8, 8)
    assert model["dp_replicate"] == 1
    assert model["global_ep"] is False
    assert model["ep_comm_backend"] == "torch"
    assert model["pp_schedule"] == "1f1b_async"
    assert model["pp_limit_receive_buffers"] is True
    assert model["pp_transfer_sparse_indices"] is True
    assert model["pp_layer_boundaries"] == [0, 8, 18, 28, 38, 48, 58, 68, 78]
    assert model["freeze_moe_router"] is True
    assert model["ac"]["keep_routed_expert_activations_from_pp_stage"] == 5
    assert model["quantization"] == {
        "type": "fp8",
        "dequantize_checkpoint": False,
        "diagnose_linear_backward": False,
        "fused_qkv_a_fp8": True,
        "expert_activation_transport": "bf16_forward_fp8_local_recompute_backward",
        "w8a16_modules": [],
    }
    assert model["optimization_dtype"] == model["reduce_dtype"] == "bfloat16"
    assert model["lora"]["rank"] == 32
    assert set(model["lora"]["target_modules"]) == TARGETS
    assert len(model["lora"]["target_modules"]) == 10
    assert config["optim"]["lr"] == 8e-6

    env_vars = config["env_vars"]
    assert env_vars == {"PRIME_LORA_MM_PROFILE": "sm89-r32"}
    forbidden = {
        "PRIME_NSYS_PROFILE",
        "PRIME_CUDA_PROFILE",
        "PRIME_PHASE_PROFILE",
        "PRIME_EXPERT_PHASE_PROFILE",
        "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD",
        "PRIME_INT4_COMBINE_ROUNDS",
    }
    assert not forbidden & set(env_vars)
    assert model["fsdp_cpu_offload"] is False
    assert model["cold_cpu_offload"] is False
    assert model["optim_cpu_offload"] is False
    assert model["native_fp8_expert_prefetch"] is False
    assert model["ac_offloading"] == "None"


def test_async_profile_is_allowlisted_and_validated_before_setup() -> None:
    cluster = (ROOT / "scripts/china/cluster.sh").read_text()
    train = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    config = (
        ROOT / "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
    ).read_text()
    training_ops = (
        ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    ).read_text()

    assert "glm52-native-pp-async" in cluster
    assert "glm52-native-pp-async" in train
    assert '"glm52-native-pp-async": 2' in training_ops
    assert '"glm52-native-pp-async",' in training_ops
    assert "validate_async_1f1b_topology" in config
    assert (
        "1f1b_async is only available in a GLM async qualification profile" in train
    )
    assert train.index("config.model.validate_async_1f1b_topology()") < train.index(
        "setup_torch_distributed("
    )


def test_async_topology_runtime_uses_model_checkpoint_config() -> None:
    """Exercise the schedule argument with the real TrainerConfig shape."""
    train_path = ROOT / "source/src/prime_rl/trainer/rl/train.py"
    tree = ast.parse(train_path.read_text())
    expression = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "async_topology":
                expression = keyword.value
                break
        if expression is not None:
            break
    assert expression is not None

    config = SimpleNamespace(
        model=SimpleNamespace(
            pp_schedule="1f1b_async",
            pp_limit_receive_buffers=True,
            ac=SimpleNamespace(keep_routed_expert_activations_from_pp_stage=5),
        )
    )
    parallel_dims = SimpleNamespace(pp=8, cp=8, ep=8)
    topology = eval(
        compile(ast.Expression(expression), str(train_path), "eval"),
        {"config": config, "parallel_dims": parallel_dims},
    )
    assert topology["cutoff"] == 5
    assert topology["pp"] == topology["cp"] == topology["ep"] == 8


def test_cuda_smoke_profile_is_small_and_n82_only() -> None:
    operations = _load("configs/operations.toml")
    assert operations["qualification"]["glm52_pp_async_cuda"] == {
        "kind": "glm52",
        "host": "n82",
        "container": "glm52_pp_async_cuda_smoke",
        "script": "source/tests/test_pp_async_cuda.py",
        "output_root": "/data/glm52-rl/qualifications/glm52-pp-async-cuda-smoke",
        "steps": 2,
        "gpus": 8,
        "transport": "glm52_trainer",
        "outer_timeout_seconds": 300,
    }
    script = (ROOT / "source/tests/test_pp_async_cuda.py").read_text()
    assert "stages = 8" in script
    assert "microbatches = 8" in script
    assert "steps = 2" in script
    assert "Schedule1F1BAsync" not in script
    tree = ast.parse(script)
    run_modes = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mode"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert run_modes == {"1f1b", "1f1b_async"}
    assert "@record\ndef main()" in script
    assert "model-fp8" not in script
    assert "/data/ZiphuAI" not in script
    assert "timeout=_process_group_timeout()" in script
    assert "device_id=_process_group_device_id(local_rank)" in script
    assert 'dist.init_process_group("nccl")' not in script


def _new_cuda_dummy_stage():
    path = ROOT / "source/tests/test_pp_async_cuda.py"
    tree = ast.parse(path.read_text())
    stage = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AffineStage"
    )
    names = {
        "begin_microbatch_step",
        "set_active_microbatch",
        "clear_active_microbatch",
        "assert_microbatch_step_complete",
    }
    contract = ast.ClassDef(
        name="AffineStageContract",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            node
            for node in stage.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
    )
    namespace = {"microbatches": 8}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[contract], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    dummy = namespace["AffineStageContract"]()
    dummy.stage_identity = 0
    dummy._active_microbatch = None
    dummy._microbatch_step = None
    dummy._microbatch_records = []
    dummy._current_record = None
    return dummy


def _load_cuda_work_lifecycle():
    path = ROOT / "source/tests/test_pp_async_cuda.py"
    tree = ast.parse(path.read_text())
    lifecycle = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkLifecycle"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[lifecycle], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["WorkLifecycle"]


def test_cuda_dummy_implements_the_pipeline_runner_microbatch_interface() -> None:
    runner = ast.parse(
        (ROOT / "source/src/prime_rl/trainer/distributed/pp_runner.py").read_text()
    )
    factory = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_keyed_pipeline_stage_cls"
    )
    keyed_stage = next(
        node
        for node in factory.body
        if isinstance(node, ast.ClassDef) and node.name == "KeyedPipelineStage"
    )
    consumers = {
        node.func.attr
        for node in ast.walk(keyed_stage)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "submod"
    }
    assert consumers == {"set_active_microbatch", "clear_active_microbatch"}
    dummy = _new_cuda_dummy_stage()
    assert dummy._active_microbatch is None
    assert all(callable(getattr(dummy, method)) for method in consumers)


def test_cuda_dummy_resets_microbatch_state_across_two_steps() -> None:
    dummy = _new_cuda_dummy_stage()
    for step in range(2):
        dummy.begin_microbatch_step(step)
        for microbatch in range(8):
            dummy.set_active_microbatch(microbatch)
            assert dummy._active_microbatch == microbatch
            dummy.clear_active_microbatch()
        dummy.assert_microbatch_step_complete()
        assert dummy._active_microbatch is None
        assert dummy._microbatch_step is None
        assert dummy._current_record is None
    assert dummy._microbatch_records == [
        {"step": step, "set": list(range(8)), "clear": list(range(8))}
        for step in range(2)
    ]


def test_cuda_work_lifecycle_retires_every_rank_and_step() -> None:
    lifecycle_class = _load_cuda_work_lifecycle()

    class Work:
        def is_completed(self):
            return True

    class Journal:
        def __init__(self):
            self.events = []

        def write(self, event, **fields):
            self.events.append((event, fields))

    for rank in range(8):
        for step in range(2):
            journal = Journal()
            lifecycle = lifecycle_class(journal)
            expected = 0
            for group in range(32 - 2 * rank):
                works = [Work() for _ in range(1 + (group + step) % 2)]
                pending = SimpleNamespace(works=works)
                lifecycle.create(pending, f"rank={rank};step={step};group={group}")
                lifecycle.mark_waited(pending)
                lifecycle.retire(pending)
                expected += len(works)
            lifecycle.assert_drained()
            assert lifecycle.counts() == {
                "created": expected,
                "waited": expected,
                "retired": expected,
                "live": 0,
            }
            assert [event for event, _fields in journal.events].count(
                "async_work_created"
            ) == expected
            assert [event for event, _fields in journal.events].count(
                "async_work_waited"
            ) == expected
            assert [event for event, _fields in journal.events].count(
                "async_work_retired"
            ) == expected

    lifecycle = lifecycle_class()
    pending = SimpleNamespace(works=[Work()])
    lifecycle.create(pending, "unretired")
    with pytest.raises(AssertionError, match="NCCL work remained active"):
        lifecycle.assert_drained()


def test_cuda_smoke_journal_is_bounded_and_fsynced() -> None:
    path = ROOT / "source/tests/test_pp_async_cuda.py"
    tree = ast.parse(path.read_text())
    selected = []
    names = {
        "journal_max_bytes",
        "journal_max_records",
        "journal_max_record_bytes",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "RankJournal":
            selected.append(node)
    namespace = {"json": json, "os": os, "time": __import__("time"), "Path": Path}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )

    with tempfile.TemporaryDirectory() as directory:
        journal = namespace["RankJournal"](4, Path(directory))
        journal.write("action_before", action="forward", microbatch=0)
        journal.close()
        rows = (Path(directory) / "rank-4.journal.jsonl").read_text().splitlines()
        assert len(rows) == 1
        assert json.loads(rows[0])["event"] == "action_before"

        with pytest.raises(FileExistsError):
            namespace["RankJournal"](4, Path(directory))


def test_pp8_native_crossover_batches_match_on_every_edge() -> None:
    microbatch_count = 8
    stage_count = 8
    for left in range(stage_count - 1):
        right = left + 1
        forward = microbatch_count - right
        backward = 0
        left_batch = {
            ("send", left, right, "forward", forward),
            ("receive", right, left, "backward", backward),
        }
        right_batch = {
            ("send", right, left, "backward", backward),
            ("receive", left, right, "forward", forward),
        }
        normalized_left = {
            (source, target, direction, microbatch)
            for _op, source, target, direction, microbatch in left_batch
        }
        normalized_right = {
            (source, target, direction, microbatch)
            for _op, source, target, direction, microbatch in right_batch
        }
        assert normalized_left == normalized_right


def test_cuda_smoke_records_all_blocking_boundaries() -> None:
    source = (ROOT / "source/tests/test_pp_async_cuda.py").read_text()
    for event in (
        "group_init_before",
        "group_init_after",
        "schedule_build_before",
        "schedule_build_after",
        "step_before",
        "step_schedule_after",
        "action_before",
        "action_after",
        "p2p_wait_before",
        "p2p_wait_after",
        "async_group_post_before",
        "async_group_post_after",
        "async_group_wait_before",
        "async_group_wait_after",
        "async_drain_before",
        "async_drain_after",
        "step_after",
    ):
        assert f'"{event}"' in source
    assert "os.O_EXCL" in source
    assert "os.O_NOFOLLOW" in source
    assert "os.fsync(self.fd)" in source


def test_cuda_smoke_overrides_have_a_closed_prime_rl_import_graph() -> None:
    entry = "source/tests/test_pp_async_cuda.py"
    mounted = {
        "prime_rl.trainer.distributed.async_schedule": (
            "source/src/prime_rl/trainer/distributed/async_schedule.py"
        ),
        "prime_rl.trainer.distributed.pp_runner": (
            "source/src/prime_rl/trainer/distributed/pp_runner.py"
        ),
        "prime_rl.trainer.phase_profiler": (
            "source/src/prime_rl/trainer/phase_profiler.py"
        ),
    }
    image_modules = {
        "prime_rl.trainer.model",
        "prime_rl.utils.act_offloading",
    }

    def imports(relative: str) -> set[str]:
        tree = ast.parse((ROOT / relative).read_text())
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("prime_rl.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("prime_rl.")
            ):
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    candidate_path = (
                        ROOT / "source/src" / (candidate.replace(".", "/") + ".py")
                    )
                    dependencies.add(
                        candidate if candidate_path.is_file() else node.module
                    )
        return dependencies

    pending = [entry]
    visited = set()
    missing = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        for module in imports(relative):
            if module in mounted:
                pending.append(mounted[module])
            elif module not in image_modules:
                missing.add(module)
    assert not missing
    assert visited == {entry, *mounted.values()}


def test_cuda_smoke_uses_the_host_network() -> None:
    source = (
        ROOT / ".agents/skills/training-ops/src/training_ops/__init__.py"
    ).read_text()
    assert "--network {shlex.quote(transport['network_mode'])} --read-only" in source
    assert 'f"torchrun --standalone --local-addr=127.0.0.1 "' in source
    assert 'f"--nproc-per-node={gpu_count} "' in source
    assert '"--log-dir /output/torchrun --redirects 3 --tee 3 "' in source


def test_cuda_smoke_uses_the_configured_trainer_transport() -> None:
    operations = _load("configs/operations.toml")
    transport = operations["transport"]["glm52_trainer"]
    assert transport == {
        "network_mode": "host",
        "process_group_timeout_seconds": 240,
        "device_id": "local_rank",
        "env": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "NCCL_SOCKET_IFNAME": "vlan130",
            "GLOO_SOCKET_IFNAME": "vlan130",
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SOCKET_NTHREADS": "1",
            "NCCL_NSOCKS_PERTHREAD": "1",
            "NCCL_BUFFSIZE": "1048576",
            "NCCL_CUMEM_HOST_ENABLE": "0",
            "NCCL_MIN_NCHANNELS": "16",
            "NCCL_MAX_NCHANNELS": "16",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "1048576",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ENABLE_MONITORING": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",
        },
    }
    qualification = operations["qualification"]["glm52_pp_async_cuda"]
    trainer = _load(operations["profiles"]["current"]["trainer"])
    assert qualification["transport"] == "glm52_trainer"
    assert transport["process_group_timeout_seconds"] == 240
    assert transport["process_group_timeout_seconds"] < 300
    assert trainer["dist_timeout_seconds"] == 5400

    launcher = (ROOT / "scripts/china/cluster.sh").read_text()
    for key, value in transport["env"].items():
        assert f"-e {key}=" in launcher
        assert value in launcher
    assert "--network host" in launcher
    assert "--ipc host" in launcher


def test_cuda_smoke_rejects_a_blocked_transport_host() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_blocked_host")
    qualification = training_ops.operations["qualification"]["glm52_pp_async_cuda"]
    old_head = training_ops.trainer_head
    old_host = qualification["host"]
    qualification["host"] = "n79"
    training_ops.trainer_head = "n79"
    try:
        with pytest.raises(training_ops.Refused, match="configured trainer head"):
            training_ops.qualify_glm52_pp_async_cuda()
    finally:
        qualification["host"] = old_host
        training_ops.trainer_head = old_head


def test_cuda_smoke_requires_transport_configuration() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_missing_transport")
    with pytest.raises(training_ops.Refused, match="not configured"):
        training_ops._pp_async_cuda_transport(
            {"transport": "missing", "outer_timeout_seconds": 300}
        )


def test_cuda_smoke_rejects_a_timeout_that_can_outlive_its_controller() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_long_timeout")
    transport = training_ops.operations["transport"]["glm52_trainer"]
    profile = training_ops.operations["qualification"]["glm52_pp_async_cuda"]
    old_timeout = transport["process_group_timeout_seconds"]
    transport["process_group_timeout_seconds"] = 300
    try:
        with pytest.raises(training_ops.Refused, match="below the outer timeout"):
            training_ops._pp_async_cuda_transport(profile)
    finally:
        transport["process_group_timeout_seconds"] = old_timeout


@pytest.mark.parametrize("value", [None, True, 300.0, 301])
def test_cuda_smoke_requires_the_fixed_outer_timeout(value) -> None:
    training_ops = _load_training_ops(f"training_ops_pp_async_outer_{value!r}")
    profile = dict(training_ops.operations["qualification"]["glm52_pp_async_cuda"])
    profile["outer_timeout_seconds"] = value
    with pytest.raises(training_ops.Refused, match="must be 300 seconds"):
        training_ops._pp_async_cuda_transport(profile)


def test_phase_profiler_is_a_shared_trainer_mount() -> None:
    training_ops = _load_training_ops("training_ops_phase_profiler_mount")
    path = "source/src/prime_rl/trainer/phase_profiler.py"
    assert training_ops._mounted_on(path, "n82")
    assert training_ops._mounted_on(
        path,
        "n82",
        profile_name="glm52-native-b2048",
    )
    assert training_ops._mounted_on(
        path,
        "n82",
        profile_name="glm52-pp-async-cuda-smoke",
    )
    assert training_ops._mounted_on(path, "n76")


def test_exited_trainer_is_allowed_and_cuda_command_is_fail_closed() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_smoke")

    smoke = "glm52-pp-async-cuda-smoke"
    test_path = "source/tests/test_pp_async_cuda.py"
    assert training_ops._mounted_on(test_path, "n82", profile_name=smoke)
    assert not training_ops._mounted_on(test_path, "n76", profile_name=smoke)
    assert not training_ops._mounted_on(test_path, "n82")

    source_paths = (
        test_path,
        "source/src/prime_rl/trainer/distributed/async_schedule.py",
        "source/src/prime_rl/trainer/distributed/pp_runner.py",
        "source/src/prime_rl/trainer/phase_profiler.py",
    )
    hashes = {
        "test_sha256": hashlib.sha256(
            (ROOT / source_paths[0]).read_bytes()
        ).hexdigest(),
        "async_schedule_sha256": hashlib.sha256(
            (ROOT / source_paths[1]).read_bytes()
        ).hexdigest(),
        "pp_runner_sha256": hashlib.sha256(
            (ROOT / source_paths[2]).read_bytes()
        ).hexdigest(),
        "phase_profiler_sha256": hashlib.sha256(
            (ROOT / source_paths[3]).read_bytes()
        ).hexdigest(),
    }
    mode = {
        "steps": [{"cuda_ms": 1.0}, {"cuda_ms": 1.0}],
        "peak_memory_bytes_max": 1,
        "all_async_nccl_work_drained": True,
    }
    result = {
        "status": "pass",
        "ranks": 8,
        "steps": 2,
        "microbatches": 8,
        "action_loss_grad_update_parity": True,
        "all_async_nccl_work_drained": True,
        "rank_reports": [
            {"rank": rank, "baseline": mode, "async": mode} for rank in range(8)
        ],
        **hashes,
    }
    calls = []

    def fake_ssh(host, command, timeout=120, **_kwargs):
        calls.append((host, command, timeout))
        if len(calls) == 1:
            return 0, "GPUS 8\nPIDS\nRUNNING\n"
        if len(calls) == 2:
            return 0, "STALE none\n"
        if len(calls) == 3:
            return 0, json.dumps(result)
        return 0, "CONTAINERS\nPIDS\n"

    deployed = []
    training_ops._assert_live_services_are_down = lambda nodes, include_inference: (
        nodes == ["n82"] and include_inference is False
    ) or (_ for _ in ()).throw(AssertionError("unsafe live-service scope"))
    training_ops._ssh = fake_ssh
    training_ops.deploy = lambda path, **kwargs: deployed.append((path, kwargs)) or True
    training_ops.secrets.token_hex = lambda _size: "0123456789abcdef"

    report = training_ops.qualify_glm52_pp_async_cuda()
    assert report["status"] == "pass"
    assert [item[0] for item in deployed] == list(source_paths)
    assert all(item[1]["nodes"] == ["n82"] for item in deployed)
    assert all(host == "n82" for host, _command, _timeout in calls)
    assert len(calls) == 4
    preflight = calls[0][1]
    assert "docker ps --format" in preflight
    assert "docker ps -a" not in preflight
    stale_cleanup = calls[1][1]
    assert "docker ps -aq --no-trunc" in stale_cleanup
    assert "docker inspect --type container" in stale_cleanup
    assert 'docker rm "$candidate_id"' in stale_cleanup
    launch = calls[2][1]
    assert "torchrun --standalone --local-addr=127.0.0.1" in launch
    assert "--nproc-per-node=8" in launch
    assert "--log-dir /output/torchrun" in launch
    assert "--redirects 3 --tee 3" in launch
    assert "--network host" in launch
    assert "--network none" not in launch
    assert "--hostname localhost" not in launch
    for key, value in training_ops.operations["transport"]["glm52_trainer"][
        "env"
    ].items():
        assert f"-e {key}={value}" in launch
    assert "-e PRIME_PROCESS_GROUP_TIMEOUT_SECONDS=240" in launch
    assert "-e PRIME_PROCESS_GROUP_DEVICE_ID=local_rank" in launch
    assert "/app/src/prime_rl/trainer/phase_profiler.py:ro" in launch
    assert "--read-only" in launch
    assert "--cidfile" in launch
    assert "--label prime.qualification=glm52-pp-async-cuda-smoke" in launch
    assert "--label prime.session=0123456789abcdef" in launch
    assert "timeout --foreground --signal=TERM --kill-after=30s 300s" in launch
    assert "trap cleanup EXIT" in launch
    assert "docker inspect --type container" in launch
    assert "docker rm -f" in launch
    assert "cleanup also failed" in launch
    assert calls[2][2] == 360
    postflight = calls[3][1]
    assert "docker ps -aq" in postflight
    assert "nvidia-smi --query-compute-apps=pid" in postflight
    assert "/models" not in launch
    assert "/data/ZiphuAI" not in launch
    assert "training.env" not in launch

    def timed_out_ssh(host, command, timeout=120, **_kwargs):
        timeout_calls.append((host, command, timeout))
        if len(timeout_calls) == 1:
            return 0, "GPUS 8\nPIDS\nRUNNING\n"
        if len(timeout_calls) == 2:
            return 0, "STALE none\n"
        if len(timeout_calls) == 3:
            return 124, "host-side timeout"
        return 0, "CONTAINERS\nPIDS\n"

    timeout_calls = []
    training_ops._ssh = timed_out_ssh
    try:
        training_ops.qualify_glm52_pp_async_cuda()
    except RuntimeError as error:
        assert "host-side timeout" in str(error)
        assert "cleanup/postflight also failed" not in str(error)
    else:
        raise AssertionError("a timed-out CUDA qualification passed")
    assert len(timeout_calls) == 5
    assert "--cidfile" in timeout_calls[2][1]
    assert "trap cleanup EXIT" in timeout_calls[2][1]

    def failed_cleanup_ssh(host, command, timeout=120, **_kwargs):
        failed_calls.append((host, command, timeout))
        if len(failed_calls) == 1:
            return 0, "GPUS 8\nPIDS\nRUNNING\n"
        if len(failed_calls) == 2:
            return 0, "STALE none\n"
        if len(failed_calls) == 3:
            return 124, "original host timeout"
        return 0, "CONTAINERS abc123\nPIDS 77\n"

    failed_calls = []
    training_ops._ssh = failed_cleanup_ssh
    try:
        training_ops.qualify_glm52_pp_async_cuda()
    except RuntimeError as error:
        message = str(error)
        assert message.index("original host timeout") < message.index(
            "cleanup/postflight also failed"
        )
        assert "containers=abc123 pids=77" in message
    else:
        raise AssertionError("a failed cleanup replaced the original error")


def test_rank_four_failure_artifact_survives_a_missing_result() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_rank_failure")
    training_ops._assert_live_services_are_down = lambda *_args, **_kwargs: None
    training_ops.deploy = lambda *_args, **_kwargs: True
    training_ops.secrets.token_hex = lambda _size: "0123456789abcdef"
    calls = []
    artifact = {
        "rank": 4,
        "error_path": "torchrun/run/attempt_0/4/error.json",
        "traceback": "Traceback: rank four failed before result.json",
    }

    def fake_ssh(host, command, timeout=120, **_kwargs):
        calls.append((host, command, timeout))
        if len(calls) == 1:
            return 0, "GPUS 8\nPIDS\nRUNNING\n"
        if len(calls) == 2:
            return 0, "STALE none\n"
        if len(calls) == 3:
            return 1, "ChildFailedError: rank 4"
        if len(calls) == 4:
            return 0, "CONTAINERS\nPIDS\n"
        return 0, json.dumps(artifact)

    training_ops._ssh = fake_ssh
    try:
        training_ops.qualify_glm52_pp_async_cuda()
    except RuntimeError as error:
        message = str(error)
        assert "failing_rank=4" in message
        assert (
            "error_path=/data/glm52-rl/qualifications/"
            "glm52-pp-async-cuda-smoke/0123456789abcdef/"
            "torchrun/run/attempt_0/4/error.json"
        ) in message
        assert artifact["traceback"] in message
        assert "worker error artifact is missing" not in message
    else:
        raise AssertionError("a missing result hid the rank-four worker failure")
    assert len(calls) == 5
    launch = calls[2][1]
    assert "--log-dir /output/torchrun" in launch
    diagnostic = calls[4][1]
    assert "error.json" in diagnostic
    assert "traceback[-6000:]" in diagnostic


def test_worker_diagnostic_reads_a_valid_rank_four_artifact() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_valid_artifact")
    with tempfile.TemporaryDirectory() as directory:
        session = Path(directory) / "session"
        rank_dir = session / "torchrun" / "run" / "attempt_0" / "4"
        rank_dir.mkdir(parents=True)
        traceback = "Traceback (most recent call last):\nrank four failed"
        (rank_dir / "error.json").write_text(
            json.dumps(
                {
                    "message": {
                        "message": "RuntimeError: rank four failed",
                        "extraInfo": {"py_callstack": traceback},
                    }
                }
            )
        )
        assert _run_worker_diagnostic(training_ops, session) == {
            "rank": 4,
            "error_path": "torchrun/run/attempt_0/4/error.json",
            "traceback": traceback,
        }


def test_worker_diagnostic_rejects_symlinks_corrupt_and_oversized_files() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_invalid_artifact")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        session = root / "session"
        session.mkdir()
        outside = root / "outside"
        (outside / "4").mkdir(parents=True)
        (outside / "4" / "error.json").write_text('{"message": "escaped"}')
        (session / "torchrun").symlink_to(outside, target_is_directory=True)
        assert _run_worker_diagnostic(training_ops, session) == {"status": "missing"}

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        session = root / "session"
        torchrun = session / "torchrun"
        torchrun.mkdir(parents=True)
        outside = root / "outside"
        (outside / "attempt_0" / "4").mkdir(parents=True)
        (outside / "attempt_0" / "4" / "error.json").write_text(
            '{"message": "escaped"}'
        )
        (torchrun / "run").symlink_to(outside, target_is_directory=True)
        assert _run_worker_diagnostic(training_ops, session) == {"status": "missing"}

    for payload in (b"not json", b"x" * 1048577):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            rank_dir = session / "torchrun" / "run" / "attempt_0" / "4"
            rank_dir.mkdir(parents=True)
            (rank_dir / "error.json").write_bytes(payload)
            assert _run_worker_diagnostic(training_ops, session) == {
                "status": "missing"
            }


def test_preflight_reports_gpu_pids_and_running_containers_separately() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_preflight")
    training_ops._assert_live_services_are_down = lambda *_args, **_kwargs: None
    training_ops.deploy = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("a refused preflight deployed files")
    )

    cases = (
        (
            "GPUS 8\nPIDS 42,43\nRUNNING\n",
            "active GPU processes",
            "pids=42,43",
        ),
        (
            "GPUS 8\nPIDS\nRUNNING glm52-optims-trainer\n",
            "running trainer or CUDA qualification container",
            "containers=glm52-optims-trainer",
        ),
    )
    for output, reason, identity in cases:
        calls = []

        def fake_ssh(host, command, timeout=120, **_kwargs):
            calls.append((host, command, timeout))
            return 0, output

        training_ops._ssh = fake_ssh
        try:
            training_ops.qualify_glm52_pp_async_cuda()
        except training_ops.Refused as error:
            assert reason in str(error)
            assert identity in str(error)
        else:
            raise AssertionError("an occupied n82 preflight passed")
        assert len(calls) == 1
        assert "docker ps --format" in calls[0][1]
        assert "docker ps -a" not in calls[0][1]


def test_stale_candidate_cleanup_is_fail_closed() -> None:
    training_ops = _load_training_ops("training_ops_pp_async_stale")
    training_ops._assert_live_services_are_down = lambda *_args, **_kwargs: None
    calls = []

    def fake_ssh(host, command, timeout=120, **_kwargs):
        calls.append((host, command, timeout))
        if len(calls) == 1:
            return 0, "GPUS 8\nPIDS\nRUNNING\n"
        return 96, "stale CUDA qualification candidate identity differs"

    training_ops._ssh = fake_ssh
    training_ops.deploy = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("invalid stale state deployed files")
    )
    try:
        training_ops.qualify_glm52_pp_async_cuda()
    except RuntimeError as error:
        assert "cannot clean a stale n82 PP async CUDA candidate" in str(error)
        assert "identity differs" in str(error)
    else:
        raise AssertionError("an invalid stale candidate was removed or ignored")
    assert len(calls) == 2
    stale_cleanup = calls[1][1]
    assert "docker ps -aq --no-trunc" in stale_cleanup
    assert "docker inspect --type container" in stale_cleanup
    assert "observed_running" in stale_cleanup
    assert 'docker rm "$candidate_id"' in stale_cleanup
