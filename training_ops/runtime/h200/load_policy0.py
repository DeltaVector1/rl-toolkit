"""Fail-closed policy-0 load checks.

This module contains the local state machine for the offline qualification
load action.  Network and client operations are injected by the caller so
tests can prove that the action never starts a service.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping


class Policy0LoadError(RuntimeError):
    """A required policy-0 load safety check failed."""


EXPECTED_LORA = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "target_modules": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "shared_expert",
        "experts",
    ),
}


def build_plan(
    *,
    config_dir: str,
    source_run: str,
    source_adapter_dir: str,
    source_config_path: str,
    source_url: str,
    qualification_slot: str,
    admin_urls: list[str],
    origin_urls: list[str],
    trainer_hosts: list[str],
    orchestrator_host: str,
    inference_hosts: list[str],
    engine_ports: Mapping[str, list[int]],
    engine_container: str,
) -> dict[str, object]:
    """Build and validate the exact local load plan."""
    if not isinstance(source_run, str) or not source_run:
        _fail("policy-0 source run is invalid")
    if not isinstance(source_adapter_dir, str) or not source_adapter_dir.endswith("/step_0"):
        _fail("source adapter directory must be the exact source run step_0 directory")
    if not isinstance(source_config_path, str) or source_config_path != source_adapter_dir + "/adapter_config.json":
        _fail("source config path must be the adapter_config.json in step_0")
    if not isinstance(source_url, str) or not source_url.endswith("/step_0"):
        _fail("policy-0 source URL must name step_0")
    if not isinstance(qualification_slot, str) or not qualification_slot:
        _fail("policy-0 qualification slot is invalid")
    for label, values in (("admin", admin_urls), ("origin", origin_urls), ("trainer", trainer_hosts), ("inference", inference_hosts)):
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
            _fail(f"policy-0 {label} endpoints are empty or invalid")
    if not isinstance(orchestrator_host, str) or not orchestrator_host:
        _fail("policy-0 orchestrator host is invalid")
    if not isinstance(engine_container, str) or not engine_container:
        _fail("policy-0 engine container is invalid")
    return {
        "config_dir": config_dir,
        "source_run": source_run,
        "source_url": source_url,
        "source_adapter_dir": source_adapter_dir,
        "source_config_path": source_config_path,
        "qualification_slot": qualification_slot,
        "admin_urls": list(admin_urls),
        "origin_urls": list(origin_urls),
        "trainer_hosts": list(trainer_hosts),
        "orchestrator_host": orchestrator_host,
        "inference_hosts": list(inference_hosts),
        "engine_ports": {host: list(values) for host, values in engine_ports.items()},
        "engine_container": engine_container,
        "expected_lora": {**EXPECTED_LORA, "target_modules": list(EXPECTED_LORA["target_modules"])},
        "repair_command": f"training-ops repair adapter-tunnels:{config_dir}",
        "checks": [
            "STABLE step-0 adapter/config hashes and sizes",
            "rank32/alpha64/dropout0 exact LoRA targets",
            "all trainer and orchestrator containers stopped and known",
            "exact H200 container ID and API PID/start identity before and after",
            "all inference endpoints healthy, running=0, waiting=0, no active stage",
            "stage require_delta=False and policy_step=None",
            "load prefetched=True with the exact returned adapter identity",
            "final staging receipts match source hashes, slot, and server instances",
        ],
    }


def _fail(message: str) -> None:
    raise Policy0LoadError(message)


def _identity(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(f"{label} identity is missing")
    result = dict(value)
    required = {
        "source_url": str,
        "target_sha256": str,
        "target_size": int,
        "config_sha256": str,
        "config_size": int,
    }
    for key, kind in required.items():
        item = result.get(key)
        if type(item) is not kind:
            _fail(f"{label} identity has invalid {key}")
        if key.endswith("sha256") and (len(item) != 64 or any(c not in "0123456789abcdef" for c in item)):
            _fail(f"{label} identity has invalid {key}")
        if key.endswith("size") and item <= 0:
            _fail(f"{label} identity has invalid {key}")
    return result


def _check_lora(value: object) -> None:
    if not isinstance(value, Mapping):
        _fail("source LoRA configuration is missing")
    for key in ("r", "lora_alpha", "lora_dropout"):
        if value.get(key) != EXPECTED_LORA[key]:
            _fail(f"source LoRA configuration differs in {key}")
    targets = value.get("target_modules")
    if not isinstance(targets, (list, tuple)) or tuple(targets) != EXPECTED_LORA["target_modules"]:
        _fail("source LoRA target_modules differ")


def _check_services(plan: Mapping[str, object], state: Mapping[str, object]) -> None:
    services = state.get("services")
    if not isinstance(services, Mapping):
        _fail("trainer and orchestrator service state is missing")
    trainers = services.get("trainers")
    expected_trainers = plan.get("trainer_hosts")
    if not isinstance(trainers, Mapping) or not isinstance(expected_trainers, (list, tuple)):
        _fail("trainer service inventory is missing")
    for host in expected_trainers:
        record = trainers.get(host)
        if not isinstance(record, Mapping) or record.get("known") is not True:
            _fail(f"trainer service state is unknown on {host}")
        if record.get("active") is not False:
            _fail(f"trainer service is active on {host}")
    orchestrator = services.get("orchestrator")
    if not isinstance(orchestrator, Mapping) or orchestrator.get("known") is not True:
        _fail("orchestrator service state is unknown")
    if orchestrator.get("active") is not False:
        _fail("orchestrator service is active")


def _check_engine(plan: Mapping[str, object], state: Mapping[str, object]) -> Mapping[str, object]:
    engine = state.get("engine")
    if not isinstance(engine, Mapping):
        _fail("inference engine state is missing")
    if engine.get("healthy") is not True:
        _fail("inference engine is not healthy")
    for key in ("running", "waiting"):
        value = engine.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            _fail(f"inference engine has {key} requests")
    if engine.get("active_stage") is True or engine.get("stage_conflict") is True:
        _fail("inference engine has an active adapter stage")
    if engine.get("stage_status") not in {"idle", "ready", "failed"}:
        _fail("inference engine staging state is unknown or active")
    for key in ("container_id", "container_started_at", "api_pid", "api_start_time"):
        value = engine.get(key)
        if key == "api_pid":
            if type(value) is not int or value <= 0:
                _fail("inference API PID identity is invalid")
        elif not isinstance(value, str) or not value:
            _fail(f"inference engine identity is missing {key}")
    endpoints = engine.get("endpoints")
    expected_endpoints = plan.get("admin_urls")
    def endpoint_key(value: str) -> str:
        return value.rstrip("/").removesuffix("/v1")

    if (
        not isinstance(endpoints, (list, tuple))
        or not endpoints
        or any(not isinstance(item, str) or not item for item in endpoints)
        or len(set(endpoints)) != len(endpoints)
        or (
            isinstance(expected_endpoints, (list, tuple))
            and {endpoint_key(item) for item in endpoints}
            != {endpoint_key(item) for item in expected_endpoints}
        )
    ):
        _fail("inference endpoint inventory is empty or differs from the plan")
    return engine


def _check_engine_identity(before: Mapping[str, object], after: Mapping[str, object]) -> None:
    for key in ("container_id", "container_started_at", "api_pid", "api_start_time"):
        if before.get(key) != after.get(key):
            _fail(f"inference engine identity changed in {key}")


def _check_source(plan: Mapping[str, object], source: object) -> dict[str, object]:
    if not isinstance(source, Mapping) or source.get("stable") is not True:
        _fail("source step-0 adapter is not STABLE")
    identity = _identity(source.get("identity"), "source adapter")
    if identity.get("source_url") != plan.get("source_url"):
        _fail("source adapter URL differs from the planned step-0 URL")
    _check_lora(source.get("lora"))
    return identity


def _check_receipts(
    plan: Mapping[str, object],
    receipts: object,
    expected_identity: Mapping[str, object],
    loaded_receipts: object | None = None,
) -> None:
    if not isinstance(receipts, list) or not receipts:
        _fail("policy-0 load returned no adapter receipts")
    expected_slot = plan.get("qualification_slot")
    expected_source = plan.get("source_url")
    instances: list[str] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            _fail("policy-0 load returned an invalid adapter receipt")
        if receipt.get("lora_name") != expected_slot:
            _fail("policy-0 load returned the wrong qualification slot")
        if receipt.get("source_url") != expected_source:
            _fail("policy-0 load receipt has the wrong source URL")
        for key in ("target_sha256", "target_size", "config_sha256", "config_size"):
            if receipt.get(key) != expected_identity.get(key):
                _fail(f"policy-0 load receipt differs in {key}")
        instance = receipt.get("server_instance")
        if not isinstance(instance, str) or not instance:
            _fail("policy-0 load receipt has no server instance")
        instances.append(instance)
    if len(set(instances)) != len(instances):
        _fail("policy-0 load returned duplicate server identities")
    if loaded_receipts is None:
        return
    if not isinstance(loaded_receipts, list):
        _fail("final staging status has no loaded receipt list")
    final_instances: list[str] = []
    for receipt in loaded_receipts:
        if not isinstance(receipt, Mapping):
            _fail("final staging status has an invalid loaded receipt")
        if receipt.get("lora_name") != expected_slot:
            _fail("final staging status has the wrong qualification slot")
        for key in ("source_url", "target_sha256", "target_size", "config_sha256", "config_size"):
            if receipt.get(key) != (expected_source if key == "source_url" else expected_identity.get(key)):
                _fail(f"final staging receipt differs in {key}")
        instance = receipt.get("server_instance")
        if not isinstance(instance, str) or not instance:
            _fail("final staging receipt has no server instance")
        final_instances.append(instance)
    if sorted(final_instances) != sorted(instances):
        _fail("final staging receipts differ from the load response")


def execute_policy0_load(
    plan: Mapping[str, object],
    *,
    read_state: Callable[[], Mapping[str, object]],
    read_source: Callable[[], Mapping[str, object]],
    stage: Callable[..., Mapping[str, object]],
    load: Callable[..., list[Mapping[str, object]]],
) -> dict[str, object]:
    """Run the guarded stage/load sequence without any service start call."""
    before = read_state()
    _check_services(plan, before)
    before_engine = _check_engine(plan, before)
    source = _check_source(plan, read_source())
    staged = stage(plan["source_url"], require_delta=False, policy_step=None)
    staged_identity = _identity(staged.get("adapter_identity"), "staged adapter")
    for key in ("source_url", "target_sha256", "target_size", "config_sha256", "config_size"):
        if staged_identity.get(key) != source.get(key):
            _fail(f"staged adapter differs in {key}")
    middle = read_state()
    _check_services(plan, middle)
    middle_engine = _check_engine(plan, middle)
    _check_engine_identity(before_engine, middle_engine)
    loaded = load(
        plan["qualification_slot"],
        plan["source_url"],
        prefetched=True,
        expected_identity=source,
    )
    after = read_state()
    _check_services(plan, after)
    after_engine = _check_engine(plan, after)
    _check_engine_identity(before_engine, after_engine)
    loaded_status = after_engine.get("loaded_receipts")
    if isinstance(loaded_status, Mapping):
        loaded_status = list(loaded_status.values())
    _check_receipts(plan, loaded, source, loaded_status)
    return {
        "status": "loaded",
        "qualification_slot": plan["qualification_slot"],
        "source_url": plan["source_url"],
        "adapter_identity": source,
        "server_instances": [item["server_instance"] for item in loaded],
    }
