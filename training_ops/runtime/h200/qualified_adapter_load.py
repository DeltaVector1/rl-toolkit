"""Fail-closed publication and load-plan checks for the reviewed step-1 adapter."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shlex
from pathlib import PurePosixPath
from typing import Any, Mapping


CANONICAL_CONFIG_DIR = "configs/trinity-h200-post-update-qualification"
QUALIFICATION_ROOT = "/data/glm52-rl/qualifications/trinity-h200-post-update-qualification"
RUNS_ROOT = "/data/glm52-rl/runs"
QUALIFICATION_SLOT = "trinity-h200-post-update-qualification-slot1"
EXPECTED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.safetensors.sha256",
    "adapter_manifest.json",
    "STABLE",
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_KEYS = frozenset(
    {
        "schema_version", "kind", "run_id", "policy0_adapter", "policy0_config",
        "step1_adapter", "step1_config", "step1_manifest_sha256", "trainer_config",
        "identities_before", "identities_after", "expected_identity", "trainer_evidence",
        "changed_element_count", "changed_tensor_count", "max_abs_delta", "l2_delta",
    }
)
_EXPECTED_IDENTITY_KEYS = frozenset(
    {
        "run_id", "reference_sha256", "trainer_config_sha256", "trainer_config_size",
        "code_sha256", "code_size", "loss_sha256", "loss_size", "loss_mode",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "run_id", "completed_optimizer_steps", "learning_rate", "loss", "grad_norm",
        "forward_backward_seconds", "export_seconds", "export_completed",
        "initial_adapter_sha256", "initial_adapter_size", "final_adapter_sha256",
        "final_adapter_size", "reference_sha256", "trainer_config_sha256",
        "trainer_config_size", "code_sha256", "code_size", "loss_identity",
    }
)
_IDENTITY_CHAIN_KEYS = frozenset({"policy0_adapter", "policy0_config", "step1_adapter", "step1_config", "step1_manifest", "step1_sidecar", "step1_stable", "artifacts"})
_RESULT_KEYS = frozenset(
    {
        "schema_version", "kind", "status", "qualification_slot", "source_url",
        "source_adapter_dir", "manifest_sha256", "receipt_sha256", "adapter_identity",
        "published", "stage_result", "load_result", "server_instances",
        "engine_identity_before", "engine_identity_after", "phase_seconds",
    }
)
_PUBLISHED_KEYS = frozenset({"status", "source_url", "source_adapter_dir", "manifest_sha256", "owner_token", "parent_token", "final_token", "files"})
_STAGE_KEYS = frozenset({"status", "adapter_identity"})
_LOAD_KEYS = frozenset({"source_url", "target_sha256", "target_size", "config_sha256", "config_size", "lora_name", "server_instance"})
_TOKEN = re.compile(r"^[0-9]+:[0-9]+(?:/[0-9]+:[0-9]+)*$")


class QualifiedAdapterLoadError(RuntimeError):
    """The qualified adapter is not safe to publish or load."""


def _fail(message: str) -> None:
    raise QualifiedAdapterLoadError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate receipt JSON key: {key}")
        result[key] = value
    return result


def _json(value: bytes | str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise QualifiedAdapterLoadError(f"receipt JSON is invalid: {error}") from error
    if not isinstance(parsed, dict):
        _fail("receipt JSON is not an object")
    return parsed


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        _fail(f"{label} SHA256 is invalid")
    return value


def _identity(value: object, label: str, *, allow_empty: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} identity is missing")
    digest = _digest(value.get("sha256"), label)
    size = value.get("size")
    if type(size) is not int or size < 0 or (size == 0 and not allow_empty):
        _fail(f"{label} size is invalid")
    return {"sha256": digest, "size": size}


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} is not finite")
    value = float(value)
    if not __import__("math").isfinite(value):
        _fail(f"{label} is not finite")
    return value


def _artifacts(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_FILES):
        _fail("receipt artifact identity set is incomplete")
    result = {name: _identity(value[name], f"receipt {name}", allow_empty=name == "STABLE") for name in EXPECTED_FILES}
    if result["STABLE"] != {"sha256": EMPTY_SHA256, "size": 0}:
        _fail("receipt STABLE identity is not the empty file")
    return result


def parse_delta_receipt(payload: bytes | str, expected_sha256: str) -> dict[str, Any]:
    """Parse one exact receipt byte stream and bind every exported artifact."""
    if not isinstance(payload, (bytes, str)):
        _fail("delta receipt payload is invalid")
    raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    _digest(expected_sha256, "expected receipt")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail("delta receipt SHA256 differs from the expected receipt")
    receipt = _json(raw)
    if set(receipt) != _RECEIPT_KEYS:
        _fail("delta receipt schema fields are incomplete or unexpected")
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "h200_step1_adapter_delta":
        _fail("delta receipt schema or kind is invalid")
    if not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]:
        _fail("delta receipt run identity is missing")
    model = _identity(receipt.get("step1_adapter"), "step-1 adapter")
    config = _identity(receipt.get("step1_config"), "step-1 config")
    manifest_sha = _digest(receipt.get("step1_manifest_sha256"), "step-1 manifest")
    before = receipt.get("identities_before")
    after = receipt.get("identities_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or set(before) != _IDENTITY_CHAIN_KEYS or set(after) != _IDENTITY_CHAIN_KEYS or before != after:
        _fail("delta receipt identity before/after values differ")
    if before.get("step1_adapter") != model or before.get("step1_config") != config:
        _fail("delta receipt step-1 identity chain is inconsistent")
    policy_model = _identity(receipt.get("policy0_adapter"), "policy-0 adapter")
    policy_config = _identity(receipt.get("policy0_config"), "policy-0 config")
    if model == policy_model or config == policy_config:
        _fail("step-1 identity must differ from policy-0")
    if before.get("policy0_adapter") != policy_model or before.get("policy0_config") != policy_config:
        _fail("delta receipt policy-0 identity chain is inconsistent")
    manifest = _identity(before.get("step1_manifest"), "step-1 manifest")
    if manifest["sha256"] != manifest_sha:
        _fail("delta receipt manifest identity differs")
    artifacts = _artifacts(before.get("artifacts"))
    if artifacts["adapter_model.safetensors"] != model or artifacts["adapter_config.json"] != config or artifacts["adapter_manifest.json"] != manifest or before.get("step1_sidecar") != artifacts["adapter_model.safetensors.sha256"] or before.get("step1_stable") != artifacts["STABLE"]:
        _fail("delta receipt artifact identity chain is inconsistent")
    if receipt.get("trainer_evidence", {}).get("export_completed") is not True:
        _fail("delta receipt does not prove export completion")
    trainer_config = _identity(receipt.get("trainer_config"), "trainer config")
    expected = receipt.get("expected_identity")
    if not isinstance(expected, Mapping) or set(expected) != _EXPECTED_IDENTITY_KEYS:
        _fail("delta receipt expected identity schema is invalid")
    if expected.get("run_id") != receipt["run_id"] or expected.get("trainer_config_sha256") != trainer_config["sha256"] or expected.get("trainer_config_size") != trainer_config["size"]:
        _fail("delta receipt expected trainer identity differs")
    for field in ("reference_sha256", "trainer_config_sha256", "code_sha256", "loss_sha256"):
        _digest(expected.get(field), f"expected {field}")
    for field in ("trainer_config_size", "code_size", "loss_size"):
        if type(expected.get(field)) is not int or expected[field] <= 0:
            _fail(f"expected {field} is invalid")
    if expected.get("loss_mode") != "644":
        _fail("expected loss mode is invalid")
    evidence = receipt.get("trainer_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_KEYS:
        _fail("delta receipt trainer evidence schema is invalid")
    if evidence.get("run_id") != receipt["run_id"] or type(evidence.get("completed_optimizer_steps")) is not int or evidence.get("completed_optimizer_steps") != 1 or type(evidence.get("learning_rate")) is not float or evidence.get("learning_rate") != 8e-6 or evidence.get("export_completed") is not True:
        _fail("delta receipt trainer evidence does not prove one update")
    if evidence.get("reference_sha256") != expected["reference_sha256"] or evidence.get("trainer_config_sha256") != expected["trainer_config_sha256"] or evidence.get("trainer_config_size") != expected["trainer_config_size"] or evidence.get("code_sha256") != expected["code_sha256"] or evidence.get("code_size") != expected["code_size"]:
        _fail("delta receipt trainer evidence identity differs")
    _finite(evidence.get("loss"), "trainer loss")
    if _finite(evidence.get("grad_norm"), "trainer grad norm") <= 0 or _finite(evidence.get("export_seconds"), "export seconds") <= 0 or not 0 <= _finite(evidence.get("forward_backward_seconds"), "forward/backward seconds") <= 600:
        _fail("delta receipt trainer timing or gradient is invalid")
    initial = _identity({"sha256": evidence.get("initial_adapter_sha256"), "size": evidence.get("initial_adapter_size")}, "initial adapter")
    final = _identity({"sha256": evidence.get("final_adapter_sha256"), "size": evidence.get("final_adapter_size")}, "final adapter")
    if initial == final:
        _fail("initial and final adapter identities are identical")
    if initial != policy_model or final != model:
        _fail("trainer initial/final adapter identities are not bound to policy-0 and step-1")
    loss = evidence.get("loss_identity")
    if not isinstance(loss, Mapping) or loss != {"sha256": expected["loss_sha256"], "size": expected["loss_size"], "mode": expected["loss_mode"]}:
        _fail("delta receipt loss identity differs")
    if type(receipt.get("changed_element_count")) is not int or receipt["changed_element_count"] <= 0 or type(receipt.get("changed_tensor_count")) is not int or receipt["changed_tensor_count"] <= 0 or _finite(receipt.get("max_abs_delta"), "maximum delta") <= 0 or _finite(receipt.get("l2_delta"), "delta norm") <= 0:
        _fail("delta metrics are invalid")
    return receipt


def parse_result_receipt(payload: bytes | str, expected_sha256: str) -> dict[str, Any]:
    """Validate the exact immutable result receipt bytes returned by n82."""
    if not isinstance(payload, (bytes, str)):
        _fail("result receipt payload is invalid")
    raw = payload if isinstance(payload, bytes) else payload.encode("utf-8")
    _digest(expected_sha256, "expected result receipt")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _fail("result receipt SHA256 differs from the expected receipt")
    result = _json(raw)
    if set(result) != _RESULT_KEYS or result.get("schema_version") != 1 or result.get("kind") != "h200_qualified_adapter_load" or result.get("status") != "loaded" or result.get("qualification_slot") != QUALIFICATION_SLOT:
        _fail("result receipt schema or status is invalid")
    for field in ("manifest_sha256", "receipt_sha256"):
        _digest(result[field], f"result {field}")
    for field in ("source_url", "source_adapter_dir"):
        if not isinstance(result[field], str) or not result[field]:
            _fail(f"result {field} is invalid")
    manifest = result["manifest_sha256"]
    expected_url = f"http://127.0.0.1:18100/qualified-adapters/{manifest}/step_1"
    expected_dir = f"{RUNS_ROOT}/qualified-adapters/{manifest}/step_1"
    if result["source_url"] != expected_url or result["source_adapter_dir"] != expected_dir:
        _fail("result source identity is not the canonical publication")
    adapter = result["adapter_identity"]
    if not isinstance(adapter, Mapping) or set(adapter) != {"source_url", "target_sha256", "target_size", "config_sha256", "config_size"} or adapter["source_url"] != result["source_url"]:
        _fail("result adapter identity is invalid")
    _digest(adapter["target_sha256"], "result target adapter")
    _digest(adapter["config_sha256"], "result adapter config")
    if type(adapter["target_size"]) is not int or adapter["target_size"] <= 0 or type(adapter["config_size"]) is not int or adapter["config_size"] < 0:
        _fail("result adapter sizes are invalid")
    published = result["published"]
    if not isinstance(published, Mapping) or set(published) != _PUBLISHED_KEYS or published.get("status") not in {"published", "existing"} or published.get("source_url") != result["source_url"] or published.get("source_adapter_dir") != result["source_adapter_dir"] or published.get("manifest_sha256") != result["manifest_sha256"]:
        _fail("result publication identity is invalid")
    if published.get("status") == "published" and (not isinstance(published.get("owner_token"), str) or not published["owner_token"] or any(character.isspace() for character in published["owner_token"])):
        _fail("result publication owner identity is invalid")
    if published.get("status") == "existing" and published.get("owner_token") != "":
        _fail("result existing publication owner identity is invalid")
    for field in ("parent_token", "final_token"):
        if not isinstance(published.get(field), str) or not _TOKEN.fullmatch(published[field]):
            _fail(f"result publication {field} is invalid")
    files = published.get("files")
    if not isinstance(files, Mapping) or set(files) != set(EXPECTED_FILES):
        _fail("result publication file identity is invalid")
    published_files = _artifacts(files)
    if published_files["adapter_model.safetensors"] != {"sha256": adapter["target_sha256"], "size": adapter["target_size"]} or published_files["adapter_config.json"] != {"sha256": adapter["config_sha256"], "size": adapter["config_size"]} or published_files["adapter_manifest.json"]["sha256"] != result["manifest_sha256"]:
        _fail("result publication adapter identity differs")
    stage = result["stage_result"]
    if not isinstance(stage, Mapping) or set(stage) != _STAGE_KEYS or stage.get("status") != "ok" or stage.get("adapter_identity") != adapter:
        _fail("result stage identity is invalid")
    loaded = result["load_result"]
    if not isinstance(loaded, list) or not loaded:
        _fail("result load receipts are invalid")
    loaded_instances = []
    for item in loaded:
        if not isinstance(item, Mapping) or set(item) != _LOAD_KEYS or item.get("lora_name") != QUALIFICATION_SLOT or item.get("source_url") != result["source_url"] or item.get("target_sha256") != adapter["target_sha256"] or item.get("target_size") != adapter["target_size"] or item.get("config_sha256") != adapter["config_sha256"] or item.get("config_size") != adapter["config_size"] or not isinstance(item.get("server_instance"), str) or not item["server_instance"]:
            _fail("result load receipt identity is invalid")
        loaded_instances.append(item["server_instance"])
    if not isinstance(result["server_instances"], list) or not result["server_instances"] or any(not isinstance(item, str) or not item for item in result["server_instances"]):
        _fail("result receipt server identity is invalid")
    if sorted(loaded_instances) != sorted(result["server_instances"]):
        _fail("result server identities differ from load receipts")
    for field in ("engine_identity_before", "engine_identity_after"):
        identity = result[field]
        if not isinstance(identity, Mapping) or set(identity) != {"container_id", "container_started_at", "api_pid", "api_start_time"} or any(not identity[key] for key in identity):
            _fail(f"result {field} is invalid")
        if type(identity["api_pid"]) is not int or identity["api_pid"] <= 0:
            _fail(f"result {field} API PID is invalid")
    phases = result["phase_seconds"]
    if not isinstance(phases, Mapping) or set(phases) != {"preflight_seconds", "publish_seconds", "stage_seconds", "load_seconds"} or any(_finite(value, f"result phase {key}") < 0 for key, value in phases.items()):
        _fail("result receipt phase timings are invalid")
    return result


def _canonical_absolute(value: object, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label} is invalid")
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label} is not canonical")
    return value.rstrip("/")


def derive_plan(
    config_dir: str,
    receipt: Mapping[str, Any],
    expected_receipt_sha256: str,
    *,
    qualification_root: str,
    runs_root: str,
    trainer_head: str,
    inference_host: str,
    inference_port: int,
    direct_route_port: int,
) -> dict[str, Any]:
    if config_dir != CANONICAL_CONFIG_DIR:
        _fail("qualified adapter load requires the canonical config directory")
    root = _canonical_absolute(qualification_root, "qualification root")
    runs = _canonical_absolute(runs_root, "runs root")
    if root != QUALIFICATION_ROOT or runs != RUNS_ROOT:
        _fail("qualified adapter roots are not the exact configured roots")
    expected_receipt = _digest(expected_receipt_sha256, "expected receipt")
    if trainer_head != "n82" or not isinstance(inference_host, str) or not inference_host:
        _fail("qualified adapter load host identity is invalid")
    if type(inference_port) is not int or not 1 <= inference_port <= 65535 or type(direct_route_port) is not int or direct_route_port != 18100:
        _fail("qualified adapter load route is invalid")
    manifest_sha = _digest(receipt.get("step1_manifest_sha256"), "step-1 manifest")
    identities_after = receipt.get("identities_after")
    if not isinstance(identities_after, Mapping):
        _fail("step-1 artifact identity is missing")
    artifacts = _artifacts(identities_after.get("artifacts"))
    if artifacts["adapter_manifest.json"]["sha256"] != manifest_sha:
        _fail("step-1 manifest identity differs")
    source_dir = f"{runs}/qualified-adapters/{manifest_sha}/step_1"
    source_url = f"http://127.0.0.1:{direct_route_port}/qualified-adapters/{manifest_sha}/step_1"
    return {
        "config_dir": config_dir,
        "qualification_root": root,
        "receipt_path": f"{root}/step1-delta-receipt.json",
        "source_step1_dir": f"{root}/run_trainer_only/broadcasts/step_1",
        "policy0_adapter": f"{root}/policy-0/adapter_model.safetensors",
        "runs_root": runs,
        "source_adapter_dir": source_dir,
        "source_url": source_url,
        "manifest_sha256": manifest_sha,
        "qualification_slot": QUALIFICATION_SLOT,
        "trainer_head": trainer_head,
        "inference_host": inference_host,
        "inference_port": inference_port,
        "direct_route_port": direct_route_port,
        "admin_urls": [f"http://127.0.0.1:{inference_port}/v1"],
        "origin_urls": [f"http://127.0.0.1:{inference_port}/v1"],
        "engine_ports": [inference_port],
        "engine_container": "glm52-decensor-h200",
        "receipt_sha256": expected_receipt,
        "artifacts": artifacts,
    }


def _validate_plan_paths(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    root = _canonical_absolute(plan.get("qualification_root"), "qualification root")
    runs = _canonical_absolute(plan.get("runs_root"), "runs root")
    if root != QUALIFICATION_ROOT or runs != RUNS_ROOT:
        _fail("qualified adapter roots are not the exact configured roots")
    manifest = _digest(plan.get("manifest_sha256"), "step-1 manifest")
    expected_dir = f"{runs}/qualified-adapters/{manifest}/step_1"
    expected_url = f"http://127.0.0.1:18100/qualified-adapters/{manifest}/step_1"
    if plan.get("source_adapter_dir") != expected_dir or plan.get("source_url") != expected_url:
        _fail("qualified adapter publication path is not canonical")
    if plan.get("qualification_slot") != QUALIFICATION_SLOT:
        _fail("qualified adapter slot is not canonical")
    if plan.get("source_step1_dir") != f"{root}/run_trainer_only/broadcasts/step_1":
        _fail("qualified adapter source path is not canonical")
    return root, runs, manifest


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _artifact_checks(variable: str, artifacts: Mapping[str, Mapping[str, Any]]) -> str:
    lines = []
    for name in EXPECTED_FILES:
        identity = artifacts[name]
        mode = "644"
        lines.extend(
            [
                f'test -f "${variable}/{name}" && test ! -L "${variable}/{name}"',
                f'test "$(stat -c %a -- "${variable}/{name}")" = {mode}',
                f'test "$(sha256sum -- "${variable}/{name}" | awk \'{{print $1}}\')" = {_q(identity["sha256"])}',
                f'test "$(stat -c %s -- "${variable}/{name}")" = {identity["size"]}',
            ]
        )
    lines.append(f'test "$(find "${variable}" -mindepth 1 -maxdepth 1 -printf \'%y:%f\\n\' | sort | paste -sd, -)" = "f:STABLE,f:adapter_config.json,f:adapter_manifest.json,f:adapter_model.safetensors,f:adapter_model.safetensors.sha256" # directory entry')
    return "\n".join(lines)


def build_publish_script(plan: Mapping[str, Any]) -> str:
    """Build the n82-only no-clobber publication and direct-route probe."""
    _validate_plan_paths(plan)
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(EXPECTED_FILES):
        _fail("publish plan artifact identities are incomplete")
    artifacts = _artifacts(artifacts)
    root = _q(plan["qualification_root"])
    source = _q(plan["source_step1_dir"])
    runs = _q(plan["runs_root"])
    source_url = _q(plan["source_url"])
    manifest = _q(plan["manifest_sha256"])
    final_root = f'{runs}/qualified-adapters/{plan["manifest_sha256"]}'
    final = f'{final_root}/step_1'
    source_checks = _artifact_checks("source", artifacts)
    tmp_checks = _artifact_checks("tmp", artifacts)
    expected_entries = ",\n".join(f"    ({name!r}, {artifacts[name]['sha256']!r}, {artifacts[name]['size']} )" for name in EXPECTED_FILES)
    return f'''set -euo pipefail
root={root}; source={source}; runs={runs}; final={_q(final)}; final_root={_q(final_root)}; source_url={source_url}; manifest={manifest}
check_ancestors() {{ current=/; old_ifs="$IFS"; IFS=/; for component in ${{1#/}}; do [ -n "$component" ] || continue; current="$current$component"; test -d "$current" && test ! -L "$current"; current="$current/"; done; IFS="$old_ifs"; }}
check_ancestors "$root"; check_ancestors "$source"; check_ancestors "$runs"
test -d "$root" && test ! -L "$root"
test -d "$source" && test ! -L "$source"
identity_tree() {{ printf 'directory entry='; find "$1" -mindepth 1 -maxdepth 1 -printf '%y:%f\\n' | sort | paste -sd, -; for item in adapter_config.json adapter_model.safetensors adapter_model.safetensors.sha256 adapter_manifest.json STABLE; do sha256sum -- "$1/$item"; stat -c '%d:%i:%s:%y:%z:%a' -- "$1/$item"; done; }}
source_before=$(identity_tree "$source")
{source_checks}
test -d "$runs" && test ! -L "$runs"
parent="$runs/qualified-adapters"
if [ -e "$parent" ] || [ -L "$parent" ]; then test -d "$parent" && test ! -L "$parent"; else mkdir -- "$parent"; fi
tmp_root=$(mktemp -d "$parent/.qualified-adapter-{plan['manifest_sha256']}.XXXXXX")
tmp="$tmp_root/step_1"; mkdir -- "$tmp"
safe_remove_published() {{
python3 - "$runs" "$manifest" "${{parent_token:-}}" "${{final_token:-}}" <<'PY'
import os, shutil, sys
runs, manifest, expected_parent, expected_final = sys.argv[1:]
def open_chain(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    ids = [os.fstat(fd)]
    try:
        for component in path.strip("/").split("/"):
            if component:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = child; ids.append(os.fstat(fd))
        return fd, "/".join(f"{{item.st_dev}}:{{item.st_ino}}" for item in ids)
    except BaseException:
        os.close(fd); raise
parent_fd, parent_identity = open_chain(os.path.join(runs, "qualified-adapters"))
if parent_identity != expected_parent: raise SystemExit("publication ancestor changed before cleanup")
value = os.stat(manifest, dir_fd=parent_fd, follow_symlinks=False)
if not os.path.isdir(os.path.join("/proc/self/fd", str(parent_fd))): raise SystemExit("publication parent is not a directory")
if not (value.st_mode & 0o170000) == 0o040000 or f"{{value.st_dev}}:{{value.st_ino}}" != expected_final: raise SystemExit("publication final changed before cleanup")
shutil.rmtree(manifest, dir_fd=parent_fd)
os.close(parent_fd)
PY
}}
safe_remove_tmp() {{
python3 - "$parent" "${{tmp_root##*/}}" <<'PY'
import os, shutil, sys
parent, temporary = sys.argv[1:]
fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
try:
    for component in parent.strip("/").split("/"):
        if component:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
    value = os.stat(temporary, dir_fd=fd, follow_symlinks=False)
    if not (value.st_mode & 0o170000) == 0o040000: raise SystemExit("temporary publication is not a directory")
    shutil.rmtree(temporary, dir_fd=fd)
finally:
    os.close(fd)
PY
}}
cleanup() {{ if [ "${{completed:-0}}" = 0 ]; then if [ "${{published_new:-0}}" = 1 ] && [ "${{status:-}}" = published ] && [ -n "${{owner_token:-}}" ] && [ -n "${{parent_token:-}}" ] && [ -n "${{final_token:-}}" ]; then safe_remove_published; fi; safe_remove_tmp || true; fi; if [ -n "${{route_tmp:-}}" ]; then rm -rf -- "$route_tmp"; fi; }}
trap cleanup EXIT HUP INT TERM
for item in adapter_config.json adapter_model.safetensors adapter_model.safetensors.sha256 adapter_manifest.json STABLE; do
  cp --reflink=auto -- "$source/$item" "$tmp/$item"
  test ! -L "$tmp/$item"
  test "$(stat -c '%d:%i' -- "$source/$item")" != "$(stat -c '%d:%i' -- "$tmp/$item")"
done
{tmp_checks.replace('$$', '$')}
source_after=$(identity_tree "$source"); test "$source_before" = "$source_after"
atomic_noreplace() {{ # RENAME_NOREPLACE
python3 - "$1" "$2" "$3" <<'PY'
import ctypes, errno, os, sys
parent, temporary, final = sys.argv[1:]
if "/" in temporary or "/" in final or not temporary or not final:
    raise SystemExit(94)
def open_directory_chain(path):
    fd = os.open("/", os.O_DIRECTORY | os.O_NOFOLLOW)
    identities = [os.fstat(fd)]
    try:
        for component in path.strip("/").split("/"):
            if component:
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
                identities.append(os.fstat(fd))
        return fd, "/".join(f"{{item.st_dev}}:{{item.st_ino}}" for item in identities)
    except BaseException:
        os.close(fd)
        raise
parent_fd, parent_token = open_directory_chain(parent)
before = os.fstat(parent_fd)
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = getattr(libc, "renameat2", None)
if renameat2 is None: raise SystemExit(95)
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int
try:
    try:
        existing = os.stat(final, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not __import__("stat").S_ISDIR(existing.st_mode) or __import__("stat").S_ISLNK(existing.st_mode):
            raise SystemExit(96)
        final_token = f"{{existing.st_dev}}:{{existing.st_ino}}"
        print(f"EXISTING parent_token={{parent_token}} final_token={{final_token}}")
    else:
        if renameat2(parent_fd, os.fsencode(temporary), parent_fd, os.fsencode(final), 1) != 0:
            error = ctypes.get_errno()
            if error != errno.EEXIST:
                raise SystemExit(95)
            existing = os.stat(final, dir_fd=parent_fd, follow_symlinks=False)
            if not __import__("stat").S_ISDIR(existing.st_mode) or __import__("stat").S_ISLNK(existing.st_mode):
                raise SystemExit(96)
            final_token = f"{{existing.st_dev}}:{{existing.st_ino}}"
            print(f"EXISTING parent_token={{parent_token}} final_token={{final_token}}")
        else:
            final_stat = os.stat(final, dir_fd=parent_fd, follow_symlinks=False)
            final_token = f"{{final_stat.st_dev}}:{{final_stat.st_ino}}"
            print(f"PUBLISHED parent_token={{parent_token}} final_token={{final_token}}")
    fresh_parent_fd, fresh_parent_identity = open_directory_chain(parent)
    try:
        if fresh_parent_identity != parent_token:
            raise SystemExit(97)
    finally:
        os.close(fresh_parent_fd)
finally:
    os.close(parent_fd)
PY
}}
published_new=0
publish_state=$(atomic_noreplace "$parent" "${{tmp_root##*/}}" "$manifest" || true)
case "$publish_state" in
PUBLISHED\\ *|EXISTING\\ *)
  parent_token=$(printf '%s\\n' "$publish_state" | sed -n 's/^\\(PUBLISHED\\|EXISTING\\) parent_token=\\([^ ]*\\) final_token=\\([^ ]*\\)$/\\2/p')
  final_token=$(printf '%s\\n' "$publish_state" | sed -n 's/^\\(PUBLISHED\\|EXISTING\\) parent_token=\\([^ ]*\\) final_token=\\([^ ]*\\)$/\\3/p')
  test -n "$parent_token" && test -n "$final_token"
  case "$publish_state" in
  PUBLISHED\\ *)
  test -d "$final_root" && test ! -L "$final_root"
  test -d "$final" && test ! -L "$final"
  published_new=1; status=published
  ;;
  EXISTING\\ *)
  test -d "$final_root" && test ! -L "$final_root"
  test -d "$final" && test ! -L "$final"
  existing="$final"; {tmp_checks.replace('tmp', 'existing')}
  rm -rf -- "$tmp_root"; published_new=1
  status=existing
  ;;
  esac
  ;;
*)
  echo 'existing destination appeared with a conflicting identity' >&2; exit 96
  ;;
esac
owner_token=""
if [ "$status" = published ]; then owner_token=$(stat -c '%d:%i:%s:%y:%z' -- "$final_root" | tr ' ' '_'); fi
route_tmp=$(mktemp -d)
route_manifest="$route_tmp/adapter_manifest.json"; route_config="$route_tmp/adapter_config.json"
curl -fsS "$source_url/adapter_manifest.json" >"$route_manifest"
curl -fsS "$source_url/adapter_config.json" >"$route_config"
test "$(sha256sum -- "$route_manifest" | awk '{{print $1}}')" = {_q(artifacts['adapter_manifest.json']['sha256'])}
test "$(stat -c %s -- "$route_manifest")" = {artifacts['adapter_manifest.json']['size']}
test "$(sha256sum -- "$route_config" | awk '{{print $1}}')" = {_q(artifacts['adapter_config.json']['sha256'])}
test "$(stat -c %s -- "$route_config")" = {artifacts['adapter_config.json']['size']}
python3 - "$route_manifest" "$route_config" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
if manifest.get("target_step") != 1 or manifest.get("mode") != "full" or config.get("peft_type") != "LORA":
    raise SystemExit("direct route content is not the exported step-1 adapter")
PY
verify_published() {{
python3 - "$runs" "$manifest" "$parent_token" "$final_token" <<'PY'
import hashlib, os, stat, sys
runs, manifest, expected_parent, expected_final = sys.argv[1:]
def open_chain(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    ids = [os.fstat(fd)]
    try:
        for component in path.strip("/").split("/"):
            if component:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = child; ids.append(os.fstat(fd))
        return fd, "/".join(f"{{item.st_dev}}:{{item.st_ino}}" for item in ids)
    except BaseException:
        os.close(fd); raise
parent_fd, parent_identity = open_chain(os.path.join(runs, "qualified-adapters"))
if parent_identity != expected_parent: raise SystemExit("publication ancestor identity changed")
final_fd = os.open(manifest, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
final_stat = os.fstat(final_fd)
if f"{{final_stat.st_dev}}:{{final_stat.st_ino}}" != expected_final: raise SystemExit("publication final identity changed")
step_fd = os.open("step_1", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=final_fd)
expected = [
{expected_entries}
]
for name, digest, size in expected:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=step_fd)
    try:
        stat_result = os.fstat(fd)
        if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size != size: raise SystemExit("publication artifact identity changed")
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            hasher.update(chunk)
        if hasher.hexdigest() != digest: raise SystemExit("publication artifact digest changed")
    finally:
        os.close(fd)
fresh_parent_fd, fresh_parent_identity = open_chain(os.path.join(runs, "qualified-adapters"))
try:
    if fresh_parent_identity != parent_identity: raise SystemExit("publication ancestor identity changed")
    fresh_final_fd = os.open(manifest, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fresh_parent_fd)
    fresh_final = os.fstat(fresh_final_fd)
    if f"{{fresh_final.st_dev}}:{{fresh_final.st_ino}}" != expected_final: raise SystemExit("publication final identity changed")
    os.close(fresh_final_fd)
finally:
    os.close(fresh_parent_fd)
os.close(step_fd); os.close(final_fd); os.close(parent_fd)
PY
}}
verify_published
source_after=$(identity_tree "$source"); test "$source_before" = "$source_after"
verify_published
completed=1
printf 'QUALIFIED_PUBLISH status=%s source_url=%s source_adapter_dir=%s manifest_sha256=%s owner_token=%s parent_token=%s final_token=%s\\n' "$status" "$source_url" "$final" "$manifest" "$owner_token" "$parent_token" "$final_token"
trap - EXIT HUP INT TERM
safe_remove_tmp 2>/dev/null || true
rm -rf -- "$route_tmp" 2>/dev/null || true
'''


def parse_publish_result(value: str | Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value) if isinstance(value, Mapping) else _json(value)
    if set(result) != _PUBLISHED_KEYS:
        _fail("publish result schema is invalid")
    if result.get("status") not in {"published", "existing"}:
        _fail("publish result status is invalid")
    if result.get("source_url") != plan.get("source_url"):
        _fail("publish result source URL differs")
    if result.get("source_adapter_dir") != plan.get("source_adapter_dir"):
        _fail("publish result source adapter path differs")
    if result.get("manifest_sha256") != plan.get("manifest_sha256"):
        _fail("publish result manifest identity differs")
    if not isinstance(result.get("parent_token"), str) or not _TOKEN.fullmatch(result["parent_token"]):
        _fail("publish result parent identity is invalid")
    if not isinstance(result.get("final_token"), str) or not _TOKEN.fullmatch(result["final_token"]):
        _fail("publish result final identity is invalid")
    if result.get("status") == "published" and (not isinstance(result.get("owner_token"), str) or not result["owner_token"] or any(character.isspace() for character in result["owner_token"])):
        _fail("publish result owner identity is missing")
    if result.get("files") != plan.get("artifacts"):
        _fail("publish result file identities differ")
    return result


def build_publication_boundary_script(plan: Mapping[str, Any], *, parent_token: str, final_token: str) -> str:
    """Re-open the published tree without symlink traversal at each load boundary."""
    _validate_plan_paths(plan)
    if not isinstance(parent_token, str) or not _TOKEN.fullmatch(parent_token) or not isinstance(final_token, str) or not _TOKEN.fullmatch(final_token):
        _fail("publication boundary identity is invalid")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping):
        _fail("publication boundary artifact identities are missing")
    artifacts = _artifacts(artifacts)
    entries = ",\n".join(f"    ({name!r}, {artifacts[name]['sha256']!r}, {artifacts[name]['size']})" for name in EXPECTED_FILES)
    runs = _q(plan["runs_root"])
    manifest = _q(plan["manifest_sha256"])
    return f'''set -euo pipefail
python3 - {runs} {manifest} {_q(parent_token)} {_q(final_token)} <<'PY'
import hashlib, os, stat, sys
runs, manifest, expected_parent, expected_final = sys.argv[1:]
def open_chain(path):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    ids = [os.fstat(fd)]
    try:
        for component in path.strip("/").split("/"):
            if component:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd); fd = child; ids.append(os.fstat(fd))
        return fd, "/".join(f"{{item.st_dev}}:{{item.st_ino}}" for item in ids)
    except BaseException:
        os.close(fd); raise
parent_fd, parent_identity = open_chain(os.path.join(runs, "qualified-adapters"))
if parent_identity != expected_parent: raise SystemExit("publication ancestor identity changed")
final_fd = os.open(manifest, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
final_stat = os.fstat(final_fd)
if f"{{final_stat.st_dev}}:{{final_stat.st_ino}}" != expected_final: raise SystemExit("publication final identity changed")
step_fd = os.open("step_1", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=final_fd)
for name, digest, size in [
{entries}
]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=step_fd)
    try:
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_size != size: raise SystemExit("publication artifact identity changed")
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            hasher.update(chunk)
        if hasher.hexdigest() != digest: raise SystemExit("publication artifact digest changed")
    finally:
        os.close(fd)
fresh_parent_fd, fresh_parent_identity = open_chain(os.path.join(runs, "qualified-adapters"))
try:
    if fresh_parent_identity != parent_identity: raise SystemExit("publication ancestor identity changed")
    fresh_final_fd = os.open(manifest, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fresh_parent_fd)
    fresh_final = os.fstat(fresh_final_fd)
    if f"{{fresh_final.st_dev}}:{{fresh_final.st_ino}}" != expected_final: raise SystemExit("publication final identity changed")
    os.close(fresh_final_fd)
finally:
    os.close(fresh_parent_fd)
os.close(step_fd); os.close(final_fd); os.close(parent_fd)
PY
printf 'QUALIFIED_PUBLICATION_BOUNDARY parent_token=%s final_token=%s\\n' {_q(parent_token)} {_q(final_token)}
'''


def build_rollback_script(plan: Mapping[str, Any], *, remove_published: bool) -> str:
    """Remove only this command's fresh slot.

    Publication cleanup uses the separate owner-token script. Keeping the
    paths separate prevents a slot rollback from removing a pre-existing
    publication or source tree.
    """
    if remove_published:
        _fail("publication rollback requires the owner-token cleanup script")
    _validate_plan_paths(plan)
    container = _q(plan.get("engine_container", "glm52-decensor-h200"))
    slot = _q(plan["qualification_slot"])
    port = int(plan["inference_port"])
    return f'''set -euo pipefail
container={container}; slot={slot}
name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null || true)
[ "$name" = "/$container" ]
curl -fsS -X POST -H 'content-type: application/json' -d "{{\"lora_name\":\"$slot\"}}" "http://127.0.0.1:{port}/v1/remove_lora_adapter" >/dev/null
printf 'QUALIFIED_ROLLBACK slot=%s\\n' "$slot"
'''


def build_publication_cleanup_script(plan: Mapping[str, Any], *, owner_token: str) -> str:
    """Remove only a publication created by this load session."""
    _validate_plan_paths(plan)
    if not isinstance(owner_token, str) or not owner_token or any(character.isspace() for character in owner_token):
        _fail("publication owner token is invalid")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(EXPECTED_FILES):
        _fail("publication cleanup artifact identities are incomplete")
    path = _q(plan["source_adapter_dir"])
    final_root = _q(str(plan["source_adapter_dir"]).rsplit("/", 1)[0])
    checks = _artifact_checks("owned", artifacts)
    return f'''set -euo pipefail
published={path}; final_root={final_root}; owner_token={_q(owner_token)}
check_ancestors() {{ current=/; old_ifs="$IFS"; IFS=/; for component in ${{1#/}}; do [ -n "$component" ] || continue; current="$current$component"; test -d "$current" && test ! -L "$current"; current="$current/"; done; IFS="$old_ifs"; }}
check_ancestors "$final_root"
test -d "$final_root" && test ! -L "$final_root"
test "$(stat -c '%d:%i:%s:%y:%z' -- "$final_root" | tr ' ' '_')" = "$owner_token"
owned="$published"
{checks}
rm -rf -- "$final_root"
printf 'QUALIFIED_PUBLICATION_ROLLBACK path=%s\\n' "$final_root"
'''


def build_result_receipt_script(plan: Mapping[str, Any], payload_b64: str) -> str:
    """Publish one immutable result receipt on the source host."""
    _validate_plan_paths(plan)
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", payload_b64):
        _fail("result receipt payload encoding is invalid")
    target = f'{plan["qualification_root"]}/qualified-adapter-load-receipt.json'
    return f'''set -euo pipefail
root={_q(plan["qualification_root"])}; target={_q(target)}
check_ancestors() {{ current=/; old_ifs="$IFS"; IFS=/; for component in ${{1#/}}; do [ -n "$component" ] || continue; current="$current$component"; test -d "$current" && test ! -L "$current"; current="$current/"; done; IFS="$old_ifs"; }}
check_ancestors "$root"
test -d "$root" && test ! -L "$root"
test ! -e "$target" && test ! -L "$target"
tmp=$(mktemp "$root/.qualified-adapter-load-receipt.XXXXXX")
trap 'rm -f -- "$tmp"' EXIT HUP INT TERM
printf %s {_q(payload_b64)} | base64 -d >"$tmp"
test -s "$tmp"
test ! -L "$tmp"
python3 - "$tmp" "$target" <<'PY' # RENAME_NOREPLACE
import ctypes, errno, os, sys
renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
if renameat2 is None: raise SystemExit(95)
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, os.fsencode(sys.argv[1]), -100, os.fsencode(sys.argv[2]), 1) != 0:
    error = ctypes.get_errno()
    raise SystemExit(17 if error == errno.EEXIST else 95)
PY
trap - EXIT HUP INT TERM
printf 'QUALIFIED_RESULT_RECEIPT path=%s sha256=%s size=%s\\n' "$target" "$(sha256sum -- "$target" | awk '{{print $1}}')" "$(stat -c %s -- "$target")"
printf '%s\\n' QUALIFIED_RESULT_RECEIPT_BEGIN
base64 -w0 -- "$target"
printf '\\n%s\\n' QUALIFIED_RESULT_RECEIPT_END
'''
