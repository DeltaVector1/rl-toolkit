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
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "h200_step1_adapter_delta":
        _fail("delta receipt schema or kind is invalid")
    if not isinstance(receipt.get("run_id"), str) or not receipt["run_id"]:
        _fail("delta receipt run identity is missing")
    model = _identity(receipt.get("step1_adapter"), "step-1 adapter")
    config = _identity(receipt.get("step1_config"), "step-1 config")
    manifest_sha = _digest(receipt.get("step1_manifest_sha256"), "step-1 manifest")
    before = receipt.get("identities_before")
    after = receipt.get("identities_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping) or before != after:
        _fail("delta receipt identity before/after values differ")
    if before.get("step1_adapter") != model or before.get("step1_config") != config:
        _fail("delta receipt step-1 identity chain is inconsistent")
    policy_model = _identity(receipt.get("policy0_adapter"), "policy-0 adapter")
    policy_config = _identity(receipt.get("policy0_config"), "policy-0 config")
    if before.get("policy0_adapter") != policy_model or before.get("policy0_config") != policy_config:
        _fail("delta receipt policy-0 identity chain is inconsistent")
    manifest = _identity(before.get("step1_manifest"), "step-1 manifest")
    if manifest["sha256"] != manifest_sha:
        _fail("delta receipt manifest identity differs")
    artifacts = _artifacts(before.get("artifacts"))
    if artifacts["adapter_model.safetensors"] != model or artifacts["adapter_config.json"] != config or artifacts["adapter_manifest.json"] != manifest:
        _fail("delta receipt artifact identity chain is inconsistent")
    if receipt.get("trainer_evidence", {}).get("export_completed") is not True:
        _fail("delta receipt does not prove export completion")
    return receipt


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
    if trainer_head != "n82" or not isinstance(inference_host, str) or not inference_host:
        _fail("qualified adapter load host identity is invalid")
    if type(inference_port) is not int or not 1 <= inference_port <= 65535 or type(direct_route_port) is not int or direct_route_port != 18100:
        _fail("qualified adapter load route is invalid")
    manifest_sha = _digest(receipt.get("step1_manifest_sha256"), "step-1 manifest")
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
        "receipt_sha256": expected_receipt_sha256,
        "artifacts": dict(receipt["identities_after"]["artifacts"]),
    }


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
    lines.append(f'test "$(find "${variable}" -mindepth 1 -maxdepth 1 -type f -printf \'%f\\n\' | sort | paste -sd, -)" = "STABLE,adapter_config.json,adapter_manifest.json,adapter_model.safetensors,adapter_model.safetensors.sha256"')
    return "\n".join(lines)


def build_publish_script(plan: Mapping[str, Any]) -> str:
    """Build the n82-only no-clobber publication and direct-route probe."""
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(EXPECTED_FILES):
        _fail("publish plan artifact identities are incomplete")
    root = _q(plan["qualification_root"])
    source = _q(plan["source_step1_dir"])
    runs = _q(plan["runs_root"])
    source_url = _q(plan["source_url"])
    manifest = _q(plan["manifest_sha256"])
    final_root = f'{runs}/qualified-adapters/{plan["manifest_sha256"]}'
    final = f'{final_root}/step_1'
    source_checks = _artifact_checks("source", artifacts)
    tmp_checks = _artifact_checks("tmp", artifacts)
    return f'''set -euo pipefail
root={root}; source={source}; runs={runs}; final={_q(final)}; final_root={_q(final_root)}; source_url={source_url}; manifest={manifest}
test -d "$root" && test ! -L "$root"
test -d "$source" && test ! -L "$source"
identity_tree() {{ for item in adapter_config.json adapter_model.safetensors adapter_model.safetensors.sha256 adapter_manifest.json STABLE; do sha256sum -- "$1/$item"; stat -c '%s %a' -- "$1/$item"; done; }}
source_before=$(identity_tree "$source")
{source_checks}
parent="$runs/qualified-adapters"; mkdir -p -- "$parent"; test ! -L "$parent"
tmp_root=$(mktemp -d "$parent/.qualified-adapter-{plan['manifest_sha256']}.XXXXXX")
tmp="$tmp_root/step_1"; mkdir -- "$tmp"
cleanup() {{ if [ "${{published_new:-0}}" = 0 ]; then rm -rf -- "$tmp_root"; fi; }}
trap cleanup EXIT HUP INT TERM
for item in adapter_config.json adapter_model.safetensors adapter_model.safetensors.sha256 adapter_manifest.json STABLE; do
  cp --reflink=auto -- "$source/$item" "$tmp/$item"
  test ! -L "$tmp/$item"
  test "$(stat -c '%d:%i' -- "$source/$item")" != "$(stat -c '%d:%i' -- "$tmp/$item")"
done
{tmp_checks.replace('$$', '$')}
source_after=$(identity_tree "$source"); test "$source_before" = "$source_after"
atomic_noreplace() {{ # RENAME_NOREPLACE
python3 - "$1" "$2" <<'PY'
import ctypes, errno, os, sys
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = getattr(libc, "renameat2", None)
if renameat2 is None: raise SystemExit(95)
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, os.fsencode(sys.argv[1]), -100, os.fsencode(sys.argv[2]), 1) != 0:
    error = ctypes.get_errno()
    raise SystemExit(17 if error == errno.EEXIST else 95)
PY
}}
published_new=0
if [ -e "$final_root" ] || [ -L "$final_root" ]; then
  test -d "$final" && test ! -L "$final"
  existing="$final"; {tmp_checks.replace('tmp', 'existing')}
  rm -rf -- "$tmp_root"; published_new=1
  status=existing
else
  if atomic_noreplace "$tmp_root" "$final_root"; then
    published_new=1; status=published
  elif [ -d "$final" ] && [ ! -L "$final" ]; then
    existing="$final"; {tmp_checks.replace('tmp', 'existing')}
    rm -rf -- "$tmp_root"; published_new=1; status=existing
  else
    echo 'existing destination appeared with a conflicting identity' >&2; exit 96
  fi
fi
curl -fsS "$source_url/adapter_manifest.json" >/dev/null
source_after=$(identity_tree "$source"); test "$source_before" = "$source_after"
printf 'QUALIFIED_PUBLISH status=%s source_url=%s source_adapter_dir=%s manifest_sha256=%s\\n' "$status" "$source_url" "$final" "$manifest"
trap - EXIT HUP INT TERM
rm -rf -- "$tmp_root" 2>/dev/null || true
'''


def parse_publish_result(value: str | Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value) if isinstance(value, Mapping) else _json(value)
    if result.get("status") not in {"published", "existing"}:
        _fail("publish result status is invalid")
    if result.get("source_url") != plan.get("source_url"):
        _fail("publish result source URL differs")
    if result.get("source_adapter_dir") != plan.get("source_adapter_dir"):
        _fail("publish result source adapter path differs")
    if result.get("manifest_sha256") != plan.get("manifest_sha256"):
        _fail("publish result manifest identity differs")
    if "files" in result and result["files"] != plan.get("artifacts"):
        _fail("publish result file identities differ")
    return result


def build_rollback_script(plan: Mapping[str, Any], *, remove_published: bool) -> str:
    """Remove only this command's fresh slot and, when owned, its publication."""
    container = _q(plan.get("engine_container", "glm52-decensor-h200"))
    slot = _q(plan["qualification_slot"])
    published = _q(plan["source_adapter_dir"])
    port = int(plan["inference_port"])
    remove_path = ""
    if remove_published:
        remove_path = f'test -d {published} && test ! -L {published} && rm -rf -- {published} || true\n'
    return f'''set -euo pipefail
container={container}; slot={slot}
name=$(sudo docker inspect --format '{{{{.Name}}}}' "$container" 2>/dev/null || true)
[ "$name" = "/$container" ]
curl -fsS -X POST -H 'content-type: application/json' -d "{{\"lora_name\":\"$slot\"}}" "http://127.0.0.1:{port}/v1/remove_lora_adapter" >/dev/null
{remove_path}printf 'QUALIFIED_ROLLBACK slot=%s\\n' "$slot"
'''


def build_publication_cleanup_script(plan: Mapping[str, Any]) -> str:
    """Remove only a publication created by this load session."""
    path = _q(plan["source_adapter_dir"])
    return f'''set -euo pipefail
published={path}
test -d "$published" && test ! -L "$published"
rm -rf -- "$published"
printf 'QUALIFIED_PUBLICATION_ROLLBACK path=%s\\n' "$published"
'''


def build_result_receipt_script(plan: Mapping[str, Any], payload_b64: str) -> str:
    """Publish one immutable result receipt on the source host."""
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", payload_b64):
        _fail("result receipt payload encoding is invalid")
    target = f'{plan["qualification_root"]}/qualified-adapter-load-receipt.json'
    return f'''set -euo pipefail
root={_q(plan["qualification_root"])}; target={_q(target)}
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
'''
