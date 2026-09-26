"""Guarded correction of legacy policy-0 adapter metadata.

This module is side-effect free until the caller explicitly invokes apply.
It only copies an adapter within the configured trainer storage root.
"""
from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import PurePosixPath
from urllib.parse import quote


LEGACY_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "experts",
)
CANONICAL_TARGETS = tuple(sorted((*LEGACY_TARGETS, "shared_expert")))


class Policy0SourceError(RuntimeError):
    """A policy-0 source correction failed closed."""


def _canonical_step(path_value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(path_value)
    if (
        not path.is_absolute()
        or path.as_posix() != path_value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.name != "step_0"
    ):
        raise Policy0SourceError(f"{label} must be one canonical absolute step_0 path")
    return path


def build_plan(config_dir, source_adapter_dir, destination_adapter_dir, *, repo_root, resolve_config, trainer_head):
    selected = resolve_config(config_dir, require_orchestrator=True)
    source = _canonical_step(source_adapter_dir, "source adapter directory")
    destination = _canonical_step(destination_adapter_dir, "destination adapter directory")
    if source == destination:
        raise Policy0SourceError("source and destination adapter directories must differ")
    try:
        with (repo_root / selected["orchestrator"]).open("rb") as stream:
            orchestrator = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Policy0SourceError(f"cannot read policy-0 source config: {error}") from error
    broadcast = orchestrator.get("weight_broadcast")
    source_url = broadcast.get("source_url") if isinstance(broadcast, dict) else None
    configured_root = broadcast.get("source_dir") if isinstance(broadcast, dict) else None
    if (
        not isinstance(source_url, str)
        or not source_url.startswith(("http://", "https://"))
        or not isinstance(configured_root, str)
    ):
        raise Policy0SourceError("weight_broadcast source URL and source_dir are required")
    source_root = PurePosixPath(configured_root)
    if (
        not source_root.is_absolute()
        or source_root.as_posix() != configured_root
        or any(part in {"", ".", ".."} for part in source_root.parts)
    ):
        raise Policy0SourceError("configured weight_broadcast.source_dir is unsafe")
    try:
        source_relative = source.relative_to(source_root)
        destination_relative = destination.relative_to(source_root)
    except ValueError as error:
        raise Policy0SourceError("source and destination must be below weight_broadcast.source_dir") from error
    if not source_relative.parts or not destination_relative.parts:
        raise Policy0SourceError("source and destination must be strictly below weight_broadcast.source_dir")
    if "broadcasts" not in source.parts[:-1]:
        raise Policy0SourceError("source adapter directory must be the preserved broadcasts step_0")
    if not any("qualification" in part.lower() for part in destination_relative.parts[:-1]):
        raise Policy0SourceError("destination must be a qualification-only path below source_dir")
    source_url = source_url.rstrip("/")
    if not source_url.endswith("/" + source_root.name):
        raise Policy0SourceError("configured source URL does not name its source_dir")
    source_url = source_url + "/" + quote(source_relative.as_posix(), safe="/")
    destination_url = source_url[: -len(quote(source_relative.as_posix(), safe="/"))]
    destination_url += quote(destination_relative.as_posix(), safe="/")
    plan = {
        "config_dir": config_dir,
        "trainer_head": trainer_head,
        "source_dir": source.as_posix(),
        "source_adapter": (source / "adapter_model.safetensors").as_posix(),
        "source_config": (source / "adapter_config.json").as_posix(),
        "destination_dir": destination.as_posix(),
        "destination_adapter": (destination / "adapter_model.safetensors").as_posix(),
        "destination_config": (destination / "adapter_config.json").as_posix(),
        "configured_source_dir": configured_root,
        "source_url": source_url,
        "destination_url": destination_url,
        "legacy_targets": list(LEGACY_TARGETS),
        "canonical_lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
            "target_modules": list(CANONICAL_TARGETS),
        },
    }
    plan["apply_script"] = build_apply_script(plan)
    return plan


def build_apply_script(plan):
    legacy_json = json.dumps(list(LEGACY_TARGETS), separators=(",", ":"))
    canonical_json = json.dumps(list(CANONICAL_TARGETS), separators=(",", ":"))
    source_dir = shlex.quote(str(plan["source_dir"]))
    source_adapter = shlex.quote(str(plan["source_adapter"]))
    source_config = shlex.quote(str(plan["source_config"]))
    destination = shlex.quote(str(plan["destination_dir"]))
    configured_root = shlex.quote(str(plan["configured_source_dir"]))
    return f"""set -euo pipefail
source_dir={source_dir}
source_adapter={source_adapter}
source_config={source_config}
destination={destination}
source_root={configured_root}
test -f "$source_adapter" && test ! -L "$source_adapter"
test -f "$source_config" && test ! -L "$source_config"
test -f "$source_dir/STABLE" && test ! -L "$source_dir/STABLE"
test ! -e "$source_dir/WRITING" && test ! -e "$source_dir/.writing" && test ! -e "$source_dir/.lock"
if find "$source_dir" -maxdepth 1 -type f -name '*.partial' -print -quit | grep -q .; then
  echo 'source adapter has a partial writer artifact' >&2
  exit 41
fi
source_real=$(realpath -e -- "$source_dir")
[ "$source_real" = "$source_dir" ]
source_root_real=$(realpath -e -- "$source_root")
case "$source_real/" in "$source_root_real/"*) ;; *) echo "source escapes source_dir" >&2; exit 43 ;; esac
python3 - "$source_root" "$source_dir" "$destination" <<'PY'
from pathlib import Path
import sys
root, source, destination = map(Path, sys.argv[1:])
for value in (source, destination.parent):
    try:
        parts = value.relative_to(root).parts
    except ValueError:
        raise SystemExit("path is outside source_dir")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise SystemExit("source or destination path contains a symlink")
PY
source_sha=$(sha256sum -- "$source_adapter" | cut -d' ' -f1)
source_size=$(stat -c '%s' -- "$source_adapter")
python3 - "$source_config" {shlex.quote(legacy_json)} <<'PY'
import json, math, sys
path, legacy_json = sys.argv[1:]
value = json.loads(open(path, encoding="utf-8").read())
allowed = {{'peft_type', 'task_type', 'base_model_name_or_path', 'r', 'lora_alpha', 'lora_dropout', 'bias', 'target_modules', 'modules_to_save'}}
if not isinstance(value, dict) or set(value) - allowed:
    raise SystemExit("adapter config has unexpected fields")
if value.get("peft_type", "LORA") != "LORA" or value.get("task_type", "CAUSAL_LM") != "CAUSAL_LM" or value.get("bias", "none") != "none":
    raise SystemExit("adapter config has unexpected PEFT settings")
if type(value.get("r")) is not int or value["r"] != 32:
    raise SystemExit("adapter config rank is not 32")
alpha = value.get("lora_alpha")
if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) != 64:
    raise SystemExit("adapter config alpha is not numeric 64")
dropout = value.get("lora_dropout")
if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(float(dropout)) or float(dropout) != 0:
    raise SystemExit("adapter config dropout is not zero")
targets = value.get("target_modules")
legacy = json.loads(legacy_json)
if not isinstance(targets, list) or len(targets) != len(legacy) or set(targets) != set(legacy):
    raise SystemExit("adapter config is not the recognized eight-target legacy form")
if "shared_expert" in targets:
    raise SystemExit("adapter config is already canonical")
PY
destination_parent=$(dirname -- "$destination")
if [ -e "$destination" ] && [ -L "$destination" ]; then
  echo "destination adapter directory is a symlink" >&2
  exit 42
fi
mkdir -p -- "$destination_parent"
parent_real=$(realpath -e -- "$destination_parent")
case "$parent_real/" in "$source_root_real/"*) ;; *) echo "destination escapes source_dir" >&2; exit 43 ;; esac
if [ -e "$destination" ]; then
  test -d "$destination" && test ! -L "$destination"
  test -f "$destination/STABLE" && test ! -L "$destination/STABLE"
  test -f "$destination/adapter_model.safetensors" && test ! -L "$destination/adapter_model.safetensors"
  test -f "$destination/adapter_config.json" && test ! -L "$destination/adapter_config.json"
  [ "$(sha256sum -- "$destination/adapter_model.safetensors" | cut -d' ' -f1)" = "$source_sha" ]
  [ "$(stat -c '%s' -- "$destination/adapter_model.safetensors")" = "$source_size" ]
  python3 - "$destination/adapter_config.json" {shlex.quote(canonical_json)} <<'PY'
import json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
if type(value.get("lora_alpha")) is not int or value["lora_alpha"] != 64:
    raise SystemExit("existing destination configuration conflicts")
if value.get("target_modules") != sorted(json.loads(sys.argv[2])):
    raise SystemExit("existing destination configuration conflicts")
PY
  config_sha=$(sha256sum -- "$destination/adapter_config.json" | cut -d' ' -f1)
  config_size=$(stat -c '%s' -- "$destination/adapter_config.json")
  printf "SOURCE_RESULT idempotent %s %s %s %s\n" "$source_sha" "$source_size" "$config_sha" "$config_size"
  exit 0
fi
tmp=$(mktemp -d "$destination_parent/.policy0-source.XXXXXX")
cleanup() {{ rm -rf -- "$tmp"; }}
trap cleanup EXIT
cp --reflink=auto -- "$source_adapter" "$tmp/adapter_model.safetensors"
[ "$(sha256sum -- "$tmp/adapter_model.safetensors" | cut -d' ' -f1)" = "$source_sha" ]
[ "$(stat -c '%s' -- "$tmp/adapter_model.safetensors")" = "$source_size" ]
python3 - "$source_config" "$tmp/adapter_config.json" {shlex.quote(canonical_json)} <<'PY'
import json, sys
source, target, canonical_json = sys.argv[1:]
value = json.loads(open(source, encoding="utf-8").read())
value["lora_alpha"] = 64
value["target_modules"] = sorted(json.loads(canonical_json))
with open(target, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2)
PY
python3 - "$tmp/adapter_config.json" {shlex.quote(canonical_json)} <<'PY'
import json, sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
if type(value.get("lora_alpha")) is not int or value["lora_alpha"] != 64 or value.get("target_modules") != sorted(json.loads(sys.argv[2])):
    raise SystemExit("canonical configuration verification failed")
PY
config_sha=$(sha256sum -- "$tmp/adapter_config.json" | cut -d' ' -f1)
config_size=$(stat -c '%s' -- "$tmp/adapter_config.json")
: > "$tmp/STABLE"
mv -- "$tmp" "$destination"
trap - EXIT
printf "SOURCE_RESULT created %s %s %s %s\n" "$source_sha" "$source_size" "$config_sha" "$config_size"
"""


def parse_result(output, plan):
    records = [line.split() for line in output.splitlines() if line.startswith("SOURCE_RESULT ")]
    if len(records) != 1 or len(records[0]) != 6 or records[0][1] not in {"created", "idempotent"}:
        raise Policy0SourceError("policy-0 source preparation returned no valid identity")
    _, status, adapter_sha, adapter_size, config_sha, config_size = records[0]
    if (
        not re.fullmatch(r"[0-9a-f]{64}", adapter_sha)
        or not re.fullmatch(r"[0-9a-f]{64}", config_sha)
        or not adapter_size.isdigit()
        or not config_size.isdigit()
    ):
        raise Policy0SourceError("policy-0 source preparation returned invalid identities")
    return {
        "status": status,
        "source_url": plan["source_url"],
        "destination_url": plan["destination_url"],
        "adapter_identity": {"sha256": adapter_sha, "size": int(adapter_size)},
        "config_identity": {"sha256": config_sha, "size": int(config_size)},
    }
