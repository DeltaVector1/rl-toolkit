#!/usr/bin/env python3
"""Build and verify a direct GLM trainer-to-inference parity record."""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


schema_version = 1
required_targets = (
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
)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_sha256_fd(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while block := os.pread(file_descriptor, 8 * 1024 * 1024, offset):
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def open_directory(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        require_directory_identity(path, descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def require_directory_identity(path: Path, descriptor: int) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise ValueError(f"directory identity changed: {path}")


def require_child_directory_identity(
    parent_descriptor: int, name: str, descriptor: int
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise ValueError(f"directory identity changed: {name}")


def require_child_regular_identity(
    parent_descriptor: int, name: str, descriptor: int
) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or any(getattr(opened, field) != getattr(current, field) for field in fields)
    ):
        raise ValueError(f"file identity changed: {name}")


def open_regular_at(directory_descriptor: int, name: str) -> int:
    if Path(name).name != name or not name:
        raise ValueError("relative evidence filename is invalid")
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor
    )
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"evidence file is not regular: {name}")
    return descriptor


def read_json_at(
    directory_descriptor: int, name: str
) -> tuple[dict[str, Any], str]:
    descriptor = open_regular_at(directory_descriptor, name)
    try:
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            value = json.load(stream)
        digest = file_sha256_fd(descriptor)
        require_child_regular_identity(directory_descriptor, name, descriptor)
    finally:
        os.close(descriptor)
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence is not an object: {name}")
    return value, digest


@dataclass(frozen=True)
class Profile:
    path: Path
    values: dict[str, Any]

    def get(self, name: str) -> Any:
        return self.values[name]


def load_profile(path: Path) -> Profile:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    exact = {
        "schema_version": schema_version,
        "quantization": "native_fp8",
        "model_architecture": "GlmMoeDsaForCausalLM",
        "context_length": 32768,
        "h200_scoring_context_length": 32769,
        "prompt_length": 16384,
        "max_output_length": 16384,
        "vocab_size": 154880,
        "hidden_layers": 78,
        "dense_layers": 3,
        "experts": 256,
        "experts_per_token": 8,
        "router_scale": 2.5,
        "lora_rank": 32,
        "lora_alpha": 64,
        "dppo_mask_low": 0.2,
        "dppo_mask_high": 0.2,
        "max_unmasked_mismatch_kl": 0.1,
        "max_masked_mismatch_kl": 0.5,
        "max_masked_fraction": 0.06,
        "max_router_weight_sum_error": 0.005,
    }
    for name, expected in exact.items():
        if values.get(name) != expected:
            raise ValueError(f"profile {name} must be {expected!r}")
    if tuple(values.get("lora_targets", ())) != required_targets:
        raise ValueError("profile does not contain the exact full LoRA target list")
    if (
        values.get("router_weight_delta_status")
        != "provisional_until_first_base_and_adapter_capture"
    ):
        raise ValueError("router weight delta status must remain provisional")
    router_weight_limit = values.get("max_router_weight_abs_delta")
    if (
        isinstance(router_weight_limit, bool)
        or not isinstance(router_weight_limit, (int, float))
        or router_weight_limit <= 0
    ):
        raise ValueError("provisional router weight delta limit must be positive")
    if values["prompt_length"] + values["max_output_length"] != values["context_length"]:
        raise ValueError("prompt and output lengths do not fill the context")
    if not isinstance(values.get("prime_rl_revision"), str) or len(values["prime_rl_revision"]) != 40:
        raise ValueError("profile needs one full Prime-RL revision")
    if values.get("max_fixture_age_hours", 0) <= 0:
        raise ValueError("max_fixture_age_hours must be positive")
    return Profile(path=path, values=values)


def parse_time(value: str | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.timezone.utc)
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("created time has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def make_fixture(
    profile: Profile,
    output: Path,
    seed: int | None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    seed = secrets.randbits(64) if seed is None else seed
    length = profile.get("context_length")
    vocab_size = profile.get("vocab_size")
    generator = np.random.Generator(np.random.PCG64(seed))
    token_ids = generator.integers(256, vocab_size, size=length, dtype=np.int64)
    positions = np.arange(length, dtype=np.int64)
    selected = np.arange(profile.get("prompt_length"), length, dtype=np.int64)
    advantages = np.where((token_ids[selected] ^ selected) & 1, 1.0, -1.0).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "token_ids": token_ids,
        "positions": positions,
        "selected_indices": selected,
        "advantages": advantages,
    }
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(output / filename, array, allow_pickle=False)
        entries[name] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_fixture",
        "created_utc": parse_time(created_utc).isoformat(),
        "seed": seed,
        "profile_sha256": file_sha256(profile.path),
        "context_length": length,
        "prompt_length": profile.get("prompt_length"),
        "max_output_length": profile.get("max_output_length"),
        "arrays": entries,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def _array_bytes_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def refresh_fixture_from_trainer_export(
    profile: Profile,
    source_fixture: Path,
    source_session_path: Path,
    token_export: Path | None,
    diagnostic_input: Path,
    output: Path,
    diagnostic_output: Path,
    expected_rank: int,
    created_utc: str | None = None,
    compact_source: Path | None = None,
    compact_manifest_path: Path | None = None,
) -> dict[str, Any]:
    source_manifest = load_manifest(source_fixture, "glm52_policy_parity_fixture")
    source_wrapper_sha256 = fixture_digest(source_fixture)
    source_arrays = {
        "token_ids": load_array(source_fixture, source_manifest, "token_ids").astype(
            np.int64
        ),
        "positions": load_array(source_fixture, source_manifest, "positions").astype(
            np.int64
        ),
        "selected_indices": load_array(
            source_fixture, source_manifest, "selected_indices"
        ).astype(np.int64),
        "advantages": load_array(
            source_fixture, source_manifest, "advantages"
        ).astype(np.float32),
    }
    recorded_source = source_manifest.get("source")
    source_fixture_sha256 = source_wrapper_sha256
    if recorded_source is not None:
        source_fields = {
            "fixture_sha256",
            "session_identity",
            "token_export_sha256",
            "rank",
            "fixture_semantic_sha256",
            "export_semantic_sha256",
        }
        fixture_semantic_hashes = {
            name: _array_bytes_sha256(array)
            for name, array in source_arrays.items()
        }
        if (
            not isinstance(recorded_source, dict)
            or set(recorded_source) != source_fields
            or re.fullmatch(
                r"[0-9a-f]{64}", str(recorded_source.get("fixture_sha256", ""))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(recorded_source.get("session_identity", ""))
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}", str(recorded_source.get("token_export_sha256", ""))
            )
            is None
            or recorded_source.get("rank") != expected_rank
            or recorded_source.get("fixture_semantic_sha256")
            != fixture_semantic_hashes
            or not isinstance(recorded_source.get("export_semantic_sha256"), dict)
        ):
            raise ValueError("refreshed fixture source provenance is invalid")
        source_fixture_sha256 = recorded_source["fixture_sha256"]
    source_session = json.loads(source_session_path.read_text())
    source_session_identity = hashlib.sha256(
        json.dumps(source_session, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        source_session.get("schema_version") != schema_version
        or source_session.get("side") != "trainer"
        or source_session.get("fixture_sha256") != source_fixture_sha256
        or source_session.get("token_ids_sha256")
        != _array_bytes_sha256(source_arrays["token_ids"])
        or re.fullmatch(
            r"[0-9a-f]{40}", str(source_session.get("model_revision", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source_session.get("adapter_sha256", ""))
        )
        is None
        or (
            recorded_source is not None
            and recorded_source.get("session_identity") != source_session_identity
        )
    ):
        raise ValueError("trainer session does not identify the source fixture")
    fields = ("token_ids", "position_ids", "loss_mask", "advantages")
    if compact_source is not None or compact_manifest_path is not None:
        if token_export is not None or compact_source is None or compact_manifest_path is None:
            raise ValueError("trainer fixture refresh source selection is invalid")
        if (
            compact_source.is_symlink()
            or compact_manifest_path.is_symlink()
            or compact_source.stat().st_size > 18 * 1024 * 1024
            or compact_manifest_path.stat().st_size > 1024 * 1024
        ):
            raise ValueError("trainer compact fixture source is too large or is a symlink")
        compact_manifest = json.loads(compact_manifest_path.read_text())
        claimed_content = compact_manifest.pop("content_sha256", None)
        if (
            compact_manifest.get("schema_version") != schema_version
            or compact_manifest.get("kind")
            != "glm52_policy_parity_trainer_fixture_source"
            or compact_manifest.get("rank") != expected_rank
            or compact_manifest.get("file") != "trainer-fixture-source.npz"
            or re.fullmatch(
                r"[0-9a-f]{64}", str(compact_manifest.get("binding_sha256", ""))
            )
            is None
            or compact_manifest.get("file_sha256") != file_sha256(compact_source)
            or claimed_content != hashlib.sha256(json_bytes(compact_manifest)).hexdigest()
        ):
            raise ValueError("trainer compact fixture manifest is invalid")
        with zipfile.ZipFile(compact_source) as archive:
            expected_members = {
                "token_ids.npy",
                "positions.npy",
                "selected_indices.npy",
                "advantages.npy",
            }
            infos = archive.infolist()
            if {info.filename for info in infos} != expected_members or len(infos) != 4:
                raise ValueError("trainer compact fixture NPZ members are invalid")
            if sum(info.file_size for info in infos) > 18 * 1024 * 1024:
                raise ValueError("trainer compact fixture NPZ is too large")
        with np.load(compact_source, allow_pickle=False) as source_values:
            token_ids = np.asarray(source_values["token_ids"])
            positions = np.asarray(source_values["positions"])
            selected = np.asarray(source_values["selected_indices"])
            advantages = np.asarray(source_values["advantages"])
        length = profile.get("context_length")
        if (
            token_ids.dtype != np.int64
            or positions.dtype != np.int64
            or selected.dtype != np.int64
            or advantages.dtype != np.float32
            or token_ids.shape != (length,)
            or positions.shape != (length,)
            or selected.shape != advantages.shape
            or selected.ndim != 1
            or selected.size < 1
            or selected[0] < 0
            or selected[-1] >= length
            or np.any(selected[1:] <= selected[:-1])
            or not np.isfinite(advantages).all()
        ):
            raise ValueError("trainer compact fixture arrays are invalid")
        entries = compact_manifest.get("arrays")
        if not isinstance(entries, dict) or set(entries) != {
            "token_ids",
            "positions",
            "selected_indices",
            "advantages",
        }:
            raise ValueError("trainer compact fixture array manifest is invalid")
        for name, value in {
            "token_ids": token_ids,
            "positions": positions,
            "selected_indices": selected,
            "advantages": advantages,
        }.items():
            entry = entries.get(name)
            if (
                not isinstance(entry, dict)
                or entry.get("dtype") != str(value.dtype)
                or entry.get("shape") != list(value.shape)
                or entry.get("sha256") != _array_bytes_sha256(value)
            ):
                raise ValueError(f"trainer compact fixture {name} hash is invalid")
        loss_mask = np.zeros(length, dtype=np.bool_)
        loss_mask[selected] = True
        full_advantages = np.zeros(length, dtype=np.float32)
        full_advantages[selected] = advantages
        first = {
            "schema_version": schema_version,
            "rank": expected_rank,
            "token_ids": token_ids.tolist(),
            "position_ids": positions.tolist(),
            "loss_mask": loss_mask.tolist(),
            "advantages": full_advantages.tolist(),
        }
        records = [first]
        trainer_source_sha256 = file_sha256(compact_source)
    else:
        if token_export is None:
            raise ValueError("trainer token export is missing")
        if token_export.stat().st_size > 100_000_000 or token_export.is_symlink():
            raise ValueError("trainer token export is too large or is a symlink")
        records = [json.loads(line) for line in token_export.read_text().splitlines() if line]
        if not records:
            raise ValueError("trainer token export contains no records")
        first = records[0]
        for record in records:
            if record.get("schema_version") != schema_version or record.get("rank") != expected_rank:
                raise ValueError("trainer token export provenance is invalid")
            if any(record.get(name) != first.get(name) for name in fields):
                raise ValueError("trainer token export contains different fixture records")
        trainer_source_sha256 = file_sha256(token_export)
    token_ids = np.asarray(first.get("token_ids"), dtype=np.int64)
    positions = np.asarray(first.get("position_ids"), dtype=np.int64)
    loss_mask = np.asarray(first.get("loss_mask"), dtype=np.bool_)
    full_advantages = np.asarray(first.get("advantages"), dtype=np.float32)
    length = profile.get("context_length")
    if any(array.shape != (length,) for array in (token_ids, positions, loss_mask, full_advantages)):
        raise ValueError("trainer token export arrays do not match the parity context")
    selected = np.flatnonzero(loss_mask).astype(np.int64)
    advantages = full_advantages[selected].astype(np.float32)
    if selected.shape != (profile.get("max_output_length"),):
        raise ValueError("trainer token export loss mask does not select the expected output")
    refreshed_arrays = {
        "token_ids": token_ids,
        "positions": positions,
        "selected_indices": selected,
        "advantages": advantages,
    }
    for name in refreshed_arrays:
        if not np.array_equal(refreshed_arrays[name], source_arrays[name]):
            raise ValueError(f"trainer token export {name} differs from its source fixture")
    expected_loss_mask = np.zeros(length, dtype=np.bool_)
    expected_loss_mask[source_arrays["selected_indices"]] = True
    expected_full_advantages = np.zeros(length, dtype=np.float32)
    expected_full_advantages[source_arrays["selected_indices"]] = source_arrays[
        "advantages"
    ]
    if not np.array_equal(loss_mask, expected_loss_mask):
        raise ValueError("trainer token export loss mask differs from its source fixture")
    if not np.array_equal(full_advantages, expected_full_advantages):
        raise ValueError("trainer token export full advantages differ from its source fixture")
    token_ids_sha256 = _array_bytes_sha256(token_ids)
    if source_session.get("token_ids_sha256") != token_ids_sha256:
        raise ValueError("trainer token export hash differs from its session")
    export_semantic_hashes = {
        "token_ids": token_ids_sha256,
        "positions": _array_bytes_sha256(positions),
        "loss_mask": _array_bytes_sha256(loss_mask),
        "advantages": _array_bytes_sha256(full_advantages),
    }
    if recorded_source is not None and (
        recorded_source.get("token_export_sha256") != trainer_source_sha256
        or recorded_source.get("export_semantic_sha256") != export_semantic_hashes
    ):
        raise ValueError("trainer export differs from recorded fixture source")
    validate_diagnostic_file(diagnostic_input)
    with np.load(diagnostic_input, allow_pickle=False) as values:
        diagnostic = {name: values[name] for name in values.files}
    expected_diagnostic = {
        "session_identity": source_session_identity,
        "fixture_sha256": source_fixture_sha256,
        "token_ids_sha256": token_ids_sha256,
        "model_revision": source_session.get("model_revision"),
        "adapter_sha256": source_session.get("adapter_sha256"),
    }
    for name, expected in expected_diagnostic.items():
        value = diagnostic.get(name)
        if value is None or str(value.item()) != expected:
            raise ValueError(f"trainer diagnostic {name} differs from its session")
    diagnostic_source = diagnostic.get("source_fixture_sha256")
    if (
        diagnostic_source is not None
        and str(diagnostic_source.item()) != source_fixture_sha256
    ):
        raise ValueError("trainer diagnostic source fixture differs from its session")
    if output.exists() or diagnostic_output.exists():
        raise FileExistsError("refreshed parity fixture output already exists")
    try:
        output.mkdir(parents=False)
        entries: dict[str, Any] = {}
        fixture_semantic_hashes: dict[str, str] = {}
        for name, array in refreshed_arrays.items():
            filename = f"{name}.npy"
            np.save(output / filename, array, allow_pickle=False)
            entries[name] = {
                "file": filename,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": file_sha256(output / filename),
            }
            fixture_semantic_hashes[name] = _array_bytes_sha256(array)
        manifest_created_utc = (
            source_manifest["created_utc"]
            if recorded_source is not None
            else parse_time(created_utc).isoformat()
        )
        manifest = {
            "schema_version": schema_version,
            "kind": "glm52_policy_parity_fixture",
            "created_utc": manifest_created_utc,
            "seed": source_manifest["seed"],
            "profile_sha256": file_sha256(profile.path),
            "context_length": length,
            "prompt_length": profile.get("prompt_length"),
            "max_output_length": profile.get("max_output_length"),
            "source": {
                "fixture_sha256": source_fixture_sha256,
                "session_identity": source_session_identity,
                "token_export_sha256": trainer_source_sha256,
                "rank": expected_rank,
                "fixture_semantic_sha256": fixture_semantic_hashes,
                "export_semantic_sha256": export_semantic_hashes,
            },
            "arrays": entries,
        }
        (output / "manifest.json").write_bytes(json_bytes(manifest))
        check_fresh(profile, output, manifest)
        refreshed_fixture_sha256 = fixture_digest(output)
        if (
            recorded_source is not None
            and refreshed_fixture_sha256 != source_wrapper_sha256
        ):
            raise ValueError("refreshed fixture wrapper reconstruction differs")
        diagnostic["source_fixture_sha256"] = np.asarray(
            source_fixture_sha256, dtype="<U64"
        )
        diagnostic["fixture_sha256"] = np.asarray(
            refreshed_fixture_sha256, dtype="<U64"
        )
        with diagnostic_output.open("xb") as stream:
            np.savez(stream, **diagnostic)
        validate_diagnostic_file(diagnostic_output)
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        diagnostic_output.unlink(missing_ok=True)
        raise
    return {
        "fixture_sha256": refreshed_fixture_sha256,
        "input_fixture_sha256": source_wrapper_sha256,
        "source_fixture_sha256": source_fixture_sha256,
        "token_ids_sha256": token_ids_sha256,
        "positions_sha256": export_semantic_hashes["positions"],
        "advantages_sha256": export_semantic_hashes["advantages"],
        "token_export_sha256": trainer_source_sha256,
        "records": len(records),
        "rank": expected_rank,
    }


@contextlib.contextmanager
def open_h200_capture_snapshot(
    path: Path, selected_arrays: set[str] | None = None
):
    kind = "glm52_policy_parity_h200_capture"
    selected_arrays = selected_arrays or set()
    root_descriptor = open_directory(path)
    try:
        manifest, manifest_sha256 = read_json_at(root_descriptor, "manifest.json")
        if (
            manifest.get("schema_version") != schema_version
            or manifest.get("kind") != kind
        ):
            raise ValueError(f"invalid {kind} manifest in {path}")
        arrays = manifest.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != {
            "full_logprobs",
            "selected_logprobs",
            "eager_full_logprobs",
            "eager_selected_logprobs",
            "router_ids",
            "router_weights",
        }:
            raise ValueError("H200 capture array inventory is incomplete")
        if not selected_arrays <= set(arrays):
            raise ValueError("requested H200 capture array is missing")
        loaded: dict[str, np.ndarray] = {}
        files: set[str] = set()
        for name, entry in arrays.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ValueError("H200 capture array entry is invalid")
            if entry["file"] in files:
                raise ValueError("H200 capture array files are not unique")
            files.add(entry["file"])
            descriptor = open_regular_at(root_descriptor, entry["file"])
            try:
                if file_sha256_fd(descriptor) != entry.get("sha256"):
                    raise ValueError(
                        f"checksum mismatch for H200 array {entry['file']}"
                    )
                if name in selected_arrays:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    with os.fdopen(os.dup(descriptor), "rb") as stream:
                        loaded[name] = np.load(stream, allow_pickle=False)
                require_child_regular_identity(
                    root_descriptor, entry["file"], descriptor
                )
            finally:
                os.close(descriptor)
        validate_retained_h200_diagnostics_fd(root_descriptor, manifest)
        require_directory_identity(path, root_descriptor)
        yield manifest, manifest_sha256, loaded
        require_directory_identity(path, root_descriptor)
    finally:
        os.close(root_descriptor)


def load_h200_capture_manifest(path: Path) -> tuple[dict[str, Any], str]:
    with open_h200_capture_snapshot(path) as (manifest, manifest_sha256, _arrays):
        return manifest, manifest_sha256


def load_manifest(path: Path, kind: str) -> dict[str, Any]:
    if kind == "glm52_policy_parity_h200_capture":
        return load_h200_capture_manifest(path)[0]
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != schema_version or manifest.get("kind") != kind:
        raise ValueError(f"invalid {kind} manifest in {path}")
    for entry in manifest.get("arrays", {}).values():
        array_path = path / entry["file"]
        if file_sha256(array_path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {array_path}")
    return manifest


def fixture_digest(path: Path) -> str:
    manifest = load_manifest(path, "glm52_policy_parity_fixture")
    return hashlib.sha256(json_bytes(manifest)).hexdigest()


def load_array(root: Path, manifest: dict[str, Any], name: str) -> np.ndarray:
    try:
        entry = manifest["arrays"][name]
    except KeyError as error:
        raise ValueError(f"missing array {name} in {root}") from error
    return np.load(root / entry["file"], allow_pickle=False)


def check_fresh(profile: Profile, fixture: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("fixture profile checksum does not match")
    created = dt.datetime.fromisoformat(manifest["created_utc"])
    if created.tzinfo is None:
        raise ValueError("fixture time has no timezone")
    age = dt.datetime.now(dt.timezone.utc) - created.astimezone(dt.timezone.utc)
    if age.total_seconds() < -300:
        raise ValueError("fixture creation time is in the future")
    if age > dt.timedelta(hours=profile.get("max_fixture_age_hours")):
        raise ValueError(f"fixture is stale: {fixture}")


def expected_router_shape(profile: Profile) -> tuple[int, int, int]:
    return (
        profile.get("context_length"),
        profile.get("hidden_layers"),
        profile.get("experts_per_token"),
    )


def validate_logprobs(profile: Profile, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    expected = profile.get("max_output_length")
    if values.shape != (expected,):
        raise ValueError(f"selected logprobs shape must be {(expected,)}, got {values.shape}")
    if not np.isfinite(values).all() or np.any(values > 0):
        raise ValueError("selected logprobs must be finite and no greater than zero")
    return values


def validate_router_ids(profile: Profile, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != expected_router_shape(profile):
        raise ValueError(f"router IDs shape must be {expected_router_shape(profile)}, got {values.shape}")
    if values.dtype.kind not in "iu":
        raise ValueError("router IDs must use an integer dtype")
    dense_layers = profile.get("dense_layers")
    dense = values[:, :dense_layers, :]
    if dense.size and np.any(dense):
        raise ValueError("dense-layer router IDs must be zero")
    sparse = values[:, dense_layers:, :]
    if sparse.size and (sparse.min() < 0 or sparse.max() >= profile.get("experts")):
        raise ValueError("router ID is outside the model expert range")
    if sparse.size and not np.any(sparse):
        raise ValueError("sparse router IDs are zero-filled")
    if sparse.size and np.any(np.diff(np.sort(sparse, axis=-1), axis=-1) == 0):
        raise ValueError("a sparse token routes to the same expert more than once")
    return np.ascontiguousarray(values.astype(np.uint8, copy=False))


def validate_router_weights(profile: Profile, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != expected_router_shape(profile):
        raise ValueError(
            f"router weights shape must be {expected_router_shape(profile)}, got {values.shape}"
        )
    sparse = values[:, profile.get("dense_layers") :, :]
    if not np.isfinite(sparse).all() or np.any(sparse <= 0):
        raise ValueError("sparse router weights must be finite and positive")
    max_sum_error = float(np.max(np.abs(sparse.sum(axis=-1) - 1.0)))
    limit = profile.get("max_router_weight_sum_error")
    if max_sum_error > limit:
        raise ValueError(f"router weight sum error is {max_sum_error:.6f}, limit is {limit:.6f}")
    return np.ascontiguousarray(values)

def validate_route_output_scales(
    profile: Profile,
    trainer_manifest: dict[str, Any],
    h200_manifest: dict[str, Any],
) -> float:
    scales = []
    for name, manifest in (("trainer", trainer_manifest), ("H200", h200_manifest)):
        if (
            manifest.get("router_weight_provenance")
            != "selected_weights_before_route_output_scale"
        ):
            raise ValueError(f"{name} router weight provenance is invalid")
        scale = manifest.get("route_output_scale")
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name} route output scale is missing or invalid")
        scales.append(float(scale))
    if scales[0] != scales[1]:
        raise ValueError("trainer and H200 route output scales differ")
    expected = profile.get("router_scale")
    if (
        not isinstance(expected, (int, float))
        or not math.isfinite(expected)
        or expected <= 0
    ):
        raise ValueError("profile route output scale is missing or invalid")
    if scales[0] != expected:
        raise ValueError("trace route output scale differs from profile")
    return scales[0]


def record_trace(
    profile: Profile,
    fixture: Path,
    output: Path,
    side: str,
    selected_logprobs_path: Path,
    router_ids_path: Path,
    router_weights_path: Path | None,
    model_revision: str,
    adapter_sha256: str,
    image: str,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    if side not in {"trainer", "h200"}:
        raise ValueError("side must be trainer or h200")
    if len(model_revision) != 40:
        raise ValueError("model revision must be a full 40-character revision")
    if len(adapter_sha256) != 64 or any(c not in "0123456789abcdef" for c in adapter_sha256):
        raise ValueError("adapter checksum must be 64 lowercase hex characters")
    logprobs = validate_logprobs(profile, np.load(selected_logprobs_path, allow_pickle=False))
    router_ids = validate_router_ids(profile, np.load(router_ids_path, allow_pickle=False))
    router_weights = None
    if router_weights_path is not None:
        router_weights = validate_router_weights(
            profile, np.load(router_weights_path, allow_pickle=False)
        )
    if router_weights is None:
        raise ValueError(f"{side} trace must include router weights")
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "selected_logprobs": logprobs,
        "router_ids": router_ids,
    }
    if router_weights is not None:
        arrays["router_weights"] = router_weights
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(output / filename, array, allow_pickle=False)
        entries[name] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_trace",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "side": side,
        "fixture_sha256": fixture_digest(fixture),
        "profile_sha256": file_sha256(profile.path),
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "lora_targets": list(required_targets),
        "quantization": profile.get("quantization"),
        "lora_rank": profile.get("lora_rank"),
        "lora_alpha": profile.get("lora_alpha"),
        "context_length": profile.get("context_length"),
        "route_output_scale": profile.get("router_scale"),
        "router_weight_provenance": "selected_weights_before_route_output_scale",
        "max_output_length": profile.get("max_output_length"),
        "arrays": entries,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean(dtype=np.float64)) if values.size else 0.0


def response_logprob(entry: Any, token_id: int, index: int) -> float:
    if index == 0 and entry is None:
        return 0.0
    if not isinstance(entry, dict) or not entry:
        raise ValueError(f"prompt logprob entry {index} is missing")
    value = None
    for key in (str(token_id), token_id, f"token_id:{token_id}"):
        if key in entry:
            value = entry[key]
            break
    if value is None and len(entry) == 1:
        value = next(iter(entry.values()))
    if isinstance(value, dict):
        value = value.get("logprob")
    elif hasattr(value, "logprob"):
        value = value.logprob
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"prompt logprob entry {index} has no numeric value for token {token_id}")
    result = float(value)
    if not math.isfinite(result) or result > 0:
        raise ValueError(f"prompt logprob entry {index} is invalid")
    return result


def graph_sampling_params(seed: int) -> dict[str, Any]:
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "prompt_logprobs": 0,
        "logprobs": 1,
        "extra_args": {"prime_exact_target_prompt_logprobs": True},
        "routed_experts_prompt_start": 0,
        "seed": seed,
        "skip_special_tokens": False,
    }


def trace_sampling_params(seed: int, trace_session: dict[str, Any]) -> dict[str, Any]:
    trace_session_id = hashlib.sha256(
        json.dumps(
            trace_session, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    result = graph_sampling_params(seed)
    result["extra_args"].update(
        {
            "prime_router_trace": True,
            "prime_router_trace_session": trace_session_id,
            "prime_router_trace_request": secrets.token_hex(16),
        }
    )
    return result


def prompt_logprobs(response: dict[str, Any], token_ids: np.ndarray) -> np.ndarray:
    entries = response.get("prompt_logprobs")
    if not isinstance(entries, list) or len(entries) != token_ids.size:
        count = len(entries) if isinstance(entries, list) else 0
        raise ValueError(
            f"H200 returned {count} prompt logprobs, expected {token_ids.size}"
        )
    return np.asarray(
        [
            response_logprob(entry, int(token_ids[index]), index)
            for index, entry in enumerate(entries)
        ],
        dtype=np.float32,
    )


def inference_path_metrics(
    profile: Profile,
    advantages: np.ndarray,
    graph_logprobs: np.ndarray,
    eager_logprobs: np.ndarray,
) -> dict[str, Any]:
    graph = np.asarray(graph_logprobs, dtype=np.float64)
    eager = np.asarray(eager_logprobs, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    if graph.shape != eager.shape or graph.shape != advantages.shape:
        raise ValueError("graph and eager scoring inputs have different shapes")
    if not np.isfinite(graph).all() or not np.isfinite(eager).all():
        raise ValueError("graph or eager scoring has a nonfinite logprob")
    log_ratio = eager - graph
    with np.errstate(over="ignore", invalid="ignore"):
        mismatch_kl = np.exp(log_ratio) - log_ratio - 1.0
    if not np.isfinite(mismatch_kl).all():
        raise ValueError("graph and eager mismatch KL is not finite")
    probability_delta = np.exp(eager) - np.exp(graph)
    masked = np.where(
        advantages > 0,
        probability_delta > profile.get("dppo_mask_high"),
        probability_delta < -profile.get("dppo_mask_low"),
    )
    masked_fraction = float(masked.mean())
    masked_kl = safe_mean(mismatch_kl[masked])
    unmasked_kl = safe_mean(mismatch_kl[~masked])
    absolute_delta = np.abs(log_ratio)
    checks = {
        "unmasked_mismatch_kl": unmasked_kl
        <= profile.get("max_unmasked_mismatch_kl"),
        "masked_mismatch_kl": masked_kl
        <= profile.get("max_masked_mismatch_kl"),
        "masked_fraction": masked_fraction <= profile.get("max_masked_fraction"),
    }
    return {
        "passed": all(checks.values()),
        "selected_tokens": int(graph.size),
        "mean_abs_logprob_delta": float(absolute_delta.mean()),
        "p95_abs_logprob_delta": float(np.quantile(absolute_delta, 0.95)),
        "max_abs_logprob_delta": float(absolute_delta.max()),
        "unmasked_mismatch_kl": unmasked_kl,
        "masked_mismatch_kl": masked_kl,
        "masked_fraction": masked_fraction,
        "checks": checks,
    }


def request_json(
    url: str,
    *,
    body: dict[str, Any] | None,
    api_key: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {"Accept": "application/json", "Connection": "close"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, separators=(",", ":")).encode()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode(errors="replace")
        raise ValueError(f"H200 request failed with HTTP {error.code}: {detail}") from error
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("H200 response is not JSON") from error
    if not isinstance(result, dict):
        raise ValueError("H200 response is not an object")
    return result


def wait_for_bootstrap_identity(
    base_url: str,
    bootstrap_token: str,
    expected_process_id: int,
    timeout: float,
) -> dict[str, Any]:
    if (
        isinstance(expected_process_id, bool)
        or not isinstance(expected_process_id, int)
        or expected_process_id <= 0
    ):
        raise ValueError("expected H200 API process ID is invalid")
    deadline = time.monotonic() + timeout
    while True:
        try:
            ready = request_json(
                f"{base_url.rstrip('/')}/parity/bootstrap-ready",
                body=None,
                api_key=bootstrap_token,
                timeout=min(timeout, 10),
            )
            break
        except ValueError as error:
            detail = str(error)
            if "HTTP 401" in detail:
                raise ValueError("H200 parity bootstrap identity check failed") from error
            if "HTTP 503" not in detail or time.monotonic() >= deadline:
                raise
            time.sleep(1)
    expected_sha256 = hashlib.sha256(bootstrap_token.encode("ascii")).hexdigest()
    if (
        ready.get("status") != "ready"
        or ready.get("process_id") != expected_process_id
        or re.fullmatch(r"[0-9a-f]{32}", str(ready.get("boot_nonce", ""))) is None
        or ready.get("token_sha256") != expected_sha256
    ):
        raise ValueError("H200 parity bootstrap identity differs from the isolated API")
    return ready


def read_secret_stdin(label: str, minimum: int, maximum: int) -> str:
    if minimum < 1 or maximum < minimum:
        raise ValueError("secret input limits are invalid")
    value = sys.stdin.buffer.readline(maximum + 2)
    if len(value) > maximum + 1 or not value.endswith(b"\n"):
        raise ValueError(f"{label} input is too long or incomplete")
    if sys.stdin.buffer.read(1):
        raise ValueError(f"{label} input contains extra data")
    token = value[:-1].decode("ascii")
    if not minimum <= len(token) <= maximum or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        for character in token
    ):
        raise ValueError(f"{label} input is invalid")
    return token


diagnostic_file_limit = 16 * 1024 * 1024
retained_diagnostic_file_limit = 64
retained_diagnostic_byte_limit = 100_000_000
retained_diagnostic_tokens = 256
retained_diagnostic_members = {
    "values.npy",
    "session_identity.npy",
    "name.npy",
    "source_dtype.npy",
    "fixture_sha256.npy",
    "token_ids_sha256.npy",
    "model_revision.npy",
    "adapter_sha256.npy",
}


def _read_npy_member_header(stream: Any, member_size: int) -> tuple[np.dtype, tuple[int, ...]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported diagnostic NPY version {version}")
    dtype = np.dtype(dtype)
    if dtype.hasobject or not isinstance(shape, tuple) or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ValueError("diagnostic NPY header is invalid")
    expected = int(dtype.itemsize) * math.prod(shape)
    if stream.tell() + expected != member_size:
        raise ValueError("diagnostic NPY member byte count is incorrect")
    return dtype, shape


def validate_diagnostic_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size > diagnostic_file_limit:
        raise ValueError("diagnostic input file is too large or missing")
    if path.suffix == ".npy":
        with path.open("rb") as stream:
            dtype, shape = _read_npy_member_header(stream, path.stat().st_size)
        if (
            dtype != np.dtype("float32")
            or len(shape) != 2
            or shape[1] != 6144
            or not 1 <= shape[0] <= 256
        ):
            raise ValueError("diagnostic NPY must be 1..256x6144 float32")
        return
    if path.suffix != ".npz":
        raise ValueError("diagnostic input must be NPY or NPZ")
    total_uncompressed = 0
    with zipfile.ZipFile(path) as archive:
        if not archive.infolist():
            raise ValueError("diagnostic NPZ is empty")
        names: set[str] = set()
        for info in archive.infolist():
            if info.filename in names:
                raise ValueError("diagnostic NPZ contains duplicate members")
            names.add(info.filename)
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("diagnostic NPZ contains an unsafe member")
            if info.file_size > diagnostic_file_limit:
                raise ValueError("diagnostic NPZ member is too large")
            if info.compress_size == 0 and info.file_size:
                raise ValueError("diagnostic NPZ member has invalid compression metadata")
            if info.compress_size and info.file_size > info.compress_size * 1000:
                raise ValueError("diagnostic NPZ member compression ratio is too high")
            total_uncompressed += info.file_size
            if total_uncompressed > diagnostic_file_limit:
                raise ValueError("diagnostic NPZ is too large when uncompressed")
            with archive.open(info) as stream:
                dtype, shape = _read_npy_member_header(stream, info.file_size)
            if member.name == "values.npy" and (
                dtype != np.dtype("float32")
                or len(shape) != 2
                or not 1 <= shape[0] <= 256
                or shape[1] != 6144
            ):
                raise ValueError("diagnostic values must be 1..256x6144 float32")


def validate_retained_diagnostic_fd(file_descriptor: int) -> None:
    file_status = os.fstat(file_descriptor)
    if (
        not stat.S_ISREG(file_status.st_mode)
        or file_status.st_size > diagnostic_file_limit
    ):
        raise ValueError("retained H200 diagnostic is too large or missing")
    total_uncompressed = 0
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(file_descriptor), "rb") as stream, zipfile.ZipFile(
        stream
    ) as archive:
        if not archive.infolist():
            raise ValueError("retained H200 diagnostic is empty")
        names: set[str] = set()
        for info in archive.infolist():
            if info.filename in names:
                raise ValueError("retained H200 diagnostic has duplicate members")
            names.add(info.filename)
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("retained H200 diagnostic has an unsafe member")
            if info.file_size > diagnostic_file_limit:
                raise ValueError("retained H200 diagnostic member is too large")
            if info.compress_size == 0 and info.file_size:
                raise ValueError("retained H200 diagnostic compression metadata is invalid")
            if info.compress_size and info.file_size > info.compress_size * 1000:
                raise ValueError("retained H200 diagnostic compression ratio is too high")
            total_uncompressed += info.file_size
            if total_uncompressed > diagnostic_file_limit:
                raise ValueError("retained H200 diagnostic is too large when uncompressed")
            with archive.open(info) as stream:
                _read_npy_member_header(stream, info.file_size)
    if names != retained_diagnostic_members:
        raise ValueError("retained H200 diagnostic members are incomplete")


def validate_retained_diagnostic_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        validate_retained_diagnostic_fd(descriptor)
    finally:
        os.close(descriptor)


def h200_diagnostic_binding(result: dict[str, Any]) -> dict[str, str]:
    session = result.get("session")
    diagnostics = result.get("diagnostics")
    results = result.get("results")
    if (
        not isinstance(session, dict)
        or session.get("schema_version") != schema_version
        or session.get("side") != "h200"
        or session.get("dense_layers") != 3
        or session.get("experts") != 256
        or session.get("topk") != 8
        or not isinstance(diagnostics, dict)
        or diagnostics.get("schema_version") != schema_version
        or diagnostics.get("kind") != "glm52_policy_parity_diagnostics"
        or diagnostics.get("side") != "h200"
        or not isinstance(results, dict)
        or results.get("layer") != 3
        or results.get("input_shape") != [retained_diagnostic_tokens, 6144]
    ):
        raise ValueError("H200 diagnostic session is invalid")
    identity = hashlib.sha256(
        json.dumps(session, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    binding = {
        "session_identity": identity,
        "fixture_sha256": session.get("fixture_sha256"),
        "token_ids_sha256": session.get("token_ids_sha256"),
        "model_revision": session.get("model_revision"),
        "adapter_sha256": session.get("adapter_sha256"),
    }
    lengths = {
        "session_identity": 64,
        "fixture_sha256": 64,
        "token_ids_sha256": 64,
        "model_revision": 40,
        "adapter_sha256": 64,
    }
    if any(
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{lengths[name]}}}", value) is None
        for name, value in binding.items()
    ):
        raise ValueError("H200 diagnostic session binding is invalid")
    if (
        diagnostics.get("session_identity") != identity
        or results.get("session_identity") != identity
    ):
        raise ValueError("H200 diagnostic manifest used another session")
    for name in ("logits_sha256", "route_ids_sha256", "route_weights_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(results.get(name, ""))) is None:
            raise ValueError("H200 direct diagnostic result hashes are invalid")
    return binding


def expected_h200_diagnostic_entries() -> dict[str, dict[str, Any]]:
    semantics: list[tuple[str, int, int, str]] = [
        ("embedding", retained_diagnostic_tokens, 6144, "torch.bfloat16")
    ]
    for layer in range(3):
        semantics.extend(
            (
                f"layer{layer}_{suffix}",
                retained_diagnostic_tokens,
                6144,
                "torch.bfloat16",
            )
            for suffix in ("input", "attention_raw", "mlp_raw", "output")
        )
    semantics.extend(
        (
            ("layer0_normalized_input", 1, 6144, "torch.bfloat16"),
            ("layer0_normalized_kv", 1, 512, "torch.bfloat16"),
            ("layer0_o_proj_input_tp0", 1, 2048, "torch.bfloat16"),
            ("layer3_router_input", retained_diagnostic_tokens, 6144, "torch.bfloat16"),
            ("layer3_h200_input", retained_diagnostic_tokens, 6144, "torch.bfloat16"),
            ("layer3_h200_logits", retained_diagnostic_tokens, 256, "torch.float32"),
            ("layer3_h200_scores", retained_diagnostic_tokens, 256, "torch.float32"),
            ("layer3_h200_selection_scores", retained_diagnostic_tokens, 256, "torch.float32"),
            ("layer3_h200_route_ids", retained_diagnostic_tokens, 8, "torch.int32"),
            ("layer3_h200_route_weights", retained_diagnostic_tokens, 8, "torch.float32"),
        )
    )
    return {
        f"{name}-rank-0-pp-0-cp-0": {
            "shape": [tokens, width],
            "source_dtype": source_dtype,
        }
        for name, tokens, width, source_dtype in semantics
    }


legacy_h200_missing_diagnostics = frozenset(
    f"{name}-rank-0-pp-0-cp-0"
    for name in (
        "layer0_normalized_input",
        "layer0_normalized_kv",
        "layer0_o_proj_input_tp0",
    )
)


def accepted_h200_diagnostic_entries(
    names: set[str],
) -> dict[str, dict[str, Any]]:
    current = expected_h200_diagnostic_entries()
    if len(current) != 23 or not legacy_h200_missing_diagnostics < set(current):
        raise RuntimeError("H200 diagnostic plan changed without a schema update")
    if names == set(current):
        return current
    legacy_names = set(current) - legacy_h200_missing_diagnostics
    if len(legacy_names) != 20:
        raise RuntimeError("legacy H200 diagnostic plan is invalid")
    if names == legacy_names:
        return {name: current[name] for name in current if name in legacy_names}
    raise ValueError("retained H200 diagnostic entry set is invalid")


def validate_retained_diagnostic_content_fd(
    file_descriptor: int,
    entry: dict[str, Any],
    binding: dict[str, str],
    result_hashes: dict[str, Any],
) -> None:
    validate_retained_diagnostic_fd(file_descriptor)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(file_descriptor), "rb") as stream, np.load(
        stream, allow_pickle=False
    ) as values:
        if "values" not in values.files:
            raise ValueError("retained H200 diagnostic has no values")
        array = np.ascontiguousarray(values["values"])
        for name in (
            "session_identity",
            "name",
            "source_dtype",
            "fixture_sha256",
            "token_ids_sha256",
            "model_revision",
            "adapter_sha256",
        ):
            metadata = values[name]
            if metadata.shape != () or metadata.dtype.kind != "U":
                raise ValueError("retained H200 diagnostic metadata is invalid")
        stored_name = str(values.get("name", "").item())
        stored_dtype = str(values.get("source_dtype", "").item())
        stored_binding = {
            name: str(values.get(name, "").item()) for name in binding
        }
    if (
        array.dtype != np.dtype("float32")
        or not np.isfinite(array).all()
        or list(array.shape) != entry["shape"]
        or stored_name != entry["name"]
        or stored_dtype != entry["source_dtype"]
        or stored_binding != binding
        or hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        != entry["sha256"]
    ):
        raise ValueError("retained H200 diagnostic content differs from its manifest")
    direct_hash_names = {
        "layer3_h200_logits-rank-0-pp-0-cp-0": "logits_sha256",
        "layer3_h200_route_weights-rank-0-pp-0-cp-0": "route_weights_sha256",
    }
    direct_hash_name = direct_hash_names.get(entry["name"])
    if direct_hash_name is not None and (
        hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        != result_hashes[direct_hash_name]
    ):
        raise ValueError("retained H200 diagnostic differs from direct replay")
    if entry["name"] == "layer3_h200_route_ids-rank-0-pp-0-cp-0":
        if (
            np.any(array != np.floor(array))
            or np.any(array < 0)
            or np.any(array > 255)
        ):
            raise ValueError("retained H200 route IDs are invalid")
        route_ids = np.ascontiguousarray(array.astype(np.int32))
        if (
            hashlib.sha256(memoryview(route_ids).cast("B")).hexdigest()
            != result_hashes["route_ids_sha256"]
        ):
            raise ValueError("retained H200 route IDs differ from direct replay")


def validate_retained_diagnostic_content(
    path: Path,
    entry: dict[str, Any],
    binding: dict[str, str],
    result_hashes: dict[str, Any],
) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        validate_retained_diagnostic_content_fd(
            descriptor, entry, binding, result_hashes
        )
    finally:
        os.close(descriptor)


def validate_retained_h200_diagnostics_fd(
    capture_descriptor: int, manifest: dict[str, Any]
) -> None:
    retained = manifest.get("retained_diagnostics")
    direct = manifest.get("direct_router_diagnostic")
    if not isinstance(retained, dict) or not isinstance(direct, dict):
        raise ValueError("retained H200 diagnostics are missing")
    source = direct.get("diagnostics")
    binding = h200_diagnostic_binding(direct)
    entries = retained.get("entries")
    source_entries = source.get("entries") if isinstance(source, dict) else None
    if (
        retained.get("schema_version") != 1
        or retained.get("kind")
        != "glm52_policy_parity_retained_h200_diagnostics"
        or not isinstance(entries, list)
        or not entries
        or not isinstance(source_entries, list)
        or retained.get("session_identity") != source.get("session_identity")
        or retained.get("side") != source.get("side")
        or retained.get("binding") != binding
        or retained.get("files") != len(entries)
        or len(entries) != len(source_entries)
        or len(entries) > retained_diagnostic_file_limit
    ):
        raise ValueError("retained H200 diagnostic manifest is invalid")
    source_by_name = {
        entry.get("name"): entry for entry in source_entries if isinstance(entry, dict)
    }
    if len(source_by_name) != len(source_entries):
        raise ValueError("H200 diagnostic source names are invalid")
    plan = accepted_h200_diagnostic_entries(set(source_by_name))
    if len(entries) != len(plan):
        raise ValueError("retained H200 diagnostic manifest is invalid")
    total_bytes = 0
    files: set[str] = set()
    source_fields = {"name", "file", "shape", "source_dtype", "sha256", "bytes"}
    diagnostic_descriptor = os.open(
        "diagnostics",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=capture_descriptor,
    )
    try:
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != source_fields | {
                "file_sha256"
            }:
                raise ValueError("retained H200 diagnostic entry is invalid")
            relative = entry.get("file")
            name = entry.get("name")
            file_sha = entry.get("file_sha256")
            if (
                not isinstance(relative, str)
                or re.fullmatch(
                    r"diagnostics/h200-rank-0-pp-0-cp-0-diagnostic-[A-Za-z0-9_-]+\.npz",
                    relative,
                )
                is None
                or relative in files
                or re.fullmatch(r"[0-9a-f]{64}", str(file_sha)) is None
            ):
                raise ValueError("retained H200 diagnostic file identity is invalid")
            files.add(relative)
            original = source_by_name.get(name)
            expected_semantic = plan.get(name)
            expected = {**entry}
            expected.pop("file_sha256", None)
            filename = Path(relative).name
            expected["file"] = filename
            if (
                not isinstance(original, dict)
                or set(original) != source_fields
                or original != expected
                or expected_semantic is None
                or entry.get("shape") != expected_semantic["shape"]
                or entry.get("source_dtype") != expected_semantic["source_dtype"]
            ):
                raise ValueError("retained H200 diagnostic source identity differs")
            size = entry.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("retained H200 diagnostic byte count is invalid")
            file_descriptor = open_regular_at(diagnostic_descriptor, filename)
            try:
                if (
                    os.fstat(file_descriptor).st_size != size
                    or file_sha256_fd(file_descriptor) != file_sha
                ):
                    raise ValueError("retained H200 diagnostic file checksum differs")
                validate_retained_diagnostic_content_fd(
                    file_descriptor, entry, binding, direct["results"]
                )
                require_child_regular_identity(
                    diagnostic_descriptor, filename, file_descriptor
                )
            finally:
                os.close(file_descriptor)
            total_bytes += size
    finally:
        try:
            require_child_directory_identity(
                capture_descriptor, "diagnostics", diagnostic_descriptor
            )
        finally:
            os.close(diagnostic_descriptor)
    if set(source_by_name) != set(plan) or retained.get("bytes") != total_bytes:
        raise ValueError("retained H200 diagnostic total bytes differ")
    if total_bytes > retained_diagnostic_byte_limit:
        raise ValueError("retained H200 diagnostic bytes exceed the fixed limit")


def validate_retained_h200_diagnostics(
    capture_root: Path, manifest: dict[str, Any]
) -> None:
    capture_descriptor = open_directory(capture_root)
    try:
        validate_retained_h200_diagnostics_fd(capture_descriptor, manifest)
        require_directory_identity(capture_root, capture_descriptor)
    finally:
        os.close(capture_descriptor)


def retain_h200_diagnostics(
    result: dict[str, Any],
    output: Path,
    trace_root: Path,
    max_files: int,
    max_bytes: int,
) -> dict[str, Any]:
    diagnostics = result.get("diagnostics")
    binding = h200_diagnostic_binding(result)
    plan = expected_h200_diagnostic_entries()
    if not isinstance(diagnostics, dict):
        raise ValueError("H200 diagnostic response has no diagnostic manifest")
    entries = diagnostics.get("entries")
    if not isinstance(entries, list) or len(entries) != len(plan):
        raise ValueError("H200 diagnostic response has no diagnostic files")
    if (
        isinstance(max_files, bool)
        or not isinstance(max_files, int)
        or max_files != retained_diagnostic_file_limit
        or isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes != retained_diagnostic_byte_limit
    ):
        raise ValueError("H200 diagnostic retention limits are invalid")
    if len(entries) > max_files:
        raise ValueError("H200 diagnostic file count exceeds its configured limit")
    trace_descriptor = open_directory(trace_root)
    try:
        disk_manifest, _disk_manifest_sha256 = read_json_at(
            trace_descriptor, "parity-diagnostics.json"
        )
        if disk_manifest != diagnostics:
            raise ValueError("H200 diagnostic disk and response manifests differ")
        output_fd = open_directory(output)
        try:
            os.mkdir("diagnostics", mode=0o750, dir_fd=output_fd)
            destination_fd = os.open(
                "diagnostics",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=output_fd,
            )
        except BaseException:
            os.close(output_fd)
            raise
    except BaseException:
        os.close(trace_descriptor)
        raise
    retained: list[dict[str, Any]] = []
    names: set[str] = set()
    files: set[str] = set()
    total_bytes = 0
    try:
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "name",
                "file",
                "shape",
                "source_dtype",
                "sha256",
                "bytes",
            }:
                raise ValueError("H200 diagnostic manifest entry is invalid")
            name = entry.get("name")
            filename = entry.get("file")
            size = entry.get("bytes")
            shape = entry.get("shape")
            source_dtype = entry.get("source_dtype")
            raw_sha256 = entry.get("sha256")
            semantic = plan.get(name) if isinstance(name, str) else None
            if (
                semantic is None
                or name in names
                or not isinstance(filename, str)
                or not re.fullmatch(
                    r"h200-rank-0-pp-0-cp-0-diagnostic-[A-Za-z0-9_-]+\.npz",
                    filename,
                )
                or filename in files
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 1
                or shape != semantic["shape"]
                or source_dtype != semantic["source_dtype"]
                or re.fullmatch(r"[0-9a-f]{64}", str(raw_sha256)) is None
            ):
                raise ValueError("H200 diagnostic manifest entry fields are invalid")
            names.add(name)
            files.add(filename)
            total_bytes += size
            if total_bytes > max_bytes:
                raise ValueError("H200 diagnostic bytes exceed their configured limit")

            source_fd = open_regular_at(trace_descriptor, filename)
            target_fd = -1
            try:
                source_stat = os.fstat(source_fd)
                if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size != size:
                    raise ValueError("H200 diagnostic source type or size differs")
                target_fd = os.open(
                    filename,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=destination_fd,
                )
                with os.fdopen(os.dup(source_fd), "rb") as source_stream, os.fdopen(
                    os.dup(target_fd), "wb"
                ) as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, 1024 * 1024)
                    target_stream.flush()
                os.fsync(target_fd)
                target_stat = os.fstat(target_fd)
                if (
                    not stat.S_ISREG(target_stat.st_mode)
                    or target_stat.st_size != size
                ):
                    raise ValueError("retained H200 diagnostic byte count differs")
                validate_retained_diagnostic_content_fd(
                    target_fd, entry, binding, result["results"]
                )
                target_sha256 = file_sha256_fd(target_fd)
                require_child_regular_identity(
                    destination_fd, filename, target_fd
                )
                require_child_regular_identity(
                    trace_descriptor, filename, source_fd
                )
                retained.append(
                    {
                        **entry,
                        "file": f"diagnostics/{filename}",
                        "file_sha256": target_sha256,
                    }
                )
            finally:
                os.close(source_fd)
                if target_fd >= 0:
                    os.close(target_fd)
        if names != set(plan):
            raise ValueError("H200 diagnostic files do not match the exact plan")
        os.fsync(destination_fd)
        os.fsync(output_fd)
        require_child_directory_identity(output_fd, "diagnostics", destination_fd)
        require_directory_identity(output, output_fd)
        require_directory_identity(trace_root, trace_descriptor)
    finally:
        os.close(destination_fd)
        os.close(output_fd)
        os.close(trace_descriptor)
    return {
        "schema_version": 1,
        "kind": "glm52_policy_parity_retained_h200_diagnostics",
        "session_identity": diagnostics.get("session_identity"),
        "side": diagnostics.get("side"),
        "binding": binding,
        "entries": retained,
        "files": len(retained),
        "bytes": total_bytes,
    }


def h200_diagnostic_retention_settings() -> tuple[Path, int, int]:
    root_text = os.environ.get("PRIME_PARITY_ROUTER_TRACE_DIR")
    qualification_text = os.environ.get("PRIME_PARITY_QUALIFICATION_ROOT")
    if not root_text or not qualification_text:
        raise ValueError("H200 diagnostic retention paths are missing")
    root = Path(root_text)
    qualification = Path(qualification_text)
    if (
        root.is_symlink()
        or qualification.is_symlink()
        or root.resolve(strict=True) != root
        or qualification.resolve(strict=True) != qualification
        or root.parent.parent != qualification
        or re.fullmatch(r"\.diagnostic-work-[0-9a-f]{16}", root.parent.name) is None
        or root.name != "h200-router-trace"
    ):
        raise ValueError("H200 diagnostic retention path is outside the isolated run")
    limits: list[int] = []
    for name in (
        "PRIME_PARITY_DIAGNOSTIC_MAX_FILES",
        "PRIME_PARITY_DIAGNOSTIC_MAX_BYTES",
    ):
        value = os.environ.get(name, "")
        if re.fullmatch(r"[1-9][0-9]*", value) is None:
            raise ValueError(f"{name} is missing or invalid")
        limits.append(int(value))
    if limits != [retained_diagnostic_file_limit, retained_diagnostic_byte_limit]:
        raise ValueError("H200 diagnostic retention limits differ from qualification")
    return root, limits[0], limits[1]


def validate_diagnostic_source(name: str, layer: int) -> None:
    if not re.fullmatch(
        rf"layer{layer}_router_input-rank-[0-9]+-pp-[0-9]+-cp-0", name
    ):
        raise ValueError("direct router replay requires the CP0 layer router input")


def run_h200_router_diagnostic(
    input_path: Path,
    session_path: Path,
    base_url: str,
    layer: int,
    diagnostic_token: str,
    timeout: float,
) -> dict[str, Any]:
    """Send one bounded trainer activation to the H200 router worker."""
    validate_diagnostic_file(input_path)
    with np.load(input_path, allow_pickle=False) as values:
        if not {"values", "session_identity", "name"}.issubset(values.files):
            raise ValueError("diagnostic input must contain values, name, and session identity")
        input_array = np.ascontiguousarray(values["values"])
        session_identity = str(values["session_identity"].item())
        source_name = str(values["name"].item())
        source_dtype = str(values.get("source_dtype", ""))
        provenance = {
            name: str(values[name].item())
            for name in (
                "fixture_sha256",
                "token_ids_sha256",
                "model_revision",
                "adapter_sha256",
            )
        }
    if (
        input_array.dtype != np.float32
        or input_array.ndim != 2
        or input_array.shape[0] < 1
        or input_array.shape[0] > 256
        or input_array.shape[1] != 6144
    ):
        raise ValueError("diagnostic input must be at most 256x6144 float32 values")
    validate_diagnostic_source(source_name, layer)
    session = json.loads(session_path.read_text())
    if session.get("side") != "h200":
        raise ValueError("diagnostic session must be an H200 session")
    encoded = {
        "data": base64.b64encode(memoryview(input_array).cast("B")).decode("ascii"),
        "dtype": input_array.dtype.name,
        "shape": list(input_array.shape),
    }
    target_session_identity = hashlib.sha256(
        json.dumps(session, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "session_identity": target_session_identity,
        "source_session_identity": session_identity,
        **{f"source_{name}": value for name, value in provenance.items()},
        "request_identity": secrets.token_hex(16),
        "layer": layer,
        "source_dtype": source_dtype,
        "router_input_sha256": hashlib.sha256(memoryview(input_array).cast("B")).hexdigest(),
        "router_input": encoded,
    }
    return request_json(
        f"{base_url.rstrip('/').removesuffix('/v1')}/parity/router-diagnostic",
        body=payload,
        api_key=diagnostic_token,
        timeout=timeout,
    )


def capture_h200(
    profile: Profile,
    fixture: Path,
    output: Path,
    base_url: str,
    control_url: str,
    model: str,
    model_revision: str,
    adapter_sha256: str,
    image: str,
    api_key_env: str | None,
    bootstrap_token: str,
    expected_api_process_id: int,
    diagnostic_input: Path | None,
    timeout: float,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    base = base_url.rstrip("/").removesuffix("/v1")
    control = control_url.rstrip("/").removesuffix("/v1")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and api_key is None:
        raise ValueError(f"environment variable {api_key_env} is not set")
    if len(bootstrap_token) != 64:
        raise ValueError("bootstrap token must have 384-bit encoded capacity")
    bootstrap_ready = wait_for_bootstrap_identity(
        control, bootstrap_token, expected_api_process_id, min(timeout, 120)
    )
    cards = request_json(
        f"{base}/v1/models", body=None, api_key=api_key, timeout=min(timeout, 120)
    ).get("data", [])
    card = next((item for item in cards if isinstance(item, dict) and item.get("id") == model), None)
    if card is None:
        raise ValueError(f"H200 does not serve model {model!r}")
    if card.get("max_model_len") != profile.get("h200_scoring_context_length"):
        raise ValueError(
            "H200 direct scoring needs max_model_len 32769 so it can prefill all "
            "32768 fixture tokens and return one output token"
        )
    token_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    token_ids_sha256 = hashlib.sha256(memoryview(np.ascontiguousarray(token_ids)).cast("B")).hexdigest()
    trace_session = {
        "schema_version": schema_version,
        "side": "h200",
        "fixture_sha256": fixture_digest(fixture),
        "token_ids_sha256": token_ids_sha256,
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "expected_tokens": profile.get("context_length"),
        "hidden_layers": profile.get("hidden_layers"),
        "dense_layers": profile.get("dense_layers"),
        "topk": profile.get("experts_per_token"),
        "experts": profile.get("experts"),
        "route_output_scale": profile.get("router_scale"),
        "weight_sum_tolerance": profile.get("max_router_weight_sum_error"),
    }
    seed = int(fixture_manifest["seed"] & 0x7FFFFFFF)
    common_body = {
        "model": model,
        "token_ids": token_ids.tolist(),
    }
    graph_body = {
        **common_body,
        "cache_salt": f"{fixture_digest(fixture)}:graph",
        "sampling_params": graph_sampling_params(seed),
    }
    graph_response = request_json(
        f"{base}/inference/v1/generate",
        body=graph_body,
        api_key=api_key,
        timeout=timeout,
    )
    graph_full_logprobs = prompt_logprobs(graph_response, token_ids)
    reset = request_json(
        f"{control}/parity/router-trace/reset",
        body=trace_session,
        api_key=bootstrap_token,
        timeout=min(timeout, 120),
    )
    if reset != trace_session and not (
        isinstance(reset, dict)
        and reset.get("session") == trace_session
        and isinstance(reset.get("diagnostic_token"), str)
    ):
        raise ValueError("H200 router trace reset returned different session metadata")
    eager_sampling_params = trace_sampling_params(seed, trace_session)
    eager_body = {
        **common_body,
        "cache_salt": f"{fixture_digest(fixture)}:eager",
        "sampling_params": eager_sampling_params,
    }
    eager_response = request_json(
        f"{base}/inference/v1/generate",
        body=eager_body,
        api_key=api_key,
        timeout=timeout,
    )
    eager_full_logprobs = prompt_logprobs(eager_response, token_ids)
    choices = eager_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("H200 response needs exactly one choice")
    encoded = choices[0].get("routed_experts")
    if not isinstance(encoded, dict):
        raise ValueError("H200 did not return routed expert IDs")
    try:
        router_ids = np.frombuffer(
            base64.b64decode(encoded["data"], validate=True),
            dtype=np.dtype(encoded["dtype"]),
        ).reshape(encoded["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("H200 routed expert payload is invalid") from error
    if int(encoded.get("start", 0)) != 0:
        raise ValueError("H200 routed expert payload does not start at token zero")
    context = profile.get("context_length")
    if router_ids.ndim != 3 or router_ids.shape[0] < context:
        raise ValueError(f"H200 routed expert payload is too short: {router_ids.shape}")
    router_ids = validate_router_ids(profile, router_ids[:context])
    route_trace = request_json(
        f"{control}/parity/router-trace",
        body=None,
        api_key=api_key,
        timeout=timeout,
    )
    if route_trace.get("session") != trace_session:
        raise ValueError("H200 router trace used different session metadata")
    runtime_marker = route_trace.get("runtime_marker")
    expected_trace_request = eager_sampling_params["extra_args"][
        "prime_router_trace_request"
    ]
    if not isinstance(runtime_marker, dict) or runtime_marker.get(
        "request_identity"
    ) != expected_trace_request:
        raise ValueError("H200 router trace has no matching runtime marker")
    try:
        traced_ids_entry = route_trace["router_ids"]
        traced_weights_entry = route_trace["router_weights"]
        traced_ids = np.frombuffer(
            base64.b64decode(traced_ids_entry["data"], validate=True),
            dtype=np.dtype(traced_ids_entry["dtype"]),
        ).reshape(traced_ids_entry["shape"])
        observed_weights = np.frombuffer(
            base64.b64decode(traced_weights_entry["data"], validate=True),
            dtype=np.dtype(traced_weights_entry["dtype"]),
        ).reshape(traced_weights_entry["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("H200 router weight trace payload is invalid") from error
    traced_ids = validate_router_ids(profile, traced_ids)
    require_exact_router_ids(router_ids, traced_ids)
    router_weights = validate_router_weights(profile, observed_weights)
    selected_indices = load_array(fixture, fixture_manifest, "selected_indices")
    graph_selected_logprobs = validate_logprobs(
        profile, graph_full_logprobs[selected_indices]
    )
    eager_selected_logprobs = validate_logprobs(
        profile, eager_full_logprobs[selected_indices]
    )
    advantages = load_array(fixture, fixture_manifest, "advantages")
    graph_eager_guard = inference_path_metrics(
        profile,
        advantages,
        graph_selected_logprobs,
        eager_selected_logprobs,
    )
    if not graph_eager_guard["passed"]:
        raise ValueError(
            "graph and eager scoring failed the existing parity limits: "
            + json.dumps(graph_eager_guard, sort_keys=True)
        )
    direct_router_diagnostic = None
    if diagnostic_input is not None:
        diagnostic_token = reset.get("diagnostic_token") if isinstance(reset, dict) else None
        if not isinstance(diagnostic_token, str) or not diagnostic_token:
            raise ValueError("H200 reset did not issue a diagnostic token")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as session_file:
            json.dump(trace_session, session_file)
            session_file.flush()
            direct_router_diagnostic = run_h200_router_diagnostic(
                diagnostic_input,
                Path(session_file.name),
                control_url,
                trace_session["dense_layers"],
                diagnostic_token,
                min(timeout, 300.0),
            )
    output.mkdir(parents=True, exist_ok=False)
    retained_diagnostics = None
    if diagnostic_input is not None:
        if not isinstance(direct_router_diagnostic, dict):
            raise ValueError("H200 direct router diagnostic response is missing")
        trace_root, diagnostic_max_files, diagnostic_max_bytes = (
            h200_diagnostic_retention_settings()
        )
        retained_diagnostics = retain_h200_diagnostics(
            direct_router_diagnostic,
            output,
            trace_root,
            diagnostic_max_files,
            diagnostic_max_bytes,
        )
    arrays = {
        "full_logprobs": graph_full_logprobs,
        "selected_logprobs": graph_selected_logprobs,
        "eager_full_logprobs": eager_full_logprobs,
        "eager_selected_logprobs": eager_selected_logprobs,
        "router_ids": router_ids,
        "router_weights": router_weights,
    }
    entries: dict[str, Any] = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        np.save(output / filename, array, allow_pickle=False)
        entries[name] = {
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": file_sha256(output / filename),
        }
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_h200_capture",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "fixture_sha256": fixture_digest(fixture),
        "profile_sha256": file_sha256(profile.path),
        "base_url": base_url,
        "control_url": control_url,
        "model": model,
        "graph_request_id": graph_response.get("request_id"),
        "eager_request_id": eager_response.get("request_id"),
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "bootstrap_identity": bootstrap_ready,
        "lora_targets": list(required_targets),
        "route_output_scale": profile.get("router_scale"),
        "router_weight_provenance": "selected_weights_before_route_output_scale",
        "graph_eager_guard": graph_eager_guard,
        "router_runtime_marker": runtime_marker,
        "direct_router_diagnostic": direct_router_diagnostic,
        "retained_diagnostics": retained_diagnostics,
        "phases": {
            "graph": {
                "fixture_sha256": fixture_digest(fixture),
                "model": model,
                "model_revision": model_revision,
                "adapter_sha256": adapter_sha256,
                "mode": "normal_cuda_graph",
            },
            "eager_trace": {
                "fixture_sha256": fixture_digest(fixture),
                "model": model,
                "model_revision": model_revision,
                "adapter_sha256": adapter_sha256,
                "mode": "force_eager_skip_compiled",
            },
        },
        "sources": {
            "trainer_reference_logprobs": "normal_cuda_graph_request",
            "h200_trace_logprobs": "normal_cuda_graph_request",
            "router_ids": "isolated_eager_trace_request",
            "router_weights": "isolated_eager_trace_request",
        },
        "arrays": entries,
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return load_manifest(output, "glm52_policy_parity_h200_capture")


def validate_h200_capture_sources(manifest: dict[str, Any]) -> None:
    expected_sources = {
        "trainer_reference_logprobs": "normal_cuda_graph_request",
        "h200_trace_logprobs": "normal_cuda_graph_request",
        "router_ids": "isolated_eager_trace_request",
        "router_weights": "isolated_eager_trace_request",
    }
    if manifest.get("sources") != expected_sources:
        raise ValueError("H200 capture output sources are missing or mixed")
    guard = manifest.get("graph_eager_guard")
    if not isinstance(guard, dict) or guard.get("passed") is not True:
        raise ValueError("H200 graph and eager scoring did not pass parity limits")
    runtime_marker = manifest.get("router_runtime_marker")
    if not isinstance(runtime_marker, dict) or not runtime_marker.get("layer_calls"):
        raise ValueError("H200 router runtime marker is missing")
    id_sources = runtime_marker.get("id_sources")
    if not isinstance(id_sources, dict) or not id_sources:
        raise ValueError("H200 router runtime marker ID sources are missing")
    if any(source != "modular_actual" for source in id_sources.values()):
        raise ValueError("H200 router IDs are not from the executed MoE path")
    scale = manifest.get("route_output_scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("H200 route output scale is missing or invalid")
    if (
        manifest.get("router_weight_provenance")
        != "selected_weights_before_route_output_scale"
    ):
        raise ValueError("H200 router weight provenance is invalid")
    scales = runtime_marker.get("route_output_scales")
    if not isinstance(scales, dict) or not scales:
        raise ValueError("H200 route output scales are missing")
    if any(value != scale for value in scales.values()):
        raise ValueError("H200 route output scale differs from capture")
    phases = manifest.get("phases")
    if not isinstance(phases, dict):
        raise ValueError("H200 capture phases are missing")
    graph = phases.get("graph")
    eager = phases.get("eager_trace")
    if not isinstance(graph, dict) or not isinstance(eager, dict):
        raise ValueError("H200 capture phase identity is missing")
    for field in ("fixture_sha256", "model", "model_revision", "adapter_sha256"):
        if graph.get(field) != eager.get(field):
            raise ValueError(f"H200 graph and eager phase {field} values differ")
    if graph.get("mode") != "normal_cuda_graph":
        raise ValueError("H200 graph phase mode is invalid")
    if eager.get("mode") != "force_eager_skip_compiled":
        raise ValueError("H200 eager trace phase mode is invalid")
    if manifest.get("direct_router_diagnostic") is not None and not isinstance(
        manifest.get("retained_diagnostics"), dict
    ):
        raise ValueError("H200 semantic diagnostics were not retained")


def require_exact_router_ids(
    executed_ids: np.ndarray, observed_ids: np.ndarray
) -> None:
    if executed_ids.shape != observed_ids.shape:
        raise ValueError("executed and observed router ID shapes differ")
    if not np.array_equal(executed_ids, observed_ids):
        raise ValueError("executed and observed router IDs differ")


def select_first_trainer_router_call(router_trace: Path, output: Path) -> None:
    """Select one identical fixture pass from a multi-microbatch trainer trace."""
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(router_trace / "session.json", output / "session.json")
    chunks = sorted(router_trace.glob("trainer-rank-*-pp-*-cp-*-layer-*-call-*.npz"))
    first_calls = [path for path in chunks if path.name.endswith("-call-0.npz")]
    if not first_calls:
        raise ValueError("trainer router trace has no first microbatch")
    for path in chunks:
        first_path = output / (path.name.rsplit("-call-", 1)[0] + "-call-0.npz")
        if path.name.endswith("-call-0.npz"):
            shutil.copy2(path, first_path)
            continue
        reference_path = router_trace / first_path.name
        if not reference_path.is_file():
            raise ValueError(f"trainer router trace is missing {reference_path.name}")
        with np.load(reference_path, allow_pickle=False) as reference, np.load(
            path, allow_pickle=False
        ) as candidate:
            for name in ("ids", "weights"):
                if not np.array_equal(reference[name], candidate[name]):
                    raise ValueError(
                        f"trainer router trace microbatch differs for {path.name}"
                    )


def collect_trainer_router_trace(
    profile: Profile, trainer_config: Path, sources: list[Path], output: Path
) -> dict[str, Any]:
    """Merge per-host PP/CP traces into one immutable capture directory."""
    if not sources:
        raise ValueError("at least one trainer router trace source is required")
    if output.exists():
        raise ValueError(f"trainer router trace output already exists: {output}")
    with trainer_config.open("rb") as stream:
        trainer_values = tomllib.load(stream)
    model_values = trainer_values.get("model", {})
    fake_values = trainer_values.get("data", {}).get("fake", {})
    pp_size = model_values.get("pp")
    cp_size = model_values.get("cp")
    batch_size = fake_values.get("batch_size")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (pp_size, cp_size, batch_size)):
        raise ValueError("trainer TOML must define positive model.pp, model.cp, and data.fake.batch_size")
    source_sessions = []
    for source in sources:
        if not source.is_dir():
            raise ValueError(f"trainer router trace source is missing: {source}")
        session_path = source / "session.json"
        if not session_path.is_file():
            raise ValueError(f"trainer router trace session is missing: {session_path}")
        source_sessions.append(json.loads(session_path.read_text()))
    if any(session != source_sessions[0] for session in source_sessions[1:]):
        raise ValueError("trainer router trace sessions differ between hosts")
    source_session = source_sessions[0]
    if source_session.get("side") != "trainer":
        raise ValueError("trainer router trace session side is not trainer")

    output.mkdir(parents=True)
    (output / "session.json").write_text(json.dumps(source_session, sort_keys=True) + "\n")
    provenances: list[dict[str, Any]] = []
    copied: set[str] = set()
    for source in sources:
        runtime_paths = sorted(source.glob("trainer-runtime-pp-*-host-*.json"))
        if len(runtime_paths) != 1:
            raise ValueError(f"trainer source must contain one host provenance record: {source}")
        record = json.loads(runtime_paths[0].read_text())
        if record.get("host") != source.name:
            raise ValueError(
                f"trainer provenance host does not match its bound source: {record.get('host')} != {source.name}"
            )
        provenances.append(record)
        for chunk in sorted(source.glob("trainer-rank-*-pp-*-cp-*-layer-*-call-*.npz")):
            name = chunk.name
            if name in copied:
                raise ValueError(f"duplicate trainer router chunk: {name}")
            copied.add(name)
            shutil.copy2(chunk, output / name)
    if not copied:
        raise ValueError("trainer router trace contains no route chunks")
    if len(provenances) != pp_size:
        raise ValueError(f"expected {pp_size} PP host provenance records, found {len(provenances)}")
    expected_layers = set(range(profile.get("dense_layers"), profile.get("hidden_layers")))
    pp_ranks = {record.get("pp_rank") for record in provenances}
    if pp_ranks != set(range(pp_size)):
        raise ValueError("trainer provenance does not cover each PP rank exactly once")
    for record in provenances:
        image_digest = record.get("image_digest")
        if not isinstance(image_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", image_digest
        ):
            raise ValueError("trainer provenance has an invalid image digest")
    module_records = {json.dumps(record.get("modules"), sort_keys=True) for record in provenances}
    if len(module_records) != 1:
        raise ValueError("trainer model module hashes differ between PP hosts")
    ownership: dict[int, int] = {}
    for record in provenances:
        pp_rank = record.get("pp_rank")
        if record.get("cp_rank") != 0 or record.get("global_rank") != pp_rank * cp_size:
            raise ValueError("trainer provenance PP/CP/global rank mapping is invalid")
        if record.get("pp_size") != pp_size or record.get("cp_size") != cp_size:
            raise ValueError("trainer provenance PP/CP sizes differ from trainer TOML")
        if record.get("batch_size") != batch_size:
            raise ValueError("trainer provenance batch size differs from trainer TOML")
        owned = set(record.get("owned_layers", ()))
        if not owned or not owned <= expected_layers:
            raise ValueError("trainer provenance has invalid owned layers")
        for layer in owned:
            if layer in ownership:
                raise ValueError(f"layer {layer} is owned by multiple PP stages")
            ownership[layer] = pp_rank
    if set(ownership) != expected_layers:
        raise ValueError("trainer PP layer ownership is incomplete")
    expected_pp = pp_size
    expected_cp = cp_size
    coverage: set[tuple[int, int, int]] = set()
    rank_by_mesh: dict[tuple[int, int], int] = {}
    for name in copied:
        match = re.fullmatch(
            r"trainer-rank-(\d+)-pp-(\d+)-cp-(\d+)-layer-(\d+)-call-(\d+)\.npz",
            name,
        )
        if match is None:
            raise ValueError(f"unexpected trainer router chunk name: {name}")
        rank, pp, cp, layer, call = (int(value) for value in match.groups())
        if pp >= expected_pp or cp >= expected_cp or layer not in expected_layers:
            raise ValueError(f"trainer router chunk is outside expected coverage: {name}")
        if ownership[layer] != pp:
            raise ValueError(f"trainer router chunk has the wrong PP owner: {name}")
        mesh = (pp, cp)
        previous_rank = rank_by_mesh.setdefault(mesh, rank)
        if previous_rank != rank:
            raise ValueError(f"trainer router rank changed for PP/CP mesh {mesh}")
        if call == 0:
            coverage.add((pp, cp, layer))
    expected = {
        (ownership[layer], cp, layer)
        for layer in expected_layers
        for cp in range(expected_cp)
    }
    if coverage != expected:
        missing = sorted(expected - coverage)
        extra = sorted(coverage - expected)
        raise ValueError(f"trainer router coverage differs; missing={missing[:3]} extra={extra[:3]}")
    expected_ranks = {pp * expected_cp + cp for pp in range(expected_pp) for cp in range(expected_cp)}
    if set(rank_by_mesh.values()) != expected_ranks:
        raise ValueError("trainer router ranks do not cover the expected PP/CP world")
    expected_trace_calls = (profile.get("hidden_layers") - profile.get("dense_layers")) * batch_size
    for record in provenances:
        limits = record.get("trace_limits")
        if not isinstance(limits, dict) or limits.get("max_calls") != expected_trace_calls or limits.get("max_files") != expected_trace_calls:
            raise ValueError("trainer trace limits do not match model layers and batch size")
    for record in provenances:
        filename = (
            f"trainer-runtime-pp-{record['pp_rank']}-host-{record['host']}.json"
        )
        (output / filename).write_text(json.dumps(record, sort_keys=True) + "\n")
    return {
        "kind": "glm52_policy_parity_trainer_trace_collection",
        "sources": [str(source) for source in sources],
        "chunks": len(copied),
        "pipeline_stages": expected_pp,
        "context_parallel": expected_cp,
        "runtime_provenance": provenances,
    }


def load_trainer_provenance(router_trace: Path) -> dict[str, Any]:
    paths = sorted(router_trace.glob("trainer-runtime-pp-*-host-*.json"))
    if not paths:
        raise ValueError("trainer host provenance records are missing")
    records = [json.loads(path.read_text()) for path in paths]
    for record in records:
        image_digest = record.get("image_digest")
        if not isinstance(image_digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", image_digest
        ):
            raise ValueError("trainer image digest is missing or invalid")
    modules = [record.get("modules") for record in records]
    if not modules or any(value != modules[0] for value in modules[1:]):
        raise ValueError("trainer model module hashes differ between host records")
    for record in records:
        if record.get("kind") != "glm52_policy_parity_trainer_runtime":
            raise ValueError("trainer runtime provenance kind is invalid")
    image_digest = records[0]["image_digest"]
    if not isinstance(modules[0], dict) or not modules[0]:
        raise ValueError("trainer model module hashes are missing")
    for name, value in modules[0].items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError("trainer model module hash record is invalid")
        if value.get("location") not in {"mounted", "unmounted"} or not re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("sha256", ""))
        ):
            raise ValueError(f"trainer model module hash is invalid: {name}")
    locations = {value["location"] for value in modules[0].values()}
    if locations != {"mounted", "unmounted"}:
        raise ValueError("mounted and unmounted trainer model module hashes are incomplete")
    return {**records[0], "host_records": records}


def trainer_session_fixture_sha256(
    fixture: Path, fixture_manifest: dict[str, Any], expected_token_hash: str
) -> str:
    source = fixture_manifest.get("source")
    if source is not None:
        semantic_hashes = (
            source.get("fixture_semantic_sha256")
            if isinstance(source, dict)
            else None
        )
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("fixture_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", source["fixture_sha256"]) is None
            or not isinstance(semantic_hashes, dict)
            or semantic_hashes.get("token_ids") != expected_token_hash
        ):
            raise ValueError("refreshed fixture trainer source provenance is invalid")
    return fixture_digest(fixture)


def capture_trainer(
    profile: Profile,
    fixture: Path,
    token_exports: Path,
    router_trace: Path,
    output: Path,
    model_revision: str,
    adapter_sha256: str,
    image: str,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    token_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    expected_token_hash = hashlib.sha256(
        memoryview(np.ascontiguousarray(token_ids)).cast("B")
    ).hexdigest()
    source_root = Path(__file__).resolve().parents[1] / "source" / "src"
    sys.path.insert(0, str(source_root))
    from prime_rl.parity_router_trace import assemble_chunks, read_session

    session = read_session(router_trace)
    session_fixture_sha256 = trainer_session_fixture_sha256(
        fixture, fixture_manifest, expected_token_hash
    )
    expected = {
        "side": "trainer",
        "fixture_sha256": session_fixture_sha256,
        "token_ids_sha256": expected_token_hash,
        "model_revision": model_revision,
        "adapter_sha256": adapter_sha256,
        "image": image,
        "expected_tokens": profile.get("context_length"),
        "hidden_layers": profile.get("hidden_layers"),
        "dense_layers": profile.get("dense_layers"),
        "topk": profile.get("experts_per_token"),
        "experts": profile.get("experts"),
        "route_output_scale": profile.get("router_scale"),
        "weight_sum_tolerance": profile.get("max_router_weight_sum_error"),
    }
    for name, value in expected.items():
        if getattr(session, name) != value:
            raise ValueError(f"trainer router trace {name} differs from the parity request")
    provenance = load_trainer_provenance(router_trace)
    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-router-") as temporary:
        first_router_trace = Path(temporary) / "first"
        select_first_trainer_router_call(router_trace, first_router_trace)
        router_ids, router_weights = assemble_chunks(first_router_trace, session)
    matching_records: list[dict[str, Any]] = []
    for path in sorted(token_exports.rglob("rank_*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("schema_version") != schema_version:
                raise ValueError(f"token export schema differs in {path}")
            if record.get("token_ids") == token_ids.tolist():
                matching_records.append(record)
    if not matching_records:
        raise ValueError(
            "expected at least one exact fixture token export, found none"
        )
    first_record = matching_records[0]
    for duplicate in matching_records[1:]:
        for name in (
            "token_ids",
            "position_ids",
            "loss_mask",
            "advantages",
            "inference_logprobs",
            "trainer_logprobs",
            "sequence_lengths",
        ):
            if duplicate.get(name) != first_record.get(name):
                raise ValueError(
                    "duplicate fixture token exports differ in "
                    f"{name}; parity cannot select one microbatch"
                )
    full_logprobs = np.asarray(matching_records[0].get("trainer_logprobs"), dtype=np.float32)
    if full_logprobs.shape != token_ids.shape:
        raise ValueError(
            f"trainer token logprobs shape is {full_logprobs.shape}, expected {token_ids.shape}"
        )
    selected_indices = load_array(fixture, fixture_manifest, "selected_indices")
    selected_logprobs = validate_logprobs(profile, full_logprobs[selected_indices])
    with tempfile.TemporaryDirectory(prefix="glm52-policy-parity-trainer-") as temporary:
        temp = Path(temporary)
        np.save(temp / "selected_logprobs.npy", selected_logprobs, allow_pickle=False)
        np.save(temp / "router_ids.npy", router_ids, allow_pickle=False)
        np.save(temp / "router_weights.npy", router_weights, allow_pickle=False)
        manifest = record_trace(
            profile,
            fixture,
            output,
            "trainer",
            temp / "selected_logprobs.npy",
            temp / "router_ids.npy",
            temp / "router_weights.npy",
            model_revision,
            adapter_sha256,
            image,
        )
    manifest["trainer_image_digest"] = provenance["image_digest"]
    manifest["trainer_image_digests"] = {
        record["host"]: record["image_digest"]
        for record in provenance["host_records"]
    }
    manifest["trainer_model_modules"] = provenance["modules"]
    manifest["trainer_host_provenance"] = provenance["host_records"]
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def make_trainer_reference(
    profile: Profile,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    with open_h200_capture_snapshot(
        h200_capture, {"full_logprobs", "router_ids"}
    ) as (capture_manifest, capture_manifest_sha256, capture_arrays):
        return _make_trainer_reference_from_snapshot(
            profile,
            fixture,
            output,
            fixture_manifest,
            capture_manifest,
            capture_manifest_sha256,
            capture_arrays,
        )


def _make_trainer_reference_from_snapshot(
    profile: Profile,
    fixture: Path,
    output: Path,
    fixture_manifest: dict[str, Any],
    capture_manifest: dict[str, Any],
    capture_manifest_sha256: str,
    capture_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    validate_h200_capture_sources(capture_manifest)
    if capture_manifest.get("fixture_sha256") != fixture_digest(fixture):
        raise ValueError("H200 capture used another fixture")
    if capture_manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("H200 capture used another profile")
    input_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    inference_logprobs = capture_arrays["full_logprobs"].astype(np.float32)
    if inference_logprobs.shape != input_ids.shape:
        raise ValueError("H200 full logprobs do not align with fixture tokens")
    router_ids = validate_router_ids(
        profile, capture_arrays["router_ids"]
    )
    router_bytes = memoryview(router_ids).cast("B")
    selected = load_array(fixture, fixture_manifest, "selected_indices").astype(np.int64)
    selected_advantages = load_array(fixture, fixture_manifest, "advantages").astype(
        np.float32
    )
    loss_mask = np.zeros(input_ids.shape, dtype=np.bool_)
    loss_mask[selected] = True
    loss_mask[0] = False
    advantages = np.zeros(input_ids.shape, dtype=np.float32)
    advantages[selected] = selected_advantages
    payload = {
        "schema_version": schema_version,
        "fixture_sha256": fixture_digest(fixture),
        "profile_sha256": file_sha256(profile.path),
        "model_revision": capture_manifest.get("model_revision"),
        "adapter_sha256": capture_manifest.get("adapter_sha256"),
        "image": capture_manifest.get("image"),
        "h200_capture_manifest_sha256": capture_manifest_sha256,
        "capture_router_ids_sha256": capture_manifest["arrays"]["router_ids"][
            "sha256"
        ],
        "capture_full_logprobs_sha256": capture_manifest["arrays"][
            "full_logprobs"
        ]["sha256"],
        "input_ids": input_ids.tolist(),
        "inference_logprobs": inference_logprobs.tolist(),
        "sequence_lengths": [profile.get("context_length")],
        "loss_mask": loss_mask.tolist(),
        "advantages": advantages.tolist(),
        "routed_experts": {
            "schema_version": schema_version,
            "encoding": "base64",
            "dtype": "uint8",
            "shape": list(router_ids.shape),
            "dense_layers": profile.get("dense_layers"),
            "experts": profile.get("experts"),
            "sha256": hashlib.sha256(router_bytes).hexdigest(),
            "data": base64.b64encode(router_bytes).decode("ascii"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(json_bytes(payload))
    return {
        "output": str(output),
        "fixture_sha256": payload["fixture_sha256"],
        "tokens": int(input_ids.size),
        "selected_tokens": int(loss_mask.sum()),
        "routed_experts_shape": list(router_ids.shape),
        "routed_experts_sha256": payload["routed_experts"]["sha256"],
        "capture_router_ids_sha256": payload["capture_router_ids_sha256"],
        "capture_full_logprobs_sha256": payload[
            "capture_full_logprobs_sha256"
        ],
        "model_revision": payload["model_revision"],
        "adapter_sha256": payload["adapter_sha256"],
    }


def prepare_trainer_router_trace(
    profile: Profile,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    capture_manifest = load_manifest(
        h200_capture, "glm52_policy_parity_h200_capture"
    )
    validate_h200_capture_sources(capture_manifest)
    if capture_manifest.get("fixture_sha256") != fixture_digest(fixture):
        raise ValueError("H200 capture used another fixture")
    if capture_manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("H200 capture used another profile")
    token_ids = load_array(fixture, fixture_manifest, "token_ids").astype(np.int64)
    token_ids_sha256 = hashlib.sha256(
        memoryview(np.ascontiguousarray(token_ids)).cast("B")
    ).hexdigest()
    source_root = Path(__file__).resolve().parents[1] / "source" / "src"
    sys.path.insert(0, str(source_root))
    from prime_rl.parity_router_trace import TraceSession

    trace_session = TraceSession(
        side="trainer",
        fixture_sha256=fixture_digest(fixture),
        token_ids_sha256=token_ids_sha256,
        model_revision=capture_manifest["model_revision"],
        adapter_sha256=capture_manifest["adapter_sha256"],
        image=profile.get("trainer_image"),
        expected_tokens=profile.get("context_length"),
        hidden_layers=profile.get("hidden_layers"),
        dense_layers=profile.get("dense_layers"),
        topk=profile.get("experts_per_token"),
        experts=profile.get("experts"),
        route_output_scale=profile.get("router_scale"),
        weight_sum_tolerance=profile.get("max_router_weight_sum_error"),
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "session.json").write_bytes(json_bytes(trace_session.as_dict()))
    return trace_session.as_dict()


def stage_trainer_artifacts(
    profile: Profile,
    fixture: Path,
    h200_capture: Path,
    output: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    capture_manifest, capture_manifest_sha256 = load_h200_capture_manifest(
        h200_capture
    )
    validate_h200_capture_sources(capture_manifest)
    if capture_manifest.get("fixture_sha256") != fixture_digest(fixture):
        raise ValueError("H200 capture used another fixture")
    if capture_manifest.get("profile_sha256") != file_sha256(profile.path):
        raise ValueError("H200 capture used another profile")
    if output.is_symlink():
        raise ValueError("trainer stage output must not be a symlink")
    output.mkdir(parents=False, exist_ok=True)
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("trainer stage output must be an empty directory")
    output.chmod(0o750)
    reference = output / "trainer-reference.json"
    router_trace = output / "trainer-router-trace"
    reference_result = make_trainer_reference(
        profile, fixture, h200_capture, reference
    )
    trace_result = prepare_trainer_router_trace(
        profile, fixture, h200_capture, router_trace
    )
    reference.chmod(0o644)
    router_trace.chmod(0o755)
    (router_trace / "session.json").chmod(0o644)
    final_capture_manifest, final_capture_manifest_sha256 = (
        load_h200_capture_manifest(h200_capture)
    )
    if (
        final_capture_manifest_sha256 != capture_manifest_sha256
        or final_capture_manifest != capture_manifest
    ):
        raise ValueError("H200 capture changed while building trainer stage")
    reference_payload = json.loads(reference.read_text())
    fixture_sha256 = fixture_digest(fixture)
    profile_sha256 = file_sha256(profile.path)
    common_identity = {
        "fixture_sha256": fixture_sha256,
        "model_revision": capture_manifest["model_revision"],
        "adapter_sha256": capture_manifest["adapter_sha256"],
    }
    for name, expected in common_identity.items():
        if reference_payload.get(name) != expected or trace_result.get(name) != expected:
            raise ValueError(f"trainer stage {name} identity differs")
    if reference_payload.get("profile_sha256") != profile_sha256:
        raise ValueError("trainer stage reference profile identity differs")
    if reference_payload.get("image") != capture_manifest["image"]:
        raise ValueError("trainer stage H200 image identity differs")
    if (
        reference_payload.get("h200_capture_manifest_sha256")
        != capture_manifest_sha256
        or reference_payload.get("capture_router_ids_sha256")
        != capture_manifest["arrays"]["router_ids"]["sha256"]
        or reference_payload.get("capture_full_logprobs_sha256")
        != capture_manifest["arrays"]["full_logprobs"]["sha256"]
    ):
        raise ValueError("trainer stage H200 array identity differs")
    if trace_result.get("image") != profile.get("trainer_image"):
        raise ValueError("trainer stage trainer image identity differs")
    manifest = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_trainer_stage",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha256": file_sha256(Path(__file__).resolve()),
        "profile_sha256": profile_sha256,
        "fixture_sha256": fixture_sha256,
        "h200_capture_manifest_sha256": capture_manifest_sha256,
        "route_output_scale": capture_manifest["route_output_scale"],
        "router_weight_provenance": capture_manifest[
            "router_weight_provenance"
        ],
        "reference_schema": {
            "schema_version": schema_version,
            "encoding": "base64",
            "dtype": "uint8",
            "shape": reference_result["routed_experts_shape"],
            "dense_layers": profile.get("dense_layers"),
            "experts": profile.get("experts"),
            "sha256": capture_manifest["arrays"]["router_ids"]["sha256"],
            "routed_experts_sha256": reference_result[
                "routed_experts_sha256"
            ],
            "full_logprobs_sha256": capture_manifest["arrays"][
                "full_logprobs"
            ]["sha256"],
        },
        "identity": {
            **common_identity,
            "profile_sha256": profile_sha256,
            "h200_image": capture_manifest["image"],
            "trainer_image": profile.get("trainer_image"),
            "token_ids_sha256": trace_result["token_ids_sha256"],
        },
        "files": {
            "trainer-reference.json": file_sha256(reference),
            "trainer-router-trace/session.json": file_sha256(
                router_trace / "session.json"
            ),
        },
    }
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    (output / "manifest.json").chmod(0o644)
    return manifest


def validate_stage_paths(
    operations_path: Path,
    fixture: Path,
    h200_capture: Path,
    output: Path,
    host_output: Path | None = None,
) -> None:
    with operations_path.open("rb") as stream:
        operations = tomllib.load(stream)
    expected = operations.get("parity", {}).get("glm52")
    if not isinstance(expected, dict):
        raise ValueError("operations config has no GLM parity paths")
    actual_sources = {
        "fixture": str(fixture),
        "h200_capture": str(h200_capture),
    }
    expected_sources = {
        "fixture": expected.get("fixture"),
        "h200_capture": expected.get("h200_capture"),
    }
    stage = PurePosixPath(str(expected.get("trainer_stage")))
    candidate = PurePosixPath(str(host_output or output))
    candidate_is_exact = (
        candidate.parent == stage.parent / ".stage-candidates"
        and re.fullmatch(
            r"trainer-stage-candidate-[0-9a-f]{16,64}", candidate.name
        )
        is not None
    )
    if (
        actual_sources != expected_sources
        or (candidate != stage and not candidate_is_exact)
        or (host_output is not None and PurePosixPath(str(output)) != PurePosixPath("/stage"))
    ):
        raise ValueError(
            "GLM parity staging paths differ from operations config: "
            f"{actual_sources | {'trainer_stage': str(candidate)}}"
        )


def expected_trainer_image_digest(operations_path: Path) -> str | None:
    if not operations_path.is_file():
        raise ValueError(f"operations config is missing: {operations_path}")
    with operations_path.open("rb") as stream:
        operations = tomllib.load(stream)
    containers = operations.get("containers", {})
    profile = operations.get("profiles", {}).get("glm52-policy-parity", {})
    expected = containers.get("trainer_image_digest") or profile.get(
        "trainer_image_digest"
    )
    if expected is None:
        return None
    if not isinstance(expected, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", expected
    ):
        raise ValueError("configured trainer image digest is invalid")
    return expected


def compare(
    profile: Profile,
    fixture: Path,
    trainer: Path,
    h200: Path,
    output: Path | None,
    operations_path: Path,
) -> dict[str, Any]:
    fixture_manifest = load_manifest(fixture, "glm52_policy_parity_fixture")
    check_fresh(profile, fixture, fixture_manifest)
    trainer_manifest = load_manifest(trainer, "glm52_policy_parity_trace")
    h200_manifest = load_manifest(h200, "glm52_policy_parity_h200_capture")
    validate_h200_capture_sources(h200_manifest)
    if trainer_manifest.get("side") != "trainer":
        raise ValueError("trainer trace side is not trainer")
    expected_fixture = fixture_digest(fixture)
    for name, manifest in (("trainer", trainer_manifest), ("h200", h200_manifest)):
        if manifest.get("fixture_sha256") != expected_fixture:
            raise ValueError(f"{name} trace used another fixture")
        if manifest.get("profile_sha256") != file_sha256(profile.path):
            raise ValueError(f"{name} trace used another profile")
        if tuple(manifest.get("lora_targets", ())) != required_targets:
            raise ValueError(f"{name} trace has the wrong LoRA targets")
    for field in ("model_revision", "adapter_sha256"):
        if trainer_manifest.get(field) != h200_manifest.get(field):
            raise ValueError(f"trace {field} values differ")
    validate_route_output_scales(profile, trainer_manifest, h200_manifest)
    trainer_image_digest = trainer_manifest.get("trainer_image_digest")
    if not isinstance(trainer_image_digest, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", trainer_image_digest
    ):
        raise ValueError("trainer manifest image digest is missing or invalid")
    expected_image_digest = expected_trainer_image_digest(operations_path)
    trainer_model_modules = trainer_manifest.get("trainer_model_modules")
    if not isinstance(trainer_model_modules, dict) or not trainer_model_modules:
        raise ValueError("trainer manifest model module hashes are missing")
    if not any(
        isinstance(value, dict) and value.get("location") == "mounted"
        for value in trainer_model_modules.values()
    ) or not any(
        isinstance(value, dict) and value.get("location") == "unmounted"
        for value in trainer_model_modules.values()
    ):
        raise ValueError("trainer manifest mounted and unmounted module hashes are incomplete")
    trainer_host_provenance = trainer_manifest.get("trainer_host_provenance")
    if not isinstance(trainer_host_provenance, list) or len(trainer_host_provenance) != 8:
        raise ValueError("trainer manifest must contain eight host provenance records")
    trainer_image_digests = trainer_manifest.get("trainer_image_digests")
    if not isinstance(trainer_image_digests, dict) or len(trainer_image_digests) != len(
        trainer_host_provenance
    ):
        raise ValueError("trainer manifest must contain one image digest per host")
    for record in trainer_host_provenance:
        host = record.get("host") if isinstance(record, dict) else None
        digest = trainer_image_digests.get(host) if isinstance(host, str) else None
        if not isinstance(digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", digest
        ) or digest != record.get("image_digest"):
            raise ValueError("trainer manifest host image digest records do not match")
    if expected_image_digest is not None and any(
        digest != expected_image_digest for digest in trainer_image_digests.values()
    ):
        raise ValueError("a trainer host image digest differs from operations config")

    trainer_logprobs = validate_logprobs(
        profile, load_array(trainer, trainer_manifest, "selected_logprobs")
    ).astype(np.float64)
    inference_logprobs = validate_logprobs(
        profile, load_array(h200, h200_manifest, "selected_logprobs")
    ).astype(np.float64)
    trainer_ids = validate_router_ids(profile, load_array(trainer, trainer_manifest, "router_ids"))
    h200_ids = validate_router_ids(profile, load_array(h200, h200_manifest, "router_ids"))
    trainer_weights = validate_router_weights(
        profile, load_array(trainer, trainer_manifest, "router_weights")
    )
    h200_weights = validate_router_weights(
        profile, load_array(h200, h200_manifest, "router_weights")
    )
    dense_layers = profile.get("dense_layers")
    trainer_sparse_ids = trainer_ids[:, dense_layers:, :]
    h200_sparse_ids = h200_ids[:, dense_layers:, :]
    route_exact_fraction = float(np.mean(trainer_sparse_ids == h200_sparse_ids))
    route_exact = bool(np.array_equal(trainer_sparse_ids, h200_sparse_ids))

    advantages = load_array(fixture, fixture_manifest, "advantages").astype(np.float64)
    log_ratio = trainer_logprobs - inference_logprobs
    with np.errstate(over="ignore", invalid="ignore"):
        importance_ratio = np.exp(log_ratio)
        mismatch_kl = importance_ratio - log_ratio - 1.0
    if not np.isfinite(mismatch_kl).all():
        raise ValueError("mismatch KL is not finite")
    probability_delta = np.exp(trainer_logprobs) - np.exp(inference_logprobs)
    masked = np.where(
        advantages > 0,
        probability_delta > profile.get("dppo_mask_high"),
        probability_delta < -profile.get("dppo_mask_low"),
    )
    masked_fraction = float(masked.mean())
    masked_kl = safe_mean(mismatch_kl[masked])
    unmasked_kl = safe_mean(mismatch_kl[~masked])
    delta = trainer_logprobs - inference_logprobs
    trainer_sparse_weights = trainer_weights[:, dense_layers:, :]
    h200_sparse_weights = h200_weights[:, dense_layers:, :]
    trainer_router_weight_sum_max_error = float(
        np.max(np.abs(trainer_sparse_weights.sum(axis=-1) - 1.0))
    )
    h200_router_weight_sum_max_error = float(
        np.max(np.abs(h200_sparse_weights.sum(axis=-1) - 1.0))
    )
    router_weight_abs_delta = np.abs(
        trainer_sparse_weights.astype(np.float64) - h200_sparse_weights.astype(np.float64)
    )
    router_weight_max_abs_delta = float(np.max(router_weight_abs_delta))
    checks = {
        "unmasked_mismatch_kl": unmasked_kl <= profile.get("max_unmasked_mismatch_kl"),
        "masked_mismatch_kl": masked_kl <= profile.get("max_masked_mismatch_kl"),
        "masked_fraction": masked_fraction <= profile.get("max_masked_fraction"),
        "router_ids_exact": route_exact,
        "trainer_router_weight_sums": trainer_router_weight_sum_max_error
        <= profile.get("max_router_weight_sum_error"),
        "h200_router_weight_sums": h200_router_weight_sum_max_error
        <= profile.get("max_router_weight_sum_error"),
    }
    provisional_checks = {
        "router_weight_delta": router_weight_max_abs_delta
        <= profile.get("max_router_weight_abs_delta")
    }
    report = {
        "schema_version": schema_version,
        "kind": "glm52_policy_parity_report",
        "passed": all(checks.values()),
        "fixture_sha256": expected_fixture,
        "profile_sha256": file_sha256(profile.path),
        "model_revision": trainer_manifest["model_revision"],
        "adapter_sha256": trainer_manifest["adapter_sha256"],
        "trainer_image_digest": trainer_image_digest,
        "trainer_image_digests": trainer_image_digests,
        "trainer_model_modules": trainer_model_modules,
        "trainer_host_provenance": trainer_host_provenance,
        "inference_logprob_source": "normal_cuda_graph_request",
        "inference_router_source": "isolated_eager_trace_request",
        "graph_eager_guard": h200_manifest["graph_eager_guard"],
        "selected_tokens": int(trainer_logprobs.size),
        "unmasked_mismatch_kl": unmasked_kl,
        "masked_mismatch_kl": masked_kl,
        "masked_fraction": masked_fraction,
        "mean_abs_logprob_delta": float(np.mean(np.abs(delta))),
        "p95_abs_logprob_delta": float(np.quantile(np.abs(delta), 0.95)),
        "max_abs_logprob_delta": float(np.max(np.abs(delta))),
        "router_id_exact_fraction": route_exact_fraction,
        "trainer_router_weight_sum_max_error": trainer_router_weight_sum_max_error,
        "h200_router_weight_sum_max_error": h200_router_weight_sum_max_error,
        "router_weight_mean_abs_delta": float(np.mean(router_weight_abs_delta)),
        "router_weight_p99_abs_delta": float(np.quantile(router_weight_abs_delta, 0.99)),
        "router_weight_max_abs_delta": router_weight_max_abs_delta,
        "thresholds": {
            "max_unmasked_mismatch_kl": profile.get("max_unmasked_mismatch_kl"),
            "max_masked_mismatch_kl": profile.get("max_masked_mismatch_kl"),
            "max_masked_fraction": profile.get("max_masked_fraction"),
            "max_router_weight_abs_delta": profile.get("max_router_weight_abs_delta"),
            "max_router_weight_sum_error": profile.get("max_router_weight_sum_error"),
        },
        "checks": checks,
        "provisional_checks": provisional_checks,
        "router_weight_delta_status": profile.get("router_weight_delta_status"),
    }
    if output is not None:
        output.write_bytes(json_bytes(report))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/qualification/glm52-policy-parity.toml"),
    )
    commands = result.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--seed", type=int)
    fixture.add_argument("--created-utc")
    refresh = commands.add_parser("refresh-fixture")
    refresh.add_argument("--source-fixture", type=Path, required=True)
    refresh.add_argument("--source-session", type=Path, required=True)
    refresh.add_argument("--token-export", type=Path)
    refresh.add_argument("--compact-source", type=Path)
    refresh.add_argument("--compact-manifest", type=Path)
    refresh.add_argument("--diagnostic-input", type=Path, required=True)
    refresh.add_argument("--diagnostic-output", type=Path, required=True)
    refresh.add_argument("--expected-rank", type=int, required=True)
    refresh.add_argument("--created-utc")
    refresh.add_argument("output", type=Path)
    h200 = commands.add_parser("capture-h200")
    h200.add_argument("--fixture", type=Path, required=True)
    h200.add_argument("--base-url", required=True)
    h200.add_argument("--control-url", required=True)
    h200.add_argument("--model", required=True)
    h200.add_argument("--model-revision", required=True)
    h200.add_argument("--adapter-sha256", required=True)
    h200.add_argument("--image", required=True)
    h200.add_argument("--api-key-env")
    h200.add_argument("--expected-api-process-id", type=int, required=True)
    h200.add_argument("--diagnostic-input", type=Path)
    h200.add_argument("--timeout", type=float, default=1800.0)
    h200.add_argument("output", type=Path)
    validate_h200 = commands.add_parser("validate-h200-capture")
    validate_h200.add_argument("capture", type=Path)
    diagnostic = commands.add_parser("diagnose-h200")
    diagnostic.add_argument("--input", type=Path, required=True)
    diagnostic.add_argument("--session", type=Path, required=True)
    diagnostic.add_argument("--base-url", required=True)
    diagnostic.add_argument("--layer", type=int, required=True)
    diagnostic.add_argument("--timeout", type=float, default=300.0)
    trainer = commands.add_parser("capture-trainer")
    trainer.add_argument("--fixture", type=Path, required=True)
    trainer.add_argument("--token-exports", type=Path, required=True)
    trainer.add_argument("--router-trace", type=Path, required=True)
    trainer.add_argument("--model-revision", required=True)
    trainer.add_argument("--adapter-sha256", required=True)
    trainer.add_argument("--image", required=True)
    trainer.add_argument("output", type=Path)
    collect = commands.add_parser("collect-trainer")
    collect.add_argument("--trainer-config", type=Path, required=True)
    collect.add_argument("--source", type=Path, action="append", required=True)
    collect.add_argument("output", type=Path)
    reference = commands.add_parser("make-trainer-reference")
    reference.add_argument("--fixture", type=Path, required=True)
    reference.add_argument("--h200-capture", type=Path, required=True)
    reference.add_argument("output", type=Path)
    trace = commands.add_parser("prepare-trainer-trace")
    trace.add_argument("--fixture", type=Path, required=True)
    trace.add_argument("--h200-capture", type=Path, required=True)
    trace.add_argument("output", type=Path)
    stage = commands.add_parser("stage-trainer")
    stage.add_argument(
        "--operations",
        type=Path,
        default=Path("configs/operations.toml"),
    )
    stage.add_argument("--fixture", type=Path, required=True)
    stage.add_argument("--h200-capture", type=Path, required=True)
    stage.add_argument("--host-output", type=Path)
    stage.add_argument("output", type=Path)
    record = commands.add_parser("record")
    record.add_argument("--fixture", type=Path, required=True)
    record.add_argument("--side", choices=("trainer", "h200"), required=True)
    record.add_argument("--selected-logprobs", type=Path, required=True)
    record.add_argument("--router-ids", type=Path, required=True)
    record.add_argument("--router-weights", type=Path)
    record.add_argument("--model-revision", required=True)
    record.add_argument("--adapter-sha256", required=True)
    record.add_argument("--image", required=True)
    record.add_argument("output", type=Path)
    check = commands.add_parser("compare")
    check.add_argument("--fixture", type=Path, required=True)
    check.add_argument("--trainer", type=Path, required=True)
    check.add_argument("--h200", type=Path, required=True)
    check.add_argument(
        "--operations", type=Path, default=Path("configs/operations.toml")
    )
    check.add_argument("--output", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        profile = load_profile(args.profile)
        if args.command == "fixture":
            result = make_fixture(profile, args.output, args.seed, args.created_utc)
        elif args.command == "refresh-fixture":
            result = refresh_fixture_from_trainer_export(
                profile,
                args.source_fixture,
                args.source_session,
                args.token_export,
                args.diagnostic_input,
                args.output,
                args.diagnostic_output,
                args.expected_rank,
                args.created_utc,
                args.compact_source,
                args.compact_manifest,
            )
        elif args.command == "capture-h200":
            result = capture_h200(
                profile,
                args.fixture,
                args.output,
                args.base_url,
                args.control_url,
                args.model,
                args.model_revision,
                args.adapter_sha256,
                args.image,
                args.api_key_env,
                read_secret_stdin("bootstrap token", 64, 64),
                args.expected_api_process_id,
                args.diagnostic_input,
                args.timeout,
            )
        elif args.command == "validate-h200-capture":
            manifest, manifest_sha256 = load_h200_capture_manifest(args.capture)
            validate_h200_capture_sources(manifest)
            result = {
                "manifest_sha256": manifest_sha256,
                "retained_files": manifest["retained_diagnostics"]["files"],
                "retained_bytes": manifest["retained_diagnostics"]["bytes"],
            }
        elif args.command == "diagnose-h200":
            result = run_h200_router_diagnostic(
                args.input,
                args.session,
                args.base_url,
                args.layer,
                read_secret_stdin("diagnostic token", 32, 256),
                args.timeout,
            )
        elif args.command == "capture-trainer":
            result = capture_trainer(
                profile,
                args.fixture,
                args.token_exports,
                args.router_trace,
                args.output,
                args.model_revision,
                args.adapter_sha256,
                args.image,
            )
        elif args.command == "collect-trainer":
            result = collect_trainer_router_trace(
                profile, args.trainer_config, args.source, args.output
            )
        elif args.command == "make-trainer-reference":
            result = make_trainer_reference(
                profile,
                args.fixture,
                args.h200_capture,
                args.output,
            )
        elif args.command == "prepare-trainer-trace":
            result = prepare_trainer_router_trace(
                profile,
                args.fixture,
                args.h200_capture,
                args.output,
            )
        elif args.command == "stage-trainer":
            validate_stage_paths(
                args.operations,
                args.fixture,
                args.h200_capture,
                args.output,
                args.host_output,
            )
            result = stage_trainer_artifacts(
                profile,
                args.fixture,
                args.h200_capture,
                args.output,
            )
        elif args.command == "record":
            result = record_trace(
                profile,
                args.fixture,
                args.output,
                args.side,
                args.selected_logprobs,
                args.router_ids,
                args.router_weights,
                args.model_revision,
                args.adapter_sha256,
                args.image,
            )
        else:
            result = compare(
                profile,
                args.fixture,
                args.trainer,
                args.h200,
                args.output,
                args.operations,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(args.command == "compare" and not result["passed"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
