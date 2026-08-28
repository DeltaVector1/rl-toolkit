#!/usr/bin/env python3
"""Fail closed when a GLM policy parity fixture is incomplete or changed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED = {
    "advantages": (
        "advantages.npy",
        65_664,
        "adc6a77593c349662403cf2d736afc9e27fbab6f3f6df580f4483bf3269a0d65",
    ),
    "positions": (
        "positions.npy",
        262_272,
        "aa1454e30b61a4e79ad1b4e8a8f142e11935b8133b8f2561b1d093c9d825d71e",
    ),
    "selected_indices": (
        "selected_indices.npy",
        131_200,
        "364f56378cf2ff70ef90ed0227a016d701794db8930c94e7551d8edd85830087",
    ),
    "token_ids": (
        "token_ids.npy",
        262_272,
        "4a50cbca403ccef528d58bf84d1026288c1bd386eb703b055542b4f397a5823a",
    ),
}
EXPECTED_PROFILE_SHA256 = "acb2ecd32fff986fe96cbf962fa31549adbed70fab582f6d0ae520ccccb37759"
EXPECTED_MANIFEST_SHA256 = "8753dfd3cd62c7e3b849fc9bbfff1bac9cbd4dcf0a1cce4772d2aebf1612b65a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    args = parser.parse_args()
    root = args.fixture
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    if not root.is_dir():
        errors.append(f"fixture directory is missing: {root}")
    if not manifest_path.is_file():
        errors.append(f"manifest is missing: {manifest_path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"ERROR: cannot read manifest: {error}", file=sys.stderr)
        return 2
    if manifest.get("schema_version") != 1:
        errors.append("manifest schema_version is not 1")
    if manifest.get("kind") != "glm52_policy_parity_fixture":
        errors.append("manifest kind is not glm52_policy_parity_fixture")
    if manifest.get("profile_sha256") != EXPECTED_PROFILE_SHA256:
        errors.append("manifest profile checksum differs")
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        errors.append("manifest checksum differs")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(EXPECTED):
        errors.append(f"manifest arrays must be exactly {sorted(EXPECTED)}")
        arrays = {}
    for key, (name, expected_size, expected_sha256) in EXPECTED.items():
        path = root / name
        if not path.is_file():
            errors.append(f"required fixture file is missing or not regular: {path}")
            continue
        if path.stat().st_size != expected_size:
            errors.append(
                f"fixture size differs for {name}: {path.stat().st_size} != {expected_size}"
            )
        entry = arrays.get(key)
        if not isinstance(entry, dict):
            errors.append(f"manifest entry is missing for {name}")
            continue
        if entry.get("file") != name:
            errors.append(f"manifest filename differs for {name}")
        actual = sha256(path)
        if actual != expected_sha256:
            errors.append(f"fixed fixture checksum differs for {name}")
        if entry.get("sha256") != actual:
            errors.append(f"fixture checksum differs for {name}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"fixture preflight passed: {root}")
    for key in sorted(EXPECTED):
        name, expected_size, _expected_sha256 = EXPECTED[key]
        print(f"{name}: {expected_size} bytes sha256={sha256(root / name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
