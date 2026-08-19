#!/usr/bin/env python3
"""Test byte-exact sparse checkpoint delta creation and reconstruction."""

import json
import tempfile
from pathlib import Path

from prime_rl.utils.checkpoint_delta import (
    DELTA_MANIFEST,
    apply_xor_delta_file,
    build_xor_delta_checkpoint,
    sha256_file,
)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = root / "step_1"
        target = root / "step_2"
        restored = root / "restored.safetensors"
        base.mkdir()
        target.mkdir()

        filename = "model-00001-of-00001.safetensors"
        index = {"weight_map": {"weight": filename}}
        for directory in (base, target):
            (directory / "model.safetensors.index.json").write_text(json.dumps(index))

        original = bytearray((index * 17 + 3) % 256 for index in range(32 * 1024 * 1024))
        updated = bytearray(original)
        for offset in range(0, len(updated), 257):
            updated[offset] ^= 0x5A
        (base / filename).write_bytes(original)
        (target / filename).write_bytes(updated)

        manifest_path = build_xor_delta_checkpoint(
            base,
            target,
            base_url="http://checkpoint/step_1",
            target_url="http://checkpoint/step_2",
        )
        assert manifest_path == target / DELTA_MANIFEST
        manifest = json.loads(manifest_path.read_text())
        spec = manifest["files"][0]
        delta_path = target / spec["delta"]
        assert delta_path.stat().st_size < len(updated) // 10

        apply_xor_delta_file(
            base / filename,
            delta_path,
            restored,
            expected_size=spec["size"],
            expected_sha256=spec["sha256"],
        )
        assert restored.read_bytes() == updated
        assert sha256_file(restored) == spec["sha256"]

    print("checkpoint delta: PASS")


if __name__ == "__main__":
    main()
