#!/usr/bin/env python3
import tempfile
from pathlib import Path

from prime_rl.inference.vllm.worker.filesystem import prune_full_checkpoint_cache


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        keep = root / "new-policy"
        for name in ("old-policy-1", "old-policy-2", keep.name, f"{keep.name}.partial"):
            (root / name).mkdir()
        (root / "new-policy.lock").touch()

        removed = prune_full_checkpoint_cache(root, keep)

        assert {path.name for path in removed} == {"old-policy-1", "old-policy-2"}
        assert {path.name for path in root.iterdir()} == {
            keep.name,
            f"{keep.name}.partial",
            "new-policy.lock",
        }

    print("full-checkpoint cache retention: PASS")


if __name__ == "__main__":
    main()
