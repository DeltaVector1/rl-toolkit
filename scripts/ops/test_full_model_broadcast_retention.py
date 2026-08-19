#!/usr/bin/env python3
import tempfile
from pathlib import Path

from prime_rl.trainer.rl.broadcast.filesystem import prepare_full_model_broadcast_dir


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("step_1", "step_2", "step_25", "step_26", "step_27", "not_a_step"):
            (root / name).mkdir()

        removed = prepare_full_model_broadcast_dir(root, save_step=27)

        assert {path.name for path in removed} == {"step_1", "step_2", "step_25"}
        assert {path.name for path in root.iterdir()} == {"step_26", "step_27", "not_a_step"}

    print("full-model broadcast retention: PASS")


if __name__ == "__main__":
    main()
