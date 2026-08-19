#!/usr/bin/env python3
"""Render orchestrator rollout log lines as one tqdm progress bar."""

from __future__ import annotations

import re
import sys

from tqdm import tqdm


BATCH_RE = re.compile(r"Train batch\s+(\d+)/(\d+)")
SUCCESS_RE = re.compile(r"SUCCESS\s+Step\s+(\d+)")


def main() -> None:
    bar: tqdm | None = None
    last_current = 0
    collecting_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    def new_bar(current: int, total: int) -> tqdm:
        return tqdm(
            total=total,
            initial=min(current, total),
            desc=f"Step {collecting_step} rollouts",
            unit="rollout",
            dynamic_ncols=True,
            mininterval=0.1,
            file=sys.stdout,
        )

    try:
        for line in sys.stdin:
            success = SUCCESS_RE.search(line)
            if success is not None:
                if bar is not None:
                    bar.update(max(0, bar.total - bar.n))
                    bar.close()
                    bar = None
                print(f"SUCCESS Step {success.group(1)}", flush=True)
                collecting_step = int(success.group(1)) + 1
                last_current = 0
                continue

            batch = BATCH_RE.search(line)
            if batch is None:
                continue

            current, total = map(int, batch.groups())
            if bar is None:
                bar = new_bar(current, total)
            elif bar.total != total or current < last_current:
                bar.close()
                bar = new_bar(current, total)
            else:
                bar.update(max(0, min(current, total) - bar.n))
            last_current = current
    except KeyboardInterrupt:
        pass
    finally:
        if bar is not None:
            bar.close()


if __name__ == "__main__":
    main()
