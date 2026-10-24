#!/usr/bin/env python3

import re
import sys

from tqdm import tqdm


batch_pattern = re.compile(r"Train batch\s+(\d+)/(\d+)")
telemetry_pattern = re.compile(
    r"PIPELINE_TELEMETRY.*?\baccepted_buffer=(\d+)\s+accepted_target=(\d+)"
)
success_pattern = re.compile(r"SUCCESS\s+Step\s+(\d+)")


def _parse_progress(line: str) -> tuple[int, int] | None:
    for pattern in (telemetry_pattern, batch_pattern):
        match = pattern.search(line)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def main() -> None:
    bar = None
    previous = 0
    collecting_step = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    try:
        for line in sys.stdin:
            success = success_pattern.search(line)
            if success:
                if bar:
                    bar.update(max(0, bar.total - bar.n))
                    bar.close()
                    bar = None
                print(f"SUCCESS Step {success.group(1)}", flush=True)
                collecting_step = int(success.group(1)) + 1
                previous = 0
                continue

            progress = _parse_progress(line)
            if progress is None:
                continue
            current, total = progress
            if bar is None or bar.total != total or current < previous:
                if bar:
                    bar.close()
                bar = tqdm(
                    total=total,
                    initial=min(current, total),
                    desc=f"Step {collecting_step} rollouts",
                    unit="rollout",
                    dynamic_ncols=True,
                    file=sys.stdout,
                )
            else:
                bar.update(max(0, min(current, total) - bar.n))
            previous = current
    except KeyboardInterrupt:
        pass
    finally:
        if bar:
            bar.close()


if __name__ == "__main__":
    main()
