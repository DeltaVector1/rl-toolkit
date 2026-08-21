#!/usr/bin/env python3
"""Measure concurrent OpenAI completion throughput on a Muse rollout proxy."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="/data/muse-glimmer/model")
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def one(index: int) -> tuple[int, float]:
        body = json.dumps(
            {
                "model": args.model,
                "prompt": (
                    "Write a coherent technical paragraph about distributed systems. "
                    f"Use request identifier {index}."
                ),
                "temperature": 1.0,
                "max_tokens": args.tokens,
                "min_tokens": args.tokens,
            }
        ).encode()
        request = urllib.request.Request(
            args.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        latency = time.monotonic() - started
        return int(result.get("usage", {}).get("completion_tokens", 0)), latency

    started = time.monotonic()
    results: list[tuple[int, float]] = []
    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, index) for index in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:  # The error count is part of the benchmark.
                errors.append(f"{type(error).__name__}: {error}")
    elapsed = time.monotonic() - started
    output_tokens = sum(tokens for tokens, _ in results)
    latencies = sorted(latency for _, latency in results)
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0.0
    report = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "succeeded": len(results),
        "failed": len(errors),
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "p95_latency_seconds": p95,
        "first_errors": errors[:3],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
