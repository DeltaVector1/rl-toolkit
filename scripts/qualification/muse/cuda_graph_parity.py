#!/usr/bin/env python3
"""Compare eager and CUDA-graph Muse serving with identical requests."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-url", default="http://127.0.0.1:8007/v1/completions")
    parser.add_argument("--graph-url", default="http://127.0.0.1:8008/v1/completions")
    parser.add_argument("--model", default="muse18-r128")
    parser.add_argument("--prompt-lengths", default="593,8192,32768,65000")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1200.0)
    return parser.parse_args()


def request(url: str, body: dict, timeout: float) -> dict:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    call = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(call, timeout=timeout) as response:
        result = json.load(response)
    choice = result["choices"][0]
    logprobs = choice["logprobs"]["token_logprobs"]
    return {
        "text": choice["text"],
        "tokens": choice["logprobs"]["tokens"],
        "logprobs": logprobs,
        "finite": all(math.isfinite(value) for value in logprobs),
    }


def main() -> None:
    args = parse_args()
    lengths = [int(value) for value in args.prompt_lengths.split(",")]
    rows = []
    failed = False
    for prompt_length in lengths:
        prompt = [1000 + index % 1000 for index in range(prompt_length)]
        body = {
            "model": args.model,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": args.output_tokens,
            "min_tokens": args.output_tokens,
            "logprobs": 1,
            "seed": 7,
        }
        eager = [request(args.eager_url, body, args.timeout) for _ in range(args.repetitions)]
        graph = [request(args.graph_url, body, args.timeout) for _ in range(args.repetitions)]
        deltas = []
        tokens_match = True
        for eager_result, graph_result in zip(eager, graph, strict=True):
            tokens_match &= eager_result["tokens"] == graph_result["tokens"]
            if len(eager_result["logprobs"]) != len(graph_result["logprobs"]):
                tokens_match = False
                continue
            deltas.extend(
                abs(left - right)
                for left, right in zip(
                    eager_result["logprobs"], graph_result["logprobs"], strict=True
                )
            )
        finite = all(result["finite"] for result in eager + graph)
        failed |= not tokens_match or not finite
        rows.append(
            {
                "prompt_tokens": prompt_length,
                "output_tokens": args.output_tokens,
                "tokens_match": tokens_match,
                "finite": finite,
                "mean_selected_logprob_delta": statistics.fmean(deltas) if deltas else None,
                "maximum_selected_logprob_delta": max(deltas) if deltas else None,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    print(json.dumps({"passed": not failed, "results": rows}, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
