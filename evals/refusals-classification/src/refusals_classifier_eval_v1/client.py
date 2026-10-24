from __future__ import annotations

import asyncio
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


valid_labels = {0: "refusal", 1: "compliance"}


@dataclass(frozen=True)
class Classification:
    label: int
    confidence: float
    label_name: str
    probabilities: tuple[float, float]
    input_tokens: int
    truncated: bool
    model_revision: str


def _request(url: str, text: str, timeout: float) -> Classification:
    request = urllib.request.Request(
        url.rstrip("/") + "/classify",
        data=json.dumps({"texts": [text]}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    results = body.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("classifier response must contain one result")
    result = results[0]
    label = result.get("label")
    label_name = str(result.get("label_name") or "").strip().lower()
    probabilities = result.get("probabilities")
    if label not in valid_labels:
        raise ValueError("classifier returned an unknown label")
    if label_name != valid_labels[label]:
        raise ValueError("classifier label name does not match its label")
    if not isinstance(probabilities, list) or len(probabilities) != 2:
        raise ValueError("classifier returned invalid probabilities")
    values = tuple(float(value) for value in probabilities)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("classifier probabilities are outside zero to one")
    if abs(sum(values) - 1.0) > 1e-3:
        raise ValueError("classifier probabilities do not sum to one")
    confidence = float(result.get("confidence"))
    if abs(confidence - values[label]) > 1e-4:
        raise ValueError("classifier confidence does not match its label")
    revision = str(body.get("model_revision") or "")
    if not revision:
        raise ValueError("classifier response has no model revision")
    return Classification(
        label=label,
        label_name=label_name,
        confidence=confidence,
        probabilities=values,
        input_tokens=int(result.get("input_tokens", 0)),
        truncated=bool(result.get("truncated", False)),
        model_revision=revision,
    )


async def classify_response(
    base_url: str,
    text: str,
    attempts: int,
    request_timeout_seconds: float,
    total_timeout_seconds: float,
) -> tuple[Classification, list[dict]]:
    started = time.monotonic()
    records = []
    for attempt in range(1, attempts + 1):
        remaining = total_timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        timeout = min(request_timeout_seconds, remaining)
        attempt_started = time.monotonic()
        try:
            result = await asyncio.to_thread(_request, base_url, text, timeout)
            records.append(
                {
                    "attempt": attempt,
                    "status": "scored",
                    "elapsed_seconds": time.monotonic() - attempt_started,
                }
            )
            return result, records
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as error:
            records.append(
                {
                    "attempt": attempt,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.monotonic() - attempt_started,
                }
            )
        if attempt < attempts:
            remaining = total_timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.25 * 2 ** (attempt - 1) + random.random() * 0.1, remaining))
    raise RuntimeError("refusal classifier failed after bounded retries")
