#!/usr/bin/env python3
"""Validate a paired GLM-5.2 H200 MTP benchmark report.

The script does not launch a service.  It consumes measurements collected by a
separate, explicitly authorised H200 run and refuses incomplete or fabricated
reports.  Both MTP-off and MTP-on phases are required before a pass is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROFILE = "glm52-mtp-h200"
SCHEMA_VERSION = 1
MAX_UNMASKED_KL = 0.1
MAX_MASKED_KL = 0.5
MAX_MASKED_FRACTION = 0.06
MAX_ROUTE_WEIGHT_ABS_ERROR = 0.01
MAX_PEAK_VRAM_MIB = 143000.0
MAX_MTP_EXTRA_VRAM_MIB = 4096.0
MIN_ACCEPTANCE_RATE = 0.50
MIN_SPEEDUP = 1.05
MAX_WORKER_RESTARTS = 0
DRAFT_ROUTE_TRACE_FLAGS = {}

PHASE_FIELDS = {
    "output_tokens",
    "elapsed_seconds",
    "peak_vram_mib",
    "worker_restarts",
    "short_decode_exact",
    "token_sha256",
    "unmasked_mismatch_kl",
    "masked_mismatch_kl",
    "masked_fraction",
    "route_ids_mismatch",
    "route_weights_max_abs_error",
    "route_trace_mode",
    "target_request_flags",
    "draft_request_flags",
    "target_route_trace_requests",
    "draft_route_trace_requests",
}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _phase_errors(name: str, phase: Any, *, mtp: bool = False) -> list[str]:
    failures: list[str] = []
    if not isinstance(phase, dict):
        return [f"{name} phase is missing"]
    expected_fields = PHASE_FIELDS | {"accepted_tokens", "proposed_tokens"} if mtp else PHASE_FIELDS
    if set(phase) != expected_fields:
        failures.append(f"{name} phase fields are incomplete or unexpected")
        return failures
    for field in ("output_tokens", "worker_restarts", "target_route_trace_requests", "draft_route_trace_requests", "route_ids_mismatch"):
        value = phase[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            failures.append(f"{name}.{field} is invalid")
    for field in ("elapsed_seconds", "peak_vram_mib", "unmasked_mismatch_kl", "masked_mismatch_kl", "masked_fraction", "route_weights_max_abs_error"):
        value = phase[field]
        if not _finite_number(value) or value < 0:
            failures.append(f"{name}.{field} is invalid")
    if phase["output_tokens"] <= 0 or phase["elapsed_seconds"] <= 0:
        failures.append(f"{name} has no positive output/time measurement")
    if phase["peak_vram_mib"] > MAX_PEAK_VRAM_MIB:
        failures.append(f"{name} peak VRAM exceeds {MAX_PEAK_VRAM_MIB:.0f} MiB")
    if phase["worker_restarts"] > MAX_WORKER_RESTARTS:
        failures.append(f"{name} had a worker restart")
    if phase["short_decode_exact"] is not True:
        failures.append(f"{name} short decode is not exact")
    if not _hex64(phase["token_sha256"]):
        failures.append(f"{name}.token_sha256 is invalid")
    if phase["unmasked_mismatch_kl"] > MAX_UNMASKED_KL:
        failures.append(f"{name} unmasked mismatch KL exceeds {MAX_UNMASKED_KL}")
    if phase["masked_mismatch_kl"] > MAX_MASKED_KL:
        failures.append(f"{name} masked mismatch KL exceeds {MAX_MASKED_KL}")
    if phase["masked_fraction"] > MAX_MASKED_FRACTION:
        failures.append(f"{name} masked fraction exceeds {MAX_MASKED_FRACTION}")
    if phase["route_ids_mismatch"] != 0:
        failures.append(f"{name} route IDs differ")
    if phase["route_weights_max_abs_error"] > MAX_ROUTE_WEIGHT_ABS_ERROR:
        failures.append(f"{name} route weight error exceeds {MAX_ROUTE_WEIGHT_ABS_ERROR}")
    if phase["route_trace_mode"] != "target-only":
        failures.append(f"{name} route trace mode is not target-only")
    target_flags = phase["target_request_flags"]
    if (
        not isinstance(target_flags, dict)
        or set(target_flags)
        != {
            "prime_router_trace",
            "prime_router_trace_session",
            "prime_router_trace_request",
        }
        or target_flags.get("prime_router_trace") is not True
        or not _hex64(target_flags.get("prime_router_trace_session"))
        or not isinstance(target_flags.get("prime_router_trace_request"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", target_flags["prime_router_trace_request"])
    ):
        failures.append(f"{name} target request does not set the exact trace flags")
    if phase["draft_request_flags"] != DRAFT_ROUTE_TRACE_FLAGS:
        failures.append(f"{name} draft request carries a route trace flag")
    if phase["target_route_trace_requests"] <= 0 or phase["draft_route_trace_requests"] != 0:
        failures.append(f"{name} route trace was not isolated to target requests")
    if mtp:
        proposed = phase["proposed_tokens"]
        accepted = phase["accepted_tokens"]
        if isinstance(proposed, bool) or not isinstance(proposed, int) or proposed <= 0:
            failures.append("mtp.proposed_tokens is invalid")
        if isinstance(accepted, bool) or not isinstance(accepted, int) or accepted < 0:
            failures.append("mtp.accepted_tokens is invalid")
        if (
            isinstance(proposed, int)
            and isinstance(accepted, int)
            and accepted > proposed
        ):
            failures.append("MTP accepted token count exceeds proposals")
    return failures


def validate_report(report: Any) -> tuple[list[str], dict[str, Any] | None]:
    if not isinstance(report, dict):
        return ["benchmark report must be an object"], None
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        failures.append("benchmark schema version differs")
    if report.get("kind") != "glm52_mtp_h200_benchmark":
        failures.append("benchmark kind differs")
    if report.get("profile") != PROFILE:
        failures.append("benchmark profile differs")
    if report.get("status") != "completed":
        failures.append("benchmark is not completed; no pass is possible")
    if set(report) != {"schema_version", "kind", "profile", "status", "baseline", "mtp"}:
        failures.append("benchmark report fields are incomplete or unexpected")
    baseline = report.get("baseline")
    mtp = report.get("mtp")
    failures.extend(_phase_errors("baseline", baseline))
    failures.extend(_phase_errors("mtp", mtp, mtp=True))
    if isinstance(baseline, dict) and isinstance(mtp, dict):
        if baseline.get("token_sha256") != mtp.get("token_sha256"):
            failures.append("MTP tokens differ from the MTP-off baseline")
        proposed = mtp.get("proposed_tokens")
        accepted = mtp.get("accepted_tokens")
        if isinstance(proposed, int) and isinstance(accepted, int) and proposed > 0 and accepted <= proposed:
            acceptance_rate = accepted / proposed
            if acceptance_rate < MIN_ACCEPTANCE_RATE:
                failures.append(f"MTP acceptance rate {acceptance_rate:.4f} is below {MIN_ACCEPTANCE_RATE:.2f}")
        else:
            acceptance_rate = None
        baseline_tps = baseline["output_tokens"] / baseline["elapsed_seconds"]
        mtp_tps = mtp["output_tokens"] / mtp["elapsed_seconds"]
        speedup = mtp_tps / baseline_tps if baseline_tps > 0 else None
        if speedup is None or speedup < MIN_SPEEDUP:
            failures.append(f"MTP throughput speedup {speedup!r} is below {MIN_SPEEDUP:.2f}")
        memory_delta = mtp["peak_vram_mib"] - baseline["peak_vram_mib"]
        if memory_delta > MAX_MTP_EXTRA_VRAM_MIB:
            failures.append(f"MTP peak VRAM delta {memory_delta:.2f} MiB exceeds {MAX_MTP_EXTRA_VRAM_MIB:.0f} MiB")
        derived = {
            "acceptance_rate": acceptance_rate,
            "baseline_output_tokens_per_second": baseline_tps,
            "mtp_output_tokens_per_second": mtp_tps,
            "throughput_speedup": speedup,
            "peak_vram_delta_mib": memory_delta,
            "worker_restarts": {"baseline": baseline["worker_restarts"], "mtp": mtp["worker_restarts"]},
            "kl_route_criteria": not any("KL" in failure or "route" in failure for failure in failures),
        }
    else:
        derived = None
    return failures, derived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.input.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        print(f"ERROR: cannot read benchmark input: {error}", file=sys.stderr)
        return 2
    failures, derived = validate_report(report)
    if failures:
        print("MTP benchmark refused: " + "; ".join(failures), file=sys.stderr)
        return 2
    result = dict(report)
    result["derived"] = derived
    result["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
