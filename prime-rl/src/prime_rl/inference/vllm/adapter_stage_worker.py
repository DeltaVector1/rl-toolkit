"""Run one adapter stage outside the inference process."""

import contextlib
import json
import math
import os
import sys

from prime_rl.inference.vllm.worker.filesystem import stage_lora_adapter_files

_MAX_RESULT_BYTES = 1_000_000


def main() -> None:
    stage_id = None
    result = {"stage_id": stage_id, "pid": os.getpid()}
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        stage_id = request.get("stage_id")
        source_url = request.get("source_url")
        require_delta = request.get("require_delta", False)
        require_bundle = request.get("require_bundle", False)
        policy_step = request.get("policy_step")
        stage_timeout = request.get("stage_timeout_seconds")
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError("stage_id is required")
        if not isinstance(source_url, str) or not source_url.startswith(("http://", "https://")):
            raise ValueError("source_url must be an HTTP URL")
        if type(require_delta) is not bool:
            raise ValueError("require_delta must be a boolean")
        if type(require_bundle) is not bool:
            raise ValueError("require_bundle must be a boolean")
        if require_delta and require_bundle:
            raise ValueError("require_delta and require_bundle are mutually exclusive")
        if require_bundle and (policy_step is not None or stage_timeout is not None):
            raise ValueError("policy_step and stage_timeout_seconds are invalid for bundle staging")
        if policy_step is not None and (type(policy_step) is not int or policy_step <= 0):
            raise ValueError("policy_step must be a positive integer")
        if stage_timeout is not None and (
            isinstance(stage_timeout, bool)
            or not isinstance(stage_timeout, (int, float))
            or not math.isfinite(float(stage_timeout))
            or stage_timeout <= 0
        ):
            raise ValueError("stage_timeout_seconds must be positive")
        result["stage_id"] = stage_id
        os.environ["PRIME_LORA_STAGE_ID"] = stage_id
        with contextlib.redirect_stdout(sys.stderr):
            destination = stage_lora_adapter_files(
                source_url,
                require_delta=require_delta,
                require_bundle=require_bundle,
                stage_timeout_seconds=stage_timeout,
                policy_step=policy_step,
            )
        result.update({"status": "ok", "path": destination.as_posix()})
    except BaseException as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4096],
            }
        )
    output = json.dumps(result, separators=(",", ":"))
    if len(output.encode("utf-8")) > _MAX_RESULT_BYTES:
        output = json.dumps(
            {
                "stage_id": stage_id,
                "pid": os.getpid(),
                "status": "error",
                "error_type": "RuntimeError",
                "error": "adapter stage worker result is too large",
            },
            separators=(",", ":"),
        )
    sys.stdout.write(output)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
