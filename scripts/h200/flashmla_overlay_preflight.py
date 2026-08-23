#!/usr/bin/env python3
"""Prepare and verify the isolated vLLM runtime overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EXPECTED_IMAGE_ID = "sha256:2f803571329212bfb4e27982efec7559f20379bcde73ac6ccee6bb8be3c46ade"
PATCHES = (
    "patches/vllm/0001-v026-flashmla-sparse-scratch-limit.patch",
    "patches/vllm/0002-v026-flashmla-sparse-mixed-batch-chunks.patch",
    "patches/vllm/0003-v026-exact-target-prompt-logprobs.patch",
    "patches/vllm/0004-v026-request-exact-router-trace.patch",
)
TEST_FILE = "patches/vllm/test_flashmla_sparse_chunk_v026.py"
FILES = {
    "envs": "vllm/envs.py",
    "flashmla": "vllm/v1/attention/backends/mla/flashmla_sparse.py",
    "logits_processor": "vllm/model_executor/layers/logits_processor.py",
    "deepseek_v2": "vllm/model_executor/models/deepseek_v2.py",
    "gpu_model_runner": "vllm/v1/worker/gpu_model_runner.py",
    "scheduler_output": "vllm/v1/core/sched/output.py",
    "scheduler": "vllm/v1/core/sched/scheduler.py",
}
SOURCE_FILES = {
    "parity_router_trace": "source/src/prime_rl/parity_router_trace.py",
}


class OverlayError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_id(image: str) -> str:
    try:
        return subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise OverlayError(f"cannot inspect base image {image}: {error.output.strip()}") from error


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as error:
        raise OverlayError(f"preflight command failed ({error.returncode}): {' '.join(command)}") from error


def _check_root(deployment: Path, overlay: Path) -> None:
    deployment = deployment.resolve()
    overlay = overlay.resolve(strict=False)
    cache = (deployment / "cache").resolve()
    if overlay == cache or not overlay.is_relative_to(cache):
        raise OverlayError(f"overlay must be inside the isolated cache {cache}: {overlay}")


def _manifest_path(overlay: Path) -> Path:
    return overlay / "manifest.json"


def verify_manifest(deployment: Path, overlay: Path, image: str) -> dict[str, object]:
    _check_root(deployment, overlay)
    manifest_path = _manifest_path(overlay)
    if not manifest_path.is_file():
        raise OverlayError(f"overlay preflight manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OverlayError(f"overlay preflight manifest is unreadable: {manifest_path}") from error
    if manifest.get("preflight_passed") is not True:
        raise OverlayError("overlay preflight marker is not passed")
    actual_image_id = image_id(image)
    if manifest.get("image_id") != actual_image_id or actual_image_id != EXPECTED_IMAGE_ID:
        raise OverlayError(
            f"overlay image identity mismatch: manifest={manifest.get('image_id')!r} "
            f"actual={actual_image_id!r} expected={EXPECTED_IMAGE_ID!r}"
        )
    for name, relative in FILES.items():
        path = overlay / relative
        if not path.is_file() or path.is_symlink():
            raise OverlayError(f"overlay file is missing or not regular: {path}")
        expected = manifest.get("files", {}).get(name)
        actual = sha256(path)
        if expected != actual:
            raise OverlayError(f"overlay hash mismatch for {name}: {actual} != {expected}")
    for relative in PATCHES:
        path = deployment / relative
        expected = manifest.get("patches", {}).get(relative)
        if not path.is_file() or expected != sha256(path):
            raise OverlayError(f"overlay patch is missing or changed: {path}")
    test_path = deployment / TEST_FILE
    if manifest.get("test_sha256") != sha256(test_path):
        raise OverlayError(f"overlay test is missing or changed: {test_path}")
    launcher = deployment / "scripts/h200/start_inference.sh"
    if not launcher.is_file():
        raise OverlayError(f"inference launcher is missing: {launcher}")
    launcher_text = launcher.read_text()
    if (
        '"$deployment/source/src/prime_rl/parity_router_trace.py":/app/src/prime_rl/parity_router_trace.py:ro'
        not in launcher_text
        or "/app/src/prime_rl/parity_router_trace.py:ro" not in launcher_text
    ):
        raise OverlayError("launcher is missing the parity_router_trace source mount")
    for name, relative in SOURCE_FILES.items():
        path = deployment / relative
        if not path.is_file() or path.is_symlink():
            raise OverlayError(f"source file is missing or not regular: {path}")
        expected = manifest.get("source_files", {}).get(name)
        actual = sha256(path)
        if expected != actual:
            raise OverlayError(f"source hash mismatch for {name}: {actual} != {expected}")
    trace_source = deployment / SOURCE_FILES["parity_router_trace"]
    if "def h200_trace_request(" not in trace_source.read_text():
        raise OverlayError("parity_router_trace does not export h200_trace_request")
    return manifest


def prepare(deployment: Path, overlay: Path, image: str) -> dict[str, object]:
    _check_root(deployment, overlay)
    if image_id(image) != EXPECTED_IMAGE_ID:
        raise OverlayError(f"refusing unexpected base image identity for {image}")
    if overlay.exists() or overlay.is_symlink():
        if overlay.is_symlink() or not overlay.resolve().is_relative_to(deployment.resolve()):
            raise OverlayError(f"refusing unsafe overlay path: {overlay}")
        shutil.rmtree(overlay)
    overlay.mkdir(parents=True)
    for relative in FILES.values():
        (overlay / relative).parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "0",
            "--entrypoint",
            "sh",
            "-v",
            f"{overlay}:/overlay",
            image,
            "-lc",
            "cp /app/.venv/lib/python3.12/site-packages/vllm/envs.py /overlay/vllm/envs.py && "
            "cp /app/.venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py "
            "/overlay/vllm/v1/attention/backends/mla/flashmla_sparse.py && "
            "cp /app/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/logits_processor.py "
            "/overlay/vllm/model_executor/layers/logits_processor.py && "
            "cp /app/.venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py "
            "/overlay/vllm/model_executor/models/deepseek_v2.py && "
            "cp /app/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py "
            "/overlay/vllm/v1/worker/gpu_model_runner.py && "
            "cp /app/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/output.py "
            "/overlay/vllm/v1/core/sched/output.py && "
            "cp /app/.venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py "
            "/overlay/vllm/v1/core/sched/scheduler.py && "
            "chmod 0644 " + " ".join(f"/overlay/{relative}" for relative in FILES.values()),
        ]
    )
    for relative in PATCHES:
        _run(["git", "apply", "--check", "--unsafe-paths", str(deployment / relative)], cwd=overlay)
        _run(["git", "apply", "--unsafe-paths", str(deployment / relative)], cwd=overlay)
    for relative in FILES.values():
        _run([sys.executable, "-m", "py_compile", str(overlay / relative)])
    test_path = deployment / TEST_FILE
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            "-v",
            f"{overlay}:/overlay:ro",
            "-v",
            f"{test_path}:/tmp/test.py:ro",
            "-e",
            "VLLM_V026_SOURCE_ROOT=/overlay",
            image,
            "-lc",
            "/app/.venv/bin/python /tmp/test.py && "
            "grep -q 'get_selected_token_logprobs' /overlay/vllm/model_executor/layers/logits_processor.py && "
            "grep -q 'class GlmMoeDsaForCausalLM' /overlay/vllm/model_executor/models/deepseek_v2.py && "
            "grep -q 'compute_selected_token_logprobs' /overlay/vllm/model_executor/models/deepseek_v2.py && "
            "grep -q 'prime_exact_target_prompt_logprobs' /overlay/vllm/v1/worker/gpu_model_runner.py && "
            "grep -q 'prime_router_trace' /overlay/vllm/v1/worker/gpu_model_runner.py && "
            "grep -q 'class RouterTracePacket' /overlay/vllm/v1/core/sched/output.py && "
            "grep -q 'prime_router_trace_packet' /overlay/vllm/v1/core/sched/scheduler.py && "
            "grep -q 'mark_h200_trace_flags' /overlay/vllm/v1/worker/gpu_model_runner.py",
        ]
    )
    for relative in SOURCE_FILES.values():
        source_path = deployment / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise OverlayError(f"source file is missing or not regular: {source_path}")
        _run([sys.executable, "-m", "py_compile", str(source_path)])
    trace_source = deployment / SOURCE_FILES["parity_router_trace"]
    if "def h200_trace_request(" not in trace_source.read_text():
        raise OverlayError("parity_router_trace does not export h200_trace_request")
    manifest = {
        "preflight_passed": True,
        "image": image,
        "image_id": image_id(image),
        "files": {name: sha256(overlay / relative) for name, relative in FILES.items()},
        "patches": {relative: sha256(deployment / relative) for relative in PATCHES},
        "test_sha256": sha256(test_path),
        "source_files": {name: sha256(deployment / relative) for name, relative in SOURCE_FILES.items()},
        "chunk_tokens": 2048,
        "selected_target_prompt_logprobs": True,
    }
    with tempfile.NamedTemporaryFile("w", dir=overlay, delete=False) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, _manifest_path(overlay))
    verify_manifest(deployment, overlay, image)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    try:
        result = (
            prepare(args.deployment, args.overlay, args.image)
            if args.action == "prepare"
            else verify_manifest(args.deployment, args.overlay, args.image)
        )
    except OverlayError as error:
        print(f"overlay preflight refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
