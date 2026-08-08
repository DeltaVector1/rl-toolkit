#!/usr/bin/env python3
"""Write and compare portable GLM-5.2 FP8 parity traces.

The GPU-side probes only need to call :class:`TraceWriter`.  The comparison
path intentionally depends on NumPy only, so traces from different PyTorch and
CUDA builds can be compared on the orchestrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, offset: int = 0, length: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        left = length
        while left is None or left:
            chunk = stream.read(8 * 1024 * 1024 if left is None else min(left, 8 * 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            if left is not None:
                left -= len(chunk)
    if left not in (None, 0):
        raise EOFError(f"short tensor payload in {path}: {left} bytes missing")
    return digest.hexdigest()


def make_fixture(path: Path, *, length: int, vocab_size: int, seed: int) -> dict[str, Any]:
    """Make token ids without using a tokenizer or a backend RNG."""
    if length < 1:
        raise ValueError("length must be positive")
    if vocab_size < 1024:
        raise ValueError("vocab_size must be at least 1024")
    # A fixed integer LCG is identical in every Python/NumPy/PyTorch build.
    state = seed & 0xFFFFFFFF
    tokens = np.empty(length, dtype=np.int64)
    for index in range(length):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        tokens[index] = 256 + state % (vocab_size - 256)
    positions = np.arange(length, dtype=np.int64)
    path.mkdir(parents=True, exist_ok=False)
    np.save(path / "input_ids.npy", tokens, allow_pickle=False)
    np.save(path / "positions.npy", positions, allow_pickle=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm52_fp8_fixture",
        "length": length,
        "vocab_size": vocab_size,
        "seed": seed,
        "input_ids_sha256": sha256_file(path / "input_ids.npy"),
        "positions_sha256": sha256_file(path / "positions.npy"),
    }
    (path / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def fixture_digest(path: Path) -> str:
    manifest = json.loads((path / "manifest.json").read_text())
    for name in ("input_ids", "positions"):
        actual = sha256_file(path / f"{name}.npy")
        if actual != manifest[f"{name}_sha256"]:
            raise ValueError(f"fixture {name} checksum mismatch")
    return hashlib.sha256(_json_bytes(manifest)).hexdigest()


def _as_numpy(tensor: Any) -> np.ndarray:
    """Convert Torch, NumPy, or Python values to a portable trace dtype."""
    if isinstance(tensor, (tuple, list)) and len(tensor) == 1:
        tensor = tensor[0]
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    if hasattr(tensor, "cpu"):
        # NumPy does not support BF16 or Float8.  Activations are compared in
        # FP32.  Weight bytes belong in the checkpoint audit, not this path.
        text_dtype = str(getattr(tensor, "dtype", ""))
        if "float8" in text_dtype or "bfloat16" in text_dtype:
            tensor = tensor.float()
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()
    array = np.asarray(tensor)
    if array.dtype.kind == "f":
        return np.ascontiguousarray(array, dtype=np.float32)
    if array.dtype.kind in "iu":
        return np.ascontiguousarray(array, dtype=np.int64)
    if array.dtype.kind == "b":
        return np.ascontiguousarray(array, dtype=np.uint8)
    raise TypeError(f"unsupported trace dtype: {array.dtype}")


class TraceWriter:
    """Write named tensors and a manifest atomically enough for offline probes."""

    def __init__(
        self,
        path: str | Path,
        *,
        side: str,
        fixture_sha256: str,
        model_revision: str,
        run: dict[str, Any] | None = None,
    ):
        if side not in {"ada", "h200"}:
            raise ValueError("side must be 'ada' or 'h200'")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "glm52_fp8_trace",
            "side": side,
            "fixture_sha256": fixture_sha256,
            "model_revision": model_revision,
            "run": run or {},
            "tensors": [],
        }
        self._names: set[str] = set()

    def record(
        self,
        name: str,
        tensor: Any,
        *,
        rows: Iterable[int] | None = None,
        row_axis: int = 0,
        semantic: str = "activation",
    ) -> None:
        if name in self._names:
            raise ValueError(f"duplicate trace tensor: {name}")
        array = _as_numpy(tensor)
        source_shape = list(array.shape)
        selected_rows = None
        if rows is not None:
            selected_rows = [int(row) for row in rows]
            if not selected_rows:
                raise ValueError("rows cannot be empty")
            array = np.take(array, selected_rows, axis=row_axis)
        filename = f"{len(self._names):04d}-{_SAFE_NAME.sub('_', name)}.npy"
        np.save(self.path / filename, array, allow_pickle=False)
        self.manifest["tensors"].append(
            {
                "name": name,
                "file": filename,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "source_shape": source_shape,
                "rows": selected_rows,
                "row_axis": row_axis,
                "semantic": semantic,
                "sha256": sha256_file(self.path / filename),
            }
        )
        self._names.add(name)

    def close(self) -> None:
        temporary = self.path / "manifest.json.tmp"
        temporary.write_bytes(_json_bytes(self.manifest))
        temporary.replace(self.path / "manifest.json")

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()


@dataclass
class TensorComparison:
    name: str
    semantic: str
    shape: list[int]
    exact_fraction: float | None
    cosine: float | None
    rmse: float | None
    relative_rmse: float | None
    max_abs: float | None
    finite: bool


def _load_trace(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema in {path}")
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    if len(entries) != len(manifest["tensors"]):
        raise ValueError(f"duplicate tensor names in {path}")
    for entry in entries.values():
        if sha256_file(path / entry["file"]) != entry["sha256"]:
            raise ValueError(f"trace checksum mismatch: {entry['name']}")
    return manifest, entries


def compare_traces(reference_path: Path, candidate_path: Path) -> list[TensorComparison]:
    reference_manifest, reference = _load_trace(reference_path)
    candidate_manifest, candidate = _load_trace(candidate_path)
    for field in ("fixture_sha256", "model_revision"):
        if reference_manifest[field] != candidate_manifest[field]:
            raise ValueError(f"trace {field} mismatch")
    missing = sorted(reference.keys() - candidate.keys())
    extra = sorted(candidate.keys() - reference.keys())
    if missing or extra:
        raise ValueError(f"trace tensor set mismatch; missing={missing}, extra={extra}")

    results: list[TensorComparison] = []
    for name in reference:
        ref_entry, got_entry = reference[name], candidate[name]
        if ref_entry["source_shape"] != got_entry["source_shape"]:
            raise ValueError(f"source shape mismatch for {name}")
        if ref_entry["rows"] != got_entry["rows"] or ref_entry["row_axis"] != got_entry["row_axis"]:
            raise ValueError(f"sampling mismatch for {name}")
        ref = np.load(reference_path / ref_entry["file"], allow_pickle=False)
        got = np.load(candidate_path / got_entry["file"], allow_pickle=False)
        if ref.shape != got.shape:
            raise ValueError(f"stored shape mismatch for {name}: {ref.shape} != {got.shape}")
        semantic = ref_entry["semantic"]
        if semantic in {"indices", "count", "digest"} or ref.dtype.kind in "iub":
            exact = float(np.mean(ref == got)) if ref.size else 1.0
            results.append(TensorComparison(name, semantic, list(ref.shape), exact, None, None, None, None, True))
            continue
        ref64, got64 = ref.astype(np.float64), got.astype(np.float64)
        finite = bool(np.isfinite(ref64).all() and np.isfinite(got64).all())
        if not finite:
            results.append(TensorComparison(name, semantic, list(ref.shape), None, None, None, None, None, False))
            continue
        delta = got64 - ref64
        rmse = float(math.sqrt(np.mean(delta * delta))) if delta.size else 0.0
        ref_rms = float(math.sqrt(np.mean(ref64 * ref64))) if ref64.size else 0.0
        relative_rmse = rmse / max(ref_rms, 1e-30)
        denominator = float(np.linalg.norm(ref64.ravel()) * np.linalg.norm(got64.ravel()))
        cosine = (
            float(np.dot(ref64.ravel(), got64.ravel()) / denominator)
            if denominator
            else float(ref64.size == 0 or np.array_equal(ref64, got64))
        )
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        results.append(
            TensorComparison(name, semantic, list(ref.shape), None, cosine, rmse, relative_rmse, max_abs, True)
        )
    return results


def _safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        header = json.loads(stream.read(length))
    return 8 + length, header


def audit_checkpoint(model: Path, output: Path | None = None) -> dict[str, Any]:
    """Audit every checkpoint FP8 weight/scale pair without allocating tensors."""
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    headers: dict[str, tuple[int, dict[str, Any]]] = {}
    entries: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    for name, relative in index.items():
        if relative not in headers:
            headers[relative] = _safetensor_header(model / relative)
        payload, header = headers[relative]
        entries[name] = (model / relative, payload, header[name])

    failures: list[str] = []
    weights: list[dict[str, Any]] = []
    for name, (path, payload, metadata) in sorted(entries.items()):
        if metadata["dtype"] not in {"F8_E4M3", "F8_E4M3FN"}:
            continue
        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
        if scale_name not in entries:
            failures.append(f"{name}: missing {scale_name}")
            continue
        scale_path, scale_payload, scale = entries[scale_name]
        if scale_path != path:
            failures.append(f"{name}: weight and scale are in different shards")
        shape = metadata["shape"]
        expected = [math.ceil(shape[0] / 128), math.ceil(shape[1] / 128)]
        if scale["shape"] != expected:
            failures.append(f"{scale_name}: shape {scale['shape']} != {expected}")
        if scale["dtype"] != "F32":
            failures.append(f"{scale_name}: dtype {scale['dtype']} != F32")
        start, end = metadata["data_offsets"]
        scale_start, scale_end = scale["data_offsets"]
        expected_weight_bytes = math.prod(shape)
        expected_scale_bytes = 4 * math.prod(scale["shape"])
        if end - start != expected_weight_bytes:
            failures.append(
                f"{name}: payload has {end - start} bytes, expected {expected_weight_bytes}"
            )
        if scale_end - scale_start != expected_scale_bytes:
            failures.append(
                f"{scale_name}: payload has {scale_end - scale_start} bytes, expected {expected_scale_bytes}"
            )
        weights.append(
            {
                "name": name,
                "shape": shape,
                "scale_shape": scale["shape"],
                "native_128x128": shape[0] % 128 == 0 and shape[1] % 128 == 0,
                "weight_sha256": sha256_file(path, payload + start, end - start),
                "scale_sha256": sha256_file(scale_path, scale_payload + scale_start, scale_end - scale_start),
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "glm52_fp8_checkpoint_audit",
        "model": str(model),
        "index_sha256": sha256_file(index_path),
        "fp8_weight_count": len(weights),
        "failures": failures,
        "weights": weights,
    }
    if output:
        output.write_bytes(_json_bytes(report))
    return report


def _print_results(results: list[TensorComparison]) -> None:
    print(f"{'name':64} {'shape':18} {'exact/cos':>12} {'rel_rmse':>12} {'max_abs':>12}")
    for item in results:
        score = item.exact_fraction if item.exact_fraction is not None else item.cosine
        print(
            f"{item.name:64.64} {str(item.shape):18.18} "
            f"{score if score is not None else float('nan'):12.6g} "
            f"{item.relative_rmse if item.relative_rmse is not None else float('nan'):12.6g} "
            f"{item.max_abs if item.max_abs is not None else float('nan'):12.6g}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--length", type=int, default=2304)
    fixture.add_argument("--vocab-size", type=int, required=True)
    fixture.add_argument("--seed", type=int, default=20260808)
    compare = subparsers.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--max-relative-rmse", type=float)
    compare.add_argument("--min-exact-fraction", type=float, default=1.0)
    compare.add_argument("--json", type=Path)
    audit = subparsers.add_parser("audit-checkpoint")
    audit.add_argument("model", type=Path)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "fixture":
        result = make_fixture(
            args.output,
            length=args.length,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "audit-checkpoint":
        report = audit_checkpoint(args.model, args.output)
        print(json.dumps({key: value for key, value in report.items() if key != "weights"}, indent=2))
        return int(bool(report["failures"]))

    results = compare_traces(args.reference, args.candidate)
    _print_results(results)
    if args.json:
        args.json.write_bytes(_json_bytes([asdict(result) for result in results]))
    failed = any(
        not result.finite
        or (result.exact_fraction is not None and result.exact_fraction < args.min_exact_fraction)
        or (
            args.max_relative_rmse is not None
            and result.relative_rmse is not None
            and result.relative_rmse > args.max_relative_rmse
        )
        for result in results
    )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
