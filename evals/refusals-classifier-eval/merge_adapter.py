from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def adapter_modules(adapter: Path) -> dict[str, tuple[str, str]]:
    with safe_open(adapter, framework="pt", device="cpu") as source:
        keys = set(source.keys())
    modules = {}
    for key in sorted(keys):
        suffix = ".lora_A.weight"
        if not key.endswith(suffix):
            continue
        module = key[: -len(suffix)]
        key_b = module + ".lora_B.weight"
        if key_b not in keys:
            raise ValueError(f"adapter has no B tensor for {module}")
        modules[module + ".weight"] = (key, key_b)
    used = {key for pair in modules.values() for key in pair}
    unexpected = keys - used
    if unexpected:
        raise ValueError(f"unexpected adapter tensors: {sorted(unexpected)[:5]}")
    return modules


def copy_model_files(base: Path, output: Path, shards: set[str]) -> None:
    output.mkdir(parents=True)
    for source in base.iterdir():
        if source.name in shards or source.is_dir():
            continue
        shutil.copy2(source, output / source.name)


def read_adapter_config(
    adapter_directory: Path,
    rank: int | None,
    alpha: float | None,
) -> tuple[int, float, float, str]:
    path = adapter_directory / "adapter_config.json"
    if path.exists():
        config = json.loads(path.read_text())
        config_rank = int(config["r"])
        config_alpha = float(config["lora_alpha"])
        dropout = float(config.get("lora_dropout", 0.0))
        if config.get("bias", "none") != "none":
            raise ValueError("merge does not support LoRA bias")
        if rank is not None and rank != config_rank:
            raise ValueError("command rank does not match adapter config")
        if alpha is not None and alpha != config_alpha:
            raise ValueError("command alpha does not match adapter config")
        return config_rank, config_alpha, dropout, str(path)
    if rank is None or alpha is None:
        raise ValueError(
            "adapter_config.json is absent; --rank and --alpha are required"
        )
    return rank, alpha, 0.0, "command line, checked against trainer config"


def merge(
    base: Path,
    adapter_directory: Path,
    output: Path,
    device: str,
    rank: int | None,
    alpha: float | None,
) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    adapter = adapter_directory / "adapter_model.safetensors"
    rank, alpha, dropout, config_source = read_adapter_config(
        adapter_directory,
        rank,
        alpha,
    )
    if rank != 128 or alpha != 256.0:
        raise ValueError(f"unexpected adapter rank or alpha: {rank}, {alpha}")
    if dropout != 0.0:
        raise ValueError("merge requires zero LoRA dropout")
    scale = alpha / rank
    modules = adapter_modules(adapter)
    if len(modules) != 416:
        raise ValueError(f"expected 416 adapted weights, found {len(modules)}")

    index_path = base / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    missing_base = sorted(set(modules) - set(weight_map))
    if missing_base:
        raise ValueError(f"adapter weights missing from base index: {missing_base[:5]}")
    shards = set(weight_map.values())
    copy_model_files(base, output, shards)

    merged_modules = []
    per_weight = {}
    torch_device = torch.device(device)
    with safe_open(adapter, framework="pt", device="cpu") as adapter_file:
        for shard_name in sorted(shards):
            source_path = base / shard_name
            tensors = load_file(source_path, device="cpu")
            with safe_open(source_path, framework="pt", device="cpu") as shard_file:
                metadata = shard_file.metadata()
            for weight_name in sorted(set(tensors) & set(modules)):
                key_a, key_b = modules[weight_name]
                weight = tensors[weight_name]
                a = adapter_file.get_tensor(key_a)
                b = adapter_file.get_tensor(key_b)
                if tuple(a.shape) != (rank, weight.shape[1]):
                    raise ValueError(f"A shape mismatch for {weight_name}")
                if tuple(b.shape) != (weight.shape[0], rank):
                    raise ValueError(f"B shape mismatch for {weight_name}")
                with torch.inference_mode():
                    delta = torch.matmul(
                        b.to(torch_device, dtype=torch.float32),
                        a.to(torch_device, dtype=torch.float32),
                    )
                    delta.mul_(scale)
                    merged = weight.to(torch_device, dtype=torch.float32).add_(delta)
                    tensors[weight_name] = merged.to(dtype=weight.dtype, device="cpu")
                    per_weight[weight_name] = {
                        "delta_abs_max": float(delta.abs().max().item()),
                        "delta_l2": float(torch.linalg.vector_norm(delta).item()),
                        "changed_after_bfloat16_cast": int(
                            torch.count_nonzero(
                                tensors[weight_name].to(torch_device) != weight.to(torch_device)
                            ).item()
                        ),
                    }
                merged_modules.append(weight_name)
                del a, b, delta, merged, weight
            save_file(tensors, output / shard_name, metadata=metadata)
            del tensors
            gc.collect()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    if set(merged_modules) != set(modules):
        missing = sorted(set(modules) - set(merged_modules))
        raise RuntimeError(f"merge missed adapted weights: {missing[:5]}")
    unchanged = [
        name
        for name, values in per_weight.items()
        if values["changed_after_bfloat16_cast"] == 0
    ]
    if unchanged:
        raise RuntimeError(f"adapter made no BF16 change to: {unchanged[:5]}")

    manifest = {
        "base": str(base),
        "base_index_sha256": sha256(index_path),
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter),
        "rank": rank,
        "alpha": alpha,
        "scale": scale,
        "adapter_config_source": config_source,
        "merged_weight_count": len(merged_modules),
        "merged_shards": {
            shard: sha256(output / shard) for shard in sorted(shards)
        },
        "per_weight": per_weight,
    }
    (output / "lora_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    args = parser.parse_args()
    manifest = merge(
        args.base,
        args.adapter,
        args.output,
        args.device,
        args.rank,
        args.alpha,
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "base_index_sha256",
                    "adapter_sha256",
                    "rank",
                    "alpha",
                    "scale",
                    "merged_weight_count",
                    "merged_shards",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
