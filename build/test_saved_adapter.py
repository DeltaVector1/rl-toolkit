#!/usr/bin/env python3
"""Small CPU/Gloo qualification for the saved-adapter loader draft."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import shutil
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from safetensors.torch import load_file, save_file


class MultiLoRAModule(nn.Module):
    def __init__(self, n_adapters: int = 1) -> None:
        super().__init__()
        self.n_adapters = n_adapters


class TinyOrdinary(MultiLoRAModule):
    def __init__(self, mesh, name: str, n_adapters: int = 1) -> None:
        super().__init__(n_adapters)
        self._name = name
        width = 3
        rank = 2
        self.lora_a = nn.ParameterList(
            [nn.Parameter(_shard(torch.empty(rank, width), mesh)) for _ in range(n_adapters)]
        )
        self.lora_b = nn.ParameterList(
            [nn.Parameter(_shard(torch.empty(width, rank), mesh)) for _ in range(n_adapters)]
        )
        for index in range(n_adapters):
            with torch.no_grad():
                self.lora_a[index].copy_(_shard(torch.arange(rank * width, dtype=torch.bfloat16).reshape(rank, width) + index, mesh))
                self.lora_b[index].copy_(_shard(torch.arange(width * rank, dtype=torch.bfloat16).reshape(width, rank) + 10 + index, mesh))

    def named_parameters_for_adapter(self, index: int):
        return [(f"{self._name}.lora_A", self.lora_a[index]), (f"{self._name}.lora_B", self.lora_b[index])]


class NativeFP8MultiLoRAGroupedExperts(MultiLoRAModule):
    def __init__(self, mesh, num_experts: int = 4, n_adapters: int = 1) -> None:
        super().__init__(n_adapters)
        self.num_experts = num_experts
        self._width = 3
        self._hidden = 4
        self._rank = 2
        for projection, shape_a, shape_b in (
            ("w1", (num_experts, self._rank, self._width), (num_experts, self._hidden, self._rank)),
            ("w2", (num_experts, self._rank, self._hidden), (num_experts, self._width, self._rank)),
            ("w3", (num_experts, self._rank, self._width), (num_experts, self._hidden, self._rank)),
        ):
            setattr(self, f"{projection}_lora_A", nn.ParameterList([nn.Parameter(_shard(torch.empty(shape_a), mesh))]))
            setattr(self, f"{projection}_lora_B", nn.ParameterList([nn.Parameter(_shard(torch.empty(shape_b), mesh))]))
        with torch.no_grad():
            value = 1
            for projection in ("w1", "w2", "w3"):
                for factor in ("A", "B"):
                    parameter = getattr(self, f"{projection}_lora_{factor}")[0]
                    full = torch.arange(parameter.shape[0] * parameter.shape[1] * parameter.shape[2], dtype=torch.bfloat16).reshape(parameter.shape) + value
                    parameter.copy_(_shard(full, mesh))
                    value += 100

    def named_parameters_for_adapter(self, index: int):
        return [
            (f"{projection}_lora_{factor}", getattr(self, f"{projection}_lora_{factor}")[index])
            for projection in ("w1", "w2", "w3")
            for factor in ("A", "B")
        ]


def _shard(value: torch.Tensor, mesh):
    if mesh is None:
        return value
    return torch.distributed.tensor.distribute_tensor(value, mesh, [Shard(0)])


class TinyStage(nn.Module):
    def __init__(self, mesh) -> None:
        super().__init__()
        self.attn = TinyOrdinary(mesh, "q_proj")
        self.mlp = nn.Module()
        self.mlp.shared_expert = TinyOrdinary(mesh, "gate_proj")
        self.experts = NativeFP8MultiLoRAGroupedExperts(mesh)


class TinyModel(nn.Module):
    def __init__(self, mesh, stages: list[str]) -> None:
        super().__init__()
        for stage in stages:
            self.add_module(stage, TinyStage(mesh))

    @classmethod
    def convert_adapter_to_hf(cls, state_dict):
        for old_key in list(state_dict):
            new_key = old_key.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
            if new_key != old_key:
                state_dict[new_key] = state_dict.pop(old_key)
        return state_dict


def _extract_function(source: Path, name: str, globals_dict: dict):
    tree = ast.parse(source.read_text(), filename=str(source))
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), globals_dict)
    return globals_dict[name]


def _load_draft(source: Path):
    native_source = source.parent / "models" / "layers" / "lora" / "native_fp8_moe.py"
    namespace = {
        "torch": torch,
        "nn": nn,
        "DTensor": DTensor,
        "MultiLoRAModule": MultiLoRAModule,
        "NativeFP8MultiLoRAGroupedExperts": NativeFP8MultiLoRAGroupedExperts,
    }
    state_method = _extract_function(native_source, "state_dict_for_adapter", namespace)
    NativeFP8MultiLoRAGroupedExperts.state_dict_for_adapter = state_method
    entries = _extract_function(source, "_initial_lora_adapter_entries", namespace)
    validate = _extract_function(source, "_validate_initial_lora_adapter_inventory", namespace)
    namespace["get_logger"] = lambda: _Logger()
    load = _extract_function(source, "load_initial_lora_adapter", namespace)
    return entries, validate, load


class _Logger:
    def info(self, message: str) -> None:
        pass


def _export(model, entries):
    entries(model, 0)
    state = {}
    for module_name, module in model.named_modules():
        if isinstance(module, NativeFP8MultiLoRAGroupedExperts):
            items = module.state_dict_for_adapter(0).items()
            for name, value in items:
                state[f"{module_name}.{name}"] = value.detach().contiguous().cpu()
        elif isinstance(module, MultiLoRAModule):
            for name, parameter in module.named_parameters_for_adapter(0):
                value = parameter.detach()
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                state[f"{module_name}.{name}.weight"] = value.contiguous().cpu()
    return model.convert_adapter_to_hf(state)


def _assert_loaded(model, state, stages):
    for key, value in state.items():
        if not any(key.startswith(f"{stage}.") for stage in stages):
            continue
        # Compare through the loader's inventory key, using an explicit module scan.
        found = False
        for module_name, candidate in model.named_modules():
            exported_name = module_name.replace(".mlp.shared_expert", ".mlp.shared_experts")
            if key.startswith(exported_name + ".") and isinstance(candidate, MultiLoRAModule):
                suffix = key[len(exported_name) + 1 :]
                if isinstance(candidate, NativeFP8MultiLoRAGroupedExperts):
                    expert, projection, _, factor, _ = suffix.split(".")
                    parameter = getattr(candidate, f"{ {'gate_proj':'w1','down_proj':'w2','up_proj':'w3'}[projection] }_lora_{factor}")[0]
                    actual = parameter.full_tensor()[int(expert)] if isinstance(parameter, DTensor) else parameter[int(expert)]
                else:
                    target, _, factor, _ = suffix.split(".")
                    parameter = candidate.lora_a[0] if factor == "A" else candidate.lora_b[0]
                    actual = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
                assert torch.equal(actual.cpu(), value), f"loaded value differs for {key}"
                found = True
                break
        assert found, f"no local module for {key}"


def _snapshot(model):
    return {name: (value.full_tensor() if isinstance(value, DTensor) else value.detach().clone()).cpu() for name, value in model.named_parameters()}


def _assert_snapshot(model, snapshot):
    now = _snapshot(model)
    assert snapshot.keys() == now.keys()
    for name in snapshot:
        assert torch.equal(snapshot[name], now[name]), f"parameter changed after rejected load: {name}"


def _write(path: Path, state) -> None:
    save_file(state, str(path))


def _inventory_failures(state, load, root: Path, rank: int) -> None:
    cases = []
    missing = dict(state)
    missing.pop(sorted(missing)[0])
    cases.append(("missing", missing))
    unexpected = dict(state)
    unexpected["unexpected.key"] = torch.zeros(1, dtype=torch.bfloat16)
    cases.append(("unexpected", unexpected))
    wrong = dict(state)
    wrong[sorted(wrong)[0]] = torch.zeros(1, dtype=torch.bfloat16)
    cases.append(("wrongshape", wrong))
    for name, variant in cases:
        path = root / f"bad-{name}.safetensors"
        if rank == 0:
            _write(path, variant)
        dist.barrier()
        model_copy = TinyModel(_mesh, ["stage0", "stage1"])
        before = _snapshot(model_copy)
        try:
            load(model_copy, str(path))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} inventory was accepted")
        _assert_snapshot(model_copy, before)
        dist.barrier()


def _worker(rank: int, world_size: int, port: int, source: str, root: str) -> None:
    global _mesh
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        _mesh = init_device_mesh("cpu", (world_size,))
        entries, _validate, load = _load_draft(Path(source))
        work = Path(root)
        model = TinyModel(_mesh, ["stage0", "stage1"])
        state = _export(model, entries)
        path = work / "adapter_model.safetensors"
        if rank == 0:
            _write(path, state)
        dist.barrier()
        fresh = TinyModel(_mesh, ["stage0", "stage1"])
        load(fresh, str(path))
        _assert_loaded(fresh, state, ["stage0", "stage1"])
        if rank == 0:
            second = work / "adapter_model-reexport.safetensors"
            _write(second, _export(fresh, entries))
            assert path.read_bytes() == second.read_bytes(), "re-export is not byte-identical"
        dist.barrier()
        _inventory_failures(state, load, work, rank)

        # Each rank represents a different pipeline stage. The global file has both stages.
        if rank == 0:
            # This export-only model is not distributed, so rank 1 can reach the barrier.
            global_model = TinyModel(None, ["stage0", "stage1"])
            union_state = _export(global_model, entries)
            _write(work / "adapter-union.safetensors", union_state)
        dist.barrier()
        local = TinyModel(_mesh, [f"stage{rank}"])
        load(local, str(work / "adapter-union.safetensors"))
        _assert_loaded(local, union_state if rank == 0 else load_file(str(work / "adapter-union.safetensors"), device="cpu"), [f"stage{rank}"])
        if rank == 0:
            print("PASS: saved adapter round-trip, native experts, inventory rejection, and PP key union")
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="trainer lora.py draft")
    args = parser.parse_args()
    if not args.source.is_file():
        raise SystemExit(f"source does not exist: {args.source}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = tempfile.mkdtemp(prefix="saved-adapter-test-")
    try:
        mp.spawn(_worker, args=(2, port, str(args.source), root), nprocs=2, join=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
