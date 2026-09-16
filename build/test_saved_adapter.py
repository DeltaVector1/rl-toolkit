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
    def __init__(self, mesh, name: str, n_adapters: int = 1, offset: int = 0) -> None:
        super().__init__(n_adapters)
        self._name = name
        width = 3
        rank = 2
        self.lora_a = nn.ParameterList(
            [nn.Parameter(_shard(torch.empty(rank, width, dtype=torch.bfloat16), mesh)) for _ in range(n_adapters)]
        )
        self.lora_b = nn.ParameterList(
            [nn.Parameter(_shard(torch.empty(width, rank, dtype=torch.bfloat16), mesh)) for _ in range(n_adapters)]
        )
        for index in range(n_adapters):
            with torch.no_grad():
                self.lora_a[index].copy_(_shard(torch.arange(rank * width, dtype=torch.bfloat16).reshape(rank, width) + offset + index, mesh))
                self.lora_b[index].copy_(_shard(torch.arange(width * rank, dtype=torch.bfloat16).reshape(width, rank) + offset + 10 + index, mesh))

    def named_parameters_for_adapter(self, index: int):
        return [(f"{self._name}.lora_A", self.lora_a[index]), (f"{self._name}.lora_B", self.lora_b[index])]


class NativeFP8MultiLoRAGroupedExperts(MultiLoRAModule):
    def __init__(self, mesh, num_experts: int = 4, n_adapters: int = 1, offset: int = 0) -> None:
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
            setattr(self, f"{projection}_lora_A", nn.ParameterList([nn.Parameter(_shard(torch.empty(shape_a, dtype=torch.bfloat16), mesh))]))
            setattr(self, f"{projection}_lora_B", nn.ParameterList([nn.Parameter(_shard(torch.empty(shape_b, dtype=torch.bfloat16), mesh))]))
        with torch.no_grad():
            value = 1 + offset
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
    def __init__(self, mesh, offset: int = 0) -> None:
        super().__init__()
        self.attn = TinyOrdinary(mesh, "q_proj", offset=offset)
        self.mlp = nn.Module()
        self.mlp.shared_expert = TinyOrdinary(mesh, "gate_proj", offset=offset)
        self.experts = NativeFP8MultiLoRAGroupedExperts(mesh, offset=offset)


class TinyModel(nn.Module):
    def __init__(self, mesh, stages: list[str], offset: int = 0) -> None:
        super().__init__()
        for stage in stages:
            self.add_module(stage, TinyStage(mesh, offset=offset))

    @classmethod
    def convert_adapter_to_hf(cls, state_dict):
        for old_key in list(state_dict):
            new_key = old_key.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
            if new_key != old_key:
                state_dict[new_key] = state_dict.pop(old_key)
        return state_dict


def _extract_function(source: Path, name: str, globals_dict: dict):
    tree = ast.parse(source.read_text(), filename=str(source))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), globals_dict)
    return globals_dict[name]


def _assert_resume_source_order(lora_source: Path) -> None:
    train_source = lora_source.parent / "rl" / "train.py"
    tree = ast.parse(train_source.read_text(), filename=str(train_source))
    train_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train"
    )

    def name(call):
        return call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", "")

    branches = []
    for node in ast.walk(train_fn):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Name) or node.test.id != "resume_adapter":
            continue
        calls = [call for call in ast.walk(node) if isinstance(call, ast.Call)]
        if any(name(call) == "wait_for_run" for call in calls) and any(
            name(call) == "load_initial_lora_adapter" for call in calls
        ):
            branches.append(node)
    assert len(branches) == 1, "resume branch must contain one native-aware load"
    calls = [call for call in ast.walk(branches[0]) if isinstance(call, ast.Call)]
    wait = [call.lineno for call in calls if name(call) == "wait_for_run"]
    load = [call.lineno for call in calls if name(call) == "load_initial_lora_adapter"]
    assert wait and load and max(wait) < min(load), "resume load must follow reset hooks"
    fresh_assignments = [
        node for node in ast.walk(train_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "fresh_lora_policy" for target in node.targets)
    ]
    assert fresh_assignments and any(
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Name)
        and node.operand.id == "resume_adapter"
        for node in ast.walk(fresh_assignments[0].value)
    ), "fresh policy guard must exclude resume_adapter"


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


def _assert_reexport_matches(model, expected_state, entries, stages):
    actual = _export(model, entries)
    expected = {
        key: value
        for key, value in expected_state.items()
        if any(key.startswith(f"{stage}.") for stage in stages)
    }
    assert actual.keys() == expected.keys(), "re-export key inventory differs"
    for key, value in expected.items():
        assert torch.equal(actual[key], value), f"loaded value differs for {key}"


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
        model_copy = TinyModel(_mesh, ["stage0", "stage1"], offset=2000)
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
        model = TinyModel(_mesh, ["stage0", "stage1"], offset=0)
        state = _export(model, entries)
        path = work / "adapter_model.safetensors"
        if rank == 0:
            _write(path, state)
        dist.barrier()
        fresh = TinyModel(_mesh, ["stage0", "stage1"], offset=1000)
        load(fresh, str(path))
        _assert_reexport_matches(fresh, state, entries, ["stage0", "stage1"])
        reexport = _export(fresh, entries)
        if rank == 0:
            second = work / "adapter_model-reexport.safetensors"
            _write(second, reexport)
        dist.barrier()

        # A configured step-0 hook may run during reset. The resume load must run after it.
        zero_path = work / "initial-zero.safetensors"
        zero_state = {key: torch.zeros_like(value) for key, value in state.items()}
        if rank == 0:
            _write(zero_path, zero_state)
        dist.barrier()
        after_reset = TinyModel(_mesh, ["stage0", "stage1"], offset=1000)
        load(after_reset, str(zero_path))
        load(after_reset, str(path))
        _assert_reexport_matches(after_reset, state, entries, ["stage0", "stage1"])
        assert all(
            torch.count_nonzero(value).item() > 0
            for key, value in state.items()
            if "lora_B" in key
        ), "trained source must contain nonzero B matrices"
        dist.barrier()
        _inventory_failures(state, load, work, rank)

        # Each rank represents a different pipeline stage. The global file has both stages.
        if rank == 0:
            # This export-only model is not distributed, so rank 1 can reach the barrier.
            global_model = TinyModel(None, ["stage0", "stage1"], offset=3000)
            union_state = _export(global_model, entries)
            _write(work / "adapter-union.safetensors", union_state)
        dist.barrier()
        local = TinyModel(None, [f"stage{rank}"], offset=4000)
        load(local, str(work / "adapter-union.safetensors"))
        union_state = union_state if rank == 0 else load_file(str(work / "adapter-union.safetensors"), device="cpu")
        _assert_reexport_matches(local, union_state, entries, [f"stage{rank}"])
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
    _assert_resume_source_order(args.source)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = tempfile.mkdtemp(prefix="saved-adapter-test-")
    try:
        mp.spawn(_worker, args=(2, port, str(args.source), root), nprocs=2, join=True)
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
