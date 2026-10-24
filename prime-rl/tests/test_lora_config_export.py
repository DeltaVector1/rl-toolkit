import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).parents[1] / "src" / "prime_rl" / "trainer" / "lora.py"


class FakeModule:
    def __init__(self, kind="plain"):
        self.kind = kind

    def adapter_target_modules(self):
        return ("q_proj", "k_proj")


class SharedModule(FakeModule):
    pass


def exported_function():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "save_lora_config"
    )
    namespace = {
        "math": math,
        "nn": SimpleNamespace(Module=object),
        "MultiLoRAModule": FakeModule,
        "MultiLoRAFP8CheckpointFeedForward": SharedModule,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["save_lora_config"]


class FakeModel:
    def __init__(self, shared):
        self.config = SimpleNamespace(_name_or_path="trinity")
        self._modules = [("", FakeModule()), ("q_proj", FakeModule())]
        if shared:
            self._modules.append(("shared_expert", SharedModule()))

    def named_modules(self):
        return iter(self._modules)

    def named_parameters(self):
        return iter(())


def export(tmp_path, *, shared, alpha=64.0):
    exported_function()(FakeModel(shared), tmp_path, 32, alpha, 0.0)
    return json.loads((tmp_path / "adapter_config.json").read_text())


def test_shared_fp8_wrapper_exports_shared_expert(tmp_path):
    config = export(tmp_path, shared=True)
    assert "shared_expert" in config["target_modules"]
    assert {"q_proj", "k_proj"} <= set(config["target_modules"])


def test_export_without_shared_wrapper_keeps_projection_targets(tmp_path):
    config = export(tmp_path, shared=False)
    assert "shared_expert" not in config["target_modules"]
    assert config["target_modules"] == ["k_proj", "q_proj"]


def test_integral_alpha_is_json_integer(tmp_path):
    config = export(tmp_path, shared=False, alpha=64.0)
    assert config["lora_alpha"] == 64
    assert type(config["lora_alpha"]) is int
