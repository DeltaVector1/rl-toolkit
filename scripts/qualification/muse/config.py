#!/usr/bin/env python3
"""Validate the Muse trainer config and Transformers registration."""

import sys
import tomllib

import torch
import transformers
from transformers import AutoConfig, AutoModelForImageTextToText

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.lora import _find_target_modules
from prime_rl.utils.vlm import get_layer_prefix, is_vlm_architecture


trainer_path, model_path = sys.argv[1:3]
expected_mode = sys.argv[3] if len(sys.argv) > 3 else "fft"
with open(trainer_path, "rb") as source:
    trainer = TrainerConfig.model_validate(tomllib.load(source))
model = AutoConfig.from_pretrained(model_path, local_files_only=True)

assert trainer.model.optimization_dtype == "bfloat16"
assert trainer.model.reduce_dtype == "bfloat16"
assert trainer.model.pp == 8
assert trainer.model.cp == 4
assert trainer.model.cp_style == "ring"
assert trainer.model.seq_len == 65536
assert trainer.optim.lr == 8e-6
assert model.model_type == "muse_glimmer"
assert is_vlm_architecture(model)
assert get_layer_prefix(model) == "model.language_model.layers."
auto_model = AutoModelForImageTextToText._model_mapping[type(model)]

if expected_mode == "fft":
    assert trainer.model.lora is None
elif expected_mode == "lora-r128":
    lora = trainer.model.lora
    assert lora is not None
    assert lora.rank == 128
    assert lora.alpha == 256.0
    assert lora.dropout == 0.0
    assert lora.modules_to_save == []
    with torch.device("meta"):
        model_instance = auto_model(model)
    targets = _find_target_modules(model_instance, lora.target_modules)
    modules = dict(model_instance.named_modules())
    assert len(targets) == 416
    assert all(name.startswith("model.language_model.layers.") for name in targets)
    assert not any(name.startswith("model.vision_tower.") for name in targets)
    attention_targets = [name for name in targets if ".self_attn." in name]
    mlp_targets = [name for name in targets if ".mlp." in name]
    assert len(attention_targets) == 260
    assert len(mlp_targets) == 156
    adapter_params = sum(
        lora.rank * (modules[name].in_features + modules[name].out_features)
        for name in targets
    )
    assert adapter_params == 838_336_512
    print(
        f"lora_targets={len(targets)} attention={len(attention_targets)} "
        f"mlp={len(mlp_targets)} adapter_params={adapter_params}"
    )
else:
    raise ValueError(f"unknown expected mode: {expected_mode}")

print(f"transformers={transformers.__version__}")
print(f"trainer_model={trainer.model.name}")
print(
    f"pp={trainer.model.pp} cp={trainer.model.cp} seq_len={trainer.model.seq_len} "
    f"lora={trainer.model.lora} dtype={trainer.model.optimization_dtype}"
)
print(f"lr={trainer.optim.lr} model_type={model.model_type}")
print(f"layer_prefix={get_layer_prefix(model)} auto_model={auto_model.__name__}")
