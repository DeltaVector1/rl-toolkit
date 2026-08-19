#!/usr/bin/env python3
"""Validate the Muse rollout configuration before a production start."""

import sys
import tomllib
import urllib.request

from renderers import create_renderer
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.trainer.model import setup_tokenizer


expected_mode = sys.argv[2] if len(sys.argv) > 2 else "fft"
with open(sys.argv[1], "rb") as source:
    raw_config = tomllib.load(source)
judge_url = raw_config["train"]["env"][0]["env"]["taskset"]["task"]["judge"]["base_url"]
user_url = raw_config["train"]["env"][0]["env"]["taskset"]["task"]["user"]["base_url"]
config = OrchestratorConfig.model_validate(raw_config)

assert config.batch_size == 2048
assert config.group_size == 16
assert config.seq_len == 65536
assert config.optim.lr == 8e-6
if expected_mode == "fft":
    assert config.model.lora is None
elif expected_mode == "lora-r128":
    assert config.model.lora is not None
    assert config.model.lora.name is not None
    assert config.model.lora.name.endswith("-r128")
    assert config.model.lora.rank == 128
    assert config.model.lora.alpha == 256.0
else:
    raise ValueError(f"unknown expected mode: {expected_mode}")
assert config.train.sampling.max_completion_tokens == 16384
assert len(config.model.client.base_url) == 2
assert len(config.model.client.admin_base_url) == 2
assert config.model.client.base_url == [
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:18101/v1",
]
assert config.model.client.admin_base_url == config.model.client.base_url
assert config.weight_broadcast.type == "filesystem"
assert config.train.env[0].pool.num_workers == 16
tokenizer = setup_tokenizer(config.tokenizer)
renderer = create_renderer(tokenizer, config.renderer)
assert renderer is not None
assert judge_url == "http://202.214.223.66:26406/v1"
assert user_url == judge_url
for base_url in set(config.model.client.base_url + [judge_url]):
    with urllib.request.urlopen(f"{base_url}/models", timeout=15) as response:
        assert response.status == 200

print(f"batch_size={config.batch_size} group_size={config.group_size}")
print(f"seq_len={config.seq_len} max_output={config.train.sampling.max_completion_tokens}")
print(f"lr={config.optim.lr} lora={config.model.lora}")
print(f"rollout_clients={len(config.model.client.base_url)} admin_clients={len(config.model.client.admin_base_url)}")
print(f"env_workers={config.train.env[0].pool.num_workers}")
print(f"renderer={type(renderer).__name__} tool_parser={config.renderer.tool_parser}")
print("policy_endpoints=200 judge_endpoint=200")
