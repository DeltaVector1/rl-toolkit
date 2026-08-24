import base64
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from prime_rl.trainer.rl.data import FakeDataLoader


def test_fake_data_ce_only_has_no_rl_tokens() -> None:
    config = SimpleNamespace(
        batch_size=2,
        generate_samples=True,
        ce_only=True,
        reference_path=None,
    )
    world = SimpleNamespace(world_size=1, rank=0)
    runs = SimpleNamespace(max_runs=1)
    with (
        patch("prime_rl.trainer.rl.data.get_world", return_value=world),
        patch("prime_rl.trainer.rl.data.get_multi_run_manager", return_value=runs),
    ):
        loader = FakeDataLoader(config, seq_len=128, dp_world_size=1, dp_rank=0)
        batch = loader.get_batch()

    assert len(batch) == 2
    for microbatch in batch:
        assert torch.count_nonzero(microbatch["rl_weights"]) == 0
        assert torch.all(microbatch["ce_weights"] == 1)
        assert sum(microbatch["sequence_lengths"]) == 128


def _reference_config(path):
    return SimpleNamespace(
        batch_size=1, generate_samples=False, ce_only=False, reference_path=path
    )


def _reference_payload(routes: bytes) -> dict:
    return {
        "input_ids": [1, 2],
        "inference_logprobs": [-0.5, -0.25],
        "sequence_lengths": [2],
        "routed_experts": {
            "schema_version": 1,
            "encoding": "base64",
            "dtype": "uint8",
            "shape": [2, 2, 2],
            "dense_layers": 1,
            "experts": 4,
            "sha256": hashlib.sha256(routes).hexdigest(),
            "data": base64.b64encode(routes).decode("ascii"),
        },
    }


def test_fake_reference_loads_routed_experts(tmp_path) -> None:
    routes = bytes([0, 0, 1, 2, 0, 0, 2, 3])
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference_payload(routes)))
    world = SimpleNamespace(world_size=1, rank=0)
    runs = SimpleNamespace(max_runs=1)
    with (
        patch("prime_rl.trainer.rl.data.get_world", return_value=world),
        patch("prime_rl.trainer.rl.data.get_multi_run_manager", return_value=runs),
    ):
        microbatch = FakeDataLoader(
            _reference_config(path), seq_len=2, dp_world_size=1, dp_rank=0
        ).get_batch()[0]
    assert microbatch["routed_experts"].dtype == torch.uint8
    assert microbatch["routed_experts"].shape == (1, 2, 2, 2)
    assert microbatch["routed_experts"].tolist() == [
        [[[0, 0], [1, 2]], [[0, 0], [2, 3]]]
    ]


def test_fake_reference_rejects_bad_routed_experts(tmp_path) -> None:
    routes = bytes([0, 0, 1, 1, 0, 0, 2, 3])
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_reference_payload(routes)))
    world = SimpleNamespace(world_size=1, rank=0)
    runs = SimpleNamespace(max_runs=1)
    with (
        patch("prime_rl.trainer.rl.data.get_world", return_value=world),
        patch("prime_rl.trainer.rl.data.get_multi_run_manager", return_value=runs),
        pytest.raises(ValueError, match="sparse router IDs are invalid"),
    ):
        FakeDataLoader(_reference_config(path), seq_len=2, dp_world_size=1, dp_rank=0)


def test_normal_fake_data_still_has_no_routed_experts() -> None:
    config = SimpleNamespace(
        batch_size=1,
        generate_samples=False,
        ce_only=False,
        reference_path=None,
    )
    world = SimpleNamespace(world_size=1, rank=0)
    runs = SimpleNamespace(max_runs=1)
    with (
        patch("prime_rl.trainer.rl.data.get_world", return_value=world),
        patch("prime_rl.trainer.rl.data.get_multi_run_manager", return_value=runs),
    ):
        microbatch = FakeDataLoader(
            config, seq_len=8, dp_world_size=1, dp_rank=0
        ).get_batch()[0]
    assert microbatch["routed_experts"] is None


def test_grouped_fake_data_has_exact_samples_and_grpo_advantages() -> None:
    config = SimpleNamespace(
        batch_size=2,
        generate_samples=True,
        ce_only=False,
        reference_path=None,
        samples_per_micro_batch=4,
        group_size=2,
    )
    world = SimpleNamespace(world_size=1, rank=0)
    runs = SimpleNamespace(max_runs=1)
    with (
        patch("prime_rl.trainer.rl.data.get_world", return_value=world),
        patch("prime_rl.trainer.rl.data.get_multi_run_manager", return_value=runs),
    ):
        batch = FakeDataLoader(
            config, seq_len=128, dp_world_size=1, dp_rank=0
        ).get_batch()

    assert len(batch) == 2
    assert sum(len(item["sequence_lengths"]) for item in batch) == 8
    for microbatch in batch:
        lengths = microbatch["sequence_lengths"]
        assert len(lengths) == 4
        assert sum(lengths) == 128
        values = []
        start = 0
        for length in lengths:
            segment = microbatch["advantages"][0, start : start + length]
            assert torch.all(segment == segment[0])
            values.append(segment[0])
            start += length
        values = torch.stack(values)
        for group in values.split(2):
            assert torch.isclose(group.mean(), torch.tensor(0.0), atol=1e-6)
            assert torch.isclose(
                group.std(unbiased=False), torch.tensor(1.0), atol=1e-6
            )


def test_fake_data_rejects_batch_not_divisible_by_data_parallel_size() -> None:
    config = SimpleNamespace(
        batch_size=3,
        generate_samples=False,
        ce_only=False,
        reference_path=None,
    )
    world = SimpleNamespace(world_size=1, rank=0)
    runs = SimpleNamespace(max_runs=1)
    with (
        patch("prime_rl.trainer.rl.data.get_world", return_value=world),
        patch("prime_rl.trainer.rl.data.get_multi_run_manager", return_value=runs),
        pytest.raises(ValueError, match="not divisible"),
    ):
        FakeDataLoader(config, seq_len=8, dp_world_size=2, dp_rank=0)
