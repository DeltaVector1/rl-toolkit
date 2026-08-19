from types import SimpleNamespace

import pytest
import torch

from prime_rl.trainer.rl.train import initialize_fresh_lora_policy


class Manager:
    def __init__(self, parameters, scale=2.0):
        self._parameters = parameters
        self.scaling_factors = [scale]

    def get_named_parameters_for_run(self, index):
        assert index == 0
        return iter(self._parameters)


def test_fresh_lora_policy_is_exactly_base_model(monkeypatch) -> None:
    a = torch.tensor([1.0, -2.0])
    b0 = torch.tensor([3.0, -4.0])
    b1 = torch.tensor([5.0])
    manager = Manager(
        [
            ("layer.lora_A.0", a),
            ("layer.lora_B.0", b0),
            ("other.lora_B.0", b1),
        ]
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    count, reset_shards = initialize_fresh_lora_policy(
        manager, SimpleNamespace(rank=128, alpha=256.0), "cpu"
    )

    assert count == 2
    assert reset_shards == 2
    assert torch.equal(a, torch.tensor([1.0, -2.0]))
    assert torch.count_nonzero(b0) == 0
    assert torch.count_nonzero(b1) == 0


@pytest.mark.parametrize(
    ("parameters", "scale"),
    [
        (
            [
                ("layer.lora_A.0", torch.tensor([float("nan")])),
                ("layer.lora_B.0", torch.ones(1)),
            ],
            2.0,
        ),
        (
            [
                ("layer.lora_A.0", torch.ones(1)),
                ("layer.lora_B.0", torch.ones(1)),
            ],
            1.0,
        ),
    ],
)
def test_fresh_lora_policy_fails_closed(monkeypatch, parameters, scale) -> None:
    manager = Manager(parameters, scale)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="initialization failed"):
        initialize_fresh_lora_policy(
            manager, SimpleNamespace(rank=128, alpha=256.0), "cpu"
        )
