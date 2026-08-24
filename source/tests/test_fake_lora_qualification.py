from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prime_rl.trainer.rl.train import (
    _validate_resume_adapter_inventory,
    register_fake_lora_run,
)


def test_register_fake_lora_run_does_not_need_orchestrator() -> None:
    manager = SimpleNamespace(
        used_idxs=[],
        idx_2_id={},
        id_2_idx={},
        unused_idxs={0},
        progress={},
        scaling_factors=[0.0],
        reset_run_parameters=Mock(),
    )
    lora = SimpleNamespace(rank=128, alpha=256.0)

    register_fake_lora_run(manager, lora)

    assert manager.idx_2_id == {0: "fake_qualification"}
    assert manager.id_2_idx == {"fake_qualification": 0}
    assert manager.scaling_factors[0] == 2.0
    manager.reset_run_parameters.assert_called_once_with(0)


def test_resume_adapter_inventory_requires_complete_distributed_union() -> None:
    expected = [
        {"layer.0.lora_A.weight": (4, 8)},
        {"layer.1.lora_B.weight": (8, 4)},
        {"layer.0.lora_A.weight": (4, 8)},
    ]
    adapter = {
        "layer.0.lora_A.weight": (4, 8),
        "layer.1.lora_B.weight": (8, 4),
    }
    _validate_resume_adapter_inventory(adapter, expected)

    with pytest.raises(ValueError, match="missing=.*layer.1"):
        _validate_resume_adapter_inventory(
            {"layer.0.lora_A.weight": (4, 8)}, expected
        )
    with pytest.raises(ValueError, match="unexpected=.*extra"):
        _validate_resume_adapter_inventory(
            {**adapter, "extra": (1,)}, expected
        )
    with pytest.raises(ValueError, match="shape_mismatch=.*layer.0"):
        _validate_resume_adapter_inventory(
            {**adapter, "layer.0.lora_A.weight": (8, 4)}, expected
        )
