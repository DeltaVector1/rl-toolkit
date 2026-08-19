from types import SimpleNamespace
from unittest.mock import Mock

from prime_rl.trainer.rl.train import register_fake_lora_run


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
