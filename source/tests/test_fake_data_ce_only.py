from types import SimpleNamespace
from unittest.mock import patch

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
