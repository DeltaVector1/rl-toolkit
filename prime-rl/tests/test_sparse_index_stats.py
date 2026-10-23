import json

import torch

from prime_rl.trainer.models.glm_moe_dsa.sparse_index_stats import (
    record_sparse_index_tensor,
    summarize_sparse_indices,
)


def test_summary_excludes_sentinel_and_reports_route_distribution():
    indices = torch.tensor([[[[0, 0, 1, 3, 4, 4, 4, 9]]]], dtype=torch.int32)
    result = summarize_sparse_indices(indices, sequence_kv=9)

    assert result["entries"] == 8
    assert result["valid_entries"] == 7
    assert result["sentinel_entries"] == 0
    assert result["invalid_entries"] == 1
    assert result["unique_keys"] == 4
    assert result["routes_per_key"]["min"] == 1.0
    assert result["routes_per_key"]["max"] == 3.0


def test_summary_reports_expected_sentinel_separately():
    indices = torch.tensor([[[[0, 1, 8, 8, -1]]]], dtype=torch.int32)
    result = summarize_sparse_indices(indices, sequence_kv=9)

    assert result["valid_entries"] == 2
    assert result["sentinel_entries"] == 2
    assert result["invalid_entries"] == 1
    assert result["invalid_fraction"] == 0.2


def test_record_is_rank_scoped_and_limited(monkeypatch, tmp_path):
    output = tmp_path / "indices"
    module = __import__(
        "prime_rl.trainer.models.glm_moe_dsa.sparse_index_stats",
        fromlist=["_COUNTS", "_MAX_PER_LAYER", "_OUTPUT", "_RANK"],
    )
    monkeypatch.setattr(module, "_OUTPUT", str(output))
    monkeypatch.setattr(module, "_MAX_PER_LAYER", 1)
    monkeypatch.setattr(module, "_RANK", 3)
    module._COUNTS.clear()
    values = torch.tensor([[[[0, 1, 1, 9]]]], dtype=torch.int32)

    record_sparse_index_tensor(values, sequence_kv=9, layer_index=2, cp_rank=1, cp_world_size=8)
    record_sparse_index_tensor(values, sequence_kv=9, layer_index=2, cp_rank=1, cp_world_size=8)

    lines = (tmp_path / "indices.rank3.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["layer_index"] == 2
    assert record["cp_rank"] == 1
    assert record["sample_index"] == 0
