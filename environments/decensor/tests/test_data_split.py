from collections import Counter

import pytest

from decensor_v1.data import build_decensor_pool
from decensor_v1.taskset import DecensorTasksetConfig


def _rows():
    for harm_type in ("harm-a", "harm-b"):
        for source_index in range(8):
            yield {
                "source_index": source_index + (100 if harm_type == "harm-b" else 0),
                "prompt": [{"role": "user", "content": f"prompt-{harm_type}-{source_index}"}],
                "primary_harm_type_code": harm_type,
                "bucket_ID": "easy",
            }


def _pool(monkeypatch, target, data_split, holdout_fraction=0.25):
    monkeypatch.setattr(
        "decensor_v1.data.iter_decensor_rows", lambda difficulty: iter(_rows())
    )
    return build_decensor_pool(
        target=target,
        decay=0.5,
        max_turns=2,
        seed=11,
        difficulty="easy",
        data_split=data_split,
        holdout_seed=23,
        holdout_fraction=holdout_fraction,
    )


def _source_indices(pool):
    return [row[3] for row in pool]


def test_heldout_is_deterministic_balanced_and_unique(monkeypatch):
    first = _pool(monkeypatch, 4, "heldout")
    second = _pool(monkeypatch, 4, "heldout")

    assert first == second
    assert len(set(_source_indices(first))) == len(first)
    assert Counter(row[2] for row in first) == {"harm-a": 2, "harm-b": 2}


def test_train_and_heldout_are_disjoint(monkeypatch):
    heldout = _pool(monkeypatch, 4, "heldout")
    train = _pool(monkeypatch, 20, "train")

    assert set(_source_indices(heldout)).isdisjoint(_source_indices(train))


def test_heldout_rejects_oversize_request(monkeypatch):
    with pytest.raises(ValueError, match="unique rows"):
        _pool(monkeypatch, 5, "heldout")


def test_taskset_exposes_split_fields():
    config = DecensorTasksetConfig()
    assert config.data_split == "all"
    assert config.holdout_seed == 42
    assert config.holdout_fraction == 0.25


def test_split_and_fraction_are_validated(monkeypatch):
    with pytest.raises(ValueError, match="holdout_fraction"):
        _pool(monkeypatch, 1, "heldout", 0.0)
    with pytest.raises(ValueError, match="holdout_fraction"):
        build_decensor_pool(1, 0.5, 2, 11, "easy", "all", 23, 1.0)
    with pytest.raises(ValueError, match="data_split"):
        build_decensor_pool(1, 0.5, 2, 11, "easy", "invalid", 23, 0.25)
    with pytest.raises(ValueError):
        DecensorTasksetConfig(data_split="invalid")
    with pytest.raises(ValueError):
        DecensorTasksetConfig(holdout_fraction=1.0)
