import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from verifiers.v1.harnesses.null.harness import NullHarness
from verifiers.v1.utils.loaders import default_harness_id, harness_class, taskset_class
from xiaomi_music.scorer import pipeline
from xiaomi_music.scorer.feats import analyze
from xiaomi_music.scorer.score import score
from xiaomi_music.taskset import MusicTaskset
from xiaomi_music._data import XiaomiTasksetConfig

ABC = "X:1\nT:t\nM:4/4\nL:1/8\nK:C\nCDEF|GABc|c4|z4|"


def test_native_package_and_text_only_default():
    assert taskset_class("xiaomi-music") is MusicTaskset
    assert harness_class(default_harness_id("xiaomi-music")) is NullHarness


def test_abc_extraction_matches_source_order():
    assert pipeline.extract_abc(f"intro\n```abc\n{ABC}\n```\nX:2") == ABC
    assert pipeline.extract_abc("X:1\nC\nX:2\nD") == "X:2\nD"
    assert pipeline.compute_score("music", "no ABC") == 0


def test_missing_converter_is_infrastructure_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(pipeline.sp, "run", missing)
    with pytest.raises(pipeline.Abc2MidiMissing):
        pipeline.compute_score("music", ABC)


@pytest.mark.parametrize(
    "record",
    [
        {"skip": "no_midi"},
        {"total": 99, "reject": 1},
        {"scorer_skip": "no_notes"},
        {"total": 42.1},
    ],
)
def test_reward_gates(monkeypatch, record):
    monkeypatch.setattr(pipeline, "do", lambda _: record)
    expected = 0.421 if record == {"total": 42.1} else 0
    assert pipeline.compute_score("music", ABC) == pytest.approx(expected)


def midi():
    track = b"\x00\xc0\x00"
    for pitch in [60, 62, 64, 65, 67, 69, 71, 72] * 2:
        track += bytes([0, 0x90, pitch, 80, 96, 0x80, pitch, 0])
    track += b"\x00\xff\x2f\x00"
    return (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 96)
        + b"MTrk"
        + struct.pack(">I", len(track))
        + track
    )


def test_full_feature_pipeline_frozen_result(tmp_path, monkeypatch):
    path = tmp_path / "scale.mid"
    path.write_bytes(midi())
    result = score(analyze(path), pipeline._get_ref())
    assert result["total"] == 19.3

    def convert(argv, **kwargs):
        Path(argv[-1]).write_bytes(midi())
        return SimpleNamespace(stdout=b"", stderr=b"")

    monkeypatch.setattr(pipeline.sp, "run", convert)
    assert pipeline.compute_score("music", ABC) == 0.193
    assert pipeline.compute_score("music", ABC.replace("K:C", "\nK:C")) == 0


def test_music_loads_local_parquet_with_own_config(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    row = {"prompt": [{"role": "user", "content": "Compose 原始 music"}], "extra_info": {"src_id": "published-music-id"}}
    path = tmp_path / "music.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    task, = MusicTaskset(XiaomiTasksetConfig(dataset_path=str(path))).load()
    assert task.data.id == "published-music-id"
    assert [message.model_dump() for message in task.data.prompt] == row["prompt"]
    assert task.data.extra_info == row["extra_info"]
    assert task.data.image == "python:3.11-slim"
