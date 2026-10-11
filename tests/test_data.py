import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xiaomi_common import data


def test_rows_and_instance_preserve_original_text(tmp_path):
    instance = {"problem_statement": "原始任务\n", "docker_image": "legacy:latest"}
    row = {"prompt": [{"role": "user", "content": "short summary"}], "extra_info": {"instance_json": json.dumps(instance)}}
    path = tmp_path / "code.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    assert data.load_rows("code", data.XiaomiTasksetConfig(dataset_path=str(path))) == [row]
    assert data.load_instance(row) == instance


def test_mapping_is_pinned_and_unknown_images_fail(tmp_path, monkeypatch):
    path = tmp_path / "mapping.jsonl"
    path.write_text(json.dumps({"dataset_image": "legacy:latest", "dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:task"}) + "\n")
    calls = []
    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)
    monkeypatch.setattr(data, "hf_hub_download", download)
    data._image_mapping.cache_clear()
    try:
        assert data.resolve_image("legacy:latest").endswith(":task")
        assert calls[0][1]["revision"] == data.DATASET_REVISION
        with pytest.raises(ValueError, match="absent"):
            data.resolve_image("invented:latest")
    finally:
        data._image_mapping.cache_clear()


@pytest.mark.parametrize("path", ["/tmp/task", "envs/../secret", "envs/task/nested", "envs"])
def test_general_assets_cannot_escape(path):
    with pytest.raises(ValueError):
        data.download_general_task(path)
