import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xiaomi_code import _data as code_data
from xiaomi_general import _data as general_data
from xiaomi_music import _data as music_data


@pytest.fixture(params=[code_data, general_data], ids=["code", "general"])
def data(request):
    return request.param


def test_rows_and_instance_preserve_original_text(tmp_path, data):
    instance = {"problem_statement": "原始任务\n", "docker_image": "legacy:latest"}
    row = {"prompt": [{"role": "user", "content": "short summary"}], "extra_info": {"instance_json": json.dumps(instance)}}
    path = tmp_path / "code.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    assert data.load_rows(next(iter(data.FILES)), data.XiaomiTasksetConfig(dataset_path=str(path))) == [row]
    assert data.load_instance(row) == instance


def test_mapping_is_pinned_and_unknown_images_fail(tmp_path, monkeypatch, data):
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
        general_data.download_general_task(path)


@pytest.mark.parametrize("targets", [
    ["docker.io/other/task:latest"],
    ["docker.io/xiaomimimo/mimo-v2.6-rl-oss:first", "docker.io/xiaomimimo/mimo-v2.6-rl-oss:second"],
])
def test_mapping_rejects_untrusted_or_conflicting_images(tmp_path, monkeypatch, data, targets):
    path = tmp_path / "mapping.jsonl"
    path.write_text("\n".join(json.dumps({"dataset_image": "legacy:latest", "dockerhub_image": target}) for target in targets))
    monkeypatch.setattr(data, "hf_hub_download", lambda *args, **kwargs: str(path))
    data._image_mapping.cache_clear()
    try:
        with pytest.raises(ValueError, match="Unexpected published image registry|Conflicting published image mapping"):
            data.resolve_image("legacy:latest")
    finally:
        data._image_mapping.cache_clear()


@pytest.mark.parametrize("data", [code_data, general_data, music_data], ids=["code", "general", "music"])
def test_each_loader_downloads_its_pinned_dataset(tmp_path, monkeypatch, data):
    row = {"prompt": [{"role": "user", "content": "original"}]}
    path = tmp_path / "row.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)

    monkeypatch.setattr(data, "hf_hub_download", download)
    domain, filename = next(iter(data.FILES.items()))
    assert data.load_rows(domain, data.XiaomiTasksetConfig()) == [row]
    assert calls == [(("XiaomiMiMo/MiMo-V2.6-RL-oss", filename), {
        "repo_type": "dataset", "revision": "639865fd3374018d6cb29b9fb82dd531406fcf5f"
    })]
