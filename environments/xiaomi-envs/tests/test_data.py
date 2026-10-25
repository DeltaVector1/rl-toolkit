import gzip
import hashlib
import json
from pathlib import Path

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


@pytest.mark.parametrize(("image", "target", "digest"), [
    (
        "format-code-task-002666:latest",
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-002666",
        "sha256:00b7faa38daee51df4bcc5c0decba8f183b9aff80bd5d6ca6d7ddf870f331848",
    ),
    (
        "format-code-task-000121:latest",
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000121",
        "sha256:85e99dbe4a8e4136fd93050e6c3a18b1decd7806bf1c55bec1d73152919e5ea8",
    ),
    (
        "format-code-task-001554:latest",
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001554",
        "sha256:98459cbc783284c9d7dc89854f5b1ce7393169528ae98da127590e357b52ca9e",
    ),
])
def test_published_code_images_use_verified_digests(
    tmp_path, monkeypatch, image, target, digest
):
    path = tmp_path / "mapping.jsonl"
    path.write_text(json.dumps({"dataset_image": image, "dockerhub_image": target}) + "\n")
    monkeypatch.setattr(code_data, "hf_hub_download", lambda *args, **kwargs: str(path))
    code_data._image_mapping.cache_clear()
    try:
        assert code_data.resolve_image(image) == f"docker.io/xiaomimimo/mimo-v2.6-rl-oss@{digest}"
    finally:
        code_data._image_mapping.cache_clear()


def test_all_published_images_use_only_verified_digest_pins(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "image-mapping-639865fd.jsonl.gz"
    mapping_bytes = gzip.decompress(fixture.read_bytes())
    assert hashlib.sha256(mapping_bytes).hexdigest() == (
        "704ad2716d746430a805b9b1ac7c6725aea2117fd8e4823ebe9d339d0313ad58"
    )

    pin_table = "".join(
        "\t".join((
            image,
            f"docker.io/xiaomimimo/mimo-v2.6-rl-oss:{image.removesuffix(':latest')}",
            digest,
        )) + "\n"
        for image, digest in sorted(code_data._code_image_digests.items())
    ).encode()
    assert len(code_data._code_image_digests) == 96
    assert hashlib.sha256(pin_table).hexdigest() == (
        "559b1025e00eb0c3804bc5b5fa4ba85617806849be114423d0a0d58692e55958"
    )

    mapping_path = tmp_path / "image-mapping.jsonl"
    mapping_path.write_bytes(mapping_bytes)
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(mapping_path)

    monkeypatch.setattr(code_data, "hf_hub_download", download)
    code_data._image_mapping.cache_clear()
    try:
        mapping = code_data._image_mapping(code_data.DATASET_REVISION)
        assert len(mapping) == 3764
        matched_pins = set()
        for image, target in mapping.items():
            digest = code_data._code_image_digests.get(image)
            expected_target = (
                "docker.io/xiaomimimo/mimo-v2.6-rl-oss:"
                f"{image.removesuffix(':latest')}"
            )
            if digest is not None and target == expected_target:
                matched_pins.add(image)
                expected = f"docker.io/xiaomimimo/mimo-v2.6-rl-oss@{digest}"
            else:
                expected = target
            assert code_data.resolve_image(image) == expected, image
            assert code_data.resolve_image(image, "other-revision") == target, image

        assert matched_pins == set(code_data._code_image_digests)
        assert [kwargs["revision"] for _, kwargs in calls] == [
            code_data.DATASET_REVISION,
            "other-revision",
        ]
    finally:
        code_data._image_mapping.cache_clear()


@pytest.mark.parametrize(("image", "target", "revision"), [
    (
        "format-code-task-002666:latest",
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-002666",
        "other-revision",
    ),
    (
        "format-code-task-002666:latest",
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:replacement",
        code_data.DATASET_REVISION,
    ),
    (
        "ordinary-code-task",
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:ordinary-code-task",
        code_data.DATASET_REVISION,
    ),
])
def test_other_revisions_and_mapping_targets_keep_published_tag(
    tmp_path, monkeypatch, image, target, revision
):
    path = tmp_path / "mapping.jsonl"
    path.write_text(json.dumps({"dataset_image": image, "dockerhub_image": target}) + "\n")
    monkeypatch.setattr(code_data, "hf_hub_download", lambda *args, **kwargs: str(path))
    code_data._image_mapping.cache_clear()
    try:
        assert code_data.resolve_image(image, revision) == target
    finally:
        code_data._image_mapping.cache_clear()


def test_pinned_code_image_still_requires_published_mapping(tmp_path, monkeypatch):
    path = tmp_path / "mapping.jsonl"
    path.write_text("")
    monkeypatch.setattr(code_data, "hf_hub_download", lambda *args, **kwargs: str(path))
    code_data._image_mapping.cache_clear()
    try:
        with pytest.raises(ValueError, match="absent"):
            code_data.resolve_image("format-code-task-002666:latest")
    finally:
        code_data._image_mapping.cache_clear()


def test_pinned_code_image_still_validates_published_mapping(tmp_path, monkeypatch):
    path = tmp_path / "mapping.jsonl"
    path.write_text(json.dumps({
        "dataset_image": "format-code-task-002666:latest",
        "dockerhub_image": "docker.io/other/repository:format-code-task-002666",
    }) + "\n")
    monkeypatch.setattr(code_data, "hf_hub_download", lambda *args, **kwargs: str(path))
    code_data._image_mapping.cache_clear()
    try:
        with pytest.raises(ValueError, match="Unexpected published image registry"):
            code_data.resolve_image("format-code-task-002666:latest")
    finally:
        code_data._image_mapping.cache_clear()


def test_pinned_code_image_rejects_conflicting_published_mapping(tmp_path, monkeypatch):
    image = "format-code-task-000014:latest"
    path = tmp_path / "mapping.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {
            "dataset_image": image,
            "dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-000014",
        },
        {
            "dataset_image": image,
            "dockerhub_image": "docker.io/xiaomimimo/mimo-v2.6-rl-oss:replacement",
        },
    ]) + "\n")
    monkeypatch.setattr(code_data, "hf_hub_download", lambda *args, **kwargs: str(path))
    code_data._image_mapping.cache_clear()
    try:
        with pytest.raises(ValueError, match="Conflicting published image mapping"):
            code_data.resolve_image(image)
    finally:
        code_data._image_mapping.cache_clear()


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
