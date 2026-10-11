"""Load the published rows and image mapping without changing their contents."""

import json
from functools import lru_cache
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

import verifiers.v1 as vf

DATASET_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATASET_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
FILES = {"code": "code.parquet", "general": "general/train.parquet", "music": "music.parquet"}


class XiaomiTasksetConfig(vf.TasksetConfig):
    dataset_path: str | None = None
    """Optional local parquet with the same published row schema."""
    revision: str = DATASET_REVISION


def load_rows(domain: str, config: XiaomiTasksetConfig) -> list[dict]:
    filename = FILES[domain]
    path = config.dataset_path or hf_hub_download(
        DATASET_ID, filename, repo_type="dataset", revision=config.revision
    )
    return pq.read_table(path).to_pylist()


def load_instance(row: dict) -> dict:
    instance = json.loads(row["extra_info"]["instance_json"])
    if not isinstance(instance, dict):
        raise ValueError("instance_json must encode an object")
    return instance


@lru_cache(maxsize=4)
def _image_mapping(revision: str) -> dict[str, str]:
    path = hf_hub_download(DATASET_ID, "image-mapping.jsonl", repo_type="dataset", revision=revision)
    mapping = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        source, target = row["dataset_image"], row["dockerhub_image"]
        if source in mapping and mapping[source] != target:
            raise ValueError(f"Conflicting published image mapping for {source!r}")
        if not target.startswith("docker.io/xiaomimimo/mimo-v2.6-rl-oss:"):
            raise ValueError(f"Unexpected published image registry: {target!r}")
        mapping[source] = target
    return mapping


def resolve_image(dataset_image: str, revision: str = DATASET_REVISION) -> str:
    try:
        return _image_mapping(revision)[dataset_image]
    except KeyError as exc:
        raise ValueError(f"Image absent from published image mapping: {dataset_image!r}") from exc


def download_general_task(env_task_dir: str, revision: str = DATASET_REVISION) -> Path:
    path = PurePosixPath(env_task_dir)
    if len(path.parts) != 2 or path.parts[0] != "envs" or path.parts[1] in {".", ".."}:
        raise ValueError(f"Expected envs/<task-id>, got {env_task_dir!r}")
    prefix = f"general/{path.as_posix()}"
    root = snapshot_download(DATASET_ID, repo_type="dataset", revision=revision, allow_patterns=[f"{prefix}/**"])
    task = Path(root) / prefix
    if not (task / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing task manifest: {task}")
    return task
