"""Load the published rows without changing their contents."""

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import verifiers.v1 as vf

DATASET_ID = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DATASET_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
FILES = {"music": "music.parquet"}


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
