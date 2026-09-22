import ast
import hashlib
import json
import os
import socket
import unittest
from pathlib import Path


class _FakeTensor:
    pass


class _FakeTorch:
    Tensor = _FakeTensor


class _FakeDist:
    def __init__(self):
        self.world_records = []
        self.group_records = []

    def get_rank(self):
        return 0

    def get_world_size(self, group=None):
        return len(self.group_records if group is not None else self.world_records)

    def all_gather_object(self, output, local, group=None):
        records = self.group_records if group is not None else self.world_records
        output[:] = records


def _load_identity_functions():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/prime_rl/trainer/rl/train.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {
        "_BATCH_IDENTITY_FIELDS",
        "_update_batch_identity",
        "_micro_batch_identity",
        "_batch_identity",
        "_batch_identity_record",
        "_format_batch_identity_records",
        "verify_pipeline_batch_identity",
    }
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_BATCH_IDENTITY_FIELDS"
            for target in node.targets
        ):
            body.append(node)
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "socket": socket,
        "torch": _FakeTorch,
        "dist": _FakeDist(),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


class PipelineBatchIdentityTest(unittest.TestCase):
    def test_count_mismatch_reports_exact_rank_host_hash_and_count(self):
        namespace = _load_identity_functions()
        dist = namespace["dist"]
        micro_batches = [{"input_ids": [1], "sequence_lengths": [1]}]
        local = namespace["_batch_identity_record"](micro_batches)
        dist.world_records = [
            local,
            {
                "rank": 7,
                "host": "n85",
                "microbatch_count": 58,
                "batch_sha256": "f0c8" + "0" * 60,
                "microbatch_sha256": [],
                "error": None,
            },
        ]
        dist.group_records = dist.world_records

        with self.assertRaisesRegex(
            RuntimeError,
            r"rank=7 host=n85 microbatches=58 sha256=f0c8",
        ):
            namespace["verify_pipeline_batch_identity"](
                micro_batches,
                object(),
                compare_all_ranks=True,
            )

    def test_data_parallel_hashes_can_differ_but_pipeline_group_must_match(self):
        namespace = _load_identity_functions()
        dist = namespace["dist"]
        micro_batches = [{"input_ids": [1], "sequence_lengths": [1]}]
        local = namespace["_batch_identity_record"](micro_batches)
        other_data_rank = dict(local, rank=8, host="n76", batch_sha256="different")
        dist.world_records = [local, other_data_rank]
        dist.group_records = [local, dict(local, rank=1, host="n77")]

        namespace["verify_pipeline_batch_identity"](
            micro_batches,
            object(),
            compare_all_ranks=False,
        )


if __name__ == "__main__":
    unittest.main()
