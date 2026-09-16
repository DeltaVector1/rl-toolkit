import ast
import hashlib
import json
import os
import socket
import unittest
from pathlib import Path
from types import SimpleNamespace


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


def _load_refusal_group_function():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/prime_rl/orchestrator/train_sink.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"_refusal_score", "should_skip_all_refusal_group", "_skip_refusal_group"}
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"Rollout": object}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["should_skip_all_refusal_group"], namespace["_skip_refusal_group"]


class RefusalGroupSkipTest(unittest.TestCase):
    _missing = object()

    @staticmethod
    def _rollout(score=_missing):
        details = {} if score is RefusalGroupSkipTest._missing else {"refusal_score": score}
        return SimpleNamespace(
            info={"decensor": details},
            samples=[object()],
            reward=0.25,
            has_error=False,
        )

    def test_all_refusal_group_is_skipped_when_enabled(self):
        skip, _ = _load_refusal_group_function()
        self.assertTrue(skip([self._rollout(9), self._rollout(9)], enabled=True))

    def test_mixed_refusal_group_is_unchanged(self):
        skip, _ = _load_refusal_group_function()
        self.assertFalse(skip([self._rollout(9), self._rollout(8)], enabled=True))

    def test_invalid_refusal_scores_are_unchanged(self):
        skip, _ = _load_refusal_group_function()
        for value in (self._rollout(), self._rollout(None), self._rollout(True), self._rollout("9"), self._rollout(float("nan"))):
            with self.subTest(value=value):
                self.assertFalse(skip([self._rollout(9), value], enabled=True))

    def test_default_off_is_unchanged(self):
        skip, _ = _load_refusal_group_function()
        self.assertFalse(skip([self._rollout(9), self._rollout(9)], enabled=False))

    def test_service_error_is_not_reclassified_as_refusal_skip(self):
        skip, _ = _load_refusal_group_function()
        errored = self._rollout(9)
        errored.has_error = True
        self.assertFalse(skip([errored, self._rollout(9)], enabled=True))

    def test_skip_removes_effective_samples_but_keeps_observation(self):
        _, mark_skip = _load_refusal_group_function()
        group = [self._rollout(9), self._rollout(9)]
        infos = [rollout.info for rollout in group]
        rewards = [rollout.reward for rollout in group]
        mark_skip(group)
        self.assertEqual([rollout.samples for rollout in group], [[], []])
        self.assertTrue(all(rollout.is_filtered for rollout in group))
        self.assertEqual([rollout.info for rollout in group], infos)
        self.assertEqual([rollout.reward for rollout in group], rewards)

    def test_config_default_is_off(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        field = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "skip_all_refusal_groups"
        )
        self.assertIsInstance(field.value, ast.Constant)
        self.assertFalse(field.value.value)


if __name__ == "__main__":
    unittest.main()
