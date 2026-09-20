import ast
import hashlib
import json
import math
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


def _load_train_sink_functions(wanted, variables=()):
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/prime_rl/orchestrator/train_sink.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in variables for target in targets):
                body.append(node)
    namespace = {"Rollout": object, "math": math}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


def _load_refusal_group_function():
    wanted = {
        "_valid_refusal_score",
        "_refusal_score",
        "should_skip_strong_refusal_group",
        "_skip_strong_refusal_group",
        "_empty_refusal_advantage_counts",
        "_refusal_band",
        "_scalar_advantage",
        "refusal_advantage_counts",
    }
    namespace = _load_train_sink_functions(wanted, {"_refusal_bands", "_advantage_signs"})
    return (
        namespace["should_skip_strong_refusal_group"],
        namespace["_skip_strong_refusal_group"],
        namespace["refusal_advantage_counts"],
    )


def _load_refusal_metric_function():
    source_path = Path(__file__).resolve().parents[1] / "src/prime_rl/orchestrator/orchestrator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_refusal_advantage_metrics"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_refusal_advantage_metrics"]


def _load_usable_group_functions():
    wanted = {
        "_valid_refusal_score",
        "_refusal_score",
        "_decensor_details",
        "is_usable_response",
        "should_skip_group_without_usable_response",
        "_filter_group",
        "_scalar_advantage",
        "should_skip_positive_strong_refusal_advantage_group",
    }
    return _load_train_sink_functions(wanted, {"_usable_response_metrics"})


class StrongRefusalGroupSkipTest(unittest.TestCase):
    _missing = object()

    @staticmethod
    def _rollout(score=_missing):
        details = {} if score is StrongRefusalGroupSkipTest._missing else {"refusal_score": score}
        return SimpleNamespace(
            info={"decensor": details},
            samples=[object()],
            reward=0.25,
            has_error=False,
        )

    def test_strong_refusal_group_is_skipped_at_threshold(self):
        skip, _, _ = _load_refusal_group_function()
        self.assertTrue(skip([self._rollout(9), self._rollout(7)], skip_threshold=7))

    def test_mixed_refusal_group_is_unchanged(self):
        skip, _, _ = _load_refusal_group_function()
        self.assertFalse(skip([self._rollout(9), self._rollout(6)], skip_threshold=7))

    def test_invalid_refusal_scores_are_unchanged(self):
        skip, _, _ = _load_refusal_group_function()
        for value in (
            self._rollout(),
            self._rollout(None),
            self._rollout(True),
            self._rollout("9"),
            self._rollout(float("nan")),
            self._rollout(0),
            self._rollout(10),
            self._rollout(10**1000),
        ):
            with self.subTest(value=value):
                self.assertFalse(skip([self._rollout(9), value], skip_threshold=7))

    def test_default_off_is_unchanged(self):
        skip, _, _ = _load_refusal_group_function()
        self.assertFalse(skip([self._rollout(9), self._rollout(9)], skip_threshold=None))

    def test_service_error_is_not_reclassified_as_refusal_skip(self):
        skip, _, _ = _load_refusal_group_function()
        errored = self._rollout(9)
        errored.has_error = True
        self.assertFalse(skip([errored, self._rollout(9)], skip_threshold=7))

    def test_skip_removes_effective_samples_but_keeps_observation(self):
        _, mark_skip, _ = _load_refusal_group_function()
        group = [self._rollout(9), self._rollout(9)]
        infos = [rollout.info for rollout in group]
        rewards = [rollout.reward for rollout in group]
        mark_skip(group)
        self.assertEqual([rollout.samples for rollout in group], [[], []])
        self.assertTrue(all(rollout.is_filtered for rollout in group))
        self.assertTrue(all(rollout.filter_results == {"strong_refusal_group": True} for rollout in group))
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
            and node.target.id == "skip_if_every_refusal_score_is_at_least"
        )
        self.assertIsInstance(field.value, ast.Call)
        self.assertIsNone(field.value.args[0].value)

    def test_refusal_band_advantage_counts(self):
        _, _, counts = _load_refusal_group_function()

        def rollout(score, advantage):
            return SimpleNamespace(
                info={"decensor": {"refusal_score": score}},
                scalar_advantage=lambda: advantage,
            )

        self.assertEqual(
            counts(
                [
                    rollout(1, 1.0),
                    rollout(2, -1.0),
                    rollout(3, 0.0),
                    rollout(6, -0.5),
                    rollout(7, 0.5),
                    rollout(9, 0.0),
                    rollout(0, 1.0),
                    rollout(10, 1.0),
                    rollout(10**1000, 1.0),
                ]
            ),
            {
                "1-2": {"positive": 1, "negative": 1, "zero": 0},
                "3-6": {"positive": 0, "negative": 1, "zero": 1},
                "7-9": {"positive": 1, "negative": 0, "zero": 1},
            },
        )

    def test_refusal_advantage_counts_are_emitted_as_rollout_metrics(self):
        metrics = _load_refusal_metric_function()(
            {"1-2": {"positive": 1, "negative": 2, "zero": 3}}
        )
        self.assertEqual(
            metrics,
            {
                "rollouts/refusal_advantage/1-2/positive_count": 1.0,
                "rollouts/refusal_advantage/1-2/negative_count": 2.0,
                "rollouts/refusal_advantage/1-2/zero_count": 3.0,
            },
        )


class UsableResponseGroupTest(unittest.TestCase):
    @staticmethod
    def _rollout(**overrides):
        details = {
            "response_rejected_because": None,
            "refusal_score": 1,
            "visible_answer_is_clear": 1,
            "reasoning_trace_is_clear": 1,
            "reasoning_has_safety_or_policy_discussion": 0,
            "meta_commentary": 0,
        }
        details.update(overrides)
        advantage = details.pop("advantage", 0.0)
        return SimpleNamespace(
            info={"decensor": details},
            samples=[object()],
            reward=0.25,
            has_error=False,
            scalar_advantage=lambda: advantage,
        )

    def test_disabled_mode_does_not_read_decensor_metrics(self):
        functions = _load_usable_group_functions()
        rollout = SimpleNamespace(has_error=False)
        self.assertFalse(functions["should_skip_group_without_usable_response"]([rollout], False))
        self.assertFalse(
            functions["should_skip_positive_strong_refusal_advantage_group"]([rollout], False)
        )

    def test_usable_requirement_and_strict_metrics(self):
        functions = _load_usable_group_functions()
        skip = functions["should_skip_group_without_usable_response"]
        self.assertFalse(skip([self._rollout(refusal_score=9), self._rollout(refusal_score=4)], True))
        self.assertTrue(skip([self._rollout(refusal_score=5), self._rollout(meta_commentary=1)], True))
        missing = self._rollout()
        del missing.info["decensor"]["reasoning_trace_is_clear"]
        with self.assertRaisesRegex(RuntimeError, "reasoning_trace_is_clear"):
            skip([self._rollout(), missing], True)
        self.assertTrue(
            skip([self._rollout(response_rejected_because="truncated")], True)
        )

    def test_positive_strong_refusal_advantage_stage(self):
        skip = _load_usable_group_functions()["should_skip_positive_strong_refusal_advantage_group"]
        positive = [
            self._rollout(refusal_score=1, advantage=-0.5),
            self._rollout(refusal_score=7, advantage=0.25),
        ]
        self.assertTrue(skip(positive, True))
        self.assertFalse(skip([self._rollout(refusal_score=9, advantage=0.0)], True))

    def test_filter_keeps_observations_and_rewards(self):
        functions = _load_usable_group_functions()
        for reason in ("no_usable_response_group", "positive_strong_refusal_advantage_group"):
            with self.subTest(reason=reason):
                group = [self._rollout(), self._rollout()]
                infos = [rollout.info for rollout in group]
                rewards = [rollout.reward for rollout in group]
                functions["_filter_group"](group, reason)
                self.assertEqual([rollout.samples for rollout in group], [[], []])
                self.assertTrue(all(rollout.is_filtered for rollout in group))
                self.assertTrue(all(rollout.filter_results == {reason: True} for rollout in group))
                self.assertEqual([rollout.info for rollout in group], infos)
                self.assertEqual([rollout.reward for rollout in group], rewards)

    def test_default_metrics_and_order(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py"
        )
        tree = ast.parse(config_path.read_text(encoding="utf-8"))
        field = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "require_usable_response_in_group"
        )
        self.assertIsInstance(field.value, ast.Constant)
        self.assertIs(field.value.value, False)
        source = (
            Path(__file__).resolve().parents[1]
            / "src/prime_rl/orchestrator/orchestrator.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"groups/no_usable_response/skipped"', source)
        self.assertIn('"groups/positive_strong_refusal_advantage/skipped"', source)

        sink_source = (
            Path(__file__).resolve().parents[1]
            / "src/prime_rl/orchestrator/train_sink.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.no_usable_response_groups_skipped += 1", sink_source)
        self.assertIn("self.positive_strong_refusal_advantage_groups_skipped += 1", sink_source)
        finalize = sink_source.index("await env.algorithm.finalize_group(survivors)")
        positive_filter = sink_source.index(
            "if should_skip_positive_strong_refusal_advantage_group(", finalize
        )
        refusal_metrics = sink_source.index(
            "for band, sign_counts in refusal_advantage_counts(survivors).items():",
            positive_filter,
        )
        zero_advantage = sink_source.index("if not any(r.is_trainable", refusal_metrics)
        self.assertTrue(finalize < positive_filter < refusal_metrics < zero_advantage)


if __name__ == "__main__":
    unittest.main()
