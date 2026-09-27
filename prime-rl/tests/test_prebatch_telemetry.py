import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_progress_class():
    path = ROOT / "src/prime_rl/orchestrator/train_sink.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.ClassDef))
        and (
            isinstance(node, ast.ClassDef)
            and node.name == "PreBatchProgress"
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_PRE_BATCH_REJECTION_REASONS"
                for target in node.targets
            )
        )
    ]
    namespace = {
        "dataclass": __import__("dataclasses").dataclass,
        "field": __import__("dataclasses").field,
    }
    exec(
        compile(
            ast.Module(
                body=[ast.parse("from __future__ import annotations").body[0], *selected],
                type_ignores=[],
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["PreBatchProgress"]


class PreBatchSinkTelemetryTest(unittest.TestCase):
    def test_periodic_pipeline_view_contains_numeric_marker(self):
        path = ROOT / "src/prime_rl/orchestrator/orchestrator.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("PIPELINE_TELEMETRY", source)
        self.assertIn("pre_batch_telemetry_fields", source)

    def test_process_group_records_each_terminal_group_outcome(self):
        path = ROOT / "src/prime_rl/orchestrator/train_sink.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        process_group = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "process_group"
        )
        source = ast.unparse(process_group)
        self.assertGreaterEqual(source.count("self._record_group"), 5)
        for reason in (
            "stale_policy",
            "errored_group",
            "trainer_seq_len_group",
            "untrainable_group",
            "zero_advantage_group",
        ):
            self.assertIn(reason, source)

    def test_group_counters_match_empty_metric_snapshot(self):
        progress = load_progress_class()()
        for _ in range(26):
            progress.record_group([object() for _ in range(16)])
        for _ in range(17):
            progress.record_group(
                [object() for _ in range(16)], rejection_reason="errored_group"
            )
        progress.policy_invalid_rollouts = 70

        fields = progress.log_fields(
            accepted_buffer=416,
            accepted_target=1024,
            accepted_unit="rollouts",
            accepted_buffer_tokens=0,
            partial_groups=16,
            partial_rollouts=161,
            oldest_partial_age_seconds=219.72,
        )
        self.assertIn("complete_groups=43", fields)
        self.assertIn("clean_groups=26", fields)
        self.assertIn("rejected_groups=17", fields)
        self.assertIn("accepted_buffer=416", fields)
        self.assertIn("partial_rollouts=161", fields)
        self.assertIn("policy_invalid_rollouts=70", fields)
        self.assertIn("reject_errored_group=17", fields)
        self.assertNotIn("prompt", fields.lower())
        self.assertNotIn("response", fields.lower())


if __name__ == "__main__":
    unittest.main()
