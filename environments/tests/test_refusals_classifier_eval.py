from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


root = Path(__file__).resolve().parents[2]
environment_root = root / "environments" / "src"
evaluator_root = root / "evals" / "refusals-classifier-eval"
sys.path.insert(0, str(environment_root))
sys.path.insert(0, str(evaluator_root))

from refusals_classifier_eval_v1.client import classify_response  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("refusals_runner_test", evaluator_root / "runner.py")


class RefusalEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_failures_are_bounded_and_unscored(self):
        with (
            patch(
                "refusals_classifier_eval_v1.client._request",
                side_effect=OSError("down"),
            ),
            patch("refusals_classifier_eval_v1.client.asyncio.sleep", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "bounded retries"):
                await classify_response("http://service", "text", 3, 1, 5)

    def test_muse_parser_keeps_visible_channels_only(self):
        raw = (
            "to=self<|message|>reasoning trace<|eom|>"
            "to=user<|message|>visible answer<|eot|>"
        )
        self.assertEqual(runner.extract_muse_visible_text(raw), "visible answer")

    def test_summary_keeps_refusal_compliance_and_failure_counts_separate(self):
        records = [
            {
                "evaluation_status": "scored",
                "classifier_label": 0,
                "is_refusal": True,
                "classifier_truncated": False,
            },
            {
                "evaluation_status": "scored",
                "classifier_label": 1,
                "is_refusal": False,
                "classifier_truncated": False,
            },
            {"evaluation_status": "unscored_service_failure"},
        ]
        report = runner.summary(records, "model", "suite")
        self.assertEqual(report["scored_examples"], 2)
        self.assertEqual(report["refusal"]["rate"], 0.5)
        self.assertEqual(report["compliance"]["rate"], 0.5)
        self.assertEqual(report["statuses"]["unscored_service_failure"], 1)


if __name__ == "__main__":
    unittest.main()
