import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class ActiveRewardRowsTest(unittest.TestCase):
    def test_selects_rows_after_active_container_start(self):
        rows = [
            {"step": 1, "time": 90, "eff": 0.1},
            {"step": 1, "time": 110, "eff": 0.2},
            {"step": 2, "time": 120, "eff": 0.3},
        ]

        selected, excluded, without_time = training_ops._select_active_reward_rows(
            rows, "x85", "x85", 100
        )

        self.assertEqual([row["eff"] for row in selected], [0.2, 0.3])
        self.assertEqual(excluded, 1)
        self.assertEqual(without_time, 0)

    def test_does_not_drop_rows_without_matching_active_run(self):
        rows = [{"step": 1, "eff": 0.1}]

        selected, excluded, without_time = training_ops._select_active_reward_rows(
            rows, "x22", "x85", 100
        )

        self.assertIs(selected, rows)
        self.assertEqual(excluded, 0)
        self.assertEqual(without_time, 0)

    def test_reports_rows_without_timestamps_when_filtering(self):
        rows = [{"step": 1, "eff": 0.1}, {"step": 2, "time": 110, "eff": 0.2}]

        selected, excluded, without_time = training_ops._select_active_reward_rows(
            rows, "x85", "x85", 100
        )

        self.assertEqual([row["step"] for row in selected], [2])
        self.assertEqual(excluded, 0)
        self.assertEqual(without_time, 1)


if __name__ == "__main__":
    unittest.main()
