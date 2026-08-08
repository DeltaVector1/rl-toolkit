import json
import unittest
from unittest import mock

import glm52_ops


def metric(step, reward):
    return {
        "step": step,
        "eff": reward,
        "all": reward / 2,
        "p10": reward / 4,
        "p90": reward * 2,
        "err": 0.1,
        "t": 60,
    }


def output(rows):
    return "REWARD_JSON " + json.dumps(rows) + "\n"


class RewardHostTest(unittest.TestCase):
    def reward_with(self, responses):
        calls = []

        def fake_ssh(host, script, timeout=120):
            calls.append((host, script, timeout))
            return responses[host]

        with mock.patch.object(glm52_ops, "_ssh", side_effect=fake_ssh):
            report = glm52_ops.reward("x-test", quiet=True)
        return report, calls

    def test_edge_metrics_on_h200a_win(self):
        report, calls = self.reward_with(
            {
                "h200a": (0, output([metric(2, 0.25)])),
                "n82": (0, output([metric(2, 0.75)])),
            }
        )

        self.assertEqual(report.steps, [2])
        self.assertEqual(report.effective, [0.25])
        self.assertCountEqual([call[0] for call in calls], glm52_ops.REWARD_HOSTS)
        self.assertTrue(all(call[2] == 180 for call in calls))

    def test_china_metrics_on_n82_remain_available(self):
        report, _ = self.reward_with(
            {
                "h200a": (0, "NORUN\n"),
                "n82": (0, output([metric(3, 0.5)])),
            }
        )

        self.assertEqual(report.steps, [3])
        self.assertEqual(report.effective, [0.5])

    def test_empty_edge_file_falls_back_to_populated_china_file(self):
        report, _ = self.reward_with(
            {
                "h200a": (0, output([])),
                "n82": (0, output([metric(4, 0.55)])),
            }
        )

        self.assertEqual(report.steps, [4])
        self.assertEqual(report.effective, [0.55])

    def test_bad_edge_output_falls_back_to_n82(self):
        report, _ = self.reward_with(
            {
                "h200a": (255, "REWARD_JSON not-json\nssh failed\n"),
                "n82": (0, output([metric(4, 0.6)])),
            }
        )

        self.assertEqual(report.steps, [4])
        self.assertEqual(report.effective, [0.6])

    def test_no_data_names_both_locations(self):
        report, _ = self.reward_with(
            {
                "h200a": (0, "NORUN\n"),
                "n82": (0, "NORUN\n"),
            }
        )

        self.assertEqual(report.verdict, "no data")
        self.assertIn("h200a or n82", report.note)


if __name__ == "__main__":
    unittest.main()
