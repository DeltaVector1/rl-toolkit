import json
import unittest
from unittest import mock

import training_ops


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
    def test_default_reads_only_the_primary_metric_host(self):
        calls = []

        def fake_ssh(host, script, timeout=120):
            calls.append(host)
            return 0, output([metric(2, 0.25)])

        with mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh):
            report = training_ops.reward("x-test", quiet=True)

        self.assertEqual(calls, [training_ops.metric_hosts[0]])
        self.assertEqual(report.effective, [0.25])

    def test_no_data_names_the_checked_host(self):
        with mock.patch.object(training_ops, "_ssh", return_value=(0, "NORUN\n")):
            report = training_ops.reward("x-test", quiet=True)

        self.assertEqual(report.verdict, "no data")
        self.assertIn("on h200a", report.note)
        self.assertIn("explicitly", report.note)

    def test_clear_positive_change_is_reported_as_rising(self):
        rows = [metric(step, 0.10 + step * 0.05) for step in range(10)]
        with mock.patch.object(
            training_ops, "_ssh", return_value=(0, output(rows))
        ):
            report = training_ops.reward("x-test", quiet=True)

        self.assertEqual(report.verdict, "rising")
        self.assertGreater(report.total_change, 0)


if __name__ == "__main__":
    unittest.main()
