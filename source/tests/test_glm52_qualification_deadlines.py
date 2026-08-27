"""Tests for the b2048 qualification fail-fast state machine."""

import ast
import os
from pathlib import Path
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]


def _deadline_class():
    source = (ROOT / "source/src/prime_rl/trainer/rl/train.py").read_text()
    tree = ast.parse(source)
    names = {
        "_qualification_deadline_environment_names",
        "_b2048_qualification_profiles",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_QualificationDeadline"
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    namespace = {"os": os, "threading": threading, "time": time}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "train.py", "exec"), namespace)
    return namespace["_QualificationDeadline"]


class QualificationDeadlineTest(unittest.TestCase):
    def setUp(self):
        self.deadline_class = _deadline_class()
        self.values = {
            "PRIME_QUALIFICATION_SESSION_DEADLINE_SECONDS": "3600",
            "PRIME_QUALIFICATION_STARTUP_DEADLINE_SECONDS": "900",
            "PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS": "900",
            "PRIME_QUALIFICATION_PROFILER_FINALIZE_DEADLINE_SECONDS": "120",
        }

    def test_each_phase_expires_and_completed_phases_do_not(self):
        deadline = self.deadline_class("glm52-native-b2048", self.values)
        deadline._started = 0
        self.assertEqual(deadline._deadline_reason(901), "startup/model-load")
        deadline.mark_startup_complete()
        deadline.begin_forward_backward()
        deadline._forward_backward_started = 0
        self.assertEqual(deadline._deadline_reason(901), "forward/backward")
        deadline.end_forward_backward()
        deadline.begin_profiler_finalize()
        deadline._profiler_finalize_started = 0
        self.assertEqual(deadline._deadline_reason(121), "profiler finalization")
        deadline.end_profiler_finalize()
        self.assertIsNone(deadline._deadline_reason(121))
        self.assertEqual(deadline._deadline_reason(3601), "session")

    def test_missing_or_changed_deadlines_are_rejected(self):
        values = dict(self.values)
        values.pop("PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.deadline_class("glm52-native-b2048", values)
        values = dict(self.values)
        values["PRIME_QUALIFICATION_STEP_DEADLINE_SECONDS"] = "901"
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            self.deadline_class("glm52-native-b2048", values)

    def test_session_evidence_is_bound_to_environment(self):
        deadline = self.deadline_class("glm52-native-b2048", self.values)
        deadline.validate_session({"deadlines": deadline.session_values})
        with self.assertRaisesRegex(RuntimeError, "evidence differs"):
            deadline.validate_session({"deadlines": {"session_seconds": 60}})

    def test_completion_suppresses_expiry_and_joins_watcher(self):
        deadline = self.deadline_class("glm52-native-b2048", self.values)
        deadline._started = 0
        deadline.start()
        deadline.mark_completed()
        self.assertFalse(deadline._thread.is_alive())
        self.assertIsNone(deadline._deadline_reason(10_000))


if __name__ == "__main__":
    unittest.main()
