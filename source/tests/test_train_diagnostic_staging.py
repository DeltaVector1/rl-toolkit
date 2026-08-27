import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TRAIN = Path(__file__).resolve().parents[1] / "src/prime_rl/trainer/rl/train.py"


class _FakeTensor:
    def __init__(self, device="cuda", detached=False, copied_to_cpu=False):
        self.device = device
        self.detached = detached
        self.copied_to_cpu = copied_to_cpu

    def detach(self):
        return _FakeTensor(self.device, detached=True, copied_to_cpu=self.copied_to_cpu)

    def to(self, device):
        return _FakeTensor(device, detached=self.detached, copied_to_cpu=device == "cpu")


class TrainDiagnosticStagingTest(unittest.TestCase):
    def _helper(self, env):
        tree = ast.parse(TRAIN.read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_DEFER_PP_DIAGNOSTICS"
                for target in node.targets
            )
        )
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_detach_pp_diagnostic"
        )
        module = ast.Module(body=[assignment, function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "os": os,
            "torch": SimpleNamespace(Tensor=_FakeTensor),
        }
        with patch.dict(os.environ, env, clear=True):
            exec(compile(module, str(TRAIN), "exec"), namespace)
        return namespace["_detach_pp_diagnostic"]

    def test_deferred_diagnostics_only_changes_post_backward_metric_storage(self):
        source = TRAIN.read_text()
        self.assertEqual(source.count("_detach_pp_diagnostic("), 4)
        self.assertLess(
            source.index("tensors.compute_stats()"),
            source.index("parity_guard_failures(tensor_stats, config.parity_guard)"),
        )

        eager = self._helper({})
        copied = eager(_FakeTensor())
        self.assertTrue(copied.detached)
        self.assertEqual(copied.device, "cpu")
        self.assertTrue(copied.copied_to_cpu)

        deferred = self._helper({"PRIME_DEFER_PP_DIAGNOSTICS": "1"})
        retained = deferred(_FakeTensor())
        self.assertTrue(retained.detached)
        self.assertEqual(retained.device, "cuda")
        self.assertFalse(retained.copied_to_cpu)


if __name__ == "__main__":
    unittest.main()
