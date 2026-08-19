from __future__ import annotations

import ast
import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/algo/grpo.py"
)
SOURCE = SOURCE_PATH.read_text()


def load_advantage_function():
    tree = ast.parse(SOURCE)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.FunctionDef))
        and (
            isinstance(node, ast.FunctionDef)
            and node.name == "calculate_decensor_advantages"
            or isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "DECENSOR_QUALITY_ADVANTAGE_WEIGHT"
                for target in node.targets
            )
        )
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["calculate_decensor_advantages"]


calculate = load_advantage_function()


def load_grpo_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.float32 = object()
    torch_stub.tensor = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("the decensor path must not use the standard torch reward path")
    )
    config_stub = types.ModuleType("prime_rl.configs.algorithm")
    config_stub.GRPOAlgoConfig = type("GRPOAlgoConfig", (), {})
    base_stub = types.ModuleType("prime_rl.orchestrator.algo.base")

    class Algorithm:
        def __init__(self, config, policy_pool):
            self.policy_pool = policy_pool

    base_stub.Algorithm = Algorithm
    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "prime_rl.configs.algorithm",
            "prime_rl.orchestrator.algo.base",
        )
    }
    sys.modules["torch"] = torch_stub
    sys.modules["prime_rl.configs.algorithm"] = config_stub
    sys.modules["prime_rl.orchestrator.algo.base"] = base_stub
    try:
        specification = importlib.util.spec_from_file_location(
            "tested_decensor_grpo",
            SOURCE_PATH,
        )
        assert specification and specification.loader
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class DecensorAdvantageTests(unittest.TestCase):
    def test_all_failed_group_has_no_positive_training_signal(self):
        combined, reasoning, quality = calculate(
            [0] * 16,
            [index / 20 for index in range(16)],
        )
        self.assertEqual(combined, [0.0] * 16)
        self.assertEqual(reasoning, [0.0] * 16)
        self.assertEqual(quality, [0.0] * 16)

    def test_all_passed_group_keeps_quality_learning(self):
        combined, reasoning, quality = calculate([1, 1, 1], [0.2, 0.5, 0.8])
        self.assertTrue(all(value == 0.0 for value in reasoning))
        self.assertEqual(combined, quality)
        self.assertAlmostEqual(sum(combined), 0.0)
        self.assertGreater(combined[-1], 0.0)
        self.assertLess(combined[0], 0.0)

    def test_mixed_group_never_promotes_a_failed_response(self):
        passes = [0, 0, 1, 1]
        quality_rewards = [0.9, 0.8, 0.1, 0.9]
        combined, _reasoning, _quality = calculate(passes, quality_rewards)
        failed = [
            value
            for value, passed in zip(combined, passes, strict=True)
            if not passed
        ]
        passed = [
            value for value, passed in zip(combined, passes, strict=True) if passed
        ]
        self.assertTrue(all(value < 0.0 for value in failed))
        self.assertGreater(min(passed), max(failed))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate([0, 1], [0.5])
        with self.assertRaises(ValueError):
            calculate([2], [0.5])
        with self.assertRaises(ValueError):
            calculate([1], [1.1])

    def test_group_scorer_uses_recorded_components(self):
        self.assertIn('details.get("advantage_scheme")', SOURCE)
        self.assertIn('details.get("reasoning_requirement_pass")', SOURCE)
        self.assertIn('details.get("quality_reward")', SOURCE)
        self.assertIn('rollout.info["decensor_training"]', SOURCE)

    def test_group_scorer_executes_the_separate_component_path(self):
        module = load_grpo_module()

        class Rollout:
            def __init__(self, passed, quality):
                self.info = {
                    "decensor": {
                        "advantage_scheme": "separate_reasoning_and_quality",
                        "reasoning_requirement_pass": passed,
                        "quality_reward": quality,
                    }
                }
                self.metrics = {}
                self.advantage = None

            def record_metric(self, name, value):
                self.metrics[name] = value

            def assign_advantages(self, value):
                self.advantage = value

        config = types.SimpleNamespace(length_penalty=None)
        algorithm = module.GRPOAlgorithm(config, None)
        group = [Rollout(0, 0.9), Rollout(1, 0.2), Rollout(1, 0.8)]
        asyncio.run(algorithm.score_group(group))
        self.assertLess(group[0].advantage, 0.0)
        self.assertEqual(group[0].metrics["decensor_quality_advantage"], 0.0)
        self.assertTrue(
            all("decensor_training" in rollout.info for rollout in group)
        )


if __name__ == "__main__":
    unittest.main()
