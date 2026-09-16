import ast
import asyncio
import math
import sys
import tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar
from unittest import mock

import pydantic
import torch
from pydantic_config import BaseConfig


root = Path(__file__).resolve().parents[1]
config_root = root / "packages/prime-rl-configs/src/prime_rl/configs"


def load_source(path, **namespace):
    """Use real local definitions; isolate unrelated renderer/service imports."""
    tree = ast.parse(path.read_text())
    tree.body = [
        node for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or not (node.module or "").startswith(("prime_rl", "renderers"))
    ]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class Algorithm:
    action_loss_type: ClassVar[str] = "rl"

    def __init__(self, config, policy_pool):
        self.policy_pool = policy_pool


shared = load_source(config_root / "shared.py", BaseConfig=BaseConfig)
configs = load_source(
    config_root / "algorithm.py", BaseConfig=BaseConfig,
    ClientConfig=shared["ClientConfig"], RendererConfig=BaseConfig, AutoRendererConfig=BaseConfig,
)
algo_config = pydantic.TypeAdapter(configs["AlgoConfig"])
tail = load_source(root / "src/prime_rl/orchestrator/algo/tail_likelihood.py", Algorithm=Algorithm)
grpo = load_source(
    root / "src/prime_rl/orchestrator/algo/grpo.py",
    Algorithm=Algorithm, GRPOAlgoConfig=configs["GRPOAlgoConfig"],
)


class Rollout:
    def __init__(self, reward, mask=(False, True, True)):
        self.reward = reward
        self.mask = mask
        self.advantages = None
        self.num_output_tokens = sum(mask)
        self.num_total_tokens = len(mask)
        self.num_turns = 1

    def assign_advantages(self, value):
        self.advantages = [value if keep else 0.0 for keep in self.mask]


class TailLikelihoodTest(unittest.TestCase):
    def test_known_canonical_example(self):
        for actual, expected in zip(tail["tail_likelihood_advantages"]([0.0, 0.5, 1.0]), [-1.0, -0.25, 1.25], strict=True):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_permutation_restores_original_order(self):
        source = tail["tail_likelihood_advantages"]([0.0, 0.5, 1.0])
        self.assertEqual(tail["tail_likelihood_advantages"]([1.0, 0.0, 0.5]), [source[2], source[0], source[1]])

    def test_ties_have_equal_weights(self):
        values = tail["tail_likelihood_advantages"]([1.0, 0.5, 0.0, 0.5])
        self.assertEqual(values[1], values[3])
        self.assertAlmostEqual(math.fsum(values), 0.0)
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])

    def test_constant_rewards_have_zero_advantages(self):
        for reward in (0.0, 0.1, 1.0):
            self.assertEqual(tail["tail_likelihood_advantages"]([reward] * 16), [0.0] * 16)

    def test_invalid_rewards_fail_before_any_assignment(self):
        for reward in (float("nan"), float("inf"), -float("inf"), -0.01, 1.000000001, 10**400, True, None, "0.5"):
            with self.subTest(reward=reward):
                group = [Rollout(0.0), Rollout(reward)]
                with self.assertRaisesRegex(ValueError, "finite numeric rewards"):
                    asyncio.run(tail["TailRLAlgorithm"](None, None).score_group(group))
                self.assertTrue(all(item.advantages is None for item in group))

    def test_incomplete_group_fails(self):
        for rewards in ([], [0.5]):
            with self.assertRaisesRegex(ValueError, "at least two"):
                tail["tail_likelihood_advantages"](rewards)

    def test_scalar_is_assigned_through_existing_mask_api(self):
        group = [Rollout(0.0, (False, True, False, True)), Rollout(1.0, (True, False))]
        asyncio.run(tail["TailRLAlgorithm"](None, None).score_group(group))
        self.assertEqual(group[0].advantages, [0.0, -1.0, 0.0, -1.0])
        self.assertEqual(group[1].advantages, [1.0, 0.0])

    def test_config_round_trip_and_invalid_settings(self):
        config = algo_config.validate_python(tomllib.loads('[algo]\ntype = "tail_likelihood"')["algo"])
        self.assertEqual(config.type, "tail_likelihood")
        self.assertEqual(config.action_loss_type, "rl")
        self.assertEqual(config.sampling.source, "policy")
        self.assertEqual(algo_config.validate_json(config.model_dump_json()), config)
        for settings in (
            {"type": "unknown"},
            {"type": "tail_likelihood", "length_penalty": {}},
            {"type": "tail_likelihood", "normalize_std": True},
            {"type": "tail_likelihood", "sampling": {"source": {"name": "frozen", "base_url": ["http://example.test/v1"]}}},
        ):
            with self.subTest(settings=settings), self.assertRaises(pydantic.ValidationError):
                algo_config.validate_python(settings)

    def test_existing_config_inheritance_and_default(self):
        tree = ast.parse((config_root / "orchestrator.py").read_text())
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OrchestratorConfig")
        default = next(node for node in model.body if isinstance(node, ast.AnnAssign) and node.target.id == "algo")
        self.assertEqual(ast.unparse(default.value), "GRPOAlgoConfig()")
        method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "inherit_env_algorithms")
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "inheritance", "exec"), namespace)
        explicit = algo_config.validate_python({"type": "grpo"})
        config = algo_config.validate_python({"type": "tail_likelihood"})
        inherited = SimpleNamespace(algo=None)
        overridden = SimpleNamespace(algo=explicit)
        owner = SimpleNamespace(algo=config, train=SimpleNamespace(env=[inherited, overridden]))
        namespace["inherit_env_algorithms"](owner)
        self.assertEqual(inherited.algo, config)
        self.assertIsNot(inherited.algo, config)
        self.assertIs(overridden.algo, explicit)

    def test_registry_imports_with_baked_modules_and_no_rae(self):
        prefix = "prime_rl.orchestrator.algo"
        modules = {}
        for name in ("prime_rl", "prime_rl.orchestrator", prefix):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        exports = {
            "base": {"Algorithm": Algorithm, "connect_frozen_pool": None},
            "echo": {"EchoAlgorithm": Algorithm},
            "grpo": {"GRPOAlgorithm": grpo["GRPOAlgorithm"]},
            "max_rl": {"MaxRLAlgorithm": Algorithm},
            "opd": {"OPDAlgorithm": Algorithm},
            "opsd": {"OPSDAlgorithm": Algorithm},
            "routing": {"stamp_advantages": None, "stamp_loss_routing": None},
            "sft": {"SFTDistillAlgorithm": Algorithm},
            "tail_likelihood": {},
        }
        for name, attributes in exports.items():
            module = modules[f"{prefix}.{name}"] = ModuleType(f"{prefix}.{name}")
            module.__dict__.update(attributes)
        modules["prime_rl.orchestrator.types"] = ModuleType("prime_rl.orchestrator.types")
        modules["prime_rl.orchestrator.types"].Rollout = Rollout
        modules[f"{prefix}.rae"] = None  # The deployed image has no RAE module.
        with mock.patch.dict(sys.modules, modules):
            for name, filename in ((f"{prefix}.tail_likelihood", "tail_likelihood.py"), (prefix, "__init__.py")):
                path = root / "src/prime_rl/orchestrator/algo" / filename
                # Execute all imports; load_source() would hide image incompatibility.
                exec(compile(path.read_text(), str(path), "exec"), modules[name].__dict__)
            registry = modules[prefix]
            self.assertEqual(set(registry.ALGORITHM_CLASSES),
                             {"grpo", "echo", "max_rl", "opd", "opsd", "sft", "tail_likelihood"})
            self.assertNotIn("RAEAlgorithm", registry.__all__)
            for name, expected in (("grpo", grpo["GRPOAlgorithm"]),
                                   ("tail_likelihood", modules[f"{prefix}.tail_likelihood"].TailRLAlgorithm)):
                self.assertIsInstance(registry.build_algorithm(algo_config.validate_python({"type": name}), None), expected)

    def test_disabled_grpo_is_byte_identical_and_numerically_unchanged(self):
        for rewards in ([0.0, 0.5, 1.0], [0.1] * 16, [0.01, 0.02, 0.95, 1.0]):
            for penalty in (None, configs["LinearLengthPenaltyConfig"]()):
                group = [Rollout(reward, (False,) + (True,) * (index + 1)) for index, reward in enumerate(rewards)]
                config = configs["GRPOAlgoConfig"](length_penalty=penalty)
                asyncio.run(grpo["GRPOAlgorithm"](config, None).score_group(group))
                values = torch.tensor(rewards, dtype=torch.float32)
                shaped = values
                if penalty is not None:
                    output = torch.tensor([r.num_output_tokens for r in group], dtype=values.dtype)
                    total = torch.tensor([r.num_total_tokens for r in group], dtype=values.dtype)
                    turns = torch.tensor([r.num_turns for r in group], dtype=values.dtype)
                    fractions = (penalty.num_output_tokens_weight * (output / output.max().clamp(min=1))
                                 + penalty.num_input_tokens_weight * ((total - output) / (total - output).max().clamp(min=1))
                                 + penalty.num_turns_weight * (turns / turns.max().clamp(min=1)))
                    shaped = values - values.mean() * fractions
                expected = shaped - shaped.mean()
                expected = expected / expected.std(unbiased=False).clamp_min(1e-6)
                self.assertTrue(torch.equal(torch.tensor([r.advantages[-1] for r in group]), expected))


if __name__ == "__main__":
    unittest.main()
