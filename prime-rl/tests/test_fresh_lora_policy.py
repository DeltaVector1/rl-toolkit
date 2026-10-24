import ast
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


source_path = Path(__file__).parents[1] / "src/prime_rl/trainer/rl/train.py"


class Dist:
    ReduceOp = SimpleNamespace(SUM="sum", MAX="max")

    def __init__(self, remote=None):
        self.remote = remote or {}
        self.calls = []

    def get_rank(self):
        return 0

    def get_world_size(self):
        return 2

    def all_gather_object(self, output, local):
        self.calls.append(local.copy())
        remote = dict(local, rank=1)
        remote.update(self.remote.get(len(self.calls), {}))
        output[:] = [local, remote]

    def all_reduce(self, value, op):
        self.calls.append(op)


def load_helpers(dist):
    tree = ast.parse(source_path.read_text())
    names = {"initialize_fresh_lora_policy", "agree_fresh_trainer_only_lora", "verify_fresh_trainer_only_lora"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "dist": dist, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace


class FreshLoraPolicyTest(unittest.TestCase):
    def setUp(self):
        self.a = torch.tensor([1.0, 2.0])
        self.b = torch.tensor([0.0, 0.0])
        self.parameters = [("layer.lora_A", self.a), ("layer.lora_B", self.b)]
        self.manager = SimpleNamespace(
            get_named_parameters_for_run=lambda idx: self.parameters,
            scaling_factors=[2.0],
        )
        self.config = SimpleNamespace(alpha=64, rank=32, initial_adapter_path=None)

    def verify(self, dist=None, checkpoint_step=None, resume_adapter=None):
        namespace = load_helpers(dist or Dist())
        self.assertIn("agree_fresh_trainer_only_lora", list(namespace))
        fresh = namespace["agree_fresh_trainer_only_lora"](self.config, checkpoint_step, resume_adapter)
        if fresh:
            return namespace["verify_fresh_trainer_only_lora"](self.manager, self.config)
        return None

    def test_zero_finite_policy_passes_without_mutation(self):
        evidence = self.verify()
        self.assertEqual(evidence, {"rank_count": 2, "local_b_count": 1, "global_b_count": 2,
                                    "nonzero_b_shards": 0, "scale": 2.0})
        self.assertEqual(self.a.tolist(), [1, 2])
        self.assertEqual(self.b.tolist(), [0, 0])

    def test_nonzero_b_fails_without_reset(self):
        self.b[0] = 1
        with self.assertRaisesRegex(RuntimeError, "zero"):
            self.verify()
        self.assertEqual(self.b.tolist(), [1, 0])

    def test_nonfinite_run_parameters_fail_without_reset(self):
        for parameter in (self.a, self.b):
            for value in (float("nan"), float("inf")):
                with self.subTest(value=value):
                    original = parameter.clone()
                    parameter[0] = value
                    with self.assertRaisesRegex(RuntimeError, "finite"):
                        self.verify()
                    self.assertFalse(torch.isfinite(parameter[0]))
                    parameter.copy_(original)

    def test_bad_or_nonfinite_scale_fails(self):
        for scale in (1.0, 2.000001, float("inf"), float("nan")):
            with self.subTest(scale=scale):
                self.manager.scaling_factors = [scale]
                with self.assertRaisesRegex(RuntimeError, "scale"):
                    self.verify()
        self.config.alpha = float("inf")
        self.manager.scaling_factors = [float("inf")]
        with self.assertRaisesRegex(RuntimeError, "scale"):
            self.verify()

    def test_missing_b_and_inspection_error_reach_collective_before_failure(self):
        for broken in (False, True):
            with self.subTest(broken=broken):
                self.parameters = []
                if broken:
                    self.manager.get_named_parameters_for_run = Mock(side_effect=ValueError("broken slot"))
                dist = Dist()
                with self.assertRaises(RuntimeError):
                    self.verify(dist)
                self.assertEqual(len(dist.calls), 2)
                self.assertTrue(dist.calls[-1]["error"])

    def test_remote_inspection_failure_rejects_valid_local_policy(self):
        with self.assertRaisesRegex(RuntimeError, "rank=1.*bad shard"):
            self.verify(Dist({2: {"error": "bad shard"}}))

    def test_local_shards_and_bfloat16_scale(self):
        self.parameters = [(name, SimpleNamespace(to_local=lambda p=value: p))
                           for name, value in self.parameters]
        self.manager.scaling_factors = torch.tensor([2.0], dtype=torch.bfloat16)
        self.assertEqual(self.verify()["scale"], 2.0)

    def test_mixed_eligibility_fails_before_inspection(self):
        self.manager.get_named_parameters_for_run = Mock(side_effect=AssertionError("must not inspect"))
        with self.assertRaisesRegex(RuntimeError, "eligibility"):
            self.verify(Dist({1: {"eligible": False}}))
        self.manager.get_named_parameters_for_run.assert_not_called()

    def test_checkpoint_resume_and_initial_adapter_are_unchanged(self):
        self.b[0] = 5
        for checkpoint, adapter, initial in ((0, None, None), (3, None, None),
                                              (None, "/resume", None), (None, None, "/initial")):
            with self.subTest(checkpoint=checkpoint, adapter=adapter, initial=initial):
                self.config.initial_adapter_path = initial
                dist = Dist()
                self.assertIsNone(self.verify(dist, checkpoint, adapter))
                self.assertEqual(len(dist.calls), 1)
                self.assertEqual(self.b.tolist(), [5, 0])

    def test_normal_production_policy_still_resets_b(self):
        self.b[:] = 3
        helper = load_helpers(Dist())["initialize_fresh_lora_policy"]
        self.assertEqual(helper(self.manager, self.config, "cpu"), (1, 1))
        self.assertEqual(self.b.tolist(), [0, 0])


class TrainerOnlyFreshPolicyTest(unittest.TestCase):
    def test_two_rank_collective_sequence_is_bounded_and_agreed_before_load(self):
        for adapters in ((None, None), ("/resume", "/resume"), (None, "/resume")):
            with self.subTest(adapters=adapters):
                barrier = Barrier(2, timeout=2)
                records = [None, None]
                traces = [[], []]

                class PairedDist(Dist):
                    def __init__(self, rank):
                        self.rank = rank

                    def get_rank(self):
                        return self.rank

                    def all_gather_object(self, output, local):
                        kind = "eligibility" if "eligible" in local else "load" if "load" in local else "evidence"
                        traces[self.rank].append(kind)
                        records[self.rank] = local.copy()
                        barrier.wait()
                        output[:] = records
                        barrier.wait()

                def rank_worker(rank):
                    dist = PairedDist(rank)
                    helpers = load_helpers(dist)
                    config = SimpleNamespace(alpha=64, rank=32, initial_adapter_path=None)
                    try:
                        fresh = helpers["agree_fresh_trainer_only_lora"](config, None, adapters[rank])
                    except RuntimeError as error:
                        return str(error)
                    if adapters[rank]:
                        dist.all_gather_object([None, None], {"load": True})
                    if fresh:
                        manager = SimpleNamespace(scaling_factors=[2], get_named_parameters_for_run=
                                                  lambda idx: [("lora_B", torch.zeros(2))])
                        return helpers["verify_fresh_trainer_only_lora"](manager, config)
                    return None

                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(rank_worker, (0, 1)))
                if adapters == (None, "/resume"):
                    self.assertEqual(traces, [["eligibility"], ["eligibility"]])
                    self.assertEqual(results[0], results[1])
                    self.assertIn("eligibility", results[0])
                else:
                    second = "load" if adapters[0] else "evidence"
                    self.assertEqual(traces, [["eligibility", second]] * 2)

    def test_verifier_runs_on_all_trainer_only_lora_ranks_after_load_before_data(self):
        tree = ast.parse(source_path.read_text())
        train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
        blocks = [node for node in train.body if isinstance(node, ast.If)
                  and any(isinstance(call, ast.Call) and ast.unparse(call.func) == "verify_fresh_trainer_only_lora"
                          for call in ast.walk(node))]
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertEqual(ast.unparse(block.test), "fresh_trainer_only_lora")
        agreement = next(node for node in train.body if isinstance(node, ast.If)
                         and any(isinstance(call, ast.Call) and ast.unparse(call.func) == "agree_fresh_trainer_only_lora"
                                 for call in ast.walk(node)))
        self.assertEqual(ast.unparse(agreement.test), "trainer_only and config.model.lora is not None")
        resume = next(node for node in train.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "resume_adapter" for target in node.targets))
        self.assertLess(resume.lineno, agreement.lineno)
        calls = [node for node in ast.walk(train) if isinstance(node, ast.Call)]
        for call in calls:
            name = ast.unparse(call.func)
            if name in {"load_initial_lora_adapter", "ckpt_manager.load", "multi_run_manager.create_trainer_only_run"}:
                self.assertLess(call.lineno, block.lineno)
            if name in {"load_initial_lora_adapter", "ckpt_manager.load", "multi_run_manager.create_trainer_only_run", "setup_model"}:
                self.assertLess(agreement.end_lineno, call.lineno)
            if name in {"DataLoader", "FakeDataLoader", "model", "model.forward"}:
                self.assertLess(block.end_lineno, call.lineno)
        namespace = {"fresh_trainer_only_lora": True, "config": SimpleNamespace(model=SimpleNamespace(lora=object())),
                     "multi_run_manager": object(), "checkpoint_step": None, "resume_adapter": None,
                     "verify_fresh_trainer_only_lora": Mock(return_value={"rank_count": 2, "local_b_count": 1,
                          "global_b_count": 2, "nonzero_b_shards": 0, "scale": 2.0}),
                     "world": SimpleNamespace(is_master=True), "logger": Mock()}
        for master in (True, False):
            namespace["world"].is_master = master
            namespace["logger"].reset_mock()
            exec(compile(ast.Module(body=[block], type_ignores=[]), str(source_path), "exec"), namespace)
            self.assertEqual(namespace["logger"].info.call_count, int(master))
            if master:
                message = namespace["logger"].info.call_args.args[0]
                for evidence in ("2 ranks", "1 local B matrices", "2 global B shards",
                                 "pre-reset nonzero B shards=0", "finite", "scale=2"):
                    self.assertIn(evidence, message)
        namespace["verify_fresh_trainer_only_lora"].reset_mock()
        namespace["fresh_trainer_only_lora"] = False
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(source_path), "exec"), namespace)
        namespace["verify_fresh_trainer_only_lora"].assert_not_called()
        self.assertIn("if fresh_lora_policy and progress.step == 1:", source_path.read_text())


if __name__ == "__main__":
    unittest.main()
