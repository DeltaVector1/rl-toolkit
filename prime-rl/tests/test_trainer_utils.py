import ast
import math
import os
import unittest
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:  # pragma: no cover - the trainer image provides Torch
    torch = None
    dist = None


utils_path = Path(__file__).parents[1] / "src/prime_rl/trainer/utils.py"


def load_utils():
    if torch is None or dist is None:
        raise unittest.SkipTest("Torch is required")
    tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {"flexible_all_gather", "Tensors"}
    ]
    namespace = {"Tensor": torch.Tensor, "torch": torch, "dist": dist, "defaultdict": defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(utils_path), "exec"), namespace)
    return namespace["flexible_all_gather"], namespace["Tensors"]


class TrainerUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None or dist is None or "RANK" not in os.environ:
            raise unittest.SkipTest("run under torchrun in the trainer image")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        cls.gather, cls.tensors_type = load_utils()

    @classmethod
    def tearDownClass(cls):
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()

    def test_uneven_noncontiguous_duplicates_and_stats(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_size = rank + 1
        source = torch.arange(local_size * 2, device="cuda", dtype=torch.float32) + rank * 100
        local = source[::2]
        self.assertFalse(local.is_contiguous())
        gathered = self.gather(local)
        self.assertEqual(gathered.numel(), world_size * (world_size + 1) // 2)
        offset = 0
        for owner in range(world_size):
            expected = torch.arange(owner + 1, device="cuda", dtype=torch.float32) * 2 + owner * 100
            self.assertTrue(torch.equal(gathered[offset : offset + owner + 1], expected))
            offset += owner + 1

        values = [1.0, 1.0, float(rank)]
        stats = self.tensors_type()
        stats["score"].append(torch.tensor(values, device="cuda"))
        metrics = stats.compute_stats()
        expected = torch.tensor(
            [item for owner in range(world_size) for item in (1.0, 1.0, float(owner))],
            device="cuda",
        )
        self.assertEqual(metrics["score/mean"], expected.mean().item())
        self.assertEqual(metrics["score/median"], torch.median(expected).item())
        self.assertEqual(metrics["score/std"], expected.std().item())
        self.assertEqual(metrics["score/min"], expected.min().item())
        self.assertEqual(metrics["score/max"], expected.max().item())
        self.assertEqual(dict(stats), {})

        empty = self.tensors_type()
        empty["empty"].append(torch.empty(0, device="cuda"))
        empty_metrics = empty.compute_stats()
        for suffix in ("mean", "median", "std", "min", "max"):
            self.assertTrue(math.isnan(empty_metrics[f"empty/{suffix}"]))
        self.assertEqual(dict(empty), {})


if __name__ == "__main__":
    unittest.main()
