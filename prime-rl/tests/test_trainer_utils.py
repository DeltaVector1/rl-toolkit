import ast
import math
import os
import unittest
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import textwrap

try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:  # pragma: no cover - the trainer image provides Torch
    torch = None
    dist = None


utils_path = Path(__file__).parents[1] / "src/prime_rl/trainer/utils.py"
train_path = Path(__file__).parents[1] / "src/prime_rl/trainer/rl/train.py"


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


def load_gradient_guard():
    source = train_path.read_text(encoding="utf-8")
    start_marker = "        found_nonfinite = torch.zeros("
    end_marker = "        # Optionally, clip the gradients"
    assert source.count(start_marker) == 1
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return ast.parse(textwrap.dedent(source[start:end])).body


class TrainerUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None or dist is None or "RANK" not in os.environ:
            raise unittest.SkipTest("run under torchrun in the trainer image")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        gather, cls.tensors_type = load_utils()
        cls.gather = staticmethod(gather)

    @classmethod
    def tearDownClass(cls):
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()

    def run_gradient_guard(self, parameters):
        logger = SimpleNamespace(errors=[])
        logger.error = lambda *args: logger.errors.append(args)
        monitor = SimpleNamespace(logs=[])
        monitor.log = lambda *args, **kwargs: monitor.logs.append((args, kwargs))
        namespace = {
            "torch": torch,
            "dist": dist,
            "model": SimpleNamespace(named_parameters=lambda: list(parameters)),
            "parallel_dims": SimpleNamespace(fsdp_gradient_divide_factor=2),
            "logger": logger,
            "monitor": monitor,
            "progress": SimpleNamespace(step=1),
            "tensor_stats": {},
        }
        function = ast.FunctionDef(
            name="run_guard",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=load_gradient_guard(),
            decorator_list=[],
        )
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(train_path), "exec"), namespace)
        try:
            namespace["run_guard"]()
        except RuntimeError as error:
            if "Non-finite gradient at step" not in str(error):
                raise
            return True, logger
        return False, logger

    def test_finite_gradient_guard_collective_paths(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.assertGreaterEqual(world_size, 2)
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cuda", (world_size,))
        linear = torch.nn.Linear(2, 2, device="cuda")

        if rank == 0:
            local = torch.ones_like(linear.weight)
            grad = DTensor.from_local(local, mesh, [Replicate()])
            parameters = [("weight", SimpleNamespace(grad=grad)), ("bias", SimpleNamespace(grad=None))]
        elif rank == 1:
            grad = torch.ones_like(linear.weight)
            parameters = [("weight", SimpleNamespace(grad=grad)), ("bias", SimpleNamespace(grad=None))]
        else:
            parameters = [("weight", SimpleNamespace(grad=None)), ("bias", SimpleNamespace(grad=None))]
        rejected, _ = self.run_gradient_guard(parameters)
        self.assertFalse(rejected)
        if rank in (0, 1):
            local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
            self.assertTrue(torch.equal(local_grad, torch.full_like(local_grad, 2)))

        rejected, _ = self.run_gradient_guard(
            [("weight", SimpleNamespace(grad=None)), ("bias", SimpleNamespace(grad=None))]
        )
        self.assertFalse(rejected)

        for value in (float("nan"), float("inf")):
            bad_grad = torch.full_like(linear.weight, value) if rank == 0 else None
            parameters = [("weight", SimpleNamespace(grad=bad_grad)), ("bias", SimpleNamespace(grad=None))]
            rejected, logger = self.run_gradient_guard(parameters)
            self.assertTrue(rejected)
            rejection_count = torch.tensor(int(rejected), device="cuda", dtype=torch.int32)
            dist.all_reduce(rejection_count, op=dist.ReduceOp.SUM)
            self.assertEqual(rejection_count.item(), world_size)
            if rank == 0:
                self.assertTrue(any("weight" in str(args) for args in logger.errors))

    def test_uneven_noncontiguous_duplicates_and_stats(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.assertGreaterEqual(world_size, 2)
        local_size = rank + 2
        source = torch.arange(local_size * 2, device="cuda", dtype=torch.float32) + rank * 100
        local = source[::2]
        self.assertFalse(local.is_contiguous())
        gathered = self.gather(local)
        self.assertEqual(gathered.numel(), world_size * (world_size + 3) // 2)
        offset = 0
        for owner in range(world_size):
            owner_size = owner + 2
            expected = torch.arange(owner_size, device="cuda", dtype=torch.float32) * 2 + owner * 100
            self.assertTrue(torch.equal(gathered[offset : offset + owner_size], expected))
            offset += owner_size

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

        sparse = self.tensors_type()
        if rank == world_size - 1:
            sparse["sparse"].append(torch.tensor([2.0, 2.0, 7.0], device="cuda"))
        sparse_metrics = sparse.compute_stats()
        sparse_expected = torch.tensor([2.0, 2.0, 7.0], device="cuda")
        self.assertEqual(sparse_metrics["sparse/mean"], sparse_expected.mean().item())
        self.assertEqual(sparse_metrics["sparse/median"], torch.median(sparse_expected).item())
        self.assertEqual(sparse_metrics["sparse/std"], sparse_expected.std().item())
        self.assertEqual(sparse_metrics["sparse/min"], sparse_expected.min().item())
        self.assertEqual(sparse_metrics["sparse/max"], sparse_expected.max().item())
        self.assertEqual(dict(sparse), {})

        empty = self.tensors_type()
        empty["empty"].append(torch.empty(0, device="cuda"))
        empty_metrics = empty.compute_stats()
        for suffix in ("mean", "median", "std", "min", "max"):
            self.assertTrue(math.isnan(empty_metrics[f"empty/{suffix}"]))
        self.assertEqual(dict(empty), {})


if __name__ == "__main__":
    unittest.main()
