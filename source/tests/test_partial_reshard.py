import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch


SOURCE = pathlib.Path(__file__).resolve().parents[1]


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Mesh:
    def __init__(self, ranks, ndim, sizes):
        self.mesh = ranks
        self.ndim = ndim
        self.sizes = sizes

    def size(self, dim=None):
        return self.sizes[dim] if dim is not None else self.sizes[0]


class PartialReshardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configs = types.ModuleType("prime_rl.configs.trainer")
        configs.ModelConfig = object
        logger = types.ModuleType("prime_rl.utils.logger")
        logger.get_logger = lambda: types.SimpleNamespace(
            info=lambda *args, **kwargs: None
        )
        modules = {
            "prime_rl": types.ModuleType("prime_rl"),
            "prime_rl.configs": types.ModuleType("prime_rl.configs"),
            "prime_rl.configs.trainer": configs,
            "prime_rl.utils": types.ModuleType("prime_rl.utils"),
            "prime_rl.utils.logger": logger,
        }
        with mock.patch.dict(sys.modules, modules):
            cls.parallel_dims = load_module(
                SOURCE / "src/prime_rl/trainer/parallel_dims.py",
                "partial_reshard_parallel_dims",
            )
        cls.partial = load_module(
            SOURCE / "src/prime_rl/trainer/distributed/partial_reshard.py",
            "partial_reshard_under_test",
        )

    def test_full_world_layout_is_created_once(self):
        dims = self.parallel_dims.ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=8,
            pp=8,
            ep=8,
            world_size=64,
        )
        dims._world_mesh = Mesh(torch.arange(64).reshape(8, 1, 8), 3, (64,))
        dims._submeshes["hsdp"] = Mesh(torch.arange(8), 1, (8,))
        calls = []

        class CapturedMesh:
            def __init__(self, device, ranks, mesh_dim_names):
                calls.append((device, ranks.clone(), mesh_dim_names))
                self.ranks = ranks

            def __getitem__(self, names):
                return (self, names)

        with mock.patch.object(self.parallel_dims, "DeviceMesh", CapturedMesh):
            first = dims.get_partial_reshard_mesh(4)
            second = dims.get_partial_reshard_mesh(4)

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0][1].shape), (8, 2, 4))
        self.assertEqual(calls[0][1].reshape(-1).tolist(), list(range(64)))

    def test_invalid_partial_size_fails_before_group_creation(self):
        dims = self.parallel_dims.ParallelDims(1, 1, 8, 8, 8, 64)
        dims._world_mesh = Mesh(torch.arange(64).reshape(8, 1, 8), 3, (64,))
        dims._submeshes["hsdp"] = Mesh(torch.arange(8), 1, (8,))
        for size in (1, 2, 3, 8):
            with self.assertRaises(ValueError):
                dims.get_partial_reshard_mesh(size)

    def test_unsupported_topology_fails_before_group_creation(self):
        dims = self.parallel_dims.ParallelDims(1, 1, 8, 4, 8, 32)
        with self.assertRaisesRegex(ValueError, "requires PP8, CP8, EP8"):
            dims.get_partial_reshard_mesh(4)

    def test_prebuilt_mesh_is_attached_to_group_and_parameters(self):
        metadata_shapes = []

        class Parameter:
            is_dtensor = False
            _orig_size = torch.Size((17, 5))

            def _init_sharded_post_forward_param_metadata(self, tensor):
                metadata_shapes.append(tensor.shape)

        group = types.SimpleNamespace(
            mesh_info=types.SimpleNamespace(shard_mesh_size=8),
            post_forward_mesh_info=None,
            fsdp_params=[Parameter(), Parameter()],
        )
        state = types.SimpleNamespace(
            _fsdp_param_group=group,
            _auto_reshard_after_forward=True,
        )
        module = types.SimpleNamespace(_get_fsdp_state=lambda: state)
        mesh = Mesh(torch.arange(8).reshape(2, 4), 2, (2, 4))

        class MeshInfo:
            def __init__(self, mesh, shard_mesh_dim, replicate_mesh_dim):
                self.mesh = mesh
                self.shard_mesh_dim = shard_mesh_dim
                self.replicate_mesh_dim = replicate_mesh_dim

        with mock.patch.object(self.partial, "HSDPMeshInfo", MeshInfo):
            count = self.partial.set_partial_reshard_mesh(module, mesh)

        self.assertEqual(count, 2)
        self.assertFalse(state._auto_reshard_after_forward)
        self.assertTrue(
            all(
                p.post_forward_mesh_info is group.post_forward_mesh_info
                for p in group.fsdp_params
            )
        )
        self.assertEqual(metadata_shapes, [torch.Size((17, 5)), torch.Size((17, 5))])

    def test_routed_experts_keep_ordinary_reshard(self):
        source = (SOURCE / "src/prime_rl/trainer/model.py").read_text()
        expert_call = source.split("fully_shard(\n                    block_mlp.experts", 1)[1]
        expert_call = expert_call.split("                )", 1)[0]
        self.assertIn("**packed_fsdp_config", expert_call)
        self.assertNotIn("set_partial_reshard_mesh", expert_call)


if __name__ == "__main__":
    unittest.main()
