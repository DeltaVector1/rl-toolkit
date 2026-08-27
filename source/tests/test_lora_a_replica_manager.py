"""Pure CPU/fake-collective tests for the opt-in EP LoRA-A replica manager."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from prime_rl.trainer.distributed.lora_a_replica_manager import (  # noqa: E402
    LoRAAReplicaManager,
    ReplicaCollectiveError,
    ReplicaLifecycleError,
    ReplicaManagerDisabledError,
    ReplicaMemoryError,
    StaleReplicaError,
)


class TwoRankCollective:
    """Deterministic two-rank fake with rank-order expert-major semantics."""

    def __init__(self, shards: list[np.ndarray], gradients: list[np.ndarray], *, fail=False):
        self.shards = shards
        self.gradients = gradients
        self.fail = fail
        self.gather_calls = 0
        self.reduce_calls = 0

    def all_gather(self, local, **_kwargs):
        if self.fail:
            raise RuntimeError("fake all-gather failure")
        self.gather_calls += 1
        return [part.copy() for part in self.shards]

    def reduce_scatter(self, full_gradient, **_kwargs):
        if self.fail:
            raise RuntimeError("fake reduce-scatter failure")
        self.reduce_calls += 1
        total = sum(self.gradients)
        # This fake represents rank zero's owner shard. The second manager in
        # a real run receives the second contiguous expert chunk.
        rows = full_gradient.shape[0] // 2
        return total[:rows].copy()


class ListStyleCollective(TwoRankCollective):
    """Fake the Torch list-style reduce_scatter(output, input_list) API."""

    def reduce_scatter(self, output, input_list, **_kwargs):
        total = sum(self.gradients)
        output[...] = total[: output.shape[0]]


class Optimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.param_groups = [{"params": [parameter]}]
        self.calls = 0

    def step(self):
        self.parameter += 1
        self.calls += 1


class LoRAAReplicaManagerTest(unittest.TestCase):
    def make_manager(self, *, budget=None, collective=None):
        shard0 = np.arange(8, dtype=np.float16).reshape(2, 4)
        shard1 = (np.arange(8, dtype=np.float16) + 100).reshape(2, 4)
        grad0 = np.ones_like(shard0)
        grad1 = np.full_like(shard1, 2)
        fake = collective or TwoRankCollective([shard0, shard1], [grad0, grad1])
        kwargs = {
            "ep_size": 2,
            "collective": fake,
            "enabled": True,
        }
        if budget is not None:
            kwargs["layer_peak_budget_bytes"] = budget
            kwargs["total_peak_budget_bytes"] = max(budget, 1024)
        manager = LoRAAReplicaManager(**kwargs)
        return manager, fake, shard0, shard1, grad0, grad1

    def test_replica_equality_preserves_rank_and_expert_order(self):
        manager, fake, shard0, shard1, *_ = self.make_manager()
        manager.register_layer("mlp.0", shard0, version=3)
        manager.begin_step(11)
        got = manager.replica("mlp.0", version=3)
        np.testing.assert_array_equal(got, np.concatenate([shard0, shard1], axis=0))
        self.assertIs(got, manager.replica("mlp.0", version=3))
        self.assertEqual(fake.gather_calls, 1)
        self.assertEqual(manager.memory.replica_live_bytes, got.nbytes)
        manager.end_step()
        self.assertEqual(manager.memory.replica_live_bytes, 0)
        self.assertEqual(manager.collective_stats.replica_lifetime_steps, 1)

    def test_reduce_scatter_updates_only_local_gradient(self):
        manager, fake, shard0, _, grad0, grad1 = self.make_manager()
        manager.register_layer("mlp.0", shard0, version=0)
        manager.begin_step(0)
        manager.replica("mlp.0")
        full_grad = np.concatenate([grad0, grad0], axis=0)
        target = np.zeros_like(shard0)
        got = manager.reduce_scatter_gradient("mlp.0", full_grad, target=target)
        np.testing.assert_array_equal(got, grad0 + grad1)
        np.testing.assert_array_equal(target, got)
        self.assertEqual(fake.reduce_calls, 1)
        self.assertEqual(manager.memory.full_gradient_live_bytes, 0)
        self.assertEqual(manager.collective_stats.reduce_scatter_wire_bytes, full_grad.nbytes - shard0.nbytes)
        manager.end_step()

    def test_list_style_reduce_scatter_is_supported(self):
        manager, _, shard0, shard1, grad0, grad1 = self.make_manager()
        manager.collective = ListStyleCollective([shard0, shard1], [grad0, grad1])
        manager.register_layer("mlp.0", shard0)
        manager.begin_step(0)
        replica = manager.replica("mlp.0")
        got = manager.reduce_scatter_gradient("mlp.0", np.ones_like(replica))
        np.testing.assert_array_equal(got, grad0 + grad1)
        manager.end_step()

    def test_optimizer_updates_local_parameter_and_invalidates_replica(self):
        manager, _, shard0, _, *_ = self.make_manager()
        manager.register_layer("mlp.0", shard0, version=4)
        manager.begin_step(4)
        old = manager.replica("mlp.0")
        optimizer = Optimizer(shard0)
        manager.assert_optimizer_owns_local_shards(optimizer)
        manager.optimizer_step(optimizer, versions={"mlp.0": 5})
        self.assertEqual(optimizer.calls, 1)
        with self.assertRaises(StaleReplicaError):
            manager.replica("mlp.0", version=4)
        manager.end_step()
        manager.begin_step(5)
        refreshed = manager.replica("mlp.0", version=5)
        self.assertIsNot(old, refreshed)
        np.testing.assert_array_equal(refreshed[:2], shard0)
        manager.end_step()

    def test_stale_version_is_rejected_before_collective(self):
        manager, fake, shard0, *_ = self.make_manager()
        manager.register_layer("mlp.0", shard0, version="policy-v2")
        manager.begin_step(1)
        with self.assertRaises(StaleReplicaError):
            manager.replica("mlp.0", version="policy-v1")
        self.assertEqual(fake.gather_calls, 0)
        manager.end_step()

    def test_error_aborts_and_cleans_all_layers(self):
        manager, fake, shard0, *_ = self.make_manager(collective=None)
        fake.fail = True
        manager.register_layer("mlp.0", shard0)
        manager.begin_step(0)
        with self.assertRaises(ReplicaCollectiveError):
            manager.replica("mlp.0")
        self.assertFalse(manager.active)
        self.assertEqual(manager.memory.replica_live_bytes, 0)
        self.assertEqual(manager.memory.full_gradient_live_bytes, 0)
        with self.assertRaises(ReplicaLifecycleError):
            manager.end_step()

    def test_multiple_layers_and_microbatches_keep_replicas_but_not_full_grads(self):
        manager, _, shard0, *_ = self.make_manager()
        manager.register_layer("mlp.0", shard0, version=0)
        manager.register_layer("mlp.1", shard0 + 10, version=0)
        manager.begin_step(0)
        first = manager.replica("mlp.0")
        second = manager.replica("mlp.1")
        for _ in range(3):
            manager.reduce_scatter_gradient("mlp.0", np.ones_like(first))
            manager.reduce_scatter_gradient("mlp.1", np.ones_like(second))
            self.assertEqual(manager.memory.full_gradient_live_bytes, 0)
        self.assertEqual(manager.memory.replica_live_bytes, first.nbytes + second.nbytes)
        manager.end_step()
        self.assertEqual(manager.memory.replica_live_bytes, 0)

    def test_gate_up_pair_reduces_each_table_without_retaining_full_dA(self):
        manager, fake, shard0, shard1, *_ = self.make_manager()
        manager.register_gate_up_layer("mlp.0", shard0, shard1, version=2)
        manager.begin_step(2)
        pair = manager.gate_up_replica("mlp.0", version=2)
        np.testing.assert_array_equal(pair.gate, np.concatenate([shard0, shard1]))
        np.testing.assert_array_equal(pair.up, np.concatenate([shard0, shard1]))
        gate, up = manager.reduce_scatter_gate_up_gradients(
            "mlp.0",
            np.ones_like(pair.gate),
            np.full_like(pair.up, 2),
        )
        self.assertEqual(gate.shape, shard0.shape)
        self.assertEqual(up.shape, shard0.shape)
        self.assertEqual(fake.reduce_calls, 2)
        self.assertEqual(manager.memory.full_gradient_live_bytes, 0)
        manager.mark_optimizer_step({"mlp.0": 3})
        manager.end_step()

    def test_memory_limits_fail_closed(self):
        shard_bytes = np.zeros((2, 4), dtype=np.float16).nbytes
        manager, _, shard0, *_ = self.make_manager(budget=shard_bytes * 2 - 1)
        manager.register_layer("mlp.0", shard0)
        manager.begin_step(0)
        with self.assertRaises(ReplicaMemoryError):
            manager.replica("mlp.0")
        self.assertFalse(manager.active)
        self.assertEqual(manager.memory.replica_live_bytes, 0)

    def test_context_manager_cleans_on_error_and_feature_is_opt_in(self):
        disabled = LoRAAReplicaManager(ep_size=1, enabled=False)
        with self.assertRaises(ReplicaManagerDisabledError):
            disabled.begin_step(0)
        manager, _, shard0, *_ = self.make_manager()
        manager.register_layer("mlp.0", shard0)
        with self.assertRaises(ValueError):
            with manager.step(0):
                manager.replica("mlp.0")
                raise ValueError("failed backward")
        self.assertFalse(manager.active)
        self.assertEqual(manager.memory.replica_live_bytes, 0)

    def test_source_compute_failure_and_bad_gradient_abort_step(self):
        manager, _, shard0, *_ = self.make_manager()
        manager.register_layer("mlp.0", shard0)
        manager.begin_step(0)
        manager.replica("mlp.0")
        with self.assertRaises(ValueError):
            manager.source_compute_and_reduce_scatter(
                "mlp.0", lambda _table: np.zeros((1, 1), dtype=np.float16)
            )
        self.assertFalse(manager.active)
        self.assertEqual(manager.memory.replica_live_bytes, 0)

        manager.begin_step(1)
        manager.replica("mlp.0")
        with self.assertRaises(ValueError):
            manager.reduce_scatter_gradient(
                "mlp.0", np.zeros((1, 1), dtype=np.float16)
            )
        self.assertFalse(manager.active)
        self.assertEqual(manager.memory.full_gradient_live_bytes, 0)


if __name__ == "__main__":
    unittest.main()
