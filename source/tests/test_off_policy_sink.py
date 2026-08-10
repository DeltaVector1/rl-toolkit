"""Regression tests for queue-inclusive off-policy rejection in TrainSink."""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest
import uuid
from dataclasses import dataclass
from unittest import mock


SOURCE = pathlib.Path(__file__).resolve().parents[1]


@dataclass
class _TrainBatch:
    rollouts: list
    samples: list


class _TrainRollouts(list):
    pass


class _Logger:
    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _load_train_sink():
    modules = {
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.configs": types.ModuleType("prime_rl.configs"),
        "prime_rl.configs.orchestrator": types.ModuleType("prime_rl.configs.orchestrator"),
        "prime_rl.orchestrator": types.ModuleType("prime_rl.orchestrator"),
        "prime_rl.orchestrator.envs": types.ModuleType("prime_rl.orchestrator.envs"),
        "prime_rl.orchestrator.filters": types.ModuleType("prime_rl.orchestrator.filters"),
        "prime_rl.orchestrator.metrics": types.ModuleType("prime_rl.orchestrator.metrics"),
        "prime_rl.orchestrator.trajectories": types.ModuleType("prime_rl.orchestrator.trajectories"),
        "prime_rl.orchestrator.types": types.ModuleType("prime_rl.orchestrator.types"),
        "prime_rl.transport": types.ModuleType("prime_rl.transport"),
        "prime_rl.utils": types.ModuleType("prime_rl.utils"),
        "prime_rl.utils.logger": types.ModuleType("prime_rl.utils.logger"),
    }
    modules["prime_rl.configs.orchestrator"].OrchestratorConfig = object
    modules["prime_rl.orchestrator.envs"].TrainEnvs = object
    modules["prime_rl.orchestrator.filters"].RolloutFilter = object
    modules["prime_rl.orchestrator.filters"].apply_filters = lambda *args: None
    modules["prime_rl.orchestrator.metrics"].TrainRollouts = _TrainRollouts
    modules["prime_rl.orchestrator.trajectories"].trace_to_samples = lambda *args, **kwargs: []
    modules["prime_rl.orchestrator.types"].Rollout = object
    modules["prime_rl.orchestrator.types"].TrainBatch = _TrainBatch
    modules["prime_rl.transport"].TrainingSample = object
    modules["prime_rl.utils.logger"].get_logger = lambda: _Logger()
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "train_sink_under_test",
            SOURCE / "src/prime_rl/orchestrator/train_sink.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


@dataclass
class _Sample:
    policy_version: int
    token_ids: list[int]


class _Rollout:
    def __init__(self, policy_version: int, group_id: uuid.UUID):
        self.policy_version = policy_version
        self.group_id = group_id
        self.env_name = "live"
        self.samples = [_Sample(policy_version, [1])]
        self.is_filtered = False
        self.filter_results = {}
        self.num_total_tokens = 1
        self.task = types.SimpleNamespace(data=types.SimpleNamespace(idx=0))


class OffPolicySinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_train_sink()

    def make_sink(self, *, target: int, maximum: int, batch_size: int):
        sink = self.module.TrainSink.__new__(self.module.TrainSink)
        sink.config = types.SimpleNamespace(max_off_policy_steps=maximum)
        live = types.SimpleNamespace(sampler=types.SimpleNamespace(samples_from_live_policy=True))
        sink.train_envs = types.SimpleNamespace(get=lambda _name: live)
        sink.training_policy_version = target
        sink.batch_size = batch_size
        sink.token_batch_size = None
        sink.pending_tokens = 0
        sink.pending_batch = []
        sink.pending_groups = {}
        sink.pending_group_episodes = {}
        sink.pending_rollouts = _TrainRollouts()
        sink.post_filters = []
        return sink

    @staticmethod
    def groups(policy_version: int, count: int) -> list[_Rollout]:
        return [
            _Rollout(policy_version, group_id)
            for _ in range(count // 16)
            for group_id in [uuid.uuid4()]
            for _ in range(16)
        ]

    def test_batch9_rejects_policy3_overflow_and_replenishes(self):
        sink = self.make_sink(target=7, maximum=4, batch_size=1024)
        stale = self.groups(3, 128)
        policy4 = self.groups(4, 896)
        sink.pending_batch = [*stale, *policy4]

        # Batch 8 targeted policy v7, so policy v3 was still at the limit.
        # Advancing to batch 9 targets v8 and makes the queued v3 groups stale.
        self.assertEqual(sink.advance_training_policy_version(8), 128)
        self.assertIsNone(sink.process_batch())
        self.assertTrue(all(r.is_filtered and not r.samples for r in stale))

        replacements = self.groups(4, 128)
        sink.pending_batch.extend(replacements)
        batch = sink.process_batch()
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(len(batch.samples), 1024)
        self.assertEqual({sample.policy_version for sample in batch.samples}, {4})

    def test_completed_stale_group_never_enters_pending_batch(self):
        sink = self.make_sink(target=8, maximum=4, batch_size=16)
        stale = self.groups(3, 16)
        group_id = stale[0].group_id
        sink.pending_groups[group_id] = stale
        sink.pending_group_episodes[group_id] = 16

        asyncio.run(sink.process_group(group_id))

        self.assertEqual(sink.pending_batch, [])
        self.assertTrue(all(r.is_filtered and not r.samples for r in stale))

    def test_zero_limit_accepts_current_policy_and_rejects_one_step_old(self):
        sink = self.make_sink(target=0, maximum=0, batch_size=16)
        current = self.groups(0, 16)
        sink.pending_batch = current
        batch = sink.process_batch()
        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual({sample.policy_version for sample in batch.samples}, {0})

        old = self.groups(0, 16)
        sink.pending_batch = old
        self.assertEqual(sink.advance_training_policy_version(1), 16)
        self.assertIsNone(sink.process_batch())
        self.assertTrue(all(r.is_filtered and not r.samples for r in old))


if __name__ == "__main__":
    unittest.main()
