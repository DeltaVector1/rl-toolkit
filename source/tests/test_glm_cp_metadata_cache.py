from __future__ import annotations

import importlib.util
import sys
import threading
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "src/prime_rl/trainer/models/glm_moe_dsa/position_metadata_cache.py"
MODEL_PATH = ROOT / "src/prime_rl/trainer/models/glm_moe_dsa/modeling_glm_moe_dsa.py"
TRAINER_CONFIG_PATH = ROOT / "packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
TRAINER_MODEL_PATH = ROOT / "src/prime_rl/trainer/model.py"
DOCKERFILE_PATH = ROOT.parent / "Dockerfile"


def _load_cache_module():
    spec = importlib.util.spec_from_file_location("glm_position_metadata_cache", CACHE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CACHE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cache_module = _load_cache_module()


def _metadata(module, value: float):
    return module.PositionMetadata(
        position_embeddings=(
            torch.full((1, 4, 2), value),
            torch.full((1, 4, 2), value + 1),
        ),
        ks=torch.arange(4, dtype=torch.int32),
        ke=torch.arange(1, 5, dtype=torch.int32),
    )


class GlmCpPositionMetadataCacheTest(unittest.TestCase):
    def _key(self, positions, hidden, *, group=None, rank=0, world=1, rotary=()):
        return cache_module.PositionMetadataCache.make_key(
            position_ids=positions,
            source_position_ids=positions,
            hidden_states=hidden,
            cp_group=group,
            cp_rank=rank,
            cp_world_size=world,
            rotary_state=rotary,
        )

    def test_default_is_off_and_does_not_retain_metadata(self):
        cache = cache_module.PositionMetadataCache()
        positions = torch.arange(4).view(1, 4)
        hidden = torch.zeros(1, 4, 2)
        key = self._key(positions, hidden)
        calls = 0

        def compute():
            nonlocal calls
            calls += 1
            return _metadata(cache_module, 1)

        cache.get_or_compute(key=key, position_ids=positions, compute=compute)
        cache.get_or_compute(key=key, position_ids=positions, compute=compute)
        self.assertFalse(cache.enabled)
        self.assertEqual(cache.entry_count, 0)
        self.assertEqual(calls, 2)

    def test_forward_and_backward_match_cached_metadata(self):
        cache = cache_module.PositionMetadataCache(max_entries=1, max_bytes=1 << 20)
        positions = torch.arange(4).view(1, 4)
        hidden = torch.randn(1, 4, 2)
        key = self._key(positions, hidden)
        fresh = _metadata(cache_module, 0.25)
        cache.put(key, positions, fresh)
        cached = cache.get(key, positions)
        self.assertIsNotNone(cached)

        def run(value, metadata):
            cos, sin = metadata.position_embeddings
            return (value * (cos + sin)).sum() + metadata.ks.float().sum()

        first = hidden.clone().requires_grad_()
        second = hidden.clone().requires_grad_()
        first_output = run(first, fresh)
        second_output = run(second, cached)
        first_output.backward()
        second_output.backward()
        torch.testing.assert_close(first_output, second_output)
        torch.testing.assert_close(first.grad, second.grad)

    def test_mutation_and_remote_position_changes_miss(self):
        cache = cache_module.PositionMetadataCache(max_entries=2, max_bytes=1 << 20)
        positions = torch.arange(4).view(1, 4)
        hidden = torch.zeros(1, 4, 2)
        key = self._key(positions, hidden, world=2)
        cache.put(key, positions, _metadata(cache_module, 0))
        self.assertIsNotNone(cache.get(key, positions))

        positions.add_(1)
        changed_key = self._key(positions, hidden, world=2)
        self.assertIsNone(cache.get(changed_key, positions))

        unchanged_revision = torch.arange(4).view(1, 4)
        equal_key = self._key(unchanged_revision, hidden, world=2)
        cache.put(equal_key, unchanged_revision, _metadata(cache_module, 2))
        remote_values = torch.arange(4).view(1, 4) + 7
        remote_key = self._key(remote_values, hidden, world=2)
        self.assertIsNone(cache.get(remote_key, remote_values))

    def test_every_parallel_and_shape_field_changes_identity(self):
        cache = cache_module.PositionMetadataCache(max_entries=8, max_bytes=1 << 20)
        positions = torch.arange(4).view(1, 4)
        hidden = torch.zeros(1, 4, 2)
        group = object()
        key = self._key(positions, hidden, group=group, rank=0, world=2, rotary=("r1",))
        cache.put(key, positions, _metadata(cache_module, 0))
        variants = [
            self._key(positions, hidden, group=object(), rank=0, world=2, rotary=("r1",)),
            self._key(positions, hidden, group=group, rank=1, world=2, rotary=("r1",)),
            self._key(positions, hidden, group=group, rank=0, world=3, rotary=("r1",)),
            self._key(positions, hidden, group=group, rank=0, world=2, rotary=("r2",)),
            self._key(positions, torch.zeros(1, 5, 2), group=group, rank=0, world=2, rotary=("r1",)),
            self._key(positions, hidden.to(torch.bfloat16), group=group, rank=0, world=2, rotary=("r1",)),
        ]
        for variant in variants:
            self.assertIsNone(cache.get(variant, positions))

    def test_rotary_revision_changes_after_buffer_update(self):
        class Rotary:
            def __init__(self):
                self.inv_freq = torch.ones(2)
                self.attention_scaling = 1.0
                self.config = type("Config", (), {"revision": "one"})()

        rotary = Rotary()
        first = cache_module.rotary_revision(rotary)
        rotary.inv_freq.add_(1)
        second = cache_module.rotary_revision(rotary)
        self.assertNotEqual(first, second)
        rotary.inv_freq.data.mul_(2)
        third = cache_module.rotary_revision(rotary)
        self.assertNotEqual(second, third)
        rotary.config.revision = "two"
        self.assertNotEqual(third, cache_module.rotary_revision(rotary))

    def test_memory_and_entry_limits_are_hard(self):
        positions = torch.arange(4).view(1, 4)
        hidden = torch.zeros(1, 4, 2)
        probe = _metadata(cache_module, 0)
        entry_bytes = sum(
            value.numel() * value.element_size()
            for value in (*probe.position_embeddings, probe.ks, probe.ke, positions)
        )
        cache = cache_module.PositionMetadataCache(max_entries=2, max_bytes=entry_bytes * 2)
        for value in range(5):
            current = positions + value
            cache.put(self._key(current, hidden), current, _metadata(cache_module, value))
        self.assertLessEqual(cache.entry_count, 2)
        self.assertLessEqual(cache.bytes, entry_bytes * 2)

    def test_concurrent_access_keeps_bounds_and_values(self):
        cache = cache_module.PositionMetadataCache(max_entries=1, max_bytes=1 << 20)
        positions = torch.arange(4).view(1, 4)
        hidden = torch.zeros(1, 4, 2)
        key = self._key(positions, hidden)
        failures = []

        def worker():
            try:
                value = cache.get_or_compute(
                    key=key,
                    position_ids=positions,
                    compute=lambda: _metadata(cache_module, 3),
                )
                if value.ks.tolist() != [0, 1, 2, 3]:
                    failures.append("wrong value")
            except Exception as error:  # pragma: no cover - failure detail
                failures.append(repr(error))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(cache.entry_count, 1)
        self.assertLessEqual(cache.bytes, 1 << 20)

    def test_model_uses_opt_in_cache_and_explicit_invalidation(self):
        source = MODEL_PATH.read_text()
        config = TRAINER_CONFIG_PATH.read_text()
        trainer_model = TRAINER_MODEL_PATH.read_text()
        dockerfile = DOCKERFILE_PATH.read_text()
        self.assertIn('getattr(config, "cp_position_metadata_cache_size", 0)', source)
        self.assertIn("self._position_metadata_cache.enabled", source)
        self.assertIn("self.model.clear_position_metadata_cache()", source)
        self.assertIn("source_position_ids=position_ids", source)
        self.assertIn("cp_position_metadata_cache_size: int = Field(0, ge=0)", config)
        self.assertIn("model_config.cp_position_metadata_cache_size = config.cp_position_metadata_cache_size", trainer_model)
        self.assertIn("position_metadata_cache.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
