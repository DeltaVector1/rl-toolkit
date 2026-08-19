"""Regression checks for bounded native-FP8 pinned host memory."""

from pathlib import Path
import unittest


TRAIN = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/trainer/rl/train.py"
).read_text()


class FP8HostCacheCleanupTest(unittest.TestCase):
    def test_offload_cache_is_released_after_backward_and_optimizer(self):
        optimizer = TRAIN.index("optimizer.step()")
        synchronize = TRAIN.index("torch.cuda.synchronize()", optimizer)
        release = TRAIN.index('getattr(torch._C, "_host_emptyCache", None)', synchronize)
        broadcast = TRAIN.index("weight_broadcast.broadcast_weights", release)
        self.assertLess(optimizer, synchronize)
        self.assertLess(synchronize, release)
        self.assertLess(release, broadcast)

    def test_missing_runtime_support_fails_closed(self):
        self.assertIn(
            "PRIME_FP8_EXPERT_ACTIVATION_OFFLOAD requires torch._C._host_emptyCache",
            TRAIN,
        )


if __name__ == "__main__":
    unittest.main()
