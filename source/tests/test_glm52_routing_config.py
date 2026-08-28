import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "src"))

from prime_rl.glm52_routing import (  # noqa: E402
    glm_routing_errors,
    require_glm_routing_config,
)
from scripts.glm52_h200_startup_preflight import load_model_config  # noqa: E402


class Glm52RoutingConfigTest(unittest.TestCase):
    @staticmethod
    def valid_config() -> SimpleNamespace:
        return SimpleNamespace(
            model_type="glm_moe_dsa",
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            num_experts_per_tok=8,
            norm_topk_prob=True,
            routed_scaling_factor=2.5,
        )

    def test_valid_trainer_values_pass(self):
        require_glm_routing_config(
            self.valid_config(),
            router_frozen=True,
            correction_bias_present=True,
            side="trainer",
        )

    def test_wrong_score_and_missing_bias_fail_closed(self):
        config = self.valid_config()
        config.scoring_func = "softmax"
        errors = glm_routing_errors(
            config,
            router_frozen=False,
            correction_bias_present=False,
        )
        self.assertIn("scoring_func must be 'sigmoid', got 'softmax'", errors)
        self.assertIn("router_frozen must be true", errors)
        self.assertIn("correction bias is missing", errors)

    def test_boolean_route_setting_is_not_accepted_as_integer(self):
        config = self.valid_config()
        config.norm_topk_prob = 1
        self.assertIn(
            "norm_topk_prob must be True, got 1",
            glm_routing_errors(config),
        )

    def test_model_config_loader_rejects_non_object(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as stream:
            stream.write("[]")
            stream.flush()
            with self.assertRaisesRegex(ValueError, "must be an object"):
                load_model_config(Path(stream.name))

    def test_tracked_sources_match_accepted_image_files(self):
        expected = {
            ROOT / "source/src/prime_rl/trainer/models/kernels/fp8_indexer.py": (
                "b7d68e6ced3a33a527c50b7caaab0833e0b818d318a906b51ab4a912769606a3"
            ),
            ROOT / "source/src/prime_rl/trainer/models/layers/rotary_emb.py": (
                "148a2ee0f7ad4c9b26194215c730f60f21b0dd3d94fcf29c8700f3f917cf094e"
            ),
        }
        for path, digest in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
