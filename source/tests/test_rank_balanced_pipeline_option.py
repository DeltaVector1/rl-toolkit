from __future__ import annotations

import pathlib
import re
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/qualification/glm52-native-interleaved-rank-balanced.toml"
EXPECTED_BOUNDARIES = [
    0, 3, 7, 11, 15, 19, 23, 27, 31,
    36, 42, 48, 54, 60, 66, 72, 78,
]
EXPECTED_TARGETS = {
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert",
    "experts",
}


class RankBalancedPipelineOptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open("rb") as file:
            cls.config = tomllib.load(file)

    def test_option_is_explicit_and_disabled_elsewhere(self):
        model = self.config["model"]
        self.assertEqual(model["pp_schedule"], "interleaved_1f1b")
        self.assertEqual(model["pp_boundary_plan"], "explicit")
        self.assertEqual(model["pp"], 8)
        self.assertNotIn("pp_virtual_layer_boundaries", tomllib.loads(
            (ROOT / "configs/qualification/glm52-native-b2048.toml").read_text()
        )["model"])

    def test_layers_are_owned_once_by_pp8_pairs(self):
        bounds = self.config["model"]["pp_virtual_layer_boundaries"]
        self.assertEqual(bounds, EXPECTED_BOUNDARIES)
        ranges = list(zip(bounds, bounds[1:]))
        self.assertEqual(len(ranges), 16)
        owned = [layer for start, end in ranges for layer in range(start, end)]
        self.assertEqual(len(owned), 78)
        self.assertEqual(sorted(owned), list(range(78)))

        counts = [end - start for start, end in ranges]
        self.assertEqual(
            [counts[rank] + counts[rank + 8] for rank in range(8)],
            [8, 10, 10, 10, 10, 10, 10, 10],
        )
        self.assertEqual(counts[0] + counts[8], 8)

    def test_cutoff_maps_to_layer_48(self):
        model = self.config["model"]
        physical = model["pp_layer_boundaries"]
        cutoff = model["ac"]["keep_routed_expert_activations_from_pp_stage"]
        self.assertEqual(physical[cutoff], 48)
        self.assertEqual(model["pp_virtual_layer_boundaries"][11], 48)

    def test_lora_targets_are_complete(self):
        self.assertEqual(
            set(self.config["model"]["lora"]["target_modules"]),
            EXPECTED_TARGETS,
        )

    def test_option_contains_no_host_assignment(self):
        text = CONFIG_PATH.read_text()
        self.assertNotRegex(text, r"(?im)^\s*(host|hostname|ssh)\s*=")
        self.assertNotRegex(text, r"\bn(?:79|82|84|85)\b")


if __name__ == "__main__":
    unittest.main()
