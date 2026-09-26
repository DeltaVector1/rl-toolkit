import copy
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT / "configs/trinity-h200-parity-production/inference.toml"
DIAGNOSTIC = ROOT / "configs/trinity-h200-batch-invariant-diagnostic/inference.toml"


def _load(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


class H200BatchInvariantConfigTest(unittest.TestCase):
    def test_diagnostic_has_only_batch_invariant_env_change(self):
        production = _load(PRODUCTION)
        diagnostic = _load(DIAGNOSTIC)

        self.assertNotIn("VLLM_BATCH_INVARIANT", production["env_vars"])
        expected = copy.deepcopy(production)
        expected["env_vars"]["VLLM_BATCH_INVARIANT"] = "1"
        self.assertEqual(diagnostic, expected)
        self.assertIs(type(diagnostic["env_vars"]["VLLM_BATCH_INVARIANT"]), str)


if __name__ == "__main__":
    unittest.main()
