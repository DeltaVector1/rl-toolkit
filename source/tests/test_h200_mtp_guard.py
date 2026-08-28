"""Keep the production H200 profile off the unstable native-MTP path."""

import pathlib
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
START = (ROOT / "scripts/h200/start_inference.sh").read_text()
OPERATIONS = tomllib.loads((ROOT / "configs/operations.toml").read_text())


class H200MTPGuardTest(unittest.TestCase):
    def test_native_mtp_is_not_enabled(self):
        self.assertNotIn('"speculative_config"', START)

    def test_production_launcher_names_the_actual_mtp_fixes(self):
        self.assertIn("PRs #46908 and #47013", START)
        self.assertIn("workspace-chunking patch", START)
        self.assertNotIn("2,415,919,104", START)
        self.assertNotIn("stride", START.lower())

    def test_glm52_production3_keeps_mtp_disabled(self):
        identity = OPERATIONS["profiles"]["glm52-production3"]["identity"]
        self.assertEqual(identity["mtp_method"], "disabled")
        self.assertEqual(identity["mtp_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
