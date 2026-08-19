"""Keep the production H200 profile off the unstable native-MTP path."""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
START = (ROOT / "scripts/h200/start_inference.sh").read_text()


class H200MTPGuardTest(unittest.TestCase):
    def test_native_mtp_is_not_enabled(self):
        self.assertNotIn('"speculative_config"', START)

    def test_flashmla_stride_failure_is_documented(self):
        self.assertIn("2,415,919,104-byte stride", START)
        self.assertIn("exceeds FlashMLA's int32 limit", START)


if __name__ == "__main__":
    unittest.main()
