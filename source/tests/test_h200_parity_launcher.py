"""Fail closed before a parity H200 container can use the wrong TOML."""

import os
import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
START = ROOT / "scripts/h200/start_inference.sh"


class H200ParityLauncherTest(unittest.TestCase):
    def _run(self, *args: str, deployment: pathlib.Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["prime_deployment"] = str(deployment)
        return subprocess.run(
            ["bash", str(START), *args],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_config_is_refused_before_docker(self):
        with tempfile.TemporaryDirectory() as directory:
            deployment = pathlib.Path(directory)
            result = self._run(deployment=deployment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("usage:", result.stderr)

    def test_config_outside_isolated_deployment_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            deployment = pathlib.Path(directory) / "deployment"
            deployment.mkdir()
            outside = pathlib.Path(directory) / "outside.toml"
            outside.write_text("[model]\nmax_model_len = 32769\n")
            result = self._run(str(outside), deployment=deployment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside isolated deployment", result.stderr)

    def test_other_in_tree_config_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            deployment = pathlib.Path(directory) / "deployment"
            expected = deployment / "configs/qualification/glm52-policy-parity-h200.toml"
            wrong = deployment / "configs/qualification/other.toml"
            expected.parent.mkdir(parents=True)
            expected.write_text("[model]\nmax_model_len = 32769\n")
            wrong.write_text("[model]\nmax_model_len = 32769\n")
            result = self._run(str(wrong), deployment=deployment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-parity inference config", result.stderr)

    def test_launcher_requires_the_exact_candidate_override(self):
        source = START.read_text()
        self.assertNotIn("prime_inference_config", source)
        self.assertIn('max_num_batched_tokens":32769', source)
        self.assertIn('max_cudagraph_capture_size":256', source)


if __name__ == "__main__":
    unittest.main()
