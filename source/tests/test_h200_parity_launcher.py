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

    def test_isolated_operation_can_require_all_container_names_absent(self):
        source = START.read_text()
        self.assertIn("prime_require_absent", source)
        self.assertIn("refusing to replace an existing isolated container", source)

    def test_launcher_mounts_only_the_non_secret_bootstrap_fifo_path(self):
        source = START.read_text()
        self.assertNotIn("PRIME_PARITY_BOOTSTRAP_TOKEN", source)
        self.assertIn("prime_parity_bootstrap_fifo", source)
        self.assertIn("! -p \"$bootstrap_fifo\"", source)
        self.assertIn("1000:1000:600", source)
        self.assertIn(
            "source=$bootstrap_fifo,target=/run/prime-parity-bootstrap,readonly",
            source,
        )
        self.assertIn(
            "PRIME_PARITY_BOOTSTRAP_FIFO=/run/prime-parity-bootstrap",
            source,
        )
        self.assertIn('restart_policy=no', source)

    def test_parity_handlers_use_bounded_stream_reader(self):
        server = (ROOT / "source/src/prime_rl/inference/vllm/server.py").read_text()
        reset = server.split('@router.post("/parity/router-trace/reset")', 1)[1].split(
            "@router.get(\"/parity/router-trace\")", 1
        )[0]
        diagnostic = server.split('@router.post("/parity/router-diagnostic")', 1)[1].split(
            "@router.post(\"/init_broadcaster\")", 1
        )[0]
        self.assertNotIn("request.json()", reset)
        self.assertNotIn("request.json()", diagnostic)
        self.assertIn("bounded_request_body(request, 64 * 1024)", reset)
        self.assertIn("bounded_request_body(request, 16 * 1024 * 1024)", diagnostic)
        self.assertGreater(
            reset.index("consume_single_use_bootstrap_token"),
            reset.index("write_session(root, session)"),
        )
        self.assertIn(
            "parity_diagnostic_bootstrap_token = consume_single_use_bootstrap_token(",
            reset,
        )


if __name__ == "__main__":
    unittest.main()
