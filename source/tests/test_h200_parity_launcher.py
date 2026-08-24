"""Fail closed before a parity H200 container can use the wrong TOML."""

import ast
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

    def test_trainer_launcher_passes_rank_zero_host_for_bundle_validation(self):
        source = (ROOT / "scripts/china/cluster.sh").read_text()
        self.assertIn('trainer_head_node=${trainer_nodes%% *}', source)
        self.assertIn('-e "PRIME_TRAINER_HEAD=n$trainer_head_node"', source)

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

    def test_launcher_requires_the_exact_prepared_router_trace_path(self):
        source = START.read_text()
        config = (
            ROOT / "configs/qualification/glm52-policy-parity-h200.toml"
        ).read_text()
        self.assertIn("prime_parity_trace_dir", source)
        self.assertIn(".diagnostic-work-[0-9a-f]{16}/h200-router-trace", source)
        self.assertIn("1000:1000:750", source)
        self.assertIn(
            'PRIME_PARITY_ROUTER_TRACE_DIR="$parity_trace_dir"', source
        )
        self.assertNotIn("PRIME_PARITY_ROUTER_TRACE_DIR", config)
        launch = source.index("docker run -d")
        self.assertLess(
            source.index("install -d -o 1000 -g 1000 -m 0755 /data/glm52-rl/lora-cache"),
            launch,
        )
        self.assertLess(
            source.index('install -d -o 1000 -g 1000 -m 0755 "$deployment/cache/vllm"'),
            launch,
        )

    def test_parity_handlers_use_bounded_stream_reader(self):
        server = (ROOT / "source/src/prime_rl/inference/vllm/server.py").read_text()
        bootstrap_ready = server.split(
            '@router.get("/parity/bootstrap-ready")', 1
        )[1].split("async def bounded_request_body", 1)[0]
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
        self.assertIn("bootstrap_readiness(", bootstrap_ready)
        self.assertNotIn("consume_single_use_bootstrap_token", bootstrap_ready)
        self.assertIn('readiness == "not_ready"', bootstrap_ready)
        self.assertIn("status_code=503", bootstrap_ready)
        self.assertIn('readiness != "ready"', bootstrap_ready)
        self.assertIn("status_code=401", bootstrap_ready)

    def test_api_process_reads_bootstrap_fifo_once_before_serving(self):
        server = (ROOT / "source/src/prime_rl/inference/vllm/server.py").read_text()
        app_init = server.split("async def custom_init_app_state", 1)[1].split(
            "import vllm.entrypoints.openai.api_server", 1
        )[0]
        entry = server.split("def server(", 1)[1]
        self.assertEqual(app_init.count("read_parity_bootstrap_fifo"), 1)
        self.assertIn("parity_diagnostic_bootstrap_token is not None", app_init)
        self.assertNotIn("read_parity_bootstrap_fifo", entry)
        parents = []

        class Calls(ast.NodeVisitor):
            def __init__(self):
                self.function = None

            def visit_FunctionDef(self, node):
                previous = self.function
                self.function = node.name
                self.generic_visit(node)
                self.function = previous

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Name(self, node):
                if (
                    isinstance(node.ctx, ast.Load)
                    and node.id == "read_parity_bootstrap_fifo"
                ):
                    parents.append(self.function)
                self.generic_visit(node)

        Calls().visit(ast.parse(server))
        self.assertEqual(parents, ["custom_init_app_state"])
        tp_worker = (
            ROOT / "source/src/prime_rl/inference/vllm/worker/filesystem.py"
        ).read_text()
        self.assertNotIn("read_parity_bootstrap_fifo", tp_worker)


if __name__ == "__main__":
    unittest.main()
