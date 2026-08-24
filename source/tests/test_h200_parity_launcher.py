"""Fail closed before a parity H200 container can use the wrong TOML."""

import ast
import os
import pathlib
import subprocess
import tempfile
import tomllib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
START = ROOT / "scripts/h200/start_inference.sh"


class H200ParityLauncherTest(unittest.TestCase):
    @staticmethod
    def _identity_helper(name: str) -> str:
        source = START.read_text().split(f"{name}() {{", 1)[1]
        return source.split("<<'PY'\n", 1)[1].split("\nPY\n}", 1)[0]

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
        self.assertIn('"host":"127.0.0.1"', source)
        self.assertIn('"kv_cache_dtype":"auto"', source)
        self.assertNotIn('"kv_cache_dtype":"fp8"', source)
        self.assertIn('max_num_batched_tokens":32769', source)
        self.assertIn('max_cudagraph_capture_size":256', source)

    def test_parity_router_is_external_and_backend_is_loopback(self):
        source = START.read_text()
        config_path = ROOT / "configs/qualification/glm52-policy-parity-h200.toml"
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
        self.assertEqual(config["server"]["host"], "0.0.0.0")
        self.assertEqual(config["server"]["port"], 8000)
        self.assertEqual(config["backend_port"], 8100)
        self.assertIn(
            "expected_vllm_extra='{\"host\":\"127.0.0.1\"",
            source,
        )
        self.assertIn('--vllm-extra "$vllm_extra"', source)

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
        self.assertIn("st.st_uid==0 and st.st_gid==0", source)
        self.assertIn('re.fullmatch(rb"[0-9a-f]{64}",data)', source)
        self.assertIn(
            "source=$bootstrap_fifo,target=/run/prime-parity-bootstrap",
            source,
        )
        self.assertNotIn(
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

    def test_launcher_records_and_validates_exact_container_identity(self):
        source = START.read_text()
        launch = source.index("container_id=$(docker run -d")
        identity = source.index("parity-engine-identity:")
        root_check = source.index('"$(id -u):$(id -g)" != "0:0"')
        self.assertLess(root_check, launch)
        self.assertIn("refusing parity engine launch by a non-root user", source)
        self.assertLess(launch, identity)
        self.assertIn('[[ ! "$container_id" =~ ^[0-9a-f]{64}$ ]]', source)
        self.assertIn("/proc/sys/kernel/random/boot_id", source)
        self.assertIn("{{.Id}}", source)
        self.assertIn("{{.Name}}", source)
        self.assertIn("{{.Config.Image}}", source)
        self.assertIn("{{json .Config.Cmd}}", source)
        self.assertIn("{{.RestartCount}}", source)
        self.assertIn("{{.HostConfig.RestartPolicy.Name}}", source)
        self.assertIn("{{json .HostConfig.Init}}", source)
        self.assertIn("\n  --init \\\n", source)
        self.assertIn('"${identity[6]}" != "true"', source)
        self.assertIn("trap cleanup_unreported_container EXIT", source)
        self.assertIn('docker rm -f "$container_id"', source)
        self.assertIn('--cidfile "$engine_cidfile"', source)
        self.assertIn("prime_parity_engine_cidfile", source)
        self.assertIn("prime_parity_engine_identity_file", source)
        self.assertIn("engine-identity.json", source)
        self.assertIn("1000:1000:600", source)
        self.assertLess(identity, source.index("mapfile -t identity", launch))
        self.assertLess(identity, source.index("trap - EXIT", identity))

    def test_identity_writer_requires_root_for_cid_temp_and_final_files(self):
        writer = self._identity_helper("write_engine_identity")
        self.assertEqual(writer.count("st.st_uid==0 and st.st_gid==0"), 3)

    def test_cid_reader_rejects_symlink_without_touching_target(self):
        code = self._identity_helper("read_engine_cid")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            work = root / "work"
            work.mkdir(mode=0o750)
            outside = root / "outside"
            outside.write_text("outside\n")
            outside.chmod(0o640)
            (work / "engine.cid").symlink_to(outside)
            result = subprocess.run(
                ["python3", "-c", code, str(work), "engine.cid"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(outside.read_text(), "outside\n")
            self.assertEqual(outside.stat().st_mode & 0o777, 0o640)

    def test_cid_reader_rejects_uid_1000_regular_file(self):
        code = self._identity_helper("read_engine_cid")
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory) / "work"
            work.mkdir(mode=0o750)
            cid = work / "engine.cid"
            cid.write_bytes(b"d" * 64)
            cid.chmod(0o600)
            result = subprocess.run(
                ["python3", "-c", code, str(work), cid.name],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)

    def test_cid_reader_accepts_exact_raw_format_for_expected_owner(self):
        code = self._identity_helper("read_engine_cid").replace(
            "st.st_uid==0 and st.st_gid==0",
            "st.st_uid==1000 and st.st_gid==1000",
        )
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory) / "work"
            work.mkdir(mode=0o750)
            cid = work / "engine.cid"
            cid.write_bytes(b"d" * 64)
            cid.chmod(0o600)
            result = subprocess.run(
                ["python3", "-c", code, str(work), cid.name],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "d" * 64 + "\n")

    def test_identity_writer_rejects_tmp_symlink_without_touching_target(self):
        code = self._identity_helper("write_engine_identity").replace(
            "st.st_uid==0 and st.st_gid==0",
            "st.st_uid==1000 and st.st_gid==1000",
        )
        container_id = "d" * 64
        boot_id = "11111111-2222-3333-4444-555555555555"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            work = root / "work"
            work.mkdir(mode=0o750)
            cid = work / "engine.cid"
            cid.write_text(container_id)
            cid.chmod(0o600)
            outside = root / "outside"
            outside.write_text("outside\n")
            outside.chmod(0o640)
            (work / "engine-identity.json.tmp").symlink_to(outside)
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    code,
                    str(work),
                    cid.name,
                    "engine-identity.json",
                    container_id,
                    boot_id,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(outside.read_text(), "outside\n")
            self.assertEqual(outside.stat().st_mode & 0o777, 0o640)

    def test_identity_writer_replaces_final_symlink_without_touching_target(self):
        code = self._identity_helper("write_engine_identity").replace(
            "st.st_uid==0 and st.st_gid==0",
            "st.st_uid==1000 and st.st_gid==1000",
        )
        container_id = "d" * 64
        boot_id = "11111111-2222-3333-4444-555555555555"
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            work = root / "work"
            work.mkdir(mode=0o750)
            cid = work / "engine.cid"
            cid.write_text(container_id)
            cid.chmod(0o600)
            outside = root / "outside"
            outside.write_text("outside\n")
            outside.chmod(0o640)
            identity = work / "engine-identity.json"
            identity.symlink_to(outside)
            result = subprocess.run(
                [
                    "python3",
                    "-c",
                    code,
                    str(work),
                    cid.name,
                    identity.name,
                    container_id,
                    boot_id,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(identity.is_symlink())
            self.assertEqual(identity.stat().st_mode & 0o777, 0o600)
            self.assertEqual(outside.read_text(), "outside\n")
            self.assertEqual(outside.stat().st_mode & 0o777, 0o640)

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
            "request.app.state.parity_bootstrap_token = consume_single_use_bootstrap_token(",
            reset,
        )
        self.assertIn("bootstrap_readiness(", bootstrap_ready)
        self.assertNotIn("consume_single_use_bootstrap_token", bootstrap_ready)
        self.assertIn('readiness == "not_ready"', bootstrap_ready)
        self.assertIn("status_code=503", bootstrap_ready)
        self.assertIn('readiness != "ready"', bootstrap_ready)
        self.assertIn("status_code=401", bootstrap_ready)
        self.assertIn('"process_id": os.getpid()', bootstrap_ready)
        self.assertIn('"boot_nonce": getattr(', bootstrap_ready)
        self.assertIn('"token_sha256": hashlib.sha256(', bootstrap_ready)

    def test_api_process_issues_bootstrap_once_before_serving(self):
        server = (ROOT / "source/src/prime_rl/inference/vllm/server.py").read_text()
        app_init = server.split("async def custom_init_app_state", 1)[1].split(
            "import vllm.entrypoints.openai.api_server", 1
        )[0]
        entry = server.split("def server(", 1)[1]
        self.assertEqual(app_init.count("issue_parity_bootstrap_token"), 1)
        self.assertIn("state.parity_bootstrap_token = bootstrap_token", app_init)
        self.assertNotIn("issue_parity_bootstrap_token", entry)
        self.assertNotIn("parity_diagnostic_bootstrap_token:", server)
        self.assertIn(
            'getattr(request.app.state, "parity_bootstrap_token", None)',
            server,
        )
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
                    and node.id == "issue_parity_bootstrap_token"
                ):
                    parents.append(self.function)
                self.generic_visit(node)

        Calls().visit(ast.parse(server))
        self.assertEqual(parents, ["custom_init_app_state"])
        tp_worker = (
            ROOT / "source/src/prime_rl/inference/vllm/worker/filesystem.py"
        ).read_text()
        self.assertNotIn("issue_parity_bootstrap_token", tp_worker)


if __name__ == "__main__":
    unittest.main()
