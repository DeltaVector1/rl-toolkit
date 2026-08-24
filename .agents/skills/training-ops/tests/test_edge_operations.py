import hashlib
import json
import subprocess
import threading
import tempfile
import unittest
import tomllib
from pathlib import Path
from unittest.mock import patch

import training_ops


class EdgeOperationsTest(unittest.TestCase):
    def test_status_parser_keeps_metric_names_with_digits(self):
        self.assertEqual(training_ops._kv("FAILS5M 3\n"), {"FAILS5M": "3"})

    def test_blocked_host_is_refused_before_ssh(self):
        with self.assertRaises(training_ops.Refused):
            training_ops._ssh("n79", "true")

    def test_h200_startup_grace_comes_from_operations_config(self):
        self.assertEqual(
            training_ops._configured_h200_engine_startup_grace(), 7200
        )
        with patch.object(
            training_ops, "_profile", return_value={"kind": "glm52"}
        ), patch.object(training_ops, "_ssh") as ssh, self.assertRaisesRegex(
            training_ops.Refused, "diagnose-glm52-parity"
        ):
            training_ops._repair_engine("h200a")
        ssh.assert_not_called()

    def test_h200_startup_grace_rejects_missing_or_invalid_config(self):
        for value in (None, True, 0, -1, "7200"):
            with patch.dict(training_ops.limits, {"h200_engine_startup_grace_seconds": value}, clear=True):
                with self.assertRaisesRegex(
                    training_ops.Refused, "positive integer"
                ):
                    training_ops._configured_h200_engine_startup_grace()

    def test_glm_qualification_is_fake_data_in_the_isolated_deployment(self):
        profile = training_ops._profile("glm52-native-small")
        with (training_ops.repo_root / profile["trainer"]).open("rb") as stream:
            config = tomllib.load(stream)
        self.assertIn("fake", config["data"])
        self.assertNotIn("rollout_cache", config["data"])
        self.assertEqual(
            training_ops.remote_repo,
            "/data/glm52-rl/deployment-glm52-optims",
        )
        self.assertEqual(training_ops.trainer_container, "glm52-optims-trainer")

    def test_policy_parity_profile_and_stage_are_registered(self):
        profile = training_ops._profile("glm52-policy-parity")
        self.assertEqual(
            profile["trainer"],
            "configs/qualification/glm52-policy-parity-trainer.toml",
        )
        with (training_ops.repo_root / profile["trainer"]).open("rb") as stream:
            config = tomllib.load(stream)
        stage = training_ops.glm_parity_paths["trainer_stage"]
        self.assertEqual(
            config["data"]["fake"]["reference_path"],
            f"{stage}/trainer-reference.json",
        )
        self.assertEqual(
            config["env_vars"]["PRIME_PARITY_ROUTER_TRACE_DIR"],
            "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces",
        )
        self.assertEqual(
            config["env_vars"]["PRIME_PARITY_ROUTER_TRACE_SOURCE"],
            f"{stage}/trainer-router-trace",
        )

    def test_parity_python_missing_numpy_fails_before_host_copy(self):
        failed = subprocess.CompletedProcess(
            [], 1, stdout="", stderr="No module named numpy"
        )
        with (
            patch.object(training_ops.subprocess, "run", return_value=failed),
            patch.object(training_ops.os, "access", return_value=True),
            self.assertRaisesRegex(RuntimeError, "no usable parity Python"),
        ):
            training_ops._parity_python()

        with (
            patch.object(
                training_ops,
                "_parity_python",
                side_effect=RuntimeError("no usable parity Python"),
            ),
            patch.object(training_ops, "_copy_parity_host_evidence") as copy_hosts,
            self.assertRaisesRegex(RuntimeError, "no usable parity Python"),
        ):
            training_ops.collect_glm52_policy_parity("test-run", nodes=["n82"])
        copy_hosts.assert_not_called()

    def test_parity_commands_use_the_selected_interpreter(self):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0)

        selected = "/usr/bin/python3.12"
        with patch.object(training_ops.subprocess, "run", side_effect=run):
            for action in ("collect-trainer", "capture-trainer", "compare"):
                training_ops._run_parity_command(
                    selected,
                    Path("scripts/glm52_policy_parity.py"),
                    Path("configs/qualification/glm52-policy-parity.toml"),
                    action,
                    ["output"],
                    check=action != "compare",
                )

        self.assertEqual(len(calls), 3)
        self.assertEqual([call[0][0] for call in calls], [selected] * 3)
        self.assertEqual(
            [call[0][4] for call in calls],
            ["collect-trainer", "capture-trainer", "compare"],
        )

    def test_parity_host_copies_overlap_and_keep_configured_order(self):
        nodes = ["n82", "n76", "n77"]
        trace_root = training_ops.parity_trace_root
        token_root = training_ops.parity_token_root
        first_trace_copy = threading.Barrier(len(nodes))
        calls = []
        lock = threading.Lock()

        def copy_tree(host, remote, local):
            with lock:
                calls.append((host, remote))
            if remote.startswith(trace_root):
                first_trace_copy.wait(timeout=2)
            local.mkdir(parents=True, exist_ok=True)
            if remote.startswith(trace_root):
                (local / "session.json").write_text("{}\n")
                (local / f"trainer-runtime-pp-0-host-{host}.json").write_text("{}\n")
            else:
                (local / "rank_0.jsonl").write_text("{}\n")

        with tempfile.TemporaryDirectory() as temporary:
            copied = training_ops._copy_parity_host_evidence(
                nodes,
                "run-1",
                Path(temporary),
                trace_root,
                token_root,
                nodes[-1],
                copy_tree,
            )

        self.assertEqual([host for host, _, _ in copied], nodes)
        self.assertEqual(
            {host for host, remote in calls if remote.startswith(trace_root)},
            set(nodes),
        )
        self.assertEqual(
            {host for host, remote in calls if remote == token_root},
            {nodes[-1]},
        )

    def test_parity_host_copies_fail_when_final_tokens_are_missing(self):
        nodes = ["n82", "n76", "n77"]
        trace_root = training_ops.parity_trace_root
        token_root = training_ops.parity_token_root

        def copy_tree(host, remote, local):
            if host == nodes[-1] and remote == token_root:
                raise RuntimeError("final token exports are missing")
            local.mkdir(parents=True, exist_ok=True)
            if remote == f"{trace_root}/run-1":
                (local / "session.json").write_text("{}\n")
                (local / f"trainer-runtime-pp-0-host-{host}.json").write_text("{}\n")

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "final token exports"):
                training_ops._copy_parity_host_evidence(
                    nodes,
                    "run-1",
                    Path(temporary),
                    trace_root,
                    token_root,
                    nodes[-1],
                    copy_tree,
                )

    def test_parity_tar_stream_uses_safe_argv_and_accepts_success(self):
        class Stream:
            def close(self):
                return None

        class Process:
            def __init__(self, stdout=None):
                self.stdout = stdout
                self.returncode = 0

            def communicate(self, timeout=None):
                del timeout
                return b"", b""

            def kill(self):
                return None

        calls = []
        processes = [Process(Stream()), Process()]

        def popen(argv, **kwargs):
            calls.append((argv, kwargs))
            return processes.pop(0)

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "trace"
            with patch.object(training_ops.subprocess, "Popen", side_effect=popen):
                training_ops._copy_parity_tree_stream(
                    "n82",
                    "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces/run-1",
                    destination,
                )

        self.assertEqual(
            calls[0][0],
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "n82",
                "tar",
                "-C",
                "/data/glm52-rl/qualifications/glm52-policy-parity/trainer-traces/run-1",
                "-cf",
                "-",
                ".",
            ],
        )
        self.assertEqual(calls[1][0][0:3], ["tar", "--extract", "--file=-"])
        self.assertNotIn("shell", calls[0][1])
        self.assertNotIn("shell", calls[1][1])

    def test_parity_tar_stream_fails_on_remote_or_local_error(self):
        class Stream:
            def close(self):
                return None

        class Process:
            def __init__(self, returncode, stderr=b"", has_stdout=False):
                self.stdout = Stream() if has_stdout else None
                self.returncode = returncode
                self.stderr = stderr

            def communicate(self, timeout=None):
                del timeout
                return b"", self.stderr

            def kill(self):
                return None

        for remote_code, local_code, message in (
            (1, 0, "remote tar"),
            (0, 1, "local tar"),
        ):
            processes = [
                Process(remote_code, b"remote failed", has_stdout=True),
                Process(local_code, b"local failed"),
            ]

            def popen(argv, **kwargs):
                del argv, kwargs
                return processes.pop(0)

            with tempfile.TemporaryDirectory() as temporary:
                with patch.object(training_ops.subprocess, "Popen", side_effect=popen):
                    with self.assertRaisesRegex(RuntimeError, message):
                        training_ops._copy_parity_tree_stream(
                            "n82",
                            f"{training_ops.parity_trace_root}/run-1",
                            Path(temporary) / "trace",
                        )

    def test_policy_parity_stage_validation_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            reference = stage / "trainer-reference.json"
            session = stage / "trainer-router-trace/session.json"
            session.parent.mkdir()
            reference.write_text("{}\n")
            session.write_text("{}\n")
            profile = (
                training_ops.repo_root
                / "configs/qualification/glm52-policy-parity.toml"
            )
            manifest = {
                "schema_version": 1,
                "kind": "glm52_policy_parity_trainer_stage",
                "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "fixture_sha256": "1" * 64,
                "h200_capture_manifest_sha256": "2" * 64,
                "route_output_scale": 2.5,
                "router_weight_provenance": (
                    "selected_weights_before_route_output_scale"
                ),
                "files": {
                    "trainer-reference.json": hashlib.sha256(
                        reference.read_bytes()
                    ).hexdigest(),
                    "trainer-router-trace/session.json": hashlib.sha256(
                        session.read_bytes()
                    ).hexdigest(),
                },
            }
            (stage / "manifest.json").write_text(json.dumps(manifest) + "\n")
            training_ops._validate_glm_parity_stage(stage)
            extra = stage / "unexpected.bin"
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "unexpected files"):
                training_ops._validate_glm_parity_stage(stage)
            extra.unlink()
            session.write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "checksum differs"):
                training_ops._validate_glm_parity_stage(stage)

    def test_policy_parity_stage_refuses_symlink_and_extra_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.mkdir()
            link = root / "stage"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "regular directory"):
                training_ops._validate_glm_parity_stage(link)

    def test_parity_staging_deploys_inventory_before_script(self):
        events = []

        def deploy(relative, nodes=None):
            events.append(("deploy", relative, nodes))
            return True

        def remote(host, command, timeout=120):
            del timeout
            events.append(("verify", host, command))
            relative = (
                "configs/operations.toml"
                if "configs/operations.toml" in command
                else "scripts/glm52_policy_parity.py"
            )
            digest = hashlib.sha256(
                (training_ops.repo_root / relative).read_bytes()
            ).hexdigest()
            return 0, digest + "\n"

        with (
            patch.object(training_ops, "deploy", side_effect=deploy),
            patch.object(training_ops, "_ssh", side_effect=remote),
        ):
            training_ops._deploy_glm_parity_staging_sources()

        self.assertEqual(
            [event[1] for event in events if event[0] == "deploy"],
            ["configs/operations.toml", "scripts/glm52_policy_parity.py"],
        )
        self.assertEqual(events[0][0:2], ("deploy", "configs/operations.toml"))
        self.assertEqual(events[1][0], "verify")
        self.assertEqual(events[2][0:2], ("deploy", "scripts/glm52_policy_parity.py"))

    def test_parity_staging_refuses_missing_or_mismatched_inventory(self):
        with (
            patch.object(training_ops, "deploy", return_value=True),
            patch.object(training_ops, "_ssh", return_value=(1, "missing")),
        ):
            with self.assertRaisesRegex(RuntimeError, "remote source is missing"):
                training_ops._deploy_glm_parity_staging_sources()
        with (
            patch.object(training_ops, "deploy", return_value=True),
            patch.object(training_ops, "_ssh", return_value=(0, "0" * 64)),
        ):
            with self.assertRaisesRegex(RuntimeError, "checksum differs"):
                training_ops._deploy_glm_parity_staging_sources()

    def test_parity_staging_verifies_sources_before_docker(self):
        source = Path(training_ops.__file__).read_text()
        method = source[source.index("def stage_glm52_policy_parity"):]
        deploy = method.index("_deploy_glm_parity_staging_sources()")
        docker = method.index("sudo docker run")
        self.assertLess(deploy, docker)

    def test_parity_stage_directory_creates_only_exact_missing_path(self):
        commands = []

        def remote(host, command, timeout=120):
            del host, timeout
            commands.append(command)
            return 0, "created\n"

        with patch.object(training_ops, "_ssh", side_effect=remote):
            self.assertEqual(
                training_ops._prepare_glm_parity_stage_directory(), "created"
            )
        command = commands[0]
        exact = training_ops.glm_parity_paths["trainer_stage"]
        self.assertIn(
            'sudo install -d -o 1000 -g 1000 -m 0750 "$stage"', command
        )
        self.assertIn(f"stage={exact}", command)
        self.assertNotIn("chmod -R", command)
        self.assertNotIn("chown -R", command)

    def test_parity_stage_directory_accepts_correct_existing_path(self):
        with patch.object(training_ops, "_ssh", return_value=(0, "existing\n")):
            self.assertEqual(
                training_ops._prepare_glm_parity_stage_directory(), "existing"
            )

    def test_parity_stage_directory_refuses_bad_existing_paths(self):
        for detail in (
            "stage owner or mode differs: 0:0:755",
            "stage has content but no manifest",
            "stage path is a symlink",
        ):
            with patch.object(training_ops, "_ssh", return_value=(44, detail)):
                with self.assertRaisesRegex(RuntimeError, "directory refused"):
                    training_ops._prepare_glm_parity_stage_directory()

    def test_parity_capture_directory_creates_only_exact_missing_path(self):
        commands = []

        def remote(host, command, timeout=120):
            del host, timeout
            commands.append(command)
            return 0, "created\n"

        with patch.object(training_ops, "_ssh", side_effect=remote):
            self.assertEqual(
                training_ops._prepare_glm_parity_capture_directory(), "created"
            )
        command = commands[0]
        exact = training_ops.glm_parity_paths["h200_capture"]
        self.assertIn(
            'sudo install -d -o 1000 -g 1000 -m 0750 "$capture"', command
        )
        self.assertIn(f"capture={exact}", command)
        self.assertNotIn("chmod -R", command)
        self.assertNotIn("chown -R", command)

    def test_parity_capture_directory_accepts_correct_existing_path(self):
        with patch.object(training_ops, "_ssh", return_value=(0, "existing\n")):
            self.assertEqual(
                training_ops._prepare_glm_parity_capture_directory(), "existing"
            )

    def test_parity_capture_directory_refuses_bad_existing_paths(self):
        for detail in (
            "capture owner or mode differs: 0:0:755",
            "capture has content but no manifest",
            "capture path is a symlink",
        ):
            with patch.object(training_ops, "_ssh", return_value=(44, detail)):
                with self.assertRaisesRegex(RuntimeError, "directory refused"):
                    training_ops._prepare_glm_parity_capture_directory()

    def test_qualification_refuses_a_live_trainer(self):
        with patch.object(
            training_ops,
            "_fan",
            return_value={"n82": (0, training_ops.protected_trainer + "\n")},
        ):
            with self.assertRaises(training_ops.Refused):
                training_ops._assert_live_services_are_down(
                    ["n82"], include_inference=False
                )

    def test_live_service_check_retries_one_transient_banner_timeout(self):
        self.assertEqual(
            training_ops._configured_live_service_check(), (2, 45, 60, 2)
        )
        with patch.object(
            training_ops,
            "_ssh",
            side_effect=[
                (255, "Connection timed out during banner exchange"),
                (0, ""),
            ],
        ) as ssh:
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        self.assertEqual(ssh.call_count, 2)
        for call in ssh.call_args_list:
            self.assertEqual(call.kwargs["connect_timeout"], 45)
            self.assertEqual(call.kwargs["timeout"], 60)

    def test_live_service_check_does_not_retry_a_hard_failure(self):
        with patch.object(
            training_ops, "_ssh", return_value=(255, "Permission denied")
        ) as ssh, self.assertRaisesRegex(
            RuntimeError, "cannot check the live trainer"
        ):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        ssh.assert_called_once()

    def test_live_service_check_banner_retry_is_bounded(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(255, "Connection timed out during banner exchange"),
        ) as ssh, self.assertRaisesRegex(
            RuntimeError, "cannot check the live trainer"
        ):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        self.assertEqual(ssh.call_count, 2)

    def test_live_service_check_still_refuses_active_service_after_retry(self):
        with patch.object(
            training_ops,
            "_ssh",
            side_effect=[
                (255, "Connection timed out during banner exchange"),
                (0, training_ops.protected_trainer + "\n"),
            ],
        ) as ssh, self.assertRaises(training_ops.Refused):
            training_ops._assert_live_services_are_down(
                ["n82"], include_inference=False
            )

        self.assertEqual(ssh.call_count, 2)

    def test_live_service_check_uses_configured_worker_bound(self):
        observed = []

        def fan(hosts, function, workers=None):
            observed.append(workers)
            return {host: function(host) for host in hosts}

        with patch.object(training_ops, "_fan", side_effect=fan), patch.object(
            training_ops, "_ssh", return_value=(0, "")
        ):
            training_ops._assert_live_services_are_down(
                ["n82", "n76"], include_inference=False
            )

        self.assertEqual(observed, [2])

    def test_qualification_launch_uses_isolated_inventory(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            return 0, ""

        def fake_fan(hosts, function, workers=None):
            return {host: function(host) for host in hosts}

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", side_effect=fake_fan),
            patch.object(training_ops, "deploy", return_value=True) as deploy_file,
            patch.object(training_ops, "_assert_live_services_are_down"),
            patch.object(training_ops, "trainer_mount_preflight"),
        ):
            training_ops.qualify("glm52-native-small", nodes=["n82"])

        deploy_file.assert_any_call(
            "source/src/prime_rl/trainer/rl/token_export.py", nodes=["n82"]
        )
        launch = next(command for host, command in calls if "cluster.sh trainer" in command)
        self.assertIn("prime_deployment=/data/glm52-rl/deployment-glm52-optims", launch)
        self.assertIn("prime_image=prime-rl:glm52-v080-qual", launch)
        self.assertIn("prime_trainer_container=glm52-optims-trainer", launch)
        self.assertIn(
            "TRAINER_CONFIG=/data/glm52-rl/deployment-glm52-optims/"
            "configs/qualification/glm52-native-small.toml",
            launch,
        )

    def test_current_run_uses_trainer_log_path_when_run_env_is_absent(self):
        command = (
            "--role=trainer\n"
            "--log-dir=/data/glm52-rl/runs/muse19/run_muse19/logs/trainer/torchrun\n"
        )
        with patch.object(training_ops, "_ssh", side_effect=[(1, ""), (0, command)]):
            self.assertEqual(training_ops.current_run(), "muse19")

    def test_start_uses_edge_launcher_and_never_starts_china_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            if command.startswith("test -f "):
                return 0, ""
            return 0, "done\n"

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", return_value={}),
            patch.object(training_ops, "deploy", return_value=True) as deploy_file,
            patch.object(training_ops, "_assert_live_services_are_down"),
            patch.object(training_ops, "trainer_mount_preflight"),
            patch.object(training_ops, "production_start_preflight"),
            patch.object(training_ops.time, "sleep"),
        ):
            training_ops.start("glm52-1", mode="cold", supervisors=False)

        ranked_trainers = [training_ops.trainer_head] + [
            host
            for host in training_ops.trainer_hosts
            if host != training_ops.trainer_head
        ]
        deploy_file.assert_any_call(
            "source/src/prime_rl/trainer/rl/token_export.py",
            nodes=ranked_trainers,
        )
        deploy_file.assert_any_call(
            "source/deps/renderers/renderers/base.py",
            nodes=[training_ops.orchestrator_host],
        )
        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and "scripts/h200/start_orchestrator.sh" in command
                and (
                    "ORCH_CONFIG="
                    + training_ops.remote_repo
                    + "/configs/paused/orchestrator.glm52-production.toml"
                )
                in command
                for host, command in calls
            )
        )
        self.assertFalse(
            any(
                host == training_ops.trainer_head
                and "cluster.sh orchestrator" in command
                for host, command in calls
            )
        )

    def test_start_refuses_a_run_name_that_does_not_match_profile_paths(self):
        with self.assertRaises(training_ops.Refused):
            training_ops.start("wrong-run-name", supervisors=False)

    def test_stop_removes_edge_orchestrator(self):
        calls = []

        def fake_ssh(host, command, timeout=120):
            calls.append((host, command))
            return 0, ""

        with (
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(training_ops, "_fan", return_value={}),
        ):
            training_ops.stop()

        self.assertTrue(
            any(
                host == training_ops.orchestrator_host
                and training_ops.orchestrator_container in command
                and training_ops.environment_container in command
                for host, command in calls
            )
        )


if __name__ == "__main__":
    unittest.main()
