import base64
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import training_ops


class DeployTest(unittest.TestCase):
    @staticmethod
    def _unix_socket(path):
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        return listener

    def test_control_master_reuses_live_unix_socket_and_cleans_adopted_state(self):
        host = training_ops.trainer_head
        with tempfile.TemporaryDirectory() as directory:
            socket = Path(directory) / "configured-master"
            listener = self._unix_socket(socket)

            try:
                def run(command, **kwargs):
                    del kwargs
                    if "-G" in command:
                        return Mock(
                            returncode=0,
                            stdout=f"controlmaster auto\ncontrolpath {socket}\n",
                            stderr="",
                        )
                    if "-O" in command:
                        return Mock(returncode=0, stdout="Master running\n", stderr="")
                    raise AssertionError(f"unexpected SSH command: {command}")

                with patch.dict(
                    training_ops._ssh_control_paths, {}, clear=True
                ), patch.object(training_ops.subprocess, "run", side_effect=run) as subprocess_run:
                    with training_ops._ssh_control_masters([host]):
                        self.assertEqual(
                            training_ops._active_ssh_control_options(host),
                            ["-o", "ControlMaster=no", "-o", f"ControlPath={socket}"],
                        )

                commands = [call.args[0] for call in subprocess_run.call_args_list]
                self.assertFalse(any("-MNf" in command for command in commands))
                self.assertFalse(any("exit" in command for command in commands))
                self.assertEqual(sum("-O" in command for command in commands), 1)
                self.assertEqual(training_ops._ssh_control_paths, {})
            finally:
                listener.close()

    def test_unmanaged_ssh_disables_configured_multiplexing(self):
        host = training_ops.trainer_head
        with patch.dict(training_ops._ssh_control_paths, {}, clear=True):
            self.assertEqual(
                training_ops._ssh_control_options(host),
                ["-o", "ControlMaster=no", "-o", "ControlPath=none"],
            )

    def test_unmanaged_builders_ignore_configured_socket(self):
        host = training_ops.trainer_head
        with patch.dict(training_ops._ssh_control_paths, {}, clear=True), patch.object(
            training_ops,
            "_configured_ssh_control_path",
            return_value="/tmp/live-configured-master",
        ) as configured:
            ssh = training_ops._ssh_transport_command(host, "true")
            scp = training_ops._scp_transport_command(
                host, "source", f"{host}:destination", prefix_options=("-q",)
            )
        configured.assert_not_called()
        for command in (ssh, scp):
            self.assertIn("ControlMaster=no", command)
            self.assertIn("ControlPath=none", command)
            self.assertNotIn("ControlPath=/tmp/live-configured-master", command)
            self.assertIn("StrictHostKeyChecking=accept-new", command)

    def test_managed_ssh_uses_only_recorded_control_path(self):
        host = training_ops.trainer_head
        with patch.dict(
            training_ops._ssh_control_paths,
            {host: "/tmp/managed-training-ops"},
            clear=True,
        ):
            self.assertEqual(
                training_ops._ssh_control_options(host),
                [
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "ControlPath=/tmp/managed-training-ops",
                ],
            )

    def test_managed_builders_use_recorded_control_path(self):
        host = training_ops.trainer_head
        path = "/tmp/managed-training-ops"
        with patch.dict(training_ops._ssh_control_paths, {host: path}, clear=True):
            commands = (
                training_ops._ssh_transport_command(host, "true"),
                training_ops._scp_transport_command(
                    host, "source", f"{host}:destination"
                ),
            )
        for command in commands:
            self.assertIn("ControlMaster=no", command)
            self.assertIn(f"ControlPath={path}", command)
            self.assertNotIn("ControlPath=none", command)

    def test_adapter_download_uses_unmanaged_transport_builder(self):
        with patch.dict(training_ops._ssh_control_paths, {}, clear=True), patch.object(
            training_ops,
            "_configured_ssh_control_path",
            side_effect=AssertionError("adapter download inspected configured state"),
        ), patch.object(training_ops, "_local_command") as local_command:
            training_ops._download_adapter_file(
                "/data/glm52-rl/runs/muse19/adapter.json", Path("/tmp/adapter.json")
            )
        command = local_command.call_args.args[0]
        self.assertEqual(command[0:2], ["scp", "-q"])
        self.assertIn("ControlMaster=no", command)
        self.assertIn("ControlPath=none", command)

    def test_remote_adapter_copy_script_disables_configured_multiplexing(self):
        path = f"{training_ops.run_root}/muse19/adapter_model.safetensors"
        with patch.object(training_ops, "_ssh", return_value=(0, "")) as ssh:
            training_ops._copy_resume_adapter(path, [training_ops.trainer_head, "n76"])
        script = ssh.call_args.args[1]
        self.assertIn(
            "ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=no -o ControlPath=none)",
            script,
        )
        self.assertIn('-e "ssh ${ssh_opts[*]}"', script)
        self.assertNotIn('ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$target"', script)

    def test_remote_adapter_copy_keeps_special_path_data_out_of_remote_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "adapter ' ; $HOME value\nnext"
            source.write_bytes(b"adapter\n")
            ssh_log = root / "ssh.jsonl"
            scp_log = root / "scp.jsonl"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_ssh = fake_bin / "ssh"
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['SSH_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "operations = {'final_digest', 'make_directory', 'partial_digest', 'install'}\n"
                "operation = next((value for value in sys.argv if value in operations), None)\n"
                "if operation is not None: sys.stdin.buffer.read()\n"
                "if operation == 'final_digest': raise SystemExit(1)\n"
                "if operation == 'partial_digest': print(os.environ['EXPECTED_DIGEST'])\n"
                "raise SystemExit(0 if operation in {'make_directory', 'partial_digest', 'install'} else 1)\n"
            )
            fake_ssh.chmod(0o700)
            fake_scp = fake_bin / "scp"
            fake_scp.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['SCP_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            fake_scp.chmod(0o700)
            with patch.object(training_ops, "run_root", str(root)), patch.object(
                training_ops, "_ssh", return_value=(0, "")
            ) as outer_ssh:
                training_ops._copy_resume_adapter(
                    str(source), [training_ops.trainer_head, "n76"]
                )
            script = outer_ssh.call_args.args[1]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "SSH_LOG": str(ssh_log),
                    "SCP_LOG": str(scp_log),
                    "EXPECTED_DIGEST": digest,
                }
            )
            completed = subprocess.run(
                ["bash", "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            ssh_calls = [json.loads(line) for line in ssh_log.read_text().splitlines()]
            operation_names = {
                "final_digest", "make_directory", "partial_digest", "install"
            }
            remote_calls = {
                next(value for value in call if value in operation_names): call
                for call in ssh_calls
                if any(value in operation_names for value in call)
            }
            self.assertEqual(set(remote_calls), {
                "final_digest", "make_directory", "partial_digest", "install"
            })
            for call in remote_calls.values():
                self.assertNotIn(str(source), call)
                operation = next(value for value in call if value in operation_names)
                marker = call.index(operation)
                self.assertEqual(
                    base64.b64decode(call[marker + 1]).decode(),
                    str(source),
                )
            scp_calls = [json.loads(line) for line in scp_log.read_text().splitlines()]
            self.assertEqual(len(scp_calls), 1)
            self.assertTrue(any(str(source) in call for call in scp_calls))
            self.assertTrue(
                all("/tmp/prime-rl-adapter." in value for call in scp_calls for value in call if ":" in value)
            )

    def test_ssh_command_does_not_use_configured_socket_without_manager(self):
        with patch.dict(training_ops._ssh_control_paths, {}, clear=True), patch.object(
            training_ops,
            "_ssh_control_options",
            return_value=["-o", "ControlMaster=no", "-o", "ControlPath=none"],
        ), patch.object(
            training_ops.subprocess,
            "run",
            return_value=Mock(returncode=0, stdout="ready\n", stderr=""),
        ) as run:
            self.assertEqual(training_ops._ssh("n82", "true"), (0, "ready\n"))

        command = run.call_args.args[0]
        self.assertIn("ControlMaster=no", command)
        self.assertIn("ControlPath=none", command)

    def test_ssh_command_uses_managed_socket(self):
        with patch.dict(
            training_ops._ssh_control_paths,
            {"n82": "/tmp/managed-training-ops"},
            clear=True,
        ), patch.object(
            training_ops.subprocess,
            "run",
            return_value=Mock(returncode=0, stdout="ready\n", stderr=""),
        ) as run:
            self.assertEqual(training_ops._ssh("n82", "true"), (0, "ready\n"))

        command = run.call_args.args[0]
        self.assertIn("ControlMaster=no", command)
        self.assertIn("ControlPath=/tmp/managed-training-ops", command)

    def test_unmanaged_ssh_does_not_inspect_stale_configured_socket(self):
        with patch.dict(training_ops._ssh_control_paths, {}, clear=True), patch.object(
            training_ops,
            "_configured_ssh_control_path",
            side_effect=AssertionError("unmanaged code inspected configured state"),
        ), patch.object(
            training_ops.subprocess,
            "run",
            return_value=Mock(returncode=0, stdout="ready\n", stderr=""),
        ) as run:
            self.assertEqual(training_ops._ssh("n82", "true"), (0, "ready\n"))
        self.assertIn("ControlPath=none", run.call_args.args[0])

    def test_control_master_refuses_final_symlink(self):
        host = training_ops.trainer_head
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            listener = self._unix_socket(target)
            configured = root / "configured-master"
            configured.symlink_to(target)

            try:
                def run(command, **kwargs):
                    del kwargs
                    if "-G" in command:
                        return Mock(returncode=0, stdout=f"controlpath {configured}\n", stderr="")
                    raise AssertionError(f"unexpected SSH command: {command}")

                with patch.dict(
                    training_ops._ssh_control_paths, {}, clear=True
                ), patch.object(training_ops.subprocess, "run", side_effect=run) as subprocess_run:
                    with self.assertRaisesRegex(
                        training_ops.Refused, "configured SSH control path is a symlink"
                    ):
                        with training_ops._ssh_control_masters([host]):
                            pass
                self.assertFalse(any("-MNf" in call.args[0] for call in subprocess_run.call_args_list))
            finally:
                listener.close()

    def test_control_master_refuses_symlinked_parent(self):
        host = training_ops.trainer_head
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_parent = root / "real"
            real_parent.mkdir()
            configured_parent = root / "configured"
            configured_parent.symlink_to(real_parent, target_is_directory=True)
            target = real_parent / "master"
            listener = self._unix_socket(target)
            configured = configured_parent / "master"

            try:
                def run(command, **kwargs):
                    del kwargs
                    if "-G" in command:
                        return Mock(returncode=0, stdout=f"controlpath {configured}\n", stderr="")
                    raise AssertionError(f"unexpected SSH command: {command}")

                with patch.dict(
                    training_ops._ssh_control_paths, {}, clear=True
                ), patch.object(training_ops.subprocess, "run", side_effect=run) as subprocess_run:
                    with self.assertRaisesRegex(
                        training_ops.Refused, "configured SSH control path has a symlinked parent"
                    ):
                        with training_ops._ssh_control_masters([host]):
                            pass
                self.assertFalse(any("-MNf" in call.args[0] for call in subprocess_run.call_args_list))
            finally:
                listener.close()

    def test_control_master_refuses_configured_regular_file(self):
        host = training_ops.trainer_head
        with tempfile.TemporaryDirectory() as directory:
            configured = Path(directory) / "configured-master"
            configured.touch()

            def run(command, **kwargs):
                del kwargs
                if "-G" in command:
                    return Mock(returncode=0, stdout=f"controlpath {configured}\n", stderr="")
                raise AssertionError(f"unexpected SSH command: {command}")

            with patch.dict(
                training_ops._ssh_control_paths, {}, clear=True
            ), patch.object(training_ops.subprocess, "run", side_effect=run) as subprocess_run:
                with self.assertRaisesRegex(
                    training_ops.Refused, "not a Unix socket"
                ):
                    with training_ops._ssh_control_masters([host]):
                        pass
            self.assertFalse(any("-MNf" in call.args[0] for call in subprocess_run.call_args_list))

    def test_control_master_refuses_existing_stale_configured_socket(self):
        host = training_ops.trainer_head
        with tempfile.TemporaryDirectory() as directory:
            socket = Path(directory) / "stale-master"
            listener = self._unix_socket(socket)

            try:
                def run(command, **kwargs):
                    del kwargs
                    if "-G" in command:
                        return Mock(returncode=0, stdout=f"controlpath {socket}\n", stderr="")
                    if "-O" in command:
                        return Mock(returncode=255, stdout="", stderr="")
                    raise AssertionError(f"unexpected SSH command: {command}")

                with patch.dict(
                    training_ops._ssh_control_paths, {}, clear=True
                ), patch.object(training_ops.subprocess, "run", side_effect=run) as subprocess_run:
                    with self.assertRaisesRegex(
                        training_ops.Refused,
                        "configured SSH control master is stale.*exited with status 255",
                    ):
                        with training_ops._ssh_control_masters([host]):
                            pass

                commands = [call.args[0] for call in subprocess_run.call_args_list]
                self.assertFalse(any("-MNf" in command for command in commands))
                self.assertEqual(training_ops._ssh_control_paths, {})
            finally:
                listener.close()

    def test_deploy_refuses_duplicate_nodes_before_fanout(self):
        with patch.object(training_ops, "_ssh") as ssh, patch.object(
            training_ops, "_ssh_control_masters"
        ) as masters:
            with self.assertRaisesRegex(training_ops.Refused, "deploy nodes must be unique"):
                training_ops.deploy("scripts/china/cluster.sh", nodes=["n78", "n78"])
        ssh.assert_not_called()
        masters.assert_not_called()

    def test_qualify_refuses_duplicate_nodes_before_fanout(self):
        with patch.object(training_ops, "_ssh") as ssh, patch.object(
            training_ops, "_ssh_control_masters"
        ) as masters:
            with self.assertRaisesRegex(
                training_ops.Refused, "qualification nodes must be unique"
            ):
                training_ops.qualify("glm52-native-small", nodes=["n78", "n78"])
        ssh.assert_not_called()
        masters.assert_not_called()

    def test_run_on_hosts_supports_inventory_trainer_head_group(self):
        seen = []

        def fan(hosts, function, workers=None):
            del workers
            seen.append(list(hosts))
            return {host: function(host) for host in hosts}

        with patch.object(training_ops, "_fan", side_effect=fan), patch.object(
            training_ops, "_ssh", return_value=(0, "ready\n")
        ) as ssh:
            result = training_ops.run_on_hosts("true", group="trainer_head")

        self.assertEqual(seen, [[training_ops.trainer_head]])
        self.assertEqual(result, {training_ops.trainer_head: (0, "ready\n")})
        ssh.assert_called_once_with(training_ops.trainer_head, "true", 120)

    def test_run_on_hosts_trainer_head_group_keeps_blocked_host_check(self):
        with patch.object(training_ops, "trainer_head", "n79"), patch.object(
            training_ops, "_ssh"
        ) as ssh:
            with self.assertRaisesRegex(training_ops.Refused, "n79 is off limits"):
                training_ops.run_on_hosts("true", group="trainer_head")
        ssh.assert_not_called()

    def test_control_master_starts_once_per_configured_host(self):
        hosts = [*training_ops.trainer_hosts, training_ops.trainer_hosts[0]]
        completed = Mock(returncode=0, stdout="", stderr="")
        with patch.dict(
            training_ops._ssh_control_paths, {}, clear=True
        ), patch.object(
            training_ops.subprocess, "run", return_value=completed
        ) as run, patch.object(
            training_ops,
            "_check_started_ssh_control_master",
            return_value=(True, ""),
        ):
            with training_ops._ssh_control_masters(hosts):
                with training_ops._ssh_control_masters(reversed(hosts)):
                    for host in training_ops.trainer_hosts:
                        self.assertIn(
                            "ControlPath=",
                            " ".join(training_ops._active_ssh_control_options(host)),
                        )

        commands = [call.args[0] for call in run.call_args_list]
        starts = [command for command in commands if "-MNf" in command]
        closes = [
            command
            for command in commands
            if "-O" in command and "exit" in command
        ]
        self.assertEqual(len(starts), len(training_ops.trainer_hosts))
        self.assertEqual(len(closes), len(training_ops.trainer_hosts))
        self.assertEqual(starts[0][-1], training_ops.trainer_head)
        self.assertTrue(
            all("ControlMaster=auto" in command for command in starts)
        )
        self.assertEqual(
            {command[-1] for command in starts}, set(training_ops.trainer_hosts)
        )
        self.assertEqual(
            len({command[command.index("-S") + 1] for command in starts}),
            len(training_ops.trainer_hosts),
        )

    def test_control_master_failure_has_bounded_retry_and_cleanup(self):
        first = training_ops.trainer_head
        second = next(host for host in training_ops.trainer_hosts if host != first)
        completed = Mock(returncode=0, stdout="", stderr="")
        refused = Mock(returncode=255, stdout="", stderr="Exceeded MaxStartups")

        def run(command, **kwargs):
            del kwargs
            if "-MNf" in command and command[-1] == second:
                return refused
            return completed

        with patch.dict(
            training_ops._ssh_control_paths, {}, clear=True
        ), patch.object(
            training_ops.subprocess, "run", side_effect=run
        ) as subprocess_run, patch.object(
            training_ops,
            "_check_started_ssh_control_master",
            return_value=(True, ""),
        ), patch.object(
            training_ops.time, "sleep"
        ) as sleep, self.assertRaisesRegex(
            RuntimeError, "after 3 attempts.*Exceeded MaxStartups"
        ):
            with training_ops._ssh_control_masters([first, second]):
                pass

        commands = [call.args[0] for call in subprocess_run.call_args_list]
        attempts = [
            command
            for command in commands
            if "-MNf" in command and command[-1] == second
        ]
        closes = [command for command in commands if "-O" in command]
        self.assertEqual(len(attempts), training_ops.ssh_master_attempts)
        self.assertEqual(sleep.call_count, training_ops.ssh_master_attempts - 1)
        self.assertLessEqual(
            len(closes), training_ops.ssh_master_attempts + 1
        )
        self.assertEqual(training_ops._ssh_control_paths, {})

    def test_control_master_cleanup_failure_is_bounded_and_keeps_socket(self):
        host = training_ops.trainer_head
        started = Mock(returncode=0, stdout="", stderr="")
        refused = Mock(returncode=255, stdout="", stderr="permission denied")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "pool"
            directory.mkdir()

            def run(command, **kwargs):
                del kwargs
                if "-G" in command:
                    return Mock(returncode=0, stdout="", stderr="")
                path = Path(command[command.index("-S") + 1])
                if "-MNf" in command:
                    path.touch()
                    return started
                return refused

            with patch.dict(
                training_ops._ssh_control_paths, {}, clear=True
            ), patch.object(
                training_ops.tempfile, "mkdtemp", return_value=str(directory)
            ), patch.object(
                training_ops.subprocess, "run", side_effect=run
            ) as subprocess_run, patch.object(
                training_ops,
                "_check_started_ssh_control_master",
                return_value=(True, ""),
            ), patch.object(
                training_ops.time, "sleep"
            ), self.assertRaisesRegex(
                RuntimeError, "control sockets retained"
            ):
                with training_ops._ssh_control_masters([host]):
                    pass

            closes = [
                call.args[0]
                for call in subprocess_run.call_args_list
                if "-O" in call.args[0]
            ]
            self.assertEqual(
                len(closes), training_ops.ssh_master_cleanup_attempts
            )
            socket = directory / hashlib.sha256(host.encode()).hexdigest()[:16]
            self.assertTrue(socket.exists())
            socket.unlink()
            directory.rmdir()

    def test_control_master_refuses_blocked_host_before_subprocess(self):
        blocked = next(iter(training_ops.blocked_hosts))
        with patch.object(training_ops.subprocess, "run") as run, self.assertRaises(
            training_ops.Refused
        ):
            with training_ops._ssh_control_masters([blocked]):
                pass
        run.assert_not_called()

    def test_started_control_master_requires_a_live_socket(self):
        host = training_ops.trainer_head
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "master"
            listener = self._unix_socket(path)
            try:
                with patch.object(
                    training_ops.subprocess,
                    "run",
                    return_value=Mock(returncode=0, stdout="Master running\n", stderr=""),
                ) as run:
                    self.assertEqual(
                        training_ops._check_started_ssh_control_master(host, str(path)),
                        (True, ""),
                    )
                command = run.call_args.args[0]
                self.assertIn("-O", command)
                self.assertIn("check", command)
            finally:
                listener.close()

    def test_started_control_master_reports_missing_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            ready, detail = training_ops._check_started_ssh_control_master(
                training_ops.trainer_head, str(Path(directory) / "missing")
            )
        self.assertFalse(ready)
        self.assertEqual(detail, "control socket was not created")

    def test_ssh_reports_silent_failure_with_exit_status(self):
        with patch.object(
            training_ops,
            "_ssh_control_options",
            return_value=["-o", "ControlMaster=no", "-o", "ControlPath=none"],
        ), patch.object(
            training_ops.subprocess,
            "run",
            return_value=Mock(returncode=255, stdout="", stderr=""),
        ):
            self.assertEqual(
                training_ops._ssh(training_ops.trainer_head, "true"),
                (255, "ssh exited with status 255"),
            )

    def test_mount_preflight_refuses_missing_local_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(training_ops, "repo_root", Path(directory)),
                patch.object(training_ops, "_mounts_in", return_value={"source/missing.py"}),
            ):
                with self.assertRaisesRegex(training_ops.Refused, "missing locally"):
                    training_ops.trainer_mount_preflight(["n82"])

    def test_mount_preflight_refuses_stale_remote_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "source" / "mounted.py"
            source.parent.mkdir(parents=True)
            source.write_text("current\n")
            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounts_in", return_value={"source/mounted.py"}),
                patch.object(
                    training_ops,
                    "_fan",
                    return_value={"n82": (0, "source/mounted.py\tdeadbeef\n")},
                ),
            ):
                with self.assertRaisesRegex(training_ops.Refused, "preflight failed"):
                    training_ops.trainer_mount_preflight(["n82"])

    def test_mount_preflight_accepts_matching_remote_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "source" / "mounted.py"
            source.parent.mkdir(parents=True)
            source.write_text("current\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounts_in", return_value={"source/mounted.py"}),
                patch.object(
                    training_ops,
                    "_fan",
                    return_value={"n82": (0, f"source/mounted.py\t{digest}\n")},
                ),
            ):
                self.assertEqual(
                    training_ops.trainer_mount_preflight(["n82"]),
                    {"source/mounted.py": digest},
                )

    def test_mount_preflight_refuses_unexpected_or_duplicate_records(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "source" / "mounted.py"
            source.parent.mkdir(parents=True)
            source.write_text("current\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounts_in", return_value={"source/mounted.py"}),
                patch.object(
                    training_ops,
                    "_fan",
                    return_value={
                        "n82": (
                            0,
                            f"source/mounted.py\t{digest}\n"
                            f"source/extra.py\t{'0' * 64}\n",
                        )
                    },
                ),
            ):
                with self.assertRaisesRegex(training_ops.Refused, "unexpected"):
                    training_ops.trainer_mount_preflight(["n82"])

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounts_in", return_value={"source/mounted.py"}),
                patch.object(
                    training_ops,
                    "_fan",
                    return_value={
                        "n82": (
                            0,
                            f"source/mounted.py\t{digest}\n"
                            f"source/mounted.py\t{digest}\n",
                        )
                    },
                ),
            ):
                with self.assertRaisesRegex(training_ops.Refused, "duplicate"):
                    training_ops.trainer_mount_preflight(["n82"])

    def test_adapter_bucket_folder_uses_hyphenated_muse_run_name(self):
        self.assertEqual(training_ops._adapter_bucket_run_folder("muse18"), "muse-18-run")
        self.assertEqual(training_ops._adapter_bucket_run_folder("muse-19"), "muse-19-run")

    def test_adapter_broadcast_parser_requires_one_run_tree(self):
        run_base = "/data/glm52-rl/runs/muse18"
        parsed = training_ops._parse_adapter_broadcasts(
            "\n".join(
                [
                    f"{run_base}/run_muse18/broadcasts/step_0",
                    f"{run_base}/run_muse18/broadcasts/step_10",
                ]
            ),
            run_base,
        )
        self.assertEqual([item.step for item in parsed], [0, 10])
        with self.assertRaises(RuntimeError):
            training_ops._parse_adapter_broadcasts(
                "/data/glm52-rl/runs/muse17/run_muse17/broadcasts/step_1",
                run_base,
            )

    def test_archive_refuses_to_remove_latest_live_adapter(self):
        run = "muse18"
        run_base = f"{training_ops.run_root}/{run}"
        stable = (
            f"{run_base}/run_muse18/broadcasts/step_10\n"
            f"{run_base}/run_muse18/broadcasts/step_11\n"
        )
        with patch.object(training_ops, "_local_command"), patch.object(
            training_ops, "_ssh"
        ) as ssh:
            ssh.side_effect = [
                (0, stable),
                (0, "STABLE\nadapter_config.json\nadapter_model.safetensors\n"),
                (0, "STABLE\nadapter_config.json\nadapter_model.safetensors\n"),
            ]
            with self.assertRaises(training_ops.Refused):
                training_ops.archive_adapters(run, [11])

    def test_archive_reads_the_active_policy_step(self):
        with patch.object(
            training_ops,
            "_ssh",
            return_value=(0, "Updated weights to step 25 in 1.2s\n"),
        ):
            self.assertEqual(training_ops._confirmed_active_policy_step("muse19"), 25)

    def test_archive_removes_source_only_after_verified_upload(self):
        run = "muse18"
        run_base = f"{training_ops.run_root}/{run}"
        stable_path = f"{run_base}/run_muse18/broadcasts/step_10"
        stable = (
            stable_path + "\n"
            + f"{run_base}/run_muse18/broadcasts/step_11\n"
        )
        expected = ["STABLE", "adapter_config.json", "adapter_model.safetensors"]
        digest = hashlib.sha256(b"adapter").hexdigest()
        events = []

        def fake_ssh(host, command, timeout=120):
            if command.startswith("test -d "):
                return 0, stable
            events.append("cleanup")
            return 0, ""

        with (
            patch.object(training_ops, "_local_command", side_effect=lambda *args, **kwargs: events.append("auth")),
            patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            patch.object(
                training_ops,
                "_remote_broadcast_files",
                side_effect=[expected, expected, expected],
            ),
            patch.object(training_ops, "_confirmed_active_policy_step", return_value=11),
            patch.object(training_ops, "_remote_sha256", return_value=digest),
            patch.object(
                training_ops,
                "_download_adapter_file",
                side_effect=lambda remote, local: (local.write_bytes(b"adapter"), events.append("download")),
            ),
            patch.object(training_ops, "_archive_one_file", side_effect=lambda *args: events.append("verified")),
        ):
            self.assertEqual(training_ops.archive_adapters(run, [10]), [10])

        self.assertLess(events.index("verified"), events.index("cleanup"))

    def test_deploy_routes_runtime_config_to_inference_hosts(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "configs" / "runtime" / "test.toml"
            local.parent.mkdir(parents=True)
            local.write_text("value = 1\n")
            want = hashlib.sha256(local.read_bytes()).hexdigest()
            copied_to = []

            def fake_run(command, **kwargs):
                if "-G" in command:
                    return Mock(returncode=0, stdout="", stderr="")
                if command[0] == "scp":
                    copied_to.append(command[-1].split(":", 1)[0])
                return Mock(returncode=0)

            def fake_ssh(host, command, timeout=120):
                if command.startswith("if test -f "):
                    return 0, "0" * 64
                if command.startswith("test -f "):
                    return 0, want
                return 0, ""

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(
                    training_ops, "_ssh_control_masters", return_value=nullcontext()
                ),
                patch.object(training_ops.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(training_ops.deploy("configs/runtime/test.toml"))

            self.assertCountEqual(copied_to, training_ops.inference_hosts)

    def test_deploy_skips_scp_when_remote_file_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "source" / "mounted.py"
            local.parent.mkdir(parents=True)
            local.write_text("current\n")
            want = hashlib.sha256(local.read_bytes()).hexdigest()

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounted_on", return_value=True),
                patch.object(
                    training_ops,
                    "_ssh",
                    return_value=(0, want),
                ) as ssh,
                patch.object(
                    training_ops,
                    "_ssh_control_masters",
                    return_value=nullcontext(),
                ),
                patch.object(training_ops.subprocess, "run") as run,
            ):
                self.assertTrue(training_ops.deploy("source/mounted.py", nodes=["n78"]))

            run.assert_not_called()
            ssh.assert_called_once()

    def test_deploy_accepts_checksum_output_with_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "source" / "mounted.py"
            local.parent.mkdir(parents=True)
            local.write_text("current\n")
            want = hashlib.sha256(local.read_bytes()).hexdigest()

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounted_on", return_value=True),
                patch.object(
                    training_ops,
                    "_ssh",
                    return_value=(0, f"{want}  /data/glm52-rl/deployment-glm52-optims/source/mounted.py\n"),
                ) as ssh,
                patch.object(
                    training_ops,
                    "_ssh_control_masters",
                    return_value=nullcontext(),
                ),
                patch.object(training_ops.subprocess, "run") as run,
            ):
                self.assertTrue(training_ops.deploy("source/mounted.py", nodes=["n78"]))

            run.assert_not_called()
            ssh.assert_called_once()

    def test_deploy_reports_silent_scp_failure_with_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "source" / "mounted.py"
            local.parent.mkdir(parents=True)
            local.write_text("current\n")
            calls = []

            def fake_ssh(host, command, timeout=120):
                del host, timeout
                calls.append(command)
                if command.startswith("if test -f "):
                    return 0, "0" * 64
                return 0, ""

            def fake_run(command, **kwargs):
                del kwargs
                if command[0] == "scp":
                    return Mock(returncode=17, stdout="", stderr="")
                raise AssertionError(f"unexpected local command: {command}")

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounted_on", return_value=True),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(
                    training_ops,
                    "_ssh_control_masters",
                    return_value=nullcontext(),
                ),
                patch.object(training_ops.subprocess, "run", side_effect=fake_run),
                patch("builtins.print") as print_call,
            ):
                self.assertFalse(training_ops.deploy("source/mounted.py", nodes=["n83"]))

            rendered = "\n".join(
                " ".join(str(arg) for arg in call.args)
                for call in print_call.call_args_list
            )
            self.assertIn("scp exited with status 17", rendered)
            self.assertEqual(len(calls), 1)

    def test_deploy_refuses_unparseable_preflight_without_scp(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "source" / "mounted.py"
            local.parent.mkdir(parents=True)
            local.write_text("current\n")

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounted_on", return_value=True),
                patch.object(
                    training_ops,
                    "_ssh",
                    return_value=(0, "sha256sum: input/output error\n"),
                ) as ssh,
                patch.object(
                    training_ops,
                    "_ssh_control_masters",
                    return_value=nullcontext(),
                ),
                patch.object(training_ops.subprocess, "run") as run,
                patch("builtins.print") as print_call,
            ):
                self.assertFalse(training_ops.deploy("source/mounted.py", nodes=["n78"]))

            run.assert_not_called()
            ssh.assert_called_once()
            rendered = "\n".join(
                " ".join(str(arg) for arg in call.args)
                for call in print_call.call_args_list
            )
            self.assertIn("no unambiguous SHA-256", rendered)

    def test_deploy_copies_when_remote_file_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "source" / "mounted.py"
            local.parent.mkdir(parents=True)
            local.write_text("current\n")
            want = hashlib.sha256(local.read_bytes()).hexdigest()
            copied_to = []
            copied_commands = []

            def fake_run(command, **kwargs):
                del kwargs
                if command[0] == "scp":
                    copied_commands.append(command)
                    copied_to.append(command[-1].split(":", 1)[0])
                return Mock(returncode=0, stdout="", stderr="")

            def fake_ssh(host, command, timeout=120):
                del host, timeout
                if command.startswith("if test -f "):
                    return 0, "0" * 64
                if command.startswith("test -f "):
                    return 0, want
                return 0, ""

            with (
                patch.object(training_ops, "repo_root", repo),
                patch.object(training_ops, "_mounted_on", return_value=True),
                patch.object(training_ops, "_ssh", side_effect=fake_ssh),
                patch.object(
                    training_ops,
                    "_ssh_control_masters",
                    return_value=nullcontext(),
                ),
                patch.object(training_ops.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(training_ops.deploy("source/mounted.py", nodes=["n83"]))

            self.assertEqual(copied_to, ["n83"])
            self.assertEqual(len(copied_commands), 1)
            self.assertIn("ControlMaster=no", copied_commands[0])
            self.assertIn("ControlPath=none", copied_commands[0])

if __name__ == "__main__":
    unittest.main()
