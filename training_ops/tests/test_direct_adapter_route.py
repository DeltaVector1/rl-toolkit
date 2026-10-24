import importlib.util
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
import training_ops.tests  # noqa: F401 - installs the local-only test guard.

source = root / "training_ops/runtime/china/direct_adapter_route.py"
spec = importlib.util.spec_from_file_location("direct_route", source)
route_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route_module)


class DirectRouteTest(unittest.TestCase):
    def setUp(self):
        self.route = route_module.settings(root / "configs/nodes.toml")
        # operate() tests use mocked Popen children, never a real manager.
        pin = mock.patch.object(route_module, "pin_child", side_effect=lambda process: os.open("/dev/null", os.O_RDONLY), create=True)
        pin.start()
        self.addCleanup(pin.stop)

    def write_receipt(self, ssh, output):
        argv = shlex.split(ssh[-1])[1:]
        ports = [int(value) for value in argv[5:]]
        value = {"token": argv[4], "ports": ports, "digest": self.route["digest"], "session": {"pid": 200000 + ports[0], "start": "1", "argv": argv}, "owner": {"pid": 300000 + ports[0], "start": "1", "argv": ["sshd: root@notty"]}, "transport": self.transport(ports)}
        output.write(json.dumps(value).encode())

    def transport(self, ports):
        return {"boot_id": "boot", "connection": ["192.0.2.1", "10000", "192.0.2.2", "22"], "connection_inode": 50000 + ports[0], "listener_inodes": {str(port): port for port in ports}}

    def test_exact_mapping_and_authentication(self):
        owners = route_module.commands(self.route)
        self.assertEqual(len(owners), 64)
        self.assertEqual([port for ports, _ in owners for port in ports], list(range(18100, 18228)))
        for ports, argv in owners:
            self.assertEqual(len(ports), 2)
            self.assertEqual(argv.count("-R"), 2)
            self.assertEqual(
                argv[-1],
                f"{self.route['ssh_user']}@{self.route['ssh_host']}",
            )
            for option in ("StrictHostKeyChecking=yes", "BatchMode=yes", "IdentitiesOnly=yes", "AddKeysToAgent=no", "ExitOnForwardFailure=yes", "ForwardAgent=no", "Compression=no", "ControlPath=none", "ProxyCommand=none"):
                self.assertIn(option, argv)
            self.assertIn(self.route["ssh_identity_file"], argv)
            self.assertIn(
                f"UserKnownHostsFile={self.route['ssh_known_hosts_file']}", argv
            )
            self.assertIn(str(self.route["ssh_port"]), argv)
            for port in ports:
                self.assertIn(f"127.0.0.1:{port}:127.0.0.1:8100", argv)

    def test_forward_ack_requires_both_exact_listeners(self):
        line = "remote forward success for: listen 127.0.0.1:{}, connect 127.0.0.1:8100"
        self.assertFalse(route_module.forwards_ready(line.format(18100), [18100, 18101], 8100))
        self.assertFalse(route_module.forwards_ready("Permission denied (publickey)", [18100, 18101], 8100))
        self.assertTrue(route_module.forwards_ready(line.format(18100) + line.format(18101), [18100, 18101], 8100))

    def test_duplicate_manager_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lock"
            fd = route_module.acquire(lock)
            try:
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    route_module.acquire(lock)
            finally:
                os.close(fd)

    def test_stale_pid_cannot_be_signalled(self):
        record = {"pid": os.getpid(), "start": "wrong", "argv": []}
        with mock.patch.object(route_module, "pidfd_signal") as send:
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                route_module.send(record, signal.SIGTERM)
            send.assert_not_called()

    def test_cleanup_has_one_shared_deadline(self):
        children = []
        records = {}
        for pid in (101, 102):
            process = mock.Mock(pid=pid)
            process.poll.return_value = None
            process.wait.side_effect = [subprocess.TimeoutExpired("ssh", 0), 0]
            argv = ["/usr/bin/ssh", str(pid)]
            records[pid] = {"pid": pid, "start": "1", "argv": argv}
            children.append((process, [], argv, [], None))
        with mock.patch.object(route_module, "identity", side_effect=records.get), mock.patch.object(route_module, "send") as send, mock.patch.object(route_module.time, "monotonic", side_effect=[0, 1, 8, 8, 9]):
            route_module.cleanup(children, 10)
        self.assertEqual([call.args[1] for call in send.call_args_list], [signal.SIGTERM, signal.SIGTERM, signal.SIGKILL, signal.SIGKILL])
        self.assertEqual(children[0][0].wait.call_args_list[0], mock.call(timeout=7))
        self.assertEqual(children[1][0].wait.call_args_list[0], mock.call(timeout=0))

    def test_listener_ownership_missing_duplicate_and_wrong_owner(self):
        rows = [f'LISTEN 0 128 127.0.0.1:{port} 0.0.0.0:* users:(("sshd",pid={1000 + (port - 18100) // 2},fd={port}))' for port in range(18100, 18228)]
        self.assertEqual(len(route_module.listener_owners("\n".join(rows), self.route["ports"])), 128)
        for fixture in (rows[:-1], rows + rows[:1], [row.replace('"sshd"', '"ssh"') for row in rows]):
            with self.assertRaises(RuntimeError):
                route_module.listener_owners("\n".join(fixture), self.route["ports"])

    def test_status_does_not_create_state_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent"
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                route_module.operate("status", {**self.route, "state_dir": str(path)})
            self.assertFalse(path.exists())

    def test_private_state_rejects_symlinks_and_permissive_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            state = parent / "state"
            state.mkdir(mode=0o700)
            route = {**self.route, "state_dir": str(state)}
            outside = parent / "outside"
            outside.write_text("unchanged")
            for name in ("state.pending", "state.json", "manager.lock", "manager.log"):
                path = state / name
                path.symlink_to(outside)
                with self.assertRaises((RuntimeError, OSError)):
                    if name.startswith("state."):
                        route_module.publish(route, {})
                    else:
                        route_module.state_open(route, name, os.O_RDWR | os.O_CREAT, truncate=True)
                self.assertEqual(outside.read_text(), "unchanged")
                path.unlink()
            state.chmod(0o755)
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                route_module.publish(route, {})
            state.chmod(0o700)
            path = state / "state.json"
            path.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                route_module.stored_state(route)
            path.unlink()
            alias = parent / "alias"
            alias.symlink_to(state)
            with self.assertRaises(OSError):
                route_module.publish({**route, "state_dir": str(alias)}, {})
            with mock.patch.object(route_module.os, "geteuid", return_value=os.geteuid() + 1):
                with self.assertRaisesRegex(RuntimeError, "unsafe"):
                    route_module.publish(route, {})

    def test_forged_manager_and_changed_config_cannot_be_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            record = route_module.identity(os.getpid())
            for digest, argv in ((route["digest"], record["argv"]), ("wrong", route_module.manager_argv(route))):
                route_module.publish(route, {"digest": digest, "manager": {**record, "argv": argv}, "status": "ready", "children": []})
                with mock.patch.object(route_module, "send") as send:
                    with self.assertRaisesRegex(RuntimeError, "refusing signal"):
                        route_module.operate("stop", route)
                    send.assert_not_called()

    def test_stopped_state_from_old_config_can_be_reconciled(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {**self.route, "state_dir": directory, "digest": "new"}
            manager = {"pid": 999999, "start": "1", "argv": route_module.manager_argv(old)}
            child = {"pid": 999998, "start": "1", "argv": ["old-child"]}
            route_module.publish(old, {"manager": manager, "digest": "old", "status": "stopped", "children": [child]})
            with mock.patch.object(route_module, "check_pidfd_support"), mock.patch.object(
                route_module, "matches", return_value=False
            ):
                route_module.operate("stop", current)

    def test_stopped_config_mismatch_refuses_live_or_ambiguous_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {**self.route, "state_dir": directory, "digest": "new"}
            absent = {"pid": 999999, "start": "1", "argv": ["old-manager"]}
            live = route_module.identity(os.getpid())
            for manager, children, message in (
                (live, [], "recorded process is live"),
                (absent, [live], "recorded process is live"),
                (absent, [{}], "malformed process identity"),
            ):
                with self.subTest(message=message, child=bool(children)):
                    route_module.publish(old, {"manager": manager, "digest": "old", "status": "stopped", "children": children})
                    with mock.patch.object(route_module, "check_pidfd_support"), self.assertRaisesRegex(RuntimeError, message):
                        route_module.operate("stop", current)

    def test_restart_replaces_authenticated_live_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {
                **self.route,
                "state_dir": directory,
                "digest": "new",
                "ssh_identity_file": str(source),
                "ssh_known_hosts_file": str(source),
            }
            manager = {"pid": 11111, "start": "1", "argv": route_module.manager_argv(old)}
            child = {"pid": 22222, "start": "2", "argv": ["/usr/bin/ssh", "old-route"]}
            new = {"pid": 33333, "start": "3", "argv": route_module.manager_argv(current)}
            route_module.publish(old, {"manager": manager, "digest": "old", "status": "ready", "children": [child]})
            identities = {manager["pid"]: manager, child["pid"]: child, new["pid"]: new}
            process = mock.Mock(pid=new["pid"])
            process.poll.return_value = None

            def stop(record, number):
                self.assertEqual((record, number), (manager, signal.SIGTERM))
                identities.pop(manager["pid"])
                identities.pop(child["pid"])

            with mock.patch.object(route_module, "check_pidfd_support"), mock.patch.object(
                route_module, "identity", side_effect=identities.get
            ), mock.patch.object(route_module, "send", side_effect=stop) as send, mock.patch.object(
                route_module.subprocess, "Popen", return_value=process
            ) as launch, mock.patch.object(
                route_module, "ready", return_value={"manager": new}
            ):
                route_module.operate("restart", current)

            send.assert_called_once_with(manager, signal.SIGTERM)
            launch.assert_called_once()

    def test_start_refuses_live_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {**self.route, "state_dir": directory, "digest": "new"}
            manager = {"pid": 11111, "start": "1", "argv": route_module.manager_argv(old)}
            route_module.publish(old, {"manager": manager, "digest": "old", "status": "ready", "children": []})
            with mock.patch.object(route_module, "check_pidfd_support"), mock.patch.object(
                route_module, "identity", return_value=manager
            ), mock.patch.object(route_module, "send") as send, self.assertRaisesRegex(
                RuntimeError, "configuration differs; refusing signal"
            ):
                route_module.operate("start", current)
            send.assert_not_called()

    def test_restart_refuses_tampered_manager_or_child_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {**self.route, "state_dir": directory, "digest": "new"}
            manager = {"pid": 11111, "start": "1", "argv": route_module.manager_argv(old)}
            child = {"pid": 22222, "start": "2", "argv": ["/usr/bin/ssh", "old-route"]}
            route_module.publish(old, {"manager": manager, "digest": "old", "status": "ready", "children": [child]})
            for changed, message in (
                ({manager["pid"]: {**manager, "start": "9"}, child["pid"]: child}, "manager identity differs"),
                ({manager["pid"]: manager, child["pid"]: {**child, "argv": ["unrelated"]}}, "child identity differs"),
            ):
                with self.subTest(message=message), mock.patch.object(
                    route_module, "check_pidfd_support"
                ), mock.patch.object(
                    route_module, "identity", side_effect=changed.get
                ), mock.patch.object(route_module, "send") as send, self.assertRaisesRegex(
                    RuntimeError, message
                ):
                    route_module.operate("restart", current)
                send.assert_not_called()

            unrelated = {**manager, "argv": ["/usr/bin/python3", "unrelated"]}
            route_module.publish(old, {"manager": unrelated, "digest": "old", "status": "ready", "children": [child]})
            with mock.patch.object(route_module, "check_pidfd_support"), mock.patch.object(
                route_module, "identity", side_effect={unrelated["pid"]: unrelated, child["pid"]: child}.get
            ), mock.patch.object(route_module, "send") as send, self.assertRaisesRegex(
                RuntimeError, "manager argv or configuration differs"
            ):
                route_module.operate("restart", current)
            send.assert_not_called()

    def test_restart_allows_missing_child_for_authenticated_manager_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {**self.route, "state_dir": directory, "digest": "new"}
            manager = {"pid": 11111, "start": "1", "argv": route_module.manager_argv(old)}
            child = {"pid": 22222, "start": "2", "argv": ["/usr/bin/ssh", "old-route"]}
            state = {"manager": manager, "digest": "old", "status": "ready", "children": [child]}
            route_module.publish(old, state)
            with mock.patch.object(route_module, "identity", side_effect={manager["pid"]: manager}.get):
                self.assertEqual(route_module.restart_state(current), state)

    def test_restart_live_mismatch_cleanup_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {**self.route, "state_dir": directory, "digest": "new", "cleanup_timeout_seconds": 5}
            manager = {"pid": 11111, "start": "1", "argv": route_module.manager_argv(old)}
            child = {"pid": 22222, "start": "2", "argv": ["/usr/bin/ssh", "old-route"]}
            route_module.publish(old, {"manager": manager, "digest": "old", "status": "ready", "children": [child]})
            identities = {manager["pid"]: manager, child["pid"]: child}
            with mock.patch.object(route_module, "check_pidfd_support"), mock.patch.object(
                route_module, "identity", side_effect=identities.get
            ), mock.patch.object(route_module, "send") as send, mock.patch.object(
                route_module.subprocess, "Popen"
            ) as launch, mock.patch.object(
                route_module.time, "monotonic", side_effect=[0, 0, 0, 9]
            ), mock.patch.object(route_module.time, "sleep"), self.assertRaisesRegex(
                TimeoutError, "cleanup deadline exceeded"
            ):
                route_module.operate("restart", current)
            send.assert_called_once_with(manager, signal.SIGTERM)
            launch.assert_not_called()

    def test_restart_accepts_stopped_digest_mismatch_without_signalling(self):
        with tempfile.TemporaryDirectory() as directory:
            old = {**self.route, "state_dir": directory, "digest": "old"}
            current = {
                **self.route,
                "state_dir": directory,
                "digest": "new",
                "ssh_identity_file": str(source),
                "ssh_known_hosts_file": str(source),
            }
            manager = {"pid": 11111, "start": "1", "argv": route_module.manager_argv(old)}
            child = {"pid": 22222, "start": "2", "argv": ["/usr/bin/ssh", "old-route"]}
            new = {"pid": 33333, "start": "3", "argv": route_module.manager_argv(current)}
            route_module.publish(old, {"manager": manager, "digest": "old", "status": "stopped", "children": [child]})
            identities = {}
            process = mock.Mock(pid=new["pid"])
            process.poll.return_value = None

            def launch(*args, **kwargs):
                identities[new["pid"]] = new
                return process

            with mock.patch.object(route_module, "check_pidfd_support"), mock.patch.object(
                route_module, "identity", side_effect=identities.get
            ), mock.patch.object(route_module, "send") as send, mock.patch.object(
                route_module.subprocess, "Popen", side_effect=launch
            ), mock.patch.object(route_module, "ready", return_value={"manager": new}):
                route_module.operate("restart", current)
            send.assert_not_called()

    def test_stale_private_pending_is_recoverable_but_symlink_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            lock = route_module.acquire(Path(directory) / "manager.lock")
            try:
                fd = route_module.state_open(route, "state.pending", os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                os.write(fd, b"interrupted publish")
                os.close(fd)
                route_module.clear_pending(route)
                route_module.publish(route, {"recovered": True})
                self.assertEqual(json.loads(route_module.state_text(route, "state.json")), {"recovered": True})
                pending = Path(directory) / "state.pending"
                pending.symlink_to(Path(directory) / "state.json")
                with self.assertRaises(OSError):
                    route_module.clear_pending(route)
                self.assertTrue(pending.is_symlink())
            finally:
                os.close(lock)

    def test_missing_pidfd_support_fails_before_state_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "absent"
            with mock.patch.object(route_module.os, "pidfd_open", None, create=True), mock.patch.object(route_module.signal, "pidfd_send_signal", None, create=True), mock.patch.object(route_module.ctypes, "CDLL", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "no route state was changed"):
                    route_module.operate("start", {**self.route, "state_dir": str(state)})
            self.assertFalse(state.exists())

    def test_libc_pidfd_fallback_is_safe_without_python_binding(self):
        process = subprocess.Popen(["/usr/bin/sleep", "30"])
        try:
            record = route_module.identity(process.pid)
            with mock.patch.object(route_module.signal, "pidfd_send_signal", None, create=True), mock.patch.object(route_module.os, "pidfd_open", None, create=True):
                route_module.send(record, signal.SIGTERM)
            self.assertEqual(process.wait(timeout=2), -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

    def test_pidfd_exit_races_are_already_clean(self):
        record = {"pid": 12345, "start": "1", "argv": ["owned"]}
        for current in (None, record):
            with mock.patch.object(route_module, "pidfd_open", return_value=99), mock.patch.object(route_module, "identity", return_value=current), mock.patch.object(route_module, "pidfd_signal", side_effect=ProcessLookupError) as send, mock.patch.object(route_module.os, "close") as close:
                route_module.send(record, signal.SIGTERM)
                self.assertEqual(send.call_count, int(current is not None))
                close.assert_called_once_with(99)

    def test_receipt_rejects_malformed_process_identity(self):
        ports = [18100, 18101]
        token = "a" * 32
        receipt = {"token": token, "ports": ports, "digest": self.route["digest"], "session": {"pid": 3000, "start": "1", "argv": route_module.session_argv(self.route, ports, token)}, "owner": {"pid": 1000, "start": "1", "argv": ["sshd: root@notty"]}}
        for field in ("session", "owner"):
            for record in ({}, {"pid": -1, "start": "1", "argv": ["x"]}, {"pid": 10, "start": "wrong", "argv": ["x"]}, {"pid": 10, "start": "1", "argv": []}):
                with self.assertRaisesRegex(RuntimeError, "malformed"):
                    route_module.validate_receipt(self.route, ports, {**receipt, field: record})

    def test_receipts_bind_exact_sessions_and_stability_snapshots(self):
        snapshot = {"digest": self.route["digest"], "status": "ready", "manager": {"pid": 10}, "children": [], "receipts": {}}
        rows = []
        for index, (ports, _) in enumerate(route_module.commands(self.route)):
            token = f"{index:032x}"
            receipt = {"token": token, "ports": ports, "digest": self.route["digest"], "session": {"pid": 3000 + index, "start": "1", "argv": route_module.session_argv(self.route, ports, token)}, "owner": {"pid": 1000 + index, "start": "1", "argv": ["sshd: root@notty"]}, "transport": self.transport(ports)}
            snapshot["receipts"][str(ports[0])] = receipt
            rows.extend(f'LISTEN 0 128 127.0.0.1:{port} 0.0.0.0:* users:(("sshd",pid={1000 + index},fd={port}))' for port in ports)
        listeners = "\n".join(rows)
        proof = mock.patch.object(route_module, "transport_proof", side_effect=lambda owner, ports, connection: self.transport(ports))
        proof.start()
        self.addCleanup(proof.stop)
        with mock.patch.object(route_module, "matches", return_value=True), mock.patch.object(route_module, "ancestor", return_value=True):
            before = route_module.verify_snapshot(self.route, snapshot, listeners)
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                route_module.verify_snapshot(self.route, snapshot, listeners.replace("pid=1000,", "pid=9000,"))
            snapshot["receipts"]["18100"]["owner"]["start"] = "2"
            self.assertNotEqual(before, route_module.verify_snapshot(self.route, snapshot, listeners))
        with mock.patch.object(route_module, "matches", return_value=False), mock.patch.object(route_module, "ancestor", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                route_module.verify_snapshot(self.route, snapshot, listeners)
        with mock.patch.object(route_module, "matches", return_value=True), mock.patch.object(route_module, "ancestor", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                route_module.verify_snapshot(self.route, snapshot, listeners)
        with mock.patch.object(route_module, "matches", return_value=True), mock.patch.object(route_module, "ancestor", return_value=True):
            snapshot["receipts"]["18100"]["transport"]["boot_id"] = "old-boot"
            with self.assertRaisesRegex(RuntimeError, "transport proof changed"):
                route_module.verify_snapshot(self.route, snapshot, listeners)

    def test_tcp_proof_rejects_shared_daemon_sibling_and_tuple_mismatch(self):
        ports = [18100, 18101]
        connection = ["192.0.2.1", "10000", "192.0.2.2", "22"]
        rows = [{"state": "01", "local": ("192.0.2.2", 22), "peer": ("192.0.2.1", 10000), "inode": 1},
                {"state": "01", "local": ("192.0.2.2", 22), "peer": ("192.0.2.1", 10001), "inode": 5}]
        rows.extend({"state": "0A", "local": ("127.0.0.1", port), "peer": ("0.0.0.0", 0), "inode": index + 2} for index, port in enumerate(ports))
        with mock.patch.object(route_module, "tcp_sockets", return_value=rows), mock.patch.object(route_module.Path, "read_text", return_value="boot"):
            with mock.patch.object(route_module, "process_sockets", return_value={1, 2, 3}):
                proof = route_module.transport_proof({"pid": 1000}, ports, connection)
                self.assertEqual(proof["connection_inode"], 1)
                self.assertEqual(proof["listener_inodes"], {"18100": 2, "18101": 3})
                with self.assertRaisesRegex(RuntimeError, "exact SSH connection"):
                    route_module.transport_proof({"pid": 1000}, ports, [connection[0], "10001", *connection[2:]])
            for held in ({2, 3, 4}, {2, 3, 5}):
                with mock.patch.object(route_module, "process_sockets", return_value=held):
                    with self.assertRaisesRegex(RuntimeError, "exact SSH connection"):
                        route_module.transport_proof({"pid": 1000}, ports, connection)
    def test_setup_cap_and_cleanup_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            identities = {os.getpid(): route_module.identity(os.getpid())}
            processes = []
            pending = [0]
            maximum = [0]

            def launch(argv, **kwargs):
                process = mock.Mock()
                process.pid = 100000 + len(processes)
                process.poll.return_value = None
                processes.append(process)
                ssh = argv[5:]
                self.write_receipt(ssh, kwargs["stdout"])
                identities[process.pid] = {"pid": process.pid, "start": "1", "argv": ssh}
                for index, value in enumerate(ssh):
                    if value == "-R":
                        port = ssh[index + 1].split(":")[1]
                        kwargs["stderr"].write(f"remote forward success for: listen 127.0.0.1:{port}, connect 127.0.0.1:8100\n".encode())
                pending[0] += 1
                maximum[0] = max(maximum[0], pending[0])
                return process

            real_ready = route_module.forwards_ready
            def acknowledge(*args):
                result = real_ready(*args)
                if result:
                    pending[0] -= 1
                return result

            def publish(_, state):
                if state["status"] == "ready":
                    raise RuntimeError("test end")

            with mock.patch.object(route_module.subprocess, "Popen", side_effect=launch), mock.patch.object(route_module, "identity", side_effect=identities.get), mock.patch.object(route_module, "forwards_ready", side_effect=acknowledge), mock.patch.object(route_module, "publish", side_effect=publish), mock.patch.object(route_module, "send") as send, mock.patch.object(route_module.signal, "signal"), mock.patch.object(route_module.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "test end"):
                    route_module.serve(route)
            self.assertEqual(len(processes), 64)
            self.assertEqual(maximum[0], 4)
            self.assertEqual(send.call_count, 64)
            self.assertTrue(all(item.wait.called for item in processes))

    def test_reconnect_has_backoff_cap_and_preserves_healthy_owners(self):
        self.reconnect_case(False)

    def test_successful_recovery_resets_retry_budget_for_later_drops(self):
        self.reconnect_case(True)

    def reconnect_case(self, repeated):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            identities = {os.getpid(): route_module.identity(os.getpid())}
            processes, starts = [], []
            now, pending, maximum = [0.0], [0], [0]
            failed_at = [None]
            rounds = [0]

            def launch(argv, **kwargs):
                process = mock.Mock(pid=100000 + len(processes))
                process.poll.return_value = None
                processes.append(process)
                starts.append(now[0])
                ssh = argv[5:]
                self.write_receipt(ssh, kwargs["stdout"])
                identities[process.pid] = {"pid": process.pid, "start": "1", "argv": ssh}
                for index, value in enumerate(ssh):
                    if value == "-R":
                        port = ssh[index + 1].split(":")[1]
                        kwargs["stderr"].write(f"remote forward success for: listen 127.0.0.1:{port}, connect 127.0.0.1:8100\n".encode())
                pending[0] += 1
                maximum[0] = max(maximum[0], pending[0])
                return process

            real_ready = route_module.forwards_ready
            def acknowledge(*args):
                result = real_ready(*args)
                if result:
                    pending[0] -= 1
                return result

            def publish(_, state):
                if state["status"] != "ready":
                    return
                rounds[0] += 1
                if rounds[0] == 1:
                    failed_at[0] = now[0]
                    for process in processes[:1 if repeated else 5]:
                        process.poll.return_value = 255
                elif repeated and rounds[0] <= 5:
                    self.assertFalse(state["reconnect_attempts"])
                    processes[-1].poll.return_value = 255
                else:
                    self.assertEqual(len(processes), 69)
                    self.assertTrue(all(not process.wait.called for process in processes[5:64]))
                    self.assertTrue(all(value >= failed_at[0] + route["reconnect_backoff_seconds"] for value in starts[64:]))
                    raise RuntimeError("reconnect passed")

            def sleep(seconds):
                now[0] += seconds
                if now[0] > 60:
                    raise RuntimeError("test deadline")

            with mock.patch.object(route_module.subprocess, "Popen", side_effect=launch), mock.patch.object(route_module, "identity", side_effect=identities.get), mock.patch.object(route_module, "forwards_ready", side_effect=acknowledge), mock.patch.object(route_module, "publish", side_effect=publish), mock.patch.object(route_module, "send"), mock.patch.object(route_module.signal, "signal"), mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, "reconnect passed"):
                    route_module.serve(route)
            self.assertEqual(maximum[0], 4)

    def test_failed_start_cleans_and_reaps_only_started_manager(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            record = {"pid": process.pid, "start": "1", "argv": [sys.executable, str(source), "serve", route["nodes_path"]]}
            now = [0]
            def sleep(seconds):
                now[0] += 100
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "identity", return_value=record), mock.patch.object(route_module, "ready", side_effect=RuntimeError("not ready")), mock.patch.object(route_module, "pidfd_signal") as send, mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    route_module.operate("start", route)
            self.assertEqual([call.args[1] for call in send.call_args_list], [0, signal.SIGTERM])
            self.assertEqual(process.wait.call_count, 1)

    def test_start_accepts_post_exec_identity_after_transient_cmdline(self):
        self.start_transition_case(1, 1)

    def test_full_64_owner_ready_at_deadline_is_not_false_timeout(self):
        self.start_transition_case(182, 100)

    def start_transition_case(self, ready_at, tick):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            manager = {"pid": 12345, "start": "1", "argv": route_module.manager_argv(route)}
            identities = {manager["pid"]: manager}
            state = {"manager": manager, "digest": route["digest"], "status": "starting", "children": [], "receipts": {}}
            for index, (ports, argv) in enumerate(route_module.commands(route)):
                token = f"{index:032x}"
                record = {"pid": 20000 + index, "start": "1", "argv": argv + [route_module.session_command(route, ports, token)]}
                state["children"].append(record)
                identities[record["pid"]] = record
                state["receipts"][str(ports[0])] = {"token": token, "ports": ports, "digest": route["digest"], "session": {"pid": 30000 + index, "start": "1", "argv": route_module.session_argv(route, ports, token)}, "owner": {"pid": 40000 + index, "start": "1", "argv": ["sshd: root@notty"]}, "transport": self.transport(ports)}
            process = mock.Mock(pid=manager["pid"])
            process.poll.return_value = None
            now = [0]
            def launch(*args, **kwargs):
                route_module.publish(route, state)
                return process
            def identity(pid):
                if pid == manager["pid"] and now[0] == 0:
                    return {**manager, "argv": ["transient-pre-exec-parent"]}
                return identities.get(pid)
            def sleep(seconds):
                now[0] += tick
                if now[0] >= ready_at:
                    state["status"] = "ready"
                    route_module.publish(route, state)
            with mock.patch.object(route_module.subprocess, "Popen", side_effect=launch), mock.patch.object(route_module, "identity", side_effect=identity), mock.patch.object(route_module, "pidfd_signal") as send, mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                route_module.operate("start", route)
                self.assertEqual(len(route_module.ready(route)["children"]), 64)
            self.assertEqual([call.args[1] for call in send.call_args_list], [0])
            process.wait.assert_not_called()

    def test_failure_with_transitioning_argv_uses_only_pinned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "ready", side_effect=ValueError("failed receipt")), mock.patch.object(route_module, "identity", return_value={"pid": 99999, "start": "different", "argv": ["unrelated"]}), mock.patch.object(route_module, "send") as numeric_send, mock.patch.object(route_module, "pidfd_signal") as send:
                with self.assertRaisesRegex(ValueError, "failed receipt"):
                    route_module.operate("start", route)
            numeric_send.assert_not_called()
            self.assertEqual([call.args[1] for call in send.call_args_list], [0, signal.SIGTERM])
            process.wait.assert_called_once()

    def test_different_exact_ready_manager_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            other = {"pid": 54321, "start": "2", "argv": route_module.manager_argv(route)}
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "ready", return_value={"manager": other}) as ready, mock.patch.object(route_module, "send") as numeric_send, mock.patch.object(route_module, "pidfd_signal") as send:
                route_module.operate("start", route)
            self.assertEqual(ready.call_count, 2)
            self.assertEqual([call.args[1] for call in send.call_args_list], [0, signal.SIGTERM])
            numeric_send.assert_not_called()
            process.wait.assert_called_once()

    def test_pinned_cleanup_has_one_deadline_and_reaps_after_kill(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("manager", 8), 0]
        with mock.patch.object(route_module, "pidfd_signal") as send, mock.patch.object(route_module.time, "monotonic", side_effect=[10, 19]):
            route_module.cleanup_started(process, 99, 10)
        self.assertEqual(send.call_args_list, [mock.call(99, signal.SIGTERM), mock.call(99, signal.SIGKILL)])
        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=8), mock.call(timeout=1)])

    def test_pin_failure_reaps_exited_child_but_never_signals_unpinned_pid(self):
        for result in (1, None):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as directory:
                route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
                process = mock.Mock(pid=12345)
                process.poll.return_value = result
                with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "pin_child", side_effect=OSError("pin failed")), mock.patch.object(route_module, "send") as numeric_send, mock.patch.object(route_module, "pidfd_signal") as send:
                    with self.assertRaisesRegex(OSError if result is not None else RuntimeError, "pin failed" if result is not None else "child may still be active; no signal"):
                        route_module.operate("start", route)
                numeric_send.assert_not_called()
                self.assertEqual([call.args[1] for call in send.call_args_list], [0])
                self.assertEqual(process.wait.call_count, int(result is not None))

    def test_exited_spawn_is_not_accepted_from_stale_ready_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            process = mock.Mock(pid=12345)
            process.poll.return_value = 1
            record = {"pid": process.pid, "start": "1", "argv": route_module.manager_argv(route)}
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "ready", return_value={"manager": record}), mock.patch.object(route_module, "pidfd_signal") as send:
                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    route_module.operate("start", route)
            self.assertEqual([call.args[1] for call in send.call_args_list], [0])
            process.wait.assert_called_once_with(timeout=0)

    def test_interrupted_start_reaps_manager(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            record = {"pid": process.pid, "start": "1", "argv": [sys.executable, str(source), "serve", route["nodes_path"]]}
            handlers = {}
            def interrupt(_):
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "identity", return_value=record), mock.patch.object(route_module, "ready", side_effect=interrupt), mock.patch.object(route_module, "pidfd_signal") as send, mock.patch.object(route_module.signal, "signal", side_effect=lambda number, handler: handlers.update({number: handler})):
                with self.assertRaisesRegex(InterruptedError, "start interrupted"):
                    route_module.operate("start", route)
            self.assertEqual([call.args[1] for call in send.call_args_list], [0, signal.SIGTERM])
            self.assertEqual(process.wait.call_count, 1)

    def test_stopped_exhausted_manager_allows_a_new_bounded_start(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            old = {"pid": 99999, "start": "1", "argv": route_module.manager_argv(route)}
            new = {"pid": 100000, "start": "2", "argv": route_module.manager_argv(route)}
            route_module.publish(route, {"manager": old, "digest": route["digest"], "status": "stopped", "children": []})
            process = mock.Mock(pid=new["pid"])
            process.poll.return_value = None
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process) as launch, mock.patch.object(route_module, "identity", side_effect=lambda pid: new if pid == new["pid"] else None), mock.patch.object(route_module, "ready", return_value={"manager": new}), mock.patch.object(route_module, "send") as send:
                route_module.operate("start", route)
            launch.assert_called_once()
            send.assert_not_called()

    def test_reconnect_exhaustion_keeps_healthy_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            identities = {os.getpid(): route_module.identity(os.getpid())}
            processes, starts = [], []
            now = [0.0]
            first_ready = [False]
            states = []
            def launch(argv, **kwargs):
                process = mock.Mock(pid=100000 + len(processes))
                process.poll.return_value = None if len(processes) < 64 else 255
                processes.append(process)
                starts.append(now[0])
                ssh = argv[5:]
                self.write_receipt(ssh, kwargs["stdout"])
                identities[process.pid] = {"pid": process.pid, "start": "1", "argv": ssh}
                for index, value in enumerate(ssh):
                    if value == "-R":
                        port = ssh[index + 1].split(":")[1]
                        kwargs["stderr"].write(f"remote forward success for: listen 127.0.0.1:{port}, connect 127.0.0.1:8100\n".encode())
                return process
            def publish(_, state):
                states.append(state["status"])
                if state["status"] == "ready" and not first_ready[0]:
                    first_ready[0] = True
                    processes[0].poll.return_value = 255
            def sleep(seconds):
                now[0] += seconds
                if now[0] > 60:
                    raise RuntimeError("test deadline")
            with mock.patch.object(route_module.subprocess, "Popen", side_effect=launch), mock.patch.object(route_module, "identity", side_effect=identities.get), mock.patch.object(route_module, "publish", side_effect=publish), mock.patch.object(route_module, "send"), mock.patch.object(route_module.signal, "signal"), mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, "exhausted reconnect"):
                    route_module.serve(route)
            self.assertEqual(len(processes), 64 + route["reconnect_attempts"])
            self.assertGreaterEqual(starts[65] - starts[64], 4)
            self.assertGreaterEqual(starts[66] - starts[65], 8)
            self.assertEqual(states[-1], "stopped")
            self.assertTrue(all(process.wait.called for process in processes[1:64]))

    def test_auth_failure_stops_all_started_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            process = mock.Mock(pid=123)
            process.poll.return_value = 255
            route = {**self.route, "state_dir": directory}
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module.signal, "signal"):
                with self.assertRaisesRegex(RuntimeError, "SSH owner"):
                    route_module.serve(route)

    def test_forward_ack_without_receipt_has_bounded_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            now = [0.0]
            def sleep(seconds):
                now[0] += 10
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "forwards_ready", return_value=True), mock.patch.object(route_module, "cleanup") as cleanup, mock.patch.object(route_module.signal, "signal"), mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, "SSH owner"):
                    route_module.serve(route)
            self.assertEqual(now[0], route["reconnect_timeout_seconds"])
            self.assertEqual(len(cleanup.call_args_list[0].args[0]), 1)

    def test_child_refuses_dead_manager(self):
        with mock.patch.object(route_module.ctypes, "CDLL") as libc, mock.patch.object(route_module, "identity", return_value=None), mock.patch.object(route_module.os, "execv") as execute:
            libc.return_value.prctl.return_value = 0
            with self.assertRaisesRegex(RuntimeError, "manager exited"):
                route_module.child(999999, "1", ["/usr/bin/ssh"])
            execute.assert_not_called()
            self.assertEqual(libc.return_value.prctl.call_args.args[1], signal.SIGTERM)

    def test_manager_death_stops_child(self):
        code = '''import importlib.util, os, subprocess, sys, time
spec = importlib.util.spec_from_file_location("route", sys.argv[1])
route = importlib.util.module_from_spec(spec); spec.loader.exec_module(route)
parent = route.identity(os.getpid())
child = subprocess.Popen([sys.executable, sys.argv[1], "child", str(os.getpid()), parent["start"], "/usr/bin/sleep", "30"])
while route.identity(child.pid)["argv"] != ["/usr/bin/sleep", "30"]: time.sleep(.01)
print(child.pid, flush=True)
time.sleep(30)
'''
        manager = subprocess.Popen([sys.executable, "-c", code, str(source)], stdout=subprocess.PIPE, text=True)
        try:
            pid = int(manager.stdout.readline())
            manager.kill()
            manager.wait(timeout=2)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                current = route_module.identity(pid)
                if current is None or current["argv"] == [""]:
                    break
                time.sleep(.02)
            else:
                self.fail("SSH child survived manager death")
        finally:
            if manager.poll() is None:
                manager.kill()
                manager.wait(timeout=2)
            manager.stdout.close()


if __name__ == "__main__":
    unittest.main()
