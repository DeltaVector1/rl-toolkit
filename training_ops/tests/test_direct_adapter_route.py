import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


root = Path(__file__).resolve().parents[2]
source = root / "training_ops/runtime/china/direct_adapter_route.py"
spec = importlib.util.spec_from_file_location("direct_route", source)
route_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route_module)


class DirectRouteTest(unittest.TestCase):
    def setUp(self):
        self.route = route_module.settings(root / "configs/nodes.toml")

    def test_exact_mapping_and_authentication(self):
        owners = route_module.commands(self.route)
        self.assertEqual(len(owners), 64)
        self.assertEqual([port for ports, _ in owners for port in ports], list(range(18100, 18228)))
        for ports, argv in owners:
            self.assertEqual(len(ports), 2)
            self.assertEqual(argv.count("-R"), 2)
            self.assertEqual(argv[-1], "root@69.63.236.164")
            for option in ("StrictHostKeyChecking=yes", "BatchMode=yes", "IdentitiesOnly=yes", "AddKeysToAgent=no", "ExitOnForwardFailure=yes", "ForwardAgent=no", "Compression=no", "ControlPath=none", "ProxyCommand=none"):
                self.assertIn(option, argv)
            self.assertIn("/home/ubuntu/.ssh/id_ed25519", argv)
            self.assertIn("UserKnownHostsFile=/home/ubuntu/.ssh/known_hosts", argv)
            self.assertIn("40299", argv)
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
        with mock.patch.object(signal, "pidfd_send_signal") as send:
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
        import re
        launcher = (root / "training_ops/runtime/h200/start_orchestrator.sh").read_text()
        function = re.search(r"(?ms)^direct_route_listeners_ready\(\) \{.*?^\}", launcher).group()
        rows = [f'LISTEN 0 128 127.0.0.1:{port} 0.0.0.0:* users:(("sshd",pid={1000 + (port - 18100) // 2},fd={port}))' for port in range(18100, 18228)]
        script = 'sudo() { printf "%s\\n" "$listener_fixture"; }\nlora_source_ports=(18100 18227)\n' + function + '\ndirect_route_listeners_ready\n'
        fixtures = ((rows, 0), (rows[:-1], 1), (rows + rows[:1], 1), ([row.replace('"sshd"', '"ssh"') for row in rows], 1))
        for fixture, expected in fixtures:
            result = subprocess.run(["bash", "-c", script], env={**os.environ, "listener_fixture": "\n".join(fixture)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)

    def test_status_does_not_create_state_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "absent"
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                route_module.operate("status", {**self.route, "state_dir": str(path)})
            self.assertFalse(path.exists())

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
                    for process in processes[:5]:
                        process.poll.return_value = 255
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
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "identity", return_value=record), mock.patch.object(route_module, "ready", side_effect=RuntimeError("not ready")), mock.patch.object(route_module, "send") as send, mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    route_module.operate("start", route)
            self.assertEqual(send.call_args_list, [mock.call(record, signal.SIGTERM)])
            self.assertEqual(process.wait.call_count, 2)

    def test_interrupted_start_reaps_manager(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory, "ssh_identity_file": str(source), "ssh_known_hosts_file": str(source)}
            process = mock.Mock(pid=12345)
            process.poll.return_value = None
            record = {"pid": process.pid, "start": "1", "argv": [sys.executable, str(source), "serve", route["nodes_path"]]}
            handlers = {}
            def interrupt(_):
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module, "identity", return_value=record), mock.patch.object(route_module, "ready", side_effect=interrupt), mock.patch.object(route_module, "send") as send, mock.patch.object(route_module.signal, "signal", side_effect=lambda number, handler: handlers.update({number: handler})):
                with self.assertRaisesRegex(InterruptedError, "start interrupted"):
                    route_module.operate("start", route)
            self.assertEqual(send.call_args_list, [mock.call(record, signal.SIGTERM)])
            self.assertEqual(process.wait.call_count, 2)

    def test_reconnect_exhaustion_keeps_healthy_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            route = {**self.route, "state_dir": directory}
            identities = {os.getpid(): route_module.identity(os.getpid())}
            processes, starts = [], []
            now = [0.0]
            first_ready = [False]
            def launch(argv, **kwargs):
                process = mock.Mock(pid=100000 + len(processes))
                process.poll.return_value = None if len(processes) < 64 else 255
                processes.append(process)
                starts.append(now[0])
                ssh = argv[5:]
                identities[process.pid] = {"pid": process.pid, "start": "1", "argv": ssh}
                for index, value in enumerate(ssh):
                    if value == "-R":
                        port = ssh[index + 1].split(":")[1]
                        kwargs["stderr"].write(f"remote forward success for: listen 127.0.0.1:{port}, connect 127.0.0.1:8100\n".encode())
                return process
            def publish(_, state):
                if state["status"] == "ready" and not first_ready[0]:
                    first_ready[0] = True
                    processes[0].poll.return_value = 255
                if state.get("failed_owners") and state["status"] != "stopped":
                    self.assertEqual(state["failed_owners"], [18100])
                    self.assertEqual(len(processes), 64 + route["reconnect_attempts"])
                    self.assertEqual(len(state["children"]), 63)
                    self.assertTrue(all(not process.wait.called for process in processes[1:64]))
                    self.assertGreaterEqual(starts[65] - starts[64], 4)
                    self.assertGreaterEqual(starts[66] - starts[65], 8)
                    raise RuntimeError("exhaustion passed")
            def sleep(seconds):
                now[0] += seconds
                if now[0] > 60:
                    raise RuntimeError("test deadline")
            with mock.patch.object(route_module.subprocess, "Popen", side_effect=launch), mock.patch.object(route_module, "identity", side_effect=identities.get), mock.patch.object(route_module, "publish", side_effect=publish), mock.patch.object(route_module, "send"), mock.patch.object(route_module.signal, "signal"), mock.patch.object(route_module.time, "monotonic", side_effect=lambda: now[0]), mock.patch.object(route_module.time, "sleep", side_effect=sleep):
                with self.assertRaisesRegex(RuntimeError, "exhaustion passed"):
                    route_module.serve(route)

    def test_auth_failure_stops_all_started_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            process = mock.Mock(pid=123)
            process.poll.return_value = 255
            route = {**self.route, "state_dir": directory}
            with mock.patch.object(route_module.subprocess, "Popen", return_value=process), mock.patch.object(route_module.signal, "signal"):
                with self.assertRaisesRegex(RuntimeError, "SSH owner"):
                    route_module.serve(route)

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

    def test_peer_and_relay_functions_unchanged(self):
        path = "training_ops/runtime/h200/start_orchestrator.sh"
        old = subprocess.check_output(["git", "show", "2d773ba04661e3746833fcde205124f163744966:" + path], cwd=root, text=True)
        new = (root / path).read_text()
        import re
        functions = re.findall(r"(?m)^(peer_[a-z_]+|adapter_relay_start)\(\) \{", old)
        for name in functions:
            pattern = rf"(?ms)^{name}\(\) \{{.*?^\}}"
            self.assertEqual(re.search(pattern, old).group(), re.search(pattern, new).group(), name)


if __name__ == "__main__":
    unittest.main()
