import importlib.util
import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/h200/glm_adapter_tunnels.py"
spec = importlib.util.spec_from_file_location("glm_adapter_tunnels", SCRIPT)
tunnels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tunnels)
tunnels._configure(ROOT / "configs/operations.toml")


class ProcessSimulation:
    def __init__(self):
        self.next_pid = 2000
        self.identities = {}
        self.listeners = set()
        self.killed = []
        self.now = 0

    def spawn(self, port):
        self.next_pid += 1
        pid = self.next_pid
        self.identities[pid] = {
            "pgid": pid,
            "starttime": pid * 10,
            "cmdline": tunnels._supervisor_command(port),
        }
        self.listeners.add(port)
        return pid

    def identity(self, pid):
        value = self.identities.get(pid)
        return dict(value) if value is not None else None

    def listening(self, port):
        return port in self.listeners

    def kill(self, pid, action):
        self.killed.append((pid, action))
        identity = self.identities.pop(pid, None)
        if identity is not None:
            self.listeners.discard(int(identity["cmdline"][-1]))

    def monotonic(self):
        self.now += 1
        return self.now


class GlmAdapterTunnelTest(unittest.TestCase):
    def run_with_simulation(self, directory, simulation):
        return (
            patch.object(tunnels, "state_path", Path(directory) / "state.json"),
            patch.object(tunnels, "_spawn_supervisor", side_effect=simulation.spawn),
            patch.object(tunnels, "_process_identity", side_effect=simulation.identity),
            patch.object(tunnels, "_listening", side_effect=simulation.listening),
            patch.object(tunnels, "_kill_group", side_effect=simulation.kill),
            patch.object(tunnels, "_script_sha256", return_value="a" * 64),
            patch.object(tunnels.time, "sleep"),
            patch.object(tunnels.time, "monotonic", side_effect=simulation.monotonic),
        )

    def test_two_consecutive_starts_replace_only_recorded_groups(self):
        simulation = ProcessSimulation()
        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                first = tunnels.start()
                tunnels.commit(first)
                first_pids = {
                    record["pid"] for record in tunnels._read_state()["records"]
                }
                second = tunnels.start()
                second_pids = {
                    record["pid"] for record in tunnels._read_state()["records"]
                }
                self.assertNotEqual(first, second)
                self.assertTrue(first_pids.isdisjoint(second_pids))
                self.assertEqual(
                    {pid for pid, action in simulation.killed if action == signal.SIGTERM},
                    first_pids,
                )
                tunnels.cleanup(second)

    def test_cleanup_after_later_launch_failure_removes_current_groups(self):
        simulation = ProcessSimulation()
        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                launch_id = tunnels.start()
                pids = {record["pid"] for record in tunnels._read_state()["records"]}
                tunnels.cleanup(launch_id)
                self.assertEqual(simulation.identities, {})
                self.assertEqual(simulation.listeners, set())
                self.assertEqual(
                    {pid for pid, action in simulation.killed if action == signal.SIGTERM},
                    pids,
                )
                self.assertIsNone(tunnels._read_state())

    def test_stale_pid_starttime_is_never_killed(self):
        simulation = ProcessSimulation()
        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                first = tunnels.start()
                tunnels.commit(first)
                stale = tunnels._read_state()["records"][0]
                simulation.identities[stale["pid"]]["starttime"] += 1
                simulation.listeners.discard(stale["port"])
                second = tunnels.start()
                self.assertNotIn(
                    stale["pid"],
                    {pid for pid, _action in simulation.killed},
                )
                tunnels.cleanup(second)

    def test_unrecorded_occupied_port_fails_before_spawn(self):
        simulation = ProcessSimulation()
        simulation.listeners.add(tunnels.ports[3])
        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                with self.assertRaisesRegex(RuntimeError, "refusing occupied"):
                    tunnels.start()
                self.assertEqual(simulation.next_pid, 2000)

    def test_state_records_exact_ports_targets_and_commands(self):
        simulation = ProcessSimulation()
        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                launch_id = tunnels.start()
                state = json.loads(tunnels.state_path.read_text())
                self.assertEqual(state["ports"], list(tunnels.ports))
                self.assertEqual(state["target"], tunnels.target)
                for record in state["records"]:
                    self.assertEqual(
                        record["cmdline"],
                        tunnels._supervisor_command(record["port"]),
                    )
                tunnels.cleanup(launch_id)

    def test_interrupt_after_three_spawns_cleans_durable_partial_state(self):
        simulation = ProcessSimulation()
        observed = []

        def interrupting_spawn(port):
            if len(observed) == 3:
                state = tunnels._read_state()
                self.assertFalse(state["committed"])
                self.assertEqual(len(state["records"]), 3)
                raise KeyboardInterrupt
            pid = simulation.spawn(port)
            observed.append(pid)
            return pid

        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with (
                patches[0],
                patch.object(tunnels, "_spawn_supervisor", side_effect=interrupting_spawn),
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
            ):
                with self.assertRaises(KeyboardInterrupt):
                    tunnels.start()
                self.assertEqual(simulation.identities, {})
                self.assertEqual(simulation.listeners, set())
                self.assertIsNone(tunnels._read_state())

    def test_status_reports_recorded_and_healthy_tunnels(self):
        simulation = ProcessSimulation()
        with tempfile.TemporaryDirectory() as directory:
            patches = self.run_with_simulation(directory, simulation)
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
                launch_id = tunnels.start()
                self.assertEqual(
                    tunnels.status(),
                    {"recorded": 12, "healthy": 12, "committed": False},
                )
                tunnels.commit(launch_id)
                self.assertTrue(tunnels.status()["committed"])
                tunnels.cleanup(launch_id)


if __name__ == "__main__":
    unittest.main()
