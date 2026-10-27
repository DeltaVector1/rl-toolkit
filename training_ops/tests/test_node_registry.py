import contextlib
import io
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

import training_ops
from training_ops.runtime import node_registry


class NodeRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "nodes.toml"
        self.path.write_text(
            """[hosts]
blocked = ["n79"]

[selection]
note = "keep this"
active_node = "old"

[single_node.old]
ssh_host = "old.example"
ssh_port = 22
ssh_user = "mango"
ssh_identity_file = "/keys/old"
runs = "/runs/old"
python = "/venv/bin/python"
role = "trainer"
custom_value = "keep this too"

[single_node.n79]
ssh_host = "n79"
ssh_port = 22
ssh_user = "blocked"
ssh_identity_file = "/keys/blocked"
runs = "/runs/blocked"
python = "/venv/bin/python"

[unrelated]
keep = true
""",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def add_args(self, name="new", *extra):
        return [
            "add", name,
            "--ssh-host", "node.example",
            "--ssh-port", "20022",
            "--ssh-user", "runner",
            "--identity", "/keys/node",
            "--runs", "/data/runs",
            "--python", "/venv/bin/python3",
            *extra,
        ]

    def test_active_host_and_profile(self):
        nodes = {
            "selection": {"active_node": "host-a"},
            "single_node": {"host-a": {"ssh_host": "host-a.example"}},
        }
        self.assertEqual(node_registry.active_host(nodes), "host-a")
        self.assertEqual(node_registry.profile(nodes, "host-a"), {"ssh_host": "host-a.example"})
        self.assertIsNone(node_registry.profile(nodes, "missing"))
        self.assertIsNone(node_registry.active_host({}))

    def test_add_preserves_other_tables_and_existing_fields(self):
        original = self.path.read_text(encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = node_registry.cli(self.add_args("new", "--role", "rollout"), path=self.path)
        self.assertEqual(result, 0)
        self.assertIn("Registered node profile new.", output.getvalue())
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(original))
        nodes = tomllib.loads(text)
        self.assertEqual(nodes["selection"], {"note": "keep this", "active_node": "old"})
        self.assertEqual(nodes["single_node"]["old"]["custom_value"], "keep this too")
        self.assertEqual(nodes["unrelated"], {"keep": True})
        self.assertEqual(
            nodes["single_node"]["new"],
            {
                "ssh_host": "node.example",
                "ssh_port": 20022,
                "ssh_user": "runner",
                "ssh_identity_file": "/keys/node",
                "runs": "/data/runs",
                "python": "/venv/bin/python3",
                "role": "rollout",
            },
        )
        self.assertEqual(node_registry.profile(nodes, "new")["runs"], "/data/runs")
        self.assertEqual(node_registry.profile(nodes, "new")["python"], "/venv/bin/python3")

    def test_add_updates_only_given_fields_and_keeps_other_profile_fields(self):
        output = io.StringIO()
        args = ["nodes", "add", "old", "--ssh-host", "new.example", "--ssh-port", "2200"]
        with contextlib.redirect_stdout(output):
            result = node_registry.cli(args, path=self.path)
        self.assertEqual(result, 0)
        self.assertIn("Registered node profile old.", output.getvalue())
        profile = tomllib.loads(self.path.read_text(encoding="utf-8"))["single_node"]["old"]
        self.assertEqual(profile["ssh_host"], "new.example")
        self.assertEqual(profile["ssh_port"], 2200)
        self.assertEqual(profile["ssh_user"], "mango")
        self.assertEqual(profile["ssh_identity_file"], "/keys/old")
        self.assertEqual(profile["runs"], "/runs/old")
        self.assertEqual(profile["python"], "/venv/bin/python")
        self.assertEqual(profile["role"], "trainer")
        self.assertEqual(profile["custom_value"], "keep this too")

    def test_use_updates_active_node_and_preserves_selection_fields(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = node_registry.cli(["use", "old"], path=self.path)
        self.assertEqual(result, 0)
        self.assertIn("Selected node profile old.", output.getvalue())
        nodes = tomllib.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(nodes["selection"], {"note": "keep this", "active_node": "old"})

    def test_list_shows_selected_profile(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = node_registry.cli(["list"], path=self.path)
        self.assertEqual(result, 0)
        self.assertIn("* old old.example (trainer)", output.getvalue())

    def test_invalid_name_port_and_blocked_hosts_do_not_write(self):
        before = self.path.read_bytes()
        cases = (
            ["add", "bad.name", "--ssh-host", "node.example", "--ssh-port", "22", "--ssh-user", "r",
             "--identity", "/k", "--runs", "/r", "--python", "/p"],
            ["add", "new", "--ssh-host", "node.example", "--ssh-port", "65536", "--ssh-user", "r",
             "--identity", "/k", "--runs", "/r", "--python", "/p"],
            ["add", "n79", "--ssh-host", "node.example", "--ssh-port", "22", "--ssh-user", "r",
             "--identity", "/k", "--runs", "/r", "--python", "/p"],
            ["add", "new", "--ssh-host", "n79", "--ssh-port", "22", "--ssh-user", "r",
             "--identity", "/k", "--runs", "/r", "--python", "/p"],
            ["use", "n79"],
        )
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(node_registry.cli(args, path=self.path), 2)
            self.assertEqual(self.path.read_bytes(), before)

    def test_add_requires_fields_for_a_new_profile(self):
        before = self.path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            result = node_registry.cli(["add", "new", "--ssh-host", "node.example"], path=self.path)
        self.assertEqual(result, 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_default_file_change_refreshes_training_ops_nodes(self):
        original_nodes = dict(training_ops.nodes)
        try:
            with mock.patch.object(node_registry, "default_path", self.path):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = node_registry.cli(["use", "old"], path=self.path)
            self.assertEqual(result, 0)
            self.assertEqual(training_ops.nodes["selection"]["active_node"], "old")
        finally:
            training_ops.nodes.clear()
            training_ops.nodes.update(original_nodes)


if __name__ == "__main__":
    unittest.main()
