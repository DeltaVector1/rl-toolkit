import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_ops


class StartDependencyInventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_root = training_ops.repo_root
        training_ops.repo_root = self.root
        for directory in (
            "training_ops/runtime/china",
            "training_ops/runtime/h200",
            "configs/general-agent",
        ):
            (self.root / directory).mkdir(parents=True)
        (self.root / "configs/nodes.toml").write_text("nodes\n", encoding="utf-8")
        (self.root / "configs/general-agent/inference.toml").write_text(
            "inference\n", encoding="utf-8"
        )
        (self.root / "training_ops/runtime/china/cluster.sh").write_text(
            "cluster\n", encoding="utf-8"
        )
        (self.root / "training_ops/runtime/h200/start.sh").write_text(
            "start\n", encoding="utf-8"
        )
        (self.root / "configs/run/trainer.toml").parent.mkdir()
        (self.root / "configs/run/trainer.toml").write_text(
            "trainer\n", encoding="utf-8"
        )

    def tearDown(self):
        training_ops.repo_root = self.old_root
        self.temp.cleanup()

    def _inventory(self, environment_id):
        orchestrator = self.root / "configs/run/orchestrator.toml"
        environment_config = (
            f'env.taskset.id = "{environment_id}"'
            if environment_id == "decensor-v1"
            else f'id = "{environment_id}"'
        )
        orchestrator.write_text(
            f"[[train.env]]\n{environment_config}\n", encoding="utf-8"
        )
        return dict(
            training_ops._start_dependency_inventory(
                {
                    "trainer": "configs/run/trainer.toml",
                    "orchestrator": "configs/run/orchestrator.toml",
                },
                [training_ops.trainer_hosts[0]],
            )
        )

    def test_decensor_id_selects_all_non_bytecode_files_for_orchestrator(self):
        decensor = self.root / "environments/decensor"
        (decensor / "src/decensor_v1").mkdir(parents=True)
        (decensor / "pyproject.toml").write_text("project\n", encoding="utf-8")
        (decensor / "src/decensor_v1/__init__.py").write_text(
            "package\n", encoding="utf-8"
        )
        (decensor / "src/decensor_v1/taskset.py").write_text(
            "taskset\n", encoding="utf-8"
        )
        (decensor / "src/decensor_v1/compiled.pyc").write_bytes(b"bytecode")
        (decensor / ".venv/lib/python3.13/site-packages").mkdir(parents=True)
        (decensor / ".venv/lib/python3.13/site-packages/cached.py").write_text(
            "cache\n", encoding="utf-8"
        )

        with mock.patch.object(training_ops, "mounted_files", return_value=[]):
            inventory = self._inventory("decensor-v1")

        self.assertEqual(inventory["environments/decensor/pyproject.toml"], ["h200a"])
        self.assertEqual(
            inventory["environments/decensor/src/decensor_v1/taskset.py"], ["h200a"]
        )
        self.assertEqual(
            inventory["environments/decensor/src/decensor_v1/__init__.py"], ["h200a"]
        )
        self.assertNotIn("environments/decensor/src/decensor_v1/compiled.pyc", inventory)
        self.assertFalse(any(path.startswith("environments/decensor/.venv/") for path in inventory))
        self.assertFalse(any(path.startswith("environments/general-agent/") for path in inventory))

    def test_general_agent_id_keeps_python_source_selection(self):
        general_agent = self.root / "environments/general-agent/general_agent"
        general_agent.mkdir(parents=True)
        (general_agent / "taskset.py").write_text("taskset\n", encoding="utf-8")
        (general_agent / "README.md").write_text("readme\n", encoding="utf-8")

        with mock.patch.object(training_ops, "mounted_files", return_value=[]):
            inventory = self._inventory("general-agent")

        self.assertEqual(
            inventory["environments/general-agent/general_agent/taskset.py"], ["h200a"]
        )
        self.assertFalse(any(path.endswith("README.md") for path in inventory))

    def test_unknown_environment_id_is_not_silently_ignored(self):
        with mock.patch.object(training_ops, "mounted_files", return_value=[]):
            with self.assertRaisesRegex(training_ops.Refused, "unsupported.*unknown"):
                self._inventory("unknown")

    def test_orchestrator_launcher_includes_both_environment_source_paths(self):
        launcher = Path(__file__).resolve().parents[1] / "runtime/h200/start_orchestrator.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn(
            'PYTHONPATH="$deployment/environments/decensor/src:$deployment/environments/general-agent:/app/src"',
            text,
        )


if __name__ == "__main__":
    unittest.main()
