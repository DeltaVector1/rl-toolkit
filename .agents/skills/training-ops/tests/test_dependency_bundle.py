import contextlib
import hashlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class DependencyBundleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_root = training_ops.repo_root
        training_ops.repo_root = self.root
        (self.root / "a.py").write_text("a\n", encoding="utf-8")
        (self.root / "nested").mkdir()
        (self.root / "nested" / "b.py").write_text("b\n", encoding="utf-8")

    def tearDown(self):
        training_ops.repo_root = self.old_root
        self.temp.cleanup()

    def test_bundle_is_one_archive_per_host_and_preserves_selection(self):
        scp_calls = []
        ssh_calls = []

        def fake_run(command, **_kwargs):
            source = Path(command[-2])
            with tarfile.open(source) as archive:
                names = archive.getnames()
                manifest = archive.extractfile(training_ops.dependency_manifest_name)
                manifest = manifest.read().decode("utf-8")
            scp_calls.append((command[-1].split(":", 1)[0], names, manifest, source))
            return training_ops.subprocess.CompletedProcess(command, 0, "", "")

        def fake_ssh(host, script, timeout=0):
            ssh_calls.append((host, script, timeout))
            return 0, "installed"

        required = [("a.py", ["n76"]), ("nested/b.py", ["n77"])]
        output = io.StringIO()
        with (
            mock.patch.object(training_ops.subprocess, "run", side_effect=fake_run),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            contextlib.redirect_stdout(output),
        ):
            training_ops._deploy_dependencies(required)

        self.assertEqual({host for host, *_ in scp_calls}, {"n76", "n77"})
        self.assertEqual(len(scp_calls), 2)
        self.assertTrue(all(not source.exists() for *_, source in scp_calls))
        for _host, names, manifest, _source in scp_calls:
            self.assertEqual(names, ["a.py", "nested/b.py", training_ops.dependency_manifest_name])
            self.assertIn(
                f"{hashlib.sha256(b'a\n').hexdigest()}\ta.py\n", manifest
            )
            self.assertIn(
                f"{hashlib.sha256(b'b\n').hexdigest()}\tnested/b.py\n", manifest
            )
        scripts = {host: script for host, script, _timeout in ssh_calls}
        self.assertIn("install_paths=(a.py)", scripts["n76"])
        self.assertIn("install_paths=(nested/b.py)", scripts["n77"])
        self.assertNotIn("nested/b.py", scripts["n76"].split("install_paths=", 1)[1].split("temp=", 1)[0])
        self.assertIn("sha256sum", scripts["n76"])
        self.assertIn("--no-same-owner", scripts["n76"])
        self.assertIn("rm -rf --", scripts["n76"])
        self.assertIn("trainer dependency bundle: complete", output.getvalue())

    def test_blocked_and_traversal_paths_are_refused_without_network(self):
        with self.assertRaises(training_ops.Refused):
            training_ops._deploy_dependencies([("a.py", ["n79"])])
        with self.assertRaises(training_ops.Refused):
            training_ops._create_dependency_bundle(
                [("../a.py", ["n76"])], self.root
            )


if __name__ == "__main__":
    unittest.main()
