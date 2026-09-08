import contextlib
import hashlib
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

    def test_bundle_keeps_manifest_and_host_selection(self):
        work = self.root / "work"
        work.mkdir()
        bundle, destinations = training_ops._create_dependency_bundle(
            [("a.py", ["n76"]), ("nested/b.py", ["n77"])], work
        )

        self.assertEqual(destinations, {"a.py": ["n76"], "nested/b.py": ["n77"]})
        with tarfile.open(bundle) as archive:
            self.assertEqual(
                archive.getnames(),
                ["a.py", "nested/b.py", training_ops.dependency_manifest_name],
            )
            manifest = archive.extractfile(training_ops.dependency_manifest_name)
            text = manifest.read().decode("utf-8")
        self.assertIn(f"{hashlib.sha256(b'a\n').hexdigest()}\ta.py\n", text)
        self.assertIn(f"{hashlib.sha256(b'b\n').hexdigest()}\tnested/b.py\n", text)

    def test_blocked_and_traversal_paths_are_refused_without_network(self):
        with self.assertRaises(training_ops.Refused):
            training_ops._deploy_dependencies([("a.py", ["n79"])])
        with self.assertRaises(training_ops.Refused):
            training_ops._create_dependency_bundle([("../a.py", ["n76"])], self.root)

    def test_copy_retry_is_bounded(self):
        copy = mock.Mock(side_effect=[(1, "closed"), (1, "closed"), (1, "closed")])
        cleanup = mock.Mock()
        with (
            mock.patch.object(training_ops.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "upload failed"),
        ):
            training_ops._retry_dependency_copy("upload", copy, cleanup)
        self.assertEqual(copy.call_count, 3)
        self.assertEqual(cleanup.call_count, 3)

    def test_copy_retry_stops_after_success(self):
        copy = mock.Mock(side_effect=[(1, "closed"), (0, "")])
        cleanup = mock.Mock()
        with mock.patch.object(training_ops.time, "sleep"):
            training_ops._retry_dependency_copy("upload", copy, cleanup)
        self.assertEqual(copy.call_count, 2)
        cleanup.assert_called_once_with()

    def test_copy_file_uses_content_checked_resumable_rsync(self):
        local = self.root / "archive.bin"
        local.write_bytes(b"replacement")
        host = training_ops.trainer_hosts[0]
        remote = f"{training_ops.storage_root}/copy-file/archive.bin"
        commands = []
        ssh_scripts = []

        def fake_ssh(_host, script, timeout=0):
            ssh_scripts.append(script)
            if script.startswith("mkdir -p "):
                return 0, ""
            if script.startswith("stat -c %s -- "):
                return 0, f"{local.stat().st_size}\n"
            self.fail(f"unexpected SSH script: {script}")

        def fake_run(command, **kwargs):
            commands.append(command)
            return training_ops.subprocess.CompletedProcess(command, 0, "", "")

        with (
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
        ):
            self.assertTrue(training_ops.copy_file(local, host, remote))

        self.assertEqual(len(commands), 1)
        self.assertIn("--partial", commands[0])
        self.assertIn("--checksum", commands[0])
        self.assertNotIn("--append-verify", commands[0])
        self.assertFalse(any(script.startswith("truncate ") for script in ssh_scripts))

    def test_trainer_bundle_uses_one_edge_upload(self):
        uploads = []
        remote_calls = []

        def fake_run(command, **kwargs):
            source = Path(command[-2])
            with tarfile.open(source) as archive:
                self.assertEqual(
                    archive.getnames(),
                    ["a.py", "nested/b.py", training_ops.dependency_manifest_name],
                )
            uploads.append((command, kwargs))
            return training_ops.subprocess.CompletedProcess(command, 0, "", "")

        def fake_ssh(host, script, timeout=0):
            remote_calls.append((host, script, timeout))
            if "checksums match" in script:
                return 1, "missing"
            return 0, "installed"

        output = io.StringIO()
        with (
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            contextlib.redirect_stdout(output),
        ):
            training_ops._deploy_dependencies(
                [("a.py", ["n76"]), ("nested/b.py", ["n77"])]
            )

        self.assertEqual(len(uploads), 1)
        self.assertIn("h200a:", uploads[0][0][-1])
        self.assertTrue(
            any(
                host == "h200a" and "china-trainer:" in script
                for host, script, _ in remote_calls
            )
        )
        scripts = {
            host: script
            for host, script, _ in remote_calls
            if "install_paths=" in script
        }
        self.assertIn("install_paths=(a.py)", scripts["n76"])
        self.assertIn("install_paths=(nested/b.py)", scripts["n77"])
        self.assertIn("trainer dependency bundle: complete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
