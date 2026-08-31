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
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
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

    def test_retries_failed_host_until_success_without_repeating_success(self):
        upload_calls = []
        remote_calls = []

        def fake_run(command, **kwargs):
            upload_calls.append((command, kwargs))
            if len(upload_calls) == 1:
                return training_ops.subprocess.CompletedProcess(
                    command, 1, "", "scp: Connection closed\n"
                )
            self.assertEqual(command[0], "ssh")
            self.assertNotEqual(kwargs["stdin"], training_ops.subprocess.DEVNULL)
            return training_ops.subprocess.CompletedProcess(command, 0, "", "")

        def fake_ssh(host, script, timeout=0):
            if script.startswith("rm -f"):
                return 0, ""
            if "checksums match" in script:
                return 1, "missing"
            remote_calls.append(host)
            return 0, "installed"

        output = io.StringIO()
        errors = io.StringIO()
        with (
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            mock.patch.object(training_ops.time, "sleep"),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            training_ops._deploy_dependencies([("a.py", ["n76"])])

        self.assertEqual(len(upload_calls), 2)
        self.assertEqual(upload_calls[0][0][0], "scp")
        self.assertEqual(upload_calls[1][0][0], "ssh")
        self.assertIn("umask 077; cat > /tmp/trainer-dependencies-", upload_calls[1][0][-1])
        self.assertEqual(remote_calls, ["n76"])
        self.assertEqual(errors.getvalue().count("scp: Connection closed"), 1)
        self.assertIn("attempt 1/3", errors.getvalue())

    def test_terminal_failure_is_bounded_at_three_attempts(self):
        upload_calls = []

        def fake_run(command, **kwargs):
            upload_calls.append((command, kwargs))
            if len(upload_calls) == 1:
                return training_ops.subprocess.CompletedProcess(
                    command, 1, "", "scp: Connection closed\n"
                )
            return training_ops.subprocess.CompletedProcess(
                command, 1, "", "ssh: stream closed\n"
            )

        def fake_ssh(_host, script, timeout=0):
            if script.startswith("rm -f"):
                return 0, ""
            if "checksums match" in script:
                return 1, "missing"
            self.fail(f"verification must not run after failed scp: {script}")

        output = io.StringIO()
        errors = io.StringIO()
        with (
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            mock.patch.object(training_ops.time, "sleep"),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to deploy dependency bundle"):
                training_ops._deploy_dependencies([("a.py", ["n76"])])

        self.assertEqual(len(upload_calls), 3)
        self.assertEqual(upload_calls[0][0][0], "scp")
        self.assertTrue(all(call[0][0] == "ssh" for call in upload_calls[1:]))
        self.assertTrue(all(call[1]["timeout"] == 60 for call in upload_calls))
        self.assertEqual(errors.getvalue().count("scp: Connection closed"), 1)
        self.assertEqual(errors.getvalue().count("ssh: stream closed"), 2)
        self.assertIn("attempt 3/3", errors.getvalue())

    def test_checksum_match_skips_upload_without_marker(self):
        ssh_calls = []

        def fake_ssh(host, script, timeout=0):
            ssh_calls.append((host, script, timeout))
            if "checksums match" in script:
                self.assertIn(hashlib.sha256(b"a\n").hexdigest(), script)
                return 0, "checksums match\n"
            self.fail(f"upload verification must not run after checksum skip: {script}")

        def fail_scp(*_args, **_kwargs):
            self.fail("SCP must not run when every remote checksum matches")

        output = io.StringIO()
        with (
            mock.patch.object(training_ops, "_run_process", side_effect=fail_scp),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            contextlib.redirect_stdout(output),
        ):
            training_ops._deploy_dependencies([("a.py", ["n76"])])

        self.assertEqual(len(ssh_calls), 1)
        self.assertEqual(ssh_calls[0][2], 60)
        self.assertIn("[skipped]", output.getvalue())

    def test_checksum_mismatch_uploads_with_bundle_timeout(self):
        scp_calls = []
        ssh_calls = []

        def fake_ssh(host, script, timeout=0):
            ssh_calls.append((host, script, timeout))
            if "checksums match" in script:
                self.assertNotIn("rm -f", script)
                self.assertIn("sha256sum", script)
                return 1, "checksum mismatch"
            return 0, "installed"

        def fake_run(command, **kwargs):
            scp_calls.append((command, kwargs))
            return training_ops.subprocess.CompletedProcess(command, 0, "", "")

        output = io.StringIO()
        with (
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            contextlib.redirect_stdout(output),
        ):
            training_ops._deploy_dependencies([("a.py", ["n76"])])

        self.assertEqual(len(scp_calls), 1)
        self.assertEqual(scp_calls[0][1]["timeout"], 60)
        self.assertIn("ServerAliveInterval", " ".join(scp_calls[0][0]))
        self.assertIn("ServerAliveCountMax", " ".join(scp_calls[0][0]))
        self.assertEqual(len(ssh_calls), 2)
        self.assertIn("[ok]", output.getvalue())

    def test_scp_timeout_then_ssh_stream_success_preserves_stderr(self):
        upload_calls = []
        errors = io.StringIO()
        remote_calls = []

        def fake_ssh(_host, script, timeout=0):
            if "checksums match" in script:
                return 1, "missing"
            if script.startswith("rm -f"):
                remote_calls.append("cleanup")
                return 0, ""
            remote_calls.append("verify")
            return 0, "installed"

        def fake_run(command, **kwargs):
            upload_calls.append((command, kwargs))
            if len(upload_calls) == 1:
                raise training_ops.subprocess.TimeoutExpired(
                    command, kwargs["timeout"], stderr=b"scp: stalled\n"
                )
            self.assertEqual(command[0], "ssh")
            self.assertIn("umask 077; cat > /tmp/trainer-dependencies-", command[-1])
            self.assertNotEqual(kwargs["stdin"], training_ops.subprocess.DEVNULL)
            self.assertEqual(kwargs["timeout"], 60)
            return training_ops.subprocess.CompletedProcess(command, 0, "", "")

        with (
            mock.patch.object(training_ops, "_run_process", side_effect=fake_run),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
            mock.patch.object(training_ops.time, "sleep"),
            contextlib.redirect_stderr(errors),
        ):
            training_ops._deploy_dependencies([("a.py", ["n76"])])

        self.assertEqual(len(upload_calls), 2)
        self.assertEqual(upload_calls[0][0][0], "scp")
        self.assertEqual(upload_calls[1][0][0], "ssh")
        self.assertEqual(remote_calls, ["cleanup", "verify"])
        self.assertEqual(errors.getvalue().count("scp: stalled\n"), 1)


if __name__ == "__main__":
    unittest.main()
