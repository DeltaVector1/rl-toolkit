import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import glm52_ops


class DeployTest(unittest.TestCase):
    def test_deploy_makes_a_regular_file_container_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            local = repo / "configs" / "test.toml"
            local.parent.mkdir(parents=True)
            local.write_text("value = 1\n")
            want = hashlib.md5(local.read_bytes()).hexdigest()
            ssh_calls = []

            def fake_ssh(host, command, timeout=120):
                ssh_calls.append((host, command))
                if command.startswith("md5sum "):
                    return 0, want
                return 0, ""

            completed = Mock(returncode=0)
            with (
                patch.object(glm52_ops, "REPO", repo),
                patch.object(glm52_ops, "_ssh", side_effect=fake_ssh),
                patch.object(glm52_ops.subprocess, "run", return_value=completed),
            ):
                self.assertTrue(
                    glm52_ops.deploy("configs/test.toml", nodes=["h200a"])
                )

            self.assertTrue(
                any(command.startswith("chmod 0644 ") for _, command in ssh_calls)
            )


if __name__ == "__main__":
    unittest.main()
