import errno
import socket
import subprocess
import sys
import unittest

import training_ops
from training_ops.tests.network_guard import install

install()


class NetworkGuardTest(unittest.TestCase):
    def test_operational_boundaries_require_mocks(self):
        for function, args, kwargs in (
            (training_ops.deploy, ("configs/nodes.toml",), {}),
            (training_ops._ssh, ("n82", "true"), {}),
            (training_ops._run_process, (["true"],), {"stdin": subprocess.DEVNULL, "timeout": 1}),
        ):
            with self.assertRaises(AssertionError):
                function(*args, **kwargs)

    def test_direct_ssh_process_is_rejected_before_start(self):
        with self.assertRaisesRegex(AssertionError, "live-capable"):
            subprocess.run(["/usr/bin/ssh", "-V"], check=False)

    def test_network_is_blocked_in_current_and_child_process(self):
        with socket.socket() as connection:
            with self.assertRaises(PermissionError) as caught:
                connection.connect(("127.0.0.1", 9))
            self.assertEqual(caught.exception.errno, errno.EPERM)
        result = subprocess.run([sys.executable, "-c", "import socket; socket.socket().connect(('127.0.0.1',9))"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PermissionError", result.stderr)
