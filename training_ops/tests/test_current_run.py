import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training_ops


class CurrentRunTest(unittest.TestCase):
    def test_socket_clutter_does_not_hide_valid_run_marker(self):
        socket_path = "/home/mango/.ssh/cm-ubuntu@10.92.20.82:22"
        with mock.patch.object(
            training_ops,
            "_ssh",
            return_value=(0, f"RUN={socket_path}\nRUN=glm52\n"),
        ) as ssh:
            self.assertEqual(training_ops.current_run(), "glm52")

        ssh.assert_called_once()

    def test_socket_clutter_without_run_marker_fails_closed(self):
        socket_path = "/home/mango/.ssh/cm-ubuntu@10.92.20.82:22"
        with mock.patch.object(
            training_ops,
            "_ssh",
            side_effect=[(1, f"RUN={socket_path}\n"), (0, socket_path)],
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot identify the active run"):
                training_ops.current_run()


if __name__ == "__main__":
    unittest.main()
