"""Regression checks for the adapter control-tunnel progress probe."""

from pathlib import Path
import tomllib
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/h200/start_control_tunnel.sh"
).read_text()
with (
    Path(__file__).resolve().parents[2] / "configs/operations.toml"
).open("rb") as stream:
    OPERATIONS = tomllib.load(stream)


class ControlTunnelWatchdogTest(unittest.TestCase):
    def test_probe_is_bounded_and_recycles_after_two_misses(self):
        self.assertIn("--max-time 15", SCRIPT)
        self.assertIn("curl --head", SCRIPT)
        self.assertIn('misses=$((misses + 1))', SCRIPT)
        self.assertIn('if [ "$misses" -ge 2 ]', SCRIPT)
        self.assertIn('kill "$tunnel_pid"', SCRIPT)

    def test_success_resets_consecutive_miss_count(self):
        self.assertIn("misses=0", SCRIPT)
        probe = SCRIPT.split("if curl", 1)[1].split("else", 1)[0]
        self.assertIn("misses=0", probe)

    def test_glm_control_tunnel_uses_uncompressed_gzip_server_path(self):
        ports = OPERATIONS["ports"]
        self.assertIn("-o Compression=no", SCRIPT)
        self.assertIn(
            f'127.0.0.1:{ports["glm_adapter_tunnel"]}:'
            f'127.0.0.1:{ports["glm_adapter"]}',
            SCRIPT,
        )


if __name__ == "__main__":
    unittest.main()
