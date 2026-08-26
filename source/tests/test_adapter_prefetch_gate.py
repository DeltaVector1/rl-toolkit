"""Regression checks for pre-policy adapter staging on the H200 hosts."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SYNC = (ROOT / "scripts/h200/sync_rollout_batches.sh").read_text()
START = (ROOT / "scripts/h200/start_orchestrator.sh").read_text()
INFERENCE_START = (ROOT / "scripts/h200/start_inference.sh").read_text()
GLM_TUNNELS = (ROOT / "scripts/h200/glm_adapter_tunnels.py").read_text()
with (ROOT / "configs/operations.toml").open("rb") as stream:
    OPERATIONS = tomllib.load(stream)


class AdapterPrefetchGateTest(unittest.TestCase):
    def test_prefetch_finishes_before_stable_marker_is_exposed(self):
        pull = SYNC.split("pull_stable_markers()", 1)[1].split("\n}\n", 1)[0]
        self.assertLess(
            pull.index('prefetch_adapter "$relative"'),
            pull.index('mv "$partial" "$final"'),
        )

    def test_prefetch_uses_root_admin_routes_on_both_hosts(self):
        self.assertIn('print(url.removesuffix("/v1"), end=" ")', START)
        self.assertIn('ADAPTER_ADMIN_URLS="$adapter_admin_urls"', START)

    def test_failed_prefetch_does_not_publish_stable_marker(self):
        pull = SYNC.split("pull_stable_markers()", 1)[1].split("\n}\n", 1)[0]
        self.assertIn('prefetch_adapter "$relative" || continue', pull)

    def test_glm_adapter_uses_twelve_independent_range_streams(self):
        inventory = OPERATIONS["ports"]
        self.assertEqual(inventory["adapter_tunnel"], 18100)
        self.assertEqual(inventory["glm_adapter"], 8101)
        self.assertEqual(inventory["glm_adapter_tunnel_first"], 18200)
        self.assertEqual(inventory["glm_adapter_tunnel_last"], 18211)
        ports = " ".join(
            str(port)
            for port in range(
                inventory["glm_adapter_tunnel_first"],
                inventory["glm_adapter_tunnel_last"] + 1,
            )
        )
        self.assertIn('adapter_tunnel_ports=("${glm_adapter_tunnel_ports[@]}")', START)
        self.assertIn(
            'model == "/models/glm52-fp8"',
            START,
        )
        self.assertIn('adapter_backend_port=$glm_adapter_backend_port', START)
        self.assertIn("adapter_compression=no", START)
        self.assertIn('adapter_backend_port=$muse_adapter_backend_port', START)
        self.assertIn("adapter_compression=yes", START)
        self.assertIn('adapter_tunnel_ports=("$muse_adapter_tunnel_port")', START)
        self.assertIn(
            'glm_tunnel_launch_id=$(python3 "$glm_tunnel_manager" '
            '--operations "$operations_file" start)',
            START,
        )
        self.assertIn("trap cleanup_uncommitted_glm_tunnels EXIT", START)
        self.assertIn(
            'python3 "$glm_tunnel_manager" --operations "$operations_file" cleanup',
            START,
        )
        self.assertIn(
            'python3 "$glm_tunnel_manager" --operations "$operations_file" commit',
            START,
        )
        self.assertLess(
            START.index("trap cleanup_uncommitted_glm_tunnels EXIT"),
            START.index('if [ ! -r "$env_config_file" ]'),
        )
        self.assertGreater(
            START.rindex("commit_glm_tunnels"),
            START.index("durable batch publisher started"),
        )
        self.assertIn('config_ports["glm_adapter_tunnel_first"]', GLM_TUNNELS)
        self.assertIn('config_ports["glm_adapter_tunnel_last"]', GLM_TUNNELS)
        self.assertIn('config_ports["glm_adapter"]', GLM_TUNNELS)
        self.assertIn('config_hosts["trainer_ssh"]', GLM_TUNNELS)
        self.assertIn('"starttime"', GLM_TUNNELS)
        self.assertIn("_record_matches", GLM_TUNNELS)
        self.assertEqual(
            OPERATIONS["paths"]["glm_adapter_tunnel_state"],
            "/data/glm52-rl/glm-adapter-tunnels.json",
        )
        self.assertIn('config_paths["glm_adapter_tunnel_state"]', GLM_TUNNELS)
        self.assertIn(
            'tunnel "$adapter_port" "127.0.0.1:$adapter_backend_port" '
            '"$adapter_ssh_target" "$adapter_compression"',
            START,
        )
        self.assertIn(
            "adapter_ports=${prime_adapter_ports:-" + ports.replace(" ", ",") + "}",
            INFERENCE_START,
        )
        self.assertIn('-e PRIME_GLM_ADAPTER_GZIP=1', INFERENCE_START)
        self.assertIn(
            'adapter_source_path="runs/$RUN/$RUN_NAME"',
            START,
        )
        self.assertIn(
            'ADAPTER_SOURCE_BASE="http://127.0.0.1:$adapter_canonical_port/',
            START,
        )
        self.assertEqual(OPERATIONS["paths"]["glm_adapter_root"], "/data/glm52-rl")

    def test_parallel_batch_copy_staggers_ssh_handshakes(self):
        copy = SYNC.split("copy_parallel()", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("PARALLEL_LAUNCH_DELAY", SYNC)
        self.assertLess(copy.index('pids+=("$!")'), copy.index('sleep "$parallel_launch_delay"'))

    def test_stalled_batch_stripes_retry_within_one_minute(self):
        self.assertIn('stripe_io_timeout=${STRIPE_IO_TIMEOUT:-60}', SYNC)
        self.assertIn('--timeout="$stripe_io_timeout"', SYNC)
        self.assertNotIn("--timeout=120", SYNC)

    def test_wedged_ssh_child_has_a_wall_clock_bound(self):
        self.assertIn('stripe_attempt_timeout=${STRIPE_ATTEMPT_TIMEOUT:-90}', SYNC)
        self.assertIn(
            'timeout --signal=TERM --kill-after=10 "$stripe_attempt_timeout"',
            SYNC,
        )

    def test_launcher_releases_lock_inherited_by_orphan_transfer(self):
        self.assertIn('lsof -t -- "$publisher_lock"', START)
        self.assertIn('ps -o pgid= -p "$holder"', START)
        self.assertIn('kill -- "-$holder_pgid"', START)
        self.assertLess(
            START.index('lsof -t -- "$publisher_lock"'),
            START.index("publisher_lock_ready=0"),
        )


if __name__ == "__main__":
    unittest.main()
