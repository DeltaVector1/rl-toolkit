"""Regression checks for pre-policy adapter staging on the H200 hosts."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SYNC = (ROOT / "scripts/h200/sync_rollout_batches.sh").read_text()
START = (ROOT / "scripts/h200/start_orchestrator.sh").read_text()


class AdapterPrefetchGateTest(unittest.TestCase):
    def test_prefetch_finishes_before_stable_marker_is_exposed(self):
        pull = SYNC.split("pull_stable_markers()", 1)[1].split("\n}\n", 1)[0]
        self.assertLess(
            pull.index('prefetch_adapter "$relative"'),
            pull.index('mv "$partial" "$final"'),
        )

    def test_prefetch_uses_root_admin_routes_on_both_hosts(self):
        self.assertIn(
            'ADAPTER_ADMIN_URLS="http://127.0.0.1:8000 http://127.0.0.1:18101"',
            START,
        )
        self.assertNotIn("127.0.0.1:8000/v1 http://127.0.0.1:18101/v1", START)

    def test_failed_prefetch_does_not_publish_stable_marker(self):
        pull = SYNC.split("pull_stable_markers()", 1)[1].split("\n}\n", 1)[0]
        self.assertIn('prefetch_adapter "$relative" || continue', pull)

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
