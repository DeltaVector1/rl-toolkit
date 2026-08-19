"""Regression checks for LoRA recovery after an inference-engine restart."""

from pathlib import Path
import unittest


WATCHER = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/orchestrator/watcher.py"
).read_text()
CLIENT = (
    Path(__file__).resolve().parents[1]
    / "src/prime_rl/utils/client.py"
).read_text()


class WatcherReconcileTest(unittest.TestCase):
    def test_idle_watcher_reconciles_current_policy(self):
        loop = WATCHER.split("    async def start", 1)[1].split("\n    async def stop", 1)[0]
        self.assertIn("await self.reconcile_current_policy()", loop)
        self.assertIn("self.reconcile_interval", loop)

    def test_reconcile_checks_every_admin_endpoint_and_reloads_missing_ones(self):
        method = WATCHER.split("    async def reconcile_current_policy", 1)[1].split(
            "\n    async def wait_for_model_express_status", 1
        )[0]
        self.assertIn("self.inference.admin_clients", method)
        self.assertIn("await load_lora_adapter(missing", method)
        self.assertIn("if not all(visible)", method)

    def test_model_visibility_probe_has_a_bounded_timeout(self):
        method = CLIENT.split("async def model_is_visible", 1)[1].split("\n\nasync def ", 1)[0]
        self.assertIn('admin_client.get(', method)
        self.assertIn("read=30.0", method)


if __name__ == "__main__":
    unittest.main()
