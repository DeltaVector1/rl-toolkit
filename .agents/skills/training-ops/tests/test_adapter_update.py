import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class AdapterUpdateTest(unittest.TestCase):
    def test_config_uses_the_active_glm_profile(self):
        config = training_ops._adapter_update_config("x85")
        self.assertEqual(config["kind"], "glm52")
        self.assertEqual(config["engine_container"], "glm52-decensor-h200")
        self.assertEqual(config["ports"], [8000])
        self.assertEqual(config["run_marker"], "/x85/")
        self.assertEqual(config["service_marker"], "/run_x85/")

    def test_stage_log_requires_a_transfer_timeout(self):
        good = "Failed to apply policy update 4\nReadTimeout /stage_lora_adapter"
        self.assertEqual(
            training_ops._validate_adapter_stage_log(good),
            "Failed to apply policy update 4",
        )
        with self.assertRaises(training_ops.Refused):
            training_ops._validate_adapter_stage_log(
                "Failed to apply policy update 4\nadapter request failed"
            )

    def test_stage_success_after_failure_refuses_repair(self):
        log = (
            "Failed to apply policy update 4\nReadTimeout stage_lora_adapter\n"
            "Staged policy v4 on every inference host"
        )
        with self.assertRaisesRegex(training_ops.Refused, "already succeeded"):
            training_ops._validate_adapter_stage_log(log)

    def test_idle_script_requires_health_and_request_metric(self):
        script = training_ops._adapter_idle_script([8000])
        self.assertIn("/health", script)
        self.assertIn("/metrics", script)
        self.assertIn("vllm:num_requests_running", script)
        self.assertIn("!found || bad || total != 0", script)

    def test_prefetch_script_only_signals_exact_curl_or_timeout(self):
        script = training_ops._adapter_prefetch_stop_script(
            "/x85/", "/run_x85/", "/data/glm52-rl/runs/x85/run_x85"
        )
        self.assertIn("case \"${exe##*/}\" in curl|timeout", script)
        self.assertIn("stage_lora_adapter", script)
        self.assertIn("SOURCE_RUN_DIR=", script)
        self.assertNotIn("pkill", script)
        self.assertNotIn("kill -- -", script)

    def test_repair_refuses_before_any_mutation_when_trainer_is_down(self):
        unhealthy = training_ops.Status(trainer_up=False, orch_up=True)
        with mock.patch.object(training_ops, "current_run", return_value="x85"), mock.patch.object(
            training_ops, "status", return_value=unhealthy
        ), mock.patch.object(training_ops, "_adapter_stage_evidence") as evidence, mock.patch.object(
            training_ops, "deploy"
        ) as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "healthy trainer"):
                training_ops._repair_adapter_update()
        evidence.assert_not_called()
        deploy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
