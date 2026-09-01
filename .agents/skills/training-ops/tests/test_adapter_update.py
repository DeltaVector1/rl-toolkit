import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import training_ops


class AdapterUpdateTest(unittest.TestCase):
    def test_config_uses_the_active_glm_profile(self):
        config = training_ops._adapter_update_config("glm-1")
        self.assertEqual(config["kind"], "glm52")
        self.assertEqual(config["engine_container"], "glm52-decensor-h200")
        self.assertEqual(config["ports"], [8000])
        self.assertEqual(config["run_marker"], "/glm-1/")
        self.assertEqual(config["service_marker"], "/run_glm-1/")

    def test_glm_adapter_server_gets_predictive_upgrade_settings(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "china"
            / "cluster.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("ADAPTER_PREDICTIVE_UPGRADE", launcher)
        self.assertIn("ADAPTER_PREDICTIVE_RUN", launcher)
        self.assertIn("ADAPTER_PREDICTIVE_SOURCE_URL", launcher)
        self.assertIn("/broadcast/\"$predictive_run\":rw", launcher)
        self.assertIn("prime_rl/utils/adapter_delta.py", launcher)

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

    def test_engine_state_requires_exact_identity_and_reports_restart_need(self):
        script = training_ops._adapter_engine_state_script(
            "glm52-decensor-h200",
            [8000],
            "/glm52/",
            "/run_glm52/",
            "/data/glm52-rl/runs/glm52/run_glm52",
        )
        self.assertIn('name=$(sudo docker inspect', script)
        self.assertIn('configured inference container identity is not exact', script)
        self.assertIn('staging_process_matches', script)
        self.assertIn('adapter staging remains', script)
        self.assertIn('exit 10', script)

    def test_glm_restart_wait_is_configured_for_model_load(self):
        script = training_ops._restart_adapter_engine_script(
            "glm52-decensor-h200", 8000
        )
        self.assertIn("seq 1 900", script)
        with self.assertRaisesRegex(training_ops.Refused, "integer >= 600"):
            training_ops._restart_adapter_engine_script(
                "glm52-decensor-h200", 8000, startup_timeout_seconds=180
            )

    def test_orchestrator_state_script_requires_exact_stopped_container(self):
        script = training_ops._orchestrator_state_script(
            "glm52-decensor-orchestrator"
        )
        self.assertIn("docker inspect", script)
        self.assertIn('configured orchestrator container identity is not exact', script)
        self.assertIn("false) echo stopped", script)
        self.assertNotIn("docker start", script)
        self.assertNotIn("docker restart", script)

    def test_stopped_orchestrator_is_started_after_guarded_repair(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1
                ),
            ],
        )
        fan_results = iter(
            [
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
                {"h200a": (0, "stopped 0"), "h200b": (0, "stopped 0")},
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
            ]
        )
        ssh_scripts = []

        def ssh(_host, script, timeout=120):
            ssh_scripts.append(script)
            if "configured orchestrator container state is unknown" in script:
                return 0, "stopped\n"
            return 0, ""

        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(
            training_ops, "_adapter_stage_evidence", return_value="Failed to apply policy update 4"
        ), mock.patch.object(
            training_ops, "_fan", side_effect=lambda *_args, **_kwargs: next(fan_results)
        ), mock.patch.object(training_ops, "deploy", return_value=True), mock.patch.object(
            training_ops, "_ssh", side_effect=ssh
        ), mock.patch.object(
            training_ops, "_ssh_address", side_effect=lambda host: f"addr-{host}"
        ):
            result = training_ops._repair_adapter_update()

        self.assertIn("recreated exact orchestrator", result)
        resume_script = ssh_scripts[-1]
        self.assertIn('sudo docker rm -f "$container"', resume_script)
        self.assertIn(
            "bash /data/glm52-rl/deployment/.agents/skills/training-ops/runtime/h200/start_orchestrator.sh",
            resume_script,
        )
        self.assertIn("RUN=glm-1", resume_script)
        self.assertIn("RUN_NAME=run_glm-1", resume_script)
        self.assertIn(
            "ORCH_CONFIG=/data/glm52-rl/deployment/configs/paused/glm52/orchestrator.toml",
            resume_script,
        )
        self.assertIn("TRAINER_CT=glm52-decensor-trainer", resume_script)
        self.assertIn("ORCH_CT=glm52-decensor-orchestrator", resume_script)
        self.assertNotIn("sudo docker start", resume_script)
        self.assertNotIn("sudo docker restart", resume_script)

    def test_stopped_status_refuses_if_container_is_running(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                )
            ],
        )
        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "_ssh", return_value=(0, "running\n")), mock.patch.object(
            training_ops, "_adapter_stage_evidence"
        ) as evidence, mock.patch.object(training_ops, "deploy") as deploy:
            with self.assertRaisesRegex(training_ops.Refused, "stopped exact orchestrator"):
                training_ops._repair_adapter_update()
        evidence.assert_not_called()
        deploy.assert_not_called()

    def test_repair_refuses_while_trainer_is_busy(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=False,
            nodes=[training_ops.NodeState(host="n82", gpu_util=80)],
        )
        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(training_ops, "_ssh") as ssh, mock.patch.object(
            training_ops, "_adapter_stage_evidence"
        ) as evidence:
            with self.assertRaisesRegex(training_ops.Refused, "trainer to be waiting"):
                training_ops._repair_adapter_update()
        ssh.assert_not_called()
        evidence.assert_not_called()

    def test_engine_repair_uses_active_run_profile(self):
        calls = []

        def ssh(_host, script, timeout=120):
            calls.append(script)
            return 0, "issued"

        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "_ssh", side_effect=ssh
        ):
            result = training_ops._repair_engine("h200a")

        self.assertEqual(result, "issued")
        self.assertTrue(any("glm52-decensor-h200" in call for call in calls))
        self.assertTrue(any("start_inference.sh" in call for call in calls))
        self.assertTrue(any("ENGINE_CONTAINER=glm52-decensor-h200" in call for call in calls))
        self.assertFalse(any("muse-glimmer-h200" in call for call in calls))
        self.assertFalse(any("start_muse_inference.sh" in call for call in calls))

    def test_healthy_engine_skips_a_second_restart(self):
        before = training_ops.Status(
            trainer_up=True,
            orch_up=True,
            engines=[
                training_ops.EngineState(
                    host="h200a", gen_code="200", probes_ok=1, probes_total=1
                ),
                training_ops.EngineState(
                    host="h200b", gen_code="200", probes_ok=1, probes_total=1
                ),
            ],
        )
        fan_results = iter(
            [
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
                {"h200a": (0, "stopped 0"), "h200b": (0, "stopped 0")},
                {
                    "h200a": (0, "adapter engine ready"),
                    "h200b": (0, "adapter engine ready"),
                },
                {"h200a": (0, "idle"), "h200b": (0, "idle")},
            ]
        )
        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
            training_ops, "status", return_value=before
        ), mock.patch.object(
            training_ops, "_adapter_stage_evidence", return_value="Failed to apply policy update 4"
        ), mock.patch.object(
            training_ops, "_fan", side_effect=lambda *_args, **_kwargs: next(fan_results)
        ), mock.patch.object(training_ops, "deploy", return_value=True) as deploy, mock.patch.object(
            training_ops, "_ssh", return_value=(0, "")
        ), mock.patch.object(
            training_ops, "_restart_adapter_engine_script"
        ) as restart_script:
            result = training_ops._repair_adapter_update()

        self.assertIn("kept healthy inference", result)
        restart_script.assert_not_called()
        self.assertEqual(deploy.call_count, 3)

    def test_prefetch_script_only_signals_exact_curl_or_timeout(self):
        script = training_ops._adapter_prefetch_stop_script(
            "/glm-1/", "/run_glm-1/", "/data/glm52-rl/runs/glm-1/run_glm-1"
        )
        self.assertIn("case \"${exe##*/}\" in curl|timeout", script)
        self.assertIn("stage_lora_adapter", script)
        self.assertIn("SOURCE_RUN_DIR=", script)
        self.assertNotIn("pkill", script)
        self.assertNotIn("kill -- -", script)

    def test_repair_refuses_before_any_mutation_when_trainer_is_down(self):
        unhealthy = training_ops.Status(trainer_up=False, orch_up=True)
        with mock.patch.object(training_ops, "current_run", return_value="glm-1"), mock.patch.object(
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
