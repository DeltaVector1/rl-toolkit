import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import training_ops
import training_ops.tests  # Install the local-only test guard before any test runs.


TELEMETRY = (
    "complete_groups=43 clean_groups=26 rejected_groups=17 "
    "accepted_buffer=624 accepted_target=1024 accepted_unit=rollouts "
    "partial_groups=16 partial_rollouts=161"
)
TIMESTAMP = "2026-09-08T22:00:01.000000000Z"


def _ansi_orchestrator_log():
    return (
        f"\x1b[2m{TIMESTAMP}\x1b[0m "
        "\x1b[32mINFO\x1b[0m "
        f"\x1b[36mPIPELINE_TELEMETRY {TELEMETRY}\x1b[0m; "
        "batch 4/1024; 2 inflight; SUCCESS Step 4"
    )


class OrchestratorProbeTelemetryTest(unittest.TestCase):
    def run_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sudo").write_text(
                "#!/bin/sh\nexec \"$@\"\n", encoding="utf-8"
            )
            (root / "docker").write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *State.Running*) printf '/orch true running\\n' ;;\n"
                "  *logs*) printf '%s\\n' \"$ORCH_LOG\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            (root / "curl").write_text(
                "#!/bin/sh\nprintf 000\n", encoding="utf-8"
            )
            (root / "ss").write_text(
                "#!/bin/sh\nexit 0\n", encoding="utf-8"
            )
            for path in (root / "sudo", root / "docker", root / "curl", root / "ss"):
                path.chmod(0o755)
            script = (
                training_ops.orchestrator_probe
                .replace("__ORCH__", "orch")
                .replace("__JUDGE_URL__", "http://judge")
                .replace("__SECONDARY_JUDGE_URL__", "http://secondary")
                .replace("__SECONDARY_JUDGE_ENABLED__", "0")
                .replace("__RUN__", "test-run")
                .replace("__RUN_NAME__", "test-run")
                .replace("__RUN_DIR__", "/tmp/test-run")
                .replace("__ROLLOUT_TRANSPORT__", "filesystem")
                .replace("__ROLLOUT_TRANSFER_DIRECTION__", "trainer_pull")
                .replace("__BATCH_SYNC_SCRIPT__", "/tmp/batch-sync")
                .replace("__TRAINER_WATCHDOG_SCRIPT__", "/tmp/watchdog")
                .replace("__TRAINER_ADDRESSES__", "n82")
                .replace("__TRAINER_CT__", "trainer")
                .replace("__TRAINER_SSH__", "trainer")
                .replace("__ZMQ_PORT__", "0")
                .replace("__ZMQ_TARGET__", "127.0.0.1:1")
            )
            environment = {
                "PATH": f"{directory}:/usr/bin:/bin",
                "ORCH_LOG": _ansi_orchestrator_log(),
            }
            result = subprocess.run(
                ["bash", "-c", script],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return training_ops._kv(result.stdout)

    def test_extracts_ansi_telemetry_and_docker_timestamp(self):
        report = self.run_probe()

        self.assertEqual(report["PIPELINE_TELEMETRY"], TELEMETRY)
        self.assertEqual(report["PIPELINE_TELEMETRY_AGE"], TIMESTAMP)
        self.assertEqual(report["ORCH_LOGAGE"], TIMESTAMP)
        self.assertEqual(report["BATCH"], "batch 4/1024; 2 inflight")
        self.assertNotIn("prompt", report["PIPELINE_TELEMETRY"].lower())
        self.assertNotIn("provider", report["PIPELINE_TELEMETRY"].lower())


class StatusTelemetryHostRoutingTest(unittest.TestCase):
    def test_status_reads_orchestrator_fields_from_configured_edge_host(self):
        calls = []
        head_output = (
            "MODE rl\n"
            "TRAINER_UP up\n"
            "TRAINER_STEPS 0\n"
            "TRAINER_COMPUTE_STATE pending\n"
            "TRAINER_FAILED 0\n"
            f"TRAINER_LOGAGE {TIMESTAMP}\n"
        )
        edge_output = (
            "ORCH_UP up\n"
            "ORCH_FAILED 0\n"
            "ORCH_LAST SUCCESS Step 4\n"
            "BATCH batch 4/1024; 2 inflight\n"
            f"PIPELINE_TELEMETRY {TELEMETRY}\n"
            f"PIPELINE_TELEMETRY_AGE {TIMESTAMP}\n"
            f"ORCH_LOGAGE {TIMESTAMP}\n"
            "FAILS5M 0\n"
            "UNSCORED5M 0\n"
            "JUDGE 200\n"
            "SECONDARY_JUDGE unused\n"
            "ZMQ_TUNNEL 1\n"
        )

        def fake_ssh(host, script, timeout=120):
            calls.append((host, script, timeout))
            if host == training_ops.trainer_head:
                return 0, head_output
            if host == training_ops.orchestrator_host:
                if "PIPELINE_TELEMETRY " in script:
                    return 0, edge_output
                return 0, edge_output.replace(f"PIPELINE_TELEMETRY {TELEMETRY}\n", "")
            raise AssertionError(f"unexpected status host: {host}")

        with (
            mock.patch.object(training_ops, "_config_for_run", return_value=None),
            mock.patch.object(training_ops, "_trainer_mode", return_value="rl"),
            mock.patch.object(training_ops, "_active_policy_probe_config", return_value=None),
            mock.patch.object(
                training_ops,
                "_status_runtime_identity",
                return_value=("test-run", "filesystem", "/tmp/test-run", 0, ""),
            ),
            mock.patch.object(training_ops, "_active_orchestrator_uses_url", return_value=False),
            mock.patch.object(training_ops, "_trainer_addresses", return_value="n82"),
            mock.patch.object(training_ops, "_fan", return_value={}),
            mock.patch.object(training_ops, "_ssh", side_effect=fake_ssh),
        ):
            status = training_ops.status("test-run", quiet=True)

        self.assertTrue(status.prebatch.available)
        self.assertEqual(status.prebatch.accepted_buffer, 624)
        self.assertEqual(status.orch_last, "SUCCESS Step 4")
        self.assertEqual(status.batch, "batch 4/1024; 2 inflight")
        self.assertAlmostEqual(
            status.orch_log_age, training_ops._age(TIMESTAMP), delta=1
        )
        self.assertEqual(
            [host for host, _, _ in calls],
            [training_ops.trainer_head, training_ops.orchestrator_host],
        )


if __name__ == "__main__":
    unittest.main()
