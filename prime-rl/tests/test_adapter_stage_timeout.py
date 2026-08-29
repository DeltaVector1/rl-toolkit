import ast
import threading
from typing import Any
from pathlib import Path
import unittest


CLIENT_PATH = Path(__file__).resolve().parents[1] / "src/prime_rl/utils/client.py"
SYNC_PATH = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/training-ops/runtime/h200/sync_rollout_batches.sh"
)
SERVER_PATH = Path(__file__).resolve().parents[1] / "src/prime_rl/inference/vllm/server.py"
WATCHDOG_PATH = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/training-ops/runtime/h200/engine_watchdog.sh"
)


class AdapterStageTimeoutTest(unittest.TestCase):
    def test_stage_state_clears_after_overlapping_operations_finish(self):
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"))
        wanted = {"_begin_lora_stage", "_end_lora_stage", "_lora_stage_status"}
        body = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in {"_lora_stage_lock", "_lora_stage_count", "_lora_stage_started_at"}
                for target in node.targets
            )
        ]
        clock = iter((100.0, 102.5, 102.5))

        class FakeTime:
            @staticmethod
            def monotonic():
                return next(clock)

        namespace = {"threading": threading, "time": FakeTime, "Any": Any}
        exec(compile(ast.Module(body=assignments + body, type_ignores=[]), str(SERVER_PATH), "exec"), namespace)

        self.assertEqual(namespace["_lora_stage_status"](), {"status": "idle", "age_seconds": 0.0})
        namespace["_begin_lora_stage"]()
        namespace["_begin_lora_stage"]()
        self.assertEqual(
            namespace["_lora_stage_status"](),
            {"status": "staging", "age_seconds": 2.5},
        )
        namespace["_end_lora_stage"]()
        self.assertEqual(namespace["_lora_stage_status"]()["status"], "staging")
        namespace["_end_lora_stage"]()
        self.assertEqual(namespace["_lora_stage_status"](), {"status": "idle", "age_seconds": 0.0})

    def test_staging_serializes_all_source_urls_on_one_host(self):
        source = (CLIENT_PATH.parent.parent / "inference/vllm/worker/filesystem.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('destination.parent / "adapter-staging.lock"', source)
        self.assertIn("fcntl.flock(host_lock, fcntl.LOCK_EX)", source)

    def test_staging_has_a_longer_explicit_budget_than_loading(self):
        tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
        values = {}
        stage_call = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.endswith("TIMEOUT_S"):
                    values[target.id] = ast.literal_eval(node.value)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage_lora_adapter":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        if child.func.id == "_admin_post":
                            stage_call = {
                                keyword.arg: getattr(keyword.value, "id", None)
                                for keyword in child.keywords
                            }

        self.assertEqual(values["LORA_LOAD_TOTAL_TIMEOUT_S"], 600.0)
        self.assertEqual(values["LORA_STAGE_ATTEMPT_TIMEOUT_S"], 3600.0)
        self.assertEqual(values["LORA_STAGE_TOTAL_TIMEOUT_S"], 7200.0)
        self.assertEqual(stage_call["timeout_s"], "LORA_STAGE_ATTEMPT_TIMEOUT_S")
        self.assertEqual(stage_call["total_timeout_s"], "LORA_STAGE_TOTAL_TIMEOUT_S")

    def test_prefetch_budget_covers_the_same_large_transfer(self):
        source = SYNC_PATH.read_text(encoding="utf-8")
        self.assertIn("adapter_prefetch_timeout=${ADAPTER_PREFETCH_TIMEOUT:-7200}", source)
        self.assertIn("adapter_request_timeout=${ADAPTER_REQUEST_TIMEOUT:-3600}", source)

    def test_server_publishes_adapter_stage_state_for_health_checks(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('@router.get("/staging_status")', source)
        self.assertIn('"status": "staging"', source)
        self.assertIn("_begin_lora_stage()", source)
        self.assertIn("_end_lora_stage()", source)
        self.assertIn("finally:", source)

    def test_watchdog_defers_only_for_bounded_reported_staging(self):
        source = WATCHDOG_PATH.read_text(encoding="utf-8")
        self.assertIn("STAGING_STATUS_TIMEOUT=${STAGING_STATUS_TIMEOUT:-5}", source)
        self.assertIn("STAGING_GRACE=${STAGING_GRACE:-7200}", source)
        self.assertIn("/staging_status", source)
        self.assertIn("adapter staging active", source)
        self.assertIn("age < grace", source)
        self.assertIn("adapter staging exceeded grace", source)

    def test_watchdog_checks_collective_liveness_before_restarting(self):
        source = WATCHDOG_PATH.read_text(encoding="utf-8")
        self.assertIn('LIVENESS_TIMEOUT=${LIVENESS_TIMEOUT:-15}', source)
        self.assertIn('"http://127.0.0.1:$PORT/liveness"', source)
        self.assertIn("engine liveness passed", source)
        self.assertIn("engine liveness failed", source)
        self.assertIn("restart deferred", source)
        self.assertIn("completion-miss restart checks resumed", source)
        self.assertIn("liveness_deferrals=0", source)

    def test_named_watchdog_repair_preserves_active_profile_without_engine_restart(self):
        import sys
        from unittest import mock

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents/skills/training-ops/src"))
        import training_ops

        deployed = []
        ssh_scripts = []

        def fake_ssh(_host, script, timeout=120):
            ssh_scripts.append((script, timeout))
            return 0, "reloaded exact engine watchdog(s) without restarting an engine)\n"

        with mock.patch.object(training_ops, "current_run", return_value="x85"), mock.patch.object(
            training_ops, "deploy", side_effect=lambda *args: deployed.append(args) or True
        ), mock.patch.object(
            training_ops, "_ssh", side_effect=fake_ssh
        ):
            result = training_ops._repair_engine_watchdog("h200a")

        self.assertIn("without restarting an engine", result)
        self.assertEqual(deployed, [(".agents/skills/training-ops/runtime/h200/engine_watchdog.sh", ["h200a"])])
        script = ssh_scripts[0][0]
        self.assertIn("STARTUP_GRACE=180", script)
        self.assertIn("MODEL=/models/glm52-fp8", script)
        self.assertIn("START=/data/glm52-rl/deployment/.agents/skills/training-ops/runtime/h200/start_inference.sh", script)
        self.assertIn("ENGINE_CONTAINER=glm52-decensor-h200", script)
        self.assertIn("PORT=8000", script)
        self.assertIn("ambiguous watchdog identity", script)
        self.assertIn("kill -TERM --", script)
        self.assertNotIn("docker", script)
        self.assertNotIn("pkill", script)


if __name__ == "__main__":
    unittest.main()
