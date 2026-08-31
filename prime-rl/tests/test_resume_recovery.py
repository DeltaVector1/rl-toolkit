import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "src/prime_rl/utils/client.py"
ORCHESTRATOR_PATH = ROOT / "src/prime_rl/orchestrator/orchestrator.py"
SERVER_PATH = ROOT / "src/prime_rl/inference/vllm/server.py"


class ResumeRecoveryTest(unittest.TestCase):
    def test_resume_has_a_longer_separate_budget(self):
        tree = ast.parse(CLIENT_PATH.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("RESUME_"):
                    values[target.id] = ast.literal_eval(node.value)

        self.assertEqual(values["RESUME_ATTEMPT_TIMEOUT_S"], 900.0)
        self.assertEqual(values["RESUME_TOTAL_TIMEOUT_S"], 1800.0)
        self.assertEqual(values["RESUME_CONFIRM_TIMEOUT_S"], 120.0)
        self.assertGreater(values["RESUME_ATTEMPT_TIMEOUT_S"], 300.0)

    def test_resume_timeout_is_confirmed_before_failure(self):
        source = CLIENT_PATH.read_text(encoding="utf-8")
        resume = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "_resume_engines"
            ),
        )
        self.assertIn('"/resume_status"', resume)
        self.assertIn('"/liveness"', resume)
        self.assertIn('state == "running"', resume)
        self.assertIn("timed out, but", resume)
        self.assertIn("was not confirmed", resume)
        self.assertIn("route_missing and await _is_live(client)", resume)
        self.assertIn("except (TimeoutError, httpx.TimeoutException, httpx.TransportError)", resume)

    def test_server_exposes_explicit_resume_state(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn('@router.get("/resume_status")', source)
        self.assertIn('_set_engine_state("resuming")', source)
        self.assertIn('_set_engine_state("running")', source)
        self.assertIn('_set_engine_state("unknown")', source)

    def test_startup_stages_once_per_host_before_loading(self):
        source = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
        setup = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
            ),
        )
        stage_at = setup.index("await stage_lora_adapter(")
        update_at = setup.index("await self.policy_inference.update_weights(", stage_at)
        self.assertLess(stage_at, update_at)
        self.assertIn("prefetched=prefetched", setup[update_at:])

    def test_server_fallback_stages_in_the_host_process(self):
        source = SERVER_PATH.read_text(encoding="utf-8")
        load = ast.get_source_segment(
            source,
            next(
                node
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "load_lora_adapter"
            ),
        )
        self.assertIn("await asyncio.to_thread(", load)
        self.assertIn("stage_lora_adapter_files", load)
        self.assertNotIn('collective_rpc("stage_lora_adapter"', load)


if __name__ == "__main__":
    unittest.main()
