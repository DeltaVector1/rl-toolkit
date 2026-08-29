import ast
from pathlib import Path
import unittest


CLIENT_PATH = Path(__file__).resolve().parents[1] / "src/prime_rl/utils/client.py"
SYNC_PATH = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/training-ops/runtime/h200/sync_rollout_batches.sh"
)


class AdapterStageTimeoutTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
