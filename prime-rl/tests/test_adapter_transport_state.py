import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src/prime_rl/inference/vllm/worker/filesystem.py"
TRAINER_BROADCAST = ROOT / "src/prime_rl/trainer/rl/broadcast/filesystem.py"
CLIENT = ROOT / "src/prime_rl/utils/client.py"
ENDPOINT = ROOT.parent / ".agents/skills/training-ops/runtime/h200/muse_file_endpoint.py"


class AdapterTransportStateTest(unittest.TestCase):
    def test_worker_has_manifest_progress_and_reader_lease(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn("_ADAPTER_MANIFEST", source)
        self.assertIn("_ADAPTER_STATUS", source)
        self.assertIn("_ADAPTER_LANE_TIMEOUT_S", source)
        self.assertIn("_ADAPTER_LANE_RETRIES", source)
        self.assertIn("adapter_reader_lock", source)
        self.assertIn("os.replace(temporary, destination)", source)
        tree = ast.parse(source)
        stage = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "stage_lora_adapter_files")
        stage_source = ast.get_source_segment(source, stage)
        assert stage_source is not None
        self.assertNotIn("shutil.rmtree(destination, ignore_errors=True)", stage_source)

    def test_source_manifest_is_written_before_stable_marker(self):
        source = TRAINER_BROADCAST.read_text(encoding="utf-8")
        self.assertIn('"adapter_manifest.json"', source)
        self.assertLess(source.index("manifest_temporary.replace"), source.index("self._notify_orchestrator"))

    def test_shared_host_staging_uses_one_request(self):
        tree = ast.parse(CLIENT.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "stage_lora_adapter")
        source = ast.get_source_segment(CLIENT.read_text(encoding="utf-8"), function)
        assert source is not None
        self.assertIn("admin_clients[0]", source)
        self.assertNotIn("asyncio.gather", source)

    def test_muse_endpoint_publishes_durable_status(self):
        source = ENDPOINT.read_text(encoding="utf-8")
        self.assertIn('app.get("/staging_status")', source)
        self.assertIn("adapter_stage_status", source)
        self.assertIn("adapter_reader_lock", source)


if __name__ == "__main__":
    unittest.main()
