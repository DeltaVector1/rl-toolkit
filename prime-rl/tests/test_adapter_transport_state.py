import ast
import hashlib
import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
MODULE_PATH = ROOT / "src/prime_rl/utils/adapter_delta.py"
SPEC = importlib.util.spec_from_file_location("adapter_delta", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter_delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter_delta)


def function_arguments(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    return [argument.arg for argument in function.args.args]


class AdapterTransportStateTest(unittest.TestCase):
    def test_exact_adjacent_delta_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "step_1"
            target = root / "step_2"
            base.mkdir()
            target.mkdir()
            original = bytes(range(251)) * 33
            changed = bytearray(original)
            changed[0] = 17
            changed[4096] = 29
            for directory, payload in ((base, original), (target, changed)):
                (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
                (directory / "adapter_model.safetensors").write_bytes(payload)

            manifest = adapter_delta.publish_adapter_manifest(
                target,
                base,
                target_url="http://source/step_2",
                base_url="http://source/step_1",
                target_step=2,
                base_step=1,
                chunk_size=1024,
                max_workers=2,
            )

            rebuilt = bytearray()
            for chunk in manifest["chunks"]:
                start = chunk["offset"]
                wire = (target / chunk["file"]).read_bytes()
                rebuilt.extend(
                    adapter_delta.decode_delta_chunk(
                        original[start : start + chunk["raw_size"]], wire, chunk["codec"]
                    )
                )
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["mode"], "delta")
            self.assertEqual(manifest["base_step"], 1)
            self.assertEqual(bytes(rebuilt), bytes(changed))
            self.assertEqual(manifest["target_sha256"], hashlib.sha256(changed).hexdigest())

    def test_bootstrap_does_not_duplicate_the_full_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "step_0"
            target.mkdir()
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")
            (target / "adapter_model.safetensors").write_bytes(bytes(range(251)) * 33)
            manifest = adapter_delta.publish_adapter_manifest(target, target_step=0)
            self.assertEqual(manifest["mode"], "full")
            self.assertEqual(manifest["chunk_count"], 0)
            self.assertEqual(list(target.glob("*.transport-*")), [])

    def test_protocol_has_one_strict_delta_flag(self):
        worker = ROOT / "src/prime_rl/inference/vllm/worker/filesystem.py"
        client = ROOT / "src/prime_rl/utils/client.py"
        server = ROOT / "src/prime_rl/inference/vllm/server.py"
        watcher = ROOT / "src/prime_rl/orchestrator/watcher.py"
        self.assertEqual(
            function_arguments(worker, "stage_lora_adapter_files"),
            ["source_url", "require_delta"],
        )
        self.assertEqual(
            function_arguments(worker, "stage_lora_adapter"),
            ["self", "source_url", "require_delta"],
        )
        for path in (client, server, watcher):
            self.assertIn("require_delta", path.read_text(encoding="utf-8"))
        worker_source = worker.read_text(encoding="utf-8")
        self.assertIn("delta adapter transport is required", worker_source)
        self.assertIn('chunk_url = f"{lane_url.rstrip(\'/\')}/{name}"', worker_source)
        self.assertIn("source_urls=[chunk_url]", worker_source)
        self.assertIn("executor.shutdown(wait=False, cancel_futures=True)", worker_source)
        self.assertIn("deadline=delta_deadline", worker_source)
        self.assertIn("asyncio.wait_for(", server.read_text(encoding="utf-8"))
        watcher_source = watcher.read_text(encoding="utf-8")
        self.assertIn('manifest.get("version") != 1', watcher_source)
        self.assertIn("next_step > 0", watcher_source)

    def test_inventory_defines_bounded_eight_lane_transfer(self):
        operations = REPOSITORY / "configs/operations.toml"
        with operations.open("rb") as stream:
            config = tomllib.load(stream)
        ports = config["ports"]
        self.assertEqual(ports["lora_source_last"] - ports["lora_source_first"] + 1, 8)
        self.assertEqual(config["limits"]["adapter_delta_stage_timeout_seconds"], 300)
        runtime = REPOSITORY / ".agents/skills/training-ops/runtime/h200"
        for name in ("start_inference.sh", "start_muse_inference.sh"):
            source = (runtime / name).read_text(encoding="utf-8")
            self.assertIn('PRIME_LORA_SOURCE_PORTS="$lora_source_port_list"', source)


if __name__ == "__main__":
    unittest.main()
