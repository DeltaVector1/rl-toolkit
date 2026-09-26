import json
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "training_ops/runtime/h200/start_inference.sh"


def settings_python() -> str:
    source = LAUNCHER.read_text()
    start = source.index("runtime_output=$(python3 -")
    start = source.index("<<'PY'\n", start) + len("<<'PY'\n")
    end = source.index("\nPY\n)", start)
    return source[start:end]


class H200InferenceMemoryTest(unittest.TestCase):
    def nodes_config(self, directory: Path) -> Path:
        path = directory / "nodes.toml"
        path.write_text(
            """
[adapter_transport]
download_streams = 8

[paths]
adapter_cache = "/data/cache"

[ports]
lora_source_first = 18101
lora_source_last = 18104

[inference.n85]
disable_nccl_shm = false

[inference.n85.trinity]
tensor_parallel_size = 8
data_parallel_size = 1
gpu_count = 8
mtp_tokens = 0
"""
        )
        return path

    def run_settings(self, nodes: Path, inference: Path):
        return subprocess.run(
            [sys.executable, "-", str(nodes), "n85", str(inference)],
            input=settings_python(),
            text=True,
            capture_output=True,
            env={"ENGINE_KIND": "trinity"},
            check=False,
        )

    def test_selected_toml_memory_value_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = self.nodes_config(root)
            inference = root / "inference.toml"
            inference.write_text("gpu_memory_utilization = 0.85\n")
            result = self.run_settings(nodes, inference)
        self.assertEqual(result.returncode, 0, result.stderr)
        settings = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(settings["gpu_memory_utilization"], 0.85)
        self.assertEqual(settings["kv_cache_dtype"], "bfloat16")

    def test_invalid_memory_value_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes = self.nodes_config(root)
            inference = root / "inference.toml"
            inference.write_text("gpu_memory_utilization = 1.0\n")
            result = self.run_settings(nodes, inference)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("strictly between 0 and 1", result.stderr)

    def test_qualifier_config_only_changes_memory_value(self):
        with (ROOT / "configs/trinity-decensor-experimental-parity/inference.toml").open("rb") as stream:
            current = tomllib.load(stream)
        with (ROOT / "configs/trinity-h200-parity-production/inference.toml").open("rb") as stream:
            qualifier = tomllib.load(stream)
        self.assertEqual(qualifier["gpu_memory_utilization"], 0.85)
        current["gpu_memory_utilization"] = qualifier["gpu_memory_utilization"]
        self.assertEqual(qualifier, current)

    def test_launcher_has_no_fixed_memory_value(self):
        source = LAUNCHER.read_text()
        self.assertNotIn("0.920", source)
        self.assertRegex(source, re.compile(r'"gpu_memory_utilization": float\(gpu_memory_utilization\)'))

    def test_launcher_mounts_and_applies_prompt_logprob_patch_after_flashmla(self):
        source = LAUNCHER.read_text()
        mount = '-v "$deployment/prime-rl/deps/vllm/prompt_logprobs_rows.patch":/tmp/prompt_logprobs_rows.patch:ro'
        self.assertIn(mount, source)
        flash_index = source.index("flashmla_sparse_mtp.patch")
        prompt_index = source.index("prompt_logprobs_rows.patch")
        self.assertLess(flash_index, prompt_index)
        self.assertIn(
            "patch --batch --forward --fuzz=0 -p1 -d /app/.venv/lib/python3.12/site-packages < /tmp/prompt_logprobs_rows.patch",
            source,
        )
        self.assertIn("grep -Fxc", source)
        self.assertIn("# prime-rl prompt logprob row chunking: vllm-0.26", source)


if __name__ == "__main__":
    unittest.main()
