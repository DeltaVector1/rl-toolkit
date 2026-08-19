import importlib.util
import re
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


def _load_filesystem_module():
    torch = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    torch_nn.Module = type("Module", (), {})
    torch.nn = torch_nn

    loader = types.ModuleType("vllm.model_executor.model_loader")
    loader.DefaultModelLoader = type("DefaultModelLoader", (), {})
    loader.get_model_loader = lambda _config: None
    weight_transfer = types.ModuleType(
        "prime_rl.inference.vllm.worker.weight_transfer"
    )
    weight_transfer.load_weights_checkpoint_layerwise = lambda *_args, **_kwargs: None

    stubs = {
        "torch": torch,
        "torch.nn": torch_nn,
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.model_loader": loader,
        "prime_rl": types.ModuleType("prime_rl"),
        "prime_rl.inference": types.ModuleType("prime_rl.inference"),
        "prime_rl.inference.vllm": types.ModuleType("prime_rl.inference.vllm"),
        "prime_rl.inference.vllm.worker": types.ModuleType(
            "prime_rl.inference.vllm.worker"
        ),
        "prime_rl.inference.vllm.worker.weight_transfer": weight_transfer,
    }
    path = (
        Path(__file__).parents[1]
        / "src/prime_rl/inference/vllm/worker/filesystem.py"
    )
    spec = importlib.util.spec_from_file_location("parallel_adapter_download", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    ranges = []

    def log_message(self, *_args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_GET(self):
        value = self.headers.get("Range")
        if value is None:
            start, end = 0, len(self.payload) - 1
            status = 200
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", value)
            if match is None:
                self.send_error(416)
                return
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else len(self.payload) - 1
            status = 206
            type(self).ranges.append((start, end))
        body = self.payload[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(self.payload)}"
            )
        self.end_headers()
        self.wfile.write(body)


class ParallelAdapterDownloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_filesystem_module()
        _RangeHandler.payload = bytes(range(251)) * 20000
        _RangeHandler.ranges = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/adapter"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_parallel_ranges_assemble_exact_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "adapter_model.safetensors"
            with (
                patch.object(self.module, "_PARALLEL_DOWNLOAD_THRESHOLD", 1),
                patch.object(self.module, "_PARALLEL_DOWNLOAD_STREAMS", 4),
            ):
                self.module.download(self.url, destination)

            self.assertEqual(destination.read_bytes(), _RangeHandler.payload)
            self.assertFalse(
                destination.with_name(f"{destination.name}.parts").exists()
            )
            self.assertGreaterEqual(len(_RangeHandler.ranges), 4)

    def test_one_range_resumes_from_existing_part(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "part"
            destination.write_bytes(_RangeHandler.payload[1000:1123])
            self.module._download_range(
                self.url,
                destination,
                1000,
                4999,
            )
            self.assertEqual(destination.read_bytes(), _RangeHandler.payload[1000:5000])

    def test_stage_marks_complete_cache_and_reuses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            calls = []

            def fake_download(url, destination):
                calls.append(url)
                destination.write_bytes(url.encode())

            with (
                patch.object(self.module, "adapter_cache_path", return_value=cache),
                patch.object(self.module, "download", side_effect=fake_download),
            ):
                first = self.module.stage_lora_adapter_files(self.url)
                second = self.module.stage_lora_adapter_files(self.url)

            self.assertEqual(first, cache)
            self.assertEqual(second, cache)
            self.assertTrue((cache / "READY").exists())
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
