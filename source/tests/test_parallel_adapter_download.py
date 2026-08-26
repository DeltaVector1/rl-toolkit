import importlib.util
import os
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
    parity_trace = types.ModuleType("prime_rl.parity_router_trace")
    parity_trace.install_vllm_parity_trace = lambda: None

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
        "prime_rl.parity_router_trace": parity_trace,
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


def _load_adapter_server_module():
    path = Path(__file__).parents[2] / "scripts/china/adapter_server.py"
    spec = importlib.util.spec_from_file_location("adapter_server", path)
    module = importlib.util.module_from_spec(spec)
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
                patch.dict(
                    os.environ,
                    {
                        "PRIME_ADAPTER_PORTS": "",
                        "PRIME_GLM_ADAPTER_GZIP": "0",
                    },
                ),
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
                [self.url],
                0,
                destination,
                1000,
                4999,
            )
            self.assertEqual(destination.read_bytes(), _RangeHandler.payload[1000:5000])

    def test_glm_gzip_ranges_are_required_and_create_only(self):
        adapter_server = _load_adapter_server_module()

        class Handler(adapter_server.RangeHandler):
            gzip_requests = 0

            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.headers.get("Accept-Encoding") == "gzip":
                    type(self).gzip_requests += 1
                return super().do_GET()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            source = root / "runs/adapter"
            source.write_bytes(_RangeHandler.payload)
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                lambda *args, **kwargs: Handler(
                    *args, directory=str(root), **kwargs
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            destination = root / "downloaded.safetensors"
            url = f"http://127.0.0.1:{server.server_port}/adapter"
            environment = {
                "PRIME_GLM_ADAPTER_GZIP": "1",
                "PRIME_ADAPTER_PORTS": ",".join([str(server.server_port)] * 4),
            }
            try:
                with (
                    patch.dict(os.environ, environment),
                    patch.object(self.module, "_PARALLEL_DOWNLOAD_THRESHOLD", 1),
                ):
                    self.module.download(url, destination, create_only=True)
                    with self.assertRaises(FileExistsError):
                        self.module.download(url, destination, create_only=True)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
            self.assertEqual(destination.read_bytes(), _RangeHandler.payload)
            self.assertGreaterEqual(Handler.gzip_requests, 4)
            rejected = root / "rejected.safetensors"
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRIME_GLM_ADAPTER_GZIP": "1",
                        "PRIME_ADAPTER_PORTS": "",
                    },
                ),
                patch.object(self.module, "_PARALLEL_DOWNLOAD_THRESHOLD", 1),
                patch.object(self.module.time, "sleep"),
                self.assertRaisesRegex(
                    OSError, "did not return required gzip"
                ),
            ):
                self.module.download(self.url, rejected)


if __name__ == "__main__":
    unittest.main()
