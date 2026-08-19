import ast
import types
import unittest
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "prime_rl"
    / "trainer"
    / "models"
    / "layers"
    / "fp8_ada.py"
)
SOURCE = SOURCE_PATH.read_text()


class FP8QuantizeMemoryTests(unittest.TestCase):
    def test_grouped_mm_quantization_retries_once_after_cache_release(self):
        tree = ast.parse(SOURCE)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_quantize_grouped_mm_input"
        )
        calls = []

        class FakeOOM(Exception):
            pass

        def quantize(*args):
            calls.append(args)
            if len(calls) == 1:
                raise FakeOOM
            return "quantized"

        cache_clears = []
        fake_torch = types.SimpleNamespace(
            float8_e4m3fn="fp8",
            OutOfMemoryError=FakeOOM,
            cuda=types.SimpleNamespace(empty_cache=lambda: cache_clears.append(True)),
        )
        namespace = {
            "torch": fake_torch,
            "block_shape": [128, 128],
            "moe_kernel_quantize_input": quantize,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"),
            namespace,
        )
        result = namespace["_quantize_grouped_mm_input"]("input")
        self.assertEqual(result, "quantized")
        self.assertEqual(cache_clears, [True])
        self.assertEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
