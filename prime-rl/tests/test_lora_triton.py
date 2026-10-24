import ast
import unittest
from pathlib import Path


_SOURCE = (
    Path(__file__).parents[1]
    / "src/prime_rl/trainer/models/layers/lora/multi_linear.py"
)


class LoraTritonTest(unittest.TestCase):
    def test_lora_mm_is_outside_torch_compile(self):
        tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_lora_mm"
        ]

        self.assertEqual(len(functions), 1)
        self.assertEqual(
            [ast.unparse(decorator) for decorator in functions[0].decorator_list],
            ["torch.compiler.disable"],
        )


if __name__ == "__main__":
    unittest.main()
