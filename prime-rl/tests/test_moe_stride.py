import ast
import unittest
from pathlib import Path


_SOURCE = (
    Path(__file__).parents[1]
    / "src/prime_rl/trainer/models/layers/moe.py"
)


class MoeStrideTest(unittest.TestCase):
    def test_combine_rows_are_materialized_before_index_select(self):
        tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "combine_expert_outputs"
        ]

        self.assertEqual(len(functions), 1)
        rows_assignments = [
            node
            for node in ast.walk(functions[0])
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "rows"
            )
        ]

        self.assertEqual(len(rows_assignments), 1)
        self.assertEqual(
            ast.unparse(rows_assignments[0].value),
            "sorted_row_for_token_slot[:, slot].contiguous()",
        )


if __name__ == "__main__":
    unittest.main()
