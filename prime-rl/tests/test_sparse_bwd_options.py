import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch


kernel_path = Path(__file__).parents[1] / "src/prime_rl/trainer/models/kernels/sparse_mla_bwd.py"


def load_options():
    tree = ast.parse(kernel_path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sparse_bwd_options"
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(kernel_path), "exec"), namespace)
    return namespace["_sparse_bwd_options"]


class SparseBackwardOptionsTest(unittest.TestCase):
    def test_experimental_modes_default_off_and_partition_count_defaults_to_one(self):
        options = load_options()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(options(), (False, False, 1))

    def test_flags_and_partition_environment_are_checked(self):
        options = load_options()
        with patch.dict(
            os.environ,
            {
                "prime_sparse_bwd_direct_dkv": "1",
                "prime_sparse_bwd_skip_sentinel": "1",
                "prime_sparse_bwd_dkv_partitions": "4",
            },
            clear=True,
        ):
            self.assertEqual(options(), (True, True, 4))
            self.assertEqual(options(2), (True, True, 2))
            with self.assertRaisesRegex(ValueError, "one of 1, 2, or 4"):
                options(3)
            with self.assertRaisesRegex(ValueError, "one of 1, 2, or 4"):
                options(True)
            os.environ["prime_sparse_bwd_direct_dkv"] = "yes"
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                options()

    def test_invalid_partition_environment_is_rejected(self):
        options = load_options()
        with patch.dict(os.environ, {"prime_sparse_bwd_dkv_partitions": "bad"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be 1, 2, or 4"):
                options()


if __name__ == "__main__":
    unittest.main()
