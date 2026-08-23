import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace


source_root = Path(os.environ["VLLM_V026_SOURCE_ROOT"])
source_path = source_root / "vllm/v1/attention/backends/mla/flashmla_sparse.py"
source_tree = ast.parse(source_path.read_text())


def top_level_node(name: str):
    for node in source_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node
    raise AssertionError(f"missing {name}")


helper_namespace = {}
helper_nodes = [
    top_level_node("fp8_sparse_scratch_bytes_per_token"),
    top_level_node("fp8_mixed_batch_chunk_size"),
]
exec(
    compile(ast.Module(helper_nodes, type_ignores=[]), str(source_path), "exec"),
    helper_namespace,
)
helper_namespace["fp8_mixed_batch_chunk_tokens"] = helper_namespace[
    "fp8_mixed_batch_chunk_size"
](512)
slice_node = top_level_node("fp8_mixed_batch_chunk_slices")
exec(
    compile(ast.Module([slice_node], type_ignores=[]), str(source_path), "exec"),
    helper_namespace,
)

chunk_size = helper_namespace["fp8_mixed_batch_chunk_size"]
chunk_slices = helper_namespace["fp8_mixed_batch_chunk_slices"]


class FakeTensor:
    def __init__(self, rows, width=576):
        self.rows = list(rows)
        self.width = width

    @property
    def shape(self):
        return (len(self.rows), 1, self.width)

    def __getitem__(self, item):
        return FakeTensor(self.rows[item], self.width)

    def __setitem__(self, item, value):
        self.rows[item] = value.rows

    def new_empty(self, shape):
        return FakeTensor([None] * shape[0], shape[-1])

    def unsqueeze(self, dimension):
        if dimension != 0:
            raise AssertionError(f"unexpected dimension {dimension}")
        return FakeBatch(self)


class FakeBatch:
    def __init__(self, tensor):
        self.tensor = tensor

    def squeeze(self, dimension):
        if dimension != 0:
            raise AssertionError(f"unexpected dimension {dimension}")
        return self.tensor


def identity_index_conversion(_request_ids, _block_table, indices, **_kwargs):
    return indices


def load_mixed_forward():
    for node in source_tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FlashMLASparseImpl":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "_forward_fp8_kv_mixed_batch"
                ):
                    namespace = {
                        "triton_convert_req_index_to_global_index": identity_index_conversion
                    }
                    code = "from __future__ import annotations\n" + ast.unparse(item)
                    exec(compile(code, str(source_path), "exec"), namespace)
                    return namespace[item.name]
    raise AssertionError("missing _forward_fp8_kv_mixed_batch")


mixed_forward = load_mixed_forward()


class FakeImpl:
    def _fp8_flash_mla_kernel(
        self, q, kv_c_and_k_pe_cache, topk_indices, kernel_metadata
    ):
        del kv_c_and_k_pe_cache, topk_indices, kernel_metadata
        output = FakeTensor((1000 + row for row in q.tensor.rows), 512)
        return FakeBatch(output), None


def run_forward(rows, slices):
    metadata = SimpleNamespace(
        fp8_extra_metadata=[(token_slice, object()) for token_slice in slices],
        req_id_per_token=[0] * len(rows),
        block_table=None,
        block_size=64,
    )
    return mixed_forward(
        FakeImpl(), FakeTensor(rows), object(), FakeTensor(rows, 64), metadata
    ).rows


class FlashMLASparseChunkTests(unittest.TestCase):
    def test_chunk_size_math(self):
        self.assertEqual(chunk_size(1), 4)
        self.assertEqual(chunk_size(512), 2048)
        self.assertEqual(chunk_size(1024), 4096)

    def test_full_coverage_and_no_overlap(self):
        for token_count in (0, 1, 2048, 2049, 32768, 32769):
            cursor = 0
            covered = []
            for token_slice in chunk_slices(token_count):
                self.assertEqual(token_slice.start, cursor)
                self.assertGreater(token_slice.stop, token_slice.start)
                self.assertLessEqual(token_slice.stop - token_slice.start, 2048)
                covered.extend(range(token_slice.start, token_slice.stop))
                cursor = token_slice.stop
            self.assertEqual(cursor, token_count)
            self.assertEqual(covered, list(range(token_count)))

    def test_output_assembly_order(self):
        rows = list(range(5001))
        self.assertEqual(
            run_forward(rows, chunk_slices(len(rows))),
            [1000 + row for row in rows],
        )

    def test_single_chunk_equivalence(self):
        rows = list(range(128))
        self.assertEqual(
            run_forward(rows, chunk_slices(len(rows))),
            [1000 + row for row in rows],
        )

    def test_invalid_limit_rejection(self):
        for invalid_limit in (0, -1, -512):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                chunk_size(invalid_limit)
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            chunk_slices(-1)


if __name__ == "__main__":
    unittest.main()
