import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch


model_path = Path(__file__).parents[1] / "src/prime_rl/trainer/model.py"


class _FakeFSDPModule:
    pass


def _load_helpers():
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    wanted = {"_shared_expert_prefetch_enabled", "_append_shared_expert_prefetch"}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"os": os, "FSDPModule": _FakeFSDPModule}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(model_path), "exec"), namespace)
    return namespace


class SharedExpertPrefetchTest(unittest.TestCase):
    def test_flag_and_both_prefetch_directions_are_optional_and_idempotent(self):
        namespace = _load_helpers()
        enabled = namespace["_shared_expert_prefetch_enabled"]
        append = namespace["_append_shared_expert_prefetch"]

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("prime_shared_expert_prefetch", None)
            self.assertFalse(enabled())
            os.environ["prime_shared_expert_prefetch"] = "1"
            self.assertTrue(enabled())
            os.environ["prime_shared_expert_prefetch"] = "0"
            self.assertFalse(enabled())
            os.environ["prime_shared_expert_prefetch"] = "invalid"
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                enabled()

        block = object()
        shared = _FakeFSDPModule()
        modules = [block]
        append(modules, shared, enabled=True)
        append(modules, shared, enabled=True)
        self.assertEqual(modules, [block, shared])
        disabled = [block]
        append(disabled, shared, enabled=False)
        append(disabled, object(), enabled=True)
        self.assertEqual(disabled, [block])

        tree = ast.parse(model_path.read_text(encoding="utf-8"))
        setup = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_fsdp")
        calls = [
            node
            for node in ast.walk(setup)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_append_shared_expert_prefetch"
        ]
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
