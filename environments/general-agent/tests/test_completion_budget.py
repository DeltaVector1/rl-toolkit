import asyncio
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


env_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(env_root))


class CompletionBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vf = importlib.import_module("verifiers")
        cls.module = importlib.import_module("general_agent.solver.local.env")

    def invoke(self, used, args, limit=16384):
        env = object.__new__(self.module.ToolComposableEnv)
        env.max_total_completion_tokens = limit
        env.get_state_usage = lambda state: None if used is None else {"output_tokens": used}
        seen = []

        async def capture(self, state, prompt, **kwargs):
            seen.append(kwargs["sampling_args"])
            return kwargs["sampling_args"]

        with patch.object(self.vf.ToolEnv, "get_model_response", capture):
            result = asyncio.run(env.get_model_response({"sampling_args": args}, []))
        return result, seen

    def test_clamps_remaining_without_mutating_args(self):
        args = {"max_tokens": 16384, "temperature": 0.2}
        result, seen = self.invoke(16000, args)
        self.assertEqual(result["max_tokens"], 384)
        self.assertEqual(seen, [result])
        self.assertEqual(args["max_tokens"], 16384)

    def test_preserves_smaller_and_completion_key(self):
        result, _ = self.invoke(1000, {"max_tokens": 128})
        self.assertEqual(result["max_tokens"], 128)
        result, _ = self.invoke(16000, {"max_tokens": None})
        self.assertEqual(result["max_tokens"], 384)
        result, _ = self.invoke(16000, {"max_completion_tokens": 16384})
        self.assertEqual(result["max_completion_tokens"], 384)

    def test_first_request_is_capped_and_missing_limit_is_added(self):
        result, _ = self.invoke(None, {})
        self.assertEqual(result["max_tokens"], 16384)

    def test_zero_budget_stops_before_model_request(self):
        env = object.__new__(self.module.ToolComposableEnv)
        env.max_total_completion_tokens = 16384
        env.timeout_seconds = 5
        env.get_state_usage = lambda state: {"output_tokens": 16384}
        calls = []

        class FakeClient:
            async def get_response(self, **kwargs):
                calls.append(kwargs)

        timing = SimpleNamespace(
            generation=SimpleNamespace(start=None, end=None),
            setup=SimpleNamespace(start=None, end=None),
            env=SimpleNamespace(spans=[]),
            model=SimpleNamespace(spans=[]),
        )

        async def init_state(*args):
            return {"timing": timing, "trajectory": [], "sampling_args": {"max_tokens": 16384}}

        async def setup_state(state):
            return None

        async def is_completed(state):
            return await env.max_total_completion_tokens_reached(state)

        async def cleanup(state):
            return None

        env.init_state = init_state
        env.setup_state = setup_state
        env.is_completed = is_completed
        env.cleanup = cleanup
        state = asyncio.run(env.rollout({}, FakeClient(), "model"))
        self.assertEqual(calls, [])
        self.assertNotIn("error", state)

    def test_budget_absent_leaves_arguments_unchanged(self):
        args = {"max_tokens": 128}
        result, _ = self.invoke(None, args, limit=-1)
        self.assertEqual(result, args)
        self.assertIsNot(result, args)


if __name__ == "__main__":
    unittest.main()
