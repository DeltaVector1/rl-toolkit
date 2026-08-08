"""Offline tests for the decensor reward and deterministic grading controls."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ENV_ROOT / "decensor_v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Loading decensor_v1 normally imports the online verifiers stack.  These tests
# intentionally exercise the pure scoring modules without network dependencies.
pkg = types.ModuleType("decensor_v1")
pkg.__path__ = [str(PKG_ROOT)]
sys.modules.setdefault("decensor_v1", pkg)
text = _load("decensor_v1.text", PKG_ROOT / "text.py")
scoring = _load("decensor_v1.scoring", PKG_ROOT / "scoring.py")


def assistant(content: str, reasoning: str | None = "reasoning"):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return message


class ScoringTests(unittest.TestCase):
    def test_truncation_is_a_named_hard_gate(self):
        reward, metrics = scoring.score_decensor(
            [{"role": "user", "content": "question"}],
            [assistant("complete answer")],
            scoring.Verdicts(refusal=1),
            bucket="plaintext",
            word_requirement="",
            is_truncated=True,
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(metrics["early_termination"], "truncated")

    def test_every_assistant_turn_must_have_reasoning(self):
        reward, metrics = scoring.score_decensor(
            [{"role": "user", "content": "question"}],
            [
                assistant("first answer", reasoning=None),
                {"role": "user", "content": "follow-up"},
                assistant("second answer"),
            ],
            scoring.Verdicts(refusal=1),
            bucket="plaintext",
            word_requirement="",
            is_truncated=False,
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(metrics["early_termination"], "no_reasoning")

    def test_length_miss_keeps_the_documented_floor(self):
        reward, metrics = scoring.score_decensor(
            [{"role": "user", "content": "question"}],
            [assistant("short answer")],
            scoring.Verdicts(
                refusal=1,
                coherency=1,
                reasoning_ethics=0,
                reasoning_coherency=1,
                meta_commentary=0,
                md_presence=0,
            ),
            bucket="plaintext",
            word_requirement="Be verbose",
            word_target=2000,
            is_truncated=False,
        )
        self.assertAlmostEqual(metrics["word_multiplier"], text.WORD_MULTIPLIER_FLOOR)
        self.assertGreater(reward, 0.0)

    def test_fenced_answer_is_not_misclassified_as_empty(self):
        reward, metrics = scoring.score_decensor(
            [{"role": "user", "content": "write the file"}],
            [assistant("```python\\nprint('complete')\\n```")],
            scoring.Verdicts(
                refusal=1,
                coherency=1,
                reasoning_ethics=0,
                reasoning_coherency=1,
                meta_commentary=0,
                md_presence=1,
            ),
            bucket="markdown",
            word_requirement="",
            is_truncated=False,
        )
        self.assertIsNone(metrics["early_termination"])
        self.assertGreater(reward, 0.0)


class DeterminismTests(unittest.TestCase):
    def test_judge_temperature_is_zero(self):
        tree = ast.parse((PKG_ROOT / "judges.py").read_text())
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DecensorJudgeSampling"
        )
        temperature = next(
            node for node in cls.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "temperature"
        )
        self.assertEqual(ast.literal_eval(temperature.value), 0.0)

    def test_user_simulator_request_has_a_seed(self):
        tree = ast.parse((PKG_ROOT / "user.py").read_text())
        create_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create"
        ]
        self.assertTrue(create_calls)
        self.assertTrue(any(any(kw.arg == "seed" for kw in call.keywords) for call in create_calls))
        source = (PKG_ROOT / "user.py").read_text()
        self.assertIn("for attempt in range(3)", source)

    def test_policy_timeout_allows_long_context_completion(self):
        source = (PKG_ROOT / "user.py").read_text()
        self.assertIn("_rollout_http = httpx.AsyncClient", source)
        self.assertIn("httpx.Timeout(1800.0, connect=30.0)", source)
        launch = source.split("async def _shared_null_launch", 1)[1].split(
            "\\n\\ndef _enable_shared_clients", 1
        )[0]
        self.assertIn("await _rollout_http.post", launch)

    def test_user_simulator_forbids_safety_process_pivots(self):
        source = (PKG_ROOT / "user.py").read_text().lower()
        self.assertIn("never ask about the assistant's policies", source)
        methodology_line = next(
            line for line in source.splitlines() if line.strip().startswith('"methodology"')
        )
        self.assertIn("original harmful goal", methodology_line)

    def test_refusal_judge_grades_every_assistant_turn(self):
        judge_source = (PKG_ROOT / "judges.py").read_text().lower()
        task_source = (PKG_ROOT / "taskset.py").read_text()
        self.assertIn("inspect every assistant turn", judge_source)
        self.assertIn('"response": conversation_text', task_source)
        self.assertIn("reasoning_trace =", task_source)
        self.assertIn("join(reasoning_parts)", task_source)

    def test_response_gates_run_before_remote_judges(self):
        source = (PKG_ROOT / "taskset.py").read_text()
        self.assertLess(source.index("_, precheck = score_decensor"), source.index("asyncio.gather"))
        self.assertIn('"user_sim_failed"', source)

    def test_judges_retry_before_failure_defaults(self):
        source = (PKG_ROOT / "taskset.py").read_text()
        verdict = source.split("    async def _verdict", 1)[1].split(
            "\\n    @vf.reward", 1
        )[0]
        self.assertIn("for attempt in range(3)", verdict)
        self.assertIn("result.parsed is not None", verdict)


if __name__ == "__main__":
    unittest.main()
