import ast
import asyncio
import random
import unittest
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


root = Path(__file__).resolve().parents[1]


def load_train_source():
    source_path = root / "src/prime_rl/orchestrator/train_source.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    train_source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrainSource")
    namespace = {
        "Iterator": Iterator,
        "random": random,
        "vf": SimpleNamespace(Task=object),
        "TrainEnvs": object,
        "get_logger": lambda: SimpleNamespace(info=lambda _message: None),
    }
    exec(compile(ast.Module(body=[train_source], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["TrainSource"]


def load_dispatcher_method(name, namespace):
    source_path = root / "src/prime_rl/orchestrator/dispatcher.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    dispatcher = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutDispatcher")
    method = next(node for node in dispatcher.body if getattr(node, "name", None) == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[name]


def load_next_fresh_group():
    class GroupState:
        def __init__(self, **values):
            self.__dict__.update(values)

    return load_dispatcher_method("next_fresh_group", {
        "GroupState": GroupState,
        "RolloutKind": str,
        "uuid": SimpleNamespace(uuid4=lambda: "group-id"),
    })


def load_try_schedule():
    return load_dispatcher_method("try_schedule", {"RolloutKind": str})


def make_env(name: str, ratio: float = 1.0):
    return SimpleNamespace(
        name=name,
        tasks=[SimpleNamespace(data=SimpleNamespace(idx=0))],
        num_tasks=1,
        config=SimpleNamespace(ratio=ratio, group_size=1),
    )


def stages(*values: tuple[int, dict[str, float]]):
    return [SimpleNamespace(start_step=start, ratios=ratios) for start, ratios in values]


class RatioStagesTest(unittest.TestCase):
    def test_stage_boundaries_and_resume_step(self):
        ratio_stages = stages(
            (1, {"easy": 8.0, "medium": 2.0, "hard": 0.5}),
            (11, {"easy": 4.0, "medium": 5.0, "hard": 1.0}),
            (21, {"easy": 1.0, "medium": 2.0, "hard": 5.0}),
        )
        source = load_train_source()(
            [make_env("easy"), make_env("medium"), make_env("hard")],
            seed=7,
            ratio_stages=ratio_stages,
        )

        self.assertEqual(source.weights_for_step(1), [8.0, 2.0, 0.5])
        self.assertEqual(source.weights_for_step(10), [8.0, 2.0, 0.5])
        self.assertEqual(source.weights_for_step(11), [4.0, 5.0, 1.0])
        self.assertEqual(source.weights_for_step(21), [1.0, 2.0, 5.0])
        self.assertEqual(source.weights_for_step(31), [1.0, 2.0, 5.0])

    def test_static_ratios_do_not_need_a_step(self):
        source = load_train_source()(
            [make_env("easy", ratio=3.0), make_env("hard", ratio=1.0)],
            seed=7,
        )
        source.rng.choices = Mock(return_value=["easy"])

        self.assertEqual(source.weights_for_step(None), [3.0, 1.0])
        self.assertEqual(source.weights_for_step(99), [3.0, 1.0])
        self.assertEqual(source.next_example(1)["env_name"], "easy")
        source.rng.choices.assert_called_once_with(["easy", "hard"], weights=[3.0, 1.0], k=1)

    def test_dispatcher_passes_the_active_progress_step(self):
        seen_steps = []

        class Source:
            def next_example(self, _available_permits, *, step):
                seen_steps.append(step)
                return {"env_name": "easy", "task_idx": 0}

        train_envs = SimpleNamespace(get=lambda _name: SimpleNamespace(config=SimpleNamespace(group_size=1)))
        dispatcher = SimpleNamespace(
            train_source=Source(),
            eval_source=None,
            available_permits=4,
            fresh_train_rollout_budget=4,
            policy=SimpleNamespace(version=3),
            train_step=lambda: 17,
        )

        load_next_fresh_group()(dispatcher, "train", train_envs)
        self.assertEqual(seen_steps, [17])

        eval_next = Mock(return_value={"env_name": "easy", "task_idx": 0})
        eval_source = SimpleNamespace(next_example=eval_next)
        dispatcher.train_source = None
        dispatcher.eval_source = eval_source
        load_next_fresh_group()(dispatcher, "eval", train_envs)
        eval_next.assert_called_once_with(4)

    def test_inflight_group_stays_pinned_at_stage_boundary(self):
        scheduled = []

        async def schedule_group_rollout(group_id, _group):
            scheduled.append(group_id)
            return True

        dispatcher = SimpleNamespace(
            groups={"old": SimpleNamespace(kind="train", env_name="easy", rollouts_to_schedule=1)},
            train_scheduling_disabled=False,
            fresh_train_groups_allowed=SimpleNamespace(is_set=lambda: True),
            policy_group_drain_pending=False,
            available_permits=1,
            train_envs=SimpleNamespace(get=lambda _name: SimpleNamespace(requires_group_scoring=False)),
            next_fresh_group=lambda *_args: (_ for _ in ()).throw(AssertionError("opened a fresh group")),
            schedule_group_rollout=schedule_group_rollout,
        )
        self.assertTrue(asyncio.run(load_try_schedule()(dispatcher, "train")))
        self.assertEqual(scheduled, ["old"])
