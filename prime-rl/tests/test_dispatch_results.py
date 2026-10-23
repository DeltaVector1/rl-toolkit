import ast
import asyncio
import time
import unittest
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


root = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DispatchFailure:
    kind: str
    env_name: str
    group_id: uuid.UUID
    step: int
    policy_version: int
    task_idx: int
    task_type: str
    task_key: str
    task_hash: str
    count: int
    error: SimpleNamespace


@dataclass(frozen=True)
class GroupCancellation:
    kind: str
    env_name: str
    group_id: uuid.UUID
    step: int
    count: int
    reason: str


class Logger:
    def debug(self, _message):
        pass

    def warning(self, _message):
        pass


def load_class(path: Path, name: str, globals_: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {**globals_, "__name__": "test_loaded_module"}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def load_train_sink():
    return load_class(
        root / "src/prime_rl/orchestrator/train_sink.py",
        "TrainSink",
        {
            "DispatchFailure": DispatchFailure,
            "GroupCancellation": GroupCancellation,
            "MAX_CONSECUTIVE_ERRORED_GROUPS": 8,
            "defaultdict": defaultdict,
            "get_logger": Logger,
            "time": time,
        },
    )


def load_eval_sink():
    class EvalRollouts(list):
        pass

    @dataclass
    class EvalBatch:
        env_name: str
        step: int
        rollouts: list

    return load_class(
        root / "src/prime_rl/orchestrator/eval_sink.py",
        "EvalSink",
        {
            "DispatchFailure": DispatchFailure,
            "GroupCancellation": GroupCancellation,
            "EvalBatch": EvalBatch,
            "EvalRollouts": EvalRollouts,
            "defaultdict": defaultdict,
            "get_logger": Logger,
        },
    )


def load_dispatcher_method(name: str):
    path = root / "src/prime_rl/orchestrator/dispatcher.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dispatcher = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutDispatcher"
    )
    method = next(node for node in dispatcher.body if getattr(node, "name", None) == name)
    namespace = {
        "asyncio": asyncio,
        "traceback": __import__("traceback"),
        "DispatchFailure": DispatchFailure,
        "GroupCancellation": GroupCancellation,
        "InflightRollout": lambda **fields: SimpleNamespace(**fields),
        "vf": SimpleNamespace(Error=lambda **fields: SimpleNamespace(**fields)),
        "get_logger": Logger,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def make_train_sink(train_sink_class, *, group_size=2):
    environment = SimpleNamespace(config=SimpleNamespace(group_size=group_size))
    sink = train_sink_class.__new__(train_sink_class)
    sink.train_envs = SimpleNamespace(get=lambda _name: environment)
    sink.pending_groups = defaultdict(list)
    sink.pending_group_episodes = defaultdict(int)
    sink.pending_group_started_at = {}
    sink.pending_group_failures = defaultdict(list)
    sink.pending_group_cancellations = {}
    sink.abandoned_groups = set()
    sink.pending_rollouts = []
    sink.pending_batch = []
    sink.pending_tokens = 0
    sink.batch_size = 8
    sink.token_batch_size = None
    sink.consecutive_errored_groups = 0
    sink._ready_batch = lambda: None
    sink._record_group = lambda *args, **kwargs: sink.recorded_groups.append((args, kwargs))
    sink.recorded_groups = []

    async def process_rollout(_rollout):
        pass

    sink.process_rollout = process_rollout
    return sink


def make_rollout(group_id, *, env_name="train", idx=0):
    return SimpleNamespace(
        group_id=group_id,
        env_name=env_name,
        task=SimpleNamespace(data=SimpleNamespace(idx=idx)),
        samples=[object()],
        has_error=False,
    )


def failure(kind, group_id, count, *, step=4):
    return DispatchFailure(
        kind=kind,
        env_name=kind,
        group_id=group_id,
        step=step,
        policy_version=3,
        task_idx=7,
        task_type="Task",
        task_key="key",
        task_hash="hash",
        count=count,
        error=SimpleNamespace(type="JudgeError", message="failed"),
    )


def cancellation(kind, group_id, count, *, step=4):
    return GroupCancellation(
        kind=kind,
        env_name=kind,
        group_id=group_id,
        step=step,
        count=count,
        reason="superseded",
    )


class DispatchResultsTest(unittest.TestCase):
    def test_dispatcher_queues_explicit_failure(self):
        class FailedTask:
            def result(self):
                raise RuntimeError("judge failed")

        errors = []
        completed = []
        task = FailedTask()
        group_id = uuid.uuid4()
        group = SimpleNamespace(
            task_idx=7,
            task=SimpleNamespace(key="task-key", hash="task-hash"),
        )
        meta = SimpleNamespace(
            group_id=group_id,
            rollout_count=2,
            kind="train",
            env_name="train",
            eval_step=None,
            policy_version=3,
        )
        dispatcher = SimpleNamespace(
            inflight={task: meta},
            groups={group_id: group},
            release=lambda count: None,
            metrics=SimpleNamespace(
                record_error=lambda **fields: errors.append(fields),
            ),
            _complete_group=lambda *args: completed.append(args),
            out_q=asyncio.Queue(),
        )

        asyncio.run(load_dispatcher_method("handle_completed_rollout")(dispatcher, task))
        result = asyncio.run(dispatcher.out_q.get())

        self.assertIsInstance(result, DispatchFailure)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.task_idx, 7)
        self.assertEqual(result.error.type, "RuntimeError")
        self.assertEqual(errors, [{"kind": "train", "env_name": "train", "n": 2}])
        self.assertEqual(completed[0][2], 2)

    def test_dispatcher_queues_group_cancellation(self):
        completed = []
        group_id = uuid.uuid4()
        group = SimpleNamespace(
            kind="train",
            env_name="train",
            rollouts_to_schedule=2,
            policy_version_at_start=5,
            eval_step=None,
        )
        cancellations = []
        dispatcher = SimpleNamespace(
            groups={group_id: group},
            transient_retries={},
            inflight={},
            metrics=SimpleNamespace(
                record_cancellation=lambda **fields: cancellations.append(fields),
            ),
            _complete_group=lambda *args: completed.append(args),
            out_q=asyncio.Queue(),
        )

        cancelled = asyncio.run(load_dispatcher_method("drop_group")(dispatcher, group_id))
        result = asyncio.run(dispatcher.out_q.get())

        self.assertEqual(cancelled, 2)
        self.assertIsInstance(result, GroupCancellation)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.reason, "superseded")
        self.assertEqual(cancellations, [{"kind": "train", "env_name": "train", "n": 2}])
        self.assertEqual(completed[0][2], 2)

    def test_success_episodes_still_fill_train_group_budget(self):
        train_sink_class = load_train_sink()
        sink = make_train_sink(train_sink_class)
        completed = []

        async def process_group(group_id):
            completed.append(group_id)
            sink.pending_groups.pop(group_id, None)
            sink.pending_group_episodes.pop(group_id, None)
            sink.pending_group_started_at.pop(group_id, None)

        sink.process_group = process_group
        group_id = uuid.uuid4()

        self.assertIsNone(asyncio.run(sink.add([make_rollout(group_id, idx=1)])))
        self.assertEqual(sink.pending_group_episodes[group_id], 1)
        self.assertIsNone(asyncio.run(sink.add([make_rollout(group_id, idx=1)])))
        self.assertEqual(completed, [group_id])

    def test_dispatch_failure_counts_group_members_without_fake_rollouts(self):
        sink = make_train_sink(load_train_sink())
        group_id = uuid.uuid4()

        self.assertIsNone(asyncio.run(sink.fail(failure("train", group_id, 2))))

        self.assertEqual(sink.pending_rollouts, [])
        self.assertNotIn(group_id, sink.pending_group_episodes)
        self.assertEqual(sink.consecutive_errored_groups, 1)
        self.assertEqual(sink.recorded_groups[0][0][2], "dispatch_failure_group")
        self.assertEqual(sink.recorded_groups[0][1]["rejected_rollouts"], 2)

    def test_cancel_completes_partial_group_without_failure_metrics(self):
        sink = make_train_sink(load_train_sink())
        group_id = uuid.uuid4()
        rollout = make_rollout(group_id)

        self.assertIsNone(asyncio.run(sink.add([rollout])))
        self.assertIsNone(asyncio.run(sink.cancel(cancellation("train", group_id, 1))))

        self.assertEqual(sink.pending_rollouts, [rollout])
        self.assertEqual(rollout.samples, [])
        self.assertTrue(rollout.is_filtered)
        self.assertEqual(sink.consecutive_errored_groups, 0)
        self.assertEqual(sink.recorded_groups, [])
        self.assertNotIn(group_id, sink.pending_group_started_at)

    def test_abandoned_group_ignores_late_cancellation(self):
        sink = make_train_sink(load_train_sink())
        group_id = uuid.uuid4()
        sink.abandon_groups({group_id})

        self.assertIsNone(asyncio.run(sink.cancel(cancellation("train", group_id, 2))))
        self.assertNotIn(group_id, sink.pending_group_episodes)
        self.assertEqual(sink.recorded_groups, [])

    def test_eval_success_and_terminal_results_share_group_budget(self):
        eval_sink_class = load_eval_sink()
        env = SimpleNamespace(config=SimpleNamespace(group_size=2), examples=[None])
        eval_sink = eval_sink_class(eval_envs=SimpleNamespace(get=lambda _name: env))
        group_id = uuid.uuid4()
        first = SimpleNamespace(
            group_id=group_id,
            env_name="eval",
            eval_step=9,
            task=SimpleNamespace(data=SimpleNamespace(idx=0)),
        )
        second = SimpleNamespace(
            group_id=group_id,
            env_name="eval",
            eval_step=9,
            task=SimpleNamespace(data=SimpleNamespace(idx=0)),
        )

        self.assertIsNone(eval_sink.add([first]))
        batch = eval_sink.add([second])
        self.assertEqual(batch.step, 9)
        self.assertEqual(list(batch.rollouts), [first, second])

        failed_group = uuid.uuid4()
        self.assertIsNone(eval_sink.fail(failure("eval", failed_group, 1, step=9)))
        batch = eval_sink.cancel(cancellation("eval", failed_group, 1, step=9))
        self.assertEqual(batch.step, 9)
        self.assertEqual(list(batch.rollouts), [])
        self.assertNotIn(("eval", 9), eval_sink.pending_batch_failures)
        self.assertNotIn(("eval", 9), eval_sink.pending_batch_cancelled)


if __name__ == "__main__":
    unittest.main()
