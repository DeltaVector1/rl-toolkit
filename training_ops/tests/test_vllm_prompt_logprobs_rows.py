import importlib.util
import math
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "prime-rl/deps/vllm/prompt_logprobs_rows.patch"
FIXTURE = ROOT / "training_ops/tests/fixtures/vllm026_prompt_logprobs"
TARGET = FIXTURE / "vllm/v1/worker/gpu_model_runner.py"
MARKER = "# prime-rl prompt logprob row chunking: vllm-0.26"


def apply_patch(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-d", str(root)],
        input=PATCH.read_text(),
        text=True,
        capture_output=True,
        check=False,
    )


def load_fixture(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Index:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _Starts:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return _Index(self.values[index])


class _Model:
    def __init__(self, module, runner):
        self.module = module
        self.runner = runner
        self.calls = []

    def compute_logits(self, hidden):
        self.calls.append((list(hidden.rows), self.runner.lora_mapping))
        return self.module.FakeTensor(hidden.rows, width=65536)


class _Sampler:
    def __init__(self, module):
        self.module = module
        self.compute_logprobs_calls = 0
        self.gather_calls = []

    def compute_logprobs(self, logits):
        self.compute_logprobs_calls += 1
        return logits

    def gather_logprobs(self, scores, count, targets):
        self.gather_calls.append((len(scores), count, list(targets.rows)))
        token_ids = self.module.FakeTensor(targets.rows, width=count + 1)
        logprobs = self.module.FakeTensor(
            [-(row + 0.25) for row in targets.rows], width=count + 1
        )
        ranks = self.module.FakeTensor([row % 17 for row in targets.rows])
        return token_ids, logprobs, ranks, None


def make_runner(module, lengths, mode="logprobs", offsets=None):
    runner = module.GpuModelRunner()
    req_ids = [f"req-{index}" for index in range(len(lengths))]
    offsets = offsets or [sum(lengths[:index]) for index in range(len(lengths))]
    requests = {}
    for req_id, length in zip(req_ids, lengths):
        requests[req_id] = types.SimpleNamespace(
            prompt_token_ids=[100000 + value for value in range(length)],
            num_computed_tokens=0,
            in_progress_prompt_logprobs_cpu=None,
        )
    runner.num_prompt_logprobs = {req_id: 3 for req_id in req_ids}
    runner.requests = requests
    runner.device = "fake"
    runner.input_batch = types.SimpleNamespace(
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)}
    )
    runner.query_start_loc = types.SimpleNamespace(np=_Starts(offsets))
    runner.model_config = types.SimpleNamespace(logprobs_mode=mode)
    runner.lora_mapping = "adapter-a"
    runner.model = _Model(module, runner)
    runner.sampler = _Sampler(module)
    runner.sync_calls = 0
    runner._sync_device = lambda: setattr(runner, "sync_calls", runner.sync_calls + 1)
    hidden_rows = sum(lengths)
    return runner, module.FakeTensor(list(range(hidden_rows))), req_ids


def freeze(result):
    frozen = {}
    for req_id, tensors in result.items():
        frozen[req_id] = (
            list(tensors.logprob_token_ids.rows),
            list(tensors.logprobs.rows),
            list(tensors.selected_token_ranks.rows),
            tensors.logprob_token_ids.width,
            tensors.logprobs.width,
        )
    return frozen


class PromptLogprobsRowsTest(unittest.TestCase):
    def patched_module(self, root):
        result = apply_patch(root)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return load_fixture(
            root / "vllm/v1/worker/gpu_model_runner.py", "patched_prompt_logprobs"
        )

    def test_patch_applies_to_representative_vllm026_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(FIXTURE, root)
            result = apply_patch(root)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            source = (root / "vllm/v1/worker/gpu_model_runner.py").read_text()
            self.assertEqual(source.count(MARKER), 1)

    def test_second_apply_is_not_silently_accepted_without_unique_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(FIXTURE, root)
            first = apply_patch(root)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            second = apply_patch(root)
            self.assertNotEqual(second.returncode, 0)
            source = (root / "vllm/v1/worker/gpu_model_runner.py").read_text()
            self.assertEqual(source.count(MARKER), 1)

    def test_context_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(FIXTURE, root)
            target = root / "vllm/v1/worker/gpu_model_runner.py"
            source = target.read_text().replace(
                "prompt_hidden_states = hidden_states[offset : offset + num_logits]",
                "prompt_hidden_states = hidden_states[offset : offset + num_logits + 1]",
                1,
            )
            target.write_text(source)
            result = apply_patch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(MARKER, target.read_text())

    def test_chunked_output_matches_oracle_for_long_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(FIXTURE, root)
            patched = self.patched_module(root)
        original = load_fixture(TARGET, "original_prompt_logprobs")
        for length in (512, 974, 8192, 11324, 32768):
            oracle_runner, hidden, req_ids = make_runner(original, [length])
            actual_runner, actual_hidden, actual_ids = make_runner(patched, [length])
            expected = oracle_runner._get_prompt_logprobs_dict(
                hidden, {req_ids[0]: length}
            )
            actual = actual_runner._get_prompt_logprobs_dict(
                actual_hidden, {actual_ids[0]: length}
            )
            self.assertEqual(freeze(actual), freeze(expected), length)
            self.assertEqual(actual_runner.model.calls[0][0], list(range(128)))
            self.assertEqual(
                len(actual_runner.model.calls), math.ceil((length - 1) / 128)
            )
            self.assertLessEqual(max(len(rows) for rows, _ in actual_runner.model.calls), 128)
            expected_tail = (length - 1) % 128 or 128
            self.assertEqual(len(actual_runner.model.calls[-1][0]), expected_tail)
            self.assertEqual(actual_runner.sync_calls, 1)

    def test_offsets_scheduler_accumulation_modes_lora_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(FIXTURE, root)
            patched = self.patched_module(root)

        runner, hidden, req_ids = make_runner(patched, [300, 260], offsets=[0, 300])
        result = runner._get_prompt_logprobs_dict(
            hidden, {req_ids[0]: 300, req_ids[1]: 260}
        )
        self.assertEqual(set(result), set(req_ids))
        self.assertEqual(runner.model.calls[0][0], list(range(128)))
        self.assertEqual(runner.model.calls[3][0], list(range(300, 428)))
        self.assertTrue(all(mapping == "adapter-a" for _, mapping in runner.model.calls))
        self.assertEqual(result[req_ids[0]].logprob_token_ids.width, 4)
        self.assertEqual(result[req_ids[0]].selected_token_ranks.width, 1)

        for mode in ("raw_logits", "processed_logits"):
            mode_runner, mode_hidden, mode_ids = make_runner(patched, [257], mode=mode)
            mode_result = mode_runner._get_prompt_logprobs_dict(
                mode_hidden, {mode_ids[0]: 257}
            )
            self.assertEqual(mode_runner.sampler.compute_logprobs_calls, 0)
            self.assertEqual(len(mode_result[mode_ids[0]].logprobs.rows), 256)

        scheduled_runner, scheduled_hidden, scheduled_ids = make_runner(patched, [300])
        scheduled_request = scheduled_runner.requests[scheduled_ids[0]]
        self.assertEqual(
            scheduled_runner._get_prompt_logprobs_dict(
                scheduled_hidden, {scheduled_ids[0]: 128}
            ),
            {},
        )
        scheduled_request.num_computed_tokens = 128
        self.assertEqual(
            scheduled_runner._get_prompt_logprobs_dict(
                scheduled_hidden, {scheduled_ids[0]: 128}
            ),
            {},
        )
        scheduled_request.num_computed_tokens = 256
        completed = scheduled_runner._get_prompt_logprobs_dict(
            scheduled_hidden, {scheduled_ids[0]: 128}
        )
        self.assertEqual(len(completed[scheduled_ids[0]].logprobs.rows), 299)
        self.assertEqual([len(rows) for rows, _ in scheduled_runner.model.calls], [128, 128, 43])

    def test_no_prompt_logprobs_makes_no_output_head_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixture"
            shutil.copytree(FIXTURE, root)
            patched = self.patched_module(root)
        runner, hidden, req_ids = make_runner(patched, [8192])
        runner.num_prompt_logprobs = {}
        self.assertEqual(runner._get_prompt_logprobs_dict(hidden, {req_ids[0]: 8192}), {})
        self.assertEqual(runner.model.calls, [])


if __name__ == "__main__":
    unittest.main()
