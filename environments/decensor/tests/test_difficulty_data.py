import sys
import tomllib
import types
import unittest
from pathlib import Path
from unittest.mock import patch


root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
try:
    import verifiers.v1  # noqa: F401
except ModuleNotFoundError:
    verifiers = types.ModuleType("verifiers")
    verifiers.__path__ = []
    verifiers.v1 = types.ModuleType("verifiers.v1")
    sys.modules["verifiers"] = verifiers
    sys.modules["verifiers.v1"] = verifiers.v1

try:
    import datasets  # noqa: F401
except ModuleNotFoundError:
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **_kwargs: None
    sys.modules["datasets"] = datasets

from decensor_v1 import data


class DifficultyDataTest(unittest.TestCase):
    def test_difficulty_rows_filter_the_requested_bucket(self):
        class Dataset:
            def __init__(self):
                self.columns = None

            def select_columns(self, columns):
                self.columns = columns
                return self

            def __iter__(self):
                return iter(
                    [
                        {"bucket_ID": "easy", "source_index": 10},
                        {"bucket_ID": "hard", "source_index": 11},
                    ]
                )

        dataset = Dataset()
        with patch.object(data, "load_dataset", return_value=dataset) as load:
            rows = list(data.iter_difficulty_decensor_rows("hard"))

        self.assertEqual([row["source_index"] for row in rows], [11])
        self.assertEqual(
            dataset.columns,
            ["source_index", "prompt", "primary_harm_type_code", "bucket_ID"],
        )
        load.assert_called_once_with(
            data.difficulty_decensor_dataset,
            split="train",
            streaming=True,
        )

    def test_invalid_difficulty_fails_before_loading_data(self):
        with patch.object(data, "load_dataset") as load:
            with self.assertRaisesRegex(ValueError, "easy, medium, or hard"):
                list(data.iter_difficulty_decensor_rows("unknown"))
        load.assert_not_called()

    def test_difficulty_pool_builds_multiturn_tasks_from_fixture_rows(self):
        rows = [
            {
                "source_index": index,
                "prompt": [{"role": "user", "content": f"Prompt {index}"}],
                "primary_harm_type_code": "harm-a" if index < 2 else "harm-b",
            }
            for index in range(4)
        ]
        with patch.object(data, "iter_difficulty_decensor_rows", return_value=rows):
            with patch.object(data, "sample_turn_counts", return_value=[1, 2, 1, 2]):
                pool = data.build_decensor_pool(4, 0.5, 2, 23, difficulty="medium")

        self.assertEqual(len(pool), 4)
        self.assertEqual({task[2] for task in pool}, {"harm-a", "harm-b"})
        self.assertEqual({task[3] for task in pool}, {0, 1, 2, 3})
        self.assertEqual({task[1] for task in pool}, {1, 2})
        self.assertTrue(
            all(
                messages[0]["role"] == "user"
                and messages[0]["content"].startswith("Prompt ")
                for messages, _, _, _ in pool
            )
        )

    def test_default_pool_keeps_existing_multiturn_dataset_path(self):
        rows = [
            {
                "conversations": [
                    {"from": "human", "value": "First request"},
                    {"from": "gpt", "value": "First answer"},
                    {"from": "human", "value": "Second request"},
                    {"from": "gpt", "value": "Second answer"},
                ],
                "harm_types": ["harm-a"],
            }
        ]
        with patch.object(data, "iter_decensor_rows", return_value=rows):
            with patch.object(data, "sample_turn_counts", return_value=[2]):
                with patch.object(data, "build_difficulty_decensor_pool") as difficulty_pool:
                    pool = data.build_decensor_pool(1, 0.5, 2, 23)

        self.assertEqual(
            pool[0][0],
            [
                {"role": "user", "content": "First request"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second request"},
            ],
        )
        self.assertEqual(pool[0][1], 2)
        difficulty_pool.assert_not_called()

    def test_difficulty_profile_keeps_equal_environment_ratios(self):
        config_path = root.parent.parent / "configs/decensor-difficulty/orchestrator.toml"
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)

        envs = config["train"]["env"]
        self.assertEqual([env["name"] for env in envs], ["decensor-easy", "decensor-medium", "decensor-hard"])
        self.assertEqual([env["ratio"] for env in envs], [1.0, 1.0, 1.0])
        self.assertNotIn("ratio_stages", config["train"])


if __name__ == "__main__":
    unittest.main()
