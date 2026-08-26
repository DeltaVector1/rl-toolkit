"""Validation tests for qualification-only LoRA export steps."""

import pytest

from prime_rl.configs.trainer import validate_adapter_export_steps


def test_adapter_export_steps_are_ordered_and_bounded() -> None:
    assert validate_adapter_export_steps([1, 2, 3], 20) == (1, 2, 3)
    assert validate_adapter_export_steps([], None) == ()


@pytest.mark.parametrize(
    ("steps", "max_steps", "message"),
    [
        ([0], 20, "positive"),
        ([-1], 20, "positive"),
        ([1, 1], 20, "unique"),
        ([2, 1], 20, "increasing"),
        ([21], 20, "within max_steps"),
        ([1], None, "finite max_steps"),
    ],
)
def test_adapter_export_steps_fail_explicitly(
    steps: list[int], max_steps: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_adapter_export_steps(steps, max_steps)
