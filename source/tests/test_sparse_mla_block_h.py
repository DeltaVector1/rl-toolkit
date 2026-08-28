import ast
import os
from pathlib import Path
from unittest import mock


SOURCE_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "prime_rl"
    / "trainer"
    / "models"
    / "kernels"
    / "sparse_mla_bwd.py"
)
SOURCE = SOURCE_PATH.read_text()


def _load_selector():
    tree = ast.parse(SOURCE)
    profiles = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_SM89_BWD_QUALIFICATION_PROFILES"
            for target in node.targets
        )
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sm89_bwd_block_heads"
    )
    module = ast.Module(body=[profiles, function], type_ignores=[])
    namespace = {"os": os, "frozenset": frozenset}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_sm89_bwd_block_heads"]


def test_sm89_backward_geometry_defaults_to_16_and_rejects_unapproved_values():
    select = _load_selector()
    with mock.patch.dict(os.environ, {}, clear=True):
        assert select() == 16
    with mock.patch.dict(
        os.environ, {"PRIME_SPARSE_BWD_BLOCK_HEADS": "32"}, clear=True
    ):
        try:
            select()
        except ValueError as error:
            assert "qualification" in str(error)
        else:
            raise AssertionError("unapproved block_H=32 was accepted")


def test_sm89_backward_geometry_requires_an_allowlisted_qualification_profile():
    select = _load_selector()
    with mock.patch.dict(
        os.environ, {"PRIME_SPARSE_BWD_USE_BLOCK_H32": "1"}, clear=True
    ):
        try:
            select()
        except RuntimeError as error:
            assert "qualification-only" in str(error)
        else:
            raise AssertionError("block_H=32 was enabled without qualification")

    with mock.patch.dict(
        os.environ,
        {
            "PRIME_SPARSE_BWD_USE_BLOCK_H32": "1",
            "PRIME_QUALIFICATION_PROFILE": "glm52-native-b2048-memory",
        },
        clear=True,
    ):
        assert select() == 32


def test_sm89_backward_geometry_rejects_malformed_or_mismatched_settings():
    select = _load_selector()
    for values, error_type in (
        ({"PRIME_SPARSE_BWD_USE_BLOCK_H32": "yes"}, ValueError),
        ({"PRIME_SPARSE_BWD_USE_BLOCK_H32": "0", "PRIME_SPARSE_BWD_BLOCK_HEADS": "8"}, ValueError),
        (
            {
                "PRIME_SPARSE_BWD_USE_BLOCK_H32": "1",
                "PRIME_QUALIFICATION_PROFILE": "glm52-native-b2048-memory",
                "PRIME_SPARSE_BWD_BLOCK_HEADS": "16",
            },
            ValueError,
        ),
    ):
        with mock.patch.dict(os.environ, values, clear=True):
            try:
                select()
            except error_type:
                pass
            else:
                raise AssertionError(f"invalid settings were accepted: {values}")


def test_backward_launch_uses_only_the_selected_geometry_on_sm89():
    tree = ast.parse(SOURCE)
    backward = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sparse_mla_bwd"
    )
    calls = [
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bwd"
    ]
    assert len(calls) == 2
    sm89_call = next(
        call
        for call in calls
        if any(
            isinstance(keyword.value, ast.Name)
            and keyword.value.id == "block_heads"
            for keyword in call.keywords
        )
    )
    assert any(
        isinstance(keyword.value, ast.Name) and keyword.value.id == "block_heads"
        for keyword in sm89_call.keywords
    )
    assert "_sm89_bwd_block_heads()" in SOURCE
    assert "PRIME_SPARSE_BWD_USE_BLOCK_H32" in SOURCE
    assert "T.atomic_add" in SOURCE
    assert "T.atomic_addx4" in SOURCE


def test_invalid_geometry_is_rejected_before_kernel_compilation():
    tree = ast.parse(SOURCE)
    backward = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sparse_mla_bwd"
    )
    selector_index = next(
        index
        for index, node in enumerate(backward.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "is_sm89"
            for target in node.targets
        )
    )
    is_sm89 = backward.body[selector_index + 1]
    assert isinstance(is_sm89, ast.If)
    block_selection = next(
        statement
        for statement in is_sm89.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "block_heads"
            for target in statement.targets
        )
    )
    preprocess_call = next(
        index
        for index, statement in enumerate(backward.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "preprocess"
    )
    assert backward.body.index(is_sm89) < preprocess_call
    assert is_sm89.body.index(block_selection) >= 0
