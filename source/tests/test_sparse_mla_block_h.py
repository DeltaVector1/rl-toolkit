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
BENCHMARK_PATH = Path(__file__).parent / "manual" / "test_sparse_mla_block_h.py"
BENCHMARK_SOURCE = BENCHMARK_PATH.read_text()


def _benchmark_constants():
    tree = ast.parse(BENCHMARK_SOURCE)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and not any(
            isinstance(child, ast.Name) and child.id == "torch"
            for child in ast.walk(node)
        )
    ]
    namespace = {"frozenset": frozenset}
    exec(
        compile(
            ast.Module(body=assignments, type_ignores=[]), str(BENCHMARK_PATH), "exec"
        ),
        namespace,
    )
    return namespace


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
        (
            {
                "PRIME_SPARSE_BWD_USE_BLOCK_H32": "0",
                "PRIME_SPARSE_BWD_BLOCK_HEADS": "8",
            },
            ValueError,
        ),
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
            isinstance(keyword.value, ast.Name) and keyword.value.id == "block_heads"
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


def test_benchmark_has_fixed_production_dimensions_and_bounded_work():
    benchmark = _benchmark_constants()
    assert benchmark["sequence_lengths"] == (32, 128)
    assert benchmark["backward_head_tiles"] == (16, 32, 64)
    assert benchmark["batch_size"] == 1
    assert benchmark["attention_heads"] == 64
    assert benchmark["selected_keys"] == 2048
    assert benchmark["backward_key_tile"] == 16
    assert benchmark["backward_threads"] == 64
    assert benchmark["warmup_runs"] == 1
    assert benchmark["timed_runs"] == 2
    assert benchmark["relative_tolerance"] == 0.05
    assert benchmark["absolute_scale_fraction"] == 0.01
    assert "this benchmark has no variable command-line options" in BENCHMARK_SOURCE


def test_benchmark_requires_one_sm89_gpu_and_one_rank():
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES")' in BENCHMARK_SOURCE
    assert 'visible.split(",")' in BENCHMARK_SOURCE
    assert '"WORLD_SIZE": (1, 1)' in BENCHMARK_SOURCE
    assert '"LOCAL_WORLD_SIZE": (1, 1)' in BENCHMARK_SOURCE
    assert '"RANK": (0, 0)' in BENCHMARK_SOURCE
    assert '"LOCAL_RANK": (0, 0)' in BENCHMARK_SOURCE
    assert "torch.cuda.device_count() != 1" in BENCHMARK_SOURCE
    assert "torch.cuda.get_device_capability(0) != (8, 9)" in BENCHMARK_SOURCE
    assert "profile not in qualification_profiles" in BENCHMARK_SOURCE


def test_benchmark_rejects_setting_overrides():
    benchmark = _benchmark_constants()
    assert benchmark["expected_optional_settings"] == {
        "PRIME_SPARSE_BWD_BLOCK_SIZE": "16",
        "PRIME_SPARSE_BWD_THREADS": "64",
        "PRIME_SPARSE_FWD_BLOCK_SIZE": "32",
        "PRIME_SPARSE_FWD_BLOCK_HEADS": "32",
        "PRIME_SPARSE_FWD_THREADS": "128",
    }
    assert benchmark["forbidden_settings"] == (
        "PRIME_SPARSE_BWD_BLOCK_HEADS",
        "PRIME_SPARSE_BWD_USE_BLOCK_H32",
    )
    assert "configured != expected" in BENCHMARK_SOURCE
    assert "if name in os.environ" in BENCHMARK_SOURCE
    assert "os.environ[name] = value" in BENCHMARK_SOURCE
    assert "_apply_exact_settings()" in BENCHMARK_SOURCE


def test_benchmark_rejects_zero_and_nonfinite_gradients():
    assert "value.numel() == 0" in BENCHMARK_SOURCE
    assert "torch.isfinite(value).all()" in BENCHMARK_SOURCE
    assert "torch.count_nonzero(value).item()" in BENCHMARK_SOURCE
    assert 'f"{name} is zero"' in BENCHMARK_SOURCE


def test_benchmark_uses_scale_aware_gradient_tolerance():
    assert "reference_scale * absolute_scale_fraction" in BENCHMARK_SOURCE
    assert (
        "absolute_tolerance + relative_tolerance * reference_float.abs()"
        in BENCHMARK_SOURCE
    )
    assert '"max_abs": float(difference.max().item())' in BENCHMARK_SOURCE
    assert '"max_rel": float(relative.max().item())' in BENCHMARK_SOURCE
    assert '"violations": int((difference > allowed).sum().item())' in BENCHMARK_SOURCE
    assert "for block_heads in (32, 64)" in BENCHMARK_SOURCE
    assert "baseline = results[16]" in BENCHMARK_SOURCE
