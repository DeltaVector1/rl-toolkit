"""Guards for the strict MTP stability profile in the trainer launcher."""

import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_cluster_accepts_only_the_exact_mtp_stability_profile_and_path():
    launcher = (ROOT / "scripts/china/cluster.sh").read_text()
    assert "glm52-native-b2048|glm52-native-stability20" in launcher
    assert "glm52-mtp-stability20)" in launcher
    assert (
        '"$deployment/configs/qualification/glm52-mtp-stability20.toml"'
        in launcher
    )
    assert "MTP stability qualification trainer config path differs" in launcher


def test_cluster_mounts_model_identity_for_lora_and_it_has_no_local_imports():
    launcher = (ROOT / "scripts/china/cluster.sh").read_text()
    for path in (
        "source/src/prime_rl/model_identity.py",
        "source/src/prime_rl/trainer/models/kernels/fp8_indexer.py",
        "source/src/prime_rl/trainer/models/layers/rotary_emb.py",
    ):
        target = "/app/src/" + path.removeprefix("source/src/")
        assert f'-v "$deployment/{path}":{target}:ro' in launcher

    for path in (
        "source/src/prime_rl/model_identity.py",
        "source/src/prime_rl/trainer/models/kernels/fp8_indexer.py",
        "source/src/prime_rl/trainer/models/layers/rotary_emb.py",
    ):
        module = ast.parse((ROOT / path).read_text())
        assert not any(
            (isinstance(node, ast.Import) and any(
                alias.name.startswith("prime_rl") for alias in node.names
            ))
            or (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("prime_rl")
            )
            for node in ast.walk(module)
        ), path
