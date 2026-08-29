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


def test_cluster_mounts_the_complete_trainer_import_closure():
    launcher = (ROOT / "scripts/china/cluster.sh").read_text()
    dependency_paths = (
        "source/src/prime_rl/_compat.py",
        "source/src/prime_rl/glm52_routing.py",
        "source/src/prime_rl/model_identity.py",
        "source/src/prime_rl/trainer/mxfp4.py",
        "source/src/prime_rl/trainer/models/kernels/fp8_indexer.py",
        "source/src/prime_rl/trainer/models/layers/rotary_emb.py",
        "source/src/prime_rl/utils/monitor/__init__.py",
        "source/src/prime_rl/utils/monitor/swanlab.py",
        "source/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py",
        "source/packages/prime-rl-configs/src/prime_rl/configs/shared.py",
    )
    for path in dependency_paths:
        if path.startswith("source/src/"):
            target = "/app/src/" + path.removeprefix("source/src/")
        else:
            target = "/app/packages/prime-rl-configs/src/" + path.removeprefix(
                "source/packages/prime-rl-configs/src/"
            )
        assert f'-v "$deployment/{path}":{target}:ro' in launcher

    # These are the local imports in the mounted trainer override graph that
    # are not supplied by the pinned image. Keep this list explicit so a new
    # override cannot silently fall back to an unverified image module.
    imported_by = {
        "source/src/prime_rl/trainer/rl/train.py": (
            "source/src/prime_rl/_compat.py",
            "source/src/prime_rl/utils/monitor/__init__.py",
        ),
        "source/src/prime_rl/trainer/model.py": (
            "source/src/prime_rl/glm52_routing.py",
        ),
        "source/src/prime_rl/trainer/models/layers/moe.py": (
            "source/src/prime_rl/trainer/mxfp4.py",
        ),
        "source/src/prime_rl/trainer/models/glm_moe_dsa/sparse_mla_attention.py": (
            "source/src/prime_rl/trainer/models/kernels/fp8_indexer.py",
            "source/src/prime_rl/trainer/models/layers/rotary_emb.py",
        ),
        "source/src/prime_rl/utils/monitor/__init__.py": (
            "source/src/prime_rl/utils/monitor/swanlab.py",
        ),
        "source/src/prime_rl/trainer/runs.py": (
            "source/packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py",
        ),
        "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py": (
            "source/packages/prime-rl-configs/src/prime_rl/configs/shared.py",
        ),
    }
    for source, targets in imported_by.items():
        module = ast.parse((ROOT / source).read_text())
        imports = set()
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        for target in targets:
            if target.startswith("source/src/"):
                module_name = target.removeprefix("source/src/").removesuffix(".py").replace(
                    "/", "."
                )
            else:
                module_name = target.removeprefix(
                    "source/packages/prime-rl-configs/src/"
                ).removesuffix(".py").replace("/", ".")
            if module_name.endswith(".__init__"):
                module_name = module_name.removesuffix(".__init__")
            assert module_name in imports, (source, module_name, imports)
