"""Guards for the strict MTP stability profile in the trainer launcher."""

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
