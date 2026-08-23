import tomllib
from pathlib import Path


root = Path(__file__).resolve().parents[2]
script = (root / "scripts/h200/engine_watchdog.sh").read_text()


def test_startup_uses_explicit_qualification_grace() -> None:
    with (root / "configs/operations.toml").open("rb") as stream:
        operations = tomllib.load(stream)
    assert operations["limits"]["h200_engine_startup_grace_seconds"] == 7200
    assert "STARTUP_GRACE=${STARTUP_GRACE:-7200}" in script
    assert 'if [ "$age" -le "$STARTUP_GRACE" ]' in script


def test_startup_and_ready_states_have_separate_repairs() -> None:
    assert 'ever_healthy=0' in script
    assert 'ever_healthy=1' in script
    assert 'restart_engine "startup exceeded ${STARTUP_GRACE}s"' in script
    assert 'restart_engine "post-ready probe failures=$misses"' in script


def test_stopped_container_is_checked_before_generation() -> None:
    status_check = script.index('if [ "$status" != "running" ]')
    generation_probe = script.index('code=$(curl')
    assert status_check < generation_probe
    assert 'restart_engine "container=${status:-missing}"' in script
