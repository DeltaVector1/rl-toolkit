from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_china_supervisor_checks_all_nodes_and_never_restarts() -> None:
    script = _read("scripts/china/run_supervisor.sh")
    assert "trainer_failures" in script
    assert "for id in $TRAINER_NODES" in script
    assert 'host="10.92.20.$id"' in script
    assert "MAX_MISSES" in script
    assert 'if [ "$id" = 82 ]' in script
    assert "pkill -9 -f PRIME-RL" not in script
    assert "refusing automatic restart" in script
    assert "bash $CLUSTER trainer" not in script


def test_edge_watchdog_stops_generation_and_publisher() -> None:
    script = _read("scripts/h200/trainer_failure_watchdog.sh")
    assert "max_misses" in script
    assert 'docker rm -f "$orchestrator_container" "$environment_container"' in script
    assert "[s]ync_rollout_batches.sh" in script


def test_edge_launcher_requires_trainer_watchdog() -> None:
    script = _read("scripts/h200/start_orchestrator.sh")
    assert "trainer_failure_watchdog.sh" in script
    assert "trainer failure watchdog exited during startup" in script
