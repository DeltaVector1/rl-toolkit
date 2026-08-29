"""Deployment destination guards for shared and inference-only files."""

import hashlib
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _training_ops():
    source = ROOT / ".agents/skills/training-ops/src"
    sys.path.insert(0, str(source))
    import training_ops

    sys.path.remove(str(source))
    return training_ops


def test_shared_trainer_source_can_deploy_to_a_mounted_trainer(monkeypatch):
    training_ops = _training_ops()
    relative = "source/packages/prime-rl-configs/src/prime_rl/configs/trainer.py"
    digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    def fake_ssh(_host, _command, *args, **kwargs):
        return 0, digest

    monkeypatch.setattr(training_ops, "_ssh", fake_ssh)
    assert training_ops.deploy(relative, nodes=["n82"]) is True


def test_inference_only_launcher_stays_forbidden_on_a_trainer():
    training_ops = _training_ops()
    with pytest.raises(training_ops.Refused, match="inference-only file"):
        training_ops.deploy("scripts/h200/start_mtp_production.sh", nodes=["n82"])
