"""Training-environment checks for code commands; no GPU workers are loaded."""

import os
from pathlib import Path
import shlex
import subprocess

import pytest

hydra = pytest.importorskip("hydra")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ["grpo", "grpo-h", "exppo"])
def test_hydra_accepts_code_training_command(method, monkeypatch):
    monkeypatch.chdir(ROOT)
    output = subprocess.check_output(["bash", "scripts/train_codereasoner.sh", "--method", method, "--dry-run"],
                                     text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    command = shlex.split(output)
    start = command.index("paper") + 1
    overrides = command[start:]
    with hydra.initialize_config_dir(config_dir=str(ROOT / "configs/training"), version_base=None):
        config = hydra.compose(config_name="paper", overrides=overrides)
    assert config.custom_reward_function.path == "evaluation/code/reward.py"
    assert config.actor_rollout_ref.rollout.max_model_len == 1536
    assert config.actor_rollout_ref.actor.entropy_coeff == (0.001 if method == "grpo-h" else 0)
    assert config.algorithm.grpo_adv_reweight_enable is (method == "exppo")
