"""Dependency-light tests: no training, downloading, or GPU initialization."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

try:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
except ImportError:
    compose = None

ROOT = Path(__file__).resolve().parents[1]


def launch(*arguments, successful=True):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train.sh"), "--dry-run", *arguments],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_BIN": "python"},
        check=False,
    )
    if successful and result.returncode:
        raise AssertionError(result.stderr)
    return result, shlex.split(result.stdout)


class TrainingLauncherTests(unittest.TestCase):
    def test_each_method_is_exclusive_and_uses_the_final_step(self):
        expected = {
            "grpo": None,
            "grpo-h": "actor_rollout_ref.actor.entropy_coeff=0.001",
            "exppo": "algorithm.grpo_adv_reweight_enable=true",
            "f-grpo": "algorithm.f_grpo_enable=true",
            "grpo-ur": "algorithm.grpo_unlikeliness_enable=true",
            "repo-r": "algorithm.repo_r_enable=true",
            "dapo": "reward_model.overlong_buffer.enable=true",
            "dapo-exppo": "algorithm.grpo_adv_reweight_enable=true",
        }
        modifiers = set(v for v in expected.values() if v and "enable" in v)
        for method, flag in expected.items():
            with self.subTest(method=method):
                _, command = launch("--method", method)
                self.assertIn("trainer.total_training_steps=900", command)
                if flag:
                    self.assertIn(flag, command)
                actual = modifiers.intersection(command)
                allowed = {flag} if flag in modifiers else set()
                if method == "dapo-exppo":
                    allowed.add("reward_model.overlong_buffer.enable=true")
                self.assertEqual(actual, allowed)

    def test_models_and_runtime_overrides(self):
        for model, name in (("qwen25", "Qwen/Qwen2.5-7B-Instruct"),
                            ("qwen3", "Qwen/Qwen3-0.6B"),
                            ("phi35", "microsoft/Phi-3.5-mini-instruct")):
            _, command = launch("--model", model, "--model-path", name,
                                "--seed", "43", "--steps", "300", "--gpus", "4")
            self.assertIn(f"actor_rollout_ref.model.path={name}", command)
            self.assertIn("trainer.total_training_steps=300", command)
            self.assertIn("trainer.n_gpus_per_node=4", command)
            self.assertIn("data.seed=43", command)
            self.assertIn("actor_rollout_ref.actor.data_loader_seed=43", command)
            self.assertIn("actor_rollout_ref.actor.fsdp_config.seed=43", command)
            self.assertIn("actor_rollout_ref.rollout.engine_kwargs.vllm.seed=43", command)

    def test_components_default_to_ablation_budget(self):
        flags = {
            "no-gate": "algorithm.grpo_adv_reweight_gate_enabled=false",
            "no-normalization": "algorithm.grpo_adv_reweight_final_normalization_enabled=false",
            "no-tanh": "algorithm.grpo_adv_reweight_tanh_enabled=false",
            "positive-only": "algorithm.grpo_adv_reweight_sign=positive",
            "negative-only": "algorithm.grpo_adv_reweight_sign=negative",
        }
        for name, flag in flags.items():
            _, command = launch("--model", "phi35", "--ablation", name)
            self.assertIn(flag, command)
            self.assertIn("trainer.total_training_steps=300", command)

    def test_invalid_combinations_fail_without_launching(self):
        for args in (("--method", "grpo", "--ablation", "positive-only"),
                     ("--lambda", "1.2"), ("--steps", "0"),
                     ("--seed", "bad"), ("--model", "unknown")):
            result, _ = launch(*args, successful=False)
            self.assertNotEqual(result.returncode, 0)

    def test_paths_with_spaces_and_custom_reward_override(self):
        _, command = launch("--train-file", "data/my training.parquet", "--output", "outputs/my run",
                            "--", "custom_reward_function.path=experiments/my_reward.py")
        self.assertIn("data.train_files=data/my training.parquet", command)
        self.assertIn("trainer.default_local_dir=outputs/my run", command)
        self.assertIn("custom_reward_function.path=experiments/my_reward.py", command)

    def test_merger_uses_genuine_fsdp_export(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/merge_checkpoint.sh"), "--dry-run",
             "outputs/checkpoints/run/global_step_900/actor"],
            capture_output=True, text=True, check=True,
        )
        command = shlex.split(result.stdout)
        self.assertIn("verl.model_merger", command)
        self.assertIn("fsdp", command)
        self.assertIn("outputs/checkpoints/run/global_step_900/actor/hf_merged", command)

    def test_completed_run_exits_without_importing_training(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "latest_checkpointed_iteration.txt").write_text("900\n")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/train.sh"), "--output", directory,
                 "--model-path", "Qwen/Qwen2.5-7B-Instruct"],
                capture_output=True, text=True, check=True,
            )
        self.assertIn("nothing to train", result.stdout)

    @unittest.skipIf(compose is None, "Install hydra-core for configuration composition tests")
    def test_all_presets_compose_without_importing_the_trainer(self):
        previous_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            for method in ("grpo", "grpo-h", "exppo", "f-grpo", "grpo-ur", "repo-r", "dapo", "dapo-exppo"):
                _, command = launch("--method", method, "--seed", "43")
                overrides = command[command.index("--config-name") + 2:]
                with initialize_config_dir(config_dir=str(ROOT / "configs/training"), version_base=None):
                    config = compose(config_name="paper", overrides=overrides)
                self.assertEqual(config.trainer.total_training_steps, 900)
                self.assertEqual(config.actor_rollout_ref.actor.fsdp_config.seed, 43)
                self.assertEqual(config.data.gen_batch_size * config.actor_rollout_ref.rollout.n, 512)
                if method.startswith("dapo"):
                    self.assertTrue(config.algorithm.filter_groups.enable)
                    self.assertEqual(config.algorithm.filter_groups.metric, "acc")
                    self.assertEqual(config.actor_rollout_ref.actor.clip_ratio_low, 0.2)
                    self.assertEqual(config.actor_rollout_ref.actor.clip_ratio_high, 0.28)
                    self.assertEqual(config.actor_rollout_ref.actor.loss_agg_mode, "token-mean")
                    self.assertEqual(config.reward_model.reward_kwargs.max_resp_len, 10240)
                    self.assertTrue(config.reward_model.reward_kwargs.overlong_buffer_cfg.enable)
                    self.assertEqual(config.reward_model.reward_kwargs.overlong_buffer_cfg.len, 2048)
                else:
                    self.assertFalse(config.algorithm.filter_groups.enable)
                OmegaConf.to_container(config, resolve=True)
        finally:
            os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
