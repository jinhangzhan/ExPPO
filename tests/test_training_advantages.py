"""CPU numerical tests of the actual vendored functions, without Ray/vLLM imports."""

import ast
import __future__
from collections import defaultdict
import math
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = torch = None


def load_functions():
    source = Path(__file__).resolve().parents[1] / "third_party/verl/verl/trainer/ppo/core_algos.py"
    tree = ast.parse(source.read_text())
    names = {"compute_grpo_outcome_advantage", "apply_repo_r_advantage", "update_repo_r_controller"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np, "torch": torch, "defaultdict": defaultdict, "math": math}
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@unittest.skipIf(torch is None, "Install the prepared torch/numpy environment for numerical tests")
class TrainingAdvantageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_functions()

    def run_method(self, rewards, surprisals, **config):
        rewards = torch.tensor(rewards, dtype=torch.float64)
        surprisals = torch.tensor(surprisals, dtype=torch.float64)
        mask = torch.ones((len(rewards), 2), dtype=torch.float64)
        mask[0, 1] = 0  # Unequal lengths must not alter mean sequence surprisal.
        token_rewards = torch.zeros_like(mask)
        token_rewards[:, 0] = rewards
        logp = -surprisals[:, None].expand_as(mask).clone().requires_grad_(True)
        advantages, _ = self.functions["compute_grpo_outcome_advantage"](
            token_rewards, mask, np.array(["q"] * len(rewards)), old_log_probs=logp,
            config={"grpo_adv_reweight_lambda": 0.2, **config})
        self.assertEqual(float(advantages[0, 1]), 0)
        self.assertFalse(advantages.requires_grad)
        return advantages[:, 0], rewards, surprisals

    def test_full_method_matches_paper_equation(self):
        actual, rewards, surprisal = self.run_method(
            [1, 1, 0, 0], [0.1, 2.0, 0.2, 3.0], grpo_adv_reweight_enable=True)
        base = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        rarity = torch.tanh((surprisal - surprisal.mean()) / surprisal.std(unbiased=False).clamp_min(1e-8))
        rho = (rewards > 0).to(torch.float64).mean()
        gate = torch.where(rarity > 0, 1 - rho, torch.where(rarity < 0, rho, 0.0))
        shaped = base * (1 + 0.2 * gate * torch.sign(base) * rarity).clamp_min(0)
        expected = shaped * base.abs().sum() / (shaped.abs().sum() + 1e-8)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.abs().sum(), base.abs().sum(), atol=1e-7, rtol=0)
        torch.testing.assert_close(torch.sign(actual), torch.sign(base))

    def test_lambda_zero_recovers_grpo(self):
        baseline, _, _ = self.run_method([1, 1, 0, 0], [0.1, 2, 0.2, 3])
        shaped, _, _ = self.run_method([1, 1, 0, 0], [0.1, 2, 0.2, 3],
                                     grpo_adv_reweight_enable=True, grpo_adv_reweight_lambda=0)
        torch.testing.assert_close(shaped, baseline)

    def test_dapo_gate_uses_verifier_outcomes_before_length_penalty(self):
        reward = torch.tensor([1., 0., 0., -.5], dtype=torch.float64)
        # The second response was correct but receives a full length penalty.
        correctness = torch.tensor([1., 1., 0., 0.], dtype=torch.float64, requires_grad=True)
        surprisal = torch.tensor([.1, 2., .2, 3.], dtype=torch.float64)
        function = self.functions["compute_grpo_outcome_advantage"]
        kwargs = dict(token_level_rewards=reward[:, None], response_mask=torch.ones(4, 1),
                      index=np.array(["q"] * 4), old_log_probs=-surprisal[:, None],
                      config={"grpo_adv_reweight_enable": True, "grpo_adv_reweight_lambda": .2})
        actual, _ = function(**kwargs, verifier_correctness=correctness)
        baseline = (reward - reward.mean()) / (reward.std(unbiased=False) + 1e-6)
        rarity = torch.tanh((surprisal - surprisal.mean()) / surprisal.std(unbiased=False))
        gate = torch.full_like(baseline, .5)
        expected = baseline * (1 + .2 * gate * baseline.sign() * rarity)
        expected *= baseline.abs().sum() / (expected.abs().sum() + 1e-8)
        torch.testing.assert_close(actual[:, 0], expected)
        old_gate, _ = function(**kwargs)
        self.assertFalse(torch.equal(actual, old_gate))
        self.assertFalse(actual.requires_grad)

    def test_dapo_verifier_outcomes_are_validated(self):
        function = self.functions["compute_grpo_outcome_advantage"]
        for correctness in (torch.zeros(4, 1), torch.tensor([0., 1., .5, 0.]),
                            torch.tensor([0., 1., float("nan"), 0.])):
            with self.assertRaisesRegex(ValueError, "verifier_correctness"):
                function(torch.zeros(4, 1), torch.ones(4, 1), np.array(["q"] * 4),
                         verifier_correctness=correctness)

    def test_uniform_groups_remain_zero(self):
        for rewards in ([1, 1, 1, 1], [0, 0, 0, 0]):
            actual, _, _ = self.run_method(rewards, [0.1, 2, 0.2, 3], grpo_adv_reweight_enable=True)
            torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_positive_and_negative_restrictions_precede_normalization(self):
        for sign in ("positive", "negative"):
            actual, rewards, _ = self.run_method([1, 1, 0, 0], [0.1, 2, 0.2, 3],
                grpo_adv_reweight_enable=True, grpo_adv_reweight_sign=sign,
                grpo_adv_reweight_final_normalization_enabled=False)
            base = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            untouched = base < 0 if sign == "positive" else base > 0
            torch.testing.assert_close(actual[untouched], base[untouched])
            self.assertFalse(torch.equal(actual[~untouched], base[~untouched]))

    def test_all_component_switches_change_the_intended_formula(self):
        for no_gate, no_tanh, no_norm in ((True, False, False), (False, True, False), (False, False, True)):
            actual, rewards, surprisal = self.run_method([1, 0, 0, 0], [0.1, 2, 0.2, 3],
                grpo_adv_reweight_enable=True, grpo_adv_reweight_gate_enabled=not no_gate,
                grpo_adv_reweight_tanh_enabled=not no_tanh,
                grpo_adv_reweight_final_normalization_enabled=not no_norm)
            base = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            rarity = (surprisal - surprisal.mean()) / surprisal.std(unbiased=False).clamp_min(1e-8)
            if not no_tanh:
                rarity = torch.tanh(rarity)
            rho = (rewards > 0).to(torch.float64).mean()
            gate = torch.ones_like(base) if no_gate else torch.where(rarity > 0, 1 - rho, rho)
            expected = base * (1 + 0.2 * gate * torch.sign(base) * rarity).clamp_min(0)
            if not no_norm:
                expected = expected * base.abs().sum() / (expected.abs().sum() + 1e-8)
            torch.testing.assert_close(actual, expected)

    def test_focal_baseline_scales_the_whole_group(self):
        actual, rewards, _ = self.run_method([1, 0, 0, 0], [0.1, 2, 0.2, 3], f_grpo_enable=True)
        base = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        torch.testing.assert_close(actual, base * math.sqrt(0.75))

    def test_unlikeliness_uses_pre_normalization_rank(self):
        actual, rewards, surprisal = self.run_method([1, 1, 0, 0], [0.1, 2, 0.2, 3],
                                                    grpo_unlikeliness_enable=True)
        rank = torch.argsort(torch.argsort(-surprisal))
        modified = rewards * (1 - 0.25 * rank / len(rewards))
        expected = (modified - modified.mean()) / (modified.std(unbiased=False) + 1e-6)
        torch.testing.assert_close(actual, expected)

    def test_repo_r_token_shaping(self):
        advantage = torch.tensor([[1., -1., 0.]])
        logp = torch.tensor([[-2., -2., -2.]], requires_grad=True)
        actual = self.functions["apply_repo_r_advantage"](advantage, logp, torch.tensor([0.1]))
        torch.testing.assert_close(actual, torch.tensor([[1.2, -0.8, 0.]]))
        self.assertFalse(actual.requires_grad)

    def test_missing_old_policy_probabilities_fail(self):
        with self.assertRaisesRegex(ValueError, "requires old_log_probs"):
            self.functions["compute_grpo_outcome_advantage"](
                torch.zeros(4, 2), torch.ones(4, 2), np.array([0] * 4),
                config={"grpo_adv_reweight_enable": True})


if __name__ == "__main__":
    unittest.main()
