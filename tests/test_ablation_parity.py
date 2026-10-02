"""Compare release algorithms with the supplied original experiment function."""

import __future__
import ast
from collections import defaultdict
import hashlib
import importlib.util
import itertools
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = Path(__file__).parent / "fixtures/grpo_ablation_original.py"
FUNCTION_NAME = "compute_grpo_outcome_advantage"

# Public release spelling -> original transferred source spelling. All other
# configuration keys, including lambda/epsilon/floor, are unchanged.
ORIGINAL_KNOBS = {
    "grpo_adv_reweight_gate_enabled": "grpo_adv_reweight_gate_enable",
    "grpo_adv_reweight_tanh_enabled": "grpo_adv_reweight_tanh_enable",
    "grpo_adv_reweight_final_normalization_enabled": "grpo_adv_reweight_final_normalization_enable",
    "grpo_adv_reweight_sign": "grpo_adv_reweight_advantage_side",
}


def load_release():
    path = ROOT / "third_party/verl/verl/trainer/ppo/core_algos.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION_NAME)
    node.decorator_list = []
    namespace = {"np": np, "torch": torch, "defaultdict": defaultdict, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[FUNCTION_NAME]


@pytest.fixture(scope="module")
def functions():
    spec = importlib.util.spec_from_file_location("original_ablation_reference", REFERENCE_PATH)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return load_release(), reference.compute_grpo_outcome_advantage


def rollout(dtype, trial):
    generator = torch.Generator().manual_seed(900 + trial)
    sizes = [1, 2, 4, 8, 16, 4, 4]
    groups = np.repeat(np.arange(len(sizes)), sizes)
    batch_size, response_length = len(groups), 7
    outcomes = torch.randint(0, 2, (batch_size,), generator=generator).to(dtype)
    outcomes[-8:-4] = 0  # Uniform failures and successes remain zero-advantage.
    outcomes[-4:] = 1
    lengths = torch.randint(1, response_length + 1, (batch_size,), generator=generator)
    mask = (torch.arange(response_length)[None, :] < lengths[:, None]).to(dtype)
    rewards = torch.zeros_like(mask)
    rewards[torch.arange(batch_size), lengths - 1] = outcomes
    log_probs = -torch.rand(batch_size, response_length, dtype=dtype, generator=generator) * 8
    if trial == 0:
        # Identical within-group surprisal exercises the variance epsilon.
        log_probs[groups == 3] = -0.5
    elif trial == 1:
        # Raw-z (no-tanh) variants exercise clamping of negative multipliers.
        log_probs[groups == 4] = -0.01
        log_probs[np.where(groups == 4)[0][0]] = -100
    order = torch.randperm(batch_size, generator=generator)
    return rewards[order], mask[order], groups[order.numpy()], log_probs[order].requires_grad_(True)


def assert_parity(functions, data, config, normalize_rewards=True):
    release, original = functions
    rewards, mask, groups, log_probs = data
    common = dict(response_mask=mask, index=groups, old_log_probs=log_probs,
                  norm_adv_by_std_in_grpo=normalize_rewards)
    actual = release(token_level_rewards=rewards.clone(), config=config, **common)
    original_config = {ORIGINAL_KNOBS.get(key, key): value for key, value in config.items()}
    expected = original(token_level_rewards=rewards.clone(), config=original_config, **common)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        assert not actual_tensor.requires_grad
        assert torch.isfinite(actual_tensor).all()
        assert torch.count_nonzero(actual_tensor[mask == 0]) == 0


def test_reference_function_matches_original_transferred_source():
    source = REFERENCE_PATH.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION_NAME)
    verbatim = ast.get_source_segment(source, node) + "\n"
    assert hashlib.sha256(verbatim.encode()).hexdigest() == (
        "4d5920513eea05be964f917600f16082a025c2bb224d013632bc732b99b467e2"
    )


@pytest.mark.parametrize("gate,tanh,normalization,side", list(itertools.product(
    (True, False), (True, False), (True, False), ("both", "positive", "negative"))))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_component_and_sign_ablations(functions, gate, tanh, normalization, side, dtype):
    for trial, normalize_rewards, floor in itertools.product(range(3), (True, False), (0.0, 1.25)):
        config = {
            "grpo_adv_reweight_enable": True,
            "grpo_adv_reweight_lambda": 0.2 if trial == 0 else 1.0,
            "grpo_adv_reweight_min": floor,
            "grpo_adv_reweight_gate_enabled": gate,
            "grpo_adv_reweight_tanh_enabled": tanh,
            "grpo_adv_reweight_final_normalization_enabled": normalization,
            "grpo_adv_reweight_sign": side,
        }
        assert_parity(functions, rollout(dtype, trial), config, normalize_rewards)


@pytest.mark.parametrize("coefficient", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
def test_original_lambda_sweep(functions, coefficient):
    for trial in range(4):
        assert_parity(functions, rollout(torch.float64, trial), {
            "grpo_adv_reweight_enable": True,
            "grpo_adv_reweight_lambda": coefficient,
        })


@pytest.mark.parametrize("config", [{}, {"f_grpo_enable": True}, {"grpo_unlikeliness_enable": True}])
def test_existing_baseline_arithmetic_matches_original(functions, config):
    for dtype, trial, normalize in itertools.product((torch.float32, torch.float64), range(3), (True, False)):
        assert_parity(functions, rollout(dtype, trial), config, normalize)


def test_scalar_grpo_uses_original_population_standard_deviation(functions):
    release, _ = functions
    rewards = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    actual, _ = release(rewards, torch.ones_like(rewards), np.array([0, 0]), config={})
    expected = torch.tensor([[0.5], [-0.5]], dtype=torch.float64) / (0.5 + 1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_original_side_string_normalization(functions):
    assert_parity(functions, rollout(torch.float64, 0), {
        "grpo_adv_reweight_enable": True,
        "grpo_adv_reweight_sign": " POSITIVE ",
    })


def test_release_retains_missing_log_probability_guard(functions):
    release, _ = functions
    rewards, mask, groups, _ = rollout(torch.float64, 0)
    with pytest.raises(ValueError, match="requires old_log_probs"):
        release(rewards, mask, groups, config={"grpo_adv_reweight_enable": True})
