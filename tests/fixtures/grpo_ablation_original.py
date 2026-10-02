# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Frozen original scalar-GRPO/ExPPO function for ablation parity tests.

Extracted verbatim (without its registry decorator) from
codereasoner_minimal_20260922.zip, verl/verl/trainer/ppo/core_algos.py.
Source-file SHA-256: 238f2142bbce24499cc450bb739bd87f40f1dec92308995d0ddede1f47c3c49a.

The fixture preserves the supplied component/sign ablations and baseline math.
Shuffle/audit experiments are outside these tests and fail explicitly below.
This is a test reference, not a second training implementation.
"""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch


def validate_exppo_shuffle_config(config):
    if config and any(config.get(key, False) for key in (
        "grpo_adv_reweight_shuffle_within_class",
        "grpo_adv_reweight_audit_enable",
        "grpo_adv_reweight_post_update_kl_enable",
    )):
        raise ValueError("Shuffle/audit is outside this ablation reference fixture")


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    old_log_probs: Optional[torch.Tensor] = None,
    config: Optional[AlgoConfig] = None,
    shuffle_group_ids: Optional[np.ndarray] = None,
    shuffle_response_ids: Optional[np.ndarray] = None,
    global_step: Optional[int] = None,
    diagnostics: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    validate_exppo_shuffle_config(config)
    shuffle_enable = bool(config.get("grpo_adv_reweight_shuffle_within_class", False)) if config else False
    audit_enable = shuffle_enable or (bool(config.get("grpo_adv_reweight_audit_enable", False)) if config else False)
    if shuffle_enable or audit_enable:
        if old_log_probs is None:
            raise ValueError("ExPPO shuffle/audit requires old_log_probs; refusing ordinary-GRPO fallback")
        if old_log_probs.shape != response_mask.shape or token_level_rewards.shape != response_mask.shape:
            raise ValueError("ExPPO shuffle/audit requires aligned rewards, log probabilities, and masks")
        if not torch.all((response_mask == 0) | (response_mask == 1)):
            raise ValueError("ExPPO shuffle/audit requires a binary response mask")
        if not torch.isfinite(old_log_probs[response_mask.bool()]).all():
            raise ValueError("ExPPO shuffle/audit requires finite valid-token old_log_probs")
    scores = token_level_rewards.sum(dim=-1)
    raw_rewards = scores.clone()

    unlikeliness_enable = bool(config.get("grpo_unlikeliness_enable", False)) if config is not None else False
    if unlikeliness_enable:
        if old_log_probs is None:
            raise ValueError("GRPO-Unlikeliness requires old_log_probs to rank responses within each group")
        for modifier in (
            "grpo_adv_reweight_enable", "policy_adv_reweight_enable", "repo_r_enable",
            "f_grpo_enable", "cue_grpo_enable",
        ):
            if config.get(modifier, False):
                raise ValueError(f"GRPO-Unlikeliness cannot be combined with {modifier}")
        rank_beta = float(config.get("grpo_unlikeliness_rank_beta", 0.25))
        if not 0.0 <= rank_beta <= 1.0:
            raise ValueError("grpo_unlikeliness_rank_beta must be in [0, 1]")

        mask = response_mask.float()
        token_count = mask.sum(dim=-1).clamp_min(1)
        seq_logprob_mean = (old_log_probs.detach() * mask).sum(dim=-1) / token_count
        modified_scores = scores.clone()
        with torch.no_grad():
            for group_id in np.unique(index):
                group_mask = torch.as_tensor(index == group_id, device=scores.device, dtype=torch.bool)
                group_scores = raw_rewards[group_mask]
                # Keep uniform outcome groups free of rank-only learning signals.
                original_nonzero_adv = (group_scores - group_scores.mean()).abs() > epsilon
                group_logp = seq_logprob_mean[group_mask]
                # Ascending ordinal ranks, including incorrect responses and ties.
                group_ranks = torch.argsort(torch.argsort(group_logp))
                rank_fraction = group_ranks.to(dtype=scores.dtype) / group_scores.numel()
                penalized_scores = group_scores * (1.0 - rank_beta * rank_fraction)
                modified_scores[group_mask] = torch.where(original_nonzero_adv, penalized_scores, group_scores)
        scores = modified_scores

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = scores.new_tensor(0.0)
                id2std[idx] = scores.new_tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = scores_tensor.mean()
                id2std[idx] = scores_tensor.std(unbiased=False)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        f_grpo_enable = bool(config.get("f_grpo_enable", False)) if config is not None else False
        reweight_enable = bool(config.get("grpo_adv_reweight_enable", False)) if config is not None else False
        if f_grpo_enable and reweight_enable:
            raise ValueError("F-GRPO and GRPO advantage reweighting cannot be enabled together")

        if f_grpo_enable:
            focal_gamma = float(config.get("f_grpo_gamma", 0.5))
            reward_correct = float(config.get("f_grpo_reward_correct", 1.0))
            reward_wrong = float(config.get("f_grpo_reward_wrong", 0.0))

            if not math.isfinite(focal_gamma) or focal_gamma < 0.0:
                raise ValueError("f_grpo_gamma must be finite and >= 0")
            if not math.isfinite(reward_correct) or not math.isfinite(reward_wrong):
                raise ValueError("F-GRPO reward endpoints must be finite")
            if reward_correct <= reward_wrong:
                raise ValueError("reward_correct must be greater than reward_wrong")

            focal_weights = torch.ones_like(scores)
            reward_span = reward_correct - reward_wrong
            endpoint_atol = max(epsilon, 1e-6)

            for idx in id2score:
                group_mask = torch.as_tensor(index == idx, device=scores.device, dtype=torch.bool)
                group_rewards = raw_rewards[group_mask]
                is_correct = torch.isclose(
                    group_rewards,
                    group_rewards.new_tensor(reward_correct),
                    rtol=0.0,
                    atol=endpoint_atol,
                )
                is_wrong = torch.isclose(
                    group_rewards,
                    group_rewards.new_tensor(reward_wrong),
                    rtol=0.0,
                    atol=endpoint_atol,
                )
                if not torch.all(is_correct | is_wrong):
                    raise ValueError("F-GRPO requires binary rewards matching its endpoints")

                empirical_success = ((group_rewards.mean() - reward_wrong) / reward_span).clamp(0.0, 1.0)
                focal_weights[group_mask] = (1.0 - empirical_success).pow(focal_gamma)

            scores = scores * focal_weights

        mask = response_mask.float()
        logp = old_log_probs.detach() if old_log_probs is not None else None

        original_scores = scores.clone()
        if reweight_enable and logp is not None:
            # Group-normalized sequence-level gated reweight rule:
            #   \bar{ell}_i = (1 / T_i) * sum_t log pi_old(y_{i,t} | q, y_{i,<t})
            #   s_i = -\bar{ell}_i
            #   z_i = group-normalized surprisal score, where z_i > 0 is rare/surprising
            #   rho_q = (1 / G) * sum_j 1[R_j > 0]
            #   gate_i = 1 - rho_q for rare-positive / rare-negative cases
            #          = rho_q     for common-positive / common-negative cases
            #   M_i = max(M_min, 1 + lambda * gate_i * sign(A_i) * tanh(z_i))
            #   A_hat_i = M_i * A_i
            #   A_tilde_i = A_hat_i * sum_j |A_j| / (sum_j |A_hat_j| + eps)
            lam = float(config.get("grpo_adv_reweight_lambda", 0.1)) if config is not None else 0.1
            rw_eps = float(config.get("grpo_adv_reweight_eps", 1e-8)) if config is not None else 1e-8
            m_min = float(config.get("grpo_adv_reweight_min", 0.0)) if config is not None else 0.0
            final_normalization_enable = (
                bool(config.get("grpo_adv_reweight_final_normalization_enable", True))
                if config is not None
                else True
            )
            gate_enable = bool(config.get("grpo_adv_reweight_gate_enable", True)) if config is not None else True
            tanh_enable = bool(config.get("grpo_adv_reweight_tanh_enable", True)) if config is not None else True
            advantage_side = (
                str(config.get("grpo_adv_reweight_advantage_side", "both")).strip().lower()
                if config is not None
                else "both"
            )
            if advantage_side not in {"both", "positive", "negative"}:
                raise ValueError(
                    "grpo_adv_reweight_advantage_side must be one of "
                    f"both, positive, or negative; got {advantage_side!r}"
                )

            token_count = mask.sum(dim=-1).clamp_min(1)
            seq_logprob_mean = (logp * mask).sum(dim=-1) / token_count
            surprisal = -seq_logprob_mean
            z = torch.zeros_like(surprisal)
            u = torch.zeros_like(surprisal)
            group_sigmas = []
            group_pos_counts = []
            group_pos_fracs = []

            for idx in id2score:
                group_mask = torch.as_tensor(index == idx, device=surprisal.device, dtype=torch.bool)
                group_s = surprisal[group_mask]
                group_adv = original_scores[group_mask]
                group_pos_count = (group_adv > 0).float().sum()
                group_pos_frac = group_pos_count / group_s.numel()
                group_pos_counts.append(group_pos_count.detach())
                group_pos_fracs.append(group_pos_frac.detach())

                group_mu = group_s.mean()
                group_sigma = group_s.std(unbiased=False).clamp_min(rw_eps)

                group_sigmas.append(group_sigma.detach())

                group_z = (group_s - group_mu) / group_sigma
                z[group_mask] = group_z
                u[group_mask] = torch.tanh(group_z) if tanh_enable else group_z

            # z = z.clamp(-2.0, 2.0)

            adv = original_scores

            # Detach all reweighting factors from logprob-derived rarity scores.
            # u is already tanh(z) for the z-score path above.
            with torch.no_grad():
                gate = torch.zeros_like(adv)
                rho_per_sample = torch.zeros_like(adv)

                rare = u > 0
                common = u < 0
                pos = adv > 0
                neg = adv < 0
                if advantage_side == "positive":
                    active_side = pos
                elif advantage_side == "negative":
                    active_side = neg
                else:
                    active_side = pos | neg
                common_pos = common & pos
                rare_neg = rare & neg

                # rho_q is the raw group pass rate, before GRPO advantage normalization.
                # Use raw reward > 0 exactly as the gate definition specifies.
                for idx in id2score:
                    group_mask = torch.as_tensor(index == idx, device=adv.device, dtype=torch.bool)
                    rho_q = (raw_rewards[group_mask] > 0).to(dtype=adv.dtype).mean()
                    rho_per_sample[group_mask] = rho_q

                    if gate_enable:
                        rare_pos = group_mask & pos & rare
                        common_pos_mask = group_mask & pos & common
                        common_neg = group_mask & neg & common
                        rare_neg_mask = group_mask & neg & rare

                        gate[rare_pos] = 1.0 - rho_q
                        gate[common_pos_mask] = rho_q
                        gate[common_neg] = rho_q
                        gate[rare_neg_mask] = 1.0 - rho_q
                    else:
                        gate[group_mask] = 1.0

                # Sign-side ablations restrict the raw EPPO modulation term.
                # The standard groupwise final normalization remains a separate
                # component and may rescale all samples after this masking.
                gate = gate * active_side.to(dtype=gate.dtype)
                raw_multiplier = 1.0 + lam * gate * torch.sign(adv) * u
                multiplier = torch.clamp(raw_multiplier, min=m_min)
                if advantage_side != "both":
                    # Keep the inactive sign at exactly ordinary-GRPO weight,
                    # even when a non-default multiplier floor exceeds one.
                    multiplier = torch.where(active_side, multiplier, torch.ones_like(multiplier))

                a_hat = adv * multiplier
                mag_scale = torch.ones_like(adv)
                if final_normalization_enable:
                    for idx in id2score:
                        group_mask = torch.as_tensor(index == idx, device=adv.device, dtype=torch.bool)
                        old_mag = adv[group_mask].abs().sum()
                        new_mag = a_hat[group_mask].abs().sum()
                        mag_scale[group_mask] = old_mag / (new_mag + rw_eps)

                reweighted_scores = a_hat * mag_scale

            # ---------------- diagnosis logs only ----------------
            debug_enable = bool(config.get("grpo_adv_reweight_debug", False)) if config is not None else False

            if debug_enable:
                max_response_len = float(config.get("max_response_length", response_mask.shape[-1])) if config is not None else float(response_mask.shape[-1])

                adv = original_scores.detach()
                length = mask.sum(dim=-1).detach()
                s = surprisal.detach()
                z_det = z.detach()
                u_det = u.detach()
                m_det = multiplier.detach()
                raw_m_det = raw_multiplier.detach()

                nonzero = adv != 0
                pos = adv > 0
                neg = adv < 0
                rare = u_det > 0
                common = u_det < 0
                hit_max = length >= max_response_len

                rare_pos = rare & pos
                rare_neg = rare & neg
                common_pos = common & pos
                common_neg = common & neg

                def sm(x):
                    return x.float().mean().item() if x.numel() > 0 else float("nan")

                def sf(x):
                    return x.float().mean().item() if x.numel() > 0 else float("nan")

                def sq(x, q):
                    return torch.quantile(x.float(), q).item() if x.numel() > 0 else float("nan")

                def scorr(x, y):
                    x = x.float()
                    y = y.float()
                    if x.numel() < 2:
                        return float("nan")
                    x = x - x.mean()
                    y = y - y.mean()
                    denom = x.norm() * y.norm()
                    if denom.item() == 0:
                        return float("nan")
                    return (x @ y / denom).item()

                group_sigmas_t = torch.stack(group_sigmas) if len(group_sigmas) > 0 else torch.empty(0, device=s.device)
                group_pos_counts_t = torch.stack(group_pos_counts) if len(group_pos_counts) > 0 else torch.empty(0, device=s.device)
                group_pos_fracs_t = torch.stack(group_pos_fracs) if len(group_pos_fracs) > 0 else torch.empty(0, device=s.device)
                group_sizes_t = torch.as_tensor([len(id2score[idx]) for idx in id2score], device=s.device, dtype=s.dtype)
                mixed_group = (group_pos_counts_t > 0) & (group_pos_counts_t < group_sizes_t)
                all_negative_group = group_pos_counts_t == 0
                all_positive_group = group_pos_counts_t == group_sizes_t
                gate_det = gate.detach()
                mag_scale_det = mag_scale.detach()
                reweighted_det = reweighted_scores.detach()
                update_mass = reweighted_det.abs()
                update_mass_denom = update_mass.sum().clamp_min(rw_eps)

                def mass_frac(mask_):
                    return (update_mass[mask_].sum() / update_mass_denom).item() if mask_.numel() > 0 else float("nan")

                log_line = (
                    f"[adv-rw-debug] "
                    f"lam={lam:.3f} "
                    f"final_norm={int(final_normalization_enable)} "
                    f"gate_enable={int(gate_enable)} "
                    f"tanh_enable={int(tanh_enable)} "
                    f"adv_side={advantage_side} "
                    f"adv_side_active_frac={sf(active_side):.3f} "
                    f"rho_mean={sm(rho_per_sample.detach()):.4f} "
                    f"gate_mean={sm(gate_det):.4f} "
                    f"mag_scale_mean={sm(mag_scale_det):.4f} "
                    f"old_abs_sum={adv.abs().sum().item():.4f} "
                    f"new_abs_sum={reweighted_det.abs().sum().item():.4f} "
                    f"pos_count_mean={sm(group_pos_counts_t):.3f} "
                    f"pos_count_p50={sq(group_pos_counts_t, 0.50):.3f} "
                    f"pos_count_p90={sq(group_pos_counts_t, 0.90):.3f} "
                    f"pos_frac_mean={sm(group_pos_fracs_t):.3f} "
                    f"mixed_group_frac={sf(mixed_group):.3f} "
                    f"all_negative_group_frac={sf(all_negative_group):.3f} "
                    f"all_positive_group_frac={sf(all_positive_group):.3f} "
                    f"adv_zero_frac={sf(adv == 0):.3f} "
                    f"len_mean={sm(length):.1f} "
                    f"len_p90={sq(length, 0.90):.1f} "
                    f"hit_max={sf(hit_max):.3f} "
                    f"adv_zero={sf(adv == 0):.3f} "
                    f"adv_pos={sf(pos):.3f} "
                    f"adv_neg={sf(neg):.3f} "
                    f"s_mean={sm(s):.4f} "
                    f"s_std={s.float().std(unbiased=False).item():.4f} "
                    f"z_std={z_det.float().std(unbiased=False).item():.4f} "
                    f"u_mean={sm(u_det):.4f} "
                    f"u_std={u_det.float().std(unbiased=False).item():.4f} "
                    f"u_abs95={sf(u_det.abs() > 0.95):.3f} "
                    f"u_pos_mean={sm(u_det[pos]):.4f} "
                    f"u_neg_mean={sm(u_det[neg]):.4f} "
                    f"corr_adv_u={scorr(adv, u_det):.3f} "
                    f"pos_common_frac={sf(common[pos]):.3f} "
                    f"pos_rare_frac={sf(rare[pos]):.3f} "
                    f"neg_rare_frac={sf(rare[neg]):.3f} "
                    f"neg_common_frac={sf(common[neg]):.3f} "
                    f"pos_m_mean={sm(m_det[pos]):.4f} "
                    f"neg_m_mean={sm(m_det[neg]):.4f} "
                    f"m_nz_mean={sm(m_det[nonzero]):.4f} "
                    f"m_pos_mean={sm(m_det[pos]):.4f} "
                    f"m_neg_mean={sm(m_det[neg]):.4f} "
                    f"raw_m_lt0={sf(raw_m_det < 0):.3f} "
                    f"m_floor={sf(m_det <= m_min + 1e-8):.3f} "
                    f"corr_len_s={scorr(length, s):.3f} "
                    f"corr_len_u={scorr(length, u_det):.3f} "
                    f"corr_len_m_nz={scorr(length[nonzero], m_det[nonzero]):.3f} "
                    f"common_pos_frac={sf(common_pos):.3f} "
                    f"common_pos_len_mean={sm(length[common_pos]):.1f} "
                    f"common_pos_adv_mean={sm(adv[common_pos]):.4f} "
                    f"common_pos_m_mean={sm(m_det[common_pos]):.4f} "
                    f"rare_pos_update_mass_frac={mass_frac(rare_pos):.3f} "
                    f"rare_neg_update_mass_frac={mass_frac(rare_neg):.3f} "
                    f"common_pos_update_mass_frac={mass_frac(common_pos):.3f} "
                    f"common_neg_update_mass_frac={mass_frac(common_neg):.3f} "
                    f"rare_pos_frac={sf(rare_pos):.3f} "
                    f"rare_pos_len_mean={sm(length[rare_pos]):.1f} "
                    f"rare_pos_len={sm(length[rare_pos]):.1f} "
                    f"rare_pos_m={sm(m_det[rare_pos]):.3f} "
                    f"rare_pos_hitmax={sf(hit_max[rare_pos]):.3f} "
                    f"rare_neg_m={sm(m_det[rare_neg]):.3f} "
                    f"rare_neg_frac={sf(rare_neg):.3f} "
                    f"rare_neg_len_mean={sm(length[rare_neg]):.1f} "
                    f"rare_neg_len={sm(length[rare_neg]):.1f} "
                    f"rare_neg_hitmax={sf(hit_max[rare_neg]):.3f} "
                    f"hit_max_frac={sf(hit_max):.3f} "
                    f"corr_len_m_nonzero={scorr(length[nonzero], m_det[nonzero]):.3f} "
                    f"common_pos_len={sm(length[common_pos]):.1f} "
                    f"common_pos_m={sm(m_det[common_pos]):.3f} "
                    f"common_neg_len_mean={sm(length[common_neg]):.1f} "
                    f"common_neg_len={sm(length[common_neg]):.1f} "
                    f"common_neg_m={sm(m_det[common_neg]):.3f} "
                    f"group_sigma_mean={sm(group_sigmas_t):.6f} "
                    f"group_sigma_p10={sq(group_sigmas_t, 0.10):.6f} "
                )

                print(log_line, flush=True)
            # -----------------------------------------------------

            scores = reweighted_scores
        # The control acts on FINAL sequence coefficients, after every ExPPO
        # multiplier/normalization and before broadcasting to response tokens.
        if shuffle_enable or audit_enable:
            before_shuffle = scores.clone()
            if shuffle_enable:
                scores = shuffle_exppo_within_class(
                    scores, raw_rewards, index, shuffle_group_ids, shuffle_response_ids,
                    config.get("grpo_adv_reweight_shuffle_seed"), global_step,
                )
            if diagnostics is not None:
                diagnostics.update(exppo_coefficient_mass_metrics(before_shuffle, scores, raw_rewards, response_mask))
                diagnostics["exppo_audit/shuffle_enabled"] = float(shuffle_enable)
        if scores.ndim == 1:
            scores = scores.unsqueeze(-1)
        scores = scores * mask

    return scores, scores
