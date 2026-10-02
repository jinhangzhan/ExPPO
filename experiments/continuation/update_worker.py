#!/usr/bin/env python3
"""Isolated, matched-response one-step policy audit on enumerable coin change.

Qwen2.5-7B local audit: standardized BF16 checkpoint inputs promoted to FP32
masters, BF16-autocast gradients, and fresh CPU-offloaded AdamW. This is not
historical continuation. Full semantic effects use direct fresh sampling,
not importance weighting of archived responses.
The objective has the exact PPO gradient at ratio one; clipping is inactive at
that point. Canonical strings measure disjoint, explicitly restricted response
events. Their summed mass is not the full semantic-mode probability.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Must be set before CUDA/cuBLAS initializes. Keep all actor forks reproducible.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'third_party/verl/verl/trainer/ppo/core_algos.py'
from task_utils import extract_integer_list, verify_response
from detokenization import verify_detokenization


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def render(tokenizer, messages):
    # Identical explicit no-thinking template to generate_generalpoints_vllm.py.
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop('enable_thinking')
        return tokenizer.apply_chat_template(messages, **kwargs)


def format_fields(response):
    text = response.strip()
    literal = False
    fenced = False
    try:
        literal = isinstance(json.loads(text), dict)
    except (ValueError, TypeError):
        pass
    if text.startswith('```') and text.endswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
        stripped = re.sub(r'\s*```$', '', stripped)
        try:
            fenced = isinstance(json.loads(stripped), dict)
        except (ValueError, TypeError):
            pass
    values = extract_integer_list(response, 'counts')
    return {
        'literal_valid_json': int(literal),
        'fenced_valid_json': int(fenced),
        'counts_extractable': int(values is not None),
        'format_class': ('plain_json' if literal else 'fenced_json' if fenced
                         else 'embedded_counts' if values is not None else 'missing_counts'),
        'extracted_vector': ','.join(map(str, values)) if values is not None else '',
    }


def advantages_numpy(rewards, means, groups, variant):
    """Independent float64 implementation of active core_algos sequence rule."""
    rewards = np.asarray(rewards, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    groups = np.asarray(groups)
    out = {name: np.zeros(len(rewards)) for name in
           ('A_pre', 'A_post', 'u', 'z', 'rho', 'gate', 'multiplier', 'mass_scale', 'modified_reward')}
    out['multiplier'][:] = 1
    out['mass_scale'][:] = 1
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        r, lp = rewards[indices], means[indices]
        if len(r) < 2:
            raise ValueError('audit requires at least two responses per group')
        a = (r - r.mean()) / (r.std(ddof=0) + 1e-6)
        surprisal = -lp
        z = (surprisal - surprisal.mean()) / max(surprisal.std(ddof=0), 1e-8)
        u = np.tanh(z)
        rho = np.mean(r > 0)
        gate = np.where(u > 0, 1 - rho, np.where(u < 0, rho, 0))
        gate[a == 0] = 0
        multiplier, scale, modified = np.ones_like(r), 1.0, r.copy()
        post = a.copy()
        if variant == 'eppo':
            multiplier = np.maximum(0, 1 + .2 * gate * np.sign(a) * u)
            provisional = a * multiplier
            scale = np.abs(a).sum() / (np.abs(provisional).sum() + 1e-8)
            post = provisional * scale
        elif variant == 'unlikeliness':
            # Active implementation uses argsort(argsort(mean_logprob)), rank/G.
            ranks = np.argsort(np.argsort(lp))
            if len(np.unique(lp)) != len(lp):
                # Match the active torch tie behavior; no invented stable tie-break.
                import torch
                ranks = torch.argsort(torch.argsort(torch.tensor(lp, dtype=torch.float64))).numpy()
            eligible = np.abs(r - r.mean()) > 1e-6
            modified = np.where(eligible, r * (1 - .25 * ranks / len(r)), r)
            post = (modified - modified.mean()) / (modified.std(ddof=0) + 1e-6)
        elif variant not in ('grpo', 'grpo_h'):
            raise ValueError(variant)
        for key, val in dict(A_pre=a, A_post=post, u=u, z=z, rho=rho, gate=gate,
                             multiplier=multiplier, mass_scale=scale,
                             modified_reward=modified).items():
            out[key][indices] = val
    return out


def active_core_function():
    """Load just the active function, preserving its code and excluding imports."""
    import torch
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'compute_grpo_outcome_advantage')
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    namespace = {'torch': torch, 'np': np, 'math': math, 'defaultdict': defaultdict}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[function.name]


def crosscheck_active_core(rewards, means, groups, variant):
    import torch
    rewards = np.asarray(rewards, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    # T=1 represents the same scalar reward and mean log-probability exactly.
    reward_tensor = torch.tensor(rewards[:, None], dtype=torch.float64)
    mask = torch.ones_like(reward_tensor)
    old_logp = torch.tensor(means[:, None], dtype=torch.float64)
    config = {'grpo_adv_reweight_enable': variant == 'eppo',
              'grpo_adv_reweight_lambda': .2, 'grpo_adv_reweight_eps': 1e-8,
              'grpo_unlikeliness_enable': variant == 'unlikeliness',
              'grpo_unlikeliness_rank_beta': .25}
    actual, _ = active_core_function()(reward_tensor, mask, np.asarray(groups),
                                      old_log_probs=old_logp, config=config)
    expected = advantages_numpy(rewards, means, groups, variant)['A_post']
    difference = float(np.max(np.abs(actual[:, 0].numpy() - expected)))
    # Float32 pass-rate gates in active core can introduce tiny roundoff.
    if difference > 2e-6:
        raise AssertionError(f'active advantage rule mismatch: {difference}')
    return difference


def load_panel(data_root, split, bank_path, tokenizer, canary, bank_k=128):
    marker = json.loads(Path(str(bank_path) + '.complete.json').read_text())
    if marker.get('status') != 'complete' or marker['outcomes_sha256'] != sha256_file(bank_path):
        raise ValueError('unsealed or corrupted response bank')
    prompt_path = Path(data_root) / f'{split}_prompts.jsonl'
    if marker['prompts_sha256'] != sha256_file(prompt_path):
        raise ValueError('bank prompt manifest mismatch')
    if marker['samples_per_prompt'] != bank_k or marker['verifier_sha256'] != sha256_file(ROOT / 'source_snapshot/coin_change_final_verifier.py'):
        raise ValueError('bank sample count or verifier mismatch')
    prompts = read_jsonl(Path(data_root) / f'{split}_prompts.jsonl')
    prompts = [{**prompt, 'split': split} for prompt in prompts]
    if canary:
        prompts = prompts[:2]
    elif len(prompts) != 32:
        raise ValueError(f'expected 32 {split} prompts, found {len(prompts)}')
    prompt_map = {row['prompt_id']: row for row in prompts}
    rows = []
    seen = set()
    for row in read_jsonl(bank_path):
        if row['prompt_id'] not in prompt_map:
            continue
        index = int(row['sample_index'])
        if index >= 16:
            continue
        key = (row['prompt_id'], index)
        if key in seen:
            raise ValueError(f'duplicate sample {key}')
        seen.add(key)
        if not row.get('response_token_ids'):
            raise ValueError(f'missing exact response tokens: {key}')
        # Historical K128 banks contain tokens but no proposal likelihoods.
        # They are never treated as an importance-sampling proposal here.
        prompt = prompt_map[row['prompt_id']]
        verified = verify_response(row['response'], prompt)
        if bool(verified['reward']) != bool(row['reward']) or verified['mode_key'] != row.get('mode_key'):
            raise ValueError(f'bank/parser mismatch {key}: {verified}')
        prompt_text = render(tokenizer, prompt['messages'])
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        if row.get('prompt_token_ids') is not None and list(row['prompt_token_ids']) != prompt_ids:
            raise ValueError(f'prompt token mismatch: {key}')
        response_ids = list(map(int, row['response_token_ids']))
        if int(row.get('response_length', len(response_ids))) != len(response_ids):
            raise ValueError(f'response length mismatch: {key}')
        decode_view = verify_detokenization(tokenizer, prompt_ids, response_ids, row['response'])
        rows.append({**row, **format_fields(row['response']), 'split': split,
                     'detokenization_verification': decode_view,
                     '_prompt_ids': prompt_ids, '_response_ids': response_ids,
                     'prompt_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(),
                     'in_update': int(split == 'train' and index < 16)})
    rows.sort(key=lambda row: (int(row['panel_index']), int(row['sample_index'])))
    for prompt in prompts:
        indices = sorted(r['sample_index'] for r in rows if r['prompt_id'] == prompt['prompt_id'])
        expected = list(range(16))
        if indices != expected:
            raise ValueError(f'{split}/{prompt["prompt_id"]}: incomplete bank: {len(indices)}')
    return prompts, rows


def canonical_rows(prompts, tokenizer):
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int):
        raise ValueError('canonical scoring requires one tokenizer EOS id')
    rows = []
    for prompt in prompts:
        valid = set(prompt['valid_mode_keys'])
        distractors = prompt.get('canonical_distractor_vectors', prompt.get('distractor_vectors', prompt['payload'].get('distractor_vectors')))
        if distractors is None:
            raise ValueError(f'predeclared distractors missing for {prompt["prompt_id"]}')
        vectors = [list(map(int, key.split(','))) for key in sorted(valid)] + list(distractors)
        keys = [','.join(map(str, vector)) for vector in vectors]
        if len(set(keys)) != len(keys) or len(valid.intersection(keys[len(valid):])):
            raise ValueError('canonical candidate sets overlap or contain duplicates')
        prompt_ids = tokenizer.encode(render(tokenizer, prompt['messages']), add_special_tokens=False)
        for vector, key in zip(vectors, keys):
            plain = json.dumps({'counts': vector})
            for style, text in [('json', plain), ('fenced_json', f'```json\n{plain}\n```')]:
                response_ids = tokenizer.encode(text, add_special_tokens=False) + [eos]
                verification = verify_response(text, prompt)
                if bool(verification['reward']) != (key in valid):
                    raise ValueError('canonical verifier disagreement')
                rows.append({'prompt_id': prompt['prompt_id'], 'panel_index': prompt['panel_index'],
                             'split': prompt['split'], 'mode_key': key, 'is_valid': int(key in valid),
                             'style': style, 'response': text, 'response_length': len(response_ids),
                             'eos_token_id': eos, 'oracle_mode_count': len(valid),
                             '_prompt_ids': prompt_ids, '_response_ids': response_ids})
    return rows


def make_batch(rows, pad_id, device):
    import torch
    lengths = [len(row['_prompt_ids']) + len(row['_response_ids']) for row in rows]
    input_ids = torch.full((len(rows), max(lengths)), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    flat_positions, targets, owners = [], [], []
    for i, row in enumerate(rows):
        prompt, response = row['_prompt_ids'], row['_response_ids']
        if not prompt or not response:
            raise ValueError('empty prompt or response')
        tokens = prompt + response
        input_ids[i, :len(tokens)] = torch.tensor(tokens, device=device)
        attention_mask[i, :len(tokens)] = 1
        flat_positions.extend(i * max(lengths) + j for j in range(len(prompt) - 1, len(tokens) - 1))
        targets.extend(response)
        owners.extend([i] * len(response))
    return (input_ids, attention_mask,
            torch.tensor(flat_positions, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device),
            torch.tensor(owners, dtype=torch.long, device=device))


def response_hidden(model, batch):
    input_ids, attention_mask, positions, targets, owners = batch
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False, return_dict=True).last_hidden_state
    return hidden.reshape(-1, hidden.shape[-1])[positions], targets, owners


def projected_statistics(model, hidden, targets, need_entropy=False):
    import torch
    logits = model.lm_head(hidden).float()
    log_z = torch.logsumexp(logits, dim=-1)
    logp = logits.gather(-1, targets[:, None]).squeeze(-1) - log_z
    entropy = None
    if need_entropy:
        probabilities = torch.softmax(logits, dim=-1)
        entropy = log_z - (probabilities * logits).sum(dim=-1)
    return logp, entropy


def crosscheck_response_scorer(model, rows, pad_id):
    """Independent full-CausalLM FP32 forward catches shifts and masking errors."""
    import torch
    model.eval()
    batch = make_batch(rows[:2], pad_id, model.device)
    input_ids, attention_mask, positions, targets, _ = batch
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        full = model(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=False, return_dict=True).logits
        selected = full.reshape(-1, full.shape[-1])[positions].float()
        reference = selected.gather(-1, targets[:, None]).squeeze(-1) - selected.logsumexp(-1)
        del full, selected
        hidden, selected_targets, _ = response_hidden(model, batch)
        actual, _ = projected_statistics(model, hidden, selected_targets)
        error = float((actual - reference).abs().max())
        sequence_error = float((actual - reference).sum().abs())
    if error > 1e-3:
        raise AssertionError(f'response-head/full-model logprob mismatch {error}')
    return {'full_model_reference_max_token_error_fp32': error,
            'full_model_reference_total_logprob_error_fp32': sequence_error}


def score_rows(model, rows, pad_id, batch_size, chunk_size, phase, score_dtype='float32'):
    import torch
    results = []
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                              enabled=score_dtype == 'bfloat16'):
        for start in range(0, len(rows), batch_size):
            chunk_rows = rows[start:start + batch_size]
            batch = make_batch(chunk_rows, pad_id, model.device)
            hidden, targets, owners = response_hidden(model, batch)
            sums = torch.zeros(len(chunk_rows), dtype=torch.float64, device=model.device)
            for pos in range(0, len(targets), chunk_size):
                lp, _ = projected_statistics(model, hidden[pos:pos + chunk_size], targets[pos:pos + chunk_size])
                sums.index_add_(0, owners[pos:pos + chunk_size], lp.double())
            values = sums.cpu().tolist()
            if not np.isfinite(values).all():
                raise ValueError('non-finite response score')
            for row, value in zip(chunk_rows, values):
                results.append((value, value / len(row['_response_ids'])))
            if start == 0 or (start + batch_size) % 512 == 0 or start + batch_size >= len(rows):
                print(json.dumps({'phase': phase, 'scored': min(start + batch_size, len(rows)),
                                  'total': len(rows)}), flush=True)
    return np.asarray(results)


def update_once(model, rows, post_advantages, variant, pad_id, batch_size, chunk_size,
                learning_rate=1e-6):
    import torch
    from torch.utils.checkpoint import checkpoint
    # Qwen activates layer checkpointing only in training mode. There is no
    # dropout in these audited models, so this does not change the policy.
    if float(getattr(model.config, 'attention_dropout', 0.)) != 0.:
        raise ValueError('nonzero attention dropout is outside this deterministic audit')
    if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules()):
        raise ValueError('nonzero module dropout is outside this deterministic audit')
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.zero_grad(set_to_none=True)
    total_tokens = sum(len(row['_response_ids']) for row in rows)
    coefficient = .001 if variant == 'grpo_h' else 0.
    loss_sum = 0.
    entropy_sum = 0.
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start:start + batch_size]
        batch = make_batch(chunk_rows, pad_id, model.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden, targets, owners = response_hidden(model, batch)
            seq_adv = torch.tensor(post_advantages[start:start + batch_size],
                                   dtype=torch.float32, device=model.device)
            token_adv = seq_adv[owners]

            def chunk_objective(h, target, advantage):
                lp, entropy = projected_statistics(model, h, target, coefficient > 0)
                # d exp(lp-stopgrad(lp))/d theta at ratio=1 is d lp/d theta.
                policy = -(advantage * torch.exp(lp - lp.detach())).sum()
                ent = entropy.sum() if entropy is not None else policy.new_zeros(())
                return torch.stack(((policy - coefficient * ent) / total_tokens,
                                    ent.detach() / total_tokens))

            losses = [checkpoint(chunk_objective, hidden[pos:pos + chunk_size],
                                 targets[pos:pos + chunk_size], token_adv[pos:pos + chunk_size],
                                 use_reentrant=False)
                      for pos in range(0, len(targets), chunk_size)]
            values = torch.stack(losses).sum(dim=0)
        values[0].backward()
        loss_sum += float(values[0].detach())
        entropy_sum += float(values[1].detach())
        if start == 0 or (start + batch_size) % 64 == 0 or start + batch_size >= len(rows):
            print(json.dumps({'phase': 'accumulate_gradient',
                              'responses': min(start + batch_size, len(rows)), 'total': len(rows)}), flush=True)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    if not torch.isfinite(grad_norm):
        raise ValueError('non-finite gradient norm')
    cpu_optimizer_step(model, learning_rate)
    model.gradient_checkpointing_disable()
    torch.cuda.empty_cache()
    return {'gradient_norm_before_clip': float(grad_norm), 'clip_factor': min(1., 1. / (float(grad_norm) + 1e-6)),
            'objective_at_ratio_one': loss_sum, 'entropy_response_token_mean': entropy_sum if coefficient else None,
            'entropy_coefficient': coefficient, 'update_response_tokens': total_tokens,
            'optimizer': {'name': 'AdamW', 'fresh_state': True, 'state_device': 'cpu', 'foreach': False, 'lr': learning_rate, 'eps': 1e-8,
                          'betas': [.9, .999], 'weight_decay': .1, 'clip_norm': 1.,
                          'optimizer_steps': 1, 'loss_reduction': 'global-response-token-mean',
                          'ppo_ratio': 1., 'kl_coefficient': 0.}}


def cpu_optimizer_step(model, learning_rate):
    """Same FP32 AdamW math, one parameter at a time on CPU.

    Fresh moments are independent per parameter: staging one parameter and its
    gradient at a time avoids retaining 4x7B optimizer arrays on any device.
    GPU global gradient clipping has already been applied. foreach=False makes
    this the explicit scalar/tensor AdamW implementation for every rule.
    """
    import torch
    start = time.time()
    count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        host = torch.nn.Parameter(parameter.detach().to('cpu', copy=True))
        host.grad = parameter.grad.detach().to('cpu', copy=True)
        optimizer = torch.optim.AdamW([host], lr=learning_rate, eps=1e-8,
                                     betas=(.9,.999), weight_decay=.1, foreach=False)
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(host.to(device=parameter.device))
        parameter.grad = None
        del optimizer, host
        count += 1
    print(json.dumps({'phase':'cpu_adamw_complete','parameter_tensors':count,
                      'elapsed_seconds':time.time()-start}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('model', 'train-bank', 'heldout-bank', 'data-root', 'output', 'start-label'):
        parser.add_argument('--' + argument, required=True)
    parser.add_argument('--variant', choices=('grpo', 'eppo', 'grpo_h'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--canary', action='store_true')
    parser.add_argument('--score-only', action='store_true', help='PRE-only calibration; no optimizer or post model')
    parser.add_argument('--bank-k', type=int, choices=(64,128), required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--logit-chunk-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--score-dtype', choices=('float32', 'bfloat16'), default='bfloat16')
    parser.add_argument('--rarity-score-dtype', choices=('float32', 'bfloat16'), default='bfloat16',
                        help='Numerical policy used for the old-logprob rarity score entering the actor update')
    parser.add_argument('--bfloat16-sensitivity', action=argparse.BooleanOptionalAction, default=False,
                        help='Also score identical pre/post events in BF16 when primary scoring is FP32')
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('The one-step model update requires an allocated CUDA GPU.')
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise RuntimeError('completed output already exists; use a distinct fork directory')
    if (output / 'model').exists():
        raise RuntimeError('model output already exists without completion; inspect before retry')
    if args.variant != 'grpo' and args.score_only:
        raise ValueError('Calibration-only records use the unshaped GRPO reference rule')
    torch.set_num_threads(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    train_prompts, train = load_panel(args.data_root, 'train', args.train_bank, tokenizer, args.canary, args.bank_k)
    heldout_prompts, heldout = load_panel(args.data_root, 'heldout', args.heldout_bank, tokenizer, args.canary, args.bank_k)
    if {p['prompt_id'] for p in train_prompts}.intersection(p['prompt_id'] for p in heldout_prompts):
        raise ValueError('train and heldout prompts overlap')
    canonical = canonical_rows(train_prompts + heldout_prompts, tokenizer)
    all_rows = train + heldout
    update_indices = [i for i, row in enumerate(all_rows) if row['in_update']]
    train_rows = [all_rows[i] for i in update_indices]
    group_ids = [row['prompt_id'] for row in train_rows]
    if set(Counter(group_ids).values()) != {16}:
        raise ValueError('update groups must have exactly G=16')
    provenance = {'args': vars(args), 'input_hashes': {
        'train_bank': sha256_file(args.train_bank), 'heldout_bank': sha256_file(args.heldout_bank),
        'train_prompts': sha256_file(Path(args.data_root) / 'train_prompts.jsonl'),
        'heldout_prompts': sha256_file(Path(args.data_root) / 'heldout_prompts.jsonl'),
        'core_algos': sha256_file(SOURCE), 'worker': sha256_file(__file__),
        'verifier_wrapper': sha256_file(ROOT / 'task_utils.py'),
        'detokenization': sha256_file(ROOT / 'detokenization.py'),
        'verifier': sha256_file(ROOT / 'source_snapshot/coin_change_final_verifier.py'),
        'train_bank_receipt': sha256_file(str(args.train_bank) + '.complete.json'),
        'heldout_bank_receipt': sha256_file(str(args.heldout_bank) + '.complete.json')},
        'tokenizer': {'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                      'chat_template_sha256': hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
                      'enable_thinking': False},
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__},
        'determinism': {'torch_deterministic_algorithms': True,
                        'cudnn_benchmark': False, 'cudnn_deterministic': True,
                        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
                        'allow_tf32': False},
        'precision': {'model_parameters': 'float32', 'input_standardization': 'BF16 checkpoint weights promoted to FP32', 'likelihood_scoring': args.score_dtype,
                      'actor_update_autocast': 'bfloat16', 'advantage_reconstruction': 'float64',
                      'rarity_logprob_view': args.rarity_score_dtype,
                      'bfloat16_likelihood_sensitivity': bool(args.bfloat16_sensitivity and args.score_dtype == 'float32')},
        'checkpoint_step': 0 if args.start_label == 'base' else 900,
        'task_variant': 'paper2048_direct',
        'historical_optimizer_available': False,
        'primary_probability_estimator': 'direct_fresh_sampling_not_importance',
        'score_only': args.score_only,
        'original_trainer_replay': False,
        'canary': args.canary, 'selection': 'first16 sample indices in every training prompt; no reward selection',
        'canonical_event': 'plain JSON+EOS and fenced JSON+EOS; restricted exact-string probability, not full semantic mass'}
    model_files = sorted(Path(args.model).glob('*.safetensors'))
    model_files += sorted(Path(args.model).glob('*.bin')) if not model_files else []
    if not model_files:
        raise ValueError('no model weight files found')
    provenance['model_hashes'] = {p.name: sha256_file(p) for p in model_files}
    model_config_paths = [p for p in Path(args.model).iterdir()
                          if p.is_file() and p.suffix in ('.json', '.jinja') and 'index' not in p.name]
    provenance['model_config_hashes'] = {p.name: sha256_file(p) for p in sorted(model_config_paths)}
    atomic_json(output / 'started.json', provenance)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa', trust_remote_code=False)
    if model.config.model_type not in ('qwen3', 'qwen2'):
        raise ValueError('response-only head worker audited for Qwen2/Qwen3 architecture')
    model.to(device='cuda', dtype=torch.float32)
    provenance['hardware'] = {'gpu': torch.cuda.get_device_name(),
        'node': os.uname().nodename, 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
        'slurm_step_id': os.getenv('SLURM_STEP_ID')}
    model.config.use_cache = False
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    scorer_check = crosscheck_response_scorer(model, all_rows, pad_id) if args.canary else {}
    pre = score_rows(model, all_rows, pad_id, args.batch_size, args.logit_chunk_size,
                     'responses_pre', args.score_dtype)
    canonical_pre = score_rows(model, canonical, pad_id, args.batch_size, args.logit_chunk_size,
                               'canonical_pre', args.score_dtype)
    auxiliary_pre = auxiliary_canonical_pre = None
    if args.bfloat16_sensitivity and args.score_dtype == 'float32':
        auxiliary_pre = score_rows(model, all_rows, pad_id, args.batch_size, args.logit_chunk_size,
                                   'responses_pre_bfloat16_sensitivity', 'bfloat16')
        auxiliary_canonical_pre = score_rows(model, canonical, pad_id, args.batch_size, args.logit_chunk_size,
                                             'canonical_pre_bfloat16_sensitivity', 'bfloat16')
    rewards = np.asarray([float(row['reward']) for row in train_rows])
    if args.rarity_score_dtype == args.score_dtype:
        update_means = pre[update_indices, 1]
    elif args.rarity_score_dtype == 'bfloat16' and auxiliary_pre is not None:
        update_means = auxiliary_pre[update_indices, 1]
    else:
        update_means = score_rows(model, train_rows, pad_id, args.batch_size, args.logit_chunk_size,
                                 'update_rarity_' + args.rarity_score_dtype, args.rarity_score_dtype)[:, 1]
    shaping = advantages_numpy(rewards, update_means, group_ids, args.variant)
    shaping['rarity_mean_logprob'] = update_means
    parity_error = crosscheck_active_core(rewards, update_means, group_ids, args.variant)
    # Persist pre-update scores before any optimization, allowing lossless failure diagnosis.
    atomic_csv(output / 'responses_pre.csv', [
        {'sample_id': row['sample_id'], 'prompt_id': row['prompt_id'], 'split': row['split'],
         'sample_index': row['sample_index'], 'sequence_logprob': float(lp[0]), 'mean_logprob': float(lp[1])}
        for row, lp in zip(all_rows, pre)])
    atomic_json(output / 'update_capture.json', {
        'start_label': args.start_label, 'variant': args.variant, 'seed': args.seed,
        'bank_hashes': provenance['input_hashes'],
        'records': [{
            'sample_id': row['sample_id'], 'prompt_id': row['prompt_id'],
            'sample_index': row['sample_index'], 'reward': bool(row['reward']),
            'mode_key': row.get('mode_key'), 'response_length': row['response_length'],
            'prompt_token_ids': row['_prompt_ids'], 'response_token_ids': row['_response_ids'],
            **{name: float(values[j]) for name, values in shaping.items()}
        } for j, row in enumerate(train_rows)]})
    if args.score_only:
        atomic_csv(output / 'response_scores.csv', [
            {**{key: row.get(key, '') for key in
                 ('sample_id','prompt_id','panel_index','split','sample_index','reward','mode_key',
                  'response_length','format_class','literal_valid_json','fenced_valid_json',
                  'counts_extractable','error','finish_reason','prompt_sha256','in_update',
                  'detokenization_verification')},
             'pre_sequence_logprob': float(lp[0]), 'pre_mean_logprob': float(lp[1]),
             'start_label': args.start_label, 'variant': 'grpo',
             **{name: float(values[update_indices.index(i)]) if i in update_indices else ''
                for name, values in shaping.items()}}
            for i,(row,lp) in enumerate(zip(all_rows,pre))])
        atomic_csv(output / 'canonical_scores.csv', [
            {**{key:val for key,val in row.items() if not key.startswith('_')},
             'pre_sequence_logprob': float(lp[0]), 'pre_mean_logprob': float(lp[1])}
            for row,lp in zip(canonical,canonical_pre)])
        atomic_json(output / 'complete.json', {
            **provenance, 'status':'complete','elapsed_seconds':time.time()-started,
            'counts':{'train_prompts':len(train_prompts),'heldout_prompts':len(heldout_prompts),
                      'train_responses':len(train),'heldout_responses':len(heldout),
                      'update_responses':0,'scored_update_candidate_responses':len(train_rows)},
            'diagnostics':{**scorer_check,'active_core_max_abs_error':parity_error,'optimizer_steps':0},
            'output_hashes':{name:sha256_file(output/name) for name in
                ('response_scores.csv','canonical_scores.csv','update_capture.json')}})
        return
    parameter_before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    diagnostics = update_once(model, train_rows, shaping['A_post'], args.variant, pad_id,
                              args.batch_size, args.logit_chunk_size, args.learning_rate)
    changed = elements = changed_bf16 = 0
    max_abs_parameter_delta = 0.
    for name, p in model.named_parameters():
        previous = parameter_before.pop(name)
        current = p.detach().cpu()
        changed += int(torch.count_nonzero(previous != current))
        changed_bf16 += int(torch.count_nonzero(previous.bfloat16() != current.bfloat16()))
        elements += current.numel()
        max_abs_parameter_delta = max(max_abs_parameter_delta, float((current - previous).abs().max()))
    diagnostics['parameter_change'] = dict(elements=elements, changed_fp32=changed,
        changed_after_bf16_cast=changed_bf16, max_abs=max_abs_parameter_delta)
    del parameter_before, previous, current
    post = score_rows(model, all_rows, pad_id, args.batch_size, args.logit_chunk_size,
                      'responses_post', args.score_dtype)
    canonical_post = score_rows(model, canonical, pad_id, args.batch_size, args.logit_chunk_size,
                                'canonical_post', args.score_dtype)
    auxiliary_post = auxiliary_canonical_post = None
    if auxiliary_pre is not None:
        auxiliary_post = score_rows(model, all_rows, pad_id, args.batch_size, args.logit_chunk_size,
                                    'responses_post_bfloat16_sensitivity', 'bfloat16')
        auxiliary_canonical_post = score_rows(model, canonical, pad_id, args.batch_size, args.logit_chunk_size,
                                              'canonical_post_bfloat16_sensitivity', 'bfloat16')
    update_map = {index: j for j, index in enumerate(update_indices)}
    response_output = []
    for i, row in enumerate(all_rows):
        result = {key: row.get(key, '') for key in
                  ('sample_id', 'prompt_id', 'panel_index', 'split', 'sample_index', 'reward', 'mode_key',
                   'response_length', 'format_class', 'literal_valid_json', 'fenced_valid_json',
                   'counts_extractable', 'extracted_vector', 'error', 'finish_reason', 'prompt_sha256', 'in_update',
                   'detokenization_verification',
                   'initial_sequence_logprob_generation', 'initial_mean_logprob_generation')}
        result.update(variant=args.variant, start_label=args.start_label, seed=args.seed,
                      pre_sequence_logprob=pre[i, 0], post_sequence_logprob=post[i, 0],
                      pre_mean_logprob=pre[i, 1], post_mean_logprob=post[i, 1],
                      delta_sequence_logprob=post[i, 0] - pre[i, 0],
                      delta_mean_logprob=post[i, 1] - pre[i, 1])
        if auxiliary_pre is not None:
            result.update(pre_bfloat16_sequence_logprob=auxiliary_pre[i, 0],
                          post_bfloat16_sequence_logprob=auxiliary_post[i, 0],
                          delta_bfloat16_sequence_logprob=auxiliary_post[i, 0] - auxiliary_pre[i, 0])
        for name, values in shaping.items():
            result[name] = float(values[update_map[i]]) if i in update_map else ''
        response_output.append(result)
    canonical_output = []
    for i, (row, before, after) in enumerate(zip(canonical, canonical_pre, canonical_post)):
        result = {key: val for key, val in row.items() if not key.startswith('_')}
        result.update(variant=args.variant, start_label=args.start_label, seed=args.seed,
                      pre_sequence_logprob=before[0], post_sequence_logprob=after[0],
                      pre_mean_logprob=before[1], post_mean_logprob=after[1],
                      delta_sequence_logprob=after[0] - before[0], delta_mean_logprob=after[1] - before[1])
        if auxiliary_canonical_pre is not None:
            result.update(pre_bfloat16_sequence_logprob=auxiliary_canonical_pre[i, 0],
                          post_bfloat16_sequence_logprob=auxiliary_canonical_post[i, 0],
                          delta_bfloat16_sequence_logprob=auxiliary_canonical_post[i, 0] - auxiliary_canonical_pre[i, 0])
        canonical_output.append(result)
    atomic_csv(output / 'response_scores.csv', response_output)
    atomic_csv(output / 'canonical_scores.csv', canonical_output)
    model.save_pretrained(output / 'model', safe_serialization=True, max_shard_size='4GB')
    tokenizer.save_pretrained(output / 'model')
    saved_model_hashes = {str(path.relative_to(output)): sha256_file(path)
                          for path in sorted((output / 'model').iterdir()) if path.is_file()}
    group_array = np.asarray(group_ids)
    mass_errors = [abs(np.abs(shaping['A_pre'][group_array == group]).sum() -
                       np.abs(shaping['A_post'][group_array == group]).sum())
                   for group in np.unique(group_array)]
    diagnostics.update(**scorer_check, active_core_max_abs_error=parity_error,
                       mean_advantage_pre=float(shaping['A_pre'].mean()),
                       mean_advantage_post=float(shaping['A_post'].mean()),
                       absolute_mass_pre=float(np.abs(shaping['A_pre']).sum()),
                       absolute_mass_post=float(np.abs(shaping['A_post']).sum()),
                       maximum_group_absolute_mass_change=max(mass_errors),
                       mixed_group_fraction=float(np.mean([0 < rewards[group_array == g].sum() < 16
                                                           for g in np.unique(group_array)])),
                       response_drift_max_abs=float(np.max(np.abs(post - pre))),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if auxiliary_pre is not None:
        for name, before, after, low_before, low_after in (
            ('response', pre, post, auxiliary_pre, auxiliary_post),
            ('canonical', canonical_pre, canonical_post, auxiliary_canonical_pre, auxiliary_canonical_post),
        ):
            discrepancy = ((after - before) - (low_after - low_before))[:, 0]
            diagnostics[name + '_fp32_vs_bf16_drift_max_abs'] = float(np.max(np.abs(discrepancy)))
            diagnostics[name + '_fp32_vs_bf16_drift_mean_abs'] = float(np.mean(np.abs(discrepancy)))
    if args.variant in ('grpo', 'eppo', 'grpo_h') and max(mass_errors) > 1e-6:
        raise AssertionError('group absolute sequence advantage mass changed unexpectedly')
    payload = {**provenance, 'status': 'complete', 'elapsed_seconds': time.time() - started,
               'counts': {'train_prompts': len(train_prompts), 'heldout_prompts': len(heldout_prompts),
                          'train_responses': len(train), 'heldout_responses': len(heldout),
                          'update_responses': len(train_rows), 'canonical_strings': len(canonical)},
               'diagnostics': diagnostics,
               'saved_model_hashes': saved_model_hashes,
               'output_hashes': {name: sha256_file(output / name)
                                 for name in ('response_scores.csv', 'canonical_scores.csv', 'update_capture.json')}}
    atomic_json(output / 'complete.json', payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
