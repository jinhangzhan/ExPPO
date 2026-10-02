"""Independent CPU-only regression audit; no real checkpoint is loaded.

Run with CUDA_VISIBLE_DEVICES='' python -B -m unittest test_update_worker -v.
These tests validate FP32 gradient logic, not CUDA/BF16 numerical equivalence.
"""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

if any(importlib.util.find_spec(name) is None for name in ('numpy', 'torch', 'transformers', 'tokenizers')):
    raise unittest.SkipTest('The continuation update test needs its optional model environment.')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/continuation'))
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import update_worker as worker

torch.set_num_threads(1)


def toy_model():
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=43, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
                        attention_dropout=0., pad_token_id=0, bos_token_id=1,
                        eos_token_id=2, tie_word_embeddings=True)
    config._attn_implementation = 'eager'
    return Qwen2ForCausalLM(config).float().eval()


def toy_rows(n=16):
    return [dict(_prompt_ids=[1, 4, 7], _response_ids=[10 + i % 5] * (1 + i % 3) + [2])
            for i in range(n)]


def independent_full_loss(model, rows, advantages, variant):
    """Full LM projection and explicit per-response indexing, no worker scorer."""
    token_count = sum(len(r['_response_ids']) for r in rows)
    loss = torch.zeros(())
    for r, advantage in zip(rows, advantages):
        ids = torch.tensor([r['_prompt_ids'] + r['_response_ids']])
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]
        first = len(r['_prompt_ids']) - 1
        response_logits = logits[first:first + len(r['_response_ids'])].float()
        log_probs = response_logits.log_softmax(-1)
        targets = torch.tensor(r['_response_ids'])
        lp = log_probs.gather(1, targets[:, None]).squeeze(1)
        loss = loss - float(advantage) * lp.sum() / token_count
        if variant == 'grpo_h':
            entropy = -(log_probs.exp() * log_probs).sum(-1)
            loss = loss - .001 * entropy.sum() / token_count
    return loss


class WorkerTests(unittest.TestCase):
    def test_active_core_shaping_all_groups(self):
        rewards=np.array([0.] * 16 + [1.] * 16 + [0., 1.] * 8 + [0.] * 15 + [1.])
        means=np.linspace(-.05, -1.8, len(rewards))
        groups=np.repeat(np.arange(4), 16)
        for variant in ('grpo', 'eppo', 'grpo_h'):
            self.assertLess(worker.crosscheck_active_core(rewards,means,groups,variant),2e-6)
            out=worker.advantages_numpy(rewards,means,groups,variant)
            np.testing.assert_array_equal(out['A_post'][:32],0)
            for group in range(4):
                take=groups==group
                self.assertAlmostEqual(np.abs(out['A_pre'][take]).sum(),np.abs(out['A_post'][take]).sum(),places=7)

    def test_full_logits_response_positions_and_score(self):
        model=toy_model(); rows=toy_rows(4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            actual=worker.score_rows(model,rows,0,3,2,'cpu_test','float32')
        expected=[]
        with torch.no_grad():
            for r in rows:
                ids=torch.tensor([r['_prompt_ids']+r['_response_ids']])
                logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0]
                first=len(r['_prompt_ids'])-1
                lp=logits[first:first+len(r['_response_ids'])].log_softmax(-1)
                value=lp.gather(1,torch.tensor(r['_response_ids'])[:,None]).sum().item()
                expected.append((value,value/len(r['_response_ids'])))
        np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=0)

    def test_actual_update_matches_independent_full_model_gradient(self):
        rows=toy_rows()
        means=np.linspace(-.1,-1.3,16)
        rewards=np.array([0.,1.] * 8)
        groups=np.zeros(16)
        real_adamw=torch.optim.AdamW
        for variant in ('grpo','eppo','grpo_h'):
            with self.subTest(variant=variant):
                actual=toy_model(); expected=deepcopy(actual)
                advantages=worker.advantages_numpy(rewards,means,groups,variant)['A_post']
                expected_optimizer=real_adamw(expected.parameters(),lr=1e-6,eps=1e-8,
                                             betas=(.9,.999),weight_decay=.1)
                independent_full_loss(expected,rows,advantages,variant).backward()
                norm=torch.nn.utils.clip_grad_norm_(expected.parameters(),1.)
                expected_grads=[p.grad.detach().clone() for p in expected.parameters()]
                expected_optimizer.step()
                captured=[]
                class RecordingAdamW(real_adamw):
                    def step(self,closure=None):
                        captured.extend(p.grad.detach().clone() for p in self.param_groups[0]['params'])
                        return super().step(closure)
                with patch.object(torch.optim,'AdamW',RecordingAdamW),warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    diagnostics=worker.update_once(actual,rows,advantages,variant,0,4,3,1e-6)
                self.assertAlmostEqual(diagnostics['gradient_norm_before_clip'],float(norm),places=5)
                for got,want in zip(captured,expected_grads):
                    torch.testing.assert_close(got,want,rtol=3e-4,atol=3e-7)
                for got,want in zip(actual.parameters(),expected.parameters()):
                    torch.testing.assert_close(got,want,rtol=1e-5,atol=1.2e-7)
                self.assertTrue(diagnostics['optimizer']['fresh_state'])

    def test_parameterwise_cpu_adam_is_identical_to_global_first_step(self):
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(721)
                self.a=torch.nn.Parameter(torch.randn(13,7))
                self.b=torch.nn.Parameter(torch.randn(19))
                self.zero=torch.nn.Parameter(torch.randn(5))
                self.missing=torch.nn.Parameter(torch.randn(3))
        actual=Small(); reference=deepcopy(actual)
        for i,(a,b) in enumerate(zip(actual.parameters(),reference.parameters())):
            if i==3:
                a.grad=b.grad=None
            else:
                a.grad=torch.zeros_like(a) if i==2 else torch.full_like(a,float(i+1))
                b.grad=a.grad.clone()
        torch.nn.utils.clip_grad_norm_(actual.parameters(),1.)
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
        optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-6,betas=(.9,.999),
                                    eps=1e-8,weight_decay=.1,foreach=False)
        optimizer.step()
        worker.cpu_optimizer_step(actual,1e-6)
        for a,b in zip(actual.parameters(),reference.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(actual.missing,Small().missing,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
