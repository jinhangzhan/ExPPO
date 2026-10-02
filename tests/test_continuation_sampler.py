#!/usr/bin/env python3
"""CPU-only fake-backend tests. Real vLLM logprobs/VRAM require the GPU canary."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/continuation'))
import generate_direct as gd


def prompt(index=0):
    return dict(prompt_id=f'q{index}', panel_index=index, task='coin_change',
                messages=[{'role': 'user', 'content': 'Find counts.'}],
                payload=dict(denominations=[1, 2, 3, 4], target=2, max_count=6),
                valid_mode_keys=['2,0,0,0', '0,1,0,0'])


def completion(index=0, text='{"counts":[2,0,0,0]}'):
    return SimpleNamespace(index=index, token_ids=[10, 11, 0], text=text,
                           logprobs=[{10: SimpleNamespace(logprob=-.1)},
                                     {11: SimpleNamespace(logprob=-.2)},
                                     {0: SimpleNamespace(logprob=-.3)}],
                           cumulative_logprob=-.6, finish_reason='stop', stop_reason=None)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        if kwargs != dict(tokenize=False, add_generation_prompt=True):
            raise ValueError('Template kwargs changed.')
        return 'CHAT:' + messages[0]['content']

    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise ValueError('Unexpected special token insertion.')
        return [1, 2]


def rows_for(p=None):
    p = p or prompt()
    text, ids = gd.render_prompt(FakeTokenizer(), p)
    request = SimpleNamespace(prompt_token_ids=ids,
                              outputs=[completion(1, '{"counts":[0,1,0,0]}'), completion(0)])
    rows = gd.make_rows(p, text, ids, request, 'pre', 2, gd.SEED, FakeTokenizer(),
                        lambda *args: 'batch_exact')
    return p, text, ids, rows


def contract_for(p):
    return dict(tag='pre', canary=False, identity={}, prompts_sha256='prompt-hash',
                rendered_prompt_ids_sha256='render-hash', sampling=gd.sampling_config(2, gd.SEED, 1),
                decoding_contract_sha256='decoder-hash', software={}, decoding={},
                source_sha256={'generate_direct.py': 'generator-hash', 'task_utils.py': 'utility-hash',
                               'source_snapshot/coin_change_final_verifier.py': 'verifier-hash'})


def seal_chunk(parts, p, rows, contract):
    path, marker = gd.chunk_paths(parts, p)
    gd.write_new_lines(path, (json.dumps(row) + '\n' for row in rows))
    gd.write_new_json(marker, dict(status='complete', contract_sha256=gd.digest(contract),
                       outcomes_sha256=gd.sha256(path), response_count=len(rows), prompt_id=p['prompt_id'],
                       runtime={'gpu_names': ['FAKE CPU TEST'], 'slurm_job_id': 'test'},
                       logprob_crosscheck_max_abs_error=0.0))
    return path, marker


class DirectSamplerTests(unittest.TestCase):
    def test_neutral_fixed_decoding(self):
        config = gd.sampling_config(128, gd.SEED, 1)
        self.assertEqual((config['max_new_tokens'], config['max_model_len']), (2048, 2560))
        self.assertEqual((config['temperature'], config['top_p'], config['top_k']), (1., 1., 0))
        self.assertEqual(config['samples_per_prompt'], 128)
        self.assertFalse(config['ignore_eos'])
        self.assertEqual(config['repetition_penalty'], 1.)

    def test_prompt_split_and_canary_are_outcome_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'prompts.jsonl'
            gd.write_new_lines(path, (json.dumps(prompt(i)) + '\n' for i in range(32, 64)))
            self.assertEqual([p['panel_index'] for p in gd.load_prompts(path, True)], [32, 33])
            self.assertEqual(len(gd.load_prompts(path)), 32)
            bad = Path(temporary) / 'bad.jsonl'
            gd.write_new_lines(bad, (json.dumps(prompt(i)) + '\n' for i in range(31)))
            with self.assertRaises(ValueError):
                gd.load_prompts(bad)

    def test_exact_default_template_and_sorted_response_ids(self):
        p, text, ids, rows = rows_for()
        self.assertEqual(text, 'CHAT:Find counts.')
        self.assertEqual([row['sample_index'] for row in rows], [0, 1])
        self.assertEqual(rows[0]['response_token_ids'], [10, 11, 0])
        self.assertEqual(rows[0]['response_length'], 3)  # EOS retained.
        gd.validate_rows(rows, p, text, ids, 'pre', 2, gd.SEED)

    def test_logprob_zero_contains_sampled_tokens(self):
        value, error = gd.checked_logprob(completion())
        self.assertAlmostEqual(value, -.6)
        self.assertLess(error, 1e-12)

    def test_installed_vllm_flat_logprobs_zero_backend_container(self):
        package = importlib.util.find_spec('vllm')
        if package is None:
            self.skipTest('Optional vLLM installation unavailable.')
        path = Path(package.origin).parent / 'logprobs.py'
        if not path.exists():
            self.skipTest('Installed vLLM has no FlatLogprobs module.')
        spec = importlib.util.spec_from_file_location('_sampler_test_vllm_logprobs', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        flat = module.FlatLogprobs()
        for token, logprob in [(10, -.1), (11, -.2), (0, -.3)]:
            module.append_logprobs_for_next_position(flat, [token], [logprob], [None], rank=1, num_logprobs=0)
        item = completion()
        item.logprobs = flat
        self.assertEqual(len(flat), 3)
        self.assertAlmostEqual(gd.checked_logprob(item)[0], -.6)

    def test_logprob_rejects_missing_nonfinite_or_wrong_cumulative(self):
        for change in ({'logprobs': None}, {'cumulative_logprob': None},
                       {'cumulative_logprob': float('nan')}, {'cumulative_logprob': -.5},
                       {'logprobs': [{}, {}, {}]}):
            with self.subTest(change=change):
                item = completion()
                for key, value in change.items():
                    setattr(item, key, value)
                with self.assertRaises(ValueError):
                    gd.checked_logprob(item)
        item = completion()
        item.logprobs[0][10].logprob = float('nan')
        with self.assertRaises(ValueError):
            gd.checked_logprob(item)

    def test_literal_final_field_not_first_or_repaired(self):
        p = prompt()
        item = completion(text='{"counts":[2,0,0,0]} then {"counts":"invalid"}')
        request = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[item])
        rows = gd.make_rows(p, 'CHAT', [1, 2], request, 'pre', 1, gd.SEED, None, lambda *args: 'batch_exact')
        self.assertFalse(rows[0]['reward'])
        self.assertIsNone(rows[0]['mode_key'])
        self.assertIsNone(rows[0]['final_counts'])

    def test_short_context_stop_and_wrong_prompt_are_rejected(self):
        p = prompt()
        item = completion()
        item.finish_reason = 'length'
        request = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[item])
        with self.assertRaisesRegex(ValueError, 'Context limit'):
            gd.make_rows(p, 'CHAT', [1, 2], request, 'pre', 1, gd.SEED, None, lambda *args: 'batch_exact')
        item.finish_reason = 'stop'
        request.prompt_token_ids = [2]
        with self.assertRaises(ValueError):
            gd.make_rows(p, 'CHAT', [1, 2], request, 'pre', 1, gd.SEED, None, lambda *args: 'batch_exact')

    def test_row_revalidation_rejects_reward_and_length_corruption(self):
        p, text, ids, rows = rows_for()
        for field, value in [('reward', False), ('response_length', 9),
                             ('initial_mean_logprob_generation', -99), ('sample_id', 'wrong')]:
            with self.subTest(field=field):
                corrupted = copy.deepcopy(rows)
                corrupted[0][field] = value
                with self.assertRaises(ValueError):
                    gd.validate_rows(corrupted, p, text, ids, 'pre', 2, gd.SEED)

    def test_valid_chunk_resume_and_immutable_final_assembly(self):
        with tempfile.TemporaryDirectory() as temporary:
            parts = Path(temporary) / 'train.jsonl.parts'
            output = Path(temporary) / 'train.jsonl'
            p, text, ids, rows = rows_for()
            contract = contract_for(p)
            self.assertIsNone(gd.check_chunk(parts, p, text, ids, contract))
            seal_chunk(parts, p, rows, contract)
            self.assertEqual(gd.check_chunk(parts, p, text, ids, contract)['response_count'], 2)
            first = gd.assemble(output, parts, [(p, text, ids)], contract)
            stamp = output.stat().st_mtime_ns
            second = gd.assemble(output, parts, [(p, text, ids)], contract)
            self.assertEqual(first, second)
            self.assertEqual(output.stat().st_mtime_ns, stamp)
            self.assertEqual(gd.read_jsonl(output), rows)
            self.assertEqual(first['response_count'], 2)

    def test_unsealed_chunk_blocks_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, text, ids, rows = rows_for()
            path, _ = gd.chunk_paths(temporary, p)
            gd.write_new_lines(path, ['partial evidence\n'])
            with self.assertRaisesRegex(RuntimeError, 'unsealed'):
                gd.check_chunk(temporary, p, text, ids, contract_for(p))
            self.assertEqual(path.read_text(), 'partial evidence\n')

    def test_chunk_contract_and_receipt_hash_corruption_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            p, text, ids, rows = rows_for()
            contract = contract_for(p)
            seal_chunk(temporary, p, rows, contract)
            changed = {**contract, 'tag': 'grpo'}
            with self.assertRaises(ValueError):
                gd.check_chunk(temporary, p, text, ids, changed)

    def test_write_never_replaces_targets_or_partial_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'evidence.json'
            gd.write_new_json(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                gd.write_new_json(path, {'value': 2})
            self.assertEqual(json.loads(path.read_text()), {'value': 1})
            other = Path(temporary) / 'partial.json'
            gd.write_new_lines(str(other) + '.tmp', ['partial'])
            with self.assertRaises(FileExistsError):
                gd.write_new_json(other, {'value': 2})

    def test_sharded_model_identity_and_post_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / 'model'
            for name, value in [('config.json', {'model_type': 'qwen2'}),
                                ('tokenizer_config.json', {}), ('tokenizer.json', {}),
                                ('model.safetensors.index.json', {'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}})]:
                gd.write_new_json(model / name, value)
            gd.write_new_lines(model / 'one.safetensors', ['fake-one'])
            gd.write_new_lines(model / 'two.safetensors', ['fake-two'])
            with self.assertRaisesRegex(ValueError, 'completed worker receipt'):
                gd.model_identity(model)
            hashes = {str(path.relative_to(model.parent)): gd.sha256(path) for path in model.iterdir()}
            gd.write_new_json(model.parent / 'complete.json', {'status': 'complete', 'saved_model_hashes': hashes})
            identity = gd.model_identity(model)
            self.assertEqual(set(identity['model_weight_files_sha256']), {'one.safetensors', 'two.safetensors'})
            self.assertEqual(identity['worker_saved_model_hashes'], hashes)

    def test_exclusive_output_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'bank.jsonl'
            with gd.output_lock(output):
                with self.assertRaisesRegex(RuntimeError, 'Another sampler'):
                    with gd.output_lock(output):
                        self.fail('Nested lock should not be acquired.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
