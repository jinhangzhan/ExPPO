#!/usr/bin/env python3
"""Immutable, prompt-chunked Qwen2.5 direct sampling for the step-900 audit.

Primary outputs contain K=128 responses per paper prompt, not an adaptive
correct-answer pool. A canary uses the first two prompts and K=16 in a separate
output. The direct cap is 2048 generated tokens, including EOS if generated.
Generation uses one prompt/request, seed 20260905 + unchanged panel_index.
No decoded response is repaired, extended, or filtered by correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
from task_utils import ROOT, VERIFIER_NAME, extract_integer_list, read_jsonl, sha256, verify_response

CAP = 2048
CONTEXT = 2560
SEED = 20260905
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_new_json(path, value):
    """Publish once, without replacing even a concurrently created destination."""
    write_new_lines(path, [json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'])


def write_new_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    with temporary.open('x') as destination:
        for line in lines:
            destination.write(line)
        destination.flush()
        os.fsync(destination.fileno())
    os.link(temporary, path)  # Unlike rename, link refuses an existing target.
    temporary.unlink()


def load_prompts(path, canary=False):
    prompts = read_jsonl(path)
    if len(prompts) != 32:
        raise ValueError('Supply one immutable 32-prompt paper split.')
    indices = [prompt['panel_index'] for prompt in prompts]
    if indices not in (list(range(32)), list(range(32, 64))):
        raise ValueError('Expected original ordered panel indices 0:32 or 32:64.')
    if len({prompt['prompt_id'] for prompt in prompts}) != 32:
        raise ValueError('Duplicate paper prompt IDs.')
    if any(prompt.get('task') != 'coin_change' or not prompt.get('messages') for prompt in prompts):
        raise ValueError('Expected the unchanged free-response coin-change messages.')
    return prompts[:2] if canary else prompts


def render_prompt(tokenizer, prompt):
    # Qwen2.5 uses its unmodified default template; no Qwen3 thinking flag.
    text = tokenizer.apply_chat_template(prompt['messages'], tokenize=False, add_generation_prompt=True)
    ids = list(map(int, tokenizer.encode(text, add_special_tokens=False)))
    if not ids or len(ids) + CAP > CONTEXT:
        raise ValueError(f'Prompt does not fit fixed direct cap/context: {prompt["prompt_id"]}')
    return text, ids


def model_identity(model):
    model = Path(model).resolve(strict=True)
    required = ('config.json', 'tokenizer_config.json', 'tokenizer.json')
    for name in required:
        if not (model / name).is_file():
            raise ValueError(f'Missing model asset {model / name}')
    config = json.loads((model / 'config.json').read_text())
    if config.get('model_type') != 'qwen2':
        raise ValueError('This sampler is restricted to Qwen2.5/Qwen2 architecture.')
    weights = None
    index_name = None
    for candidate in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (model / candidate).exists():
            index_name = candidate
            weights = sorted(set(json.loads((model / candidate).read_text())['weight_map'].values()))
            break
    if weights is None:
        weights = [next((name for name in ('model.safetensors', 'pytorch_model.bin')
                         if (model / name).is_file()), '')]
    if not weights or not all(weights):
        raise ValueError(f'No supported model weights under {model}')
    for name in weights:
        weight_path = (model / name).resolve(strict=True)
        if weight_path.parent != model:
            raise ValueError(f'Weight index must resolve to files inside model directory: {name}')
    assets = set(TOKENIZER_FILES) | {'config.json', 'generation_config.json'}
    if index_name:
        assets.add(index_name)
    receipts = {name: sha256(model / name) for name in ('export_complete.json', 'export.json')
                if (model / name).is_file()}
    result = {'model': str(model), 'model_weight_files_sha256': {name: sha256(model / name) for name in weights},
              'model_config_hashes': {name: sha256(model / name) for name in sorted(assets)
                                     if (model / name).is_file()}, 'model_export_receipts_sha256': receipts}
    worker_marker = model.parent / 'complete.json'
    if model.name in ('model', 'post_model'):
        if not worker_marker.is_file():
            raise ValueError('Post-update model lacks its completed worker receipt.')
        receipt = json.loads(worker_marker.read_text())
        saved_hashes = receipt.get('saved_model_hashes', {})
        observed = {str(path.relative_to(model.parent)): sha256(path)
                    for path in sorted(model.iterdir()) if path.is_file()}
        if receipt.get('status') != 'complete' or not saved_hashes or saved_hashes != observed:
            raise ValueError('Post-update saved model differs from completed worker receipt.')
        result.update(worker_completion_sha256=sha256(worker_marker),
                      worker_completion_path=str(worker_marker), worker_saved_model_hashes=saved_hashes)
    return result


def sampling_config(k, seed, tp):
    return dict(samples_per_prompt=k, max_new_tokens=CAP, max_model_len=CONTEXT,
                temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                seed=seed, per_prompt_seed_rule='seed + unchanged panel_index',
                generation_dtype='bfloat16', tensor_parallel_size=tp,
                prompt_batch_size=1, max_num_seqs=128, gpu_memory_utilization=0.80,
                enable_prefix_caching=True, generation_config='auto',
                ignore_eos=False, stop=[], stop_token_ids=[], min_tokens=0,
                skip_special_tokens=True, include_stop_str_in_output=False,
                detokenize=True, logprobs=0, flat_logprobs=True,
                tokenizer_template='unmodified default; add_generation_prompt=True; no thinking override',
                token_cap_interpretation='direct max_tokens; EOS counts if generated; no continuation')


def checked_logprob(completion):
    """Check sampled-token logprobs even with vLLM logprobs=0/FlatLogprobs."""
    tokens = list(map(int, completion.token_ids))
    if not tokens or len(tokens) > CAP:
        raise ValueError('Empty response or generation cap violation.')
    token_logprobs = completion.logprobs
    if token_logprobs is None or len(token_logprobs) != len(tokens):
        raise ValueError('logprobs=0 must retain every sampled-token probability.')
    values = []
    for token, position in zip(tokens, token_logprobs):
        if token not in position:
            raise ValueError('Sampled token missing from generation logprobs.')
        value = float(position[token].logprob)
        if not math.isfinite(value) or value > 1e-6:
            raise ValueError('Invalid sampled-token log probability.')
        values.append(value)
    cumulative = completion.cumulative_logprob
    if cumulative is None or not math.isfinite(float(cumulative)):
        raise ValueError('Missing/non-finite cumulative generation log probability.')
    cumulative = float(cumulative)
    difference = abs(sum(values) - cumulative)
    if difference > 1e-6 + 1e-9 * abs(cumulative):
        raise ValueError(f'Cumulative/chosen-token generation LP mismatch: {difference}')
    return cumulative, difference


def make_rows(prompt, text, prompt_ids, request, tag, k, seed, tokenizer, decode_check):
    if list(request.prompt_token_ids) != prompt_ids or len(request.outputs) != k:
        raise ValueError('vLLM prompt identity or sample count mismatch.')
    outputs = sorted(request.outputs, key=lambda completion: completion.index)
    if [completion.index for completion in outputs] != list(range(k)):
        raise ValueError('vLLM returned duplicate/missing completion indices.')
    rows = []
    for completion in outputs:
        ids = list(map(int, completion.token_ids))
        logprob, difference = checked_logprob(completion)
        if completion.finish_reason not in ('stop', 'length'):
            raise ValueError(f'Unexpected generation finish reason: {completion.finish_reason}')
        if completion.finish_reason == 'length' and len(ids) != CAP:
            raise ValueError('Context limit ended generation before direct 2048-token cap.')
        decode_view = decode_check(tokenizer, prompt_ids, ids, completion.text)
        verification = verify_response(completion.text, prompt)
        counts = extract_integer_list(completion.text)
        sample_index = int(completion.index)
        rows.append(dict(
            sample_id=digest([tag, prompt['prompt_id'], sample_index, ids])[:24],
            prompt_id=prompt['prompt_id'], panel_index=prompt['panel_index'], sample_index=sample_index,
            backbone='Qwen2.5-7B-Instruct', method=tag, tag=tag,
            split='train' if prompt['panel_index'] < 32 else 'heldout',
            messages=prompt['messages'], prompt=text, prompt_token_ids=prompt_ids,
            response=completion.text, response_token_ids=ids, response_length=len(ids),
            initial_sequence_logprob_generation=logprob,
            initial_mean_logprob_generation=logprob / len(ids),
            initial_surprisal_generation=-logprob / len(ids),
            generation_logprob_crosscheck_abs_error=difference,
            sampling_seed=seed + int(prompt['panel_index']),
            finish_reason=completion.finish_reason, stop_reason=getattr(completion, 'stop_reason', None),
            truncated=completion.finish_reason == 'length',
            detokenization_verification=decode_view, final_counts=counts, format_valid=counts is not None,
            verifier_name=VERIFIER_NAME, **verification))
    return rows


def validate_rows(rows, prompt, text, ids, tag, k, seed):
    if len(rows) != k or [row['sample_index'] for row in rows] != list(range(k)):
        raise ValueError('Sealed chunk sample-index/count mismatch.')
    for row in rows:
        if (row['prompt_id'] != prompt['prompt_id'] or row['panel_index'] != prompt['panel_index']
                or row['prompt_token_ids'] != ids or row['prompt'] != text
                or row['messages'] != prompt['messages'] or row['tag'] != tag
                or row['sampling_seed'] != seed + int(prompt['panel_index'])):
            raise ValueError('Sealed chunk prompt/tag/seed mismatch.')
        response_ids = row['response_token_ids']
        if not 0 < len(response_ids) <= CAP or row['response_length'] != len(response_ids):
            raise ValueError('Sealed chunk invalid response length.')
        logprob = row['initial_sequence_logprob_generation']
        if not math.isfinite(logprob) or logprob > 1e-6:
            raise ValueError('Sealed chunk invalid generation probability.')
        if not math.isclose(row['initial_mean_logprob_generation'], logprob / len(response_ids),
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Sealed chunk mean LP does not match sequence LP/token length.')
        expected = verify_response(row['response'], prompt)
        if any(row[key] != expected[key] for key in ('reward', 'mode_key', 'error')):
            raise ValueError('Sealed chunk verifier mismatch.')
        if row['sample_id'] != digest([tag, prompt['prompt_id'], row['sample_index'], response_ids])[:24]:
            raise ValueError('Sealed chunk sample identity mismatch.')


def chunk_paths(parts, prompt):
    path = Path(parts) / f'p{int(prompt["panel_index"]):03d}.jsonl'
    return path, Path(str(path) + '.complete.json')


def check_chunk(parts, prompt, text, ids, contract):
    path, marker = chunk_paths(parts, prompt)
    for target in (path, marker):
        if Path(str(target) + '.tmp').exists():
            raise RuntimeError(f'Preserve/review interrupted temporary chunk: {target}.tmp')
    if not marker.exists():
        if path.exists():
            raise RuntimeError(f'Preserve/review unsealed chunk; refusing redraw: {path}')
        return None
    receipt = json.loads(marker.read_text())
    if (receipt.get('status') != 'complete' or receipt.get('contract_sha256') != digest(contract)
            or not path.is_file() or receipt.get('outcomes_sha256') != sha256(path)):
        raise ValueError(f'Invalid chunk hash/contract: {marker}')
    rows = read_jsonl(path)
    validate_rows(rows, prompt, text, ids, contract['tag'], contract['sampling']['samples_per_prompt'],
                  contract['sampling']['seed'])
    if receipt.get('response_count') != len(rows) or receipt.get('prompt_id') != prompt['prompt_id']:
        raise ValueError('Chunk receipt count/prompt mismatch.')
    return receipt


def assemble(output, parts, rendered, contract):
    """Assemble only fully validated chunks, without resampling or reordering."""
    output = Path(output)
    marker = Path(str(output) + '.complete.json')
    receipts = []
    for prompt, text, ids in rendered:
        receipt = check_chunk(parts, prompt, text, ids, contract)
        if receipt is None:
            raise ValueError('Cannot seal an incomplete split.')
        receipts.append(receipt)
    chunk_hashes = {chunk_paths(parts, prompt)[0].name: receipt['outcomes_sha256']
                    for (prompt, _, _), receipt in zip(rendered, receipts)}
    receipt_hashes = {chunk_paths(parts, prompt)[1].name: sha256(chunk_paths(parts, prompt)[1])
                      for prompt, _, _ in rendered}
    joined = hashlib.sha256()
    for prompt, _, _ in rendered:
        with chunk_paths(parts, prompt)[0].open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                joined.update(block)
    if marker.exists():
        final = json.loads(marker.read_text())
        if (final.get('status') != 'complete' or final.get('contract_sha256') != digest(contract)
                or not output.is_file() or final.get('outcomes_sha256') != sha256(output)
                or final.get('outcomes_sha256') != joined.hexdigest()
                or final.get('chunk_outcomes_sha256') != chunk_hashes
                or final.get('chunk_receipts_sha256') != receipt_hashes
                or final.get('response_count') != len(rendered) * contract['sampling']['samples_per_prompt']):
            raise ValueError('Final split hash/contract/count mismatch.')
        return final
    if output.exists() or Path(str(output) + '.tmp').exists():
        raise RuntimeError(f'Preserve/review unsealed final assembly: {output}')
    def lines():
        for prompt, _, _ in rendered:
            with chunk_paths(parts, prompt)[0].open() as source:
                yield from source
    write_new_lines(output, lines())
    final = dict(status='complete', origin='fresh_direct_fixed_k_sampling',
                 tag=contract['tag'], canary=contract['canary'],
                 split='train' if rendered[0][0]['panel_index'] < 32 else 'heldout',
                 prompt_count=len(rendered), response_count=len(rendered) * contract['sampling']['samples_per_prompt'],
                 **contract['sampling'], **contract['identity'],
                 outcomes_sha256=sha256(output), prompts_sha256=contract['prompts_sha256'],
                 rendered_prompt_ids_sha256=contract['rendered_prompt_ids_sha256'],
                 contract_sha256=digest(contract), decoding_contract_sha256=contract['decoding_contract_sha256'],
                 generator_sha256=contract['source_sha256']['generate_direct.py'],
                 verifier_sha256=contract['source_sha256']['source_snapshot/coin_change_final_verifier.py'],
                 task_utils_sha256=contract['source_sha256']['task_utils.py'],
                 chunk_outcomes_sha256=chunk_hashes,
                 chunk_receipts_sha256=receipt_hashes,
                 software=contract['software'], decoding=contract['decoding'], gpu_names=sorted({name for item in receipts
                                                                for name in item['runtime']['gpu_names']}),
                 slurm_job_ids=sorted({item['runtime']['slurm_job_id'] for item in receipts
                                       if item['runtime']['slurm_job_id'] is not None}),
                 logprob_crosscheck_max_abs_error=max(item['logprob_crosscheck_max_abs_error'] for item in receipts),
                 sample_count_is_not_pass_at_k=True,
                 primary_estimator='direct empirical frequency; no importance weights')
    write_new_json(marker, final)
    return final


@contextlib.contextmanager
def output_lock(output):
    lock_path = Path(str(output) + '.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another sampler owns {output}') from error
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Final train.jsonl or heldout.jsonl path.')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--k', type=int, default=128, choices=(128,))
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--tp', type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument('--canary', action='store_true', help='Separate output, first two prompts, K=16.')
    args = parser.parse_args()
    prompts = load_prompts(args.prompts, args.canary)
    if not args.tag or not all(character.isalnum() or character in '_-.' for character in args.tag):
        raise ValueError('Tag must be a nonempty stable alphanumeric identifier.')
    import torch
    import transformers
    import vllm
    import tokenizers
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from detokenization import verify_detokenization

    k = 16 if args.canary else args.k
    with output_lock(args.output):
        identity = model_identity(args.model)
        if args.tag in ('grpo', 'eppo', 'grpo_h') and 'worker_completion_sha256' not in identity:
            raise ValueError('A primary post-update tag requires a hash-validated worker model.')
        model = identity['model']
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
        if not tokenizer.is_fast:
            raise ValueError('Exact vLLM detokenization check requires the fast tokenizer.')
        rendered = [(prompt, *render_prompt(tokenizer, prompt)) for prompt in prompts]
        parameters = sampling_config(k, args.seed, args.tp)
        software = dict(torch=torch.__version__, transformers=transformers.__version__,
                        vllm=vllm.__version__, tokenizers=tokenizers.__version__)
        generation_path = Path(model) / 'generation_config.json'
        generation_config = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        model_eos = json.loads((Path(model) / 'config.json').read_text()).get('eos_token_id')
        generation_eos = generation_config.get('eos_token_id', model_eos)
        generation_eos = generation_eos if isinstance(generation_eos, list) else [generation_eos]
        effective_eos = sorted({int(value) for value in [tokenizer.eos_token_id, *generation_eos]
                                if value is not None})
        decoding = dict(sampling=parameters, software=software, tokenizer_class=type(tokenizer).__name__,
                        tokenizer_eos_token_id=tokenizer.eos_token_id,
                        effective_eos_token_ids=effective_eos,
                        tokenizer_backend_sha256=digest(json.loads(tokenizer.backend_tokenizer.to_str())),
                        chat_template_sha256=digest(tokenizer.chat_template),
                        model_eos_token_id=model_eos)
        contract = dict(schema_version=1, tag=args.tag, canary=args.canary, identity=identity,
                        prompts_path=str(args.prompts.resolve()), prompts_sha256=sha256(args.prompts),
                        prompt_ids=[prompt['prompt_id'] for prompt in prompts], sampling=parameters,
                        rendered_prompt_ids_sha256=digest([(prompt['prompt_id'], ids) for prompt, _, ids in rendered]),
                        rendered_prompt_text_sha256=digest([(prompt['prompt_id'], text) for prompt, text, _ in rendered]),
                        source_sha256={name: sha256(ROOT / name) for name in (
                            'generate_direct.py', 'task_utils.py', 'detokenization.py',
                            'source_snapshot/coin_change_final_verifier.py')},
                        software=software, hf_generation_config=generation_config,
                        decoding=decoding, decoding_contract_sha256=digest(decoding))
        parts = Path(str(args.output) + '.parts')
        manifest = parts / 'contract.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != contract:
                raise ValueError('Scientific input/implementation changed; use a separately declared output.')
        else:
            if parts.exists() and any(parts.iterdir()):
                raise ValueError('Unowned chunk directory; preserve it for review.')
            if args.output.exists() or Path(str(args.output) + '.complete.json').exists():
                raise ValueError('Output exists without its chunk contract; refusing adoption.')
            write_new_json(manifest, contract)
        pending = [(prompt, text, ids) for prompt, text, ids in rendered
                   if check_chunk(parts, prompt, text, ids, contract) is None]
        if not pending:
            print(json.dumps(assemble(args.output, parts, rendered, contract), indent=2), flush=True)
            return
        if Path(str(args.output) + '.complete.json').exists() or args.output.exists():
            raise ValueError('Final bank exists but some chunks are missing.')
        if not torch.cuda.is_available():
            raise RuntimeError('Generation requires allocated CUDA GPUs.')
        if torch.cuda.device_count() != args.tp:
            raise RuntimeError('Visible GPU count must equal requested tensor parallelism.')
        runtime = dict(node=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                       slurm_step_id=os.environ.get('SLURM_STEP_ID'),
                       gpu_names=[torch.cuda.get_device_name(index) for index in range(args.tp)])
        llm = LLM(model=model, tokenizer=model, tensor_parallel_size=args.tp, dtype='bfloat16',
                  max_model_len=CONTEXT, max_num_seqs=128, gpu_memory_utilization=0.80,
                  enable_prefix_caching=True, generation_config='auto', trust_remote_code=False, seed=args.seed)
        engine_generation_config = llm.llm_engine.vllm_config.model_config.try_get_generation_config()
        eos_check = SamplingParams(n=1, stop_token_ids=[], ignore_eos=False)
        eos_check.update_from_generation_config(engine_generation_config, tokenizer.eos_token_id)
        if sorted(eos_check.all_stop_token_ids) != effective_eos:
            raise ValueError('Actual vLLM EOS rules differ from the frozen decoding contract.')
        for prompt, text, ids in pending:
            began = time.time()
            sampling = SamplingParams(n=k, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                max_tokens=CAP, min_tokens=0, seed=args.seed + int(prompt['panel_index']),
                stop=[], stop_token_ids=[], ignore_eos=False, skip_special_tokens=True,
                detokenize=True, include_stop_str_in_output=False, logprobs=0, flat_logprobs=True)
            requests = llm.generate([TokensPrompt(prompt_token_ids=ids)], [sampling], use_tqdm=True)
            if len(requests) != 1:
                raise ValueError('Expected exactly one generated request.')
            rows = make_rows(prompt, text, ids, requests[0], args.tag, k, args.seed, tokenizer, verify_detokenization)
            validate_rows(rows, prompt, text, ids, args.tag, k, args.seed)
            path, marker = chunk_paths(parts, prompt)
            write_new_lines(path, (json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n' for row in rows))
            receipt = dict(status='complete', contract_sha256=digest(contract), prompt_id=prompt['prompt_id'],
                           panel_index=prompt['panel_index'], response_count=len(rows), outcomes_sha256=sha256(path),
                           runtime=runtime, elapsed_seconds=time.time() - began,
                           engine_generation_config=engine_generation_config,
                           logprob_crosscheck_max_abs_error=max(row['generation_logprob_crosscheck_abs_error'] for row in rows),
                           detokenization_checks=dict(Counter(row['detokenization_verification'] for row in rows)))
            write_new_json(marker, receipt)
            print(json.dumps(dict(tag=args.tag, prompt_id=prompt['prompt_id'], panel_index=prompt['panel_index'],
                                  rows=len(rows), sealed_chunk=str(path))), flush=True)
        print(json.dumps(assemble(args.output, parts, rendered, contract), indent=2), flush=True)


if __name__ == '__main__':
    main()
