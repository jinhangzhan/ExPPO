# Evaluation

Use the evaluation environment (`requirements-eval.txt`) from the repository root.
GPU inference uses the original `eval_local_passk.py` and answer checkers.
No plotting code or recorded figure measurements are included.

```bash
python scripts/prepare_data.py --datasets reasoning
bash scripts/evaluate.sh --model qwen25 --checkpoint checkpoints/qwen25/exppo/global_step_900/actor/hf_merged --out outputs/qwen25/exppo
bash scripts/evaluate_high_k.sh --model qwen25 --checkpoint checkpoints/qwen25/exppo/global_step_900/actor/hf_merged --out outputs/qwen25/exppo-high-k
bash scripts/evaluate_controlled.sh --model qwen25 --checkpoint checkpoints/qwen25/exppo/global_step_900/actor/hf_merged --method ExPPO --out outputs/controlled/qwen25/exppo
```

Add `--dry-run` to inspect commands without loading models or generating responses.
Use `--model qwen3` or `--model phi35` for the other families. Omit `--checkpoint`
to evaluate downloaded initialization weights in `models/<alias>`. Set `--tp` and
`--pp` to match available GPUs; `PYTHON_BIN` can select the evaluation interpreter.
Qwen3 uses Qwen/Qwen3-0.6B and explicitly disables thinking, matching recorded
evaluation metadata.

## Inputs and metrics

The preparation script pins upstream revisions and validates ordered prompt/answer
fingerprints against the archived paper inputs. Fingerprints apply the evaluator's
boxed system prompt and strip answer-boundary whitespace. Downloaded inputs are
not bundled; obtain any necessary dataset access/license permissions first. GPQA
requires accepted upstream terms and Hugging Face authentication.

| Benchmark | Source / split | Prompts | k |
|---|---|---:|---:|
| AIME24 | `wh-zhu/aime-24`, train, stable deduplication | 30 | 16 |
| AIME25 | `math-ai/aime25`, test | 30 | 16 |
| Minerva | `math-ai/minervamath`, test | 272 | 8 |
| OlympiadBench | `QwenLM/Qwen2.5-Math`, `evaluation/data/olympiadbench/test.jsonl` | 675 | 4 |
| GPQA | `idavidrein/gpqa`, `gpqa_main.csv` / train | 448 | 4 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro`, validation | 70 | 4 |
| Safety | `ThWu/safety_benchmark`, train, safer-response choice | 1,000 | 4 |
| TruthfulQA-MC | pinned `sylinrl/TruthfulQA` MC1 four-choice construction | 684 | 4 |

`lighteval/MATH` in archived ID records is a reward-routing label, not their source
dataset. AIME24's published conversion contains repeated records; each distinct
prompt/answer pair is evaluated once with 16 samples. The pinned Qwen mathematical
evaluation conversion preserves the paper's 675-question OlympiadBench set;
the Hothan release contains 674. TruthfulQA retains questions with at least four choices, samples
three incorrect choices when necessary, then deterministically shuffles choices.

Import archived inputs with `--source-dir path/to/evaluation/data` instead of
downloading. A fingerprint mismatch stops preparation rather than substituting
a different benchmark. Revisions and fingerprints are in
[`benchmarks.json`](../configs/evaluation/benchmarks.json).

Standard sampling uses temperature 0.7, top-p 0.95 and 10,240 new tokens.
Coverage uses `1 - C(n-c,k)/C(n,k)`; at `n=k` it equals any-success coverage.
Response accuracy is the mean verified-response share. High-k evaluation samples
512 responses per AIME prompt and 128 per Minerva/OlympiadBench prompt, reporting
the former through k=256. Its response-accuracy estimate uses the entire pool.
`scripts/summarize_evaluations.py` computes per-benchmark mean/sample SD and first
averages each four-task suite within each independent run. Specify whether the
repetitions vary training or decoding:

```bash
python scripts/summarize_evaluations.py outputs/run42/metrics.json outputs/run43/metrics.json outputs/run44/metrics.json --variation training --output outputs/summary.json
```

## Controlled multi-answer task

`controlled/prompts.jsonl` contains the frozen synthetic input bank, not model
outputs. Run `python -m evaluation.controlled.bank` to exhaustively verify its
64 prompts and 4–46 valid modes per prompt. Every response is verified by the
original final-most literal JSON `"counts"` parser.

Generation uses K=128, temperature/top-p 1.0, and chunks of 16. Qwen2.5 uses a
2,048-token cap; Qwen3 uses 8,192. Phi generates 16,384-token responses, then
re-verifies their exact 2,048-token prefixes. Retained token IDs enable this
protocol; direct generation at 2,048 is a different experiment.

```bash
python -m evaluation.controlled.metrics \
  --bank GRPO=outputs/controlled/qwen25/grpo \
  --bank GRPO-H=outputs/controlled/qwen25/grpo-h \
  --bank ExPPO=outputs/controlled/qwen25/exppo \
  --output outputs/controlled/qwen25/comparison.json
```

For Phi, paired comparisons must use the same 2,048-token prefix protocol as
the per-method wrapper (not the full 16,384-token response):

```bash
python -m evaluation.controlled.metrics \
  --bank GRPO=outputs/controlled/phi35/grpo \
  --bank GRPO-H=outputs/controlled/phi35/grpo-h \
  --bank ExPPO=outputs/controlled/phi35/exppo \
  --prefix-tokens 2048 --tokenizer models/phi35 \
  --output outputs/controlled/phi35/comparison.json
```

Observed yield counts distinct correct vectors per prompt; effective yield is
the exponentiated entropy of their frequencies. Both include zero for unsolved
prompts. Correct responses/prompt counts all verified successes divided by the
prompt count. Conditional diversity is the probability that two distinct correct
responses have different vectors, restricted to prompts with two successes for
both compared methods. Paired intervals resample these shared prompts 10,000 times.

## CodeReasoner

The recovered CodeReasoner builder, training reward, and literal-answer evaluator
are integrated separately from the mathematical boxed-answer verifier. See the
[CodeReasoner workflow](../docs/codereasoner.md) for pinned inputs, the exact
12,000/1,800/1,800 split, 16-response evaluation, and the four-task OOD comparison.
The general `--test-file` mathematical evaluator is not the CodeReasoner scorer.
