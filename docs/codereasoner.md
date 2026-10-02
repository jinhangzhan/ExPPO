# CodeReasoner experiment

This workflow uses the recovered experiment's preparation, literal verifier,
sampling and metrics. It predicts frozen Python return values; neither reference
programs nor generated code are executed.

## Prepare and train

Use the separate environments from [environment.md](environment.md).
Prepare data in the evaluation environment, then switch to training:

```bash
source .venv-eval/bin/activate
python scripts/download_models.py --model phi35
python scripts/prepare_codereasoner.py
python scripts/prepare_data.py --datasets mmlu_pro safety_benchmark truthful_qa_mc gpqa
deactivate
source .venv-train/bin/activate
bash scripts/train_codereasoner.sh --method exppo --seed 42
```

`--method grpo` and `--method grpo-h` select the reference comparisons. Repeat
each training command with `--seed 43` and `--seed 44` for the three-run table.
The launcher defaults to the paper's 900 updates and accepts `--steps`,
`--lambda`, `--ablation` and `--dry-run`; outputs are separated from math runs.
The paper's component/sign ablations and coefficient sweep use the **math**
launcher, `scripts/train.sh`, rather than this code-training dataset.

The original code recipe uses Phi-3.5-mini-instruct; prompt/response limits
1,024/512; nominal batch 64; generation/PPO minibatch 32; 16 responses per
prompt; learning rate 1e-6; clipping 0.2; ExPPO coefficient 0.2; and GRPO-H
entropy coefficient 0.001. Actor token microbatches are 4,608, or 3,072 for
GRPO-H with chunked/checkpointed entropy. Other optimizer settings inherit
`configs/training/paper.yaml`.

## Pinned data and split

All three sources in [data_sources.json](../configs/code/data_sources.json)
are downloaded at fixed revisions and verified against their original SHA-256.
`--offline` reuses downloaded files; `--list` prints sources; `--dry-run` checks
existing sources without writing; existing prepared data require `--force`
to replace.

The CodeReasoner RL corpus contributes both prompt directions, converted to
forward output prediction from `func_str`, `func_args`, and `result`. Exact
code/argument duplicates and invalid literals are removed. Normalized-function
groups overlapping either external code benchmark are excluded before assigning
whole groups with split random state 42: training 12,000, validation 1,800,
ID test 1,800. The builder audits function and code/input overlap and retains
CRUXEval-O's 800 and LiveCodeBench execution-v2's 479 official rows. Those two
external code tasks are available for additional evaluation; the paper's OOD
table uses MMLU-Pro, Safety, TruthfulQA-MC and GPQA instead.

AST normalization reproduces the original Python 3.13 representation, including
empty type-parameter fields on Python 3.11. Generated JSONL/parquet and a
provenance/overlap manifest are written under `data/codereasoner/` and excluded
from the submission archive.

## Evaluate and aggregate

Merge each actor with `scripts/merge_checkpoint.sh` in the training environment,
then switch back to the evaluation environment for generation and aggregation:

```bash
deactivate
source .venv-eval/bin/activate
bash scripts/evaluate_codereasoner.sh --method exppo \
  --checkpoint outputs/checkpoints/phi35-code-exppo-none-l0.2-s42/global_step_900/actor/hf_merged \
  --out outputs/code-eval/exppo-s42
# Initialization, using the same evaluation protocol:
bash scripts/evaluate_codereasoner.sh --out outputs/code-eval/base
```

The default `--suite paper` runs CodeReasoner ID plus the four standard OOD
benchmarks. Code ID uses 16 samples, temperature 0.7, top-p 0.95, 512 output
tokens, prompt batch 32, and generation random state 2027. OOD follows the
standard evaluation script: four samples, 10,240 output tokens, prompt batch
64, and generation random state 42. `--code-seed` and `--ood-seed` explicitly
control these separately. Default evaluation parallelism is TP=4, PP=1;
set `--tp`/`--pp` for the allocated GPUs.

`--suite id`, `ood`, `external`, or `all` selects subsets/additional code tasks.
The native code evaluator reports unbiased pass@1/5/8/16, average correctness,
first-attempt correctness, strict-type diagnostics, and cluster-macro results.
The paper uses official-compatible literal equality and row-macro pass@16;
the standard four OOD results use pass@4. Both pass and average metrics are
combined in `metrics.json` on a 0–100 scale.

```bash
python scripts/summarize_codereasoner.py \
  outputs/code-eval/exppo-s42/metrics.json \
  outputs/code-eval/exppo-s43/metrics.json \
  outputs/code-eval/exppo-s44/metrics.json --variation training
```

Repeat evaluation for the matching independently trained models. Aggregation
reports per-task mean/sample SD and computes each run's OOD average before its
SD. Initialization uses one result and `--variation single`.

## Provenance and continuation

The original campaign ran 300 updates and continued toward 1,000 with the
optimizer reset while restoring scheduler/data/random states. This release
uses the requested last checkpoint at 900 and saves full optimizer state for
new resumable runs. Reproducing a historical interrupted trajectory requires
its restart boundaries and original checkpoint state; a new uninterrupted
900-update run follows the released recipe but is not that identical trajectory.
