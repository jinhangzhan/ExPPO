# Reproducing the experiments

Run commands from the project root. Activate the training environment for
training/merging and the evaluation environment for data preparation/scoring.
Use `--dry-run` to inspect a launcher without starting GPU work.

## Shared settings

Training uses 17,398 DAPO math prompts, prompt/response limits 2,048/10,240,
16 rollouts per prompt, a configured prompt batch of 64, generation batch 32,
policy mini-batch 32, AdamW learning rate 1e-6, 10 warmup steps, weight decay
0.1, gradient clipping 1, token-mean loss, and zero KL penalties. Core GRPO runs
process 32 prompts (512 responses) per logged step. DAPO accumulates 64 retained
prompt groups per update from 32-prompt candidate batches.

Core GRPO clipping is 0.2/0.2. ExPPO uses lambda 0.2; GRPO-H entropy coefficient
is 0.001; GRPO-UR rank coefficient is 0.25; F-GRPO exponent is 0.5; REPO-R
starts at 0.001 with magnitude bounds 0.0001–0.05. Configuration:
[`configs/training/paper.yaml`](../configs/training/paper.yaml).

The eight-task evaluator preserves each task's prompt construction and verifier.
Coverage uses the unbiased estimator `1 - choose(n-c,k)/choose(n,k)`;
response accuracy is the fraction of verified responses. At the standard
budget `n=k`, coverage is the observed any-success indicator. Report unweighted
four-task ID and OOD means. Compute each run's suite mean before taking the
mean and sample standard deviation across runs; do not average task SDs.

## Experiment map

| Study | Models / methods | Training and evaluation |
| --- | --- | --- |
| Core comparison | qwen25, qwen3, phi35; grpo, grpo-h, exppo | 900 steps; initialization and checkpoints 100–900 |
| Closest baselines | qwen25; grpo, grpo-h, grpo-ur, repo-r, f-grpo, exppo | One 900-step checkpoint per method; three decoding repetitions |
| Coefficient sweep | phi35; exppo | 300 steps; lambda 0.1, 0.2, 0.3, 0.5, 0.75, 1.0 |
| Component ablations | phi35; exppo | 300 steps; no gate, no final normalization, no tanh, positive only, negative only |
| Matched F-GRPO comparison | phi35; f-grpo, exppo | 300 steps; three independent training runs |
| Independent core runs | phi35; grpo, exppo | 900 steps; three independent training runs |
| DAPO extension | qwen25; dapo, dapo-exppo | 900 steps; standard eight-task evaluation |
| Large sampling budgets | all three; initialization and core methods | Pools of 512 for AIME24/25 (reported through k=256), 128 for Minerva/OlympiadBench |
| Controlled multi-answer | all three; core methods; additional qwen25 baselines | 64 fixed prompts, 128 responses per prompt |
| Common-checkpoint continuation | qwen25; grpo, grpo-h, exppo updates | Shared ExPPO-900 weights; 32 update / 32 held-out prompts |
| Code reasoning | phi35; grpo, grpo-h, exppo | Recovered preparation, literal reward and evaluator; 900-step target; three training runs |

The main comparison uses training state 42; independently trained repetitions
use 42, 43, and 44. Standard evaluation budgets are 16 for AIME24/25, 8 for
Minerva, and 4 for OlympiadBench and each OOD task. Temperature/top-p are
0.7/0.95, with 10,240 new tokens. Closest-baseline error bars are decoding
variability at one fixed trained checkpoint, not independent training runs.

## Training variants

```bash
bash scripts/train.sh --model phi35 --method exppo --steps 300 --seed 43
bash scripts/train.sh --model phi35 --method f-grpo --steps 300 --seed 43
bash scripts/train.sh --model phi35 --method exppo --steps 300 --lambda 0.75
bash scripts/train.sh --model phi35 --method exppo --ablation no-gate
bash scripts/train.sh --model phi35 --method exppo --ablation no-normalization
bash scripts/train.sh --model phi35 --method exppo --ablation no-tanh
bash scripts/train.sh --model phi35 --method exppo --ablation positive-only
bash scripts/train.sh --model phi35 --method exppo --ablation negative-only
bash scripts/train.sh --model qwen25 --method dapo-exppo
```

Use distinct `--output` directories for studies with different step budgets.
Pass additional Hydra overrides after `--`. Hardware overrides can change
execution order and numerical results; retain the data and optimizer settings
when comparing algorithms. Each method should start from the same downloaded
initialization, not another method's trained weights.

## Evaluation and aggregation

```bash
# Initialization (pinned model downloaded first).
bash scripts/evaluate.sh --model phi35 --out outputs/evaluation/phi35-initialization

# Fixed checkpoint: repeat with --seed 43 and --seed 44 for decoding variability.
bash scripts/evaluate.sh --model phi35 --checkpoint outputs/RUN/global_step_900/actor/hf_merged \
  --seed 42 --out outputs/evaluation/RUN-decode42

# High-budget evaluation writes numerical metrics, not plots.
bash scripts/evaluate_high_k.sh --model phi35 --checkpoint outputs/RUN/global_step_900/actor/hf_merged \
  --out outputs/evaluation/RUN-high-k
```

Repeat the fixed-checkpoint command for saved steps 100, 200, …, 900 to recover
checkpoint-wise metrics. Baselines saved every 200 steps also save the final
step. Plot construction and historical result files are intentionally absent.

Aggregate complete eight-task repetitions with:

```bash
python scripts/summarize_evaluations.py outputs/evaluation/RUN-decode42/metrics.json \
  outputs/evaluation/RUN-decode43/metrics.json outputs/evaluation/RUN-decode44/metrics.json \
  --variation decoding
```

Use `--variation training` when inputs come from independently trained policies.

For overhead measurements, run each Qwen2.5 method for 100 steps on the same
eight-H200 allocation and capture its console log. GRPO-H's reported cost uses
unchunked entropy with entropy checkpointing disabled. Supply the full job wall
time (including setup) to `scripts/summarize_overhead.py LOG --allocation-seconds N`.
Throughput and memory use steps 11–100; estimated 900-step GPU-hours scale the
full 100-step allocation by nine. New measurements depend on hardware/runtime.

Controlled-task commands and exact metrics are described in
[`evaluation/README.md`](../evaluation/README.md). The synthetic prompt bank is
an evaluation input, not a historical result file. Phi's published controlled
scores use prefix re-verification of longer responses; direct short-cap
sampling is a different experiment. The continuation workflow is documented in
[`experiments/continuation/README.md`](../experiments/continuation/README.md).

## Implementation provenance

- **Qwen3 initialization:** the verified checkpoint is `Qwen/Qwen3-0.6B`
  revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Ablations:** the original component/sign-ablation source has been recovered
  and compared with the portable implementation. Positive-only and negative-only
  variants restrict the raw multiplier before shared group normalization; the
  inactive side can therefore also be rescaled. The standard ExPPO rule and its
  component variants retain the original population-standard-deviation convention.
- **CodeReasoner:** the recovered builder converts prompts to forward output
  prediction and splits normalized-function groups into 12,000 training, 1,800
  validation, and 1,800 ID examples. Its dedicated literal verifier, 16-response
  evaluator, and pinned upstream inputs are included. Code training uses
  1,024/512 prompt/response limits, distinct from math training. See the
  [CodeReasoner workflow](codereasoner.md).
- **DAPO:** release presets now implement the paper-described dynamic group
  filtering, asymmetric clipping, token-mean loss, and overlong-response penalty.
  These features were not all active in the previously available launcher.
  The updated recipe requires new runs to establish its numerical results;
  the change is not evidence that historical DAPO measurements used this path.
- **Resumption:** release checkpoints retain optimizer state. Some historical
  launchers saved model/extra state without AdamW state, so reproducing a resumed
  historical trajectory requires knowing those restart boundaries.
- **Numerical reproduction:** original checkpoints, response banks, and figure
  data are excluded. Fresh training and sampling reproduce the implemented
  procedure, not guaranteed bit-for-bit historical scores. The common-checkpoint
study can also accept separately supplied original sealed response banks.

Source integration, configuration checks, and CPU tests do not replace a full
GPU reproduction run. The review ZIP contains code and documentation, without
model weights, sampled responses, recorded figure data, or manuscript files.
