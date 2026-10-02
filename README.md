# ExPPO

Exploration-Preserving Policy Optimization: prompt-relative surprisal shaping
for group-relative reinforcement learning with verifiable rewards.

This release contains the modified trainer, GRPO/GRPO-H and closest baselines,
data preparation, evaluation, ablations, and controlled-task experiments.
The review ZIP contains code and documentation; manuscript files, plotting
scripts, recorded figure data, and the original transfer archive are excluded.

## Quick start

Set up the separate [training and evaluation environments](docs/environment.md).
Then, in the evaluation environment:

```bash
python scripts/download_models.py --model all
python scripts/prepare_data.py --datasets train reasoning
```

In the training environment:

```bash
bash scripts/train.sh --model qwen25 --method exppo --output outputs/qwen25-exppo
bash scripts/merge_checkpoint.sh outputs/qwen25-exppo/global_step_900/actor
```

Back in the evaluation environment:

```bash
bash scripts/evaluate.sh --model qwen25 \
  --checkpoint outputs/qwen25-exppo/global_step_900/actor/hf_merged \
  --out outputs/evaluation/qwen25-exppo
```

Defaults follow the paper's 900-step schedule, 16 responses per prompt, and
ExPPO coefficient 0.2. Launchers support `--help` and `--dry-run`.

## Reproduction and submission

- [Experiment recipes and provenance](docs/reproduction.md)
- [Training implementation and ablations](configs/training/README.md)
- [Benchmark preparation and evaluation](evaluation/README.md)
- [CodeReasoner preparation, training, and evaluation](docs/codereasoner.md)
- [Validation results](docs/validation.md)

Model and dataset revisions are pinned. Qwen3 uses **Qwen/Qwen3-0.6B**.
See the reproduction notes for historical restart behavior and the newly
enabled paper-aligned DAPO recipe.

```bash
make check                     # portable-source and syntax checks
make package                   # dist/ExPPO-code.zip and its SHA-256 checksum
```

Generated outputs, datasets, model weights, and credentials are ignored by Git.
No experiments are launched by installation or packaging. Existing upstream
licenses are retained; see [attribution](THIRD_PARTY.md).
