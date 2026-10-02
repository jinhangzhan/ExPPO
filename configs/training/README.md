# Training configuration

`paper.yaml` contains the shared FSDP/vLLM settings. `scripts/train.sh` selects
the model and one method modifier; trailing Hydra overrides adjust runtime
resources or replace the data/reward function.

```sh
bash scripts/train.sh --model qwen25 --method exppo --dry-run
bash scripts/train.sh --model phi35 --method exppo --steps 300 --seed 43 --lambda 0.2
bash scripts/train.sh --model phi35 --method exppo --ablation negative-only
```

Main runs use 900 steps; component ablations default to 300. Use `--steps 300`
for the matched GRPO/F-GRPO and full-ExPPO ablation references. The coefficient
sweep is `0.1, 0.2, 0.3, 0.5, 0.75, 1.0`. Independent runs set `--seed` to
42, 43 and 44; it controls data order, policy minibatches and rollout sampling.

The configured prompt batch is 64; non-DAPO runs consume 32 prompts
with 16 responses each (512 responses per step). DAPO generates candidates in
32-prompt batches, filters groups with uniform correctness, and collects 64
accepted groups (1,024 responses) before each update. Both DAPO variants use
this same collection rule. The default allocation is
eight GPUs. Adapt GPU count, tensor parallelism and token microbatching to the
available hardware without changing the global sample budget.

Downloaded models in `models/{qwen25,qwen3,phi35}` take priority. Qwen3 defaults
to the verified `Qwen/Qwen3-0.6B` checkpoint.
Use `--model-path` for an explicit alternative.

Component and sign-specific switches were checked against the recovered
external-cluster implementation. See the [runtime notes](../../third_party/verl/README.md)
for DAPO configuration semantics and checkpoint-resume changes.

DAPO uses clipping `(0.2, 0.28)`, token-mean loss and a linear overlong penalty
over the final 2,048 tokens of the 10,240-token response budget (coefficient 1).
Filtering and ExPPO's pass-rate gate use raw binary verifier correctness before
that penalty; group-relative advantages use the shaped reward. The default
32-generation-batch limit fails explicitly when an update cannot be filled;
override `algorithm.filter_groups.max_num_gen_batches=0` for no per-update limit.
Increase `trainer.total_epochs` if the dataset is exhausted before 900 updates.
This is the paper-aligned DAPO implementation added to the release, not a
verified replay of the archived DAPO run; new GPU runs are needed to validate it.
