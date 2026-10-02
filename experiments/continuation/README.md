# Common-checkpoint continuation

This appendix experiment forks the same merged Qwen2.5-7B ExPPO last checkpoint
into one GRPO, GRPO-H, or ExPPO update. It is a fresh-optimizer intervention,
not resumption of the original optimizer. No plotting scripts or response banks
are included.

From the repository root:

```sh
python scripts/continuation.py prepare
python scripts/continuation.py run --model checkpoints/qwen25-exppo/hf_merged --dry-run
# Run inside your GPU allocation after checking the commands:
python scripts/continuation.py run --model checkpoints/qwen25-exppo/hf_merged
```

The update worker needs one large-memory CUDA GPU and substantial host RAM
(FP32 parameter masters and a full CPU parameter snapshot). The original study
used H200-class hardware. Sampling defaults to one GPU; `--tp` controls its
tensor parallelism. This is separate from the distributed 900-step trainer.
Use the evaluation environment (`requirements-eval.txt`): the sampler was
executed with PyTorch 2.9.0, Transformers 4.57.1, vLLM 0.12.0, and Tokenizers
0.22.1. GPU execution should be checked on the target hardware before a full run.

The frozen 64-prompt bank is split by its original order: first 32 for updating,
last 32 for held-out evaluation. All policies use 128 responses per prompt,
temperature/top-p 1, and a direct 2,048-token cap. Updates use the first 16
responses of every update prompt, without selection by correctness. A single
AdamW step uses learning rate `1e-6`, weight decay `0.1`, betas `(0.9, 0.999)`,
epsilon `1e-8`, gradient clipping `1`, and global response-token-mean loss.
BF16 input weights are promoted to FP32 masters; scoring and update forwards
use BF16. ExPPO uses lambda `0.2`; GRPO-H uses entropy coefficient `0.001`.

The published update used an archival response bank. By default this workflow
generates a fresh update bank using randomization key `20260903`, then separate
PRE/POST banks with key `20260905` plus the original prompt index. Fresh draws
and execution settings can change numerical outcomes. To reuse an existing
compatible archival bank, provide both `--train-bank` and `--heldout-bank`
(each JSONL requires its matching `.complete.json` receipt).

Outputs stay under `outputs/continuation`: sealed generation chunks, forked
models, exact-token likelihood and coefficient captures, and `summary.json`.
The summary reports both splits, conditional correct-pair diversity, fixed-budget
correctness/mode yield, 10,000 paired prompt-bootstrap draws, and signed
post-normalization credit changes. Probability metrics use 0--1 units; multiply
differences by 100 to obtain the paper's percentage-point contrasts. Undefined
diversity stays null and each comparison lists its shared eligible prompts.
To recompute the CPU summary, run `python scripts/continuation.py summarize`.

Completed sampler chunks are verified and reused. Completed update directories
are immutable; use a separate work directory for a new run. Retain partial
outputs for inspection after interruption.
