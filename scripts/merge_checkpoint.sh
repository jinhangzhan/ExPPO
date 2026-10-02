#!/usr/bin/env bash
# Run on a suitably provisioned compute node after training has saved the actor.
set -euo pipefail

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then dry_run=true; shift; fi
if [[ "${1:-}" == --help || $# -lt 1 || $# -gt 2 ]]; then
    printf '%s\n' 'Usage: bash scripts/merge_checkpoint.sh [--dry-run] ACTOR_DIRECTORY [OUTPUT_DIRECTORY]'
    printf '%s\n' 'Example: bash scripts/merge_checkpoint.sh outputs/checkpoints/RUN/global_step_900/actor'
    [[ "${1:-}" == --help ]] && exit 0
    exit 2
fi
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$release_root"
actor_directory=$1
output_directory=${2:-$actor_directory/hf_merged}
export PYTHONPATH="$release_root/third_party/verl:$release_root${PYTHONPATH:+:$PYTHONPATH}"
command=("${PYTHON_BIN:-python}" -m verl.model_merger merge --backend fsdp
    --local_dir "$actor_directory" --target_dir "$output_directory")
if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
if [[ ! -d "$actor_directory/huggingface" ]]; then
    printf 'Actor directory must contain the saved huggingface model/tokenizer configuration.\n' >&2
    exit 2
fi
if [[ -e "$output_directory" ]]; then
    printf 'Output already exists: %s. Choose a new output directory.\n' "$output_directory" >&2
    exit 2
fi
exec "${command[@]}"
