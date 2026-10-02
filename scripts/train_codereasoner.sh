#!/usr/bin/env bash
# CodeReasoner recipe; shared algorithms and checkpoint handling live in train.sh.
set -euo pipefail
release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$release_root"
method=exppo
training_seed=42
coefficient=0.2
ablation=none
data_dir=data/codereasoner/prepared
steps=900
output=''
forward=()
extra=()
while (($#)); do
    case "$1" in
        --help|-h)
            printf '%s\n' 'Usage: bash scripts/train_codereasoner.sh --method grpo|grpo-h|exppo [options] [-- HYDRA_OVERRIDES]' \
                'Defaults: Phi-3.5, 900 steps, seed 42, lambda 0.2, 8 GPUs.' \
                'Options: --steps N --seed N --lambda X --ablation NAME --data-dir PATH --output PATH' \
                'Also accepts train.sh options --model-path, --train-file, --val-file, --gpus, --nodes, --dry-run.'
            exit 0 ;;
        --) shift; extra=("$@"); break ;;
        --dry-run) forward+=(--dry-run); shift ;;
        --method|--steps|--seed|--lambda|--ablation|--data-dir|--output|--model-path|--train-file|--val-file|--gpus|--nodes)
            (($# >= 2)) || { printf 'Missing value for %s\n' "$1" >&2; exit 2; }
            case "$1" in
                --method) method=$2 ;;
                --steps) steps=$2 ;;
                --seed) training_seed=$2 ;;
                --lambda) coefficient=$2 ;;
                --ablation) ablation=$2 ;;
                --data-dir) data_dir=$2 ;;
                --output) output=$2 ;;
                *) forward+=("$1" "$2") ;;
            esac
            shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$method" in grpo|grpo-h|exppo|f-grpo|grpo-ur|repo-r) ;;
    *) printf 'Unsupported code method: %s\n' "$method" >&2; exit 2 ;;
esac
experiment="phi35-code-$method-$ablation-l$coefficient-s$training_seed"
output=${output:-outputs/checkpoints/$experiment}
overrides=(
    "trainer.experiment_name=$experiment"
    data.max_prompt_length=1024 data.max_response_length=512
    data.train_batch_size=64 data.gen_batch_size=32
    data.trust_remote_code=false actor_rollout_ref.model.trust_remote_code=false
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4608
    actor_rollout_ref.rollout.n=16 actor_rollout_ref.rollout.max_num_seqs=32
    actor_rollout_ref.rollout.max_model_len=1536
    actor_rollout_ref.rollout.max_num_batched_tokens=1536
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4608
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4608
    custom_reward_function.path=evaluation/code/reward.py
    custom_reward_function.name=compute_score
    trainer.save_freq=100 trainer.max_actor_ckpt_to_keep=10
)
if [[ "$method" == grpo-h ]]; then
    overrides+=(actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
        actor_rollout_ref.actor.entropy_checkpointing=true)
fi
exec bash scripts/train.sh --model phi35 --method "$method" --steps "$steps" \
    --seed "$training_seed" --lambda "$coefficient" --ablation "$ablation" \
    --train-file "$data_dir/codereasoner_train_12000.nosnappy.parquet" \
    --val-file "$data_dir/codereasoner_validation_1800.nosnappy.parquet" \
    --output "$output" "${forward[@]}" -- "${overrides[@]}" "${extra[@]}"
