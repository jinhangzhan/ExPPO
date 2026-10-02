#!/usr/bin/env bash
# Launch from any working directory; all project-relative paths resolve at root.
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash scripts/train.sh --model qwen25|qwen3|phi35 --method METHOD [options] [-- HYDRA_OVERRIDES]' \
        'Methods: grpo, grpo-h, exppo, f-grpo, grpo-ur, repo-r, dapo, dapo-exppo' \
        'Options:' \
        '  --steps N             Optimization steps (900; component ablations: 300)' \
        '  --seed N              Data, minibatch and rollout random state (42)' \
        '  --lambda X            ExPPO coefficient in [0,1] (0.2)' \
        '  --ablation NAME       none, no-gate, no-normalization, no-tanh, positive-only, negative-only' \
        '  --model-path ID/PATH   Override the model checkpoint' \
        '  --train-file PATH     Training parquet (data/train/dapo_math_17k.parquet)' \
        '  --val-file PATH       Validation parquet (training file; validation disabled by default)' \
        '  --output PATH         Checkpoint directory (outputs/checkpoints/MODEL-METHOD-ABLATION-lLAMBDA-sSEED)' \
        '  --gpus N              GPUs per node (8)' \
        '  --nodes N             Number of nodes in an already available Ray cluster (1)' \
        '  --dry-run             Print the exact command without importing GPU libraries or launching work'
}

model=qwen25
method=exppo
steps=''
training_seed=42
coefficient=0.2
ablation=none
model_path=''
train_file=data/train/dapo_math_17k.parquet
val_file=''
output=''
gpus=8
nodes=1
dry_run=false
extra=()

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=true; shift ;;
        --) shift; extra=("$@"); break ;;
        --model|--method|--steps|--seed|--lambda|--ablation|--model-path|--train-file|--val-file|--output|--gpus|--nodes)
            if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
            case "$1" in
                --model) model=$2 ;;
                --method) method=$2 ;;
                --steps) steps=$2 ;;
                --seed) training_seed=$2 ;;
                --lambda) coefficient=$2 ;;
                --ablation) ablation=$2 ;;
                --model-path) model_path=$2 ;;
                --train-file) train_file=$2 ;;
                --val-file) val_file=$2 ;;
                --output) output=$2 ;;
                --gpus) gpus=$2 ;;
                --nodes) nodes=$2 ;;
            esac
            shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$model" in
    qwen25) default_model=Qwen/Qwen2.5-7B-Instruct ;;
    # Verified against the original downloaded checkpoint's weight digest.
    qwen3) default_model=Qwen/Qwen3-0.6B ;;
    phi35) default_model=microsoft/Phi-3.5-mini-instruct ;;
    *) printf 'Unknown model: %s\n' "$model" >&2; exit 2 ;;
esac

method_args=()
save_freq=100
case "$method" in
    grpo) ;;
    grpo-h)
        method_args+=(actor_rollout_ref.actor.entropy_coeff=0.001)
        if [[ "$model" == qwen3 ]]; then
            method_args+=(actor_rollout_ref.actor.entropy_from_logits_with_chunking=true
                actor_rollout_ref.actor.entropy_checkpointing=true)
        fi ;;
    exppo) method_args+=(algorithm.grpo_adv_reweight_enable=true) ;;
    f-grpo) method_args+=(algorithm.f_grpo_enable=true); save_freq=200 ;;
    grpo-ur) method_args+=(algorithm.grpo_unlikeliness_enable=true); save_freq=200 ;;
    repo-r) method_args+=(algorithm.repo_r_enable=true); save_freq=200 ;;
    dapo|dapo-exppo)
        method_args+=(actor_rollout_ref.actor.clip_ratio_high=0.28
            algorithm.filter_groups.enable=true algorithm.filter_groups.metric=acc
            reward_model.reward_manager=dapo reward_model.overlong_buffer.enable=true
            '+reward_model.reward_kwargs.max_resp_len=${data.max_response_length}'
            '+reward_model.reward_kwargs.overlong_buffer_cfg=${reward_model.overlong_buffer}')
        if [[ "$method" == dapo-exppo ]]; then
            method_args+=(algorithm.grpo_adv_reweight_enable=true)
        fi ;;
    *) printf 'Unknown method: %s\n' "$method" >&2; exit 2 ;;
esac

if [[ "$ablation" != none && "$method" != exppo ]]; then
    printf 'Component ablations require --method exppo.\n' >&2; exit 2
fi
case "$ablation" in
    none) ;;
    no-gate) method_args+=(algorithm.grpo_adv_reweight_gate_enabled=false) ;;
    no-normalization) method_args+=(algorithm.grpo_adv_reweight_final_normalization_enabled=false) ;;
    no-tanh) method_args+=(algorithm.grpo_adv_reweight_tanh_enabled=false) ;;
    positive-only) method_args+=(algorithm.grpo_adv_reweight_sign=positive) ;;
    negative-only) method_args+=(algorithm.grpo_adv_reweight_sign=negative) ;;
    *) printf 'Unknown ablation: %s\n' "$ablation" >&2; exit 2 ;;
esac
if [[ -z "$steps" ]]; then
    if [[ "$ablation" == none ]]; then steps=900; else steps=300; fi
fi
for positive_integer in "$steps" "$gpus" "$nodes"; do
    if [[ ! "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
        printf 'Steps, GPUs and nodes must be positive integers.\n' >&2; exit 2
    fi
done
if [[ ! "$training_seed" =~ ^[0-9]+$ ]]; then
    printf 'The random-state identifier must be a nonnegative integer.\n' >&2; exit 2
fi
if [[ ! "$coefficient" =~ ^(0([.][0-9]*)?|[.][0-9]+|1([.]0*)?)$ ]]; then
    printf 'ExPPO lambda must be a number in [0,1].\n' >&2; exit 2
fi

release_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$release_root"
if [[ -z "$model_path" ]]; then
    if [[ -f "models/$model/config.json" ]]; then
        model_path="models/$model"
    elif "$dry_run"; then
        model_path=$default_model
    else
        printf 'Download the pinned model into models/%s first, or pass --model-path explicitly.\n' "$model" >&2
        exit 2
    fi
fi
val_file=${val_file:-$train_file}
experiment="$model-$method-$ablation-l$coefficient-s$training_seed"
output=${output:-outputs/checkpoints/$experiment}

export PYTHONPATH="$release_root/third_party/verl:$release_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-outputs/wandb}
export WANDB_RUN_ID_FILE=${WANDB_RUN_ID_FILE:-$output/.wandb_run_id}
export WANDB_RESUME=${WANDB_RESUME:-allow}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export VERL_STEP_LOG_EVERY=${VERL_STEP_LOG_EVERY:-1}
export PYTHONHASHSEED=$training_seed

command=("${PYTHON_BIN:-python}" -m verl.trainer.main_ppo
    --config-path "$release_root/configs/training" --config-name paper
    "actor_rollout_ref.model.path=$model_path"
    "data.train_files=$train_file" "data.val_files=$val_file"
    "data.seed=$training_seed" "actor_rollout_ref.actor.data_loader_seed=$training_seed"
    "actor_rollout_ref.actor.fsdp_config.seed=$training_seed"
    "actor_rollout_ref.rollout.engine_kwargs.vllm.seed=$training_seed"
    "algorithm.grpo_adv_reweight_lambda=$coefficient"
    "trainer.experiment_name=$experiment" "trainer.default_local_dir=$output"
    "trainer.total_training_steps=$steps" "trainer.save_freq=$save_freq"
    "trainer.n_gpus_per_node=$gpus" "trainer.nnodes=$nodes"
    "${method_args[@]}")
if [[ "$model" == qwen3 ]]; then
    command+=(+data.apply_chat_template_kwargs.enable_thinking=false)
fi
command+=("${extra[@]}")

if "$dry_run"; then
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
latest_marker="$output/latest_checkpointed_iteration.txt"
if [[ -f "$latest_marker" ]]; then
    latest_step=$(tr -d '[:space:]' < "$latest_marker")
    if [[ ! "$latest_step" =~ ^[0-9]+$ ]]; then
        printf 'Invalid checkpoint marker: %s\n' "$latest_marker" >&2; exit 2
    fi
    if ((10#$latest_step >= steps)); then
        printf 'Checkpoint %s already reaches the requested %s steps; nothing to train.\n' "$latest_step" "$steps"
        exit 0
    fi
fi
if [[ ! -f "$train_file" || ! -f "$val_file" ]]; then
    printf 'Prepare the training/validation parquet files before launching.\n' >&2; exit 2
fi
mkdir -p -- "$output" "$WANDB_DIR" outputs/logs
exec "${command[@]}"
