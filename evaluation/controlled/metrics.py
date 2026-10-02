"""Correct-mode yield and paired correct-pair diversity from response banks."""

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.controlled.bank import PROMPTS, audit
from evaluation.controlled.verifier import verify_response


def prompt_metrics(modes):
    counts = Counter(mode for mode in modes if mode is not None)
    correct = sum(counts.values())
    effective = math.exp(-sum((v / correct) * math.log(v / correct) for v in counts.values())) if correct else 0.0
    diversity = 1 - sum(v * (v - 1) for v in counts.values()) / (correct * (correct - 1)) if correct >= 2 else None
    return {"correct": correct, "samples": len(modes), "accuracy": correct / len(modes), "solved": int(correct > 0),
            "observed_yield": len(counts), "effective_yield": effective, "diversity": diversity}


def paired_diversity(baseline, treatment, draws=10000, random_state=20260903):
    shared = sorted(key for key in baseline.keys() & treatment.keys()
                    if baseline[key]["diversity"] is not None and treatment[key]["diversity"] is not None)
    if not shared:
        return {"shared_prompts": 0, "baseline_d": None, "treatment_d": None, "difference": None, "ci95": None}
    differences = [treatment[key]["diversity"] - baseline[key]["diversity"] for key in shared]
    rng = random.Random(random_state)
    estimates = sorted(statistics.mean(rng.choices(differences, k=len(shared))) for _ in range(draws))
    def percentile(p):
        index = (len(estimates) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(estimates) - 1)
        return estimates[lo] + (estimates[hi] - estimates[lo]) * (index - lo)
    return {"shared_prompts": len(shared), "baseline_d": statistics.mean(baseline[key]["diversity"] for key in shared),
            "treatment_d": statistics.mean(treatment[key]["diversity"] for key in shared),
            "difference": statistics.mean(differences), "ci95": [percentile(.025), percentile(.975)] if len(shared) > 1 else None}


def evaluate_bank(paths, prompts, k=128, tokenizer=None, prefix_tokens=None):
    grouped = defaultdict(dict)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            pid, index = row["prompt_id"], int(row["sample_index"])
            if pid not in prompts or index in grouped[pid]:
                raise ValueError("Unknown prompt or duplicate sample: use one complete bank per method")
            response = row["response"]
            if prefix_tokens is not None:
                if tokenizer is None or "response_token_ids" not in row:
                    raise ValueError("Exact prefix verification requires tokenizer and original token IDs")
                response = tokenizer.decode(row["response_token_ids"][:prefix_tokens], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            result = verify_response(response, prompts[pid])
            grouped[pid][index] = result["mode_key"] if result["reward"] else None
    if set(grouped) != set(prompts) or any(set(values) != set(range(k)) for values in grouped.values()):
        raise ValueError("Every prompt must have exactly K distinct sample indices")
    metrics = {key: prompt_metrics([grouped[key][i] for i in range(k)]) for key in prompts}
    n = len(metrics)
    return {"prompts": n, "samples_per_prompt": k, "correct_responses": sum(r["correct"] for r in metrics.values()),
            "correct_responses_per_prompt": statistics.mean(r["correct"] for r in metrics.values()),
            "accuracy": 100 * statistics.mean(r["accuracy"] for r in metrics.values()),
            "solved_prompts": sum(r["solved"] for r in metrics.values()),
            "observed_yield": statistics.mean(r["observed_yield"] for r in metrics.values()),
            "effective_yield": statistics.mean(r["effective_yield"] for r in metrics.values()), "per_prompt": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", required=True, help="METHOD=directory containing parts/**/*.jsonl or one response JSONL")
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--prefix-tokens", type=int)
    parser.add_argument("--tokenizer", help="Required for exact Phi prefix re-verification")
    parser.add_argument("--treatment", default="ExPPO")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = {r["prompt_id"]: r for r in audit(args.prompts)}
    tokenizer = None
    if args.prefix_tokens:
        if not args.tokenizer:
            parser.error("--prefix-tokens requires --tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    result = {}
    for entry in args.bank:
        label, source = entry.split("=", 1)
        path = Path(source)
        paths = [path] if path.is_file() else sorted(path.glob("**/parts/*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"No response parts found under {path}")
        if label in result:
            raise ValueError("Each method must be supplied once")
        result[label] = evaluate_bank(paths, prompts, args.k, tokenizer, args.prefix_tokens)
    pairs = {}
    if args.treatment in result:
        for baseline in result:
            if baseline != args.treatment:
                pairs[baseline] = paired_diversity(result[baseline]["per_prompt"], result[args.treatment]["per_prompt"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"methods": result, "paired_diversity": pairs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
