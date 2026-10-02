"""Portable orchestration and non-plotting summaries of the continuation study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from task_utils import ROOT, atomic_json, atomic_jsonl, read_jsonl, sha256, verify_response

REPOSITORY = ROOT.parents[1]
PROMPTS = REPOSITORY / "evaluation/controlled/prompts.jsonl"
PROMPT_SHA256 = "eeee29807fd7095ebd572a7c7806847965da72fbe1bd767d7a612a9918e0b78d"
RULES = ("grpo", "grpo_h", "eppo")
LABELS = {"pre": "Shared ExPPO checkpoint", "grpo": "GRPO", "grpo_h": "GRPO-H", "eppo": "ExPPO"}
VERIFIER = ROOT / "source_snapshot/coin_change_final_verifier.py"


def canonical_metadata(prompt):
    """Keep the original task and deterministically enumerate scoring strings."""
    payload = prompt["payload"]
    vectors = list(itertools.product(range(payload["max_count"] + 1),
                                     repeat=len(payload["denominations"])))

    def residual(vector):
        return sum(a * b for a, b in zip(vector, payload["denominations"])) - payload["target"]

    valid = [vector for vector in vectors if residual(vector) == 0]
    if {",".join(map(str, vector)) for vector in valid} != set(prompt["valid_mode_keys"]):
        raise ValueError("Enumerated answer modes differ from the fixed prompt bank")
    wrong = sorted((vector for vector in vectors if residual(vector) != 0),
                   key=lambda vector: (abs(residual(vector)), vector))[:len(valid)]
    return dict(prompt, canonical_valid_vectors=[list(v) for v in valid],
                canonical_distractor_vectors=[list(v) for v in wrong],
                canonical_distractor_selection="in-range wrong sums by absolute residual then lexicographic vector; count equals valid-mode count")


def prepare(workdir, source=PROMPTS):
    if sha256(source) != PROMPT_SHA256:
        raise ValueError("Continuation requires the unchanged paper prompt bank")
    prompts = read_jsonl(source)
    if len(prompts) != 64 or [p["panel_index"] for p in prompts] != list(range(64)):
        raise ValueError("Expected the original ordered 64-prompt bank")
    prompts = [canonical_metadata(prompt) for prompt in prompts]
    for split, selected in (("train", prompts[:32]), ("heldout", prompts[32:])):
        atomic_jsonl(Path(workdir) / "data" / f"{split}_prompts.jsonl", selected)
    atomic_json(Path(workdir) / "data/complete.json", {
        "status": "complete", "original_prompts_sha256": PROMPT_SHA256,
        "split": "first 32 update prompts; last 32 held-out prompts",
        "update_selection": "first 16 sample indices per update prompt, independent of reward",
        "samples_per_prompt": 128, "max_new_tokens": 2048,
    })


def commands(args):
    work = args.workdir
    sampler = [sys.executable, str(ROOT / "generate_direct.py")]
    train_bank = args.train_bank or work / "banks/update/train.jsonl"
    heldout_bank = args.heldout_bank or work / "banks/update/heldout.jsonl"
    jobs = []

    def sample(model, tag, split, output, seed):
        jobs.append(sampler + ["--model", str(model), "--prompts", str(work / "data" / f"{split}_prompts.jsonl"),
                              "--output", str(output), "--tag", tag, "--k", "128",
                              "--seed", str(seed), "--tp", str(args.tp)])

    if args.train_bank is None:
        sample(args.model, "update_pre", "train", train_bank, args.update_seed)
        sample(args.model, "update_pre", "heldout", heldout_bank, args.update_seed)
    # Independent fixed-budget PRE evaluation, with the same per-prompt draw rule as POST.
    for split in ("train", "heldout"):
        sample(args.model, "pre", split, work / "direct/pre" / f"{split}.jsonl", args.seed)
    for rule in RULES:
        jobs.append([sys.executable, str(ROOT / "update_worker.py"), "--model", str(args.model),
                     "--train-bank", str(train_bank), "--heldout-bank", str(heldout_bank),
                     "--data-root", str(work / "data"), "--output", str(work / "runs" / rule),
                     "--start-label", "eppo900", "--variant", rule, "--seed", str(args.seed),
                     "--bank-k", "128", "--batch-size", "1", "--logit-chunk-size", "128",
                     "--learning-rate", "1e-6", "--score-dtype", "bfloat16",
                     "--rarity-score-dtype", "bfloat16"])
        for split in ("train", "heldout"):
            sample(work / "runs" / rule / "model", rule, split,
                   work / "direct" / rule / f"{split}.jsonl", args.seed)
    return jobs


def load_bank(path, prompt_path):
    """Validate sealed generation outputs and independently reverify answers."""
    path, prompt_path = Path(path), Path(prompt_path)
    marker = json.loads(Path(str(path) + ".complete.json").read_text())
    required = {"status": "complete", "outcomes_sha256": sha256(path),
                "prompts_sha256": sha256(prompt_path), "verifier_sha256": sha256(VERIFIER),
                "prompt_count": 32, "response_count": 4096, "samples_per_prompt": 128,
                "max_new_tokens": 2048}
    if any(marker.get(key) != value for key, value in required.items()) or marker.get("canary", False):
        raise ValueError(f"Invalid or mismatched bank receipt: {path}")
    prompts = {p["prompt_id"]: p for p in read_jsonl(prompt_path)}
    groups = defaultdict(list)
    seen = set()
    for row in read_jsonl(path):
        prompt = prompts.get(row["prompt_id"])
        if prompt is None or row["panel_index"] != prompt["panel_index"]:
            raise ValueError("Bank contains an unexpected prompt")
        if row["sample_id"] in seen:
            raise ValueError("Duplicated sample identity")
        seen.add(row["sample_id"])
        expected = verify_response(row["response"], prompt)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("Recorded correctness differs from the final-answer verifier")
        if not 0 < len(row["response_token_ids"]) == row["response_length"] <= 2048:
            raise ValueError("Response violates the direct token cap")
        groups[row["prompt_id"]].append(row)
    if set(groups) != set(prompts) or len(seen) != 4096:
        raise ValueError("Incomplete bank")
    for rows in groups.values():
        if sorted(r["sample_index"] for r in rows) != list(range(128)):
            raise ValueError("Missing or duplicate within-prompt samples")
    return groups


def prompt_metrics(rows):
    counts = Counter(row["mode_key"] for row in rows if row["reward"])
    correct = sum(counts.values())
    diversity = (1 - sum(n * (n - 1) for n in counts.values()) / (correct * (correct - 1))
                 if correct >= 2 else None)
    effective = math.exp(-sum((n / correct) * math.log(n / correct) for n in counts.values())) if correct else 0.
    return {"accuracy": correct / len(rows), "conditional_d": diversity,
            "correct_responses": correct, "observed_modes": len(counts), "effective_modes": effective}


def estimate(values, seed, draws, *labels):
    """Original prompt-percentile bootstrap, with stable contrast-specific streams."""
    import numpy as np
    values = np.asarray([value for value in values if value is not None and math.isfinite(value)], float)
    if not len(values):
        return {"mean": None, "ci_low": None, "ci_high": None, "n_prompts": 0}
    payload = json.dumps([int(seed), *labels], separators=(",", ":")).encode()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16], "big"))
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(samples, .025)),
            "ci_high": float(np.quantile(samples, .975)), "n_prompts": len(values)}


def coefficient_audit(capture):
    groups = defaultdict(list)
    totals = {(correct, high): 0. for correct in (True, False) for high in (True, False)}
    for row in capture["records"]:
        groups[row["prompt_id"]].append(row)
        if row["u"] != 0:
            totals[(bool(row["reward"]), row["u"] > 0)] += row["A_post"] - row["A_pre"]
    errors = [abs(sum(abs(row["A_post"]) for row in rows) - sum(abs(row["A_pre"]) for row in rows))
              for rows in groups.values()]
    if any(len(rows) != 16 for rows in groups.values()) or not errors or max(errors) > 1e-6:
        raise ValueError("ExPPO capture violates group size or sequence absolute-mass preservation")
    return {"categories": [{"correct": correct, "surprisal": "high" if high else "low",
                             "sum_A_post_minus_A_pre": value} for (correct, high), value in totals.items()],
            "max_group_absolute_mass_error": max(errors), "update_prompts": len(groups),
            "units": "sequence-advantage units before token aggregation"}


def summarize(workdir, draws=10000, seed=20260905):
    work = Path(workdir)
    outcomes, contrasts = [], []
    reference_banks = None
    shared_decoding = None
    for rule in RULES:
        receipt = json.loads((work / "runs" / rule / "complete.json").read_text())
        if receipt.get("status") != "complete" or receipt.get("score_only") or receipt.get("canary"):
            raise ValueError("Every update arm must finish before comparison")
        if receipt["args"]["variant"] != rule or receipt["counts"]["update_responses"] != 512:
            raise ValueError("Update arm identity or sample count differs from the protocol")
        optimizer = receipt["diagnostics"]["optimizer"]
        expected_optimizer = {"name": "AdamW", "fresh_state": True, "state_device": "cpu", "foreach": False,
                              "lr": 1e-6, "eps": 1e-8, "betas": [.9, .999], "weight_decay": .1,
                              "clip_norm": 1., "optimizer_steps": 1,
                              "loss_reduction": "global-response-token-mean", "ppo_ratio": 1., "kl_coefficient": 0.}
        if optimizer != expected_optimizer:
            raise ValueError("Update optimizer differs from the one-step paper protocol")
        for name, digest in receipt["output_hashes"].items():
            if sha256(work / "runs" / rule / name) != digest:
                raise ValueError("Worker output hash mismatch")
        hashes = receipt["input_hashes"]
        identity = (receipt["model_hashes"], hashes["train_bank"], hashes["heldout_bank"],
                    hashes["train_prompts"], hashes["heldout_prompts"])
        if reference_banks is None:
            reference_banks = identity
        elif reference_banks != identity:
            raise ValueError("Update arms must share the same model and response banks")
    for split in ("heldout", "train"):
        panel_order = [p["prompt_id"] for p in read_jsonl(work / "data" / f"{split}_prompts.jsonl")]
        measures = {}
        for policy in ("pre", *RULES):
            bank_path = work / "direct" / policy / f"{split}.jsonl"
            marker = json.loads(Path(str(bank_path) + ".complete.json").read_text())
            if marker.get("tag") != policy or marker.get("seed") != seed:
                raise ValueError("Direct evaluation policy or randomization key mismatch")
            if policy == "pre":
                if marker.get("model_weight_files_sha256") != reference_banks[0]:
                    raise ValueError("PRE evaluation is not the common starting checkpoint")
            elif marker.get("worker_completion_sha256") != sha256(work / "runs" / policy / "complete.json"):
                raise ValueError("POST evaluation is not bound to its completed update")
            if shared_decoding is None:
                shared_decoding = marker["decoding_contract_sha256"]
            elif shared_decoding != marker.get("decoding_contract_sha256"):
                raise ValueError("Direct evaluations use different decoding settings")
            groups = load_bank(bank_path,
                               work / "data" / f"{split}_prompts.jsonl")
            # Preserve the original analyzer's ID ordering for identical bootstrap streams.
            measures[policy] = {pid: prompt_metrics(rows) for pid, rows in sorted(groups.items())}
            for metric in ("accuracy", "conditional_d", "observed_modes", "effective_modes"):
                outcomes.append({"split": split, "policy": LABELS[policy], "metric": metric,
                                 **estimate([r[metric] for r in measures[policy].values()], seed, draws,
                                            "summary", split, policy, metric)})
        for left, right in (("eppo", "grpo"), ("eppo", "grpo_h"), *((r, "pre") for r in RULES)):
            if set(measures[left]) != set(measures[right]):
                raise ValueError("Unmatched comparison prompts")
            for metric in ("accuracy", "conditional_d"):
                primary = left == "eppo" and right in ("grpo", "grpo_h")
                order = panel_order if primary else measures[left]
                eligible = [pid for pid in order if measures[left][pid][metric] is not None
                            and measures[right][pid][metric] is not None]
                values = [measures[left][pid][metric] - measures[right][pid][metric] for pid in eligible]
                # The paper reports the independent audit's panel-ordered bootstrap.
                labels = (("independent_audit", split, left, right, "d" if metric == "conditional_d" else metric)
                          if primary else ("contrast", split, left, right, metric))
                contrasts.append({"split": split, "left": LABELS[left], "right": LABELS[right],
                                  "metric": metric, "shared_prompt_ids": eligible,
                                  **estimate(values, seed, draws, *labels)})
    capture = json.loads((work / "runs/eppo/update_capture.json").read_text())
    result = {"status": "complete", "bootstrap_draws": draws, "analysis_seed": seed,
              "uncertainty": "paired prompt-percentile intervals conditional on the realized policies and banks; unadjusted",
              "accuracy_and_diversity_units": "probability; multiply by 100 for paper percentage-point contrasts",
              "outcomes": outcomes, "paired_contrasts": contrasts, "coefficient_audit": coefficient_audit(capture)}
    atomic_json(work / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "summarize"):
        command = sub.add_parser(name)
        command.add_argument("--workdir", type=Path, default=Path("outputs/continuation"))
        if name == "run":
            command.add_argument("--model", type=Path, required=True, help="Merged Qwen2.5-7B ExPPO last checkpoint")
            command.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1)
            command.add_argument("--train-bank", type=Path, help="Optional sealed archival update bank")
            command.add_argument("--heldout-bank", type=Path, help="Optional sealed archival held-out bank")
            command.add_argument("--update-seed", type=int, default=20260903)
            command.add_argument("--dry-run", action="store_true")
        if name != "prepare":
            command.add_argument("--bootstrap-draws", type=int, default=10000)
            command.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.workdir)
        print(f"Prepared 32 update / 32 held-out prompts in {args.workdir / 'data'}")
    elif args.command == "run":
        if bool(args.train_bank) != bool(args.heldout_bank):
            parser.error("Supply both archival banks, or neither")
        if args.bootstrap_draws < 1 or args.seed < 0 or args.update_seed < 0:
            parser.error("Positive bootstrap draws and non-negative randomization keys are required")
        jobs = commands(args)
        if args.dry_run:
            print("Prepare the fixed 32/32 prompt split, then run:")
            for job in jobs:
                print(shlex.join(job))
            print("Finally write the paired-bootstrap and coefficient summary.")
            return
        prepare(args.workdir)
        for job in jobs:
            print(shlex.join(job), flush=True)
            subprocess.run(job, check=True)
        summarize(args.workdir, args.bootstrap_draws, args.seed)
    else:
        if args.bootstrap_draws < 1:
            parser.error("Positive bootstrap draws are required")
        summarize(args.workdir, args.bootstrap_draws, args.seed)


if __name__ == "__main__":
    main()
