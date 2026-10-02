"""Run code ID and the paper's four OOD benchmarks with their original protocols."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

OOD = ("mmlu_pro", "safety_benchmark", "truthful_qa_mc", "gpqa")
CODE_DATASETS = {
    "id": ("codereasoner_id_test_1800.jsonl", "CodeReasoner-ID-test", "codereasoner", "id"),
    "cruxeval": ("cruxeval_output_test_800.jsonl", "CRUXEval-output-test", "cruxeval_output", "ood"),
    "lcb": ("livecodebench_execution_v2_test_479.jsonl", "LiveCodeBench-execution-v2-test", "livecodebench_output", "ood"),
}


def code_entry(summary):
    """Convert the native 0–1 code summary to the shared 0–100 result schema."""
    if not summary.get("complete") or summary.get("is_canary") or summary.get("limited"):
        raise ValueError("A complete, non-canary CodeReasoner evaluation is required")
    if summary["dataset"] != "CodeReasoner-ID-test" or summary["samples"] != 1800:
        raise ValueError("Expected the original 1,800-row CodeReasoner ID test split")
    pool = summary["sample_pool_size"]
    return {str(pool): {"prompts": summary["samples"], "pool_size": pool,
                        "pass_at_k": 100 * summary["pass_at_k"],
                        "avg_at_k": 100 * summary["avg_at_k"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/phi35", help="Merged local checkpoint or initialization")
    parser.add_argument("--method", default="base", help="base, grpo, grpo-h, exppo, or ablation label")
    parser.add_argument("--step", type=int, help="0 for base, 900 for trained methods by default")
    parser.add_argument("--suite", choices=("paper", "id", "ood", "external", "all"), default="paper")
    parser.add_argument("--data-dir", type=Path, default=Path("data/codereasoner/prepared"))
    parser.add_argument("--ood-data-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16, help="Code-task response pool size; paper uses 16")
    parser.add_argument("--code-seed", type=int, default=2027)
    parser.add_argument("--ood-seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--code-batch-size", type=int, default=32)
    parser.add_argument("--ood-batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    step = args.step if args.step is not None else (0 if args.method == "base" else 900)
    if step < 0 or (args.method == "base") != (step == 0):
        parser.error("Use step 0 for base and a positive step for trained methods")
    if min(args.k, args.tp, args.pp, args.code_batch_size, args.ood_batch_size) < 1:
        parser.error("Budgets and parallelism must be positive")
    result = {}
    keys = {"paper": ("id",), "id": ("id",), "ood": (),
            "external": ("cruxeval", "lcb"), "all": tuple(CODE_DATASETS)}[args.suite]
    if not args.dry_run and (args.out / "metrics.json").exists():
        raise FileExistsError(f"Results already exist: {args.out / 'metrics.json'}; choose a fresh --out")
    for key in keys:
        filename, dataset, source, distribution = CODE_DATASETS[key]
        output = args.out / "code" / key
        command = [sys.executable, "-m", "evaluation.code.eval_code_literal_passk",
                   "--model", args.checkpoint, "--method", args.method, "--step", str(step),
                   "--test_file", str(args.data_dir / filename), "--output_dir", str(output),
                   "--dataset", dataset, "--data_source", source, "--distribution", distribution,
                   "--run_tag", f"{args.method}-step{step}-{key}", "--k", str(args.k),
                   "--seed", str(args.code_seed), "--temperature", "0.7", "--top_p", "0.95",
                   "--max_new_tokens", "512", "--prompt_batch_size", str(args.code_batch_size),
                   "--tensor_parallel_size", str(args.tp), "--pipeline_parallel_size", str(args.pp),
                   "--gpu_memory_utilization", "0.90", "--max_model_len", "4096",
                   "--max_num_seqs", "128", "--dtype", "bfloat16"]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        before = set(output.glob("*.summary.json"))
        subprocess.run(command, check=True)
        new = set(output.glob("*.summary.json")) - before
        if len(new) != 1:
            raise RuntimeError(f"Expected one new code summary in {output}")
        summary = json.loads(new.pop().read_text())
        if key == "id":
            result["codereasoner"] = code_entry(summary)
    if args.suite in {"paper", "ood", "all"}:
        output = args.out / "ood"
        command = [sys.executable, "-m", "evaluation.run", "--model", "phi35",
                   "--checkpoint", args.checkpoint, "--data-dir", str(args.ood_data_dir),
                   "--out", str(output), "--benchmarks", *OOD, "--seed", str(args.ood_seed),
                   "--tp", str(args.tp), "--pp", str(args.pp), "--batch-size", str(args.ood_batch_size)]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
            result.update(json.loads((output / "metrics.json").read_text()))
    if not args.dry_run and result:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "metrics.json").open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
