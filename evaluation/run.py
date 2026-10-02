"""Portable launcher for the paper's fixed-budget and high-k evaluations."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from evaluation.data.prepare import benchmark_specs, load_local, validate
from evaluation.metrics.coverage import read_groups, summarize

MODELS = {"qwen25": "Qwen/Qwen2.5-7B-Instruct", "qwen3": "Qwen/Qwen3-0.6B", "phi35": "microsoft/Phi-3.5-mini-instruct"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--checkpoint", help="Merged Hugging Face checkpoint; omit to evaluate initialization")
    parser.add_argument("--data-dir", type=Path, default=Path("data/evaluation"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", help="Subset of the eight benchmark names")
    parser.add_argument("--high-k", action="store_true")
    parser.add_argument("--test-file", type=Path, help="Explicit separately prepared dataset, e.g. CodeReasoner")
    parser.add_argument("--data-id", default="code_reasoner")
    parser.add_argument("--k", type=int, help="Required with --test-file; otherwise the paper budget is used")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=10240)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking", choices=["true", "false", "default"], default="default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = benchmark_specs()
    names = args.benchmarks or [name for name, spec in specs.items() if not args.high_k or spec["domain"] == "id"]
    if args.test_file:
        if args.k is None or args.k < 1:
            parser.error("--test-file requires an explicit positive --k")
        names = [args.data_id]
    results = {}
    for name in names:
        if not args.test_file and name not in specs:
            parser.error(f"Unknown benchmark: {name}")
        spec = specs.get(name, {})
        if args.high_k and "max_k" not in spec:
            parser.error("The paper's high-k suite contains the four ID tasks")
        k = args.k if args.test_file else spec["pool_size" if args.high_k else "k"]
        source = args.test_file or args.data_dir / spec["domain"] / f"{name}.jsonl"
        output = args.out / name
        model = args.checkpoint or f"models/{args.model}"
        if not args.dry_run and not args.checkpoint and not Path(model).is_dir():
            raise FileNotFoundError(f"Missing initialization {model}; run scripts/download_models.py first")
        cmd = [sys.executable, "-m", "evaluation.eval_local_passk", "--model", model,
               "--test_file", str(source), "--out", str(output), "--data_id", name, "--k", str(k),
               "--temperature", "0.7", "--top_p", "0.95", "--max_new_tokens", str(args.max_new_tokens),
               "--batch_size", str(args.batch_size), "--seed", str(args.seed),
               "--vllm_tensor_parallel_size", str(args.tp), "--vllm_pipeline_parallel_size", str(args.pp),
               "--vllm_max_model_len", str(args.max_model_len), "--vllm_gpu_memory_utilization", str(args.gpu_memory_utilization),
               "--vllm_dtype", "bfloat16"]
        thinking = "false" if args.model == "qwen3" and args.enable_thinking == "default" else args.enable_thinking
        if thinking != "default":
            cmd += ["--enable_thinking", thinking]
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        if not args.test_file:
            validate(load_local(source), spec)
        previous = set(output.glob("*.log.jsonl")) if output.exists() else set()
        subprocess.run(cmd, check=True)
        created = set(output.glob("*.log.jsonl")) - previous
        if len(created) != 1:
            raise RuntimeError(f"Expected one new detailed evaluation log under {output}")
        ks = [value for value in [1, 4, 8, 16, 32, 64, 128, 256] if value <= spec["max_k"]] if args.high_k else [k]
        results[name] = summarize(read_groups(created.pop(), repeat_group_size=1 if k == 1 else 0), ks)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
