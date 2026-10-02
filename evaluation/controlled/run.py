"""Launch the controlled K=128 protocol, including Phi's nested-token horizon."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from evaluation.run import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--checkpoint", help="Local merged model; omit to download/evaluate initialization")
    parser.add_argument("--method", required=True, help="Method name; use Initialization for base weights")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model = args.checkpoint or f"models/{args.model}"
    if not args.checkpoint and not args.dry_run:
        if not Path(model).is_dir():
            raise FileNotFoundError(f"Missing initialization {model}; run scripts/download_models.py first")
    cap = {"qwen25": 2048, "qwen3": 8192, "phi35": 16384}[args.model]
    cmd = [sys.executable, "-m", "evaluation.controlled.sample", "--model", model,
           "--prompts", "evaluation/controlled/prompts.jsonl", "--output-dir", args.out,
           "--method", args.method, "--backbone", args.model, "--max-new-tokens", str(cap),
           "--max-model-len", str(cap + 512), "--tensor-parallel-size", str(args.tp),
           "--enable-thinking", "false" if args.model == "qwen3" else "default", "--seed", str(args.seed)]
    if args.dry_run:
        cmd.append("--dry-run")
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    analysis = [sys.executable, "-m", "evaluation.controlled.metrics", "--bank", f"{args.method}={args.out}",
                "--output", str(Path(args.out) / "metrics.json")]
    if args.model == "phi35":
        analysis += ["--prefix-tokens", "2048", "--tokenizer", model]
    print(shlex.join(analysis), flush=True)
    if not args.dry_run:
        subprocess.run(analysis, check=True)


if __name__ == "__main__":
    main()
