#!/usr/bin/env python3
"""Prepare paper training/benchmark inputs without implicit split substitutions."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.data.prepare import benchmark_specs, download, load_local, validate, write_jsonl


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["reasoning"], help="reasoning, train, codereasoner-raw, or benchmark names")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, help="Import archived JSONL/parquet inputs instead of downloading benchmarks")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = benchmark_specs()
    names = list(dict.fromkeys(name for group in args.datasets for name in (list(specs) if group == "reasoning" else [group])))
    for name in names:
        if name not in specs and name not in {"train", "codereasoner-raw"}:
            parser.error(f"Unknown dataset: {name}")
        if args.dry_run:
            print(json.dumps({"name": name, "source": specs.get(name, {"dataset": "wh-zhu/dapo" if name == "train" else "just-for-paper-submission/CodeReasoner-Dataset"}), "output_dir": str(args.output_dir)}))
            continue
        if name in specs:
            spec = specs[name]
            if args.source_dir:
                candidates = [args.source_dir / spec["domain"] / f"{name}.jsonl", args.source_dir / f"{name}.jsonl", args.source_dir / f"{name}.parquet"]
                source = next((p for p in candidates if p.is_file()), None)
                if source is None:
                    raise FileNotFoundError(f"Missing archived input {name} under {args.source_dir}")
                rows = load_local(source)
            else:
                rows = download(name, spec)
            validate(rows, spec)
            destination = args.output_dir / "evaluation" / spec["domain"] / f"{name}.jsonl"
            write_jsonl(destination, rows, args.overwrite)
            print(f"Verified {name}: {len(rows)} prompts -> {destination}")
        elif name == "train":
            from datasets import load_dataset
            source = load_dataset("wh-zhu/dapo", revision="96cf34c6d7447bce46fba99eae021fc45dd45e0f", split="train")
            if len(source) != 17398:
                raise ValueError(f"Expected 17,398 DAPO prompts, received {len(source)}")
            destination = args.output_dir / "train" / "dapo_math_17k.parquet"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.to_parquet(str(destination))
            print(destination)
        else:
            from huggingface_hub import hf_hub_download
            import shutil
            destination = args.output_dir / "codereasoner_raw"
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ["dataset_rl.json", "dataset_it.json", "README.md"]:
                target = destination / filename
                if target.exists() and not args.overwrite:
                    raise FileExistsError(target)
                source = hf_hub_download("just-for-paper-submission/CodeReasoner-Dataset", filename,
                                         repo_type="dataset", revision="08c784b2aa0e58e511c0cfdbdf61a61380a8c92f")
                shutil.copyfile(source, target)
            print("Downloaded upstream CodeReasoner. Run scripts/prepare_codereasoner.py for the paper's prepared split and verification protocol.")


if __name__ == "__main__":
    main()
