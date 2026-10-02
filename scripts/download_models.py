#!/usr/bin/env python3
"""Download the pinned initialization checkpoints; never downloads on --dry-run."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", "qwen25", "qwen3", "phi35"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    specs = json.loads((ROOT / "configs/models.json").read_text())
    selected = specs if args.model == "all" else {args.model: specs[args.model]}
    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return
    from huggingface_hub import snapshot_download
    os.chdir(ROOT)
    for alias, spec in selected.items():
        destination = Path(spec["directory"])
        receipt = destination / "exppo_source.json"
        identity = {"repository": spec["repository"], "revision": spec["revision"]}
        if destination.exists() and any(destination.iterdir()):
            if receipt.exists() and json.loads(receipt.read_text()) != identity:
                raise SystemExit(f"{destination} contains another initialization; choose a new directory.")
            if not receipt.exists():
                raise SystemExit(f"{destination} has no ExPPO source receipt; preserve it and use --model-path for training.")
        destination.mkdir(parents=True, exist_ok=True)
        # Record the intended source before download so an interrupted download can resume.
        receipt.write_text(json.dumps(identity, indent=2) + "\n")
        snapshot_download(repo_id=identity["repository"], revision=identity["revision"],
                          local_dir=str(destination), ignore_patterns=["*.msgpack", "*.h5", "*.onnx"])
        print(f"{alias}: {destination}")


if __name__ == "__main__":
    main()
