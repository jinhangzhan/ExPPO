#!/usr/bin/env python3
"""Check the local reproduction environment without downloading or training."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import sys


REQUIRED = {
    "torch": "2.9.0", "vllm": "0.12.0", "transformers": "4.57.1",
    "tokenizers": "0.22.1", "ray": "2.49.1", "tensordict": "0.11.0",
    "torchdata": "0.11.0", "datasets": "3.6.0", "numpy": "1.26.4",
    "pandas": "2.2.3", "pyarrow": "20.0.0", "duckdb": "1.2.0", "flash-attn": "2.8.3",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Skip CUDA and FlashAttention checks.")
    parser.add_argument("--evaluation", action="store_true", help="Check the separate evaluation environment.")
    args = parser.parse_args()
    problems = []
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is required; reference version is 3.11.")
    versions = {}
    for package, expected in REQUIRED.items():
        if args.evaluation and package in {"flash-attn", "tensordict", "torchdata"}:
            continue
        if args.cpu and package in {"flash-attn", "vllm"}:
            continue
        try:
            actual = metadata.version(package)
            versions[package] = actual
            if actual.split("+")[0] != expected:
                problems.append(f"{package}: expected {expected}, found {actual}")
        except metadata.PackageNotFoundError:
            problems.append(f"{package}: missing")
    if not args.evaluation and importlib.util.find_spec("verl") is None:
        problems.append("Install the vendored trainer: pip install --no-deps -e third_party/verl")
    try:
        expected_antlr = "4.11.1" if args.evaluation else "4.9.3"
        actual_antlr = metadata.version("antlr4-python3-runtime")
        if actual_antlr != expected_antlr:
            problems.append(f"ANTLR: expected {expected_antlr}, found {actual_antlr}")
    except metadata.PackageNotFoundError:
        problems.append("antlr4-python3-runtime: missing")
    devices = []
    if not args.cpu and "torch" in versions:
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("CUDA is unavailable; run training checks on an allocated GPU host.")
            devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:
            problems.append(f"PyTorch runtime import failed: {exc}")
    print(json.dumps({"python": sys.version.split()[0], "packages": versions,
                      "gpus": devices, "problems": problems}, indent=2))
    return bool(problems)


if __name__ == "__main__":
    raise SystemExit(main())
