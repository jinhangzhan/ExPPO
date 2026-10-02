#!/usr/bin/env python3
"""Aggregate eight-task repetitions, computing domain means before sample SDs."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.data.prepare import benchmark_specs


def aggregate(paths):
    specs = benchmark_specs()
    runs = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if set(payload) != set(specs):
            raise ValueError(f"{path}: expected the complete eight-benchmark suite")
        row = {}
        for name, spec in specs.items():
            entry = payload[name][str(spec["k"])]
            if entry["prompts"] != spec["count"] or entry["pool_size"] < spec["k"]:
                raise ValueError(f"{path}: wrong prompt count or response budget for {name}")
            row[name] = {}
            for metric in ("pass_at_k", "avg_at_k"):
                value = entry[metric]
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f"{path}: invalid {metric} for {name}")
                row[name][metric] = value
        runs.append(row)
    if not runs:
        raise ValueError("At least one run is required")
    result = {}
    groups = {name: [name] for name in specs}
    groups.update({f"{domain}_average": [name for name in specs if specs[name]["domain"] == domain]
                   for domain in ("id", "ood")})
    groups["overall_average"] = list(specs)
    for name, tasks in groups.items():
        result[name] = {}
        for metric in ("pass_at_k", "avg_at_k"):
            values = [statistics.mean(run[task][metric] for task in tasks) for run in runs]
            result[name][metric] = {"mean": statistics.mean(values),
                                     "sd": statistics.stdev(values) if len(values) > 1 else None,
                                     "repetitions": len(values)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+", help="metrics.json files from evaluate.sh")
    parser.add_argument("--variation", choices=("training", "decoding", "single"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len({path.resolve() for path in args.metrics}) != len(args.metrics):
        parser.error("A repeated input file is not an independent repetition")
    if args.variation == "single" and len(args.metrics) != 1:
        parser.error("--variation single takes exactly one result file")
    if args.variation != "single" and len(args.metrics) < 2:
        parser.error("Sample SD requires at least two repetitions")
    text = json.dumps({"variation": args.variation, "units": "0-100",
                       "results": aggregate(args.metrics)}, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as destination:
            destination.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
