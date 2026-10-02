#!/usr/bin/env python3
"""Aggregate CodeReasoner plus four OOD tasks; compute macro means before sample SD."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.code.run import OOD
from evaluation.data.prepare import benchmark_specs


def aggregate(paths):
    specs = benchmark_specs()
    settings = {name: (specs[name]["k"], specs[name]["count"]) for name in OOD}
    settings["codereasoner"] = (16, 1800)
    runs = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if set(payload) != set(settings):
            raise ValueError(f"{path}: expected CodeReasoner ID and the four paper OOD tasks")
        row = {}
        for name, (budget, count) in settings.items():
            value = payload[name][str(budget)]
            if value["prompts"] != count or value["pool_size"] != budget:
                raise ValueError(f"{path}: wrong response budget or row count for {name}")
            row[name] = {}
            for metric in ("pass_at_k", "avg_at_k"):
                score = value[metric]
                if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
                    raise ValueError(f"{path}: invalid {metric} for {name}")
                row[name][metric] = score
        runs.append(row)
    if not runs:
        raise ValueError("At least one evaluation is required")
    groups = {name: [name] for name in settings}
    groups["ood_average"] = OOD
    result = {}
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
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--variation", choices=("training", "decoding", "single"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len({path.resolve() for path in args.metrics}) != len(args.metrics):
        parser.error("Repeated input paths are not independent repetitions")
    if args.variation == "single" and len(args.metrics) != 1:
        parser.error("--variation single requires one result")
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
