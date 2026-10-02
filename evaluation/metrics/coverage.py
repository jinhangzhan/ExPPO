"""Unbiased pass@k and response accuracy from recorded attempts (no plots)."""

import argparse
import json
import math
import statistics
from pathlib import Path


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def read_groups(path, repeat_group_size=0):
    groups, scalar = [], []
    with Path(path).open() as handle:
        lines = list(handle)
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") in {"meta", "summary"}:
            continue
        if "attempt_correct" in row:
            groups.append([bool(x) for x in row["attempt_correct"]])
        elif "correct" in row:
            scalar.append(bool(row["correct"]))
    if groups and scalar:
        raise ValueError("Mixed direct-sampling and repeated-record formats")
    if scalar:
        if repeat_group_size < 1 or len(scalar) % repeat_group_size:
            raise ValueError("Scalar records require a valid --repeat-group-size")
        groups = [scalar[i:i + repeat_group_size] for i in range(0, len(scalar), repeat_group_size)]
    if not groups or any(not group for group in groups):
        raise ValueError("No complete response groups")
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Incomplete response pool: unequal attempts per prompt")
    return groups


def summarize(groups, ks):
    if not groups:
        raise ValueError("No prompts")
    return {str(k): {"pass_at_k": 100 * statistics.mean(pass_at_k(len(g), sum(g), k) for g in groups),
                     "avg_at_k": 100 * statistics.mean(sum(g) / len(g) for g in groups),
                     "prompts": len(groups), "pool_size": len(groups[0])} for k in ks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--ks", default="1,4,8,16,32,64,128")
    parser.add_argument("--repeat-group-size", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(read_groups(args.log, args.repeat_group_size), [int(k) for k in args.ks.split(",")])
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
