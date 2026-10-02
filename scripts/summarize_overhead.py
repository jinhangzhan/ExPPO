#!/usr/bin/env python3
"""Summarize the paper's 100-step overhead profile from fresh console logs."""

import argparse
import json
import math
import re
from pathlib import Path

FIELDS = ("timing_s/step", "perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def load_metrics(text):
    rows = {}
    for line in text.splitlines():
        line = ANSI.sub("", line)
        if "training/global_step:" not in line or "timing_s/step:" not in line:
            continue
        values = {}
        for key in ("training/global_step", *FIELDS):
            found = re.search(re.escape(key) + r"\s*:\s*([-+0-9.eE]+)", line)
            if not found:
                raise ValueError(f"Missing profile field {key}")
            value = float(found.group(1))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite profile field {key}")
            values[key] = value
        step = int(values.pop("training/global_step"))
        if step in rows and rows[step] != values:
            raise ValueError(f"Conflicting resumed entries at step {step}")
        rows[step] = values
    if set(rows) != set(range(1, 101)):
        raise ValueError("Expected a complete 100-step profile, with unique steps 1–100")
    return rows


def summarize(rows, allocation_seconds, gpus=8):
    selected = [rows[step] for step in range(11, 101)]
    seconds = sum(row["timing_s/step"] for row in selected)
    if seconds <= 0 or allocation_seconds <= 0 or gpus < 1:
        raise ValueError("Timing and GPU allocation must be positive")
    return {"estimated_gpu_hours_900_steps": allocation_seconds * gpus / 3600 * 9,
            "samples_per_second": 512 * 90 / seconds, "steps_per_second": 90 / seconds,
            "peak_allocated_gib_per_gpu": max(row[FIELDS[1]] for row in selected),
            "peak_reserved_gib_per_gpu": max(row[FIELDS[2]] for row in selected),
            "profile_steps": 100, "measurement_steps": [11, 100],
            "samples_per_step": 512, "gpus": gpus}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--allocation-seconds", type=float, required=True,
                        help="Full allocated wall time of the 100-step job, including setup.")
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    rows = load_metrics(args.log.read_text(errors="replace"))
    print(json.dumps(summarize(rows, args.allocation_seconds, args.gpus), indent=2))


if __name__ == "__main__":
    main()
