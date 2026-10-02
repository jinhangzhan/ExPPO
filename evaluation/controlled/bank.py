"""Audit/reconstruct the frozen 64-prompt input bank by exhaustive enumeration."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

PROMPTS = Path(__file__).with_name("prompts.jsonl")
PROMPT_SHA256 = "eeee29807fd7095ebd572a7c7806847965da72fbe1bd767d7a612a9918e0b78d"


def enumerate_modes(payload):
    return sorted(",".join(map(str, counts)) for counts in itertools.product(range(payload["max_count"] + 1), repeat=4)
                  if sum(c * d for c, d in zip(counts, payload["denominations"])) == payload["target"])


def audit(path=PROMPTS):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != PROMPT_SHA256:
        raise ValueError("Frozen prompt bank fingerprint mismatch")
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    if len(rows) != 64 or [r["panel_index"] for r in rows] != list(range(64)):
        raise ValueError("Expected all 64 ordered prompts")
    for row in rows:
        actual = enumerate_modes(row["payload"])
        if actual != row["valid_mode_keys"] or len(actual) != row["oracle_mode_count"]:
            raise ValueError(f"Enumeration mismatch for prompt {row['prompt_id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional exact copy of the audited input bank")
    args = parser.parse_args()
    rows = audit()
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(PROMPTS.read_bytes())
    counts = [len(row["valid_mode_keys"]) for row in rows]
    print(f"Verified {len(rows)} prompts; {min(counts)}–{max(counts)} exact solution modes per prompt.")


if __name__ == "__main__":
    main()
