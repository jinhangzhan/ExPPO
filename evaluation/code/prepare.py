#!/usr/bin/env python3
"""Build deterministic CodeReasoner PPO data and frozen output eval sets.

The builder is deliberately non-executing: it parses source programs only to
normalize their ASTs, parses calls only as syntax, and validates expected
answers with the bounded ``ast.literal_eval`` layer shared with training.
Neither reference code nor model-produced code is ever run.

All CodeReasoner source rows are converted to forward output prediction from
their frozen ``func_str``, ``func_args``, and ``result`` fields.  This avoids
the non-unique inverse-input task while retaining both halves of the released
RL corpus.  Exact ``(func_str, func_args)`` duplicates are removed before
whole normalized-function groups are assigned to train, validation, ID test,
or an unused reserve.

Run ``python scripts/prepare_codereasoner.py`` to download and prepare the pinned data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import platform
import sys
import tempfile
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.code.reward import (
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_LITERAL_CHARS,
    PYTHON_FEATURE_VERSION,
    LiteralValidationError,
    parse_literal_bounded,
)


DEFAULT_ROOT = Path("data/codereasoner")
DEFAULT_RAW_ROOT = DEFAULT_ROOT / "raw"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "prepared"

CODEREASONER_FILENAME = "codereasoner_dataset_rl_08c784b2.json"
CRUXEVAL_FILENAME = "cruxeval_test_b96af045.jsonl"
LCB_FILENAME = "livecodebench_execution_v2_ff6ea0e2.parquet"

SYSTEM_PROMPT = """You predict the exact return value of a Python function call.

Analyze the provided source code and call without running either one. Reason step by step, then finish with exactly one tag pair containing only a valid Python literal:
<final_output>YOUR_LITERAL</final_output>

Do not put an assertion, prose, Markdown, or another final_output tag inside the tag pair."""


@dataclass(frozen=True)
class SourceSpec:
    key: str
    repository: str
    revision: str
    remote_path: str
    local_filename: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repository}/resolve/"
            f"{self.revision}/{self.remote_path}"
        )


SOURCES = {
    "codereasoner": SourceSpec(
        key="codereasoner",
        repository="just-for-paper-submission/CodeReasoner-Dataset",
        revision="08c784b2aa0e58e511c0cfdbdf61a61380a8c92f",
        remote_path="dataset_rl.json",
        local_filename=CODEREASONER_FILENAME,
        sha256="2d4e18fa896bd8c84b54f4a1ae521d9e379f7589efe314822d50e5b352295d9b",
    ),
    "cruxeval": SourceSpec(
        key="cruxeval",
        repository="cruxeval-org/cruxeval",
        revision="b96af0450242eb4da433032b90998f25588a5d0f",
        remote_path="test.jsonl",
        local_filename=CRUXEVAL_FILENAME,
        sha256="8368b81047dc5014e4caf5a2f97604eff7644e0ecd7415e3ceeb184bbc2e0c96",
    ),
    "livecodebench": SourceSpec(
        key="livecodebench",
        repository="livecodebench/execution-v2",
        revision="ff6ea0e2a638001006ddcc41259eff23a4283fb2",
        remote_path="data/test-00000-of-00001.parquet",
        local_filename=LCB_FILENAME,
        sha256="c54e4377642f689758c2ee8664457af1b95ba95813a18208003c7a5a1efa410f",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--validation-size", type=int, default=1_800)
    parser.add_argument("--id-test-size", type=int, default=1_800)
    parser.add_argument("--dry-run", action="store_true", help="validate and plan without writing")
    parser.add_argument("--force", action="store_true", help="replace canonical generated artifacts")
    return parser.parse_args(argv)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: Path, spec: SourceSpec) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pinned {spec.key} source {path}; expected {spec.url}"
        )
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise RuntimeError(
            f"Pinned source hash mismatch for {path}: got {actual}, expected {spec.sha256}"
        )
    return {
        "key": spec.key,
        "repository": spec.repository,
        "revision": spec.revision,
        "remote_path": spec.remote_path,
        "url": spec.url,
        "local_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def parse_python(source: str, *, mode: str) -> ast.AST:
    return ast.parse(
        source,
        mode=mode,
        feature_version=PYTHON_FEATURE_VERSION,
    )


def _reject_post_py311_fstring_syntax(source: str) -> None:
    """Reject PEP 701 backslashes that Python 3.11 cannot parse.

    ``ast.parse(feature_version=(3, 11))`` on Python 3.12+ still accepts some
    PEP 701 f-string constructs.  On runtimes exposing f-string tokens, track
    replacement-field braces and reject a backslash-bearing expression token,
    matching Python 3.11's grammar without evaluating the source.
    """

    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)
    if fstring_start is None or fstring_end is None:
        return
    fstring_levels: list[int] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for item in tokens:
            if item.type == fstring_start:
                fstring_levels.append(0)
                continue
            if item.type == fstring_end:
                if fstring_levels:
                    fstring_levels.pop()
                continue
            if not fstring_levels:
                continue
            if item.type == tokenize.OP and item.string == "{":
                fstring_levels[-1] += 1
                continue
            if item.type == tokenize.OP and item.string == "}":
                fstring_levels[-1] = max(0, fstring_levels[-1] - 1)
                continue
            if fstring_levels[-1] > 0 and "\\" in item.string:
                raise SyntaxError("f-string expression contains a backslash (Python 3.11 target)")
    except tokenize.TokenError as exc:
        raise SyntaxError(f"invalid token stream: {exc}") from exc


class _FunctionNameAnonymizer(ast.NodeTransformer):
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name == self.function_name:
            node.name = "__function__"
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if node.name == self.function_name:
            node.name = "__function__"
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.function_name:
            node.id = "__function__"
        return node


def canonical_ast_dump(tree: ast.AST) -> str:
    """Retain the original Python 3.13 AST hash on Python 3.11 as well.

    Python 3.12 added empty type-parameter lists to function/class AST nodes.
    Source syntax is restricted to Python 3.11, so these lists are always empty;
    explicitly adding them on older interpreters preserves the released split.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if "type_params" not in node._fields:
                node._fields = (*node._fields, "type_params")
                node.type_params = []
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def normalized_function(
    code: str,
    *,
    expected_function_name: str | None = None,
) -> tuple[str, str]:
    _reject_post_py311_fstring_syntax(code)
    module = parse_python(code, mode="exec")
    if not isinstance(module, ast.Module):
        raise ValueError("function source did not parse as a module")
    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if expected_function_name is None:
        if len(functions) != 1:
            raise ValueError(f"expected exactly one top-level function, found {len(functions)}")
        function_name = functions[0].name
    else:
        matching = [node.name for node in functions if node.name == expected_function_name]
        if len(matching) != 1:
            raise ValueError(
                f"expected one top-level function named {expected_function_name!r}, found {matching}"
            )
        function_name = expected_function_name
    anonymized = _FunctionNameAnonymizer(function_name).visit(module)
    ast.fix_missing_locations(anonymized)
    normalized = canonical_ast_dump(anonymized)
    return function_name, sha256_text(normalized)


def normalize_call(call_text: str, function_hash: str) -> str:
    expression = parse_python(call_text, mode="eval")
    if not isinstance(expression, ast.Expression) or not isinstance(expression.body, ast.Call):
        raise ValueError("input must be exactly one Python call expression")
    call = expression.body
    anonymized = ast.Call(
        func=ast.Name(id="__function__", ctx=ast.Load()),
        args=call.args,
        keywords=call.keywords,
    )
    normalized = canonical_ast_dump(anonymized)
    return sha256_text(f"{function_hash}\0{normalized}")


def original_prompt_direction(prompt: str) -> str:
    if "== ????" in prompt:
        return "forward_output"
    if "????" in prompt:
        return "backward_input_repurposed_to_forward_output"
    raise ValueError("CodeReasoner prompt has no ???? marker")


def codereasoner_call(function_name: str, raw_args: str, direction: str) -> str:
    fragment = raw_args.strip()
    candidates = [fragment]
    if direction.startswith("backward_") and fragment.startswith("(") and fragment.endswith(")"):
        # Backward records store the replacement text including an outer call
        # parenthesis pair.  Removing exactly that pair recovers the argument
        # fragment.  Parsing below is syntax-only and never invokes the call.
        candidates = [fragment[1:-1].strip(), fragment]
    errors: list[str] = []
    for candidate in candidates:
        call_text = f"{function_name}({candidate})"
        try:
            expression = parse_python(call_text, mode="eval")
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(expression, ast.Expression) and isinstance(expression.body, ast.Call):
            return call_text
    raise ValueError(f"could not normalize CodeReasoner call syntax: {errors}")


def user_prompt(code: str, call_text: str) -> str:
    return (
        "Predict the return value of this function call.\n\n"
        f"Python source:\n```python\n{code.strip()}\n```\n\n"
        f"Call:\n```python\n{call_text.strip()}\n```"
    )


def blank_extra_info() -> dict[str, Any]:
    return {
        "index": None,
        "split": None,
        "source_repository": None,
        "source_revision": None,
        "source_row_index": None,
        "source_id": None,
        "original_prompt_direction": None,
        "function_name": None,
        "normalized_function_hash": None,
        "code_input_hash": None,
        "normalized_call": None,
        "source_input": None,
        "question_id": None,
        "sample_id": None,
        "problem_id": None,
        "contest_id": None,
        "contest_date": None,
        "difficulty": None,
        "numsteps": None,
    }


def common_row(
    *,
    example_id: str,
    dataset: str,
    data_source: str,
    code: str,
    call_text: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    parse_literal_bounded(ground_truth)
    return {
        "id": example_id,
        "dataset": dataset,
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(code, call_text)},
        ],
        "ability": "code_output_reasoning",
        "reward_model": {"style": "rule", "ground_truth": ground_truth.strip()},
        "extra_info": extra_info,
    }


def load_codereasoner(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError("CodeReasoner source must be one JSON list")

    spec = SOURCES["codereasoner"]
    seen_pairs: dict[tuple[str, str], tuple[str, int]] = {}
    rows: list[dict[str, Any]] = []
    duplicate_rows = 0
    duplicate_groups: set[tuple[str, str]] = set()
    rejected: list[dict[str, Any]] = []
    directions = Counter()

    for source_index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise RuntimeError(f"CodeReasoner row {source_index} is not an object")
        required = ("func_str", "func_args", "result", "prompt_str")
        if any(not isinstance(raw.get(field), str) for field in required):
            raise RuntimeError(f"CodeReasoner row {source_index} has missing/non-string fields")
        code = raw["func_str"]
        raw_args = raw["func_args"]
        expected = raw["result"].strip()
        pair = (code, raw_args)
        if pair in seen_pairs:
            previous_expected, previous_index = seen_pairs[pair]
            if expected != previous_expected:
                raise RuntimeError(
                    "Conflicting outputs for exact duplicate CodeReasoner pair at "
                    f"rows {previous_index} and {source_index}"
                )
            duplicate_rows += 1
            duplicate_groups.add(pair)
            continue
        seen_pairs[pair] = (expected, source_index)

        try:
            direction = original_prompt_direction(raw["prompt_str"])
            function_name, function_hash = normalized_function(code)
            call_text = codereasoner_call(function_name, raw_args, direction)
            input_hash = normalize_call(call_text, function_hash)
            parse_literal_bounded(expected)
        except (SyntaxError, ValueError, LiteralValidationError) as exc:
            rejected.append(
                {
                    "source_row_index": source_index,
                    "pair_sha256": sha256_text(f"{code}\0{raw_args}"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        directions[direction] += 1
        source_id = sha256_text(f"{code}\0{raw_args}")
        extra = blank_extra_info()
        extra.update(
            {
                "source_repository": spec.repository,
                "source_revision": spec.revision,
                "source_row_index": source_index,
                "source_id": source_id,
                "original_prompt_direction": direction,
                "function_name": function_name,
                "normalized_function_hash": function_hash,
                "code_input_hash": input_hash,
                "normalized_call": call_text,
                "source_input": raw_args,
            }
        )
        rows.append(
            common_row(
                example_id=f"codereasoner:{source_id}",
                dataset="CodeReasoner-unassigned",
                data_source="codereasoner",
                code=code,
                call_text=call_text,
                ground_truth=expected,
                extra_info=extra,
            )
        )

    diagnostics = {
        "source_rows": len(payload),
        "unique_exact_code_args_pairs": len(seen_pairs),
        "exact_duplicate_rows_removed": duplicate_rows,
        "exact_duplicate_pair_groups": len(duplicate_groups),
        "invalid_rows_rejected": len(rejected),
        "rejected_rows": rejected,
        "validated_candidate_rows": len(rows),
        "original_prompt_directions_after_filtering": dict(sorted(directions.items())),
    }
    expected_counts = {
        "source_rows": 18_796,
        "unique_exact_code_args_pairs": 18_608,
        "exact_duplicate_rows_removed": 188,
        "invalid_rows_rejected": 1,
        "validated_candidate_rows": 18_607,
    }
    mismatched = {
        key: {"actual": diagnostics[key], "expected": expected}
        for key, expected in expected_counts.items()
        if diagnostics[key] != expected
    }
    if mismatched:
        raise RuntimeError(f"Pinned CodeReasoner preparation counts changed: {mismatched}")
    return rows, diagnostics


def load_cruxeval(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = SOURCES["cruxeval"]
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            raw = json.loads(line)
            for field in ("id", "code", "input", "output"):
                if not isinstance(raw.get(field), str):
                    raise RuntimeError(f"CRUXEval row {source_index} has invalid {field}")
            function_name, function_hash = normalized_function(raw["code"])
            call_text = f"{function_name}({raw['input'].strip()})"
            input_hash = normalize_call(call_text, function_hash)
            expected = raw["output"].strip()
            parse_literal_bounded(expected)
            extra = blank_extra_info()
            extra.update(
                {
                    "index": source_index,
                    "split": "cruxeval_output_test",
                    "source_repository": spec.repository,
                    "source_revision": spec.revision,
                    "source_row_index": source_index,
                    "source_id": raw["id"],
                    "function_name": function_name,
                    "normalized_function_hash": function_hash,
                    "code_input_hash": input_hash,
                    "normalized_call": call_text,
                    "source_input": raw["input"],
                    "sample_id": raw["id"],
                }
            )
            rows.append(
                common_row(
                    example_id=f"cruxeval:{raw['id']}",
                    dataset="CRUXEval-output-test",
                    data_source="cruxeval_output",
                    code=raw["code"],
                    call_text=call_text,
                    ground_truth=expected,
                    extra_info=extra,
                )
            )
    if len(rows) != 800:
        raise RuntimeError(f"Pinned CRUXEval source has {len(rows)} rows, expected 800")
    return rows, {
        "source_rows": len(rows),
        "artifact_rows": len(rows),
        "unique_exact_code_input_pairs": len(
            {(row["extra_info"]["code_input_hash"]) for row in rows}
        ),
        "unique_normalized_functions": len(
            {row["extra_info"]["normalized_function_hash"] for row in rows}
        ),
    }


def load_livecodebench(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = SOURCES["livecodebench"]
    # Preserve the source builder's DuckDB reader and uncompressed PyArrow outputs.
    connection = duckdb.connect(database=":memory:")
    result = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
    columns = [description[0] for description in result.description]
    raw_rows = [dict(zip(columns, values)) for values in result.fetchall()]
    connection.close()
    rows: list[dict[str, Any]] = []
    exact_pairs = Counter()
    for source_index, raw in enumerate(raw_rows):
        for field in ("id", "function_name", "code", "input", "output"):
            if not isinstance(raw.get(field), str):
                raise RuntimeError(f"LiveCodeBench row {source_index} has invalid {field}")
        function_hash_name, function_hash = normalized_function(
            raw["code"],
            expected_function_name=raw["function_name"],
        )
        call_text = raw["input"].strip()
        input_hash = normalize_call(call_text, function_hash)
        expected = raw["output"].strip()
        parse_literal_bounded(expected)
        exact_pairs[(raw["code"], raw["input"])] += 1
        question_id = raw.get("question_id")
        if not isinstance(question_id, int):
            raise RuntimeError(f"LiveCodeBench row {source_index} has invalid question_id")
        contest_date = raw.get("contest_date")
        contest_date_text = contest_date.isoformat() if hasattr(contest_date, "isoformat") else str(contest_date)
        problem_id = raw.get("problem_id")
        if problem_id is not None:
            problem_id = [int(value) for value in problem_id]
        extra = blank_extra_info()
        extra.update(
            {
                "index": source_index,
                "split": "livecodebench_execution_v2_test",
                "source_repository": spec.repository,
                "source_revision": spec.revision,
                "source_row_index": source_index,
                "source_id": raw["id"],
                "function_name": raw["function_name"] or function_hash_name,
                "normalized_function_hash": function_hash,
                "code_input_hash": input_hash,
                "normalized_call": call_text,
                "source_input": raw["input"],
                "question_id": question_id,
                "sample_id": raw["id"],
                "problem_id": problem_id,
                "contest_id": None if raw.get("contest_id") is None else str(raw["contest_id"]),
                "contest_date": contest_date_text,
                "difficulty": None if raw.get("difficulty") is None else str(raw["difficulty"]),
                "numsteps": None if raw.get("numsteps") is None else int(raw["numsteps"]),
            }
        )
        rows.append(
            common_row(
                example_id=f"livecodebench:{question_id}:{raw['id']}",
                dataset="LiveCodeBench-execution-v2-test",
                data_source="livecodebench_output",
                code=raw["code"],
                call_text=call_text,
                ground_truth=expected,
                extra_info=extra,
            )
        )
    if len(rows) != 479:
        raise RuntimeError(f"Pinned LiveCodeBench source has {len(rows)} rows, expected 479")
    duplicate_groups = [count for count in exact_pairs.values() if count > 1]
    return rows, {
        "source_rows": len(rows),
        "artifact_rows": len(rows),
        "official_rows_preserved_without_deduplication": True,
        "unique_exact_code_input_pairs": len(exact_pairs),
        "diagnostic_exact_duplicate_pair_groups": len(duplicate_groups),
        "diagnostic_exact_duplicate_extra_rows": sum(count - 1 for count in duplicate_groups),
        "unique_question_ids": len({row["extra_info"]["question_id"] for row in rows}),
        "unique_normalized_functions": len(
            {row["extra_info"]["normalized_function_hash"] for row in rows}
        ),
    }


def stable_order_key(seed: int, namespace: str, value: str) -> tuple[str, str]:
    return sha256_text(f"{seed}\0{namespace}\0{value}"), value


def code_input_hash_diagnostics(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["extra_info"]["code_input_hash"] for row in rows)
    duplicate_groups = [count for count in counts.values() if count > 1]
    return {
        "code_input_rows": len(rows),
        "unique_code_input_hashes": len(counts),
        "duplicate_code_input_hash_groups": len(duplicate_groups),
        "duplicate_code_input_extra_rows": sum(count - 1 for count in duplicate_groups),
    }


def allocate_splits(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    targets: list[tuple[str, int, str]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["extra_info"]["normalized_function_hash"]].append(row)
    available = sorted(groups, key=lambda key: stable_order_key(seed, "function-group", key))
    selected: dict[str, list[dict[str, Any]]] = {}
    split_diagnostics: dict[str, Any] = {}

    for split_name, target, dataset_name in targets:
        picked_hashes: list[str] = []
        remaining_hashes: list[str] = []
        count = 0
        for function_hash in available:
            group_size = len(groups[function_hash])
            if count + group_size <= target:
                picked_hashes.append(function_hash)
                count += group_size
            else:
                remaining_hashes.append(function_hash)
        if count != target:
            raise RuntimeError(
                f"Could not allocate whole function groups for {split_name}: got {count}, target {target}"
            )
        rows = [row for function_hash in picked_hashes for row in groups[function_hash]]
        rows.sort(
            key=lambda row: stable_order_key(seed, split_name, row["id"])
        )
        for index, row in enumerate(rows):
            row["dataset"] = dataset_name
            row["extra_info"]["index"] = index
            row["extra_info"]["split"] = split_name
        selected[split_name] = rows
        split_diagnostics[split_name] = {
            "rows": len(rows),
            "normalized_function_groups": len(picked_hashes),
            "function_group_set_sha256": sha256_text("\n".join(sorted(picked_hashes))),
            **code_input_hash_diagnostics(rows),
        }
        available = remaining_hashes

    unused = [row for function_hash in available for row in groups[function_hash]]
    unused.sort(key=lambda row: stable_order_key(seed, "unused", row["id"]))
    split_diagnostics["unused"] = {
        "rows": len(unused),
        "normalized_function_groups": len(available),
        "function_group_set_sha256": sha256_text("\n".join(sorted(available))),
        **code_input_hash_diagnostics(unused),
    }
    return selected, unused, split_diagnostics


def hash_set(rows: Iterable[dict[str, Any]], field: str) -> set[str]:
    return {str(row["extra_info"][field]) for row in rows}


def overlap_counts(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "normalized_function_hashes": len(
            hash_set(left, "normalized_function_hash")
            & hash_set(right, "normalized_function_hash")
        ),
        "code_input_hashes": len(
            hash_set(left, "code_input_hash") & hash_set(right, "code_input_hash")
        ),
    }


def validate_common_rows(rows: list[dict[str, Any]], expected_source: str) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{expected_source}: generated IDs are not unique")
    for row in rows:
        if row["data_source"] != expected_source:
            raise RuntimeError(f"{expected_source}: invalid data_source")
        if row["ability"] != "code_output_reasoning":
            raise RuntimeError(f"{expected_source}: invalid ability")
        if [message["role"] for message in row["prompt"]] != ["system", "user"]:
            raise RuntimeError(f"{expected_source}: invalid prompt roles")
        parse_literal_bounded(row["reward_model"]["ground_truth"])


def arrow_schema() -> pa.Schema:
    message = pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
    reward_model = pa.struct([pa.field("style", pa.string()), pa.field("ground_truth", pa.string())])
    extra_info = pa.struct(
        [
            pa.field("index", pa.int64()),
            pa.field("split", pa.string()),
            pa.field("source_repository", pa.string()),
            pa.field("source_revision", pa.string()),
            pa.field("source_row_index", pa.int64()),
            pa.field("source_id", pa.string()),
            pa.field("original_prompt_direction", pa.string()),
            pa.field("function_name", pa.string()),
            pa.field("normalized_function_hash", pa.string()),
            pa.field("code_input_hash", pa.string()),
            pa.field("normalized_call", pa.string()),
            pa.field("source_input", pa.string()),
            pa.field("question_id", pa.int64()),
            pa.field("sample_id", pa.string()),
            pa.field("problem_id", pa.list_(pa.int64())),
            pa.field("contest_id", pa.string()),
            pa.field("contest_date", pa.string()),
            pa.field("difficulty", pa.string()),
            pa.field("numsteps", pa.int64()),
        ]
    )
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("data_source", pa.string()),
            pa.field("prompt", pa.list_(message)),
            pa.field("ability", pa.string()),
            pa.field("reward_model", reward_model),
            pa.field("extra_info", extra_info),
        ]
    )


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows, schema=arrow_schema())
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression=None)
        roundtrip = pq.read_table(temporary)
        if roundtrip.num_rows != len(rows) or not roundtrip.schema.equals(table.schema):
            raise RuntimeError(f"Parquet round-trip validation failed for {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        with temporary.open("r", encoding="utf-8") as handle:
            checked = sum(1 for line in handle if json.loads(line))
        if checked != len(rows):
            raise RuntimeError(f"JSONL round-trip validation failed for {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        with temporary.open("r", encoding="utf-8") as handle:
            json.load(handle)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def artifact_paths(
    output_dir: Path,
    train_size: int,
    validation_size: int,
    id_test_size: int,
    cruxeval_size: int,
    lcb_size: int,
) -> dict[str, tuple[Path, ...]]:
    return {
        "train": (output_dir / f"codereasoner_train_{train_size}.nosnappy.parquet",),
        "validation": (
            output_dir / f"codereasoner_validation_{validation_size}.nosnappy.parquet",
        ),
        "id_test": (
            output_dir / f"codereasoner_id_test_{id_test_size}.nosnappy.parquet",
            output_dir / f"codereasoner_id_test_{id_test_size}.jsonl",
        ),
        "cruxeval": (
            output_dir / f"cruxeval_output_test_{cruxeval_size}.nosnappy.parquet",
            output_dir / f"cruxeval_output_test_{cruxeval_size}.jsonl",
        ),
        "livecodebench": (
            output_dir / f"livecodebench_execution_v2_test_{lcb_size}.nosnappy.parquet",
            output_dir / f"livecodebench_execution_v2_test_{lcb_size}.jsonl",
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if min(args.train_size, args.validation_size, args.id_test_size) <= 0:
        raise ValueError("all CodeReasoner split sizes must be positive")
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    source_paths = {
        key: raw_root / spec.local_filename for key, spec in SOURCES.items()
    }
    source_manifest = [
        validate_source(source_paths[key], SOURCES[key])
        for key in ("codereasoner", "cruxeval", "livecodebench")
    ]

    cruxeval_rows, cruxeval_diagnostics = load_cruxeval(source_paths["cruxeval"])
    lcb_rows, lcb_diagnostics = load_livecodebench(source_paths["livecodebench"])
    candidate_rows, codereasoner_diagnostics = load_codereasoner(
        source_paths["codereasoner"]
    )

    external_function_hashes = hash_set(cruxeval_rows, "normalized_function_hash") | hash_set(
        lcb_rows, "normalized_function_hash"
    )
    raw_external_overlap_rows = [
        row
        for row in candidate_rows
        if row["extra_info"]["normalized_function_hash"] in external_function_hashes
    ]
    eligible_rows = [
        row
        for row in candidate_rows
        if row["extra_info"]["normalized_function_hash"] not in external_function_hashes
    ]
    external_excluded_hashes = hash_set(raw_external_overlap_rows, "normalized_function_hash")

    selected, unused, split_diagnostics = allocate_splits(
        eligible_rows,
        seed=args.seed,
        targets=[
            ("train", args.train_size, "CodeReasoner-train"),
            ("validation", args.validation_size, "CodeReasoner-validation"),
            ("id_test", args.id_test_size, "CodeReasoner-ID-test"),
        ],
    )
    if (args.seed, args.train_size, args.validation_size, args.id_test_size) == (42, 12000, 1800, 1800):
        expected_groups = {
            "train": "9a4de7edcb98d94d08ded47bdfc5c4e9e3bf0dc93b22acdecc1ba2740925abe2",
            "validation": "1228c4917dac54d880db8af92d3fe47d7bdfeb4e7eae5c3efed6863befae1943",
            "id_test": "05936a2c978a94cc4ea7528762c03dd303f741460a12d2bf592df32fa3deafb2",
        }
        for split, expected in expected_groups.items():
            if split_diagnostics[split]["function_group_set_sha256"] != expected:
                raise RuntimeError(f"Original {split} function-group assignment changed; check AST/runtime compatibility")

    train_rows = selected["train"]
    validation_rows = selected["validation"]
    id_test_rows = selected["id_test"]
    validate_common_rows(train_rows, "codereasoner")
    validate_common_rows(validation_rows, "codereasoner")
    validate_common_rows(id_test_rows, "codereasoner")
    validate_common_rows(cruxeval_rows, "cruxeval_output")
    validate_common_rows(lcb_rows, "livecodebench_output")

    named_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "id_test": id_test_rows,
        "cruxeval": cruxeval_rows,
        "livecodebench": lcb_rows,
    }
    overlap_pairs = {
        "train_vs_validation": overlap_counts(train_rows, validation_rows),
        "train_vs_id_test": overlap_counts(train_rows, id_test_rows),
        "validation_vs_id_test": overlap_counts(validation_rows, id_test_rows),
        "candidate_vs_cruxeval": overlap_counts(candidate_rows, cruxeval_rows),
        "candidate_vs_livecodebench": overlap_counts(candidate_rows, lcb_rows),
        "train_vs_cruxeval": overlap_counts(train_rows, cruxeval_rows),
        "train_vs_livecodebench": overlap_counts(train_rows, lcb_rows),
        "validation_vs_cruxeval": overlap_counts(validation_rows, cruxeval_rows),
        "validation_vs_livecodebench": overlap_counts(validation_rows, lcb_rows),
        "id_test_vs_cruxeval": overlap_counts(id_test_rows, cruxeval_rows),
        "id_test_vs_livecodebench": overlap_counts(id_test_rows, lcb_rows),
        "cruxeval_vs_livecodebench": overlap_counts(cruxeval_rows, lcb_rows),
    }
    for pair in ("train_vs_validation", "train_vs_id_test", "validation_vs_id_test"):
        if overlap_pairs[pair]["normalized_function_hashes"]:
            raise RuntimeError(f"Function leakage detected: {pair}={overlap_pairs[pair]}")
    for pair in ("train_vs_cruxeval", "train_vs_livecodebench"):
        if any(overlap_pairs[pair].values()):
            raise RuntimeError(f"External evaluation leakage detected: {pair}={overlap_pairs[pair]}")

    paths = artifact_paths(
        output_dir,
        args.train_size,
        args.validation_size,
        args.id_test_size,
        len(cruxeval_rows),
        len(lcb_rows),
    )
    manifest_path = output_dir / "code_reasoning_manifest.json"
    prospective = [path for group in paths.values() for path in group] + [manifest_path]
    if not args.dry_run and not args.force:
        existing = [str(path) for path in prospective if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite existing canonical outputs without --force: {existing}"
            )

    for name, rows in named_rows.items():
        print(
            f"[validated] {name}: rows={len(rows)} "
            f"functions={len(hash_set(rows, 'normalized_function_hash'))}"
        )
        for path in paths[name]:
            print(f"[planned] {path}")
    print(f"[validated] unused: rows={len(unused)}")
    print(f"[overlap] {json.dumps(overlap_pairs, sort_keys=True)}")
    if args.dry_run:
        print("[dry-run] no files written")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for name, rows in named_rows.items():
        for path in paths[name]:
            if path.suffix == ".jsonl":
                atomic_jsonl(path, rows)
                artifact_format = "jsonl"
            else:
                atomic_parquet(path, rows)
                artifact_format = "parquet-uncompressed"
            artifact = {
                "name": name,
                "format": artifact_format,
                "path": str(path),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            artifacts.append(artifact)
            print(f"[wrote] {path} rows={len(rows)} sha256={artifact['sha256']}")

    manifest = {
        "format_version": 1,
        "builder": str(Path(__file__).resolve()),
        "python_runtime": platform.python_version(),
        "seed": args.seed,
        "canonical_root": str(DEFAULT_ROOT),
        "deduplication_scope": "CodeReasoner only; official external eval rows are preserved",
        "forward_only_conversion": (
            "Every retained CodeReasoner row uses func_str + normalized func_args call -> result; "
            "the original backward-input prompt is never used for scoring."
        ),
        "normalization": {
            "python_feature_version": ".".join(str(value) for value in PYTHON_FEATURE_VERSION),
            "function_hash": (
                "sha256(AST dump after renaming the target top-level function and every Name reference "
                "to __function__; attributes excluded)"
            ),
            "code_input_hash": "sha256(function_hash + NUL + anonymized call AST dump)",
            "split_group_key": "normalized_function_hash",
        },
        "literal_contract": {
            "parser": "ast.literal_eval after explicit AST validation; no eval/exec/source execution",
            "max_chars": MAX_LITERAL_CHARS,
            "max_ast_nodes": MAX_AST_NODES,
            "max_ast_depth": MAX_AST_DEPTH,
            "complex_allowed": False,
            "model_tag_pairs_required": 1,
            "equality": "Python assertion == semantics, including bool/int equality",
        },
        "sources": source_manifest,
        "codereasoner_preparation": {
            **codereasoner_diagnostics,
            "candidate_code_input_hash_diagnostics": code_input_hash_diagnostics(
                candidate_rows
            ),
            "eligible_code_input_hash_diagnostics": code_input_hash_diagnostics(
                eligible_rows
            ),
            "external_normalized_function_groups_excluded_before_split": len(
                external_excluded_hashes
            ),
            "external_overlap_rows_excluded_before_split": len(raw_external_overlap_rows),
            "eligible_rows_after_external_exclusion": len(eligible_rows),
            "target_rows": {
                "train": args.train_size,
                "validation": args.validation_size,
                "id_test": args.id_test_size,
            },
            "unused_rows": len(unused),
        },
        "external_evaluation_preparation": {
            "cruxeval": cruxeval_diagnostics,
            "livecodebench_execution_v2": lcb_diagnostics,
        },
        "split_diagnostics": split_diagnostics,
        "overlap_audit": {
            "counts_are_unique_hash_intersections": True,
            "pairs": overlap_pairs,
            "train_external_overlap_policy": (
                "Exclude any candidate normalized-function group matching external eval before splitting; "
                "then fail if train still has normalized-code or code-input overlap."
            ),
        },
        "artifacts": artifacts,
    }
    atomic_json(manifest_path, manifest)
    print(f"[wrote] {manifest_path} sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
