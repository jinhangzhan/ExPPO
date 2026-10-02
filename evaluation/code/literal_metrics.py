"""Bounded Python-literal parsing and pass@k metrics for code reasoning.

This module never calls :func:`eval`, :func:`exec`, or any submitted/reference
function.  A model answer is accepted only when it contains exactly one
``<final_output>...</final_output>`` pair whose payload is a bounded Python
literal.  A single assertion can be enabled for compatibility with upstream
benchmarks, but its left-hand side must match a predeclared AST and only its
literal right-hand side is inspected.
"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any


FINAL_OPEN = "<final_output>"
FINAL_CLOSE = "</final_output>"
MAX_LITERAL_CHARS = 4096
MAX_AST_NODES = 512
MAX_AST_DEPTH = 32
DEFAULT_KS = (1, 5, 8, 16)


@dataclass(frozen=True)
class LiteralParseResult:
    """Result of parsing one model completion or frozen gold literal."""

    valid: bool
    value: Any = None
    reason: str | None = None
    payload: str | None = None
    used_assert: bool = False


def _invalid(reason: str, payload: str | None = None) -> LiteralParseResult:
    return LiteralParseResult(valid=False, reason=reason, payload=payload)


def _bounded_ast(source: str, *, mode: str) -> ast.AST:
    if len(source) > MAX_LITERAL_CHARS:
        raise ValueError(f"literal exceeds {MAX_LITERAL_CHARS} characters")
    if not source:
        raise ValueError("empty literal")
    try:
        tree = ast.parse(source, mode=mode)
    except (MemoryError, RecursionError, SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid Python syntax: {type(exc).__name__}") from exc

    count = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES:
            raise ValueError(f"literal AST exceeds {MAX_AST_NODES} nodes")
        if depth > MAX_AST_DEPTH:
            raise ValueError(f"literal AST exceeds depth {MAX_AST_DEPTH}")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _validate_literal_value(value: Any, *, depth: int = 1) -> None:
    """Reject values outside the benchmark's deterministic literal subset."""

    if depth > MAX_AST_DEPTH:
        raise ValueError(f"literal value exceeds depth {MAX_AST_DEPTH}")
    if value is None or type(value) in {bool, int, str, bytes}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not accepted")
        return
    if type(value) in {list, tuple, set}:
        for item in value:
            _validate_literal_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            _validate_literal_value(key, depth=depth + 1)
            if not isinstance(key, Hashable):
                raise ValueError("dictionary key is not hashable")
            _validate_literal_value(item, depth=depth + 1)
        return
    raise ValueError(f"unsupported literal type: {type(value).__name__}")


def _literal_from_node(node: ast.AST) -> Any:
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.Call):
            continue
        if not (
            isinstance(descendant.func, ast.Name)
            and descendant.func.id == "set"
            and not descendant.args
            and not descendant.keywords
        ):
            raise ValueError("only the zero-argument set() literal call is accepted")
    try:
        value = ast.literal_eval(node)
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError("answer is not a Python literal") from exc
    _validate_literal_value(value)
    return value


def _same_ast(left: ast.AST, right: ast.AST) -> bool:
    return ast.dump(left, annotate_fields=True, include_attributes=False) == ast.dump(
        right,
        annotate_fields=True,
        include_attributes=False,
    )


def _parse_payload(
    payload: str,
    *,
    allow_assert: bool,
    expected_assert_lhs: str | None,
) -> LiteralParseResult:
    stripped = payload.strip()
    if len(stripped) > MAX_LITERAL_CHARS:
        return _invalid(f"literal exceeds {MAX_LITERAL_CHARS} characters", stripped)
    try:
        expression = _bounded_ast(stripped, mode="eval")
    except ValueError as expression_error:
        if not allow_assert:
            return _invalid(str(expression_error), stripped)
    else:
        assert isinstance(expression, ast.Expression)
        try:
            value = _literal_from_node(expression.body)
        except ValueError as exc:
            return _invalid(str(exc), stripped)
        return LiteralParseResult(valid=True, value=value, payload=stripped)

    # Compatibility mode is deliberately narrower than executing an assert:
    # exactly one assertion, no message, one equality, a predeclared LHS, and a
    # literal RHS.  Neither side is ever evaluated.
    try:
        module = _bounded_ast(stripped, mode="exec")
    except ValueError as exc:
        return _invalid(str(exc), stripped)
    if not isinstance(module, ast.Module) or len(module.body) != 1:
        return _invalid("assert mode requires exactly one statement", stripped)
    statement = module.body[0]
    if not isinstance(statement, ast.Assert) or statement.msg is not None:
        return _invalid("assert mode requires one assertion without a message", stripped)
    comparison = statement.test
    if (
        not isinstance(comparison, ast.Compare)
        or len(comparison.ops) != 1
        or len(comparison.comparators) != 1
        or not isinstance(comparison.ops[0], ast.Eq)
    ):
        return _invalid("assert mode requires exactly one == comparison", stripped)
    if not expected_assert_lhs:
        return _invalid("assert mode requires a predeclared left-hand side", stripped)
    try:
        expected_tree = _bounded_ast(expected_assert_lhs.strip(), mode="eval")
    except ValueError as exc:
        return _invalid(f"invalid predeclared assert LHS: {exc}", stripped)
    assert isinstance(expected_tree, ast.Expression)
    if not _same_ast(comparison.left, expected_tree.body):
        return _invalid("assert left-hand side does not match the prompt", stripped)
    try:
        value = _literal_from_node(comparison.comparators[0])
    except ValueError as exc:
        return _invalid(str(exc), stripped)
    return LiteralParseResult(
        valid=True,
        value=value,
        payload=stripped,
        used_assert=True,
    )


def parse_tagged_literal(
    completion: str,
    *,
    allow_assert: bool = False,
    expected_assert_lhs: str | None = None,
) -> LiteralParseResult:
    """Parse exactly one tagged literal from a model completion.

    Text outside the sole tag pair may contain reasoning, but a second opening
    or closing tag makes the submission malformed.  Only the payload is handed
    to Python's AST parser, after the explicit character limit is enforced.
    """

    if not isinstance(completion, str):
        return _invalid("completion is not a string")
    if completion.count(FINAL_OPEN) != 1 or completion.count(FINAL_CLOSE) != 1:
        return _invalid("completion must contain exactly one final_output tag pair")
    start = completion.find(FINAL_OPEN) + len(FINAL_OPEN)
    end = completion.find(FINAL_CLOSE)
    if end < start:
        return _invalid("final_output closing tag precedes opening tag")
    payload = completion[start:end].strip()
    return _parse_payload(
        payload,
        allow_assert=allow_assert,
        expected_assert_lhs=expected_assert_lhs,
    )


def parse_gold_literal(literal: Any) -> LiteralParseResult:
    """Parse a frozen common-schema ground truth without requiring tags."""

    if not isinstance(literal, str):
        return _invalid("ground truth is not a canonical literal string")
    return _parse_payload(literal, allow_assert=False, expected_assert_lhs=None)


def official_equal(prediction: Any, ground_truth: Any) -> bool:
    """Mirror the upstream Python ``==`` assertion without executing code."""

    try:
        result = prediction == ground_truth
    except Exception:
        return False
    return result if type(result) is bool else False


def strict_type_equal(prediction: Any, ground_truth: Any) -> bool:
    """Recursively require identical Python types as well as equal values."""

    if type(prediction) is not type(ground_truth):
        return False
    if type(prediction) in {list, tuple}:
        return len(prediction) == len(ground_truth) and all(
            strict_type_equal(left, right)
            for left, right in zip(prediction, ground_truth)
        )
    if type(prediction) in {dict, set}:
        if len(prediction) != len(ground_truth):
            return False
        predicted_items = prediction.items() if type(prediction) is dict else prediction
        gold_items = ground_truth.items() if type(ground_truth) is dict else ground_truth
        unmatched = list(gold_items)
        for predicted_item in predicted_items:
            for index, gold_item in enumerate(unmatched):
                if type(prediction) is dict:
                    predicted_key, predicted_value = predicted_item
                    gold_key, gold_value = gold_item
                    matches = strict_type_equal(
                        predicted_key,
                        gold_key,
                    ) and strict_type_equal(predicted_value, gold_value)
                else:
                    matches = strict_type_equal(predicted_item, gold_item)
                if matches:
                    unmatched.pop(index)
                    break
            else:
                return False
        return not unmatched
    try:
        result = prediction == ground_truth
    except Exception:
        return False
    return result if type(result) is bool else False


def score_tagged_literal(
    completion: str,
    ground_truth: Any,
    *,
    allow_assert: bool = False,
    expected_assert_lhs: str | None = None,
) -> tuple[bool, bool, LiteralParseResult]:
    """Return official-compatible correctness, strict correctness, and parse."""

    gold = parse_gold_literal(ground_truth)
    if not gold.valid:
        raise ValueError(f"invalid frozen ground truth: {gold.reason}")
    parsed = parse_tagged_literal(
        completion,
        allow_assert=allow_assert,
        expected_assert_lhs=expected_assert_lhs,
    )
    if not parsed.valid:
        return False, False, parsed
    return (
        official_equal(parsed.value, gold.value),
        strict_type_equal(parsed.value, gold.value),
        parsed,
    )


def pass_at_k_estimate(num_samples: int, num_correct: int, k: int) -> float:
    """Return the standard unbiased pass@k estimate for one problem."""

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if not 0 <= num_correct <= num_samples:
        raise ValueError("num_correct must be between zero and num_samples")
    if not 1 <= k <= num_samples:
        raise ValueError("k must be between one and num_samples")
    failures = num_samples - num_correct
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(num_samples, k)


def pass_metrics(
    attempt_flags: Sequence[Sequence[bool]],
    *,
    requested_ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float | int]:
    """Compute unbiased pass@k, avg@n, and first-attempt accuracy."""

    if not attempt_flags:
        raise ValueError("cannot compute metrics for an empty evaluation")
    widths = {len(flags) for flags in attempt_flags}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("every problem must have the same positive attempt count")
    pool_size = widths.pop()
    ks = sorted(set(requested_ks) | {pool_size})
    invalid = [k for k in ks if not 1 <= k <= pool_size]
    if invalid:
        raise ValueError(f"requested k exceeds sample pool n={pool_size}: {invalid}")

    result: dict[str, float | int] = {
        "sample_pool_size": pool_size,
        "samples": len(attempt_flags),
        "first_attempt_accuracy": sum(bool(flags[0]) for flags in attempt_flags)
        / len(attempt_flags),
    }
    for k in ks:
        result[f"pass_at_{k}"] = sum(
            pass_at_k_estimate(pool_size, sum(bool(flag) for flag in flags), k)
            for flags in attempt_flags
        ) / len(attempt_flags)
    avg = sum(
        sum(bool(flag) for flag in flags) / pool_size for flags in attempt_flags
    ) / len(attempt_flags)
    result[f"avg_at_{pool_size}"] = avg
    result["pass_at_k"] = result[f"pass_at_{pool_size}"]
    result["avg_at_k"] = avg
    return result


def prefixed_metrics(
    attempt_flags: Sequence[Sequence[bool]],
    *,
    prefix: str,
    requested_ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float | int]:
    """Prefix metric names while leaving count metadata unambiguous."""

    metrics = pass_metrics(attempt_flags, requested_ks=requested_ks)
    return {
        f"{prefix}{key}": value
        for key, value in metrics.items()
        if key not in {"sample_pool_size", "samples"}
    }


def cluster_macro_metrics(
    attempt_flags: Sequence[Sequence[bool]],
    cluster_ids: Sequence[str],
    *,
    requested_ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float | int]:
    """Equal-weight clusters after averaging row-level metrics in each cluster."""

    if len(attempt_flags) != len(cluster_ids):
        raise ValueError("attempt rows and cluster IDs have different lengths")
    if not attempt_flags:
        raise ValueError("cannot compute cluster metrics for an empty evaluation")
    widths = {len(flags) for flags in attempt_flags}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("every problem must have the same positive attempt count")
    pool_size = widths.pop()
    ks = sorted(set(requested_ks) | {pool_size})
    if any(not 1 <= k <= pool_size for k in ks):
        raise ValueError("cluster metric k exceeds sample pool")

    grouped: dict[str, list[Sequence[bool]]] = defaultdict(list)
    for cluster_id, flags in zip(cluster_ids, attempt_flags):
        grouped[str(cluster_id)].append(flags)

    metric_names = [f"pass_at_{k}" for k in ks]
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    values["first_attempt_accuracy"] = []
    values[f"avg_at_{pool_size}"] = []
    for rows in grouped.values():
        row_metrics = pass_metrics(rows, requested_ks=ks)
        for name in values:
            values[name].append(float(row_metrics[name]))

    result: dict[str, float | int] = {
        "cluster_count": len(grouped),
        "cluster_size_min": min(map(len, grouped.values())),
        "cluster_size_max": max(map(len, grouped.values())),
        "cluster_size_mean": len(attempt_flags) / len(grouped),
    }
    for name, per_cluster in values.items():
        result[f"cluster_macro_{name}"] = sum(per_cluster) / len(per_cluster)
    result["cluster_macro_pass_at_k"] = result[f"cluster_macro_pass_at_{pool_size}"]
    result["cluster_macro_avg_at_k"] = result[f"cluster_macro_avg_at_{pool_size}"]
    return result


def percentile(values: Sequence[int], quantile: float) -> float:
    """Return a linearly interpolated quantile without third-party packages."""

    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def recursive_plain_value(value: Any) -> Any:
    """Return a JSON-friendly rendering that preserves non-JSON literals."""

    if type(value) is bytes:
        return {"python_type": "bytes", "repr": repr(value)}
    if type(value) is tuple:
        return {
            "python_type": "tuple",
            "items": [recursive_plain_value(item) for item in value],
        }
    if type(value) is list:
        return [recursive_plain_value(item) for item in value]
    if type(value) is set:
        return {
            "python_type": "set",
            "items": [
                recursive_plain_value(item)
                for item in sorted(value, key=lambda item: (type(item).__name__, repr(item)))
            ],
        }
    if type(value) is dict:
        return {
            "python_type": "dict",
            "items": [
                [recursive_plain_value(key), recursive_plain_value(item)]
                for key, item in value.items()
            ],
        }
    return value

