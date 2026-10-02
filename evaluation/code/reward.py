#!/usr/bin/env python3
"""Bounded, non-executing reward for Python output-prediction tasks.

The model must put its answer in exactly one literal
``<final_output>...</final_output>`` block.  Both the prediction and the
reference are parsed with :func:`ast.literal_eval` only after a small,
explicit AST has passed length, node-count, depth, node-type, container, and
finite-number checks.  Source programs and model text are never evaluated or
executed.

Equality deliberately follows Python assertion semantics.  In particular,
``True == 1`` and ``False == 0`` are correct, while lists and tuples remain
different container types.  Complex numbers are outside the benchmark's
deterministic literal subset.
"""

from __future__ import annotations

import ast
import math
from typing import Any


FINAL_OUTPUT_OPEN = "<final_output>"
FINAL_OUTPUT_CLOSE = "</final_output>"
MAX_LITERAL_CHARS = 4096
MAX_AST_NODES = 512
MAX_AST_DEPTH = 32
PYTHON_FEATURE_VERSION = (3, 11)

_SUPPORTED_DATA_SOURCES = frozenset(
    {
        "codereasoner",
        "cruxeval_output",
        "livecodebench_output",
    }
)
_SCALAR_TYPES = (str, bytes, int, float, bool, type(None))


class LiteralValidationError(ValueError):
    """Raised when text is not a bounded, supported Python literal."""


def extract_last_final_output(solution_str: Any) -> str | None:
    """Extract exactly one well-ordered final-output tag pair.

    The historical function name is retained for launcher compatibility, but
    multiple tag pairs are malformed rather than selecting the last one.  This
    is the same contract used by ``evaluation.code.literal_metrics``.
    """

    if not isinstance(solution_str, str):
        return None
    if solution_str.count(FINAL_OUTPUT_OPEN) != 1:
        return None
    if solution_str.count(FINAL_OUTPUT_CLOSE) != 1:
        return None
    start = solution_str.find(FINAL_OUTPUT_OPEN)
    content_start = start + len(FINAL_OUTPUT_OPEN)
    end = solution_str.find(FINAL_OUTPUT_CLOSE, content_start)
    if end < 0:
        return None
    answer = solution_str[content_start:end].strip()
    if not answer or len(answer) > MAX_LITERAL_CHARS:
        return None
    return answer


def _measure_ast(tree: ast.AST) -> tuple[int, int]:
    count = 0
    max_depth = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        max_depth = max(max_depth, depth)
        if count > MAX_AST_NODES:
            raise LiteralValidationError(
                f"literal AST has more than {MAX_AST_NODES} nodes"
            )
        if depth > MAX_AST_DEPTH:
            raise LiteralValidationError(
                f"literal AST is deeper than {MAX_AST_DEPTH} nodes"
            )
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return count, max_depth


def _validate_constant(value: Any) -> None:
    if not isinstance(value, _SCALAR_TYPES):
        raise LiteralValidationError(f"unsupported literal scalar: {type(value).__name__}")
    if value is Ellipsis:
        raise LiteralValidationError("ellipsis is not a supported output literal")
    if isinstance(value, float) and not math.isfinite(value):
        raise LiteralValidationError("non-finite floats are not supported")


def _validate_literal_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_literal_ast(node.body)
        return
    if isinstance(node, ast.Constant):
        _validate_constant(node.value)
        return
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for element in node.elts:
            _validate_literal_ast(element)
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise LiteralValidationError("dictionary unpacking is not supported")
            _validate_literal_ast(key)
            _validate_literal_ast(value)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        _validate_literal_ast(node.operand)
        return
    # ast.literal_eval supports this one call as the spelling of an empty set.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "set"
        and not node.args
        and not node.keywords
    ):
        return
    raise LiteralValidationError(f"unsupported literal syntax: {type(node).__name__}")


def _validate_literal_value(value: Any, depth: int = 1) -> None:
    if depth > MAX_AST_DEPTH:
        raise LiteralValidationError("literal value exceeds the container-depth limit")
    if isinstance(value, _SCALAR_TYPES):
        _validate_constant(value)
        return
    if isinstance(value, (tuple, list, set)):
        for element in value:
            _validate_literal_value(element, depth + 1)
        return
    if isinstance(value, dict):
        for key, element in value.items():
            _validate_literal_value(key, depth + 1)
            _validate_literal_value(element, depth + 1)
        return
    raise LiteralValidationError(f"unsupported parsed container: {type(value).__name__}")


def parse_literal_bounded(text: Any) -> Any:
    """Parse one supported Python literal without evaluating arbitrary code."""

    if not isinstance(text, str):
        raise LiteralValidationError("literal text must be a string")
    candidate = text.strip()
    if not candidate:
        raise LiteralValidationError("literal text is empty")
    if len(candidate) > MAX_LITERAL_CHARS:
        raise LiteralValidationError(
            f"literal text exceeds the {MAX_LITERAL_CHARS}-character limit"
        )
    try:
        tree = ast.parse(
            candidate,
            mode="eval",
            feature_version=PYTHON_FEATURE_VERSION,
        )
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise LiteralValidationError("literal has invalid Python syntax") from exc
    _measure_ast(tree)
    _validate_literal_ast(tree)
    try:
        value = ast.literal_eval(tree)
    except (ValueError, TypeError, MemoryError, RecursionError) as exc:
        raise LiteralValidationError("literal could not be parsed safely") from exc
    _validate_literal_value(value)
    return value


def literals_equal(left_text: Any, right_text: Any) -> bool:
    """Return Python ``==`` semantics for two validated literal strings."""

    left = parse_literal_bounded(left_text)
    right = parse_literal_bounded(right_text)
    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def compute_score(
    data_source: Any,
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **_: Any,
) -> float:
    """Return a strict binary reward for a final Python literal answer."""

    del extra_info
    source = str(data_source).strip().lower()
    if source not in _SUPPORTED_DATA_SOURCES:
        raise ValueError(
            "Code-output scorer received unexpected "
            f"data_source={data_source!r}; expected one of {sorted(_SUPPORTED_DATA_SOURCES)}"
        )

    # Invalid references are a data-integrity failure, not a model mistake.
    try:
        parse_literal_bounded(ground_truth)
    except LiteralValidationError as exc:
        raise ValueError(f"invalid stored ground-truth literal: {ground_truth!r}") from exc

    predicted_text = extract_last_final_output(solution_str)
    if predicted_text is None:
        return 0.0
    try:
        return float(literals_equal(predicted_text, ground_truth))
    except LiteralValidationError:
        return 0.0


__all__ = [
    "FINAL_OUTPUT_CLOSE",
    "FINAL_OUTPUT_OPEN",
    "LiteralValidationError",
    "MAX_AST_DEPTH",
    "MAX_AST_NODES",
    "MAX_LITERAL_CHARS",
    "compute_score",
    "extract_last_final_output",
    "literals_equal",
    "parse_literal_bounded",
]

