#!/usr/bin/env python3
"""Unit tests for the bounded CodeReasoner output reward."""

from __future__ import annotations

import ast
import builtins
import inspect
import re
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.code import reward
from evaluation.code import literal_metrics


def tagged(value: str) -> str:
    return f"<final_output>{value}</final_output>"


class CodeReasonerRewardTest(unittest.TestCase):
    def test_accepts_nested_supported_literals(self) -> None:
        literal = "{'items': [1, (2, None), {'x', 'y'}], 'blob': b'abc'}"
        self.assertEqual(reward.compute_score("codereasoner", tagged(literal), literal), 1.0)

    def test_uses_python_assertion_bool_int_equality(self) -> None:
        self.assertEqual(reward.compute_score("codereasoner", tagged("True"), "1"), 1.0)
        self.assertEqual(reward.compute_score("codereasoner", tagged("False"), "0"), 1.0)
        self.assertEqual(reward.compute_score("codereasoner", tagged("True"), "2"), 0.0)

    def test_preserves_container_equality_semantics(self) -> None:
        self.assertEqual(reward.compute_score("codereasoner", tagged("[1, 2]"), "(1, 2)"), 0.0)
        self.assertEqual(reward.compute_score("codereasoner", tagged("{True: 'x'}"), "{1: 'x'}"), 1.0)

    def test_rejects_multiple_complete_answers(self) -> None:
        response = f"first {tagged('1')} corrected {tagged('2')}"
        self.assertIsNone(reward.extract_last_final_output(response))
        self.assertEqual(reward.compute_score("codereasoner", response, "2"), 0.0)
        self.assertEqual(reward.compute_score("codereasoner", response, "1"), 0.0)

    def test_does_not_fall_back_from_later_malformed_answer(self) -> None:
        cases = (
            f"earlier {tagged('1')} later <final_output>[1, 2",
            f"earlier {tagged('1')} later <final_output>2",
            f"earlier {tagged('1')} later <final_output></final_output>",
        )
        for response in cases:
            self.assertIsNone(reward.extract_last_final_output(response))
            self.assertEqual(reward.compute_score("codereasoner", response, "1"), 0.0)

    def test_rejects_malformed_or_missing_tags(self) -> None:
        malformed = (
            "1",
            "<final_output>1",
            "1</final_output>",
            "<FINAL_OUTPUT>1</FINAL_OUTPUT>",
            "<final_output></final_output>",
            "<final_output>lambda: 1</final_output>",
            None,
        )
        for response in malformed:
            self.assertEqual(reward.compute_score("codereasoner", response, "1"), 0.0)

    def test_caps_answer_characters(self) -> None:
        maximum = repr("x" * (reward.MAX_LITERAL_CHARS - 2))
        self.assertEqual(len(maximum), reward.MAX_LITERAL_CHARS)
        self.assertEqual(reward.compute_score("codereasoner", tagged(maximum), maximum), 1.0)
        overlong = repr("x" * (reward.MAX_LITERAL_CHARS - 1))
        self.assertGreater(len(overlong), reward.MAX_LITERAL_CHARS)
        self.assertEqual(reward.compute_score("codereasoner", tagged(overlong), "'x'"), 0.0)

    def test_caps_ast_node_count(self) -> None:
        # A flat list has Expression + List + Load plus one Constant per item.
        too_many = "[" + ",".join("0" for _ in range(reward.MAX_AST_NODES)) + "]"
        with self.assertRaises(reward.LiteralValidationError):
            reward.parse_literal_bounded(too_many)

    def test_caps_ast_depth(self) -> None:
        too_deep = "0"
        for _ in range(reward.MAX_AST_DEPTH + 2):
            too_deep = f"[{too_deep}]"
        with self.assertRaises(reward.LiteralValidationError):
            reward.parse_literal_bounded(too_deep)

    def test_rejects_non_finite_numbers(self) -> None:
        for literal in ("1e309", "-1e309", "1e309j"):
            with self.assertRaises(reward.LiteralValidationError):
                reward.parse_literal_bounded(literal)
            self.assertEqual(reward.compute_score("codereasoner", tagged(literal), "0.0"), 0.0)

    def test_rejects_all_complex_literals(self) -> None:
        for literal in ("1j", "1+2j", "complex(1, 2)"):
            with self.assertRaises(reward.LiteralValidationError):
                reward.parse_literal_bounded(literal)
            self.assertEqual(reward.compute_score("codereasoner", tagged(literal), "0"), 0.0)

    def test_rejects_unsupported_syntax_before_literal_eval(self) -> None:
        unsupported = (
            "open('/tmp/x')",
            "[x for x in [1]]",
            "lambda: 1",
            "{'x': object()}",
            "...",
            "{**{'x': 1}}",
        )
        for literal in unsupported:
            with mock.patch.object(ast, "literal_eval", side_effect=AssertionError("must not be called")):
                with self.assertRaises(reward.LiteralValidationError):
                    reward.parse_literal_bounded(literal)

    def test_accepts_only_the_restricted_empty_set_call(self) -> None:
        self.assertEqual(reward.parse_literal_bounded("set()"), set())
        for literal in ("set([1])", "frozenset()", "dict()", "list()"):
            with self.assertRaises(reward.LiteralValidationError):
                reward.parse_literal_bounded(literal)

    def test_rejects_bad_reference_and_wrong_routing(self) -> None:
        with self.assertRaises(ValueError):
            reward.compute_score("math", tagged("1"), "1")
        with self.assertRaises(ValueError):
            reward.compute_score("codereasoner", tagged("1"), "not a literal")

    def test_all_eval_routes_use_the_same_reward(self) -> None:
        for source in ("codereasoner", "cruxeval_output", "livecodebench_output"):
            self.assertEqual(reward.compute_score(source, tagged("{'a': [1]}"), "{'a': [1]}"), 1.0)

    def test_matches_evaluator_contract(self) -> None:
        cases = (
            (tagged("1"), "1"),
            (tagged("True"), "1"),
            (tagged("{'a': [1, (2, None)]}"), "{'a': [1, (2, None)]}"),
            (tagged("set()"), "set()"),
            (tagged("b'abc'"), "b'abc'"),
            (tagged("-3.5"), "-3.5"),
            (f"{tagged('1')} {tagged('1')}", "1"),
            ("<final_output>1", "1"),
            (tagged("1+2j"), "0"),
            (tagged("1e309"), "0.0"),
            (tagged("[x for x in [1]]"), "[1]"),
        )
        for completion, ground_truth in cases:
            official, _strict, _parsed = literal_metrics.score_tagged_literal(
                completion,
                ground_truth,
            )
            training_score = reward.compute_score(
                "codereasoner",
                completion,
                ground_truth,
            )
            self.assertEqual(training_score, float(official), completion)

        for literal in ("1", "True", "{'a': [1]}", "set()", "1+2j", "1e309"):
            evaluator_parse = literal_metrics.parse_gold_literal(literal)
            try:
                reward.parse_literal_bounded(literal)
            except reward.LiteralValidationError:
                reward_valid = False
            else:
                reward_valid = True
            self.assertEqual(reward_valid, evaluator_parse.valid, literal)

    def test_module_contains_no_dynamic_eval_or_exec_calls(self) -> None:
        source = inspect.getsource(reward)
        forbidden = re.compile(r"\b(?:eval|exec|compile)\s*\(")
        self.assertIsNone(forbidden.search(source))
        tree = ast.parse(source)
        forbidden_names = {"eval", "exec", "compile"}
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(calls.isdisjoint(forbidden_names))
        # Keep the builtins references live so static checkers see that this
        # test intentionally distinguishes literal parsing from code execution.
        self.assertTrue(callable(builtins.eval) and callable(builtins.exec))


if __name__ == "__main__":
    unittest.main()

