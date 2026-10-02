from __future__ import annotations

import json
import unittest

from evaluation.code.literal_metrics import (
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_LITERAL_CHARS,
    cluster_macro_metrics,
    official_equal,
    parse_gold_literal,
    parse_tagged_literal,
    pass_at_k_estimate,
    pass_metrics,
    recursive_plain_value,
    score_tagged_literal,
    strict_type_equal,
)
from evaluation.code.eval_code_literal_passk import (
    _cluster_field,
    _cluster_id,
    _messages,
    _validated_data_source,
    _validate_row,
)


def _tag(payload: str) -> str:
    return f"reasoning\n<final_output>{payload}</final_output>\n"


class LiteralParserTests(unittest.TestCase):
    def test_accepts_supported_literals_and_reasoning_outside_tag(self) -> None:
        examples = (
            ("None", None),
            ("True", True),
            ("-17", -17),
            ("8.5", 8.5),
            (r"'x==y'", "x==y"),
            (r"b'abc'", b"abc"),
            ("[1, ('a', False)]", [1, ("a", False)]),
            ("{'a': [1, 2]}", {"a": [1, 2]}),
            ("{1, 2}", {1, 2}),
            ("set()", set()),
        )
        for literal, expected in examples:
            with self.subTest(literal=literal):
                parsed = parse_tagged_literal(_tag(literal))
                self.assertTrue(parsed.valid, parsed.reason)
                self.assertEqual(parsed.value, expected)

    def test_requires_exactly_one_tag_pair(self) -> None:
        for completion in (
            "42",
            "<final_output>42",
            "42</final_output>",
            _tag("1") + _tag("2"),
            "</final_output><final_output>1",
        ):
            with self.subTest(completion=completion):
                self.assertFalse(parse_tagged_literal(completion).valid)

    def test_rejects_executable_or_nonliteral_expressions(self) -> None:
        for payload in (
            "__import__('os').system('touch /tmp/never')",
            "open('/etc/passwd').read()",
            "1 + 1",
            "[x for x in range(3)]",
            "f'{1}'",
            "frozenset()",
            "set([1])",
            "{x for x in [1]}",
        ):
            with self.subTest(payload=payload):
                self.assertFalse(parse_tagged_literal(_tag(payload)).valid)

    def test_enforces_character_node_and_depth_budgets(self) -> None:
        oversized = repr("x" * MAX_LITERAL_CHARS)
        parsed = parse_tagged_literal(_tag(oversized))
        self.assertFalse(parsed.valid)
        self.assertIn("characters", parsed.reason or "")

        many_nodes = "[" + ",".join("0" for _ in range(MAX_AST_NODES)) + "]"
        parsed = parse_tagged_literal(_tag(many_nodes))
        self.assertFalse(parsed.valid)
        self.assertIn("nodes", parsed.reason or "")

        deep = "0"
        for _ in range(MAX_AST_DEPTH + 4):
            deep = f"[{deep}]"
        parsed = parse_tagged_literal(_tag(deep))
        self.assertFalse(parsed.valid)
        self.assertIn("depth", parsed.reason or "")

    def test_safe_assert_is_opt_in_and_lhs_must_ast_match(self) -> None:
        completion = _tag("assert f(3) == 'x==y'")
        self.assertFalse(parse_tagged_literal(completion).valid)

        parsed = parse_tagged_literal(
            completion,
            allow_assert=True,
            expected_assert_lhs="f(3)",
        )
        self.assertTrue(parsed.valid, parsed.reason)
        self.assertTrue(parsed.used_assert)
        self.assertEqual(parsed.value, "x==y")

        for payload, lhs in (
            ("assert f(4) == 7", "f(3)"),
            ("assert f(3) == 7 == 7", "f(3)"),
            ("assert f(3) == 7, 'message'", "f(3)"),
            ("assert f(3) == dangerous()", "f(3)"),
        ):
            with self.subTest(payload=payload):
                self.assertFalse(
                    parse_tagged_literal(
                        _tag(payload),
                        allow_assert=True,
                        expected_assert_lhs=lhs,
                    ).valid
                )

    def test_gold_is_bounded_literal_string_without_tags(self) -> None:
        self.assertEqual(parse_gold_literal("set()").value, set())
        self.assertFalse(parse_gold_literal(17).valid)
        self.assertFalse(parse_gold_literal("<final_output>1</final_output>").valid)

    def test_set_details_are_json_serializable(self) -> None:
        rendered = recursive_plain_value({1, "a"})
        self.assertEqual(rendered["python_type"], "set")
        json.dumps(rendered)


class EqualityTests(unittest.TestCase):
    def test_official_and_strict_type_semantics_are_both_reported(self) -> None:
        self.assertTrue(official_equal(1, True))
        self.assertFalse(strict_type_equal(1, True))
        self.assertTrue(official_equal([1], [True]))
        self.assertFalse(strict_type_equal([1], [True]))
        self.assertTrue(strict_type_equal({1, 2}, {2, 1}))

    def test_scoring_counts_malformed_as_false(self) -> None:
        official, strict, parsed = score_tagged_literal("answer: 3", "3")
        self.assertFalse(official)
        self.assertFalse(strict)
        self.assertFalse(parsed.valid)

        official, strict, parsed = score_tagged_literal(_tag("1"), "True")
        self.assertTrue(official)
        self.assertFalse(strict)
        self.assertTrue(parsed.valid)


class PassMetricTests(unittest.TestCase):
    def test_unbiased_pass_at_k(self) -> None:
        self.assertAlmostEqual(pass_at_k_estimate(4, 2, 1), 0.5)
        self.assertAlmostEqual(pass_at_k_estimate(4, 2, 2), 5 / 6)
        self.assertEqual(pass_at_k_estimate(4, 2, 4), 1.0)

    def test_pool_metrics_include_first_attempt_and_average(self) -> None:
        metrics = pass_metrics(
            [[False, True], [True, True], [False, False]],
            requested_ks=(1, 2),
        )
        self.assertAlmostEqual(metrics["pass_at_1"], 0.5)
        self.assertAlmostEqual(metrics["first_attempt_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["avg_at_2"], 0.5)

    def test_cluster_macro_equal_weights_question_ids(self) -> None:
        # Cluster A has two failed rows, cluster B has one perfect row.  Row
        # macro is 1/3 while question-id cluster macro is (0 + 1) / 2.
        metrics = cluster_macro_metrics(
            [[False, False], [False, False], [True, True]],
            ["A", "A", "B"],
            requested_ks=(1, 2),
        )
        self.assertEqual(metrics["cluster_count"], 2)
        self.assertAlmostEqual(metrics["cluster_macro_pass_at_1"], 0.5)
        self.assertAlmostEqual(metrics["cluster_macro_first_attempt_accuracy"], 0.5)


class CommonSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {
            "id": "lcb-0",
            "dataset": "LiveCodeBench-execution-v2-test",
            "data_source": "livecodebench_output",
            "prompt": [
                {"role": "system", "content": "Return a literal."},
                {"role": "user", "content": "Predict the output."},
            ],
            "ability": "code_output_reasoning",
            "reward_model": {"style": "rule", "ground_truth": "[1, 2]"},
            "extra_info": {"question_id": 2777, "sample_id": "sample_0"},
        }

    def test_validates_and_preserves_question_cluster(self) -> None:
        _validate_row(self.row, 0, "LiveCodeBench-execution-v2-test")
        self.assertEqual(_cluster_id(self.row), "question_id:2777")
        self.assertEqual(len(_messages(self.row)), 2)

    def test_rejects_schema_and_dataset_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            _validate_row(self.row, 0, "CRUXEval-output-test")
        broken = dict(self.row)
        broken.pop("reward_model")
        with self.assertRaises(ValueError):
            _validate_row(broken, 0, "LiveCodeBench-execution-v2-test")

    def test_data_source_must_be_singleton_and_match_cli(self) -> None:
        self.assertEqual(
            _validated_data_source([self.row], "livecodebench_output"),
            "livecodebench_output",
        )
        with self.assertRaises(ValueError):
            _validated_data_source([self.row], "cruxeval_output")
        mixed = dict(self.row)
        mixed["data_source"] = "cruxeval_output"
        with self.assertRaises(ValueError):
            _validated_data_source([self.row, mixed], "livecodebench_output")

    def test_non_lcb_rows_cluster_by_normalized_function_when_present(self) -> None:
        row = dict(self.row)
        row["data_source"] = "codereasoner"
        row["extra_info"] = {
            "question_id": None,
            "normalized_function_hash": "abc123",
        }
        self.assertEqual(_cluster_id(row), "normalized_function_hash:abc123")
        self.assertEqual(
            _cluster_field([row], "codereasoner"),
            "extra_info.normalized_function_hash",
        )


if __name__ == "__main__":
    unittest.main()

