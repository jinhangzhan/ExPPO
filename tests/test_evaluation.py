import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.controlled.bank import audit
from evaluation.controlled.metrics import paired_diversity, prompt_metrics
from evaluation.controlled.verifier import extract_final_counts, verify_response
from evaluation.data.prepare import benchmark_specs, canonical_hash, transform, validate
from evaluation.metrics.coverage import pass_at_k, read_groups, summarize

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_unbiased_coverage(self):
        self.assertAlmostEqual(pass_at_k(10, 2, 4), 1 - math.comb(8, 4) / math.comb(10, 4))
        self.assertEqual(pass_at_k(4, 1, 4), 1)
        self.assertEqual(pass_at_k(4, 0, 4), 0)
        with self.assertRaises(ValueError):
            pass_at_k(4, 1, 8)

    def test_response_accuracy_and_pool_estimator(self):
        result = summarize([[True, False, False, False], [False] * 4], [1, 4])
        self.assertEqual(result["1"]["pass_at_k"], 12.5)
        self.assertEqual(result["4"]["pass_at_k"], 50)
        self.assertEqual(result["4"]["avg_at_k"], 12.5)

    def test_original_bank_and_final_parser(self):
        bank = audit()
        self.assertEqual(len(bank), 64)
        self.assertEqual((min(r["oracle_mode_count"] for r in bank), max(r["oracle_mode_count"] for r in bank)), (4, 46))
        row = bank[0]
        counts = [int(v) for v in row["valid_mode_keys"][0].split(",")]
        self.assertTrue(verify_response(json.dumps({"counts": counts}), row)["reward"])
        self.assertFalse(verify_response(json.dumps({"counts": counts}) + '\n{"counts": [false]}', row)["reward"])
        self.assertEqual(extract_final_counts('{"counts": [1,2,3,4]} then {"counts": [0,0,0,0]}'), [0, 0, 0, 0])

    def test_yields_include_unsolved_prompts(self):
        result = prompt_metrics(["a", "a", "b", None])
        self.assertEqual(result["observed_yield"], 2)
        self.assertAlmostEqual(result["diversity"], 2 / 3)
        self.assertEqual(prompt_metrics([None] * 4)["effective_yield"], 0)
        self.assertIsNone(prompt_metrics(["a", None])["diversity"])

    def test_paired_support_is_row_specific(self):
        a = {"a": prompt_metrics(["x", "x"]), "b": prompt_metrics([None, None])}
        b = {"a": prompt_metrics(["x", "y"]), "b": prompt_metrics(["x", "y"])}
        result = paired_diversity(a, b, draws=20)
        self.assertEqual(result["shared_prompts"], 1)
        self.assertEqual(result["difference"], 1)
        self.assertIsNone(result["ci95"])

    def test_dataset_fingerprint_is_prompt_sensitive(self):
        row = {"prompt": [{"role": "user", "content": "question"}], "reward_model": {"ground_truth": "42"}}
        changed = {**row, "prompt": [{"role": "user", "content": "another question"}]}
        spec = {"count": 1, "sha256": canonical_hash([row])}
        validate([row], spec)
        with self.assertRaises(ValueError):
            validate([changed], spec)

    def test_truthfulqa_four_choice_filter(self):
        rows = transform("truthful_qa_mc", [{"question": "q", "mc1_targets": {"a": 1, "b": 0, "c": 0}},
                                              {"question": "q", "mc1_targets": {"a": 1, "b": 0, "c": 0, "d": 0, "e": 0}}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows, transform("truthful_qa_mc", [{"question": "q", "mc1_targets": {"a": 1, "b": 0, "c": 0}},
                                                         {"question": "q", "mc1_targets": {"a": 1, "b": 0, "c": 0, "d": 0, "e": 0}}]))

    def test_standard_and_high_k_contract(self):
        spec = benchmark_specs()
        self.assertEqual([spec[name]["k"] for name in ["aime-24", "aime-25", "minerva", "olympiadBench"]], [16, 16, 8, 4])
        self.assertEqual(spec["aime-24"]["pool_size"], 512)
        for command in [["-m", "evaluation.run", "--model", "qwen3", "--out", "outputs/test", "--dry-run"],
                        ["-m", "evaluation.run", "--model", "phi35", "--high-k", "--out", "outputs/test", "--dry-run"],
                        ["-m", "evaluation.controlled.run", "--model", "phi35", "--method", "ExPPO", "--out", "outputs/test", "--dry-run"]]:
            result = subprocess.run([sys.executable] + command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max-new-tokens 16384", result.stdout)
        self.assertIn("--prefix-tokens 2048", result.stdout)

    def test_jsonl_unicode_line_separator(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "example.jsonl"
            path.write_text(json.dumps({"prompt": "a\u2028b", "attempt_correct": [True, False]}, ensure_ascii=False) + "\n")
            self.assertEqual(read_groups(path), [[True, False]])

    def test_custom_single_sample_evaluation(self):
        from evaluation.run import main
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "evaluation"

            def generate(command, check):
                directory = Path(command[command.index("--out") + 1])
                directory.mkdir(parents=True)
                (directory / "sample.log.jsonl").write_text(
                    '{"type":"meta"}\n{"correct":true}\n{"correct":false}\n'
                )

            args = ["evaluation.run", "--model", "phi35", "--checkpoint", "example-model",
                    "--test-file", "example.jsonl", "--k", "1", "--out", str(out)]
            with mock.patch.object(sys, "argv", args), mock.patch("evaluation.run.subprocess.run", side_effect=generate):
                main()
            metrics = json.loads((out / "metrics.json").read_text())["code_reasoner"]["1"]
            self.assertEqual(metrics["pass_at_k"], 50.0)
            self.assertEqual(metrics["pool_size"], 1)


if __name__ == "__main__":
    unittest.main()
