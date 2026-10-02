"""CPU checks for fixed splits, estimator semantics, and update command parity."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/continuation"))
import workflow


class ContinuationWorkflowTests(unittest.TestCase):
    def test_prepare_exact_partition_and_no_oracle_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            workflow.prepare(work)
            train = workflow.read_jsonl(work / "data/train_prompts.jsonl")
            heldout = workflow.read_jsonl(work / "data/heldout_prompts.jsonl")
            self.assertEqual([p["panel_index"] for p in train], list(range(32)))
            self.assertEqual([p["panel_index"] for p in heldout], list(range(32, 64)))
            for prompt in train + heldout:
                valid = {tuple(v) for v in prompt["canonical_valid_vectors"]}
                distractors = {tuple(v) for v in prompt["canonical_distractor_vectors"]}
                self.assertEqual(len(valid), len(distractors))
                self.assertFalse(valid & distractors)
            workflow.prepare(work)  # Identical preparation is safely idempotent.

    def test_direct_pair_diversity_is_without_replacement(self):
        rows = [{"reward": True, "mode_key": "a"}] * 2 + [{"reward": True, "mode_key": "b"}]
        rows += [{"reward": False, "mode_key": None}]
        metrics = workflow.prompt_metrics(rows)
        self.assertEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["conditional_d"], 2 / 3)
        self.assertEqual(metrics["observed_modes"], 2)
        self.assertAlmostEqual(metrics["effective_modes"], 1.88988157484231)

    def test_undefined_diversity_and_unsolved_yield(self):
        empty = workflow.prompt_metrics([{"reward": False, "mode_key": None}])
        self.assertIsNone(empty["conditional_d"])
        self.assertEqual(empty["effective_modes"], 0)
        singleton = workflow.prompt_metrics([{"reward": True, "mode_key": "a"}])
        self.assertIsNone(singleton["conditional_d"])
        self.assertEqual(singleton["effective_modes"], 1)

    def test_three_identical_starting_models_and_update_bank(self):
        args = argparse.Namespace(workdir=Path("outputs/check"), model=Path("checkpoints/exppo"),
                                  train_bank=None, heldout_bank=None, update_seed=20260903,
                                  seed=20260905, tp=1)
        commands = workflow.commands(args)
        updates = [c for c in commands if c[1].endswith("update_worker.py")]
        self.assertEqual(len(updates), 3)
        for command in updates:
            self.assertEqual(command[command.index("--model") + 1], "checkpoints/exppo")
            self.assertEqual(command[command.index("--train-bank") + 1], "outputs/check/banks/update/train.jsonl")
            self.assertEqual(command[command.index("--bank-k") + 1], "128")
            self.assertEqual(command[command.index("--learning-rate") + 1], "1e-6")
        self.assertEqual(len(commands), 13)  # Two update banks, eight evaluation banks, three updates.

    def test_signed_coefficient_changes_preserve_absolute_mass(self):
        row = {"prompt_id": "p", "reward": True, "u": 1., "A_pre": 1., "A_post": 1.2}
        other = {"prompt_id": "p", "reward": True, "u": -1., "A_pre": 1., "A_post": .8}
        result = workflow.coefficient_audit({"records": [row, other] * 8})
        self.assertLess(result["max_group_absolute_mass_error"], 1e-12)
        self.assertAlmostEqual(result["categories"][0]["sum_A_post_minus_A_pre"], 1.6)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy optional for bootstrap test")
    def test_bootstrap_is_repeatable_and_keeps_null_undefined(self):
        result = workflow.estimate([.1, .2, .3, None], 7, 100, "contrast")
        self.assertEqual(result, workflow.estimate([.1, .2, .3, None], 7, 100, "contrast"))
        self.assertEqual(result["n_prompts"], 3)
        self.assertAlmostEqual(result["mean"], .2)
        self.assertIsNone(workflow.estimate([None], 7, 100)["mean"])


if __name__ == "__main__":
    unittest.main()
