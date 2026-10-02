"""Run in requirements-eval.txt to exercise the actual paper answer checkers."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("sympy"), "Install requirements-eval.txt for mathematical verifier tests")
class AnswerVerifierTests(unittest.TestCase):
    def test_boxed_exact_answer(self):
        from evaluation.eval.openmathinst_utils import process_results
        self.assertTrue(process_results(r"Reasoning. \boxed{42}", "42", response_extract_from_boxed=True))
        self.assertFalse(process_results(r"Reasoning. \boxed{41}", "42", response_extract_from_boxed=True))

    def test_multiple_choice_answer(self):
        from evaluation.eval.openmathinst_utils import process_results
        self.assertTrue(process_results(r"The correct choice is \boxed{B}", "B", response_extract_from_boxed=True))
        self.assertFalse(process_results(r"The correct choice is \boxed{A}", "B", response_extract_from_boxed=True))

    def test_olympiad_checker(self):
        from evaluation.eval.objudge import OBJudge
        self.assertTrue(OBJudge().judge("2", r"\boxed{2}", 1e-8))


if __name__ == "__main__":
    unittest.main()
