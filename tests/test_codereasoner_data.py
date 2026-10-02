"""CPU checks for the recovered data builder, launchers and five-task aggregation."""

import ast
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from evaluation.code import prepare
from evaluation.code.run import code_entry, OOD
from scripts.prepare_codereasoner import download_sources, source_specs
from scripts.summarize_codereasoner import aggregate
from evaluation.data.prepare import benchmark_specs

ROOT = Path(__file__).resolve().parents[1]


def test_source_manifest_matches_original_builder():
    for source in source_specs():
        expected = prepare.SOURCES[source["key"]]
        for field in ("repository", "revision", "remote_path", "local_filename", "sha256", "url"):
            assert source[field] == getattr(expected, field)


def test_function_normalization_ignores_name_but_preserves_body():
    first = prepare.normalized_function("def f(x):\n return f(x - 1) if x else 0\n")[1]
    renamed = prepare.normalized_function("def other(x):\n return other(x - 1) if x else 0\n")[1]
    changed = prepare.normalized_function("def f(x):\n return x + 1\n")[1]
    assert first == renamed
    assert first != changed


def test_python311_ast_dump_matches_original_modern_ast():
    tree = ast.parse("def f(x):\n return x\n")
    expected = prepare.canonical_ast_dump(copy.deepcopy(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node._fields = tuple(field for field in node._fields if field != "type_params")
            if hasattr(node, "type_params"):
                del node.type_params
    assert prepare.canonical_ast_dump(tree) == expected


def test_inverse_rows_become_forward_calls_without_execution():
    assert prepare.codereasoner_call("f", "(1, 2)", "backward_input_repurposed_to_forward_output") == "f(1, 2)"
    assert prepare.codereasoner_call("f", "(1, 2)", "forward_output") == "f((1, 2))"
    dangerous_source = "def f():\n raise RuntimeError('must never run')\n"
    assert len(prepare.normalized_function(dangerous_source)[1]) == 64
    assert len(prepare.normalize_call("f()", "frozen-hash")) == 64


def sample_rows():
    rows = []
    for group in range(4):
        for index in range(2):
            extra = prepare.blank_extra_info()
            extra.update(normalized_function_hash=f"group-{group}", code_input_hash=f"input-{group}-{index}")
            rows.append(prepare.common_row(example_id=f"{group}-{index}", dataset="unassigned",
                        data_source="codereasoner", code="def f(x): return x", call_text=f"f({index})",
                        ground_truth=str(index), extra_info=extra))
    return rows


def test_split_assignment_is_deterministic_and_whole_group():
    targets = [("train", 4, "train"), ("validation", 2, "val")]
    original = sample_rows()
    split, unused, diagnostic = prepare.allocate_splits(copy.deepcopy(original), seed=42, targets=targets)
    reversed_split, _, _ = prepare.allocate_splits(list(reversed(copy.deepcopy(original))), seed=42, targets=targets)
    assert split == reversed_split
    assert len(unused) == 2
    assert diagnostic["train"]["normalized_function_groups"] == 2
    assert prepare.overlap_counts(split["train"], split["validation"]) == {
        "normalized_function_hashes": 0, "code_input_hashes": 0}


def test_arrow_and_jsonl_roundtrip(tmp_path):
    rows = sample_rows()
    parquet = tmp_path / "train.parquet"
    jsonl = tmp_path / "test.jsonl"
    prepare.atomic_parquet(parquet, rows)
    prepare.atomic_jsonl(jsonl, rows)
    assert pq.read_table(parquet).to_pylist() == rows
    assert [json.loads(line) for line in jsonl.read_text().splitlines()] == rows


def test_corrupt_existing_source_is_not_overwritten(tmp_path):
    source = source_specs()[0]
    target = tmp_path / source["local_filename"]
    target.write_text("corrupted existing data")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        download_sources(tmp_path)
    assert target.read_text() == "corrupted existing data"


def command_output(*arguments):
    env = dict(os.environ, PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.check_output(arguments, cwd=ROOT, env=env, text=True)


@pytest.mark.parametrize("method,token_budget", [("grpo", "4608"), ("exppo", "4608"), ("grpo-h", "3072")])
def test_code_training_retains_original_budgets_and_paper_steps(method, token_budget):
    words = shlex.split(command_output("bash", "scripts/train_codereasoner.sh", "--method", method, "--dry-run"))
    overrides = dict(word.split("=", 1) for word in words if "=" in word)
    assert overrides["trainer.total_training_steps"] == "900"
    assert overrides["data.max_prompt_length"] == "1024"
    assert overrides["data.max_response_length"] == "512"
    assert overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == token_budget
    assert overrides["custom_reward_function.path"] == "evaluation/code/reward.py"
    assert "phi35-code-" in overrides["trainer.default_local_dir"]


def test_paper_evaluation_combines_code_with_four_ood():
    lines = command_output("bash", "scripts/evaluate_codereasoner.sh", "--method", "exppo",
                           "--checkpoint", "outputs/example checkpoint", "--out", "outputs/evaluation", "--dry-run").splitlines()
    assert len(lines) == 2
    code, ood = map(shlex.split, lines)
    assert code[code.index("--step") + 1] == "900"
    assert code[code.index("--k") + 1] == "16"
    assert code[code.index("--model") + 1] == "outputs/example checkpoint"
    assert all(name in ood for name in OOD)
    assert "cruxeval" not in lines[0].lower()


def test_external_code_tasks_are_opt_in():
    output = command_output("bash", "scripts/evaluate_codereasoner.sh", "--suite", "external",
                            "--out", "outputs/evaluation", "--dry-run")
    assert "CRUXEval-output-test" in output
    assert "LiveCodeBench-execution-v2-test" in output
    assert "evaluation.run" not in output


def test_native_code_summary_conversion_rejects_partial_evaluations():
    summary = {"complete": True, "dataset": "CodeReasoner-ID-test", "samples": 1800,
               "sample_pool_size": 16, "pass_at_k": .5, "avg_at_k": .25}
    assert code_entry(summary)["16"]["pass_at_k"] == 50
    with pytest.raises(ValueError, match="non-canary"):
        code_entry(dict(summary, is_canary=True))


def test_macro_sd_is_taken_after_per_run_averaging(tmp_path):
    specs = benchmark_specs()
    paths = []
    for repetition, scores in enumerate(((10, 20, 30, 40), (20, 10, 40, 30))):
        payload = {}
        for name, score in zip(OOD, scores):
            payload[name] = {"4": {"prompts": specs[name]["count"], "pool_size": 4,
                                   "pass_at_k": score, "avg_at_k": score / 2}}
        payload["codereasoner"] = {"16": {"prompts": 1800, "pool_size": 16,
                                          "pass_at_k": 40 + 2 * repetition, "avg_at_k": 20}}
        path = tmp_path / f"run{repetition}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    result = aggregate(paths)
    assert result["ood_average"]["pass_at_k"]["mean"] == 25
    assert result["ood_average"]["pass_at_k"]["sd"] == 0
    assert result["codereasoner"]["pass_at_k"]["sd"] == pytest.approx(2 ** .5)
    payload["codereasoner"]["16"]["pool_size"] = 8
    paths[-1].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="wrong response budget"):
        aggregate(paths)


@pytest.mark.parametrize("pool", [1, 16])
def test_native_evaluator_cpu_generation_stub(tmp_path, monkeypatch, pool):
    """Exercise the whole scoring/output path without loading model weights."""
    from evaluation.code import eval_code_literal_passk as evaluator
    rows = sample_rows()[:2]
    for row in rows:
        row["dataset"] = "CodeReasoner-ID-test"
    source = tmp_path / "input.jsonl"
    prepare.atomic_jsonl(source, rows)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def generate(self, prompts, parameters, **kwargs):
            sample = SimpleNamespace(text="<final_output>1</final_output>", token_ids=[1, 2], finish_reason="stop")
            return [SimpleNamespace(outputs=[sample] * parameters.n) for _ in prompts]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer())))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM,
                                                          SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["eval", "--model", str(model), "--test_file", str(source),
        "--output_dir", str(output), "--dataset", "CodeReasoner-ID-test", "--data_source", "codereasoner",
        "--distribution", "id", "--method", "base", "--step", "0", "--k", str(pool)])
    evaluator.main()
    summary = json.loads(next(output.glob("*.summary.json")).read_text())
    assert summary["complete"]
    assert summary["generated_attempts"] == 2 * pool
    assert summary["pass_at_k"] == summary["avg_at_k"] == .5
    assert summary["format_rate_all_attempts"] == 1
    assert len(next(output.glob("*.details.jsonl")).read_text().splitlines()) == 4
    with pytest.raises(FileExistsError):
        evaluator.main()
