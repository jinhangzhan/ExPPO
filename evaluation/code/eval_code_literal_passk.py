#!/usr/bin/env python3
"""Evaluate common-schema code-output reasoning without executing any code.

The input rows must have the schema produced by ``evaluation.code.prepare``.  The
model is sampled with vLLM, but scoring only performs bounded AST parsing of a
single ``<final_output>`` literal and compares it with the frozen literal in
``reward_model.ground_truth``.  Reference functions and model text are never
passed to ``eval`` or ``exec``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.code.literal_metrics import (
    DEFAULT_KS,
    FINAL_CLOSE,
    FINAL_OPEN,
    MAX_AST_DEPTH,
    MAX_AST_NODES,
    MAX_LITERAL_CHARS,
    cluster_macro_metrics,
    parse_gold_literal,
    pass_at_k_estimate,
    pass_metrics,
    percentile,
    prefixed_metrics,
    recursive_plain_value,
    score_tagged_literal,
)


SCHEMA_VERSION = "code-output-reasoning-v1"
REQUIRED_ROW_FIELDS = {
    "id",
    "dataset",
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged local checkpoint directory")
    parser.add_argument("--test_file", required=True, help="Common-schema JSONL/JSON/parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_source", required=True)
    parser.add_argument("--distribution", choices=("id", "ood"), required=True)
    parser.add_argument("--method", required=True, help="Method or ablation label; base is reserved for step 0")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--run_tag", default="code-literal")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt_batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="First-N canary only")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=128)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--allow_safe_assert",
        action="store_true",
        help="Allow one tagged assert only when extra_info.assert_lhs is predeclared and AST-matches",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing summary for this exact run tag",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.k > 1 and args.temperature <= 0:
        parser.error("--temperature must be positive when --k > 1")
    if args.prompt_batch_size < 1:
        parser.error("--prompt_batch_size must be positive")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.step < 0:
        parser.error("--step cannot be negative")
    if args.method == "base" and args.step != 0:
        parser.error("the base model must use --step 0")
    if args.method != "base" and args.step == 0:
        parser.error("trained methods must use a positive --step")
    return args


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        records = pq.read_table(path).to_pylist()
    elif suffix in {".json", ".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            first = ""
            while not first:
                first = handle.read(1)
                if not first:
                    return []
                if first.isspace():
                    first = ""
            handle.seek(0)
            records = json.load(handle) if first == "[" else [
                json.loads(line) for line in handle if line.strip()
            ]
    else:
        raise ValueError(f"unsupported eval-file suffix: {path.suffix}")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("evaluation input must be a list/stream of objects")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "code-literal"


def _validate_row(row: dict[str, Any], index: int, expected_dataset: str) -> None:
    missing = sorted(REQUIRED_ROW_FIELDS - row.keys())
    if missing:
        raise ValueError(f"row {index} lacks common-schema fields: {missing}")
    if str(row["dataset"]) != expected_dataset:
        raise ValueError(
            f"row {index} dataset={row['dataset']!r}, expected {expected_dataset!r}"
        )
    if row["ability"] != "code_output_reasoning":
        raise ValueError(f"row {index} has unexpected ability {row['ability']!r}")
    if not str(row["id"]):
        raise ValueError(f"row {index} has an empty id")
    if not str(row["data_source"]):
        raise ValueError(f"row {index} has an empty data_source")
    reward_model = row["reward_model"]
    if not isinstance(reward_model, dict) or reward_model.get("style") != "rule":
        raise ValueError(f"row {index} has invalid reward_model metadata")
    if not isinstance(reward_model.get("ground_truth"), str):
        raise ValueError(f"row {index} ground truth is not a canonical literal string")
    if not isinstance(row["extra_info"], dict):
        raise ValueError(f"row {index} extra_info must be an object")
    prompt = row["prompt"]
    if not isinstance(prompt, (list, tuple, str)) or not prompt:
        raise ValueError(f"row {index} prompt is empty or invalid")


def _messages(row: dict[str, Any]) -> list[dict[str, str]] | str:
    prompt = row["prompt"]
    if isinstance(prompt, str):
        stripped = prompt.strip()
        if stripped.startswith("["):
            try:
                prompt = json.loads(stripped)
            except json.JSONDecodeError:
                return prompt
        else:
            return prompt
    if isinstance(prompt, tuple):
        prompt = list(prompt)
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"record {row['id']!r} has an invalid prompt")
    messages = []
    for message in prompt:
        if not isinstance(message, dict):
            raise ValueError(f"record {row['id']!r} has a non-object prompt message")
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if not role or not content:
            raise ValueError(f"record {row['id']!r} has an empty prompt role/content")
        messages.append({"role": role, "content": content})
    return messages


def _render_prompt(tokenizer: Any, prompt: list[dict[str, str]] | str) -> str:
    if isinstance(prompt, str):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise RuntimeError("tokenizer chat template failed for code-reasoning messages") from exc


def _ground_truth(row: dict[str, Any]) -> str:
    return str(row["reward_model"]["ground_truth"])


def _assert_lhs(row: dict[str, Any]) -> str | None:
    value = row["extra_info"].get("assert_lhs")
    return str(value) if value is not None else None


def _question_id(row: dict[str, Any]) -> Any:
    return row["extra_info"].get("question_id")


def _validated_data_source(rows: list[dict[str, Any]], expected: str) -> str:
    observed = {str(row["data_source"]) for row in rows}
    if len(observed) != 1:
        raise ValueError(f"dataset mixes data sources: {sorted(observed)}")
    data_source = next(iter(observed))
    if data_source != expected:
        raise ValueError(f"data_source={data_source!r}, expected {expected!r}")
    return data_source


def _cluster_id(row: dict[str, Any]) -> str:
    question_id = _question_id(row)
    if row["data_source"] == "livecodebench_output" and question_id is not None:
        return f"question_id:{question_id}"
    function_hash = row["extra_info"].get("normalized_function_hash")
    if function_hash:
        return f"normalized_function_hash:{function_hash}"
    return f"row_id:{row['id']}"


def _cluster_field(rows: list[dict[str, Any]], data_source: str) -> str:
    if data_source == "livecodebench_output":
        return "extra_info.question_id"
    if rows and all(row["extra_info"].get("normalized_function_hash") for row in rows):
        return "extra_info.normalized_function_hash"
    return "id"


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    flat = {
        key: value
        for key, value in summary.items()
        if not isinstance(value, (dict, list, tuple))
    }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    temporary.replace(path)


def _prefixed_cluster(
    flags: list[list[bool]],
    cluster_ids: list[str],
    *,
    prefix: str,
    requested_ks: tuple[int, ...],
) -> dict[str, float | int]:
    metrics = cluster_macro_metrics(flags, cluster_ids, requested_ks=requested_ks)
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def main() -> None:
    args = parse_args()
    test_file = Path(args.test_file).resolve()
    model_path = Path(args.model).resolve()
    if not test_file.is_file():
        raise FileNotFoundError(test_file)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config is missing: {model_path / 'config.json'}")

    all_rows = _load_records(test_file)
    dataset_total_samples = len(all_rows)
    if not all_rows:
        raise ValueError(f"no evaluation rows loaded from {test_file}")
    for index, row in enumerate(all_rows):
        _validate_row(row, index, args.dataset)
    ids = [str(row["id"]) for row in all_rows]
    duplicated_ids = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicated_ids:
        raise ValueError(f"dataset has duplicate row IDs: {duplicated_ids[:10]}")
    observed_data_source = _validated_data_source(all_rows, args.data_source)

    rows = all_rows[: args.limit] if args.limit else all_rows
    gold_results = [parse_gold_literal(_ground_truth(row)) for row in rows]
    invalid_gold = [
        (row["id"], result.reason)
        for row, result in zip(rows, gold_results)
        if not result.valid
    ]
    if invalid_gold:
        raise ValueError(f"frozen dataset has invalid literals: {invalid_gold[:10]}")
    if args.allow_safe_assert:
        missing_lhs = [row["id"] for row in rows if not _assert_lhs(row)]
        if missing_lhs:
            raise ValueError(
                "--allow_safe_assert requires extra_info.assert_lhs on every row; "
                f"missing for {missing_lhs[:10]}"
            )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_tag(f"{args.run_tag}_k{args.k}_seed{args.seed}")
    details_path = output_dir / f"{stem}.details.jsonl"
    summary_path = output_dir / f"{stem}.summary.json"
    summary_csv_path = output_dir / f"{stem}.summary.csv"
    if not args.overwrite and any(
        path.exists() for path in (details_path, summary_path, summary_csv_path)
    ):
        raise FileExistsError(
            f"run outputs already exist for {stem}; use --overwrite only for an intentional rerun"
        )
    details_temporary = details_path.with_name(
        f".{details_path.name}.tmp-{os.getpid()}"
    )
    dataset_hash = _sha256(test_file)
    model_config_hash = _sha256(model_path / "config.json")

    # Import model dependencies only after all schema/gold validation succeeds.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = [_render_prompt(tokenizer, _messages(row)) for row in rows]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
    )

    ground_truth_types = Counter(type(result.value).__name__ for result in gold_results)
    cluster_ids = [_cluster_id(row) for row in rows]
    cluster_sizes = Counter(cluster_ids)
    question_ids = [_question_id(row) for row in rows]
    started = time.time()
    metadata: dict[str, Any] = {
        "type": "meta",
        "schema_version": SCHEMA_VERSION,
        "complete": False,
        "model": str(model_path),
        "model_config_sha256": model_config_hash,
        "test_file": str(test_file),
        "dataset_sha256": dataset_hash,
        "dataset": args.dataset,
        "data_source": observed_data_source,
        "distribution": args.distribution,
        "method": args.method,
        "step": args.step,
        "run_tag": args.run_tag,
        "limit": args.limit,
        "is_canary": bool(args.limit) or "canary" in args.run_tag.casefold(),
        "limited": len(rows) < dataset_total_samples,
        "dataset_total_samples": dataset_total_samples,
        "samples": len(rows),
        "k": args.k,
        "sample_pool_size": args.k,
        "seed": args.seed,
        "seed_schedule": "seed + prompt_batch_start",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt_batch_size": args.prompt_batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "trust_remote_code": args.trust_remote_code,
        "allow_safe_assert": args.allow_safe_assert,
        "strict_answer_contract": f"exactly one {FINAL_OPEN}Python literal{FINAL_CLOSE}",
        "parser_max_chars": MAX_LITERAL_CHARS,
        "parser_max_ast_nodes": MAX_AST_NODES,
        "parser_max_ast_depth": MAX_AST_DEPTH,
        "scoring_execution": False,
        "ground_truth_types": dict(sorted(ground_truth_types.items())),
        "cluster_field": _cluster_field(rows, observed_data_source),
        "cluster_count": len(cluster_sizes),
        "cluster_size_histogram": dict(
            sorted(Counter(cluster_sizes.values()).items())
        ),
        "question_id_present": sum(value is not None for value in question_ids),
    }

    official_flags: list[list[bool]] = []
    strict_flags: list[list[bool]] = []
    format_flags: list[list[bool]] = []
    response_token_lengths: list[int] = []
    response_char_lengths: list[int] = []
    format_failure_reasons: Counter[str] = Counter()
    assert_answers_used = 0
    generated_attempts = 0
    requested_ks = tuple(k for k in DEFAULT_KS if k <= args.k)

    print(
        f"[INFO] model={model_path} dataset={args.dataset} samples={len(rows)} "
        f"k={args.k} seed={args.seed} scoring=literal-only",
        flush=True,
    )
    with details_temporary.open("w", encoding="utf-8") as details:
        details.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for start in range(0, len(rows), args.prompt_batch_size):
            stop = min(len(rows), start + args.prompt_batch_size)
            params = SamplingParams(
                n=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                seed=args.seed + start,
            )
            generated = llm.generate(prompts[start:stop], params, use_tqdm=False)
            if len(generated) != stop - start:
                raise RuntimeError(
                    f"vLLM returned {len(generated)} requests for {stop - start} prompts"
                )
            for offset, request_output in enumerate(generated):
                index = start + offset
                if len(request_output.outputs) != args.k:
                    raise RuntimeError(
                        f"row {index} returned {len(request_output.outputs)} attempts; expected {args.k}"
                    )
                completions = [candidate.text for candidate in request_output.outputs]
                scored = [
                    score_tagged_literal(
                        completion,
                        _ground_truth(rows[index]),
                        allow_assert=args.allow_safe_assert,
                        expected_assert_lhs=_assert_lhs(rows[index]),
                    )
                    for completion in completions
                ]
                row_official = [official for official, _, _ in scored]
                row_strict = [strict for _, strict, _ in scored]
                parsed = [result for _, _, result in scored]
                row_formats = [result.valid for result in parsed]
                for result in parsed:
                    if result.used_assert:
                        assert_answers_used += 1
                    if not result.valid:
                        format_failure_reasons[result.reason or "unknown"] += 1

                token_lengths = [
                    len(getattr(candidate, "token_ids", ()) or ())
                    for candidate in request_output.outputs
                ]
                char_lengths = [len(completion) for completion in completions]
                finish_reasons = [
                    getattr(candidate, "finish_reason", None)
                    for candidate in request_output.outputs
                ]
                official_flags.append(row_official)
                strict_flags.append(row_strict)
                format_flags.append(row_formats)
                response_token_lengths.extend(token_lengths)
                response_char_lengths.extend(char_lengths)
                generated_attempts += len(completions)

                num_correct = sum(row_official)
                strict_num_correct = sum(row_strict)
                sample_pass = {
                    f"pass_at_{k}": pass_at_k_estimate(args.k, num_correct, k)
                    for k in requested_ks
                }
                strict_sample_pass = {
                    f"strict_pass_at_{k}": pass_at_k_estimate(
                        args.k,
                        strict_num_correct,
                        k,
                    )
                    for k in requested_ks
                }
                extra_info = rows[index]["extra_info"]
                record = {
                    "type": "sample",
                    "sample_index": index,
                    "id": rows[index]["id"],
                    "question_id": _question_id(rows[index]),
                    "cluster_id": cluster_ids[index],
                    "data_source": rows[index]["data_source"],
                    "source_sample_id": extra_info.get("sample_id"),
                    "function_hash": extra_info.get("normalized_function_hash"),
                    "problem_id": extra_info.get("problem_id"),
                    "contest_id": extra_info.get("contest_id"),
                    "contest_date": extra_info.get("contest_date"),
                    "difficulty": extra_info.get("difficulty"),
                    "numsteps": extra_info.get("numsteps"),
                    "ground_truth_literal": _ground_truth(rows[index]),
                    "ground_truth_type": type(gold_results[index].value).__name__,
                    "parsed_predictions": [
                        recursive_plain_value(result.value) if result.valid else None
                        for result in parsed
                    ],
                    "prediction_types": [
                        type(result.value).__name__ if result.valid else None
                        for result in parsed
                    ],
                    "parse_reasons": [result.reason for result in parsed],
                    "used_safe_assert": [result.used_assert for result in parsed],
                    "correct": row_official,
                    "strict_type_correct": row_strict,
                    "valid_format": row_formats,
                    "num_correct": num_correct,
                    "strict_num_correct": strict_num_correct,
                    "first_attempt_correct": bool(row_official[0]),
                    "strict_first_attempt_correct": bool(row_strict[0]),
                    **sample_pass,
                    **strict_sample_pass,
                    f"avg_at_{args.k}": num_correct / args.k,
                    f"strict_avg_at_{args.k}": strict_num_correct / args.k,
                    "response_token_lengths": token_lengths,
                    "response_char_lengths": char_lengths,
                    "finish_reasons": finish_reasons,
                    "completions": completions,
                }
                details.write(json.dumps(record, ensure_ascii=False) + "\n")
            details.flush()
            print(f"[INFO] generated and scored {stop}/{len(rows)} prompts", flush=True)

        expected_attempts = len(rows) * args.k
        if generated_attempts != expected_attempts:
            raise RuntimeError(
                f"generated {generated_attempts} attempts; expected {expected_attempts}"
            )
        metrics = pass_metrics(official_flags, requested_ks=requested_ks)
        strict_metrics = prefixed_metrics(
            strict_flags,
            prefix="strict_",
            requested_ks=requested_ks,
        )
        cluster_metrics = cluster_macro_metrics(
            official_flags,
            cluster_ids,
            requested_ks=requested_ks,
        )
        strict_cluster_metrics = _prefixed_cluster(
            strict_flags,
            cluster_ids,
            prefix="strict_",
            requested_ks=requested_ks,
        )
        all_format_rate = sum(map(sum, format_flags)) / expected_attempts
        first_format_rate = sum(flags[0] for flags in format_flags) / len(rows)
        summary = {
            **metadata,
            "type": "summary",
            "complete": True,
            **metrics,
            **strict_metrics,
            **cluster_metrics,
            **strict_cluster_metrics,
            "format_rate_at_1": first_format_rate,
            "format_rate_all_attempts": all_format_rate,
            "generated_attempts": generated_attempts,
            "scored_attempts": expected_attempts,
            "malformed_attempts": expected_attempts - sum(map(sum, format_flags)),
            "format_failure_reasons": dict(format_failure_reasons.most_common()),
            "safe_assert_answers_used": assert_answers_used,
            "response_tokens_mean": sum(response_token_lengths)
            / len(response_token_lengths),
            "response_tokens_p50": percentile(response_token_lengths, 0.50),
            "response_tokens_p95": percentile(response_token_lengths, 0.95),
            "response_tokens_max": max(response_token_lengths),
            "response_chars_mean": sum(response_char_lengths)
            / len(response_char_lengths),
            "response_chars_p50": percentile(response_char_lengths, 0.50),
            "response_chars_p95": percentile(response_char_lengths, 0.95),
            "response_chars_max": max(response_char_lengths),
            "elapsed_seconds": time.time() - started,
            "details_path": str(details_path),
        }
        details.write(json.dumps(summary, ensure_ascii=False) + "\n")

    details_temporary.replace(details_path)
    _atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    _write_summary_csv(summary_csv_path, summary)

    print(f"first-attempt accuracy: {metrics['first_attempt_accuracy']:.8f}")
    for k in requested_ks:
        print(f"pass@{k}: {metrics[f'pass_at_{k}']:.8f}")
    print(f"avg@{args.k}: {metrics['avg_at_k']:.8f}")
    print(f"strict pass@1: {strict_metrics['strict_pass_at_1']:.8f}")
    print(f"format rate: {all_format_rate:.8f}")
    print(f"Summary: {summary_path}")
    print(f"Detailed log: {details_path}")


if __name__ == "__main__":
    main()

