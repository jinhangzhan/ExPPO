#!/usr/bin/env python3
"""Resume-safe fixed-K coin-change sampling with a frozen final-answer verifier."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from evaluation.controlled.verifier import VERIFIER_NAME, verify_response


EXPECTED_VERIFIER_SHA = "8efedcc847a4a6ee3ead4b347cae67d78ddc0ae60607d38ac7d66e7062884cfa"


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"invalid JSON at {path}:{number}") from exc
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def render_prompt(tokenizer, messages: list[dict[str, str]], enable_thinking: bool | None) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def stable_id(*parts: object) -> str:
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()[:24]


def part_path(parts_dir: Path, panel_index: int, round_index: int) -> Path:
    return parts_dir / f"p{panel_index:03d}_r{round_index:03d}.jsonl"


def validate_part(
    path: Path,
    prompt: dict[str, Any],
    config_id: str,
    round_index: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    expected_indices = list(range(round_index * chunk_size, (round_index + 1) * chunk_size))
    checks = {
        "row_count": len(rows) == chunk_size,
        "prompt_id": all(row.get("prompt_id") == prompt["prompt_id"] for row in rows),
        "panel_index": all(row.get("panel_index") == prompt["panel_index"] for row in rows),
        "config_id": all(row.get("config_id") == config_id for row in rows),
        "sample_indices": [row.get("sample_index") for row in rows] == expected_indices,
        "verification_contract": all(row.get("verification_contract") == VERIFIER_NAME for row in rows),
    }
    if not all(checks.values()):
        raise IntegrityError(f"part validation failed for {path}: {checks}")
    for index, row in enumerate(rows):
        reproduced = verify_response(row.get("response", ""), prompt)
        for field in ("reward", "mode_key", "error"):
            if reproduced[field] != row.get(field):
                raise IntegrityError(f"verifier mismatch for {path}:{index}/{field}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--enable-thinking", choices=("true", "false", "default"), required=True)
    parser.add_argument("--expected-prompts-sha256", default="eeee29807fd7095ebd572a7c7806847965da72fbe1bd767d7a612a9918e0b78d")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.samples_per_prompt <= 0 or args.chunk_size <= 0:
        raise ValueError("sample and chunk sizes must be positive")
    if args.samples_per_prompt % args.chunk_size:
        raise ValueError("samples-per-prompt must be divisible by chunk-size")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.temperature != 1.0 or args.top_p != 1.0:
        raise ValueError("this sealed experiment requires temperature=1 and top-p=1")

    prompts_path = Path(args.prompts)
    if sha256_file(prompts_path) != args.expected_prompts_sha256:
        raise IntegrityError("prompt hash mismatch")
    verifier_path = Path(__file__).with_name("verifier.py")
    if sha256_file(verifier_path) != EXPECTED_VERIFIER_SHA:
        raise IntegrityError("frozen verifier hash mismatch")
    model_path = Path(args.model)
    config_path = model_path / "config.json"
    if not args.dry_run and not config_path.is_file():
        raise IntegrityError("expected a local Hugging Face model directory; download the initialization or merge the actor first")

    prompts = [json.loads(line) for line in prompts_path.open(encoding="utf-8") if line.strip()]
    if len(prompts) != 64 or len({row["prompt_id"] for row in prompts}) != 64:
        raise IntegrityError("expected exactly 64 unique prompts")
    prompts.sort(key=lambda row: int(row["panel_index"]))
    if [row["panel_index"] for row in prompts] != list(range(64)):
        raise IntegrityError("panel indices must be 0..63")
    selected = [row for row in prompts if int(row["panel_index"]) % args.shard_count == args.shard_index]
    enable_thinking = None if args.enable_thinking == "default" else args.enable_thinking == "true"

    static_config = {
        "backbone": args.backbone,
        "method": args.method,
        "model": str(model_path),
        "checkpoint_config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "prompts": str(prompts_path),
        "prompts_sha256": args.expected_prompts_sha256,
        "prompt_count_full": 64,
        "panel_indices": [row["panel_index"] for row in selected],
        "samples_per_prompt": args.samples_per_prompt,
        "chunk_size": args.chunk_size,
        "round_count": args.samples_per_prompt // args.chunk_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "prompt_batch_size": args.prompt_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "verification_contract": VERIFIER_NAME,
        "verifier_sha256": EXPECTED_VERIFIER_SHA,
    }
    config_id = hashlib.sha256(
        json.dumps(static_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    static_config["config_id"] = config_id
    if args.dry_run:
        print(json.dumps(static_config, indent=2, sort_keys=True))
        return

    output_dir = Path(args.output_dir)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / "run.lock").open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    tokenized: dict[str, list[int]] = {}
    for prompt in selected:
        rendered = render_prompt(tokenizer, prompt["messages"], enable_thinking)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) + args.max_new_tokens > args.max_model_len:
            raise IntegrityError(f"context limit exceeded for panel {prompt['panel_index']}")
        tokenized[prompt["prompt_id"]] = ids

    rounds = args.samples_per_prompt // args.chunk_size
    missing: dict[int, list[dict[str, Any]]] = {}
    for round_index in range(rounds):
        missing[round_index] = []
        for prompt in selected:
            path = part_path(parts_dir, prompt["panel_index"], round_index)
            if path.is_file():
                validate_part(path, prompt, config_id, round_index, args.chunk_size)
            else:
                missing[round_index].append(prompt)

    if any(missing.values()):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        llm = LLM(
            model=str(model_path),
            tokenizer=str(model_path),
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=False,
        )
        for round_index in range(rounds):
            pending = missing[round_index]
            for start in range(0, len(pending), args.prompt_batch_size):
                batch = pending[start : start + args.prompt_batch_size]
                requests = [TokensPrompt(prompt_token_ids=tokenized[row["prompt_id"]]) for row in batch]
                parameters = [
                    SamplingParams(
                        n=args.chunk_size,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=0,
                        max_tokens=args.max_new_tokens,
                        seed=args.seed + int(row["panel_index"]) * 1000 + round_index,
                        skip_special_tokens=True,
                    )
                    for row in batch
                ]
                generated = llm.generate(requests, parameters, use_tqdm=True)
                for prompt, request_output in zip(batch, generated):
                    if len(request_output.outputs) != args.chunk_size:
                        raise IntegrityError("vLLM returned an incomplete fixed-K chunk")
                    rows = []
                    for offset, completion in enumerate(request_output.outputs):
                        token_ids = list(map(int, completion.token_ids))
                        if not token_ids:
                            raise IntegrityError(f"empty completion for panel {prompt['panel_index']}")
                        sample_index = round_index * args.chunk_size + offset
                        verification = verify_response(completion.text, prompt)
                        rows.append(
                            {
                                "sample_id": stable_id(
                                    args.backbone,
                                    args.method,
                                    prompt["prompt_id"],
                                    sample_index,
                                    token_ids,
                                ),
                                "config_id": config_id,
                                "prompt_id": prompt["prompt_id"],
                                "panel_index": prompt["panel_index"],
                                "backbone": args.backbone,
                                "method": args.method,
                                "model": str(model_path),
                                "sample_index": sample_index,
                                "sampling_seed": args.seed + int(prompt["panel_index"]) * 1000 + round_index,
                                "response": completion.text,
                                "response_token_ids": token_ids,
                                "response_length": len(token_ids),
                                "finish_reason": completion.finish_reason,
                                "verification_contract": VERIFIER_NAME,
                                **verification,
                            }
                        )
                    path = part_path(parts_dir, prompt["panel_index"], round_index)
                    atomic_jsonl(path, rows)
                    validate_part(path, prompt, config_id, round_index, args.chunk_size)
                print(
                    json.dumps(
                        {
                            "round": round_index,
                            "completed_in_round": min(start + len(batch), len(pending)),
                            "pending_in_round": len(pending),
                            "shard": args.shard_index,
                        }
                    ),
                    flush=True,
                )

    manifest = []
    response_count = 0
    correct_count = 0
    for round_index in range(rounds):
        for prompt in selected:
            path = part_path(parts_dir, prompt["panel_index"], round_index)
            rows = validate_part(path, prompt, config_id, round_index, args.chunk_size)
            response_count += len(rows)
            correct_count += sum(bool(row["reward"]) for row in rows)
            manifest.append({"path": path.name, "sha256": sha256_file(path), "rows": len(rows)})
    manifest_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    marker = {
        "status": "complete",
        **static_config,
        "selected_prompt_count": len(selected),
        "response_count": response_count,
        "correct_count": correct_count,
        "part_count": len(manifest),
        "part_manifest": manifest,
        "part_manifest_sha256": manifest_sha,
    }
    atomic_json(output_dir / "complete.json", marker)
    print(json.dumps({key: marker[key] for key in ("status", "backbone", "method", "shard_index", "response_count", "correct_count", "part_manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
