#!/usr/bin/env python3
"""Frozen final-answer verifier for the controlled coin-change task.

The task asks for one final ``counts`` vector.  Model responses often mention
templates or candidate vectors while reasoning, so using the first occurrence
does not identify the submitted semantic mode.  This verifier selects the last
literal JSON ``"counts": ...`` field in the response.  That field itself must
contain a JSON integer array; an invalid final field is a failure rather than a reason
to fall back to an earlier candidate.
"""

from __future__ import annotations

import json
import re
from typing import Any


VERIFIER_NAME = "coin_change_final_counts_v1"

_FIELD = re.compile(r'"counts"\s*:\s*')


def extract_final_counts(response: str) -> list[int] | None:
    """Return the last submitted literal ``counts`` list, if well formed."""

    fields = list(_FIELD.finditer(response))
    if not fields:
        return None
    tail = response[fields[-1].end() :].lstrip()
    try:
        values, _ = json.JSONDecoder().raw_decode(tail)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(values, list) or not all(type(value) is int for value in values):
        return None
    return values


def verify_response(response: str, prompt: dict[str, Any]) -> dict[str, Any]:
    """Verify the final answer and return its exact count-vector mode."""

    if prompt.get("task") != "coin_change":
        raise ValueError("the frozen final-answer verifier only supports coin_change")
    values = extract_final_counts(response)
    if values is None:
        return {"reward": False, "mode_key": None, "error": "missing_or_invalid_final_counts"}

    payload = prompt["payload"]
    denominations = payload["denominations"]
    if len(values) != len(denominations) or any(
        value < 0 or value > payload["max_count"] for value in values
    ):
        return {"reward": False, "mode_key": None, "error": "invalid_counts"}
    if sum(count * denomination for count, denomination in zip(values, denominations)) != payload["target"]:
        return {"reward": False, "mode_key": None, "error": "wrong_sum"}

    mode = ",".join(map(str, values))
    if mode not in set(prompt["valid_mode_keys"]):
        raise RuntimeError(f"verifier/enumerator mismatch: coin_change/{mode}")
    return {"reward": True, "mode_key": mode, "error": None}
