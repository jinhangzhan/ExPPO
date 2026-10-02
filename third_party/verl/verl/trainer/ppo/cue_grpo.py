"""Deterministic strategy-cue extraction and Cue-GRPO rarity weights.

This module is an adaptation of ``skeleton_extractor.py`` from the official
Cue-GRPO implementation at commit
19979295458ec5a85eb46f2e24a06f9b82a2812e:
https://github.com/CzZ12/When-Correct-Solutions-Repeat-Rarity-Aware-Credit-Redistribution-for-GRPO

The adapted portions are distributed under the following MIT license:

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

# Operation definitions are ordered exactly as in the official implementation.
# Higher-priority symbolic cues are matched before natural-language cues.
_OPERATION_PATTERNS = [
    (
        "modulo_op",
        2,
        [r"\\pmod", r"\\bmod", r"\\mod", r"\\equiv\s*\d*\s*\\pmod", r"\\mod\{", r"\bmod\b"],
        [],
    ),
    (
        "combinatorics",
        2,
        [r"\\binom", r"_\{?\d+\}?C_\{?\d+\}?", r"_C_", r"\\choose", r"combination", r"permutation"],
        [],
    ),
    ("discriminant", 2, [r"\\Delta", r"b\^2\s*-\s*4ac", r"discriminant", r"b\^2-4ac"], []),
    ("radicals", 2, [r"\\sqrt", r"\\sqrt\[]"], []),
    ("summation", 2, [r"\\sum", r"\\prod"], []),
    (
        "calculus",
        2,
        [r"\\int", r"\\frac\{d\}", r"\lim", r"\\infty", r"derivative", r"integral", r"\\partial"],
        [],
    ),
    ("log_exp", 2, [r"\\log", r"\\ln", r"\\exp", r"\\e\^", r"\\log_"], []),
    (
        "trigonometry",
        2,
        [r"\\sin", r"\\cos", r"\\tan", r"\\arcsin", r"\\arccos", r"\\arctan", r"\\csc", r"\\sec", r"\\cot"],
        [],
    ),
    (
        "inequality",
        2,
        [r"\\le\s", r"\\ge\s", r"\\leq", r"\\geq", r"\\neq", r"\\leqslant", r"\\geqslant", r"inequal"],
        [],
    ),
    ("set_theory", 2, [r"\\cup", r"\\cap", r"\\subset", r"\\in\b", r"\\emptyset", r"\\mathbb\{"], []),
    (
        "matrix_op",
        2,
        [
            r"\\begin\{matrix\}",
            r"\\begin\{pmatrix\}",
            r"\\begin\{bmatrix\}",
            r"\\begin\{vmatrix\}",
            r"\\det",
            r"determinant",
        ],
        [],
    ),
    ("gcd_lcm", 2, [r"\\gcd", r"\\lcm", r"gcd\(", r"lcm\(", r"greatest common"], []),
    (
        "define",
        1,
        [],
        [
            r"\b[Ll]et\s+[\w\s]+?\b(?:be\b|=|denote|represent|equal)",
            r"\b[Ss]et\s+[\w\s]+?\b(?:as|=|to(?:\s+be)?)",
            r"\b[Dd]efine\b",
            r"\b[Dd]enote\b",
            r"\b[Ss]uppose\b",
            r"\b[Aa]ssume\b",
            r"\b[Gg]iven\b",
            r"\b[Ww]rite\b.*\b(as|in the form)\b",
            r"\b[Cc]onsider\b",
            r"\b[Ll]abel\b",
        ],
    ),
    (
        "equation_setup",
        1,
        [],
        [
            r"(?i)(?:system\s+of\s+(?:linear\s+)?equations?)",
            r"(?i)(?:set\s+up|form\s+an?\s+equation|equations?\s*(?:as\s+follows|:)\s*\n)",
            r"(?i)(?:we\s+(?:have|get|obtain)\s+(?:the\s+)?(?:following\s+)?equations?)",
            r"(?i)(?:from\s+the\s+(?:given|problem|conditions?))",
        ],
    ),
    (
        "substitute",
        1,
        [],
        [
            r"\b[Ss]ubstitut",
            r"\b[Pp]lug\b",
            r"\b[Rr]eplac(?:e|ing)\b",
            r"\b[Bb]ack.substitut",
            r"\b[iI]nto\b",
            r"\b[Ii]nsert\b",
            r"(?i)(?:substitut(?:e|ing)\s+(?:this|it|back|into))",
        ],
    ),
    (
        "solve_linear",
        1,
        [],
        [
            r"\b[Ss]olv(?:e|ing)\b(?!.*quadrat)(?!.*cubic)",
            r"\b[Ff]ind\b.*\b(?:value|root|solution)\b",
            r"\b[Ii]solat(?:e|ing)\b",
            r"\b[Rr]earrang(?:e|ing)\b",
            r"\b[Dd]ivid(?:e|ing)\s+both\s+sides\b",
            r"\b[Mm]ultiply(?:ing)?\s+both\s+sides\b",
            r"\b[Aa]dd(?:ing)?\s+(?:to\s+)?both\s+sides\b",
            r"\b[Ss]ubtract(?:ing)?\s+(?:from\s+)?both\s+sides\b",
        ],
    ),
    (
        "solve_quadratic",
        1,
        [],
        [
            r"\b[Qq]uadrat(?:ic)?\s+(?:formula|equation)",
            r"\b[Dd]iscriminant\b",
            r"b\^?2\s*[--]\s*4\s*a\s*c",
            r"\b[Cc]omplet(?:e|ing)\s+(?:the\s+)?square\b",
            r"\b[Ff]actor(?:ing|isation)?\s*(?:the\s+)?(?:\w+\s*=\s*0)",
            r"\b[Vv]ieta\b",
            r"(?i)(?:roots?\s+of\s+the\s+quadratic)",
        ],
    ),
    (
        "solve_cubic",
        1,
        [],
        [
            r"\b[Cc]ubic\s+(?:formula|equation)",
            r"\b[Rr]ational\s+[Rr]oot\s+[Tt]heorem\b",
            r"\b[Ss]ynthetic\s+division\b",
        ],
    ),
    (
        "case_split",
        1,
        [],
        [
            r"\b[Cc]ase\s+\d+\b",
            r"\b[Cc]onsider\s+(?:two|three|both|each)\s+cases?\b",
            r"\b[Ii]f\s+.*\b(?:otherwise|else)\b",
            r"\b[Ww]hen\s+\w+\s*(?:is|are|=|>|<)\b",
            r"(?i)(?:there\s+are\s+(?:two|three|several)\s+(?:possibilit|case|scenario))",
        ],
    ),
    (
        "theorem_apply",
        1,
        [],
        [
            r"\b[Tt]heorem\b",
            r"\b[Pp]ythagorean\b",
            r"\b[Ff]ermat\b",
            r"\b[Ll]aw\s+of\s+(?:cosines?|sines?)\b",
            r"\b[Bb]inomial\s+[Tt]heorem\b",
            r"\b[Ee]uler\b",
            r"\b[Ww]ilson\b",
            r"\b[Ff]ormula\b",
            r"\b[Pp]ropert(?:y|ies)\b.*\b(?:of|in)\b",
            r"\b[Aa]rithmetic\s+(?:sequence|progression|series)\b",
            r"\b[Gg]eometric\s+(?:sequence|progression|series)\b",
        ],
    ),
    (
        "modulo_arith",
        1,
        [],
        [
            r"\b[Mm]od(?:ulo)?\b(?!.*op)",
            r"\b[Cc]ongruen(?:t|ce)\b",
            r"\b[Uu]nit(?:s)?\s+digit\b",
            r"\b[Rr]emainder\b",
            r"\b[Cc]hinese\s+[Rr]emainder\b",
            r"\b[Ff]ermat.s\s+[Ll]ittle\b",
            r"\b[Ee]uler.s\s+totient\b",
        ],
    ),
    (
        "induction",
        1,
        [],
        [
            r"\b[Ii]nduction\b",
            r"\b[Bb]ase\s+case\b",
            r"\b[Ii]nductive\s+(?:step|hypothesis)\b",
            r"\b[Pp]rove\s+(?:by|via|using)\s+induction\b",
        ],
    ),
    (
        "combin_count",
        1,
        [],
        [
            r"\b[Cc]ount(?:ing)?\b",
            r"\b[Pp]igeonhole\b",
            r"\b[Ss]tars?\s*(?:and|&)\s*bars?\b",
            r"\b[Nn]umber\s+of\s+ways?\b",
            r"\b[Hh]ow\s+many\b",
            r"\bchoose\b.*\bfrom\b",
            r"\bselect\b.*\bfrom\b",
        ],
    ),
    (
        "simplify",
        1,
        [],
        [
            r"\b[Ss]implif",
            r"\b[Rr]educ(?:e|ing)\b",
            r"\b[Cc]ancel\b",
            r"\b[Cc]ombin(?:e|ing)\s+(?:like\s+)?terms?\b",
            r"\b[Dd]istribut(?:e|ing)\b",
            r"\b[Ee]xpand\b",
        ],
    ),
    (
        "compute",
        1,
        [],
        [
            r"\b[Cc]omput(?:e|ing)\b",
            r"\b[Cc]alculat(?:e|ing)\b",
            r"\b[Ee]valuat(?:e|ing)\b",
            r"\b[Ww]ork(?:ing)?\s*out\b",
        ],
    ),
    (
        "verify",
        1,
        [],
        [
            r"\b[Cc]heck\b",
            r"\b[Vv]erif",
            r"\b[Tt]est\b",
            r"\b[Cc]onfirm\b",
            r"\b[Vv]alidat",
            r"\b[Dd]ouble.check\b",
            r"\b[Ee]nsure\b",
        ],
    ),
    (
        "conclude",
        1,
        [],
        [
            r"\b[Tt]herefore\b",
            r"\b[Tt]hus\b",
            r"\b[Hh]ence\b",
            r"\b[Ss]o\b(?!\s*(?:that|as|we|the|I|let|if|now|far|long))",
            r"\b[Tt]his\s+(?:gives|yields|means|implies|shows)\b",
            r"\b[Aa]s\s+a\s+result\b",
            r"\b[Cc]onsequently\b",
            r"\b[Ff]inally\b",
            r"\b[Ii]n\s+conclusion\b",
            r"\\boxed",
            r"\b[Aa]nswer\s*(?:is|=|:)\s*",
            r"\b[Rr]esult\s*(?:is|=|:)\s*",
            r"\b[Ww]e\s+(?:get|find|obtain|arrive\s+at)\b",
        ],
    ),
]


def _safe_compile(pattern: str) -> re.Pattern[str]:
    """Compile the upstream LaTeX-heavy patterns on supported Python versions."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        fixed = re.sub(r"\\([a-zA-Z])", r"\\\\\1", pattern)
        return re.compile(fixed, re.IGNORECASE)


_OPERATIONS = [
    (
        name,
        priority,
        [_safe_compile(pattern) for pattern in latex_patterns],
        [_safe_compile(pattern) for pattern in text_patterns],
    )
    for name, priority, latex_patterns, text_patterns in _OPERATION_PATTERNS
]
_OPERATIONS.sort(key=lambda item: -item[1])

_STEP_BOUNDARY = re.compile(
    r"\n\s*\n"
    r"|\n\s*(?:\d+[\.\)]\s)"
    r"|\n\s*(?:Step|First|Next|Then|Finally|Therefore|Thus|Hence|So\b)[\s,:,,]"
    r"|\n\s*(?:Now|Consequently|As a result|This gives|This yields|This means)[\s,:,,]"
    r"|\n\s*(?:We|Let|Consider|Suppose|Assume|Define|Given|Note|Observe|Recall)[\s,]"
    r"|\n\s*(?:Since|Because|If|When|For|From|By|Using|In|To|Now)[\s,]",
    re.IGNORECASE,
)
_NOISE_LABELS = {"other", "verify"}


def split_steps(text: str) -> list[str]:
    """Split one completion into the deterministic reasoning steps used by Cue-GRPO."""
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    steps = [part.strip() for part in _STEP_BOUNDARY.split(text) if len(part.strip()) >= 20]
    return steps if steps else [text]


def classify_step(step: str) -> str:
    """Return the first matching cue in stable priority/catalog order."""
    for name, _priority, latex_patterns, text_patterns in _OPERATIONS:
        if any(pattern.search(step) for pattern in latex_patterns):
            return name
        if any(pattern.search(step) for pattern in text_patterns):
            return name
    return "other"


def extract_skeleton(text: str) -> tuple[list[str], list[str]]:
    """Extract a de-noised, consecutive-deduplicated strategy-cue skeleton."""
    if not isinstance(text, str):
        raise TypeError(f"Cue-GRPO completions must be strings, got {type(text).__name__}")
    steps = split_steps(text)
    labels = [classify_step(step) for step in steps]
    labels = [label for label in labels if label not in _NOISE_LABELS]
    if not labels:
        labels = ["empty"]

    merged: list[str] = []
    for label in labels:
        if not merged or merged[-1] != label:
            merged.append(label)
    return steps, merged


def bag_of_cues_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Cosine similarity between two bag-of-cue count vectors."""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left_counts, right_counts = Counter(left), Counter(right)
    cues = set(left_counts) | set(right_counts)
    dot = sum(left_counts.get(cue, 0) * right_counts.get(cue, 0) for cue in cues)
    left_norm = math.sqrt(sum(value**2 for value in left_counts.values()))
    right_norm = math.sqrt(sum(value**2 for value in right_counts.values()))
    if left_norm < 1e-9 or right_norm < 1e-9:
        return 1.0
    return dot / (left_norm * right_norm)


def cluster_skeletons(skeletons: Sequence[Sequence[str]], similarity_threshold: float = 0.50) -> list[int]:
    """Partition skeletons by connected components of the cosine-similarity graph."""
    count = len(skeletons)
    if count <= 1:
        return [0] * count

    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left in range(count):
        for right in range(left + 1, count):
            if bag_of_cues_similarity(skeletons[left], skeletons[right]) >= similarity_threshold:
                adjacency[left].append(right)
                adjacency[right].append(left)

    cluster_ids = [-1] * count
    cluster_id = 0
    for root in range(count):
        if cluster_ids[root] != -1:
            continue
        cluster_ids[root] = cluster_id
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if cluster_ids[neighbor] == -1:
                    cluster_ids[neighbor] = cluster_id
                    stack.append(neighbor)
        cluster_id += 1
    return cluster_ids


def validate_cue_grpo_parameters(
    *,
    alpha: float,
    similarity_threshold: float,
    common_cue_threshold: float,
    target_floor: float,
    clip_min: float,
    clip_max: float,
    beta: float,
    max_workers: int,
) -> None:
    """Fail fast on invalid Cue-specific settings."""
    finite_values = {
        "cue_grpo_alpha": alpha,
        "cue_grpo_similarity_threshold": similarity_threshold,
        "cue_grpo_common_cue_threshold": common_cue_threshold,
        "cue_grpo_target_floor": target_floor,
        "cue_grpo_clip_min": clip_min,
        "cue_grpo_clip_max": clip_max,
        "cue_grpo_beta": beta,
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if alpha < 0.0:
        raise ValueError(f"cue_grpo_alpha must be >= 0, got {alpha}")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(f"cue_grpo_similarity_threshold must be in [0, 1], got {similarity_threshold}")
    if not 0.0 <= common_cue_threshold <= 1.0:
        raise ValueError(f"cue_grpo_common_cue_threshold must be in [0, 1], got {common_cue_threshold}")
    if target_floor <= 0.0:
        raise ValueError(f"cue_grpo_target_floor must be > 0, got {target_floor}")
    if clip_min <= 0.0 or clip_max < clip_min:
        raise ValueError(f"Cue-GRPO requires 0 < clip_min <= clip_max, got {clip_min}, {clip_max}")
    if beta < 0.0:
        raise ValueError(f"cue_grpo_beta must be >= 0, got {beta}")
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError(f"cue_grpo_max_workers must be an integer >= 1, got {max_workers!r}")


def compute_group_rarity_weights(
    skeletons: Sequence[Sequence[str]],
    *,
    alpha: float = 0.8,
    similarity_threshold: float = 0.50,
    common_cue_threshold: float = 0.75,
    target_floor: float = 1.05,
    clip_min: float = 0.3,
    clip_max: float = 3.0,
    beta: float = 0.0,
) -> tuple[list[float], list[list[str]], list[int]]:
    """Compute paper-default weights for one prompt's correct completions."""
    correct_count = len(skeletons)
    if correct_count == 0:
        return [], [], []
    if correct_count == 1:
        weight = np.ones(1, dtype=np.float32)
        weight[0] = 1.0 + beta
        minimum = weight.min()
        if minimum < target_floor:
            weight[:] = weight - minimum + target_floor
        return weight.tolist(), [list(skeletons[0])], [0]

    cue_counts: Counter[str] = Counter()
    for skeleton in skeletons:
        cue_counts.update(set(skeleton))
    cutoff = common_cue_threshold * correct_count
    suppressed = {cue for cue, count in cue_counts.items() if count > cutoff}
    cleaned = [[cue for cue in skeleton if cue not in suppressed] for skeleton in skeletons]

    cluster_ids = cluster_skeletons(cleaned, similarity_threshold=similarity_threshold)
    cluster_sizes = Counter(cluster_ids)
    rarity_scores = [cluster_sizes[cluster_id] ** (-alpha) for cluster_id in cluster_ids]
    normalizer = sum(rarity_scores) / correct_count + 1e-8
    rarity_scores = [float(np.clip(score / normalizer, clip_min, clip_max)) for score in rarity_scores]

    median_length = float(np.median([len(skeleton) for skeleton in cleaned]))
    for position, (skeleton, cluster_id) in enumerate(zip(cleaned, cluster_ids, strict=True)):
        if cluster_sizes[cluster_id] != 1:
            continue
        skeleton_length = len(skeleton)
        too_short = skeleton_length < max(2.0, 0.3 * median_length)
        too_long = skeleton_length > max(6.0, 3.0 * median_length)
        if too_short or too_long:
            rarity_scores[position] = 1.0

    # Match the official implementation's float32 weight buffer and in-buffer
    # additive floor. The trainer ultimately consumes float32 advantages.
    weights = np.asarray([score * (1.0 + beta) for score in rarity_scores], dtype=np.float32)
    minimum = weights.min()
    if minimum < target_floor:
        weights[:] = weights - minimum + target_floor
    return weights.tolist(), cleaned, cluster_ids


def compute_cue_grpo_weights(
    completions: Sequence[str],
    correct: Sequence[bool],
    group_ids: Sequence[Any],
    *,
    alpha: float = 0.8,
    similarity_threshold: float = 0.50,
    common_cue_threshold: float = 0.75,
    target_floor: float = 1.05,
    clip_min: float = 0.3,
    clip_max: float = 3.0,
    beta: float = 0.0,
    max_workers: int = 8,
) -> tuple[list[float], list[list[str]], list[int], dict[str, float]]:
    """Compute Cue multipliers without ever clustering across prompt groups.

    Only correct completions in mixed pass/fail groups need extraction: uniform
    reward groups already have exactly zero ordinary GRPO advantage.
    """
    validate_cue_grpo_parameters(
        alpha=alpha,
        similarity_threshold=similarity_threshold,
        common_cue_threshold=common_cue_threshold,
        target_floor=target_floor,
        clip_min=clip_min,
        clip_max=clip_max,
        beta=beta,
        max_workers=max_workers,
    )
    batch_size = len(completions)
    if len(correct) != batch_size or len(group_ids) != batch_size:
        raise ValueError(
            "Cue-GRPO expects aligned completions, correctness flags, and group ids; "
            f"got lengths {batch_size}, {len(correct)}, and {len(group_ids)}"
        )

    grouped_positions: dict[Any, list[int]] = defaultdict(list)
    for position, group_id in enumerate(group_ids):
        grouped_positions[group_id].append(position)

    # The official trainer treats alpha=beta=0 as the vanilla-GRPO control,
    # bypassing both clustering and the positive floor.
    if alpha == 0.0 and beta == 0.0:
        return (
            [1.0] * batch_size,
            [[] for _ in range(batch_size)],
            [-1] * batch_size,
            {
                "group_count": float(len(grouped_positions)),
                "active_group_count": 0.0,
                "correct_count": 0.0,
                "cluster_count": 0.0,
                "weight_mean": 1.0,
                "weight_min": 1.0,
                "weight_max": 1.0,
            },
        )

    active_groups: list[list[int]] = []
    extraction_positions: list[int] = []
    for positions in grouped_positions.values():
        correct_count = sum(bool(correct[position]) for position in positions)
        if 0 < correct_count < len(positions):
            active_groups.append(positions)
            extraction_positions.extend(position for position in positions if bool(correct[position]))

    extracted: dict[int, list[str]] = {}
    if extraction_positions:
        texts = [completions[position] for position in extraction_positions]
        if len(texts) == 1 or max_workers == 1:
            results = [extract_skeleton(text) for text in texts]
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(texts))) as executor:
                results = list(executor.map(extract_skeleton, texts))
        extracted = {
            position: skeleton for position, (_steps, skeleton) in zip(extraction_positions, results, strict=True)
        }

    weights = [1.0] * batch_size
    all_skeletons: list[list[str]] = [[] for _ in range(batch_size)]
    all_cluster_ids = [-1] * batch_size
    cluster_count = 0
    for positions in active_groups:
        correct_positions = [position for position in positions if bool(correct[position])]
        group_skeletons = [extracted[position] for position in correct_positions]
        group_weights, cleaned, cluster_ids = compute_group_rarity_weights(
            group_skeletons,
            alpha=alpha,
            similarity_threshold=similarity_threshold,
            common_cue_threshold=common_cue_threshold,
            target_floor=target_floor,
            clip_min=clip_min,
            clip_max=clip_max,
            beta=beta,
        )
        cluster_count += len(set(cluster_ids))
        for position, weight, skeleton, cluster_id in zip(
            correct_positions, group_weights, cleaned, cluster_ids, strict=True
        ):
            weights[position] = weight
            all_skeletons[position] = skeleton
            all_cluster_ids[position] = cluster_id

    active_weights = [weights[position] for position in extraction_positions]
    diagnostics = {
        "group_count": float(len(grouped_positions)),
        "active_group_count": float(len(active_groups)),
        "correct_count": float(len(extraction_positions)),
        "cluster_count": float(cluster_count),
        "weight_mean": float(np.mean(active_weights)) if active_weights else 1.0,
        "weight_min": float(min(active_weights)) if active_weights else 1.0,
        "weight_max": float(max(active_weights)) if active_weights else 1.0,
    }
    return weights, all_skeletons, all_cluster_ids, diagnostics
