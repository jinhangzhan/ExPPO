"""DAPO prompt-group filtering, independent of the GPU training workers.

Matches the collection rule in verl's Apache-2.0 DAPO recipe (v0.6.1):
https://github.com/volcengine/verl/blob/v0.6.1/recipe/dapo/dapo_ray_trainer.py
"""

from collections import defaultdict

import numpy as np


class DynamicGroupCollector:
    """Accumulate whole, non-uniform groups under a single unchanged policy.

    Input rewards and diagnostics must already be attached to the DataProto.
    Excess accepted groups are discarded at the update boundary, never carried
    into the next policy version. Input row order need not be group-contiguous.
    """

    def __init__(self, target_prompts, responses_per_prompt, metric="acc", max_num_gen_batches=0):
        if target_prompts < 1 or responses_per_prompt < 2:
            raise ValueError("Dynamic sampling requires positive prompt count and at least two responses per prompt")
        if not metric:
            raise ValueError("Dynamic sampling requires a filtering metric")
        self.target_prompts = target_prompts
        self.responses_per_prompt = responses_per_prompt
        self.metric = metric
        self.max_num_gen_batches = max_num_gen_batches
        self.reset()

    def reset(self):
        self.parts = []
        self.seen_uids = set()
        self.generated_batches = 0
        self.generated_prompts = 0
        self.accepted_prompts = 0
        self.reward_extra_keys = None

    def add(self, batch, reward_extra_keys):
        """Return a full update batch, or None when more generation is needed."""
        keys = tuple(sorted(reward_extra_keys))
        if self.reward_extra_keys is not None and self.reward_extra_keys != keys:
            raise ValueError("Reward diagnostic columns changed while collecting one DAPO update")
        self.reward_extra_keys = keys
        if self.metric in ("seq_reward", "seq_final_reward"):
            field = "token_level_scores" if self.metric == "seq_reward" else "token_level_rewards"
            values = batch.batch[field].detach().cpu().sum(dim=-1).numpy()
        else:
            if self.metric not in batch.non_tensor_batch:
                raise ValueError(f"Dynamic-sampling metric {self.metric!r} is absent from reward diagnostics")
            values = np.asarray(batch.non_tensor_batch[self.metric], dtype=float)
        if values.shape != (len(batch),) or not np.isfinite(values).all():
            raise ValueError("Dynamic-sampling metrics must contain one finite scalar per response")

        groups = defaultdict(list)
        for row, uid in enumerate(batch.non_tensor_batch["uid"]):
            groups[uid].append(row)
        if self.seen_uids.intersection(groups):
            raise ValueError("Prompt UIDs must be unique across candidate generation batches")
        self.seen_uids.update(groups)
        accepted = []
        for rows in groups.values():
            if len(rows) != self.responses_per_prompt:
                raise ValueError("Dynamic sampling received an incomplete prompt group")
            if np.ptp(values[rows]) > 0:
                accepted.append(rows)

        self.generated_batches += 1
        self.generated_prompts += len(groups)
        # Keep complete groups, even if the candidate batch was reordered.
        needed = self.target_prompts - self.accepted_prompts
        retained = accepted[:needed]
        if retained:
            part = batch[[row for rows in retained for row in rows]]
            # Per-generation timing/row counts are recomputed for the final batch.
            part.meta_info = {key: value for key, value in batch.meta_info.items()
                              if key not in {"timing", "global_token_num", "images_seqlens", "metrics"}}
            self.parts.append(part)
            self.accepted_prompts += len(retained)
        if self.accepted_prompts == self.target_prompts:
            return type(batch).concat(self.parts)
        if 0 < self.max_num_gen_batches <= self.generated_batches:
            raise RuntimeError(
                f"DAPO collected {self.accepted_prompts}/{self.target_prompts} informative prompt groups "
                f"after {self.generated_batches} generation batches. Increase "
                "algorithm.filter_groups.max_num_gen_batches or use 0 for no per-update limit."
            )
        return None

    def metrics(self):
        return {
            "dapo/generation_batches": self.generated_batches,
            "dapo/generated_prompts": self.generated_prompts,
            "dapo/accepted_prompts": self.accepted_prompts,
            "dapo/retained_fraction": self.accepted_prompts / max(self.generated_prompts, 1),
        }
