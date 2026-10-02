# ExPPO's vendored verl runtime

This source snapshot is based on [verl](https://github.com/volcengine/verl),
version `0.8.0.dev`, with the ExPPO experiment modifications. Upstream copyright
headers, `LICENSE` and `Notice.txt` are retained. A source commit identifier was
not present in the supplied working copy.

Included modifications implement sequence-level ExPPO, GRPO-UR, F-GRPO,
REPO-R (including its checkpointed entropy controller), checkpoint serialization,
and the experiment's model/runtime compatibility changes. Runtime Python modules,
Hydra configurations and the FSDP/Hugging Face merger are included; upstream
examples, deployment scripts, caches and documentation are excluded.

The release exposes ExPPO component switches, defaulting to the unchanged
full-method arithmetic. These were verified against the recovered external-cluster
implementation. `no-gate` sets the gate to one on nonzero advantages;
`no-tanh` replaces bounded rarity by its z-score; `no-normalization` removes
the final group rescaling. Sign-specific variants restrict shaping before the
shared group normalization, so the other sign can still be rescaled. Within-group
reward and surprisal standardization use population standard deviations.
An enabled ExPPO invocation without old-policy log probabilities now fails
explicitly instead of silently reverting to GRPO.

The DAPO preset now implements the paper's asymmetric clipping `(0.2, 0.28)`,
dynamic sampling, token-mean loss and linear overlong-response penalty. Reward
configuration is passed through `reward_kwargs`, which both supported reward
manager paths consume. The dynamic collector follows the
[upstream DAPO recipe](https://github.com/volcengine/verl/blob/v0.6.1/recipe/dapo/dapo_ray_trainer.py):
keep non-uniform-correctness prompt groups, gather a full update under unchanged
weights, then balance responses and update the policy. It preserves response,
reward and diagnostic alignment, discards excess whole groups before the update,
and records generation counts separately from optimization steps. Epoch progress
is saved with checkpoints because variable candidate counts invalidate step-based
epoch inference. The rule-based, zero-reward-KL setting matches the paper.
Filtering and ExPPO's pass-rate gate use detached binary verifier outcomes;
advantages use rewards including the overlong penalty. Thus a correct response
at the length cap remains a verifier success even when its shaped reward is zero.

This paper-aligned implementation replaces inactive/missing features in the
archived DAPO launcher. CPU tests cover its collection and reward wiring; the
historical DAPO values have not been regenerated with it.

The release saves model, optimizer and random-generator state. Historical launchers
saved model and extra state without AdamW state; resumed historical trajectories
therefore need not match a continuously trained or fully restored run.

Install this checkout using the root project's environment instructions. The
package metadata's tensordict upper bound is updated to include the recorded
`0.11.0` runtime. No training outputs or model weights are distributed here.
