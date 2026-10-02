# Attribution and release scope

- `third_party/verl/` contains the modified training framework used by this
  project. Its upstream Apache-2.0 license and source-file notices are retained.
  ExPPO and baseline changes are in `verl/trainer/ppo/core_algos.py` and the
  trainer/actor integration. Release-specific changes are documented alongside
  the training configuration.
- Evaluation helpers retain their original NVIDIA, Microsoft, OpenAI,
  OlympiadBench, and other source notices where present. Existing third-party
  licenses continue to apply; a wrapper does not replace those licenses.
- The CodeReasoner builder and literal verifier come from the recovered
  experiment source; data/model providers retain their respective terms.
  The frozen original advantage function is included as a test reference with
  its upstream notice, not as a second runtime implementation.
- Manuscript files and rendered figures are excluded from the code-only ZIP.
- Model weights and external benchmark data are downloaded from their original
  repositories under the applicable terms. No access credentials are included.

The authors should select a license for original ExPPO contributions before
public distribution. This extraction preserves existing licenses and does not
assign new ownership or relicense third-party material.
