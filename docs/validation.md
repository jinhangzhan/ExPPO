# Release validation

Integration checks completed on 2026-09-22 (CPU tests use Python 3.12):

| Check | Result |
| --- | --- |
| Training-side CPU tests | 182 passed, 1 optional-backend test skipped |
| Evaluation-side CPU tests | 91 passed, 1 optional-backend test skipped |
| Original ExPPO and ablations | 645 exact numerical comparisons with the recovered scalar implementation |
| DAPO | Real batch-container collection, both reward managers, raw-correctness gating, and simulated trainer-flow tests |
| CodeReasoner preparation | All three pinned downloads verified; all eight generated dataset files match the recovered builder byte-for-byte |
| Cross-version data stability | Full CodeReasoner rebuilds under Python 3.11 and 3.12 produce identical dataset files |
| Dependency resolution | Updated training and evaluation requirements resolved separately |
| Source audit | No detected hard-coded cluster paths, credentials, Python syntax errors, or shell syntax errors |
| Archive filtering | Code-only, with no manuscript, nested transfer archive, runtime data, model weights, or logs |
| Relocated checkout | Both test suites passed from the extracted code-only ZIP in a path containing spaces; documentation links and launcher dry-runs passed |

The two test groups overlap. The optional skipped test requires the installed
vLLM sampling backend. Trainer-flow tests use CPU worker stand-ins rather than
GPU generation. Dependency resolution is not a GPU runtime test.

The earlier extraction checks also validated the 17,398-prompt math training
download, eight reasoning-benchmark fingerprints, trainer wheel packaging, and
continuation statistics against the archived response banks. Those banks and
generated datasets are not distributed.

Full training and GPU evaluation were not rerun. The paper-aligned DAPO changes
need fresh GPU runs; historical optimizer-reset trajectories also differ from
new fully resumable runs. These distinctions are recorded in the
[reproduction notes](reproduction.md). The original uploaded source archive is
preserved locally but excluded from the review ZIP.
