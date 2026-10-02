# Environment

Use Linux, Python 3.11, and CUDA-capable NVIDIA GPUs. The full paper recipe
uses an eight-GPU node; the memory-cost experiment used eight H200s.
CPU checks do not require model weights or dataset downloads.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade 'pip>=24' 'setuptools>=77,<81' wheel
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-cuda.txt
python -m pip install --no-deps -e third_party/verl
python scripts/check_environment.py
```

Use a separate evaluation environment. Hydra 1.3.2 requires ANTLR 4.9, while
the benchmark judge's SymPy LaTeX parser requires ANTLR 4.11; installing both
into one environment creates an incompatible dependency set.

```bash
deactivate
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install --upgrade 'pip>=24' 'setuptools>=77,<81' wheel
python -m pip install -r requirements-eval.txt
python scripts/check_environment.py --evaluation
```

Data preparation, standard/high-budget evaluation, controlled evaluation, and
the common-checkpoint continuation run in the evaluation environment. This also
includes CodeReasoner preparation and its literal-answer evaluation. Training
and checkpoint merging run in the training environment. `PYTHON_BIN` can select
the corresponding interpreter for shell wrappers.

Install the NVIDIA driver and a CUDA toolkit compatible with the selected
PyTorch wheel before building FlashAttention. A site-provided CUDA environment
can be used instead of a virtual environment; no cluster modules or activation
paths are embedded in this project.

The reference stack follows the recorded training environment: PyTorch 2.9.0,
vLLM 0.12.0, Transformers 4.57.1, Ray 2.49.1, and FlashAttention 2.8.3.
Upstream [vLLM 0.12.0 metadata](https://pypi.org/pypi/vllm/0.12.0/json)
requires PyTorch 2.9.0. Pins use public package versions rather than
cluster-specific wheel suffixes. This is a portable reference specification,
not a byte-identical archive of the cluster environment.

CodeReasoner data preparation additionally pins DuckDB 1.2.0 and PyArrow 20.0.0.
Its recovered split has been rebuilt identically under Python 3.11 and 3.12;
the builder normalizes the AST representation used by the original Python 3.13
environment before hashing function groups.

Model and dataset repositories retain their own access conditions. Authenticate
with `hf auth login` when required; credentials are read from the user's
environment and are never included in the release. Experiment outputs and
downloaded data are ignored by Git.

Optional developer checks:

```bash
python -m pip install -r requirements-dev.txt
make test-training     # training environment
# Activate the evaluation environment, install requirements-dev.txt there, then:
make test-evaluation
```
