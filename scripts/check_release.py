#!/usr/bin/env python3
"""Audit release contents, Python syntax, and local-path/credential leaks."""

import argparse
import ast
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = {".git", ".venv", "venv", "dist", "build", "data", "models",
                 "outputs", "results", "checkpoints", "logs", "wandb", "ray_results"}
CACHE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache", "build"}
BUILD_SUFFIXES = {".aux", ".blg", ".log", ".fls", ".fdb_latexmk", ".out", ".ptc", ".toc"}
PRIVATE_NAMES = {".env", ".deepspeed_env", ".netrc", ".pypirc", "credentials",
                 "token", "id_rsa", "id_ed25519", "id_ecdsa"}
PRIVATE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".7z")
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".jsonl", ".toml",
                 ".txt", ".md", ".tex", ".sty", ".cfg", ".ini", ".cff"}


def release_files(root=ROOT):
    """Yield deliverable files; exclude generated data, never vendored models code."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if (rel.parts[0] in RUNTIME_ROOTS or rel.parts[0].startswith(".venv-")
                or any(part in CACHE_DIRS or part.endswith(".egg-info") for part in rel.parts)):
            continue
        if path.suffix in BUILD_SUFFIXES or path.name.endswith((".synctex.gz", ".pyc")):
            continue
        # Uploaded transfer bundles can contain unreviewed cluster files; package
        # integrated, audited source files, never nested archives.
        if path.name.lower().endswith(ARCHIVE_SUFFIXES):
            continue
        if (path.name in PRIVATE_NAMES or path.suffix in PRIVATE_SUFFIXES
                or (path.name.startswith(".env.") and path.name != ".env.example")):
            continue
        if path.is_symlink():
            yield path
        elif path.is_file():
            yield path


def audit(root=ROOT):
    errors = []
    # Cluster-specific roots are never appropriate as embedded release defaults.
    cluster_path = re.compile(r"/(?:home|scratch|project|projects|cvmfs|lustre|gpfs)/[A-Za-z0-9]")
    secret = re.compile(r"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|"
                        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
    count = 0
    for path in release_files(root):
        count += 1
        rel = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"{rel}: symlink; vendor a regular file instead")
            continue
        if "experiment_evidence" in rel.parts or "source_snapshot_manifest.json" == path.name:
            errors.append(f"{rel}: internal experiment evidence excluded from release")
        if path.suffix in {".safetensors", ".pt", ".pth", ".ckpt"}:
            errors.append(f"{rel}: model/checkpoint binary excluded from release")
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"Makefile", "Dockerfile"}:
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if cluster_path.search(line):
                errors.append(f"{rel}:{number}: embedded cluster-specific path")
            if secret.search(line):
                errors.append(f"{rel}:{number}: possible credential (value suppressed)")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(rel))
                if rel.parts[0] != "third_party":
                    for node in ast.walk(tree):
                        module = getattr(node, "module", "") or ""
                        imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [module]
                        if any(name.startswith(("matplotlib", "seaborn", "plotly")) for name in imports):
                            errors.append(f"{rel}: plotting dependency in release script")
                            break
            except SyntaxError as exc:
                errors.append(f"{rel}:{exc.lineno}: Python syntax error: {exc.msg}")
        if path.suffix == ".sh":
            result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f"{rel}: shell syntax error")
    return count, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    count, errors = audit(args.root)
    for error in errors:
        print(error)
    print(f"Release audit: {count} files, {len(errors)} issue(s).")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
