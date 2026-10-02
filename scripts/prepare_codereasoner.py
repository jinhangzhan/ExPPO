#!/usr/bin/env python3
"""Download hash-pinned code sources and rebuild the original function-group splits."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_specs():
    return json.loads((ROOT / "configs/code/data_sources.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(raw_root):
    raw_root.mkdir(parents=True, exist_ok=True)
    for spec in source_specs():
        target = raw_root / spec["local_filename"]
        if target.exists():
            if sha256(target) != spec["sha256"]:
                raise ValueError(f"Existing file has the wrong SHA-256: {target}; move it aside first")
            print(f"Verified {target}", flush=True)
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=raw_root, delete=False) as output:
                temporary = Path(output.name)
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "ExPPO-reproduction/1"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
            if sha256(temporary) != spec["sha256"]:
                raise ValueError(f"Downloaded source hash mismatch: {spec['key']}")
            temporary.replace(target)
            print(f"Downloaded and verified {target}", flush=True)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/codereasoner/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/codereasoner/prepared"))
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded source files")
    parser.add_argument("--list", action="store_true", help="Print pinned sources without downloading")
    parser.add_argument("--dry-run", action="store_true", help="Validate and plan from existing files without writes")
    parser.add_argument("--force", action="store_true", help="Replace existing prepared artifacts")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(source_specs(), indent=2))
        return
    if args.download_only and (args.dry_run or args.offline):
        parser.error("--download-only cannot be combined with --dry-run or --offline")
    if not args.offline and not args.dry_run:
        download_sources(args.raw_root)
    if args.download_only:
        return
    from evaluation.code.prepare import main as build
    command = ["--raw-root", str(args.raw_root), "--output-dir", str(args.output_dir)]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    build(command)


if __name__ == "__main__":
    main()
