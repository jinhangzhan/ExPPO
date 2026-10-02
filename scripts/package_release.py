#!/usr/bin/env python3
"""Build the code-only review ZIP; optionally package separate paper materials."""

import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

from check_release import ROOT, audit, release_files


def build_archives(root, output, include_paper=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    _, errors = audit(root)
    if errors:
        raise SystemExit("Release audit failed; run python scripts/check_release.py first.")
    # Freeze inputs before opening either archive; never include the output tree.
    inputs = [path for path in release_files(root) if not path.is_relative_to(output)]
    if not inputs:
        raise SystemExit("The archive output directory must not contain the project root.")
    pdf = root / "paper" / "iclr2027_conference.pdf"
    if include_paper and not pdf.is_file():
        raise SystemExit("Build the manuscript first: make -C paper")
    output.mkdir(parents=True, exist_ok=True)
    code = output / "ExPPO-code.zip"
    with zipfile.ZipFile(code, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in inputs:
            relative = path.relative_to(root)
            if relative.parts[0] == "paper" or relative.as_posix() == "docs/paper-materials.md":
                continue
            archive.write(path, "ExPPO/" + relative.as_posix())
    checksum = output / "ExPPO-code.zip.sha256"
    checksum.write_text(hashlib.sha256(code.read_bytes()).hexdigest() + "  " + code.name + "\n")
    artifacts = [code, checksum]
    if include_paper:
        paper = output / "ExPPO-paper-source.zip"
        with zipfile.ZipFile(paper, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in inputs:
                relative = path.relative_to(root)
                if relative.parts[0] == "paper" and path != pdf:
                    archive.write(path, path.relative_to(root / "paper").as_posix())
        submission_pdf = output / "ExPPO-paper.pdf"
        shutil.copy2(pdf, submission_pdf)
        artifacts.extend([paper, submission_pdf])
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--include-paper", action="store_true", help="Also build separate paper artifacts; the code ZIP stays code-only")
    args = parser.parse_args()
    for path in build_archives(ROOT, args.output, args.include_paper):
        print(path.name, f"{path.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
