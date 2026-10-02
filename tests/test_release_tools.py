import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_runtime_exclusions_keep_runtime_model_code(self):
        audit = load_script("check_release")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("outputs/results.json", "models/model.py", "evaluation/data/prepare.py",
                         "third_party/verl/verl/models/runtime.py", "paper/figures/rendered.pdf",
                         ".env.local", "nested/.env.production", ".netrc", "nested/id_ed25519",
                         "original_transfer.zip", "nested/private.tar.gz"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            files = {path.relative_to(root).as_posix() for path in audit.release_files(root)}
            self.assertEqual(files, {"evaluation/data/prepare.py", "third_party/verl/verl/models/runtime.py",
                                     "paper/figures/rendered.pdf"})

    def test_review_zip_is_code_only_and_needs_no_latex(self):
        with mock.patch.dict(sys.modules, {"check_release": load_script("check_release")}):
            module = load_script("package_release")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "scripts/evaluate.py", "paper/source.tex",
                         "paper/figure.pdf", "docs/paper-materials.md", "transfer.zip",
                         "data/private.json", ".env.local"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = root / "custom-artifacts"
            output.mkdir()
            (output / "previous-output.txt").touch()
            artifacts = module.build_archives(root, output)
            with zipfile.ZipFile(artifacts[0]) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(set(archive.namelist()), {"ExPPO/README.md", "ExPPO/scripts/evaluate.py"})
            expected = hashlib.sha256(artifacts[0].read_bytes()).hexdigest()
            self.assertEqual(artifacts[1].read_text(), expected + "  ExPPO-code.zip\n")
            self.assertFalse((output / "ExPPO-paper.pdf").exists())

    def test_archive_output_cannot_contain_source(self):
        with mock.patch.dict(sys.modules, {"check_release": load_script("check_release")}):
            module = load_script("package_release")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").touch()
            with self.assertRaises(SystemExit):
                module.build_archives(root, root)

    def test_macro_sd_uses_within_run_means(self):
        module = load_script("summarize_evaluations")
        specs = module.benchmark_specs()
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for index in range(3):
                payload = {}
                for task_index, (name, spec) in enumerate(specs.items()):
                    # Individual task variability cancels in every domain average.
                    value = 50 + (index - 1) * (1 if task_index % 2 else -1) * 10
                    payload[name] = {str(spec["k"]): {"pass_at_k": value, "avg_at_k": value / 2,
                                                       "pool_size": spec["k"], "prompts": spec["count"]}}
                path = Path(directory) / f"run{index}.json"
                path.write_text(json.dumps(payload))
                files.append(path)
            result = module.aggregate(files)
            self.assertEqual(result["id_average"]["pass_at_k"]["sd"], 0)
            self.assertEqual(result["ood_average"]["pass_at_k"]["sd"], 0)
            self.assertEqual(result["aime-24"]["pass_at_k"]["sd"], 10)

    def test_overhead_measurement_window_and_gpu_hours(self):
        module = load_script("summarize_overhead")
        lines = [f"training/global_step:{i} timing_s/step:2 perf/max_memory_allocated_gb:10 "
                 f"perf/max_memory_reserved_gb:12" for i in range(1, 101)]
        result = module.summarize(module.load_metrics("\n".join(lines)), 3600)
        self.assertEqual(result["estimated_gpu_hours_900_steps"], 72)
        self.assertEqual(result["steps_per_second"], 0.5)
        self.assertEqual(result["samples_per_second"], 256)
        with self.assertRaises(ValueError):
            module.load_metrics("\n".join(lines[:-1]))


if __name__ == "__main__":
    unittest.main()
