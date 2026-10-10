"""Workflow C safeguards: dataset download/verification and opt-in training reproduction (no training, no real data)."""

import hashlib
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from tests._common import ROOT

import download_dataset  # noqa: E402  (scripts/)
from artemis_x.config import paths  # noqa: E402
from artemis_x.datasets import release  # noqa: E402
from artemis_x.ml import training_repro as T  # noqa: E402


def _mini_release(d: Path) -> tuple[Path, dict]:
    files = {"config_A/images/train/a.png": b"img-a", "config_A/labels/train/a.txt": b"0 0.5 0.5 0.1 0.1\n",
             "images/val/b.png": b"img-b", "labels/val/b.txt": b"", "manifests/split_manifest.csv": b"frame_id,split\n"}
    src = d / "yolo-rock-v1.1"
    for rel, data in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(data)
    manifest = {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}
    z = d / "release.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for rel in files:
            zf.write(src / rel, f"yolo-rock-v1.1/{rel}")
    return z, manifest


class TestDownloadDataset(unittest.TestCase):
    def test_fails_clearly_without_source(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTEMIS_DATASET_SOURCE", None)
            self.assertEqual(download_dataset.main([]), download_dataset.EXIT_NO_SOURCE)

    def test_release_record_is_not_published_and_has_no_urls(self):
        rec = release.release_record()
        self.assertTrue(rec["status"].startswith("NOT PUBLISHED"))
        self.assertEqual(rec["archives"], [])
        self.assertEqual(rec["file_manifest_sha256"], release.sha256_file(release.FILE_MANIFEST))
        self.assertTrue(all(rec["registry_proof"][k]["matches_registry"] for k in ("labels_tree", "images_tree")))

    def test_committed_manifest_counts(self):
        m = release.read_file_manifest()
        self.assertEqual(len(m), 7490)
        self.assertEqual(sum(1 for r in m if r.startswith("config_A/images/train/")), 1903)
        self.assertFalse(any(r.startswith("config_B/") for r in m))

    def test_archive_install_verify_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            z, manifest = _mini_release(d)
            dest = d / "out" / "yolo-rock-v1.1"
            local_yaml = d / "local.yaml"
            with mock.patch.object(release, "read_file_manifest", return_value=manifest), \
                 mock.patch.object(release, "LOCAL_YAML", local_yaml), \
                 mock.patch.object(release, "write_local_data_yaml", lambda root, out=local_yaml: out):
                good = hashlib.sha256(z.read_bytes()).hexdigest()
                self.assertEqual(download_dataset.main(["--source", str(z), "--dest", str(dest), "--archive-sha256", "0" * 64]),
                                 download_dataset.EXIT_ARCHIVE)
                self.assertEqual(download_dataset.main(["--source", str(z), "--dest", str(dest), "--archive-sha256", good]), 0)
                self.assertTrue((dest / "config_A/images/train/a.png").exists())
                (dest / "images/val/b.png").write_bytes(b"tampered")
                self.assertEqual(download_dataset.main(["--check-only", "--dest", str(dest)]), download_dataset.EXIT_VERIFY)


class TestTrainingReproduction(unittest.TestCase):
    def test_overrides_match_historical_run(self):
        self.assertEqual(T.historical_config()["overrides"], T.E1C_OVERRIDES)

    def test_refuses_protected_output(self):
        with mock.patch.object(T, "REPRO_ROOT", paths.EXPERIMENTS / "task-e"):
            with self.assertRaises(PermissionError):
                T.plan_run(name="full-training-config-a-200-s42")

    def test_launch_disabled_without_both_safeguards(self):
        plan = T.RunPlan("x", Path("never"), Path("never.yaml"), dict(T.E1C_OVERRIDES), Path("never.jsonl"))
        with mock.patch.object(T.subprocess, "run", side_effect=AssertionError("must not start")):
            self.assertIsNone(T.launch(plan))
            self.assertIsNone(T.launch(plan, run=True, authorisation="yes"))
            self.assertIsNone(T.launch(plan, run=False, authorisation=T.AUTH_PHRASE))

    def test_cli_uses_separate_registry_and_dry_run(self):
        plan = T.RunPlan("x", Path("r"), Path("d.yaml"), dict(T.E1C_OVERRIDES), T.REPRO_REGISTRY)
        dry = plan.cli(authorised=False)
        self.assertIn("--dry-run", dry); self.assertNotIn("--authorized", dry)
        self.assertEqual(dry[dry.index("--registry") + 1], str(T.REPRO_REGISTRY))
        self.assertNotEqual(T.REPRO_REGISTRY, paths.EXPERIMENTS / "task-e" / "runs_registry.jsonl")


if __name__ == "__main__":
    unittest.main()
