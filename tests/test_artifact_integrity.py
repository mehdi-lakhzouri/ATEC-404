"""Published artifacts, frozen model and once-only test evidence are intact and free of local paths."""

import json
import re
import unittest
from pathlib import Path

from tests._common import ROOT

import verify_artifacts  # noqa: E402


class TestArtifactIntegrity(unittest.TestCase):
    def test_all_verification_checks_pass(self):
        res = verify_artifacts.verify()
        failed = {k: v for k, v in res.items() if v["ok"] is False}
        self.assertEqual(failed, {})

    def test_manifest_covers_key_evidence(self):
        man = json.loads((ROOT / "artifacts/provenance/ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
        paths = {f["path"] for f in man["files"]}
        for p in ("models/artemis_rock_yolo11n.pt", "artifacts/provenance/FROZEN_MODEL_CONFIG.json", "artifacts/provenance/TEST_RUN_LOCK.json",
                  "artifacts/evaluation/test_results.json", "artifacts/training/results.csv", "data/manifests/split_manifest.csv"):
            self.assertIn(p, paths)
        self.assertEqual(man["missing_sources"], [])

    def test_published_text_has_no_local_paths(self):
        pat = re.compile(r"[A-Za-z]:(\\\\|\\|/)Users|" + re.escape(Path.home().name))
        offenders = []
        for d in ("artifacts", "data", "docs", "notebooks", "src", "scripts", "tests", "future", "README.md", "models/model_card.md", "CITATION.cff"):
            base = ROOT / d
            files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
            for p in files:
                if p.suffix.lower() in (".json", ".md", ".csv", ".yaml", ".yml", ".py", ".ipynb", ".txt", ".sha256", ".jsonl", ".cff"):
                    if "audits/raw" in p.as_posix() or "data/local" in p.as_posix():   # git-ignored, machine-specific
                        continue
                    if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
                        offenders.append(p.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_frozen_config_unchanged_values(self):
        cfg = json.loads((ROOT / "artifacts/provenance/FROZEN_MODEL_CONFIG.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["frozen"]); self.assertFalse(cfg["test_split_used"])
        self.assertEqual(cfg["model"]["sha256"], "413d0481a728a310f5b919d069676ee5ddd2091e81b4b663831e1e977a0752eb")
        inf = cfg["inference"]
        self.assertEqual((inf["imgsz"], inf["conf"], inf["nms_iou"], inf["max_det"], inf["min_box_area_px2"]), (736, 0.2, 0.6, 300, 50.0))


if __name__ == "__main__":
    unittest.main()
