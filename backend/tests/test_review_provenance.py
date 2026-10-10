"""Provenance tests: AI reviews and legacy exports must never be ingested as human review."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from analysis.yolo_dataset.__main__ import _ingest_human, is_human_export

ROOT = Path(__file__).resolve().parents[2]
AI = ROOT / "experiments/yolo-dataset/ai_review/taskd_ai_review_36_decisions.json"
GALLERY = ROOT / "experiments/yolo-dataset-v1.1/review_gallery/index.html"


class TestIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, obj):
        (self.tmp / name).write_text(json.dumps(obj), encoding="utf-8")

    def test_ai_audit_is_not_human(self):
        if not AI.exists():
            self.skipTest("AI audit not imported")
        j = json.loads(AI.read_text(encoding="utf-8"))
        self.assertFalse(is_human_export(j))
        shutil.copy(AI, self.tmp / AI.name)
        r = _ingest_human(self.tmp, {1, 676})
        self.assertEqual(r["scenes_reviewed"], 0)
        self.assertEqual(r["decisions"], {"NOT_REVIEWED": 2})
        self.assertEqual(len(r["rejected_files"]), 1)

    def test_legacy_v1_export_with_human_true_is_rejected(self):
        # v1 gallery stamped human_review=true on everything, including imported AI decisions
        self._write("old.json", {"gallery": "artemis-rock-review-v1", "human_review": True,
                                 "scenes": [{"frame_id": 1, "decision": "APPROVE"}]})
        r = _ingest_human(self.tmp, {1})
        self.assertEqual(r["scenes_reviewed"], 0)

    def test_v2_human_export_counts_only_human_ui_decisions(self):
        self._write("h.json", {"gallery": "artemis-rock-review-v2", "provenance": "human_ui", "human_review": True, "reviewer": "x",
                               "scenes": [{"frame_id": 1, "decision": "APPROVE", "source": "human_ui"},
                                          {"frame_id": 676, "decision": "FLAG", "source": None},
                                          {"frame_id": 999, "decision": "APPROVE", "source": "human_ui"}]})
        r = _ingest_human(self.tmp, {1, 676})
        self.assertEqual(r["scenes_reviewed"], 1)                     # 676 not human_ui, 999 not in gallery
        self.assertEqual(r["decisions"], {"APPROVE": 1, "NOT_REVIEWED": 1})

    def test_spoofed_provenance_without_v2_marker_rejected(self):
        self._write("s.json", {"gallery": "something", "provenance": "human_ui", "human_review": True,
                               "scenes": [{"frame_id": 1, "decision": "APPROVE", "source": "human_ui"}]})
        self.assertEqual(_ingest_human(self.tmp, {1})["scenes_reviewed"], 0)


@unittest.skipUnless(GALLERY.exists(), "v1.1 gallery not built")
class TestGalleryHtml(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = GALLERY.read_text(encoding="utf-8")

    def test_old_unconditional_human_stamp_is_gone(self):
        self.assertNotIn("human_review: true, scenes: D.scenes.map(s => ({frame_id: s.frame_id, split: s.split, ...dec(s.frame_id)}))", self.html)
        self.assertIn("provenance: 'human_ui'", self.html)
        self.assertIn("j.provenance === 'human_ui'", self.html)

    def test_ai_reference_embedded_as_non_human(self):
        data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', self.html, re.S).group(1).replace("<\\/", "</"))
        ai = data["ai_reference"]
        self.assertIs(ai.get("human_review"), False)
        self.assertEqual(len(ai.get("scenes", {})), 36)
        self.assertTrue(all(s["frame_id"] for s in data["scenes"]))

    def test_script_parses_with_node(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        js = re.findall(r"<script>(.*?)</script>", self.html, re.S)[0]
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(js)
        res = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
        Path(f.name).unlink(missing_ok=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
