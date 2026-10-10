"""Regression tests for Task C v1.1 (rule D05 applied to watershed split pieces).

Run from backend/:  uv run --no-sync python -m unittest discover -s tests -t . -v
Integration tests read dataset/processed/curated-dataset-v1{,.1} and skip if those are absent.
"""

from __future__ import annotations

import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.dataset_cleaning.build import duplicates_csv

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "dataset" / "processed" / "curated-dataset-v1"
V11 = ROOT / "dataset" / "processed" / "curated-dataset-v1.1"
# byte-identical copy of yolo-rock-v1/manifests/export_exclusions.csv (sha256 recorded in RETIRED_VERSIONS.json);
# yolo-rock-v1 was retired on 2026-10-10
EXCL = Path(__file__).resolve().parent / "fixtures" / "yolo_rock_v1_export_exclusions.csv"


def _rows(p: Path) -> dict[str, dict[str, str]]:
    with p.open(newline="", encoding="utf-8") as f:
        return {r["ann_id"]: r for r in csv.DictReader(f)}


class TestDuplicatesCsvUnit(unittest.TestCase):
    cb = np.array([[100.0, 100.0, 200.0, 180.0]])

    def test_blue_piece_inside_csv_is_duplicate(self):
        self.assertTrue(duplicates_csv(np.array([120, 110, 160, 150]), "large", self.cb))   # fully inside, small IoU

    def test_blue_piece_partly_inside_low_iou_is_not_duplicate(self):
        self.assertFalse(duplicates_csv(np.array([180, 100, 260, 180]), "large", self.cb))  # 25% inside, IoU 0.14

    def test_green_needs_iou_0_5(self):
        self.assertTrue(duplicates_csv(np.array([105, 100, 200, 180]), "small", self.cb))   # IoU 0.95
        self.assertFalse(duplicates_csv(np.array([120, 110, 160, 150]), "small", self.cb))  # inside but IoU 0.2

    def test_no_csv_boxes(self):
        self.assertFalse(duplicates_csv(np.array([0, 0, 10, 10]), "large", np.zeros((0, 4))))


@unittest.skipUnless(V1.exists() and V11.exists() and EXCL.exists(), "curated datasets not present")
class TestTaskCV11Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = _rows(V1 / "annotations" / "annotations_all.csv")
        cls.v11 = _rows(V11 / "annotations" / "annotations_all.csv")
        cls.excluded = set(_rows(EXCL))
        cls.changed = {k for k in cls.v1 if cls.v1[k] != cls.v11.get(k)}

    def test_the_146_export_exclusions_are_fixed_at_source(self):
        self.assertEqual(len(self.excluded), 146)
        for a in self.excluded:
            r = self.v11[a]
            self.assertEqual(r["status"], "REJECTED", a)
            self.assertEqual(r["reasons"], "duplicate_of_csv_instance", a)
            self.assertIn("split_piece_dedup", r["flags"], a)

    def test_valid_set_shrinks_by_exactly_the_146(self):
        v1_valid = {k for k, r in self.v1.items() if r["status"] == "VALID"}
        v11_valid = {k for k, r in self.v11.items() if r["status"] == "VALID"}
        self.assertEqual(v1_valid - v11_valid, self.excluded)
        self.assertEqual(v11_valid - v1_valid, set())

    def test_same_ann_ids_no_records_added_or_lost(self):
        self.assertEqual(set(self.v1), set(self.v11))

    def test_every_change_is_a_split_piece_dedup(self):
        for a in self.changed:
            r = self.v11[a]
            self.assertTrue(r["source"].endswith("_split"), a)
            self.assertEqual((r["status"], r["reasons"]), ("REJECTED", "duplicate_of_csv_instance"), a)
            for k in ("x0", "y0", "x1", "y1", "frame_id", "parent_id", "source"):
                self.assertEqual(self.v1[a][k], r[k], (a, k))

    def test_csv_instances_identical(self):
        csv1 = {k: r for k, r in self.v1.items() if r["source"] == "csv_large"}
        self.assertEqual(len(csv1), 17685)
        for k, r in csv1.items():
            self.assertEqual(r, self.v11[k], k)

    def test_no_valid_split_piece_duplicates_a_csv_instance(self):
        by_frame = defaultdict(list)
        for r in self.v11.values():
            by_frame[r["frame_id"]].append(r)
        for f, rs in by_frame.items():
            cb = np.array([[float(r[k]) for k in ("x0", "y0", "x1", "y1")] for r in rs if r["source"] == "csv_large"]).reshape(-1, 4)
            for r in rs:
                if r["status"] == "VALID" and r["source"].endswith("_split"):
                    cls = "large" if r["source"].startswith("mask_blue") else "small"
                    box = np.array([float(r[k]) for k in ("x0", "y0", "x1", "y1")])
                    self.assertFalse(duplicates_csv(box, cls, cb), r["ann_id"])

    def test_counts_and_policy_version(self):
        counts = json.loads((V11 / "metadata" / "counts.json").read_text(encoding="utf-8"))["status_counts"]
        self.assertEqual(counts, {"REJECTED": 50005, "REVIEW_REQUIRED": 192784, "VALID": 261239})
        self.assertEqual(json.loads((V11 / "metadata" / "policy.json").read_text(encoding="utf-8"))["policy_version"], "S2-H/1.1")


if __name__ == "__main__":
    unittest.main()
