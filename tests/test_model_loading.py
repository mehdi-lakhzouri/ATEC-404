"""The frozen detector loads with checksum verification and reproduces the historical predictions."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests._common import ROOT

from artemis_x.config import paths  # noqa: E402
from artemis_x.ml import model_registry  # noqa: E402
from artemis_x.ml.inference import RockDetector  # noqa: E402

CACHED = paths.E1C_RUN / "ignore_aware_val" / "best" / "val" / "labels"   # Task F cached predictions (conf >= 0.001)


class TestFrozenSpec(unittest.TestCase):
    def test_spec_matches_frozen_config(self):
        s = model_registry.load_frozen_spec()
        self.assertEqual(s.class_names, {0: "rock"})
        self.assertEqual(s.predict_kwargs(), {"imgsz": 736, "conf": 0.2, "iou": 0.6, "max_det": 300})
        self.assertEqual((s.min_box_area_px2, s.epoch), (50.0, 111))

    def test_tampered_weights_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "artemis_rock_yolo11n.pt"
            shutil.copyfile(paths.FROZEN_MODEL, bad)
            with bad.open("r+b") as fh:
                fh.seek(1000); b = fh.read(1); fh.seek(1000); fh.write(bytes([b[0] ^ 0xFF]))
            with mock.patch.object(paths, "FROZEN_MODEL", bad):
                with self.assertRaises(model_registry.ModelIntegrityError):
                    model_registry.load_frozen_spec()


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = RockDetector(device="cpu")

    def test_blank_frame_runs_on_cpu(self):
        r = self.cpu.predict(np.zeros((480, 720, 3), dtype=np.uint8))
        self.assertEqual(r.device, "cpu")
        self.assertEqual(r.image_hw, (480, 720))
        self.assertTrue(all(d.area >= 50 and d.confidence >= 0.2 for d in r.detections))

    @unittest.skipUnless(CACHED.is_dir() and paths.local_dataset_available(), "local experiments/dataset not present")
    def test_parity_with_task_f_cached_predictions(self):
        """New entry point == historical protocol: cached conf>=0.001 predictions filtered at the frozen conf and area."""
        from artemis_x.evaluation.ignore_aware import iou, yolo_to_xyxy
        det = RockDetector()
        for name in ("render0115", "render5589", "render0575", "render6352"):
            live = det.predict(paths.YOLO_DATASET / "images" / "val" / f"{name}.png").as_array()
            c = yolo_to_xyxy((CACHED / f"{name}.txt").read_text().splitlines(), with_score=True)
            c = c[c[:, 4] >= det.spec.conf]
            c = c[(c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) >= det.spec.min_box_area_px2]
            self.assertEqual(len(live), len(c), name)
            if len(live):
                self.assertTrue((iou(live[:, :4], c[:, :4]).max(axis=1) > 0.99).all(), name)


if __name__ == "__main__":
    unittest.main()
