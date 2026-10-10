"""Unit tests for the ignore-aware evaluator (analysis.evaluation.ignore_eval)."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.evaluation.ignore_eval import ImageEval, ap_101, evaluate, match_image, yolo_to_xyxy

G1 = [100, 100, 140, 130]
G2 = [300, 200, 360, 260]


def img(gt, pred, ign=()):
    return ImageEval(np.array(gt, float).reshape(-1, 4), np.array(pred, float).reshape(-1, 5), np.array(ign, float).reshape(-1, 4))


class TestMatching(unittest.TestCase):
    def test_perfect_predictions(self):
        r = evaluate({"a": img([G1, G2], [G1 + [0.9], G2 + [0.8]])})
        self.assertAlmostEqual(r["AP50"], 1.0)
        self.assertEqual((r["tp@conf"], r["fp@conf"], r["fn@conf"]), (2, 0, 0))

    def test_prediction_on_ignore_region_is_not_a_false_positive(self):
        review = [500, 300, 560, 350]
        with_ign = evaluate({"a": img([G1], [G1 + [0.9], review + [0.95]], [review])})
        without = evaluate({"a": img([G1], [G1 + [0.9], review + [0.95]])})
        self.assertEqual(with_ign["fp@conf"], 0)
        self.assertAlmostEqual(with_ign["precision@conf"], 1.0)
        self.assertEqual(without["fp@conf"], 1)
        self.assertLess(without["AP50"], with_ign["AP50"])

    def test_ignore_region_never_creates_false_negatives(self):
        r = evaluate({"a": img([G1], [G1 + [0.9]], [[500, 300, 560, 350]])})
        self.assertEqual((r["gt"], r["fn@conf"]), (1, 0))

    def test_small_prediction_inside_large_ignore_region_uses_ioa(self):
        big = [400, 200, 600, 400]
        s, l, n = match_image(img([], [[450, 250, 470, 265, 0.9]], [big]), 0.5)
        self.assertEqual(len(l), 0)                       # IoU tiny, IoA = 1 -> ignored

    def test_missed_gt_lowers_recall(self):
        r = evaluate({"a": img([G1, G2], [G1 + [0.9]])})
        self.assertAlmostEqual(r["recall@conf"], 0.5)

    def test_duplicate_detection_is_false_positive(self):
        r = evaluate({"a": img([G1], [G1 + [0.9], [101, 101, 141, 131, 0.8]])})
        self.assertEqual((r["tp@conf"], r["fp@conf"]), (1, 1))

    def test_duplicate_matching_ignore_region_after_gt_is_ignored_not_tp(self):
        # a second prediction on a GT that also lies in an ignore region is ignored, not counted twice
        r = evaluate({"a": img([G1], [G1 + [0.9], [101, 101, 141, 131, 0.8]], [G1])})
        self.assertEqual((r["tp@conf"], r["fp@conf"]), (1, 0))

    def test_min_pred_area_scope(self):
        tiny = [10, 10, 15, 14, 0.99]                   # 20 px^2
        r = evaluate({"a": img([G1], [G1 + [0.9], tiny])}, min_pred_area=50)
        self.assertEqual(r["fp@conf"], 0)

    def test_conf_threshold(self):
        r = evaluate({"a": img([G1, G2], [G1 + [0.9], G2 + [0.1]])}, conf=0.25)
        self.assertEqual(r["tp@conf"], 1)
        self.assertAlmostEqual(r["AP50"], 1.0)           # AP is threshold-free

    def test_size_bins_ignore_other_sizes(self):
        small = [10, 10, 25, 25]                          # 225 px^2
        large = [200, 200, 300, 300]                      # 10000 px^2
        r = evaluate({"a": img([small, large], [small + [0.9], large + [0.8]])})
        self.assertEqual(r["by_size"]["small"]["gt"], 1)
        self.assertAlmostEqual(r["by_size"]["small"]["AP50"], 1.0)
        self.assertAlmostEqual(r["by_size"]["large"]["AP50"], 1.0)
        self.assertEqual(r["by_size"]["medium"]["gt"], 0)


class TestAP(unittest.TestCase):
    def test_hand_computed_101_point_ap(self):
        # TP(0.9), FP(0.8), TP(0.7), 2 GT -> precision envelope 1.0 up to recall 0.5, 2/3 up to recall 1.0
        ap = ap_101(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1], np.int8), 2)["ap"]
        self.assertAlmostEqual(ap, (51 * 1.0 + 50 * (2 / 3)) / 101, places=6)

    def test_no_gt_gives_nan(self):
        self.assertTrue(np.isnan(ap_101(np.array([0.5]), np.array([0], np.int8), 0)["ap"]))

    def test_agrees_with_ultralytics_ap_per_class_without_ignores(self):
        """Same TP/FP decisions -> AP within 0.02 of Ultralytics (it integrates with trapz, we average 101 points)."""
        from ultralytics.utils.metrics import ap_per_class
        rng = np.random.default_rng(0)
        images, tps, confs, n_gt = {}, [], [], 0
        for k in range(40):
            gt = []
            for _ in range(rng.integers(1, 6)):
                x, y = rng.uniform(0, 600), rng.uniform(0, 400)
                gt.append([x, y, x + rng.uniform(10, 80), y + rng.uniform(10, 60)])
            pred = []
            for g in gt:
                if rng.random() < 0.8:
                    j = rng.normal(0, 3, 4)
                    pred.append(list(np.array(g) + j) + [rng.uniform(0.3, 1.0)])
            for _ in range(rng.integers(0, 3)):
                x, y = rng.uniform(0, 600), rng.uniform(0, 400)
                pred.append([x, y, x + 30, y + 30, rng.uniform(0.0, 0.7)])
            images[str(k)] = img(gt, pred)
            s, l, n = match_image(images[str(k)], 0.5)
            tps.append(l); confs.append(s); n_gt += n
        ours = evaluate(images)["AP50"]
        tp = np.concatenate(tps).astype(bool)[:, None]
        conf = np.concatenate(confs)
        out = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_gt))
        ul_ap = float(np.asarray(out[5]).reshape(-1)[0])  # ap array (nc, n_iou) -> AP50
        self.assertAlmostEqual(ours, ul_ap, delta=0.02)


class TestRealValidationSplit(unittest.TestCase):
    """End-to-end on yolo-rock-v1.1 val without any model: label files as perfect predictions, REVIEW boxes as extra ones."""

    @classmethod
    def setUpClass(cls):
        import json
        from pathlib import Path
        from analysis.evaluation.ignore_eval import load_split
        root = Path(__file__).resolve().parents[2]
        cls.ds = root / "dataset/processed/yolo-rock-v1.1"
        if not (cls.ds / "labels/val").exists():
            raise unittest.SkipTest("yolo-rock-v1.1 not present")
        cls.ign = cls.ds / "manifests/eval_ignore_regions.json"
        base = load_split(cls.ds, "val", None, [cls.ign])
        cls.imgs = {}
        for name, im in base.items():
            gt_pred = np.c_[im.gt, np.full(len(im.gt), 0.9)]
            rv_pred = np.c_[im.ignore, np.full(len(im.ignore), 0.95)]  # model "detects" every REVIEW region
            cls.imgs[name] = ImageEval(im.gt, np.vstack([gt_pred, rv_pred]).reshape(-1, 5), im.ignore)
        cls.n_review = sum(len(v) for v in json.loads(cls.ign.read_text(encoding="utf-8"))["regions"]["val"].values())

    def test_review_detections_are_ignored_not_false_positives(self):
        r = evaluate(self.imgs)
        self.assertEqual(r["fp@conf"], 0)
        self.assertAlmostEqual(r["AP50"], 1.0, places=6)
        self.assertEqual(r["gt"], 25822)

    def test_without_ignore_regions_the_same_detections_are_penalised(self):
        stripped = {k: ImageEval(v.gt, v.pred, np.zeros((0, 4))) for k, v in self.imgs.items()}
        r = evaluate(stripped)
        self.assertGreater(r["fp@conf"], 0.9 * self.n_review)    # REVIEW detections become FP (a few hit VALID boxes)
        self.assertLess(r["precision@conf"], 0.6)


class TestYoloParsing(unittest.TestCase):
    def test_yolo_to_xyxy(self):
        b = yolo_to_xyxy(["0 0.5 0.5 0.1 0.2 0.75"], with_score=True)
        np.testing.assert_allclose(b[0], [324, 192, 396, 288, 0.75])


if __name__ == "__main__":
    unittest.main()
