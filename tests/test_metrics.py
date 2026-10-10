"""Metric helpers and the ignore-aware wrapper are consistent with the historical implementation and results."""

import json
import unittest

from tests._common import ROOT

from artemis_x.evaluation import ignore_aware  # noqa: E402
from artemis_x.evaluation.metrics import false_positives_per_image, precision_recall_f, wilson_interval  # noqa: E402

import numpy as np  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_prf(self):
        m = precision_recall_f(8, 2, 2)
        self.assertAlmostEqual(m["precision"], 0.8); self.assertAlmostEqual(m["recall"], 0.8); self.assertAlmostEqual(m["f1"], 0.8)
        self.assertEqual(precision_recall_f(0, 0, 0)["f1"], 0.0)
        self.assertAlmostEqual(precision_recall_f(1, 1, 0, beta=2)["f2"], 5 * 0.5 / (4 * 0.5 + 1))

    def test_wilson(self):
        lo, hi = wilson_interval(50, 100)
        self.assertLess(lo, 0.5); self.assertGreater(hi, 0.5); self.assertAlmostEqual((lo + hi) / 2, 0.5, places=6)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_reproduces_published_test_counts(self):
        r = json.loads((ROOT / "artifacts/evaluation/test_results.json").read_text(encoding="utf-8"))["policies"]["PRIMARY_review_ignores"]
        m = precision_recall_f(r["tp"], r["fp"], r["fn"])
        self.assertAlmostEqual(m["precision"], r["precision"], places=9)
        self.assertAlmostEqual(m["recall"], r["recall"], places=9)
        self.assertAlmostEqual(m["f1"], r["f1"], places=9)
        self.assertAlmostEqual(false_positives_per_image(r["fp"], r["images"]), r["fp_per_image"], places=9)


class TestIgnoreAwareWrapper(unittest.TestCase):
    def test_is_the_proven_implementation(self):
        from analysis.evaluation import ignore_eval
        self.assertIs(ignore_aware.evaluate, ignore_eval.evaluate)
        self.assertIs(ignore_aware.match_image, ignore_eval.match_image)

    def test_review_region_prediction_is_ignored(self):
        im = ignore_aware.ImageEval(gt=np.array([[0, 0, 10, 10.]]), pred=np.array([[0, 0, 10, 10, .9], [50, 50, 60, 60, .8]]),
                                    ignore=np.array([[48, 48, 62, 62.]]))
        s, l, n = ignore_aware.match_image(im, 0.5, 0.5)
        self.assertEqual((list(l), n), ([1], 1))         # TP kept, prediction on the REVIEW region neither TP nor FP


if __name__ == "__main__":
    unittest.main()
