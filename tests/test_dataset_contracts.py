"""Dataset contracts: split membership, leakage groups, class map and registry hashes."""

import hashlib
import unittest

import yaml

from tests._common import ROOT

from artemis_x.config import paths  # noqa: E402
from artemis_x.datasets import registry  # noqa: E402


class TestDatasetContracts(unittest.TestCase):
    def test_split_counts_and_no_leakage(self):
        c = registry.split_counts()
        self.assertEqual(c["splits"], registry.EXPECTED_SPLIT)
        self.assertEqual(c["config_A_train"], registry.EXPECTED_CONFIG_A_TRAIN)
        self.assertEqual(c["groups_spanning_splits"], 0)

    def test_config_a_is_subset_of_train(self):
        rows = registry.split_manifest()
        self.assertTrue(all(r["split"] == "train" for r in rows if r["in_config_A_train"] == "True"))

    def test_published_manifest_matches_registry(self):
        v = registry.versions()["versions"]["yolo-rock-v1.1"]
        got = hashlib.sha256((ROOT / "data/manifests/split_manifest.csv").read_bytes()).hexdigest()
        self.assertEqual(got, v["files"]["manifests/split_manifest.csv"])
        self.assertIn("NOT frozen", v["status"])

    def test_single_rock_class(self):
        for cfg in ("A", "B"):
            d = yaml.safe_load((ROOT / f"data/manifests/data_{cfg}.yaml").read_text(encoding="utf-8"))
            self.assertEqual((d["names"], d["nc"]), ({0: "rock"}, 1))

    @unittest.skipUnless(paths.local_dataset_available(), "local dataset not present")
    def test_local_dataset_matches_registry(self):
        self.assertTrue(all(registry.verify_local_yolo_dataset().values()))


if __name__ == "__main__":
    unittest.main()
