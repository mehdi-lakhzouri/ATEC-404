"""Training scaffolding must resolve configs without training and must refuse to train without authorisation."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from training import train


class TestTrainingGuard(unittest.TestCase):
    def test_refuses_without_env_and_flag(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTEMIS_TRAINING_AUTHORIZED", None)
            with mock.patch.object(train, "train", side_effect=AssertionError("must not train")):
                self.assertEqual(train.main(["--arm", "A_steps", "--seed", "0"]), 3)
                self.assertEqual(train.main(["--arm", "A_steps", "--seed", "0", "--authorized"]), 3)

    def test_env_alone_is_not_enough(self):
        with mock.patch.dict(os.environ, {"ARTEMIS_TRAINING_AUTHORIZED": "1"}):
            with mock.patch.object(train, "train", side_effect=AssertionError("must not train")):
                self.assertEqual(train.main(["--arm", "B_steps", "--seed", "1"]), 3)

    def test_dry_run_resolves_fair_arms(self):
        a = train.resolve("A_steps", 0)["ultralytics_args"]
        b = train.resolve("B_steps", 0)["ultralytics_args"]
        same = {k for k in a if k not in ("data", "epochs", "name")}
        self.assertTrue(all(a[k] == b[k] for k in same))               # identical hyperparameters
        iters = lambda n, e: -(-n // 16) * e                           # noqa: E731  ceil(n/batch) * epochs
        self.assertLess(abs(iters(1903, a["epochs"]) - iters(7353, b["epochs"])) / iters(7353, b["epochs"]), 0.01)
        self.assertTrue(a["data"].endswith("data_A.yaml") and b["data"].endswith("data_B.yaml"))
        self.assertEqual((a["seed"], a["deterministic"], a["patience"], a["batch"]), (0, True, 0, 16))

    def test_dry_run_records_provenance_and_unfrozen_status(self):
        m = train.resolve("A_steps", 2)
        self.assertEqual(len(m["manifest_sha256"]), 64)
        self.assertFalse(m["dataset_status"]["frozen"])
        self.assertFalse(m["dataset_status"]["approved_for_training"])


class TestLossRecord(unittest.TestCase):
    """Regression for the smoke-test-001 crash: Ultralytics 8.4 tloss is a dict of tensors."""

    def test_dict_of_tensors(self):
        import torch
        rec, ok = train.loss_record({"box_loss": torch.tensor(2.36), "cls_loss": torch.tensor(2.25), "dfl_loss": torch.tensor(1.17)})
        self.assertTrue(ok)
        self.assertEqual(set(rec), {"box_loss", "cls_loss", "dfl_loss"})
        self.assertAlmostEqual(rec["box_loss"], 2.36, places=4)

    def test_tensor_sequence_and_none(self):
        import torch
        rec, ok = train.loss_record(torch.tensor([1.0, 2.0, 3.0]))
        self.assertTrue(ok and len(rec) == 3)
        self.assertEqual(train.loss_record(None), (None, True))

    def test_non_finite_detected(self):
        import torch
        _, ok = train.loss_record({"box_loss": torch.tensor(float("nan")), "cls_loss": torch.tensor(1.0)})
        self.assertFalse(ok)
        _, ok = train.loss_record({"box_loss": torch.tensor(float("inf"))})
        self.assertFalse(ok)

    def test_overrides_parsing(self):
        self.assertEqual(train._parse_overrides(["epochs=3", "amp=true", "lr0=0.01", "device=0"]),
                         {"epochs": 3, "amp": True, "lr0": 0.01, "device": 0})
        with self.assertRaises(ValueError):
            train._parse_overrides(["epochs"])


class TestClassNaming(unittest.TestCase):
    """E.1a regression: checkpoints must carry {0: 'rock'} (single_cls=True renamed it to 'item' in smoke-test-001)."""

    def test_validator_accepts_only_rock(self):
        train.check_class_names({0: "rock"})
        train.check_class_names(["rock"])
        for bad in ({0: "item"}, {0: "rock", 1: "shadow"}, {0: "Rock"}, {}, ["item"]):
            with self.assertRaises(ValueError, msg=str(bad)):
                train.check_class_names(bad)

    def test_config_disables_single_cls(self):
        for arm in ("A_steps", "B_steps", "A_epochs"):
            self.assertIs(train.resolve(arm, 0)["ultralytics_args"]["single_cls"], False, arm)

    def test_data_yamls_declare_rock(self):
        import yaml
        for cfg in ("A", "B"):
            d = yaml.safe_load((train.ROOT / f"dataset/processed/yolo-rock-v1.1/data_{cfg}.yaml").read_text(encoding="utf-8"))
            self.assertEqual((d["names"], d["nc"]), ({0: "rock"}, 1), cfg)

    def test_smoke_test_checkpoint_is_rejected(self):
        ck = train.ROOT / "experiments/task-e/smoke-test-001/weights/best.pt"
        if not ck.exists():
            self.skipTest("smoke-test checkpoint not present")
        from ultralytics import YOLO
        with self.assertRaises(ValueError):
            train.check_class_names(YOLO(str(ck)).names)

    def test_effective_args_mismatch_is_a_limit_violation(self):
        from types import SimpleNamespace
        expected = {"epochs": 1, "batch": 16, "time": None, "single_cls": False}
        ok = SimpleNamespace(epochs=1, batch=16, time=None, single_cls=False)
        self.assertEqual(train.check_effective_args(ok, {0: "rock"}, expected)["epochs"], 1)
        for k, v in (("epochs", 2), ("batch", 8), ("time", 0.5), ("single_cls", True)):
            bad = SimpleNamespace(**{**vars(ok), k: v})
            with self.assertRaises(train.LimitViolation, msg=k):
                train.check_effective_args(bad, {0: "rock"}, expected)
        with self.assertRaises(ValueError):
            train.check_effective_args(ok, {0: "item"}, expected)


class TestEpochLimits(unittest.TestCase):
    """E.1c: per-epoch limits and completed-epoch checkpoint naming."""

    def test_in_order_epochs_pass(self):
        for i in range(200):
            train.check_epoch_start(i, i, 200, 16, 16, 0)

    def test_extra_epoch_is_refused(self):
        with self.assertRaises(train.LimitViolation):
            train.check_epoch_start(200, 200, 200, 16, 16, 0)

    def test_repeated_epoch_is_refused(self):           # Ultralytics NaN recovery redoes the epoch index
        with self.assertRaises(train.LimitViolation):
            train.check_epoch_start(41, 42, 200, 16, 16, 0)

    def test_oom_reduced_batch_is_refused(self):
        for bs, retries in ((8, 0), (16, 1)):
            with self.assertRaises(train.LimitViolation):
                train.check_epoch_start(0, 0, 200, bs, 16, retries)

    def test_aligned_checkpoint_names(self):
        names = [train.aligned_checkpoint(i, 10) for i in range(200)]
        kept = [n for n in names if n]
        self.assertEqual(len(kept), 20)
        self.assertEqual((kept[0], kept[-1]), ("epoch_completed_010.pt", "epoch_completed_200.pt"))
        self.assertIsNone(train.aligned_checkpoint(9, 0))


if __name__ == "__main__":
    unittest.main()
