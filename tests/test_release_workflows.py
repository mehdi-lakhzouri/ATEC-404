"""Release-only regression guards. No external service or user data is modified."""
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._common import ROOT
import jury
import run_webots_smoke

spec = importlib.util.spec_from_file_location("fetch_assets_release", ROOT / "artemis-webots/tools/fetch_polyhaven.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class TestLockedAssets(unittest.TestCase):
    def fixture(self, directory, filename="texture/map.jpg"):
        payload = b"test fixture, not lunar imagery"
        manifest = {"assets": {"texture": {"type": "texture", "files": [{"file": filename,
                    "url": "https://example.invalid/fixture", "sha256": hashlib.sha256(payload).hexdigest()}]}}}
        path = directory / "ASSETS.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return payload, path.read_bytes()

    def test_download_uses_frozen_manifest_and_never_rewrites_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload, manifest = self.fixture(directory)
            with patch.object(fetch, "DEST", directory), patch.object(fetch, "_get", return_value=payload) as download, patch("sys.argv", ["fetch", "--locked"]):
                self.assertEqual(fetch.main(), 0)
                download.assert_called_once_with("https://example.invalid/fixture")
            self.assertEqual((directory / "ASSETS.json").read_bytes(), manifest)
            self.assertEqual((directory / "texture/map.jpg").read_bytes(), payload)

    def test_cached_verified_asset_does_not_request_latest_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            payload, _ = self.fixture(directory)
            (directory / "texture").mkdir()
            (directory / "texture/map.jpg").write_bytes(payload)
            with patch.object(fetch, "DEST", directory), patch.object(fetch, "_get") as download, patch("sys.argv", ["fetch", "--locked"]):
                self.assertEqual(fetch.main(), 0)
                download.assert_not_called()

    def test_checksum_mismatch_fails_without_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, manifest = self.fixture(directory)
            with patch.object(fetch, "DEST", directory), patch.object(fetch, "_get", return_value=b"wrong"), patch("sys.argv", ["fetch", "--locked"]):
                with self.assertRaisesRegex(SystemExit, "SHA256 mismatch"):
                    fetch.main()
            self.assertEqual((directory / "ASSETS.json").read_bytes(), manifest)
            self.assertFalse((directory / "texture/map.jpg").exists())

    def test_path_escape_is_refused_before_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, "../escape.bin")
            with patch.object(fetch, "DEST", directory), patch.object(fetch, "_get") as download, patch("sys.argv", ["fetch", "--locked"]):
                with self.assertRaisesRegex(SystemExit, "outside destination"):
                    fetch.main()
                download.assert_not_called()


@unittest.skipUnless(os.name == "nt", "Windows ownership identity guard")
class TestOwnership(unittest.TestCase):
    def test_creation_identity_is_stable_for_current_process(self):
        value = jury.process_identity(os.getpid())
        self.assertIsNotNone(value)
        self.assertEqual(value, jury.process_identity(os.getpid()))

    def test_foreign_manifest_is_refused_without_killing_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "active.json"
            path.write_text(json.dumps({"repo": "other repository", "run": temporary}))
            with patch.object(jury, "STATE", path), patch.object(jury.subprocess, "run") as kill:
                with self.assertRaisesRegex(RuntimeError, "outside this repository"):
                    jury.stop()
                kill.assert_not_called()

    def test_smoke_timeout_stops_only_its_launched_tree(self):
        from subprocess import TimeoutExpired
        from unittest.mock import Mock
        child = Mock(pid=123456, returncode=-9)
        child.communicate.side_effect = [TimeoutExpired("owned webots", 1), ("partial log", None)]
        with patch.object(run_webots_smoke.subprocess, "Popen", return_value=child), patch.object(run_webots_smoke.subprocess, "run") as kill:
            code, log = run_webots_smoke.run_webots(ROOT / ".jury/test-unused", False, 1)
            self.assertEqual(code, -9)
            self.assertIn("TIMEOUT", log)
            kill.assert_called_once_with(["taskkill", "/T", "/F", "/PID", "123456"], capture_output=True)


if __name__ == "__main__":
    unittest.main()
