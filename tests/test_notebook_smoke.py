"""Notebooks are valid, executed, error-free, small, and can never train or touch the held-out test split."""

import re
import unittest

from tests._common import ROOT

try:
    import nbformat
except ImportError:  # notebook tooling is layered on demand: uv run --with nbformat ...
    nbformat = None

import build_notebooks  # noqa: E402  (scripts/)

NOTEBOOKS = sorted((ROOT / "notebooks").glob("0*.ipynb"))


@unittest.skipIf(nbformat is None, "nbformat not installed (run with: uv run --with nbformat ...)")
class TestNotebooks(unittest.TestCase):
    def test_nine_notebooks(self):
        self.assertEqual([p.name[:2] for p in NOTEBOOKS], [f"{i:02d}" for i in range(1, 10)])

    def test_training_notebook_is_disabled_by_default(self):
        nb = nbformat.read(ROOT / "notebooks/09_reproduce_training.ipynb", as_version=4)
        src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        self.assertIn("RUN_TRAINING = False", src)
        self.assertIn('AUTHORISATION = ""', src)
        out = "\n".join(str(o.get("text", "")) for c in nb.cells if c.cell_type == "code" for o in c.outputs)
        self.assertIn("Training NOT started", out)

    def test_each_notebook(self):
        for p in NOTEBOOKS:
            with self.subTest(p.name):
                nb = nbformat.read(p, as_version=4)
                nbformat.validate(nb)
                self.assertLess(p.stat().st_size, 1_000_000)
                code = [c for c in nb.cells if c.cell_type == "code"]
                self.assertTrue(all(c.execution_count is not None for c in code), "not fully executed")
                src = "\n".join(c.source for c in code)
                for pat in build_notebooks.FORBIDDEN_SOURCE:
                    self.assertIsNone(re.search(pat, src), pat)
                for c in code:
                    for o in c.outputs:
                        self.assertNotEqual(o.get("output_type"), "error")
                        text = str(o.get("text", "")) + str(o.get("data", {}).get("text/plain", ""))
                        for pat in build_notebooks.FORBIDDEN_OUTPUT:
                            self.assertIsNone(re.search(pat, text), pat)
                md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
                self.assertIn("Provisional", md)


if __name__ == "__main__":
    unittest.main()
