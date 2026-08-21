"""The example scripts must actually run.

lume-impact does the same for the Impact-Z examples. Each script asserts its own
physics, so running it to completion is a real check, not just an import test.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

EXAMPLES = sorted(
    (pathlib.Path(__file__).resolve().parents[2] / "examples").glob("*.py")
)


def test_examples_are_discovered():
    names = {path.name for path in EXAMPLES}
    assert names == {
        "space_charge_expansion.py",
        "csr_chicane.py",
        "resistive_wall_wake.py",
    }


@pytest.mark.slow
@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(script, tmp_path, monkeypatch):
    """Run each example end to end, in a temp dir so its PNG lands there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script)])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(script), run_name="__main__")
    assert excinfo.value.code == 0, f"{script.name} exited non-zero"

    written = list(tmp_path.glob("*.png"))
    assert written, f"{script.name} produced no figure"
