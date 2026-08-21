#!/usr/bin/env python3
"""Generate ``upstream/impactx.py`` from ``lume_impactx/utils.py``.

The converters are written so they can become
``beamphysics/interfaces/impactx.py`` in openPMD-beamphysics, alongside the existing
``astra``/``bmad``/``elegant``/``impact`` interfaces. Rather than maintain a second copy
by hand, this script slices the ImpactX-free core out of ``utils.py`` and rewrites the
imports to openPMD-beamphysics' relative style.

``lume_impactx/tests/test_upstream.py`` regenerates and compares, so the committed file
cannot drift from the implementation this package actually uses.

Usage::

    python scripts/make_upstream_interface.py          # write upstream/impactx.py
    python scripts/make_upstream_interface.py --check  # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "lume_impactx" / "utils.py"
TARGET = ROOT / "upstream" / "impactx.py"

HEADER = '''"""ImpactX beam data <-> ParticleGroup.

`ImpactX <https://impactx.readthedocs.io>`_ is an s-based beam dynamics code. Its
particles are held at a common ``s`` with a spread in arrival time, which is
z-coordinates on this side -- all ``z`` equal, ``t`` varying -- so the conversion is a
direct algebraic map, like the Bmad interface and unlike the time-based ones.

Generated from lume-impactx; see https://github.com/lume-science/lume-impactx.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..particles import ParticleGroup
from ..species import charge_of, e_charge, mass_of

C_LIGHT = 299792458.0
'''


def extract(text: str, begin: str, end: str) -> str:
    """Return the text between two marker comments, exclusive."""
    try:
        start = text.index(begin) + len(begin)
        stop = text.index(end)
    except ValueError as exc:  # pragma: no cover - marker drift
        raise RuntimeError(f"Marker not found in {SOURCE}: {exc}") from exc
    return text[start:stop].strip("\n")


def render() -> str:
    text = SOURCE.read_text()
    core = extract(text, "# BEGIN UPSTREAM CORE", "# END UPSTREAM CORE")
    reader = extract(text, "# BEGIN UPSTREAM READER", "# END UPSTREAM READER")
    body = f"{core}\n\n\n{reader}"

    exports = [
        "ImpactXRefPart",
        "particlegroup_to_impactx",
        "impactx_to_particlegroup_data",
        "pmd_species_of",
        "refpart_from_openpmd",
        "read_beam_monitor",
    ]
    all_block = "__all__ = [\n" + "".join(f'    "{n}",\n' for n in exports) + "]\n"
    return f"{HEADER}\n{all_block}\n\n{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="only verify the file is up to date"
    )
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = TARGET.read_text() if TARGET.exists() else ""
        if current != rendered:
            print(
                f"{TARGET.relative_to(ROOT)} is out of date; "
                "run scripts/make_upstream_interface.py",
                file=sys.stderr,
            )
            return 1
        print(f"{TARGET.relative_to(ROOT)} is up to date")
        return 0

    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(rendered)
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(rendered.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
