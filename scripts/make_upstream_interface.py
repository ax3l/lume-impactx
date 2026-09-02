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

`ImpactX <https://impactx.readthedocs.io>`_ is an s-based beam dynamics code, the
successor of IMPACT-Z. Its particles are held at a common ``s`` with a spread in
arrival time, which is z-coordinates on this side -- all ``z`` equal, ``t`` varying --
so the conversion is a direct algebraic map, like the Bmad interface and unlike the
time-based ones.

Coordinates and frames
----------------------
ImpactX describes each particle at fixed ``s`` by ``(x, y, t, px, py, pt)``:

- ``x``, ``y`` [m] are the transverse displacement from the reference particle, in the
  local (curvilinear) frame that follows the reference orbit.
- ``t`` [m] is ``c`` times the difference between the particle's and the reference
  particle's arrival time, i.e. a length, not a time.
- ``px``, ``py``, ``pt`` are dimensionless, normalized by the magnitude of the
  reference momentum: ``px = Delta(beta_x gamma) / (beta_0 gamma_0)`` and
  ``pt = -Delta(gamma) / (beta_0 gamma_0)``.

`ParticleGroup` is a lab-frame container, so the mapping has to choose a frame:

- The transverse coordinates stay in the **local frame**: ``x`` and ``y`` are the
  displacement from the reference particle and ``z`` is zero, the reference plane.
  Adding ``x_ref``/``z_ref`` would be wrong wherever the reference orbit is bent,
  because local ``x`` is then not lab ``x``. Use `ImpactXRefPart` (``x``, ``y``, ``z``,
  ``s``, ``px``, ``py``, ``pz``) if you need to place the bunch in the lab.
- The time is **absolute**: ``t = t_ref + position_t / c``. That one is unambiguous, it
  is what openPMD's ``position/t + positionOffset/t`` means in ImpactX output, and it
  keeps quantities like `ParticleGroup.average_current` meaningful.

Originally developed in `lume-impactx <https://github.com/lume-science/lume-impactx>`_.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..particles import ParticleGroup
from ..species import charge_of, e_charge, mass_of
from ..status import ParticleStatus
from ..units import c_light

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
        "UnrepresentableParticleData",
        "beam_monitor_iterations",
        "impactx_to_particlegroup_data",
        "particle_id_from_idcpu",
        "particlegroup_to_impactx",
        "pmd_species_of",
        "read_beam_monitor",
        "read_beam_monitor_data",
        "refpart_from_openpmd",
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
