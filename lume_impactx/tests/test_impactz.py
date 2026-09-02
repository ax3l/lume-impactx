"""Three-way comparison: Bmad, Impact-Z and ImpactX on one lattice and one bunch.

Every other physics test here compares this translation against Bmad. That is the right
reference, but it is a single one -- a shared misreading of a Bmad convention would look
like agreement. Impact-Z is an independent third opinion: a different code, wrapped by a
different package (lume-impact), with its own Tao translator that shares no
implementation with this one.

Skipped unless the whole toolchain is present. It needs `pytao`, `lume-impact` and an
`ImpactZexe` binary, all of which are on conda-forge::

    micromamba create -n lume-impactx-z -c conda-forge python=3.12 impactx bmad pytao \\
        impact-z distgen
    pip install lume-impact

Slower than the rest of the suite, because Impact-Z is an external executable driven
through files rather than an in-process library.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile

import numpy as np
import pytest

pytao = pytest.importorskip("pytao", reason="Bmad/pytao is not installed")
pytest.importorskip("impact", reason="lume-impact is not installed")

from impact import ImpactZ  # noqa: E402
from impact.z import ImpactZInput  # noqa: E402

try:
    from beamphysics import ParticleGroup
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup

pytestmark = pytest.mark.skipif(
    shutil.which("ImpactZexe") is None and shutil.which("ImpactZexe-mpi") is None,
    reason="no ImpactZexe on PATH",
)

TOTAL_ENERGY_EV = 100e6
ELECTRON_MASS_EV = 0.51099895069e6
N_PARTICLES = 512


def _bunch(seed: int = 7) -> ParticleGroup:
    rng = np.random.default_rng(seed)
    p0c = math.sqrt(TOTAL_ENERGY_EV**2 - ELECTRON_MASS_EV**2)
    n = N_PARTICLES
    return ParticleGroup(
        data={
            "x": rng.normal(0, 2e-4, n),
            "y": rng.normal(0, 2e-4, n),
            "z": np.zeros(n),
            "px": rng.normal(0, 1e-5, n) * p0c,
            "py": rng.normal(0, 1e-5, n) * p0c,
            "pz": p0c * (1 + rng.normal(0, 1e-4, n)),
            "t": rng.normal(0, 1e-13, n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1e-12 / n),
            "species": "electron",
        }
    )


def _worst(a: ParticleGroup, b: ParticleGroup) -> float:
    """Worst relative coordinate difference between two bunches.

    Sorted per coordinate: the three codes do not preserve particle ordering, and only
    the distribution is being compared.
    """
    return max(
        np.abs(np.sort(a[key]) - np.sort(b[key])).max()
        / max(np.abs(a[key]).max(), 1e-30)
        for key in ("x", "y", "px", "py")
    )


@pytest.fixture
def three_way(tmp_path, monkeypatch):
    """Track one bunch through Bmad, Impact-Z and ImpactX; return the three bunches."""
    from lume_impactx import ImpactXSimulator

    def run(body: str, line: str):
        bunch = _bunch()
        directory = tempfile.mkdtemp(dir=tmp_path)
        with open(os.path.join(directory, "lat.bmad"), "w") as handle:
            handle.write(
                "parameter[geometry] = open\nparameter[particle] = electron\n"
                f"parameter[e_tot] = {TOTAL_ENERGY_EV}\n"
                "beginning[beta_a] = 10\nbeginning[beta_b] = 10\n"
                f"d1: drift, l = 0.4\n{body}\nlat: line = ({line})\nuse, lat\n"
            )
        monkeypatch.chdir(directory)
        bunch.write("beam.h5")

        tao = pytao.Tao(lattice_file="lat.bmad", noplot=True)
        tao.cmds(
            [
                "set beam_init position_file = beam.h5",
                f"set beam_init n_particle = {N_PARTICLES}",
                f"set beam_init bunch_charge = {bunch.charge}",
                "set beam_init saved_at = *",
                "set global track_type = beam",
            ]
        )

        impactz_input = ImpactZInput.from_tao(tao)
        impactz_input.space_charge_off()
        impactz = ImpactZ(impactz_input).run().particles["final_particles"]

        impactx = ImpactXSimulator.from_tao(tao, nslice=16).final_particles
        return tao.particles("END"), impactz, impactx

    return run


@pytest.mark.parametrize(
    ("label", "body", "line", "impactx_tol", "impactz_tol"),
    [
        ("drift", "", "d1", 1e-13, 1e-13),
        ("quadrupole", "q: quadrupole, l = 0.3, k1 = 2.0", "d1, q, d1", 1e-12, 1e-12),
        # Impact-Z is the looser side here by ~2800x, not this translation. Its
        # tolerance is set to what it actually achieves, so a regression on either
        # side is still caught.
        ("sbend", "b: sbend, l = 0.5, angle = 0.12", "d1, b, d1", 1e-9, 1e-6),
        (
            "four FODO cells",
            "qf: quadrupole, l = 0.3, k1 = 2.0\nqd: quadrupole, l = 0.3, k1 = -2.0",
            "4*(qf, d1, qd, d1)",
            1e-11,
            1e-11,
        ),
    ],
)
def test_three_codes_agree(three_way, label, body, line, impactx_tol, impactz_tol):
    bmad, impactz, impactx = three_way(body, line)

    assert impactz.n_particle == bmad.n_particle == impactx.n_particle
    assert 0.0 < _worst(bmad, impactx) < impactx_tol, f"{label}: Bmad vs ImpactX"
    assert 0.0 < _worst(bmad, impactz) < impactz_tol, f"{label}: Bmad vs Impact-Z"


def test_impactx_tracks_a_bend_closer_to_bmad_than_impactz(three_way):
    """An independent check that the bend work was worth doing.

    Both packages translate the same Tao lattice with no shared code. This one models
    the body with ImpactX's exact sector bend and the pole faces with a nonlinear
    DipEdge; Impact-Z's translator reaches 1.2e-7 on the same case where this reaches
    4.3e-11. Asserted as an ordering rather than a fixed ratio, which would be brittle.
    """
    bmad, impactz, impactx = three_way("b: sbend, l = 0.5, angle = 0.12", "d1, b, d1")
    assert _worst(bmad, impactx) < _worst(bmad, impactz) / 100.0
