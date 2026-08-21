"""Shared fixtures.

The ImpactX session fixtures live here rather than in individual tests because of a
process-lifetime hazard: on an MPI-enabled ImpactX build, ``ImpactX.finalize()`` calls
``MPI_Finalize()`` unless mpi4py owns MPI. ``lume_impactx`` arranges that on import (see
``lume_impactx._mpi``), and importing the package here makes sure it happens before any
test constructs an ``ImpactX``.
"""

from __future__ import annotations

import numpy as np
import pytest

import lume_impactx  # noqa: F401  -- MPI bootstrap, must precede any ImpactX()

impactx = pytest.importorskip("impactx", reason="ImpactX is not installed")

from beamphysics import ParticleGroup  # noqa: E402

from lume_impactx.utils import ImpactXRefPart  # noqa: E402

#: 2 GeV electrons, matching ImpactX's own examples/fodo.
KIN_ENERGY_MEV = 2.0e3
BUNCH_CHARGE_C = 1.0e-9
ELECTRON_MASS_MEV = 0.5109989506917532


@pytest.fixture(scope="session")
def electron_ref() -> ImpactXRefPart:
    """A 2 GeV on-axis electron reference particle, with no ImpactX session needed."""
    gamma = 1.0 + KIN_ENERGY_MEV / ELECTRON_MASS_MEV
    beta_gamma = np.sqrt(gamma**2 - 1.0)
    return ImpactXRefPart(
        x=0.0,
        y=0.0,
        z=0.0,
        t=0.0,
        px=0.0,
        py=0.0,
        pz=beta_gamma,
        pt=-gamma,
        mass_MeV=ELECTRON_MASS_MEV,
        charge_qe=-1.0,
    )


@pytest.fixture(scope="session")
def bunch(electron_ref) -> ParticleGroup:
    """A synthetic 2 GeV electron bunch in z-coordinates, matched to ``electron_ref``."""
    rng = np.random.default_rng(20260820)
    n = 2000
    mass_eV = electron_ref.mass_eV
    p_ref = electron_ref.beta_gamma * mass_eV

    px = rng.normal(0.0, 1.0e-5, n) * p_ref
    py = rng.normal(0.0, 1.0e-5, n) * p_ref
    pz = p_ref * (1.0 + rng.normal(0.0, 2.0e-3, n))
    return ParticleGroup(
        data={
            "x": rng.normal(0.0, 4.0e-5, n),
            "y": rng.normal(0.0, 4.0e-5, n),
            "z": np.zeros(n),  # z-coordinates: one plane, spread in t
            "px": px,
            "py": py,
            "pz": pz,
            "t": rng.normal(0.0, 1.0e-3, n) / 299792458.0,
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, BUNCH_CHARGE_C / n),
            "species": "electron",
        }
    )


@pytest.fixture
def impactx_session():
    """A fresh, finalized ImpactX session.

    Creating and finalizing one session per test is exactly what
    ``ImpactXSimulator.track()`` does, so if this fixture ever breaks on some build,
    the whole rebuild-per-track design is in trouble and we want to know.
    """
    sim = impactx.ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.init_grids()
    yield sim
    sim.finalize()


@pytest.fixture
def fodo_lattice():
    """The FODO cell from ImpactX's own examples/fodo, so numbers stay comparable."""
    from impactx import elements

    ns = 5
    return [
        elements.Drift(name="drift1", ds=0.25, nslice=ns),
        elements.Quad(name="quad1", ds=1.0, k=1.0, nslice=ns),
        elements.Drift(name="drift2", ds=0.5, nslice=ns),
        elements.Quad(name="quad2", ds=1.0, k=-1.0, nslice=ns),
        elements.Drift(name="drift3", ds=0.25, nslice=ns),
    ]


@pytest.fixture
def waterbag():
    """The matched Waterbag distribution from examples/fodo."""
    from impactx import distribution

    return distribution.Waterbag(
        lambdaX=3.9984884770e-5,
        lambdaY=3.9984884770e-5,
        lambdaT=1.0e-3,
        lambdaPx=2.6623538760e-5,
        lambdaPy=2.6623538760e-5,
        lambdaPt=2.0e-3,
        muxpx=-0.846574929020762,
        muypy=0.846574929020762,
        mutpt=0.0,
    )


@pytest.fixture
def fodo_simulator(fodo_lattice, waterbag):
    """A tracked FODO simulator: 2 GeV electrons, 1000 macroparticles, no space charge."""
    from lume_impactx.simulator import ImpactXSimulator

    return ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": KIN_ENERGY_MEV},
        distribution=waterbag,
        npart=1000,
        bunch_charge_C=BUNCH_CHARGE_C,
    )
