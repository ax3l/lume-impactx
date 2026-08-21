#!/usr/bin/env python3
"""Check that many ImpactX simulations can be built, tracked and finalized in one process.

This is the assumption :class:`lume_impactx.simulator.ImpactXSimulator` rests on: it
rebuilds the simulation on every ``track()`` rather than trying to rewind a container
that tracking has already consumed.

Run it after changing ImpactX versions or switching between serial and MPI builds::

    python scripts/spike_amrex_cycles.py 200
"""

from __future__ import annotations

import sys
import time

import numpy as np

# Import lume_impactx *first*: it hands MPI ownership to mpi4py, without which the
# second simulation in this process aborts. See lume_impactx/_mpi.py.
import lume_impactx  # noqa: F401
import impactx
from impactx import ImpactX, distribution, elements

NPART = 1000


def fodo(k1: float, k2: float, ns: int = 5) -> list:
    return [
        elements.Drift(name="drift1", ds=0.25, nslice=ns),
        elements.Quad(name="quad1", ds=1.0, k=k1, nslice=ns),
        elements.Drift(name="drift2", ds=0.5, nslice=ns),
        elements.Quad(name="quad2", ds=1.0, k=k2, nslice=ns),
        elements.Drift(name="drift3", ds=0.25, nslice=ns),
    ]


def run_once(k1: float, k2: float) -> dict:
    sim = ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.init_grids()

    sim.beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    distr = distribution.Waterbag(
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
    sim.add_particles(1.0e-9, distr, NPART)
    sim.lattice.extend(fodo(k1, k2))
    sim.track_particles()

    moments = dict(sim.beam.beam_moments())
    n = sim.beam.total_number_of_particles()
    s = sim.beam.ref.s
    sim.finalize()
    return {"sigma_x": moments["sigma_x"], "n": n, "s": s}


def main() -> int:
    n_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print(
        f"impactx {impactx.__version__} "
        f"have_mpi={impactx.Config.have_mpi} precision={impactx.Config.precision}",
        flush=True,
    )
    print(f"running {n_cycles} build/track/finalize cycles, npart={NPART}", flush=True)

    t0 = time.monotonic()
    results = [run_once(1.0, -1.0) for _ in range(n_cycles)]
    dt = time.monotonic() - t0

    sigma_x = np.array([r["sigma_x"] for r in results])
    spread = float(np.ptp(sigma_x) / np.mean(sigma_x))
    alt = run_once(1.2, -1.0)

    print(
        f"  {n_cycles} cycles in {dt:.1f}s ({dt / n_cycles * 1e3:.0f} ms/cycle)",
        flush=True,
    )
    print(f"  sigma_x relative spread across cycles: {spread:.3e}", flush=True)
    print(f"  k=1.0 -> {sigma_x[0]:.6e} | k=1.2 -> {alt['sigma_x']:.6e}", flush=True)

    deterministic = spread < 1e-12
    responsive = not np.isclose(alt["sigma_x"], sigma_x[0], rtol=1e-6)
    ok = deterministic and responsive
    if not deterministic:
        print("  FAIL: repeated identical runs disagree", flush=True)
    if not responsive:
        print("  FAIL: changing quad strength did not change the beam", flush=True)
    print("RESULT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
