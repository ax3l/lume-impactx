#!/usr/bin/env python3
"""Resistive-wall wakefield in a long beam pipe.

ImpactX has no wakefield lattice element -- it ships the numerical primitives
(``wakeconvolution.deposit_charge`` / ``derivative_charge`` / ``w_l_csr`` /
``convolve_fft``) and leaves the assembly to the user. :mod:`lume_impactx.wakes`
assembles the resistive-wall case into a thin ``Programmable`` element you can drop into
a lattice.

The physics: a bunch travelling down a resistive pipe drives image currents that lag
behind it, so trailing particles are decelerated. The head loses least and the tail
most, which costs mean energy and imprints a correlated energy spread along the bunch.

Run::

    python examples/resistive_wall_wake.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from impactx import distribution, elements  # noqa: E402

from lume_impactx import ImpactXSimulator  # noqa: E402
from lume_impactx.wakes import (  # noqa: E402
    ResistiveWallWake,
    resistive_wall_s0,
    resistive_wall_wake,
)

REFERENCE = {"species": "electron", "kin_energy_MeV": 100.0}
BUNCH_CHARGE_C = 1.0e-9
PIPE_LENGTH_M = 20.0
PIPE_RADIUS_M = 0.005
COPPER_S_PER_M = 5.96e7

BEAM = distribution.Gaussian(
    lambdaX=2.0e-4,
    lambdaY=2.0e-4,
    lambdaT=5.0e-5,
    lambdaPx=1.0e-5,
    lambdaPy=1.0e-5,
    lambdaPt=1.0e-5,
)


def build(wake: bool) -> ImpactXSimulator:
    lattice = [elements.Drift(name="pipe", ds=PIPE_LENGTH_M, nslice=40)]
    if wake:
        lattice.append(
            ResistiveWallWake(
                "rw",
                length=PIPE_LENGTH_M,
                pipe_radius=PIPE_RADIUS_M,
                conductivity=COPPER_S_PER_M,
                num_bins=128,
            )
        )
    return ImpactXSimulator(
        lattice=lattice,
        ref=REFERENCE,
        distribution=BEAM,
        npart=20_000,
        bunch_charge_C=BUNCH_CHARGE_C,
        settings={"particle_shape": 2},
    )


def main() -> int:
    s0 = resistive_wall_s0(PIPE_RADIUS_M, COPPER_S_PER_M)
    print(f"pipe: {PIPE_LENGTH_M} m of copper, radius {PIPE_RADIUS_M * 1e3:.1f} mm")
    print(f"characteristic length s0 = {s0 * 1e6:.2f} um")
    print(f"bunch length sigma_t     = {5.0e-5 * 1e6:.0f} um  ({5.0e-5 / s0:.1f} s0)")
    print(
        f"W_L(0) = {float(resistive_wall_wake(0.0, PIPE_RADIUS_M, COPPER_S_PER_M)):.4e} V/C/m"
    )

    off, on = build(wake=False), build(wake=True)
    plain, kicked = off.final_particles, on.final_particles

    print(f"\n{'quantity':22s} {'no wake':>16s} {'with wake':>16s}")
    for key in ("mean_energy", "sigma_energy", "sigma_x"):
        print(f"{key:22s} {plain[key]:16.6e} {kicked[key]:16.6e}")

    lost_eV = plain["mean_energy"] - kicked["mean_energy"]
    print(f"\nmean energy lost:  {lost_eV:.4e} eV  ({lost_eV / 1e6:.4f} MeV)")
    assert lost_eV > 0.0, "a resistive wall must decelerate the bunch"
    assert kicked["sigma_x"] == plain["sigma_x"], "the wake is purely longitudinal"

    # The head-to-tail energy profile is the signature of a wake: the tail sits deepest.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    lags = np.linspace(0.0, 8.0 * s0, 400)
    axes[0].plot(lags * 1e6, resistive_wall_wake(lags, PIPE_RADIUS_M, COPPER_S_PER_M))
    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].set_xlabel("distance behind source [um]")
    axes[0].set_ylabel("$W_L$ [V/C/m]")
    axes[0].set_title("resistive-wall wake")

    for label, bunch in (("no wake", plain), ("with wake", kicked)):
        axes[1].scatter(
            bunch.t * 1e12, bunch.energy / 1e6, s=1, alpha=0.25, label=label
        )
    axes[1].set_xlabel("arrival time [ps]")
    axes[1].set_ylabel("energy [MeV]")
    axes[1].set_title("energy along the bunch")
    axes[1].legend(markerscale=8, fontsize=8)

    figure.tight_layout()
    figure.savefig("resistive_wall_wake.png", dpi=110, bbox_inches="tight")
    print("wrote resistive_wall_wake.png")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
