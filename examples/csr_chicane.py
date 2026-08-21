#!/usr/bin/env python3
"""Coherent synchrotron radiation through a bunch compressor.

The counterpart of Bmad's ``tao_examples/csr_beam_tracking``, which tracks a bunch
through bends with ``csr_method = 1_dim``. ImpactX exposes CSR as a pair of
settings -- ``csr`` and ``csr_bins`` -- so the model-level knobs are ``sim:csr`` and
``sim:csr_bins``.

CSR acts inside bends: the radiation emitted on the curved orbit catches up with the
bunch and redistributes its energy, costing mean energy and adding correlated spread.
This runs the same chicane with CSR off and on and shows the difference, then scans
``sim:csr_bins`` to show the answer is resolution-dependent -- worth knowing before
trusting a single number. It also checks the condition under which the 1D CSR model is
valid at all.

Run::

    python examples/csr_chicane.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from impactx import distribution, elements  # noqa: E402

import numpy as np  # noqa: E402

from lume_impactx import ImpactXSimulator, LUMEImpactXModel  # noqa: E402
from lume_impactx.wakes import csr_validity_margin  # noqa: E402

REFERENCE = {"species": "electron", "kin_energy_MeV": 250.0}
BUNCH_CHARGE_C = 1.0e-9

#: A 100 um bunch at 250 MeV: a strong but physical CSR regime, of the kind a bunch
#: compressor actually operates in. Shortening the bunch or dropping the energy pushes
#: this into a regime where CSR removes several percent of the beam energy and the
#: result stops converging in ``csr_bins``.
BEAM = distribution.Waterbag(
    lambdaX=1.0e-4,
    lambdaY=1.0e-4,
    lambdaT=1.0e-4,
    lambdaPx=1.0e-5,
    lambdaPy=1.0e-5,
    lambdaPt=1.0e-4,
)


def chicane(bend_angle_deg: float = 5.0, nslice: int = 20) -> list:
    """A four-bend chicane: the classic CSR-generating geometry."""
    drift = 0.5
    return [
        elements.ExactSbend(name="b1", ds=0.5, phi=bend_angle_deg, nslice=nslice),
        elements.Drift(name="d1", ds=drift, nslice=nslice),
        elements.ExactSbend(name="b2", ds=0.5, phi=-bend_angle_deg, nslice=nslice),
        elements.Drift(name="d2", ds=drift, nslice=nslice),
        elements.ExactSbend(name="b3", ds=0.5, phi=-bend_angle_deg, nslice=nslice),
        elements.Drift(name="d3", ds=drift, nslice=nslice),
        elements.ExactSbend(name="b4", ds=0.5, phi=bend_angle_deg, nslice=nslice),
    ]


def build(csr: bool, csr_bins: int = 150) -> ImpactXSimulator:
    return ImpactXSimulator(
        lattice=chicane(),
        ref=REFERENCE,
        distribution=BEAM,
        npart=10_000,
        bunch_charge_C=BUNCH_CHARGE_C,
        settings={
            "csr": csr,
            "csr_bins": csr_bins,
            "particle_shape": 2,
            "slice_step_diagnostics": True,
        },
    )


def main() -> int:
    # The 1D CSR model projects the bunch onto a line, which only holds while the bunch
    # is transversely thin compared with the CSR formation width. Worth checking before
    # reading anything into the numbers -- see Sagan & Mayes, IPAC2017 THPAB076.
    bend_radius = 0.5 / np.deg2rad(5.0)
    margin = csr_validity_margin(
        bend_radius=bend_radius, sigma_z=1.0e-4, sigma_transverse=1.0e-4
    )
    print(f"bend radius R = {bend_radius:.2f} m")
    print(f"1D CSR validity margin: {margin:.1f}x  (want >> 1)")
    assert margin > 10.0, "the 1D CSR model is not justified for this bunch"

    off, on = build(csr=False), build(csr=True)

    print(f"{'quantity':22s} {'CSR off':>16s} {'CSR on':>16s}")
    for key in ("mean_pt", "sigma_pt", "sigma_t", "emittance_x"):
        print(
            f"{key:22s} {off.results['moments'][key]:16.6e} "
            f"{on.results['moments'][key]:16.6e}"
        )

    lost_eV = off.final_particles["mean_energy"] - on.final_particles["mean_energy"]
    print(f"\nmean energy lost to CSR: {lost_eV:.4e} eV")
    assert lost_eV > 0.0, "CSR must remove energy from the bunch"

    spread_growth = (
        on.results["moments"]["sigma_pt"] / off.results["moments"]["sigma_pt"]
    )
    print(f"energy-spread growth:    {spread_growth:.2f}x")
    assert spread_growth > 1.0

    # The CSR model bins the bunch longitudinally, so the answer depends on how many
    # bins you give it. Always check this before quoting a number.
    print("\nresolution dependence (sim:csr_bins):")
    model = LUMEImpactXModel.from_simulator(on)
    for bins in (50, 100, 150, 300):
        model.set({"sim:csr_bins": bins})
        print(
            f"    csr_bins={bins:4d}  mean_pt={model.get('moment_final:mean_pt'):+.6e}"
            f"  sigma_pt={model.get('moment_final:sigma_pt'):.6e}"
        )

    figure = on.plot(y=("sigma_t",), y2=("sigma_pt",), include_labels=True)
    figure.savefig("csr_chicane.png", dpi=110, bbox_inches="tight")
    print("\nwrote csr_chicane.png")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
