#!/usr/bin/env python3
"""3D space charge: a bunch expanding under its own field.

The counterpart of Bmad's ``tao_examples/space_charge`` and of ImpactX's own
``expanding_beam`` benchmark. A cold, uniformly-filled ellipsoidal bunch coasts down a
drift; with space charge on it expands to twice its initial size, which is the
analytically known result and what the assertion below checks.

Run::

    python examples/space_charge_expansion.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from impactx import distribution, elements  # noqa: E402

from lume_impactx import ImpactXSimulator, LUMEImpactXModel  # noqa: E402

#: A cold 250 MeV electron bunch: a 1 mm ball in its own rest frame.
BEAM = distribution.Kurth6D(
    lambdaX=4.472135955e-4,
    lambdaY=4.472135955e-4,
    lambdaT=9.12241869e-7,
    lambdaPx=0.0,
    lambdaPy=0.0,
    lambdaPt=0.0,
)
REFERENCE = {"species": "electron", "kin_energy_MeV": 250.0}
BUNCH_CHARGE_C = 1.0e-9

#: Mesh and solver. particle_shape is required whenever a collective effect is on --
#: ImpactX refuses init_grids() without it.
SPACE_CHARGE_SETTINGS = {
    "space_charge": "3D",
    "poisson_solver": "multigrid",
    "particle_shape": 2,
    "n_cell": [32, 32, 40],
    "prob_relative": [3.0],
    "dynamic_size": True,
    "slice_step_diagnostics": True,
}


def build(space_charge: bool) -> ImpactXSimulator:
    settings = dict(SPACE_CHARGE_SETTINGS)
    if not space_charge:
        settings["space_charge"] = False
    return ImpactXSimulator(
        lattice=[elements.Drift(name="drift", ds=6.0, nslice=40)],
        ref=REFERENCE,
        distribution=BEAM,
        npart=10_000,
        bunch_charge_C=BUNCH_CHARGE_C,
        settings=settings,
    )


def main() -> int:
    off, on = build(space_charge=False), build(space_charge=True)

    model = LUMEImpactXModel.from_simulator(on)
    print(f"{len(model.supported_variables)} variables; space-charge knobs exposed:")
    for name in sorted(n for n in model.supported_variables if n.startswith("sim:")):
        print(f"    {name:32s} = {model.get(name)!r}")

    print(f"\n{'quantity':16s} {'no space charge':>18s} {'3D space charge':>18s}")
    for key in ("sigma_x", "sigma_y", "sigma_t", "emittance_x"):
        print(
            f"{key:16s} {off.results['moments'][key]:18.6e} "
            f"{on.results['moments'][key]:18.6e}"
        )

    growth = on.results["moments"]["sigma_x"] / off.results["moments"]["sigma_x"]
    print(f"\ntransverse growth from space charge: {growth:.3f}x")
    assert 1.9 < growth < 2.1, f"expected the known 2x expansion, got {growth:.3f}"

    figure = on.plot(y=("sigma_x", "sigma_y"), y2=("emittance_x",), include_labels=True)
    figure.savefig("space_charge_expansion.png", dpi=110, bbox_inches="tight")
    print("wrote space_charge_expansion.png")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
