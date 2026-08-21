"""Short-range wakefields.

ImpactX ships wake primitives but no wake element, so this module assembles one. The
tests lean on properties that hold exactly -- an analytic value at s=0, and linearity in
charge and length -- rather than on regression numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from lume_impactx.wakes import (
    C_LIGHT,
    Z0,
    ResistiveWallWake,
    resistive_wall_s0,
    resistive_wall_wake,
)

COPPER = 5.96e7


def test_wake_at_zero_matches_the_analytic_value():
    """At s=0 the Bane-Sands integral is analytic and the wake collapses to Z0 c/(4 pi b^2).

    The integral term is int_0^inf x^2/(x^6+8) dx = pi/(12 sqrt 2), so the bracket is
    1/3 - 1/12 = 1/4. This pins the numerical quadrature to a closed form.
    """
    for radius in (0.002, 0.005, 0.02):
        analytic = Z0 * C_LIGHT / (4.0 * np.pi * radius**2)
        assert float(resistive_wall_wake(0.0, radius, COPPER)) == pytest.approx(
            analytic, rel=1e-10
        )


def test_wake_is_causal():
    assert float(resistive_wall_wake(-1e-6, 0.005, COPPER)) == 0.0
    assert float(resistive_wall_wake(-1.0, 0.005, COPPER)) == 0.0
    values = resistive_wall_wake(np.array([-2e-5, -1e-9, 0.0, 1e-9]), 0.005, COPPER)
    np.testing.assert_array_equal(values[:2], [0.0, 0.0])
    assert values[2] > 0.0


def test_wake_decays_behind_the_source():
    s0 = resistive_wall_s0(0.005, COPPER)
    s = np.array([0.0, 0.5, 1.0, 2.0]) * s0
    wake = resistive_wall_wake(s, 0.005, COPPER)
    assert np.all(np.diff(wake) < 0.0), "the wake must fall off behind the source"


def test_s0_scales_as_expected():
    assert resistive_wall_s0(0.01, COPPER) == pytest.approx(
        (2 * 0.01**2 / (Z0 * COPPER)) ** (1 / 3), rel=1e-12
    )
    # s0 grows as b^(2/3)
    assert resistive_wall_s0(0.02, COPPER) / resistive_wall_s0(
        0.01, COPPER
    ) == pytest.approx(2 ** (2 / 3), rel=1e-12)


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match="pipe_radius"):
        resistive_wall_s0(0.0, COPPER)
    with pytest.raises(ValueError, match="conductivity"):
        resistive_wall_s0(0.01, -1.0)


# -- applied to a real bunch ------------------------------------------------------------


def _energy_loss(charge=1e-9, length=10.0, radius=0.005, npart=2000):
    """Mean energy removed by the wake, against an otherwise identical run."""
    from impactx import distribution, elements

    from lume_impactx.simulator import ImpactXSimulator

    beam = distribution.Gaussian(
        lambdaX=2e-4,
        lambdaY=2e-4,
        lambdaT=5e-5,
        lambdaPx=1e-5,
        lambdaPy=1e-5,
        lambdaPt=1e-5,
    )
    reference = {"species": "electron", "kin_energy_MeV": 100.0}

    def run(lattice):
        return ImpactXSimulator(
            lattice=lattice,
            ref=reference,
            distribution=beam,
            npart=npart,
            bunch_charge_C=charge,
            settings={"particle_shape": 2},
        ).final_particles["mean_energy"]

    pipe = elements.Drift(name="pipe", ds=1.0, nslice=4)
    without = run([pipe])
    with_wake = run(
        [pipe, ResistiveWallWake("rw", length=length, pipe_radius=radius, num_bins=96)]
    )
    return without - with_wake


@pytest.mark.slow
def test_wake_removes_energy_and_adds_spread():
    from impactx import distribution, elements

    from lume_impactx.simulator import ImpactXSimulator

    beam = distribution.Gaussian(
        lambdaX=2e-4,
        lambdaY=2e-4,
        lambdaT=5e-5,
        lambdaPx=1e-5,
        lambdaPy=1e-5,
        lambdaPt=1e-5,
    )
    reference = {"species": "electron", "kin_energy_MeV": 100.0}
    pipe = elements.Drift(name="pipe", ds=1.0, nslice=4)

    plain = ImpactXSimulator(
        lattice=[pipe],
        ref=reference,
        distribution=beam,
        npart=2000,
        bunch_charge_C=1e-9,
        settings={"particle_shape": 2},
    ).final_particles
    kicked = ImpactXSimulator(
        lattice=[pipe, ResistiveWallWake("rw", length=10.0, pipe_radius=0.005)],
        ref=reference,
        distribution=beam,
        npart=2000,
        bunch_charge_C=1e-9,
        settings={"particle_shape": 2},
    ).final_particles

    assert kicked["mean_energy"] < plain["mean_energy"], "the bunch must lose energy"
    assert kicked["sigma_energy"] > 10.0 * plain["sigma_energy"]
    # transverse dynamics are untouched by a purely longitudinal wake
    assert kicked["sigma_x"] == pytest.approx(plain["sigma_x"], rel=1e-12)


@pytest.mark.slow
def test_energy_loss_is_linear_in_charge():
    reference = _energy_loss(charge=1e-9)
    for factor in (0.5, 2.0, 4.0):
        assert _energy_loss(charge=factor * 1e-9) == pytest.approx(
            factor * reference, rel=1e-6
        )


@pytest.mark.slow
def test_energy_loss_is_linear_in_length():
    reference = _energy_loss(length=10.0)
    for length in (5.0, 20.0):
        assert _energy_loss(length=length) == pytest.approx(
            length / 10.0 * reference, rel=1e-6
        )


@pytest.mark.slow
def test_a_wider_pipe_costs_less_energy():
    losses = [_energy_loss(radius=b) for b in (0.0025, 0.005, 0.01)]
    assert losses[0] > losses[1] > losses[2] > 0.0


@pytest.mark.slow
def test_wake_element_is_a_lattice_element():
    """It has to survive being held by the simulator and copied into the lattice."""
    from lume_impactx.elements import element_type

    wake = ResistiveWallWake("rw", length=1.0, pipe_radius=0.01)
    assert element_type(wake) == "Programmable"
    assert wake.ds == 0.0
    assert wake.name == "rw"


def test_csr_validity_margin_matches_the_closed_form():
    """sigma_perp << R (sigma_z/R)^(2/3), per Saldin / Sagan & Mayes THPAB076."""
    from lume_impactx.wakes import csr_validity_margin

    radius, sigma_z, sigma_perp = 5.73, 1.0e-4, 1.0e-4
    expected = radius * (sigma_z / radius) ** (2 / 3) / sigma_perp
    assert csr_validity_margin(radius, sigma_z, sigma_perp) == pytest.approx(
        expected, rel=1e-12
    )
    assert csr_validity_margin(radius, sigma_z, sigma_perp) == pytest.approx(
        38.6, abs=0.1
    )


def test_csr_validity_margin_falls_for_a_fatter_bunch():
    from lume_impactx.wakes import csr_validity_margin

    wide = csr_validity_margin(5.73, 1e-4, 1e-3)
    narrow = csr_validity_margin(5.73, 1e-4, 1e-4)
    assert narrow == pytest.approx(10.0 * wide, rel=1e-12)
    assert wide < narrow


def test_csr_validity_margin_rejects_nonsense():
    from lume_impactx.wakes import csr_validity_margin

    for args in [(0.0, 1e-4, 1e-4), (5.0, 0.0, 1e-4), (5.0, 1e-4, -1.0)]:
        with pytest.raises(ValueError, match="must be positive"):
            csr_validity_margin(*args)
