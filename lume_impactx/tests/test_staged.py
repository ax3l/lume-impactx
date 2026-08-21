"""Chaining ImpactX sections through ``StagedModel``."""

from __future__ import annotations

import pytest

from lume_impactx.config import VariableMappingConfig
from lume_impactx.model import LUMEImpactXModel
from lume_impactx.simulator import ImpactXSimulator
from lume_impactx.staged import StagedImpactXModel

pytestmark = pytest.mark.slow


def _halves():
    """The FODO cell of examples/fodo, split after the mid-drift."""
    from impactx import elements

    ns = 5
    first = [
        elements.Drift(name="drift1", ds=0.25, nslice=ns),
        elements.Quad(name="quad1", ds=1.0, k=1.0, nslice=ns),
        elements.Drift(name="drift2a", ds=0.25, nslice=ns),
    ]
    second = [
        elements.Drift(name="drift2b", ds=0.25, nslice=ns),
        elements.Quad(name="quad2", ds=1.0, k=-1.0, nslice=ns),
        elements.Drift(name="drift3", ds=0.25, nslice=ns),
    ]
    return first, second


@pytest.fixture
def staged(waterbag):
    first, second = _halves()
    ref = {"species": "electron", "kin_energy_MeV": 2.0e3}

    upstream = ImpactXSimulator(
        lattice=first, ref=ref, distribution=waterbag, npart=2000, bunch_charge_C=1e-9
    )
    downstream = ImpactXSimulator(
        lattice=second, ref=ref, initial_particles=upstream.final_particles
    )
    return StagedImpactXModel.from_simulators(
        [upstream, downstream], prefixes=["a:", "b:"]
    )


def test_prefixes_keep_variable_names_unique(staged):
    """Both halves use element name 'quad*', so StagedModel needs prefixes."""
    names = set(staged.supported_variables)
    assert "a:ele:quad1:k" in names
    assert "b:ele:quad2:k" in names
    assert staged.get("a:ele:quad1:k") == pytest.approx(1.0)
    assert staged.get("b:ele:quad2:k") == pytest.approx(-1.0)


def test_duplicate_names_across_stages_are_rejected(waterbag):
    """Without prefixes the two halves collide, and StagedModel must say so."""
    first, second = _halves()
    ref = {"species": "electron", "kin_energy_MeV": 2.0e3}
    upstream = ImpactXSimulator(
        lattice=first, ref=ref, distribution=waterbag, npart=200, bunch_charge_C=1e-9
    )
    downstream = ImpactXSimulator(
        lattice=second, ref=ref, initial_particles=upstream.final_particles
    )
    plain = VariableMappingConfig()
    with pytest.raises(ValueError, match="defined in both model"):
        StagedImpactXModel.from_simulators(
            [upstream, downstream], configs=[plain, plain]
        )


def test_staging_reproduces_the_whole_cell(waterbag):
    """Two chained halves must agree with one run of the full lattice."""
    from impactx import elements

    ns = 5
    whole = [
        elements.Drift(name="drift1", ds=0.25, nslice=ns),
        elements.Quad(name="quad1", ds=1.0, k=1.0, nslice=ns),
        elements.Drift(name="drift2", ds=0.5, nslice=ns),
        elements.Quad(name="quad2", ds=1.0, k=-1.0, nslice=ns),
        elements.Drift(name="drift3", ds=0.25, nslice=ns),
    ]
    ref = {"species": "electron", "kin_energy_MeV": 2.0e3}
    reference = ImpactXSimulator(
        lattice=whole, ref=ref, distribution=waterbag, npart=2000, bunch_charge_C=1e-9
    )

    first, second = _halves()
    upstream = ImpactXSimulator(
        lattice=first, ref=ref, distribution=waterbag, npart=2000, bunch_charge_C=1e-9
    )
    downstream = ImpactXSimulator(
        lattice=second, ref=ref, initial_particles=upstream.final_particles
    )

    # The hand-off goes through a ParticleGroup round-trip, so agreement is at the
    # converter's precision rather than bit-exact.
    for key in ["sigma_x", "sigma_y", "emittance_x", "emittance_y"]:
        assert downstream.results["moments"][key] == pytest.approx(
            reference.results["moments"][key], rel=1e-9
        ), key


def test_upstream_change_propagates_downstream(staged):
    """The fix for StagedModel._set: a re-seeded stage must re-run.

    Upstream writes touch no downstream variable, so the stock implementation would
    hand the downstream stage new particles and never re-track it.
    """
    downstream = staged.lume_model_instances[1].simulator
    before_count = downstream.track_count
    before_sigma = staged.get("b:moment_final:sigma_x")

    staged.set({"a:ele:quad1:k": 1.3})

    assert downstream.track_count > before_count, "downstream stage did not re-run"
    assert staged.get("b:moment_final:sigma_x") != before_sigma


def test_downstream_change_does_not_disturb_upstream(staged):
    upstream_sigma = staged.get("a:moment_final:sigma_x")
    staged.set({"b:ele:quad2:k": -1.3})
    assert staged.get("a:moment_final:sigma_x") == upstream_sigma


def test_staged_reset(staged):
    base = staged.get("b:moment_final:sigma_x")
    staged.set({"a:ele:quad1:k": 1.3})
    assert staged.get("b:moment_final:sigma_x") != base

    staged.reset()
    assert staged.get("a:ele:quad1:k") == pytest.approx(1.0)


def test_mismatched_prefix_count_is_rejected(waterbag):
    first, _ = _halves()
    sim = ImpactXSimulator(
        lattice=first,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    with pytest.raises(ValueError, match="2 prefixes for 1 simulators"):
        StagedImpactXModel.from_simulators([sim], prefixes=["a:", "b:"])


def test_model_requires_initial_particles_for_staging(fodo_simulator):
    """A distribution-seeded simulator has no ParticleGroup to hand back as input."""
    model = LUMEImpactXModel.from_simulator(fodo_simulator)
    with pytest.raises(RuntimeError, match="seeds its beam from a distribution"):
        _ = model.initial_particles


def test_staging_across_a_bend_matches_one_shot():
    """A hand-off must survive a bent reference orbit.

    Beam coordinates are relative to the reference orbit, so the downstream reference
    stays on its own axis; seeding it with the upstream's lab position and angle would
    bend it away. This failed when the converter mixed local and lab frames: stage two
    launched the bunch 0.26 m off-axis at ~500 mrad.
    """
    from impactx import distribution, elements

    from lume_impactx.simulator import ImpactXSimulator

    waterbag = distribution.Waterbag(
        lambdaX=1e-4,
        lambdaY=1e-4,
        lambdaT=1e-3,
        lambdaPx=1e-5,
        lambdaPy=1e-5,
        lambdaPt=1e-3,
    )
    ref = {"species": "electron", "kin_energy_MeV": 100.0}
    bend = [elements.ExactSbend(name="b", ds=1.0, phi=30.0, nslice=8)]
    drift = [elements.Drift(name="d", ds=0.5, nslice=5)]

    upstream = ImpactXSimulator(
        lattice=bend, ref=ref, distribution=waterbag, npart=4000, bunch_charge_C=1e-12
    )
    downstream = ImpactXSimulator(
        lattice=drift,
        ref=ref,
        initial_particles=upstream.final_particles,
        ref_origin=upstream.results["ref_final"],
    )
    one_shot = ImpactXSimulator(
        lattice=[*bend, *drift],
        ref=ref,
        distribution=waterbag,
        npart=4000,
        bunch_charge_C=1e-12,
    )

    staged, whole = downstream.final_particles, one_shot.final_particles
    for key in ["sigma_x", "sigma_y", "mean_energy", "sigma_energy", "norm_emit_x"]:
        assert staged[key] == pytest.approx(whole[key], rel=1e-9), key


def test_reference_energy_is_carried_across_a_handoff():
    """An upstream RF cavity changes the reference energy; the next stage must inherit it.

    Beam momenta are normalized by the reference momentum, so a downstream stage holding
    its own configured energy silently rescales the incoming bunch. With an accelerating
    ShortRF between the stages this put the staged result 100% away from the one-shot
    lattice.
    """
    from impactx import distribution, elements

    from lume_impactx.simulator import ImpactXSimulator

    waterbag = distribution.Waterbag(
        lambdaX=1e-4,
        lambdaY=1e-4,
        lambdaT=1e-3,
        lambdaPx=1e-5,
        lambdaPy=1e-5,
        lambdaPt=1e-3,
    )
    ref = {"species": "electron", "kin_energy_MeV": 250.0}
    first = [
        elements.Drift(name="d1", ds=0.3, nslice=4),
        elements.ShortRF(name="rf", V=5e6, freq=1.3e9, phase=0.0),
    ]
    second = [
        elements.Drift(name="d2", ds=0.5, nslice=4),
        elements.Quad(name="q", ds=0.2, k=1.0, nslice=4),
    ]

    upstream = ImpactXSimulator(
        lattice=first, ref=ref, distribution=waterbag, npart=2000, bunch_charge_C=1e-12
    )
    exit_ref = upstream.results["ref_final"]
    exit_energy = (exit_ref.gamma - 1.0) * exit_ref.mass_MeV
    assert exit_energy != pytest.approx(250.0), "the RF must actually accelerate"

    downstream = ImpactXSimulator(
        lattice=second,
        ref=ref,
        initial_particles=upstream.final_particles,
        ref_origin=exit_ref,
    )
    one_shot = ImpactXSimulator(
        lattice=[*first, *second],
        ref=ref,
        distribution=waterbag,
        npart=2000,
        bunch_charge_C=1e-12,
    )

    staged, whole = downstream.final_particles, one_shot.final_particles
    for key in ["sigma_x", "sigma_px", "mean_energy", "sigma_energy", "norm_emit_x"]:
        assert staged[key] == pytest.approx(whole[key], rel=1e-9), key


def test_energy_is_read_only_on_a_stage_fed_from_upstream():
    """A stage seeded from an upstream section takes its energy from that section.

    Leaving `ref:kin_energy_MeV` writable there accepted a value, read it back, and
    discarded it on the next track -- the silent no-op this package refuses elsewhere.
    """
    from impactx import distribution, elements
    from lume.exceptions import ReadOnlyError

    from lume_impactx.model import LUMEImpactXModel
    from lume_impactx.simulator import ImpactXSimulator

    waterbag = distribution.Waterbag(
        lambdaX=1e-4,
        lambdaY=1e-4,
        lambdaT=1e-3,
        lambdaPx=1e-5,
        lambdaPy=1e-5,
        lambdaPt=1e-3,
    )
    ref = {"species": "electron", "kin_energy_MeV": 100.0}
    upstream = ImpactXSimulator(
        lattice=[elements.Drift(name="d1", ds=0.3, nslice=4)],
        ref=ref,
        distribution=waterbag,
        npart=500,
        bunch_charge_C=1e-12,
    )
    downstream = ImpactXSimulator(
        lattice=[elements.Drift(name="d2", ds=0.3, nslice=4)],
        ref=ref,
        initial_particles=upstream.final_particles,
        ref_origin=upstream.results["ref_final"],
    )

    standalone = LUMEImpactXModel.from_simulator(upstream)
    staged = LUMEImpactXModel.from_simulator(downstream)

    # writable where it has an effect ...
    assert standalone.supported_variables["ref:kin_energy_MeV"].read_only is False
    # ... read-only where the value comes from upstream
    assert staged.supported_variables["ref:kin_energy_MeV"].read_only is True
    with pytest.raises(ReadOnlyError):
        staged.set({"ref:kin_energy_MeV": 250.0})
