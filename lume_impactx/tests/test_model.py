"""End-to-end tests: a real ImpactX FODO cell driven through the LUME model API.

Every test here builds and tears down ImpactX sessions, which is exactly what
``ImpactXSimulator.track()`` does. If they start aborting rather than failing, the MPI
bootstrap in ``lume_impactx._mpi`` has stopped working -- see that module.
"""

from __future__ import annotations

import numpy as np
import pytest
from lume.exceptions import ReadOnlyError

from lume_impactx.model import LUMEImpactXModel
from lume_impactx.simulator import ImpactXSimulator

pytestmark = pytest.mark.slow


@pytest.fixture
def model(fodo_simulator):
    return LUMEImpactXModel.from_simulator(fodo_simulator)


def test_simulator_predicts_its_own_history_length(fodo_simulator):
    """n_steps == periods * sum(nslice), known before the run and used as NDVariable shape."""
    assert fodo_simulator.n_steps == 25
    assert len(fodo_simulator.results["moments_history"]) == 25


def test_rebuild_per_track_is_deterministic(fodo_simulator):
    """A fresh session re-seeds the RNG, so repeated tracks are bit-identical."""
    first = fodo_simulator.results["moments"]["sigma_x"]
    fodo_simulator.track()
    assert fodo_simulator.results["moments"]["sigma_x"] == first
    fodo_simulator.track()
    assert fodo_simulator.results["moments"]["sigma_x"] == first
    assert fodo_simulator.track_count == 3


def test_set_retracks_and_changes_the_beam(model):
    base = model.get("moment_final:sigma_x")
    model.set({"ele:quad1:k": 1.2})
    assert model.get("ele:quad1:k") == pytest.approx(1.2)
    assert model.get("moment_final:sigma_x") != base


def test_set_is_reproducible(model):
    model.set({"ele:quad1:k": 1.2})
    once = model.get("moment_final:sigma_x")
    model.set({"ele:quad1:k": 1.0})
    model.set({"ele:quad1:k": 1.2})
    assert model.get("moment_final:sigma_x") == once


def test_reset_restores_exactly(model):
    base = model.get("moment_final:sigma_x")
    model.set({"ele:quad1:k": 1.2, "ele:drift2:ds": 0.7})
    assert model.get("moment_final:sigma_x") != base

    model.reset()
    assert model.get("ele:quad1:k") == pytest.approx(1.0)
    assert model.get("ele:drift2:ds") == pytest.approx(0.5)
    assert model.get("moment_final:sigma_x") == base


def test_read_only_variables_reject_writes(model):
    for name in [
        "moment_final:sigma_x",  # an output
        "ele:drift1:aperture_x",  # ImpactX refuses it
        "ele:quad1:nslice",  # we refuse it: it fixes every s-series shape
        "sim:periods",
    ]:
        with pytest.raises(ReadOnlyError):
            model.set({name: 1.0})


def test_unknown_variable_is_rejected(model):
    with pytest.raises(ValueError, match="not supported"):
        model.get("moment_final:does_not_exist")
    with pytest.raises(ValueError, match="not supported"):
        model.set({"ele:nope:k": 1.0})


def test_moment_history_has_the_exact_shape(model):
    history = model.get("moment:sigma_x")
    assert history.shape == (25,)
    assert np.isfinite(history).all(), "exact sizing means no NaN padding"
    assert history[-1] == pytest.approx(model.get("moment_final:sigma_x"), rel=1e-12)


def test_variables_carry_metadata_for_pva(model):
    """lume-pva builds PVs from unit / read_only / shape, so they must be populated."""
    k = model.supported_variables["ele:quad1:k"]
    assert k.unit == "1/m^2" and k.read_only is False

    sigma_x = model.supported_variables["moment_final:sigma_x"]
    assert sigma_x.unit == "m" and sigma_x.read_only is True

    history = model.supported_variables["moment:sigma_x"]
    assert history.shape == (25,) and history.unit == "m"


def test_legacy_moment_aliases_are_dropped(model):
    names = set(model.supported_variables)
    assert "moment_final:sigma_x" in names
    assert "moment_final:sig_x" not in names
    assert "moment_final:x_mean" not in names


def test_dummy_run_defers_tracking(fodo_simulator):
    model = LUMEImpactXModel.from_simulator(fodo_simulator, dummy_run=True)
    before = fodo_simulator.track_count
    model.set({"ele:quad1:k": 1.2})
    assert fodo_simulator.track_count == before, "dummy_run must not re-track"

    fodo_simulator.track()
    assert fodo_simulator.track_count == before + 1


def test_reference_particle_is_writable(model):
    """Writing the reference energy must reach the rebuilt session.

    ``sigma_x`` deliberately does *not* move: ImpactX's ``Quad.k`` is in m^-2 (MADX
    convention, already normalized by rigidity) and the Waterbag ``lambda*`` are
    absolute, so a pure energy change leaves the transverse dynamics identical. The
    reference particle is where the change shows up.
    """
    electron_mass_MeV = 0.5109989506917532
    assert model.get("ref_final:s") == pytest.approx(3.0)
    assert model.get("ref_final:gamma") == pytest.approx(
        1.0 + 2.0e3 / electron_mass_MeV, rel=1e-9
    )
    sigma_x_before = model.get("moment_final:sigma_x")

    model.set({"ref:kin_energy_MeV": 2.5e3})

    assert model.get("ref:kin_energy_MeV") == pytest.approx(2.5e3)
    assert model.get("ref_final:gamma") == pytest.approx(
        1.0 + 2.5e3 / electron_mass_MeV, rel=1e-9
    )
    assert model.get("moment_final:sigma_x") == sigma_x_before


def test_final_particles_round_trip(model):
    particles = model.get("particles:final_particles")
    assert particles.n_particle == 1000
    assert particles.species == "electron"
    assert particles.charge == pytest.approx(1.0e-9, rel=1e-9)


def test_initial_particles_seeding(fodo_lattice, model):
    """A ParticleGroup out of one run can seed the next -- the staging primitive."""
    bunch = model.get("particles:final_particles")

    staged = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        initial_particles=bunch,
    )
    staged_model = LUMEImpactXModel.from_simulator(staged)
    assert "particles:initial_particles" in staged_model.supported_variables
    assert staged_model.get("run_info:n_particles") == 1000
    assert staged_model.get("moment_final:sigma_x") > 0.0


# --------------------------------------------------------------------------------------
# Parity with the behaviours lume-impact, lume-bmad and lume-cheetah cover.
# --------------------------------------------------------------------------------------


def test_simulator_requires_exactly_one_beam_source(fodo_lattice, waterbag, bunch):
    """lume-cheetah's CheetahSimulator makes the same XOR check."""
    ref = {"species": "electron", "kin_energy_MeV": 2.0e3}
    with pytest.raises(ValueError, match="exactly one of"):
        ImpactXSimulator(lattice=fodo_lattice, ref=ref)
    with pytest.raises(ValueError, match="exactly one of"):
        ImpactXSimulator(
            lattice=fodo_lattice,
            ref=ref,
            distribution=waterbag,
            npart=100,
            bunch_charge_C=1e-9,
            initial_particles=bunch,
        )
    with pytest.raises(ValueError, match="npart"):
        ImpactXSimulator(lattice=fodo_lattice, ref=ref, distribution=waterbag)


def test_setting_initial_particles_through_the_model(fodo_lattice, model):
    """lume-impact's test_particle_group_set_initial / lume-bmad's equivalent."""
    seed = model.get("particles:final_particles")

    staged = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        initial_particles=seed,
    )
    staged_model = LUMEImpactXModel.from_simulator(staged)
    before = staged_model.get("moment_final:sigma_x")

    # a visibly different bunch: same beam, half the particles
    half = seed[: seed.n_particle // 2]
    staged_model.set({"particles:initial_particles": half})

    assert staged_model.get("run_info:n_particles") == half.n_particle
    assert staged_model.get("moment_final:sigma_x") != before
    assert staged_model.get("particles:initial_particles").n_particle == half.n_particle


def test_element_reads_are_live_but_moments_are_cached(model, fodo_simulator):
    """The two halves of the model have deliberately different freshness.

    Element variables read the lattice directly, so an external mutation shows up at
    once. Moments come from the last track's snapshot -- the container is gone by then
    -- so they only move when something re-tracks. lume-cheetah exposes the same split
    through its explicit ``update_state()``.
    """
    moments_before = model.get("moment_final:sigma_x")

    fodo_simulator.lattice[1].k = 1.4
    assert model.get("ele:quad1:k") == pytest.approx(1.4)  # live
    assert model.get("moment_final:sigma_x") == moments_before  # cached

    fodo_simulator.track()
    assert model.get("moment_final:sigma_x") != moments_before


def test_register_and_replace_action_variables(model):
    """ActionModel's registration API, as lume-impact tests it."""
    from lume_impactx.actions import MomentAction

    extra = MomentAction(
        name="moment_final:custom", moment_name="sigma_y", unit="m", read_only=True
    )
    model.register_action_variable(extra)
    assert "moment_final:custom" in model.supported_variables
    assert model.get("moment_final:custom") == pytest.approx(
        model.get("moment_final:sigma_y")
    )

    replacement = MomentAction(
        name="moment_final:custom", moment_name="sigma_x", unit="m", read_only=True
    )
    model.register_action_variable(replacement)
    assert model.get("moment_final:custom") == pytest.approx(
        model.get("moment_final:sigma_x")
    )
    assert (
        len([n for n in model.supported_variables if n == "moment_final:custom"]) == 1
    )

    model.unregister_action_variable("moment_final:custom")
    assert "moment_final:custom" not in model.supported_variables


@pytest.mark.slow
def test_linear_transfer_maps_are_captured_and_correctly_oriented(waterbag):
    """lume-bmad exposes mat6/vec0; this is the ImpactX equivalent.

    The orientation matters and is easy to get wrong: AMReX SmallMatrix is Fortran
    ordered, so ``np.asarray`` on one returns the transpose. A drift must be
    ``[[1, L], [0, 1]]``; the transposed form ``[[1, 0], [L, 1]]`` looks just as
    plausible.
    """
    import numpy as np
    from impactx import elements

    length = 0.5
    simulator = ImpactXSimulator(
        lattice=[elements.Drift(name="d", ds=length, nslice=1)],
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    results = simulator.results

    transfer_map = results["transfer_map"]
    assert transfer_map.shape == (6, 6)
    np.testing.assert_allclose(
        transfer_map[:2, :2], [[1.0, length], [0.0, 1.0]], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        transfer_map[2:4, 2:4], [[1.0, length], [0.0, 1.0]], rtol=1e-12, atol=1e-12
    )

    assert results["cumulative_maps"].shape == (len(results["map_s"]), 6, 6)
    assert results["map_names"][-1] == "d"
    np.testing.assert_allclose(results["map_s"], [0.0, length], rtol=1e-12)


@pytest.mark.slow
def test_cumulative_maps_run_from_the_lattice_start(waterbag):
    """map_trace is cumulative, not per element -- entry i is start -> element i exit.

    Multiplying the entries together would be wrong; the last one already *is* the
    whole-lattice map.
    """
    import numpy as np
    from impactx import elements

    simulator = ImpactXSimulator(
        lattice=[
            elements.Drift(name="d1", ds=0.4, nslice=1),
            elements.Quad(name="q", ds=0.3, k=1.2, nslice=1),
            elements.Drift(name="d2", ds=0.2, nslice=1),
        ],
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    results = simulator.results
    maps, s_values = results["cumulative_maps"], results["map_s"]

    np.testing.assert_allclose(s_values, [0.0, 0.4, 0.7, 0.9], rtol=1e-12)
    np.testing.assert_allclose(maps[0], np.eye(6), atol=1e-12)
    np.testing.assert_allclose(
        maps[-1], results["transfer_map"], rtol=1e-12, atol=1e-14
    )

    # the first drift, read off the cumulative map at s = 0.4
    np.testing.assert_allclose(maps[1][:2, :2], [[1.0, 0.4], [0.0, 1.0]], atol=1e-12)


from lume_impactx import actions as actions_module  # noqa: E402
from lume_impactx.tests.conftest import BUNCH_CHARGE_C, KIN_ENERGY_MEV  # noqa: E402


# -- the addressing surface a virtual accelerator needs --------------------------------
#
# SLAC's virtual-accelerator addresses elements by *name* across all three of its
# backends -- impact.ele[name], getattr(cheetah.segment, name), bmad's
# ele_gen_attribs(name) -- and drives composite devices through simulator[group][key].
# ImpactX names are not unique, so these carry the same ##2/##3 disambiguation the
# captured bunches use.


def test_elements_are_addressable_by_name(fodo_simulator):
    elements = fodo_simulator.ele
    assert elements["quad1"].k == pytest.approx(1.0)
    assert elements["quad2"].k == pytest.approx(-1.0)
    # Bmad hands names back upper case; lookup folds it.
    assert elements["QUAD1"].k == pytest.approx(1.0)
    # Live elements, so a write reaches the lattice the next track uses.
    elements["quad1"].k = 1.5
    assert fodo_simulator.lattice[1].k == pytest.approx(1.5)


def test_repeated_element_names_are_disambiguated(fodo_lattice, waterbag):
    """A lattice may use one element twice, and lattice_from_tao splits a single Bmad
    element into several. Names alone cannot address those."""
    from impactx import elements as impactx_elements

    from lume_impactx.simulator import ImpactXSimulator

    lattice = fodo_lattice + [impactx_elements.Quad(name="quad1", ds=1.0, k=2.0)]
    simulator = ImpactXSimulator(
        lattice=lattice,
        ref={"species": "electron", "kin_energy_MeV": KIN_ENERGY_MEV},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=BUNCH_CHARGE_C,
        track_on_init=False,
    )
    assert simulator.ele["quad1"].k == pytest.approx(1.0)
    assert simulator.ele["quad1##2"].k == pytest.approx(2.0)
    assert len(simulator.ele.all_named("quad1")) == 2


def test_an_unknown_element_name_lists_what_there_is(fodo_simulator):
    with pytest.raises(KeyError, match="quad1"):
        fodo_simulator.ele["nowhere"]


def test_a_group_writes_every_member(fodo_simulator):
    """simulator[group][key] = value, the shape lume-impact's control groups have."""
    fodo_simulator.groups = {"QUADS": ["quad1", "quad2"]}
    group = fodo_simulator["QUADS"]
    assert len(group) == 2
    assert group["k"] == pytest.approx(1.0)  # reads the first member

    group["k"] = 3.0
    assert fodo_simulator.ele["quad1"].k == pytest.approx(3.0)
    assert fodo_simulator.ele["quad2"].k == pytest.approx(3.0)


def test_a_group_says_so_when_an_attribute_is_wrong(fodo_simulator):
    fodo_simulator.groups = {"MIXED": ["quad1", "drift1"]}
    with pytest.raises(KeyError, match="has no attribute"):
        fodo_simulator["MIXED"]["k"] = 1.0


def test_a_bare_subscript_means_a_group_and_says_so(fodo_simulator):
    """Narrower than lume-impact's Impact.__getitem__, which also resolves elements,
    header keys, end_ stats and particles: paths. One meaning, and the error points at
    the other two accessors."""
    with pytest.raises(KeyError) as caught:
        fodo_simulator["quad1"]
    assert ".ele[name]" in str(caught.value)
    assert ".particles[name]" in str(caught.value)


def test_particles_can_be_called_as_well_as_subscripted(fodo_simulator):
    """lume-impact subscripts, lume-bmad calls. Both work."""
    assert (
        fodo_simulator.particles("final")["sigma_x"]
        == fodo_simulator.particles["final"]["sigma_x"]
    )


def test_reference_energy_is_reported_per_element(fodo_simulator):
    """Anything converting a magnet setting to kG needs the rigidity at that element."""
    total_eV = (KIN_ENERGY_MEV + 0.51099895069) * 1e6
    for name in ("quad1", "quad2"):
        assert fodo_simulator.reference_energy_at(name) == pytest.approx(
            total_eV, rel=1e-9
        )
    assert set(fodo_simulator.energies) == set(fodo_simulator.ele)


def test_reference_energy_follows_an_accelerating_cavity(fodo_lattice, waterbag):
    """A ShortRF moves ImpactX's reference, so two quads with the same name on either
    side of one are at different energies. Replayed analytically -- ShortRF.H:207 does
    pt -= V*cos(phase) with pt = -gamma -- so no run is needed."""
    from impactx import elements as impactx_elements

    from lume_impactx.simulator import ImpactXSimulator

    mass_eV = 0.51099895069e6
    gain_eV = 20.0e6
    lattice = [
        impactx_elements.Quad(name="q", ds=0.1, k=1.0, nslice=1),
        impactx_elements.ShortRF(
            name="cav", V=gain_eV / mass_eV, freq=1.3e9, phase=0.0
        ),
        impactx_elements.Quad(name="q", ds=0.1, k=1.0, nslice=1),
    ]
    simulator = ImpactXSimulator(
        lattice=lattice,
        ref={"species": "electron", "kin_energy_MeV": KIN_ENERGY_MEV},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=BUNCH_CHARGE_C,
        track_on_init=False,
    )
    before = simulator.reference_energy_at("q")
    after = simulator.reference_energy_at("q##2")
    assert after - before == pytest.approx(gain_eV, rel=1e-9)


# -- bases a facility subclasses for its own PVs ---------------------------------------


class _QuadKGVariable(actions_module.ImpactXWritableScalarVariable):
    """A stand-in for a facility's BCTRL: kG from gradient x magnetic rigidity."""

    unit: str = "kG"

    @staticmethod
    def _rigidity(energy_eV: float) -> float:
        return 33.356 * energy_eV / 1e9  # kG-m

    def _get(self, simulator):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        return element.k * element.ds * self._rigidity(energy)

    def _set(self, simulator, value):
        element, energy = self._resolve_element_and_energy(simulator, self.element_name)
        element.k = value / (element.ds * self._rigidity(energy))


class _QuadKGReadback(actions_module.ImpactXReadOnlyActionMixin, _QuadKGVariable):
    """The BACT to that BCTRL: same conversion, no write."""


def test_a_facility_can_subclass_the_name_keyed_bases(fodo_simulator):
    """The conversion logic belongs in the facility's repo; the addressing belongs here.

    This is the shape lume-cheetah exports and SLAC's virtual accelerator subclasses.
    """
    control = _QuadKGVariable(name="QUAD1:BCTRL", element_name="quad1")

    kG = control._get(fodo_simulator)
    assert kG != 0.0
    control._set(fodo_simulator, kG * 2.0)
    assert fodo_simulator.ele["quad1"].k == pytest.approx(2.0)  # was 1.0
    assert control._get(fodo_simulator) == pytest.approx(kG * 2.0)


def test_a_readback_reuses_its_controls_conversion(fodo_simulator):
    """BACT is a one-line subclass of BCTRL, not a second copy of the physics."""
    from lume.exceptions import ReadOnlyError

    readback = _QuadKGReadback(name="QUAD1:BACT", element_name="quad1")
    control = _QuadKGVariable(name="QUAD1:BCTRL", element_name="quad1")

    assert readback._get(fodo_simulator) == pytest.approx(control._get(fodo_simulator))
    assert readback.read_only is True
    with pytest.raises(ReadOnlyError):
        readback._set(fodo_simulator, 1.0)


def test_the_name_keyed_base_reports_the_energy_at_that_element(fodo_simulator):
    element, energy = actions_module._ElementByNameMixin._resolve_element_and_energy(
        fodo_simulator, "quad2"
    )
    assert element is fodo_simulator.lattice[3]
    assert energy == pytest.approx(fodo_simulator.reference_energy_at("quad2"))


def test_a_name_keyed_variable_survives_a_lattice_edit(fodo_simulator, fodo_lattice):
    """Unlike the index-keyed generated variables, a name still finds its element after
    the lattice is edited -- which is the point of addressing this way."""
    from impactx import elements as impactx_elements

    control = _QuadKGVariable(name="QUAD1:BCTRL", element_name="quad1")
    before = control._get(fodo_simulator)

    fodo_simulator.lattice.insert(0, impactx_elements.Drift(name="new", ds=0.1))
    assert control._get(fodo_simulator) == pytest.approx(before)


def test_a_bunch_at_element_variable_reads_a_capture(fodo_simulator):
    variable = actions_module.ImpactXBunchAtElementVariable(
        name="probe", element_name="final"
    )
    assert variable._get(fodo_simulator)["sigma_x"] == pytest.approx(
        fodo_simulator.final_particles["sigma_x"]
    )
