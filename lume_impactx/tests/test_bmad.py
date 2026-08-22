"""Bmad/Tao translation.

Skipped unless pytao is installed. It is not a dependency: Bmad comes from conda-forge
(`conda install -c conda-forge bmad pytao`) and most users of this package never need it.

The lattice fixture is a copy of lume-bmad's FODO cell, so both packages are exercised
against the same model.
"""

from __future__ import annotations

import itertools
import math
import pathlib
import warnings

import numpy as np
import pytest

pytao = pytest.importorskip("pytao", reason="Bmad/pytao is not installed")

from lume_impactx.interfaces.bmad import (  # noqa: E402
    TaoTranslationWarning,
    UnsupportedElementError,
    beam_from_tao,
    lattice_from_tao,
    particles_from_tao,
    reference_from_tao,
)

LATTICE_DIR = pathlib.Path(__file__).parent / "bmad"
ELECTRON_MASS_EV = 510998.95069


@pytest.fixture
def tao(tmp_path, monkeypatch):
    """A Tao FODO model with a tracked, saved beam."""
    for name in ("fodo.init", "fodo.bmad"):
        (tmp_path / name).write_text((LATTICE_DIR / name).read_text())
    monkeypatch.chdir(tmp_path)

    instance = pytao.Tao(init_file="fodo.init", noplot=True)
    instance.cmd("set global track_type = beam")
    instance.cmd("set beam saved_at = *")
    return instance


@pytest.fixture
def untracked_tao(tmp_path, monkeypatch):
    """The same model with no beam tracked, for the error path."""
    for name in ("fodo.init", "fodo.bmad"):
        (tmp_path / name).write_text((LATTICE_DIR / name).read_text())
    monkeypatch.chdir(tmp_path)
    return pytao.Tao(init_file="fodo.init", noplot=True)


# -- beam ------------------------------------------------------------------------------


def test_particles_come_across_as_a_particlegroup(tao):
    particles = particles_from_tao(tao)
    assert particles.n_particle == 1000
    assert particles.species == "electron"
    assert particles.charge == pytest.approx(1e-9, rel=1e-9)
    # z-coordinates is what the ImpactX converter expects
    assert particles.in_z_coordinates


def test_reference_uses_the_design_energy_not_the_bunch_mean(tao):
    reference = reference_from_tao(tao)
    assert reference["species"] == "electron"

    total_eV = float(tao.ele_gen_attribs("BEGINNING")["E_TOT"])
    assert reference["kin_energy_MeV"] == pytest.approx(
        (total_eV - ELECTRON_MASS_EV) / 1e6, rel=1e-12
    )
    # the bunch mean differs slightly; the design reference is the faithful choice
    assert particles_from_tao(tao)["mean_energy"] != pytest.approx(total_eV, rel=1e-12)


def test_untracked_tao_gives_an_actionable_error(untracked_tao):
    with pytest.raises(RuntimeError, match="no tracked bunch") as excinfo:
        particles_from_tao(untracked_tao)
    message = str(excinfo.value)
    assert "track_type = beam" in message
    assert "saved_at" in message


def test_beam_from_tao_agrees_with_its_parts(tao):
    reference, particles = beam_from_tao(tao)
    assert reference == reference_from_tao(tao)
    assert particles.n_particle == particles_from_tao(tao).n_particle


# -- lattice ---------------------------------------------------------------------------
#
# Each mapping below was established by comparing Tao's `ele_mat6`, taken into ImpactX's
# basis, and by tracking the same bunch through both codes. The numbers quoted in the
# assertions and comments are measured, not aspirational.


@pytest.fixture
def make_tao(tmp_path, monkeypatch):
    """Build a Tao model from an inline lattice body and beam line."""
    counter = itertools.count()

    def build(body: str, line: str, particle: str = "electron", e_tot: float = 100e6):
        directory = tmp_path / f"lat{next(counter)}"
        directory.mkdir()
        (directory / "lat.bmad").write_text(
            f"parameter[geometry] = open\n"
            f"parameter[particle] = {particle}\n"
            f"parameter[e_tot] = {e_tot}\n"
            f"beginning[beta_a] = 10\nbeginning[beta_b] = 10\n"
            f"{body}\nlat: line = ({line})\nuse, lat\n"
        )
        (directory / "tao.init").write_text(
            "&tao_start\n/\n&tao_design_lattice\n"
            '  design_lattice(1)%file = "lat.bmad"\n/\n'
        )
        monkeypatch.chdir(directory)
        return pytao.Tao(init_file="tao.init", noplot=True)

    return build


def kinds(lattice):
    return [type(element).__name__ for element in lattice]


def test_fodo_translates_to_the_models_that_match_bmad(tao):
    """Bmad's quadrupole body is paraxial in (x, y) but exact in energy, which is
    ImpactX's Chr* family -- not Exact*. Measured: ChrQuad 1.9e-14, ExactQuad 4.5e-9,
    linear Quad 8.0e-5. Drifts are the other way round: Bmad's drift really is exact.

    Nothing about a plain FODO cell is lossy, so it must translate without warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", TaoTranslationWarning)
        lattice = lattice_from_tao(tao, nslice=5)

    assert kinds(lattice) == [
        "Marker",  # BEGINNING
        "ChrQuad",
        "ExactDrift",
        "ChrQuad",
        "ExactDrift",
        "Marker",  # END
    ]
    quad = lattice[1]
    assert quad.ds == pytest.approx(0.25)
    assert quad.nslice == 5
    assert quad.k == pytest.approx(1.2)  # Bmad's K1 sign convention is ImpactX's


def test_impactx_reproduces_bmad_tracking(tao):
    """The translated FODO must track to the same beam Bmad does.

    The beam hand-off is exact, so the energy distribution matches to machine
    precision. The transverse residual is ImpactX's own quadrupole-model floor, ~1e-8;
    Bmad's own integrators differ from each other by about as much.
    """
    from lume_impactx import ImpactXSimulator

    with warnings.catch_warnings():
        warnings.simplefilter("error", TaoTranslationWarning)
        simulator = ImpactXSimulator.from_tao(tao, nslice=16)

    bmad_end = tao.particles("END")
    impactx_end = simulator.final_particles

    assert impactx_end.n_particle == bmad_end.n_particle
    assert impactx_end.charge == pytest.approx(bmad_end.charge, rel=1e-12)
    assert impactx_end["mean_energy"] == pytest.approx(
        bmad_end["mean_energy"], rel=1e-12
    )
    assert impactx_end["sigma_energy"] == pytest.approx(
        bmad_end["sigma_energy"], rel=1e-12
    )
    for key in ("sigma_x", "sigma_y", "norm_emit_x", "norm_emit_y"):
        assert impactx_end[key] == pytest.approx(bmad_end[key], rel=1e-6), key


# -- bends -----------------------------------------------------------------------------

BEND = "b: sbend, l = 0.5, angle = 0.12, e1 = 0.06, e2 = 0.06"


def test_fringe_type_none_gives_no_edges(make_tao):
    """Adding a DipEdge anyway is wrong by 8.9e-2 on the transfer map."""
    tao = make_tao(f"{BEND}, fringe_type = none", "b")
    assert kinds(lattice_from_tao(tao)) == ["Marker", "ExactSbend", "Marker"]


def test_bmads_default_bend_fringe_is_a_nonlinear_dipedge(make_tao):
    """`basic_bend` is Bmad's default and maps exactly, so it must not warn.

    The edge model must be `nonlinear`: with `linear`, tracking is wrong by 2.2e-4
    where `nonlinear` agrees to 2.1e-9.
    """
    tao = make_tao(BEND, "b")
    with warnings.catch_warnings():
        warnings.simplefilter("error", TaoTranslationWarning)
        lattice = lattice_from_tao(tao)

    assert kinds(lattice) == ["Marker", "DipEdge", "ExactSbend", "DipEdge", "Marker"]
    entry, exit_ = lattice[1], lattice[3]
    assert entry.psi == pytest.approx(0.06)
    assert exit_.psi == pytest.approx(0.06)
    assert entry.rc == pytest.approx(0.5 / 0.12)
    for edge in (entry, exit_):
        assert edge.to_dict()["model"] == "nonlinear"
        assert edge.g == 0.0
    assert entry.to_dict()["location"] == "entry"
    assert exit_.to_dict()["location"] == "exit"


def test_a_zero_pole_face_angle_still_gets_an_edge(make_tao):
    """Regression: ImpactX's nonlinear edge keeps a loc/(2*rc) term at psi = 0.

    Skipping the edge because `psi == 0` made a bend wrong by 3.9e-4.
    """
    tao = make_tao("b: sbend, l = 0.5, angle = 0.12", "b")
    lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "DipEdge", "ExactSbend", "DipEdge", "Marker"]
    assert lattice[1].psi == 0.0


def test_bmads_default_fringe_carries_fint_and_hgap(make_tao):
    """Regression: `basic_bend` is Bmad's default AND it keeps the soft edge.

    fringe_mod.f90:280 zeroes fint_gap only for `hard_edge_only` and `sad_full`.
    Treating `basic_bend` as hard-edge-only dropped FINT/HGAP silently and tracked
    3.2e-2 away from Bmad; carrying it reaches 2.2e-9.
    """
    tao = make_tao(f"{BEND}, fint = 0.5, hgap = 0.03", "b")
    with warnings.catch_warnings():
        warnings.simplefilter("error", TaoTranslationWarning)
        lattice = lattice_from_tao(tao)
    entry = lattice[1]
    assert entry.g == pytest.approx(0.06)  # g = 2*HGAP
    assert entry.to_dict()["K2"] == pytest.approx(0.5)  # K2 = FINT
    # K3 defaults to 1/6 in ImpactX and adds a y^3 term Bmad has no counterpart for,
    # whose 1/g factor grows as the gap shrinks: 1.0e-6 against 2.1e-9 with K3 = 0.
    assert entry.to_dict()["K3"] == 0.0


def test_the_fringe_types_that_genuinely_differ_say_so(make_tao):
    """`full` uses a PTC Lie map in Bmad, not DipEdge's Hwang & Lee map."""
    tao = make_tao(f"{BEND}, fringe_type = full, fint = 0.5, hgap = 0.03", "b")
    with pytest.warns(TaoTranslationWarning, match="different edge map"):
        lattice_from_tao(tao)


def test_fringe_at_selects_which_ends_get_an_edge(make_tao):
    tao = make_tao(f"{BEND}, fringe_at = entrance_end", "b")
    assert kinds(lattice_from_tao(tao)) == [
        "Marker",
        "DipEdge",
        "ExactSbend",
        "Marker",
    ]


def test_ref_tilt_becomes_a_rotation_of_the_whole_bend(make_tao):
    """A vertical bend: `rotation = degrees(REF_TILT)` reproduces Bmad to 1.2e-11.

    The rotation must reach the edges too, or the bend is wrong by 3.9e-4.
    """
    tao = make_tao(f"{BEND}, ref_tilt = pi/2", "b")
    lattice = lattice_from_tao(tao)
    for element in lattice[1:4]:
        assert element.rotation == pytest.approx(90.0)


def test_roll_becomes_a_kick_between_two_half_bends(make_tao):
    """No ImpactX element expresses ROLL, so it is carried as its physical effect.

    `rotation`, `PlaneXYRot` and `PRot` all turn the reference orbit with the magnet,
    which is REF_TILT, not ROLL. An on-axis particle leaves a bend rolled by psi with
    px = ANGLE*(1 - cos psi) and py = -ANGLE*sin psi; applying that between two half
    bends reproduces 99.93% of the roll, measured end to end against Bmad.
    """
    roll = 0.01
    tao = make_tao(f"{BEND}, roll = {roll}", "b")
    with pytest.warns(TaoTranslationWarning, match="ROLL"):
        lattice = lattice_from_tao(tao, nslice=8)

    assert kinds(lattice) == [
        "Marker",
        "DipEdge",
        "ExactSbend",
        "Kicker",
        "ExactSbend",
        "DipEdge",
        "Marker",
    ]
    first, kick, second = lattice[2], lattice[3], lattice[4]
    assert first.ds == pytest.approx(0.25)
    assert second.ds == pytest.approx(0.25)
    assert first.nslice == 4  # the halves share the requested slice count
    assert kick.xkick == pytest.approx(0.12 * (1.0 - math.cos(roll)))
    assert kick.ykick == pytest.approx(-0.12 * math.sin(roll))
    # The bend plane itself is untouched: a roll is not a REF_TILT.
    assert first.rotation == pytest.approx(0.0)


def test_an_unrolled_bend_is_not_split(make_tao):
    tao = make_tao(BEND, "b")
    assert kinds(lattice_from_tao(tao)) == [
        "Marker",
        "DipEdge",
        "ExactSbend",
        "DipEdge",
        "Marker",
    ]


def test_a_combined_function_bend_is_split_in_bmads_own_model(make_tao):
    """ImpactX's CFbend is non-chromatic (1.6e-4 at dp/p = 5e-4, 1.1e-3 at 2e-3).

    Bmad's combined-function body is paraxial-but-chromatic, so drift-kick-drift with
    ChrQuad and ThinDipole reproduces it and converges: 7.5e-5 at 8 steps, 4.7e-6 at 32.
    """
    tao = make_tao("b: sbend, l = 0.5, angle = 0.12, k1 = 0.8", "b")
    with pytest.warns(TaoTranslationWarning, match="ChrQuad/ThinDipole"):
        lattice = lattice_from_tao(tao, nslice=4)
    assert kinds(lattice).count("ChrQuad") == 8
    assert kinds(lattice).count("ThinDipole") == 4
    assert "CFbend" not in kinds(lattice)


def test_vertically_pure_multipoles_use_exactcfbend(make_tao):
    """ImpactX's ExactCFbend is curvilinear, so it matches only that Bmad setting.

    Verified to be a convention floor rather than a convergence one: unmoved by Bmad
    num_steps=400 and integrator_order=6, and by ImpactX int_order 2->6 and mapsteps
    10->400. ImpactX's own defaults are under-converged, hence int_order=4/mapsteps=100.
    """
    tao = make_tao(
        "b: sbend, l = 0.5, angle = 0.12, k1 = 0.8, exact_multipoles = vertically_pure",
        "b",
    )
    with pytest.warns(TaoTranslationWarning, match="ExactCFbend"):
        lattice = lattice_from_tao(tao, nslice=4)
    bend = next(e for e in lattice if type(e).__name__ == "ExactCFbend")
    assert bend.to_dict()["int_order"] == 4
    assert bend.to_dict()["mapsteps"] == 100


def test_a_bends_sextupole_component_is_carried(make_tao):
    """Regression: K2 on a bend was read by nobody.

    It does not enter the linear map, so a transfer-matrix check cannot see it;
    tracking moved 4.6e-2. Carried as a thin Multipole per step it reaches 1.3e-5.
    """
    tao = make_tao("b: sbend, l = 0.5, angle = 0.12, k2 = 30", "b")
    with pytest.warns(TaoTranslationWarning):
        lattice = lattice_from_tao(tao, nslice=4)
    sextupoles = [e for e in lattice if type(e).__name__ == "Multipole"]
    assert len(sextupoles) == 4
    assert sextupoles[0].to_dict()["multipole"] == 3
    assert sextupoles[0].to_dict()["K_normal"] == pytest.approx(30 * 0.5 / 4)


def test_a_bend_field_error_is_reported(make_tao):
    tao = make_tao("b: sbend, l = 0.5, angle = 0.12, dg = 0.01", "b")
    with pytest.warns(TaoTranslationWarning, match="DG"):
        lattice_from_tao(tao)


# -- other elements --------------------------------------------------------------------


def test_multipoles_use_exact_multipole_at_order_minus_one(make_tao):
    """ImpactX's k_normal is indexed by order - 1; index 1 reproduces a quadrupole."""
    tao = make_tao(
        "sx: sextupole, l = 0.2, k2 = 25.0\noc: octupole, l = 0.15, k3 = 80.0", "sx, oc"
    )
    lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "ExactMultipole", "ExactMultipole", "Marker"]
    assert list(lattice[1].to_dict()["k_normal"]) == [0.0, 0.0, 25.0]
    assert list(lattice[2].to_dict()["k_normal"]) == [0.0, 0.0, 0.0, 80.0]


def test_solenoid_uses_the_chromatic_model(make_tao):
    """ChrAcc, not Sol: `Sol` has no pt dependence and misses Bmad by 1.4e-4.

    `ChrAcc(ez=0)` keeps the exact pt dependence and agrees to 2.9e-9, so a solenoid
    translates without warning. Its `bz` is charge*Bz/(m*c) where Bmad's `KS` is
    charge*Bz/p0c, hence the beta*gamma factor.
    """
    tao = make_tao("s: solenoid, l = 0.3, ks = 0.4", "s", e_tot=100e6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TaoTranslationWarning)
        lattice = lattice_from_tao(tao)

    assert kinds(lattice) == ["Marker", "ChrAcc", "Marker"]
    solenoid = lattice[1]
    gamma = 100e6 / ELECTRON_MASS_EV
    beta_gamma = math.sqrt(gamma * gamma - 1.0)
    assert solenoid.to_dict()["ez"] == 0.0
    assert solenoid.to_dict()["bz"] == pytest.approx(0.4 * beta_gamma, rel=1e-6)


def test_kickers_carry_their_kick(make_tao):
    tao = make_tao("hk: hkicker, kick = 1e-4\nvk: vkicker, kick = -7e-5", "hk, vk")
    lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "Kicker", "Kicker", "Marker"]
    assert lattice[1].xkick == pytest.approx(1e-4)
    assert lattice[1].ykick == 0.0
    assert lattice[2].ykick == pytest.approx(-7e-5)


def test_a_stray_kick_on_another_element_is_reported(make_tao):
    """Bmad lets any element steer; ImpactX has no such field on a quadrupole."""
    tao = make_tao("q: quadrupole, l = 0.3, k1 = 2.0, hkick = 1e-4", "q")
    with pytest.warns(TaoTranslationWarning, match="HKICK"):
        lattice_from_tao(tao)


def test_a_switched_off_element_keeps_its_length(make_tao):
    tao = make_tao("q: quadrupole, l = 0.3, k1 = 2.0, is_on = F", "q")
    with pytest.warns(TaoTranslationWarning, match="switched off"):
        lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "ExactDrift", "Marker"]
    assert lattice[1].ds == pytest.approx(0.3)


def test_collimator_limits_become_an_aperture(make_tao):
    """Bmad's aperture_type and aperture_at decide the shape and the placement."""
    tao = make_tao("ec: ecollimator, l = 0.2, x_limit = 0.01, y_limit = 0.008", "ec")
    lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "ExactDrift", "Aperture", "Marker"]
    aperture = lattice[2]
    assert aperture.to_dict()["shape"] == "elliptical"
    assert aperture.aperture_x == pytest.approx(0.01)
    assert aperture.aperture_y == pytest.approx(0.008)


def test_an_asymmetric_aperture_is_reported(make_tao):
    """ImpactX apertures are centred, so an off-centre one is too permissive."""
    tao = make_tao("rc: rcollimator, l = 0.2, x1_limit = 0.01, x2_limit = 0.02", "rc")
    with pytest.warns(TaoTranslationWarning, match="asymmetric"):
        lattice = lattice_from_tao(tao)
    assert lattice[2].to_dict()["shape"] == "rectangular"
    assert lattice[2].aperture_x == pytest.approx(0.02)


def test_rfcavity_normalises_the_voltage_by_the_lattice_species_mass(make_tao):
    """ShortRF takes energy gain / (m c^2), so the mass must come from the lattice."""
    tao = make_tao(
        "c: rfcavity, l = 0.4, voltage = 1e5, rf_frequency = 1.3e9, phi0 = 0.1",
        "c",
        particle="proton",
        e_tot=2e9,
    )
    with pytest.warns(TaoTranslationWarning, match="ShortRF"):
        lattice = lattice_from_tao(tao, nslice=2)
    cavities = [e for e in lattice if type(e).__name__ == "ShortRF"]
    assert len(cavities) == 2
    proton_mass_eV = 938.27208816e6
    assert cavities[0].V == pytest.approx(1e5 / proton_mass_eV / 2, rel=1e-4)
    assert cavities[0].to_dict()["freq"] == pytest.approx(1.3e9)
    # Bmad's phi0 is in units of 2*pi; ImpactX counts degrees from the crest at 90.
    assert cavities[0].to_dict()["phase"] == pytest.approx(90.0 - 36.0)


def test_an_unsupported_element_raises_and_can_be_skipped(make_tao):
    tao = make_tao("s: sol_quad, l = 0.4, ks = 0.3, k1 = 1.0", "s")
    with pytest.raises(UnsupportedElementError, match="sol_quad"):
        lattice_from_tao(tao)
    with pytest.warns(TaoTranslationWarning, match="Replaced by a marker"):
        lattice = lattice_from_tao(tao, skip_unsupported=True)
    assert kinds(lattice) == ["Marker", "Marker", "Marker"]


def test_lcavity_is_supported_and_carries_the_reference_energy(make_tao):
    """ImpactX *can* change the reference energy -- ShortRF.H:207 does exactly that.

    Bmad's E_tot(exit) = E_tot(in) + VOLTAGE*cos(2*pi*PHI0), so the ImpactX phase is
    360*PHI0 degrees (cos here, where the rfcavity convention uses sin). Measured 9.6e-6
    on a travelling-wave cavity; a standing-wave one is far worse and warns loudly.
    """
    tao = make_tao(
        "c: lcavity, l = 1.0, gradient = 10e6, rf_frequency = 1.3e9, phi0 = 0.1,"
        " cavity_type = traveling_wave",
        "c",
    )
    with pytest.warns(TaoTranslationWarning, match="Rosenzweig-Serafini"):
        lattice = lattice_from_tao(tao, nslice=4)
    cavities = [e for e in lattice if type(e).__name__ == "ShortRF"]
    assert len(cavities) == 4
    assert cavities[0].to_dict()["phase"] == pytest.approx(36.0)


def test_pitch_and_z_offset_are_reported_as_dropped(make_tao):
    tao = make_tao("q: quadrupole, l = 0.3, k1 = 2.0, x_pitch = 1e-4", "q")
    with pytest.warns(TaoTranslationWarning, match="no pitch"):
        lattice_from_tao(tao)


def test_transverse_offsets_and_tilt_come_across(make_tao):
    """Signs verified against the negated alternative, which is wrong by O(1).

    Note Impact-Z needs the opposite tilt sign: the two codes are not interchangeable.
    """
    tao = make_tao(
        "q: quadrupole, l = 0.3, k1 = 2.0, x_offset = 1e-4, y_offset = -2e-4, tilt = 0.05",
        "q",
    )
    quad = lattice_from_tao(tao)[1]
    assert quad.dx == pytest.approx(1e-4)
    assert quad.dy == pytest.approx(-2e-4)
    assert quad.rotation == pytest.approx(math.degrees(0.05))


# -- the LUME model --------------------------------------------------------------------


def test_model_from_tao_builds_a_driveable_model(tao):
    """The one-step Tao to LUME path: translate, track, and generate variables."""
    from lume_impactx import LUMEImpactXModel

    model = LUMEImpactXModel.from_tao(tao, nslice=16)
    names = set(model.supported_variables)

    # Bmad element names come across as-is, and repeats are disambiguated by index.
    assert "ele:QF:k" in names
    assert {"ele:D#1:ds", "ele:D#2:ds"} <= names
    assert "moment_final:sigma_x" in names
    assert {"particles:initial_particles", "particles:final_particles"} <= names
    assert "ref:kin_energy_MeV" in names

    baseline = model.get("moment_final:sigma_x")
    model.set({"ele:QF:k": 1.5})
    assert model.get("moment_final:sigma_x") != pytest.approx(baseline)
    model.reset()
    assert model.get("moment_final:sigma_x") == pytest.approx(baseline)


def test_model_from_tao_matches_building_it_in_two_steps(tao):
    """`from_tao` is exactly `from_simulator(ImpactXSimulator.from_tao(...))`."""
    from lume_impactx import ImpactXSimulator, LUMEImpactXModel

    one_step = LUMEImpactXModel.from_tao(tao, nslice=16)
    two_step = LUMEImpactXModel.from_simulator(
        ImpactXSimulator.from_tao(tao, nslice=16)
    )

    assert set(one_step.supported_variables) == set(two_step.supported_variables)
    assert one_step.get("moment_final:sigma_x") == pytest.approx(
        two_step.get("moment_final:sigma_x"), rel=1e-12
    )


def test_model_from_tao_forwards_translator_options(tao):
    """Keyword arguments reach the lattice translation, not just the simulator."""
    from lume_impactx import LUMEImpactXModel

    model = LUMEImpactXModel.from_tao(tao, nslice=3)
    assert model.simulator.lattice[1].nslice == 3


def test_model_from_tao_honours_dummy_run(tao):
    """With `dummy_run`, `set()` writes but does not re-track until asked."""
    from lume_impactx import LUMEImpactXModel

    model = LUMEImpactXModel.from_tao(tao, nslice=8, dummy_run=True)
    baseline = model.get("moment_final:sigma_x")

    model.set({"ele:QF:k": 1.5})
    assert model.get("moment_final:sigma_x") == pytest.approx(baseline)

    model.simulator.track()
    assert model.get("moment_final:sigma_x") != pytest.approx(baseline)


# -- regressions -----------------------------------------------------------------------
#
# Each of these reproduces a translation that was silently wrong: it produced a lattice
# with no warning and tracked far away from Bmad.


def test_a_zero_angle_bend_keeps_its_k1(make_tao):
    """Regression: it became a drift, discarding K1 entirely -- 100% wrong."""
    tao = make_tao("b: sbend, l = 0.5, g = 0, k1 = 1.7", "b")
    lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "ChrQuad", "Marker"]
    assert lattice[1].k == pytest.approx(1.7)


def test_a_switched_off_bend_is_not_a_drift(make_tao):
    """Regression: track_a_bend.f90:90-94 zeroes g_tot but keeps g, so the particle
    goes straight through a still-curved coordinate system. Calling that a drift
    tracked 100% wrong, with a warning that read as reassurance."""
    tao = make_tao("b: sbend, l = 0.5, angle = 0.12, is_on = F", "b")
    with pytest.raises(UnsupportedElementError, match="not a drift"):
        lattice_from_tao(tao)


def test_a_zero_length_collimator_keeps_its_aperture(make_tao):
    """Regression: the marker shortcut ran before apertures were computed, so the
    canonical thin collimator translated to a bare Marker."""
    tao = make_tao("c: rcollimator, x_limit = 1e-4, y_limit = 1e-4", "c")
    lattice = lattice_from_tao(tao)
    assert "Aperture" in kinds(lattice)
    aperture = next(e for e in lattice if type(e).__name__ == "Aperture")
    assert aperture.aperture_x == pytest.approx(1e-4)


def test_a_half_limited_aperture_does_not_invent_a_metre(make_tao):
    """Bmad falls back to bmad_com%max_aperture_limit (1000 m), not 1 m."""
    tao = make_tao("c: rcollimator, l = 0.2, x_limit = 1e-3", "c")
    aperture = next(e for e in lattice_from_tao(tao) if type(e).__name__ == "Aperture")
    assert aperture.aperture_x == pytest.approx(1e-3)
    assert aperture.aperture_y == pytest.approx(1000.0)


def test_multipole_error_tables_are_reported(make_tao):
    """Regression: A_n/B_n were never read. `has#ab_multipoles` is True for every
    quadrupole, so the live values must come from ele_multipoles()['data']."""
    tao = make_tao("q: quadrupole, l = 0.3, k1 = 2.0, a2 = 5.0", "q")
    with pytest.warns(TaoTranslationWarning, match="multipole error terms"):
        lattice_from_tao(tao)


def test_a_plain_quadrupole_does_not_claim_multipole_errors(make_tao):
    tao = make_tao("q: quadrupole, l = 0.3, k1 = 2.0", "q")
    with warnings.catch_warnings():
        warnings.simplefilter("error", TaoTranslationWarning)
        lattice_from_tao(tao)


def test_quadrupole_fringe_uses_quadedge(make_tao):
    """QuadEdge is Bmad's quad fringe, not just a similar name: ImpactX's
    a = +-(-k/12)/(1+delta) is Bmad's hard_multipole_edge_kick at n=1. Body alone
    5.0e-7; with both edges 1.9e-14."""
    tao = make_tao("q: quadrupole, l = 0.3, k1 = 2.0, fringe_type = full", "q")
    lattice = lattice_from_tao(tao)
    assert kinds(lattice) == ["Marker", "QuadEdge", "ChrQuad", "QuadEdge", "Marker"]
    assert lattice[1].to_dict()["flag"] == "entry"
    assert lattice[3].to_dict()["flag"] == "exit"


def test_superposition_does_not_truncate_or_duplicate_the_lattice(make_tao):
    """Regression: `-no_slaves` keeps lords while bare indices address the tracking
    branch, so the loop counted one set and walked another. A superposed lattice came
    out 1.6 m of 2.6 m, or 3.2 m with a super-lord translated twice."""
    tao = make_tao(
        "q: quadrupole, l = 0.6, k1 = 1.0\n"
        "m1: marker, superimpose, ref = q, offset = -0.1\n"
        "m2: marker, superimpose, ref = q, offset = 0.1\n"
        "d: drift, l = 1.0",
        "q, d",
    )
    lattice = lattice_from_tao(tao)
    length = sum(getattr(e, "ds", 0.0) or 0.0 for e in lattice)
    assert length == pytest.approx(1.6)  # 0.6 quad (in slices) + 1.0 drift


# -- tracking against Bmad -------------------------------------------------------------


@pytest.fixture
def track_both(tmp_path, monkeypatch):
    """Track one bunch through Bmad and through the translation, and compare.

    The structural tests above check that the right elements come out; these check
    that the physics agrees, which is the claim that actually matters.
    """
    from beamphysics import ParticleGroup

    from lume_impactx import ImpactXSimulator

    counter = itertools.count()

    def run(body: str, line: str, nslice: int = 32, e_tot: float = 100e6) -> float:
        n = 64
        rng = np.random.default_rng(7)
        p0c = math.sqrt(e_tot**2 - ELECTRON_MASS_EV**2)
        bunch = ParticleGroup(
            data={
                "x": rng.normal(0, 3e-4, n),
                "y": rng.normal(0, 3e-4, n),
                "z": np.zeros(n),
                "px": rng.normal(0, 2e-5, n) * p0c,
                "py": rng.normal(0, 2e-5, n) * p0c,
                "pz": p0c * (1 + rng.normal(0, 5e-4, n)),
                "t": rng.normal(0, 1e-13, n),
                "status": np.ones(n, dtype=int),
                "weight": np.full(n, 1e-12),
                "species": "electron",
            }
        )
        directory = tmp_path / f"trk{next(counter)}"
        directory.mkdir()
        bunch.write(str(directory / "beam.h5"))
        (directory / "lat.bmad").write_text(
            f"parameter[geometry] = open\nparameter[particle] = electron\n"
            f"parameter[e_tot] = {e_tot}\nbeginning[beta_a] = 10\n"
            f"beginning[beta_b] = 10\nd1: drift, l = 0.4\n{body}\n"
            f"lat: line = ({line})\nuse, lat\n"
        )
        (directory / "tao.init").write_text(
            "&tao_start\n/\n&tao_design_lattice\n"
            '  design_lattice(1)%file = "lat.bmad"\n/\n'
            "&tao_beam_init\n  beam_init%position_file = 'beam.h5'\n"
            f"  beam_init%n_particle = {n}\n"
            # Bmad silently re-centres and re-scales the input bunch otherwise, which
            # fakes a ~5e-5 error that has nothing to do with the translation.
            "  beam_init%renorm_center = F\n  beam_init%renorm_sigma = F\n/\n"
        )
        monkeypatch.chdir(directory)
        instance = pytao.Tao(init_file="tao.init", noplot=True)
        instance.cmd("set global track_type = beam")
        instance.cmd("set beam saved_at = *")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulator = ImpactXSimulator.from_tao(instance, nslice=nslice)
        reference, translated = instance.particles("END"), simulator.final_particles
        return max(
            np.abs(np.sort(reference[key]) - np.sort(translated[key])).max()
            / max(np.abs(reference[key]).max(), 1e-30)
            for key in ("x", "y", "px", "py", "t")
        )

    return run


@pytest.mark.parametrize(
    ("label", "body", "line", "tolerance"),
    [
        ("drift", "", "d1", 1e-12),
        ("quadrupole", "q: quadrupole, l = 0.3, k1 = 2.0", "d1, q, d1", 1e-12),
        ("bend with pole faces", BEND, "d1, b, d1", 1e-8),
        (
            "bend, default fringe, with fint/hgap",
            f"{BEND}, fint = 0.5, hgap = 0.03",
            "d1, b, d1",
            1e-8,
        ),
        (
            "zero-angle bend with k1",
            "b: sbend, l = 0.5, g = 0, k1 = 1.7",
            "d1, b, d1",
            1e-12,
        ),
        ("solenoid", "s: solenoid, l = 0.3, ks = 0.4", "d1, s, d1", 1e-8),
        (
            "quadrupole with fringe",
            "q: quadrupole, l = 0.3, k1 = 2.0, fringe_type = full",
            "d1, q, d1",
            1e-12,
        ),
        (
            "rfcavity followed by a quadrupole",
            "c: rfcavity, l = 0.4, voltage = 5e6, rf_frequency = 1.3e9, phi0 = 0.25\n"
            "q: quadrupole, l = 0.3, k1 = 2.0",
            "d1, c, q, d1",
            1e-6,
        ),
        (
            "lcavity followed by a quadrupole",
            "c: lcavity, l = 1.0, gradient = 10e6, rf_frequency = 1.3e9, phi0 = 0,"
            " cavity_type = traveling_wave, fringe_type = none\n"
            "q: quadrupole, l = 0.3, k1 = 2.0",
            "d1, c, q, d1",
            1e-4,
        ),
    ],
)
def test_tracking_matches_bmad(track_both, label, body, line, tolerance):
    """Every accuracy claim in this module, actually asserted."""
    worst = track_both(body, line)
    # A comparison that returns exactly zero means the two sides were never really
    # compared -- guard against the harness silently degenerating.
    assert 0.0 < worst < tolerance, f"{label}: {worst:.3e}"
