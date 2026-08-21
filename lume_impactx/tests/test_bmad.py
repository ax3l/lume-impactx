"""Bmad/Tao translation.

Skipped unless pytao is installed. It is not a dependency: Bmad comes from conda-forge
(`conda install -c conda-forge bmad pytao`) and most users of this package never need it.

The lattice fixture is a copy of lume-bmad's FODO cell, so both packages are exercised
against the same model.
"""

from __future__ import annotations

import pathlib

import pytest

pytao = pytest.importorskip("pytao", reason="Bmad/pytao is not installed")

from lume_impactx.interfaces.bmad import (  # noqa: E402
    TaoTranslationWarning,
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


def test_lattice_bridge_warns_about_what_it_drops(tao):
    with pytest.warns(TaoTranslationWarning, match="numerics control"):
        lattice = lattice_from_tao(tao, nslice=5)
    assert [type(e).__name__ for e in lattice] == [
        "Quad",
        "Drift",
        "Quad",
        "Drift",
        "Marker",
    ]
    quad = lattice[0]
    assert quad.ds == pytest.approx(0.25)
    assert quad.nslice == 5


def test_tao_madx_beam_definition_is_normalized():
    """Tao writes a labelled `x: Beam, ...;;`, which ImpactX reads as an empty species.

    Without rewriting it the load fails outright with "Unknown MAD-X particle species
    requires explicit MASS and CHARGE"; dropping it does not help either, because the
    parser requires a BEAM command.
    """
    from lume_impactx.interfaces.bmad import _normalize_tao_madx

    tao_output = (
        "// Bmad lattice file: fodo.bmad;\n\n"
        "beam_def: Beam, Particle = Electron, Energy =  6.0E-003, Npart = \n"
        "     0.0E+000;;\n\n"
        "QF: quadrupole, l = 0.25, k1 = 1.2;\n"
    )
    normalized = _normalize_tao_madx(tao_output, "electron", 5.489001049)
    assert "beam_def" not in normalized
    assert "particle=electron" in normalized
    assert "QF: quadrupole" in normalized


# -- end to end ------------------------------------------------------------------------


@pytest.mark.slow
def test_impactx_reproduces_bmad_tracking(tao):
    """The translated model must track to the same beam Bmad does.

    The beam translation is exact -- energy spread matches to machine precision. The
    transverse residuals are the lattice-model difference, since the MAD-X bridge
    carries no per-element numerics control.
    """
    from lume_impactx import ImpactXSimulator

    with pytest.warns(TaoTranslationWarning):
        simulator = ImpactXSimulator.from_tao(tao, nslice=20)

    bmad_end = tao.particles("END")
    impactx_end = simulator.final_particles

    assert impactx_end.n_particle == bmad_end.n_particle
    assert impactx_end.charge == pytest.approx(bmad_end.charge, rel=1e-12)
    # exact: nothing in the beam hand-off touches the energy distribution
    assert impactx_end["sigma_energy"] == pytest.approx(
        bmad_end["sigma_energy"], rel=1e-12
    )
    assert impactx_end["mean_energy"] == pytest.approx(
        bmad_end["mean_energy"], rel=1e-12
    )
    # approximate: the lattice went through MAD-X
    assert impactx_end["sigma_x"] == pytest.approx(bmad_end["sigma_x"], rel=1e-3)
    assert impactx_end["sigma_y"] == pytest.approx(bmad_end["sigma_y"], rel=1e-3)
    assert impactx_end["norm_emit_x"] == pytest.approx(
        bmad_end["norm_emit_x"], rel=1e-3
    )


def test_min_model_is_passed_through_when_supported(tao, monkeypatch):
    """`min_model` selects the element-model tier: exact gives ExactQuad/ExactDrift.

    It landed after ImpactX 26.08, so older builds raise TypeError and we fall back to
    linear models with a warning. This checks the call is made, without needing a build
    new enough to honour it.
    """
    from impactx import elements

    seen = {}
    original = elements.KnownElementsList.load_file

    def spy(self, filename, nslice=1, **kwargs):
        seen.update(kwargs)
        return original(self, filename, nslice)  # emulate an older build

    monkeypatch.setattr(elements.KnownElementsList, "load_file", spy)
    with pytest.warns(TaoTranslationWarning):
        lattice_from_tao(tao, nslice=3, min_model="exact")
    assert seen.get("min_model") == "exact"


def test_older_impactx_falls_back_to_linear_with_a_warning(tao, monkeypatch):
    from impactx import elements

    original = elements.KnownElementsList.load_file

    def only_positional(self, filename, nslice=1, **kwargs):
        if kwargs:
            raise TypeError(
                "load_file() got an unexpected keyword argument 'min_model'"
            )
        return original(self, filename, nslice)

    monkeypatch.setattr(elements.KnownElementsList, "load_file", only_positional)
    with pytest.warns(TaoTranslationWarning, match="does not accept min_model"):
        lattice = lattice_from_tao(tao, nslice=3, min_model="exact")
    assert [type(e).__name__ for e in lattice][:2] == ["Quad", "Drift"]
