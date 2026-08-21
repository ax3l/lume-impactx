"""Archiving and the openPMD BeamMonitor reader."""

from __future__ import annotations

import numpy as np
import pytest

from lume_impactx.archive import (
    archive,
    dicts_to_lattice,
    element_to_dict,
    load_archive,
)


def test_degree_elements_survive_a_dict_round_trip():
    """The bug this guards: to_dict() gives radians, from_dicts() expects degrees.

    Without ``in_degrees=True`` an ExactSbend built with phi=30 deg comes back as
    0.0091 rad instead of 0.5236 rad -- a silent factor of 57.3.
    """
    from impactx import elements

    original = elements.ExactSbend(name="b", ds=1.0, phi=30.0)
    rebuilt = dicts_to_lattice([element_to_dict(original)])[0]
    assert rebuilt.phi == pytest.approx(np.deg2rad(30.0), rel=1e-12)
    assert rebuilt.phi == pytest.approx(original.phi, rel=1e-12)


def test_mixed_lattice_round_trips():
    from impactx import elements

    lattice = [
        elements.Drift(name="d", ds=1.0, nslice=3),
        elements.Quad(name="q", ds=0.5, k=2.0, nslice=4),
        elements.Marker(name="mk"),
        elements.ThinDipole(name="td", theta=15.0, rc=10.0),
        elements.Aperture(name="ap", aperture_x=1e-3, aperture_y=2e-3),
        elements.BeamMonitor("mon", backend="h5"),
    ]
    rebuilt = dicts_to_lattice([element_to_dict(e) for e in lattice])
    assert len(rebuilt) == len(lattice)
    for before, after in zip(lattice, rebuilt):
        assert type(after).__name__ == type(before).__name__
        assert getattr(after, "name", None) == getattr(before, "name", None)
    assert rebuilt[1].k == pytest.approx(2.0)
    assert rebuilt[1].nslice == 4
    assert rebuilt[4].aperture_y == pytest.approx(2e-3)


def test_elements_outlive_the_temporary_list():
    """dicts_to_lattice returns a plain list; pybind11 keep-alive must hold."""
    import gc

    lattice = dicts_to_lattice(
        [{"type": "Quad", "name": "q", "ds": 1.0, "k": 2.0, "nslice": 4}]
    )
    gc.collect()
    lattice[0].k = 7.0
    assert lattice[0].k == pytest.approx(7.0)


def test_rejects_a_foreign_file(tmp_path):
    import h5py

    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["dataType"] = "something-else"
    with pytest.raises(ValueError, match="Not a lume-impactx archive"):
        load_archive(path)


@pytest.mark.slow
def test_archive_round_trip(fodo_simulator, tmp_path):
    path = tmp_path / "fodo.h5"
    archive(fodo_simulator, path)
    restored = load_archive(path)

    assert [type(e).__name__ for e in restored.lattice] == [
        type(e).__name__ for e in fodo_simulator.lattice
    ]
    assert restored.ref == fodo_simulator.ref
    assert restored.settings == fodo_simulator.settings
    assert restored.n_steps == fodo_simulator.n_steps

    before, after = fodo_simulator.results, restored.results
    assert after["moments"]["sigma_x"] == pytest.approx(
        before["moments"]["sigma_x"], rel=1e-15
    )
    assert after["n_particles"] == before["n_particles"]
    np.testing.assert_allclose(
        after["moments_history"]["sigma_x"],
        before["moments_history"]["sigma_x"],
        rtol=1e-15,
    )
    assert after["final_particles"].n_particle == before["final_particles"].n_particle
    assert after["final_particles"]["sigma_x"] == pytest.approx(
        before["final_particles"]["sigma_x"], rel=1e-12
    )


@pytest.mark.slow
def test_a_restored_archive_serves_a_model(fodo_simulator, tmp_path):
    """Reading an archive needs no ImpactX run: the model answers from cached results."""
    from lume_impactx.model import LUMEImpactXModel

    path = tmp_path / "fodo.h5"
    archive(fodo_simulator, path)
    restored = load_archive(path)

    model = LUMEImpactXModel.from_simulator(restored)
    original = LUMEImpactXModel.from_simulator(fodo_simulator)
    assert set(model.supported_variables) == set(original.supported_variables)
    assert model.get("moment_final:sigma_x") == pytest.approx(
        original.get("moment_final:sigma_x"), rel=1e-15
    )
    assert model.get("ele:quad1:k") == pytest.approx(1.0)


@pytest.mark.slow
def test_archive_of_a_particle_seeded_simulator_can_retrack(
    fodo_lattice, fodo_simulator, tmp_path
):
    from lume_impactx.simulator import ImpactXSimulator

    seeded = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        initial_particles=fodo_simulator.final_particles,
    )
    path = tmp_path / "seeded.h5"
    archive(seeded, path)

    restored = load_archive(path, track=True)
    assert restored.track_count == 1
    assert restored.results["moments"]["sigma_x"] == pytest.approx(
        seeded.results["moments"]["sigma_x"], rel=1e-9
    )


@pytest.mark.slow
def test_distribution_seeded_archive_cannot_retrack(fodo_simulator, tmp_path):
    path = tmp_path / "fodo.h5"
    archive(fodo_simulator, path)
    with pytest.raises(ValueError, match="seeded from a distribution"):
        load_archive(path, track=True)


@pytest.mark.slow
def test_beam_monitor_reader_matches_the_in_memory_bunch(tmp_path, monkeypatch):
    """The openPMD path and the in-memory path must agree on the same step."""
    import lume_impactx  # noqa: F401  -- MPI bootstrap
    from impactx import ImpactX, distribution, elements

    from lume_impactx.utils import (
        particle_container_to_particlegroup,
        read_beam_monitor,
    )

    monkeypatch.chdir(tmp_path)
    sim = ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.space_charge = False
    sim.slice_step_diagnostics = False
    sim.diagnostics = True
    sim.init_grids()
    sim.beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    sim.add_particles(
        1e-9,
        distribution.Waterbag(
            lambdaX=4e-5,
            lambdaY=4e-5,
            lambdaT=1e-3,
            lambdaPx=2.7e-5,
            lambdaPy=2.7e-5,
            lambdaPt=2e-3,
        ),
        500,
    )
    monitor = elements.BeamMonitor("monitor", backend="h5")
    sim.lattice.extend([monitor, elements.Drift(name="d", ds=1.0, nslice=5), monitor])
    sim.track_particles()
    in_memory = particle_container_to_particlegroup(sim.beam)
    sim.finalize()

    from_file = read_beam_monitor(str(tmp_path / "diags" / "openPMD" / "monitor.h5"))

    assert from_file.n_particle == in_memory.n_particle
    assert from_file.species == in_memory.species
    assert from_file.in_z_coordinates
    for stat in ["sigma_x", "sigma_y", "norm_emit_x", "mean_energy"]:
        assert from_file[stat] == pytest.approx(in_memory[stat], rel=1e-12), stat
    assert from_file.charge == pytest.approx(in_memory.charge, rel=1e-12)


@pytest.mark.slow
def test_archive_preserves_linear_optics(fodo_simulator, tmp_path):
    """A restored simulator must generate the same optics:* variables as the original."""
    path = tmp_path / "optics.h5"
    archive(fodo_simulator, path)
    restored = load_archive(path)

    before, after = fodo_simulator.results, restored.results
    for key in ("transfer_map", "cumulative_maps", "map_s"):
        np.testing.assert_allclose(after[key], before[key], rtol=1e-15, err_msg=key)
    assert after["map_names"] == before["map_names"]
