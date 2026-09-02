"""Converter tests.

The pure-core tests need no ImpactX session, so they are fast and carry none of the
process-lifetime hazards described in conftest.
"""

from __future__ import annotations

import numpy as np
import pytest
from beamphysics import ParticleGroup

from lume_impactx.utils import (
    ImpactXRefPart,
    c_light,
    impactx_to_particlegroup_data,
    particlegroup_to_impactx,
    pmd_species_of,
)


def test_qm_units_relate_by_c_squared(electron_ref):
    """to_df() reports qm in C/kg while add_n_particles wants 1/eV."""
    assert electron_ref.qm_SI / c_light**2 == pytest.approx(
        electron_ref.qm_eV, rel=1e-15
    )
    # the value ImpactX itself reports for an electron
    assert electron_ref.qm_SI == pytest.approx(-1.758820008e11, rel=1e-9)


def test_species_inference(electron_ref):
    assert pmd_species_of(electron_ref) == "electron"

    positron = ImpactXRefPart(**{**electron_ref.__dict__, "charge_qe": 1.0})
    assert pmd_species_of(positron) == "positron"

    unknown = ImpactXRefPart(**{**electron_ref.__dict__, "mass_MeV": 42.0})
    with pytest.raises(ValueError, match="Cannot infer a species"):
        pmd_species_of(unknown)


def test_roundtrip_is_exact(bunch, electron_ref):
    """ParticleGroup -> ImpactX -> ParticleGroup must return the same bunch."""
    data = particlegroup_to_impactx(bunch, electron_ref)
    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))

    for key, tol in [
        ("x", 1e-15),
        ("y", 1e-15),
        ("px", 1e-14),
        ("py", 1e-14),
        ("pz", 1e-14),
        ("t", 1e-14),
        ("weight", 1e-15),
    ]:
        np.testing.assert_allclose(
            back[key], bunch[key], rtol=tol, atol=0.0, err_msg=f"{key} round-trip"
        )
    assert back.species == bunch.species


def test_roundtrip_preserves_beam_statistics(bunch, electron_ref):
    data = particlegroup_to_impactx(bunch, electron_ref)
    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))

    for stat in ["sigma_x", "sigma_y", "norm_emit_x", "norm_emit_y", "mean_energy"]:
        assert back[stat] == pytest.approx(bunch[stat], rel=1e-12), stat
    assert back.charge == pytest.approx(bunch.charge, rel=1e-14)


def test_result_is_in_z_coordinates(bunch, electron_ref):
    """ImpactX holds one s-plane, which is z-coordinates on the ParticleGroup side."""
    data = particlegroup_to_impactx(bunch, electron_ref)
    back = ParticleGroup(data=impactx_to_particlegroup_data(data, electron_ref))
    assert back.in_z_coordinates
    assert np.ptp(back.z) == 0.0
    assert np.ptp(back.t) > 0.0


def test_weighting_counts_real_particles(bunch, electron_ref):
    """ImpactX weighting is a particle count; ParticleGroup weight is a charge."""
    data = particlegroup_to_impactx(bunch, electron_ref)
    e_charge = 1.602176634e-19
    np.testing.assert_allclose(data["weighting"], bunch.weight / e_charge, rtol=1e-14)
    assert data["qm"] == pytest.approx(electron_ref.qm_eV, rel=1e-15)


def test_momentum_t_matches_the_plain_difference(bunch, electron_ref):
    """The algebraic identity must agree with the plain gamma - gamma_ref difference.

    Not a precision claim. Measured against a longdouble reference across
    10 MeV..10 GeV and dp/p 1e-3..1e-12, the identity's error ratio to the plain
    difference is 0.12x..1.57x -- noise around unity, worse as often as better, since
    pg.p's float64 representation sets the floor for both. This asserts equivalence and
    that the round trip closes, which is what actually matters.
    """
    data = particlegroup_to_impactx(bunch, electron_ref)

    mass_eV = electron_ref.mass_eV
    gamma = np.sqrt(1.0 + (bunch.p / mass_eV) ** 2)
    naive = -(gamma - electron_ref.gamma) / electron_ref.pz

    # Same answer to within the naive form's own precision ...
    np.testing.assert_allclose(data["momentum_t"], naive, rtol=1e-9)
    # ... but ours round-trips far better than the naive one does.
    back = impactx_to_particlegroup_data(data, electron_ref)
    np.testing.assert_allclose(back["pz"], bunch.pz, rtol=1e-14)


def test_input_particlegroup_is_not_mutated(electron_ref):
    """A t-coordinates input gets drifted on a copy, never in place."""
    rng = np.random.default_rng(0)
    n = 64
    p_ref = electron_ref.beta_gamma * electron_ref.mass_eV
    pg = ParticleGroup(
        data={
            "x": np.zeros(n),
            "y": np.zeros(n),
            "z": rng.normal(0.0, 1e-3, n),  # t-coordinates: spread in z
            "px": np.zeros(n),
            "py": np.zeros(n),
            "pz": np.full(n, p_ref),
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1e-12),
            "species": "electron",
        }
    )
    assert pg.in_t_coordinates
    z_before = pg.z.copy()
    particlegroup_to_impactx(pg, electron_ref)
    np.testing.assert_array_equal(pg.z, z_before)


# --------------------------------------------------------------------------------------
# Independent cross-check against ImpactX's own reference implementation.
#
# These four functions are copied verbatim from
# impactx/examples/initialize_from_array/transformation_utilities.py, which is *not*
# shipped in the impactx package. They take a different route to the same place -- global
# fixed-t -> reference-relative fixed-t -> fixed-s -- so agreeing with them is real
# evidence that the direct algebraic map in utils.py has its signs and factors right.
# --------------------------------------------------------------------------------------


def _example_to_ref_part_t_from_global_t(ref, x, y, z, px, py, pz):
    dx, dy, dz = x - ref.x, y - ref.y, z - ref.z
    dpx = (px - ref.px) / ref.pz
    dpy = (py - ref.py) / ref.pz
    dpz = (pz - ref.pz) / ref.pz
    return dx, dy, dz, dpx, dpy, dpz


def _example_to_s_from_t(ref, dx, dy, dz, dpx, dpy, dpz):
    ref_pz, ref_pt = ref.pz, ref.pt
    denom = ref_pz + ref_pz * dpz
    dxs = dx - ref_pz * dpx * dz / denom
    dys = dy - ref_pz * dpy * dz / denom
    pt = -np.sqrt(
        1 + (ref_pz + ref_pz * dpz) ** 2 + (ref_pz * dpx) ** 2 + (ref_pz * dpy) ** 2
    )
    dt = pt * dz / denom
    dpt = (pt - ref_pt) / ref_pz
    return dxs, dys, dt, dpx, dpy, dpt


def test_agrees_with_impactx_example_transforms(electron_ref):
    """Our direct map must reproduce ImpactX's own two-step example conversion.

    Valid only for an on-axis reference particle: the example normalizes momenta by
    ``ref.pz`` and adds ``ref.x``/``ref.px`` back, which coincides with normalizing by
    ``|p_ref|`` in the local frame exactly when the reference orbit is straight. See
    ``test_bend_preserves_energy_spread`` for the case that separates them.
    """
    rng = np.random.default_rng(7)
    n = 500
    mass_eV = electron_ref.mass_eV
    p_ref = electron_ref.beta_gamma * mass_eV

    # A bunch at fixed t (spread in z) -- what the ImpactX example starts from. Its z is
    # mean-centred because the example's transforms express everything about a reference
    # particle at z = 0 ("It is assumed that the local and global coordinate frames
    # align"), while our converter drifts a t-coordinate bunch to its own mean plane.
    # Without centring, the two land on planes 1.3e-4 m apart and disagree at 7e-10.
    x = rng.normal(0.0, 4e-5, n)
    y = rng.normal(0.0, 4e-5, n)
    z = rng.normal(0.0, 1e-3, n)
    z -= z.mean()
    px = rng.normal(0.0, 1e-5, n) * p_ref
    py = rng.normal(0.0, 1e-5, n) * p_ref
    pz = p_ref * (1.0 + rng.normal(0.0, 2e-3, n))

    # route A: the example's two-step transform, on beta*gamma momenta
    dx, dy, dz, dpx, dpy, dpz = _example_to_ref_part_t_from_global_t(
        electron_ref, x, y, z, px / mass_eV, py / mass_eV, pz / mass_eV
    )
    ex = _example_to_s_from_t(electron_ref, dx, dy, dz, dpx, dpy, dpz)

    # route B: ours, via a ParticleGroup that drift_to_z() puts on one plane
    pg = ParticleGroup(
        data={
            "x": x,
            "y": y,
            "z": z,
            "px": px,
            "py": py,
            "pz": pz,
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1e-12),
            "species": "electron",
        }
    )
    assert pg.in_t_coordinates
    ours = particlegroup_to_impactx(pg, electron_ref)

    for ours_key, expected in [
        ("position_x", ex[0]),
        ("position_y", ex[1]),
        ("position_t", ex[2]),
        ("momentum_x", ex[3]),
        ("momentum_y", ex[4]),
        ("momentum_t", ex[5]),
    ]:
        np.testing.assert_allclose(
            ours[ours_key], expected, rtol=1e-9, atol=1e-18, err_msg=ours_key
        )


@pytest.mark.slow
def test_live_container_roundtrip(impactx_session, bunch, electron_ref):
    """Inject a ParticleGroup into a real ImpactX container and read it back."""
    from lume_impactx.utils import (
        add_particlegroup,
        particle_container_to_particlegroup,
        refpart_snapshot,
    )

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)

    # the fixture's reference particle must match what ImpactX builds for 2 GeV electrons
    live_ref = refpart_snapshot(beam.ref)
    assert live_ref.pz == pytest.approx(electron_ref.pz, rel=1e-12)
    assert live_ref.pt == pytest.approx(electron_ref.pt, rel=1e-12)

    add_particlegroup(beam, bunch)
    assert beam.total_number_of_particles() == len(bunch.x)

    back = particle_container_to_particlegroup(beam)
    assert back is not None
    assert back.species == "electron"
    assert back.in_z_coordinates

    for stat in ["sigma_x", "sigma_y", "norm_emit_x", "norm_emit_y", "mean_energy"]:
        assert back[stat] == pytest.approx(bunch[stat], rel=1e-9), stat
    assert back.charge == pytest.approx(bunch.charge, rel=1e-9)


@pytest.mark.slow
def test_qm_unit_depends_on_insertion_path(impactx_session, bunch):
    """``qm`` is not reported in a single unit, which is why the readers ignore it.

    ``add_n_particles`` stores the 1/eV value it is given verbatim, while
    ``ImpactX.add_particles`` stores C/kg. The two differ by c**2. If this test starts
    failing because the two agree, ImpactX has fixed the inconsistency upstream and the
    note in utils.py can be dropped.
    """
    from impactx import distribution

    from lume_impactx.utils import add_particlegroup, refpart_snapshot

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    ref = refpart_snapshot(beam.ref)

    add_particlegroup(beam, bunch)
    qm_via_add_n = float(beam.to_df(local=True)["qm"].iloc[0])
    assert qm_via_add_n == pytest.approx(ref.qm_eV, rel=1e-9)

    beam.clear_particles()
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    impactx_session.add_particles(
        1.0e-9,
        distribution.Waterbag(
            lambdaX=4e-5,
            lambdaY=4e-5,
            lambdaT=1e-3,
            lambdaPx=2.7e-5,
            lambdaPy=2.7e-5,
            lambdaPt=2e-3,
        ),
        100,
    )
    qm_via_add = float(beam.to_df(local=True)["qm"].iloc[0])
    assert qm_via_add == pytest.approx(ref.qm_SI, rel=1e-9)
    assert qm_via_add / c_light**2 == pytest.approx(qm_via_add_n, rel=1e-9)


# --------------------------------------------------------------------------------------
# Refusing to drop per-particle data ParticleGroup cannot hold.
#
# ImpactX bunches can carry spin and arbitrary runtime SoA components. Nothing in LUME
# analyses, chains or plots those today, so lume-impactx does not carry them -- but it
# must not hand back a silently zeroed bunch either, which would look right and be
# wrong.
# --------------------------------------------------------------------------------------


def _seed(beam, n=64, **extra):
    """Put n minimal particles into a container, with optional spin arrays."""
    from lume_impactx.utils import refpart_snapshot

    zeros = np.zeros(n)
    ref = refpart_snapshot(beam.ref)
    beam.add_n_particles(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        ref.qm_eV,
        w=np.full(n, 1.0e6),
        **extra,
    )
    return n


@pytest.mark.slow
def test_zero_spin_converts_normally(impactx_session):
    """ImpactX always allocates spin; all-zero must not trip the guard."""
    from lume_impactx.utils import particle_container_to_particlegroup

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    n = _seed(beam)

    result = particle_container_to_particlegroup(beam)
    assert isinstance(result, ParticleGroup)
    assert result.n_particle == n


@pytest.mark.slow
def test_spin_refuses_loudly(impactx_session):
    from lume_impactx.utils import (
        UnrepresentableParticleData,
        particle_container_to_particlegroup,
    )

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    n = 64
    _seed(beam, n=n, sx=np.zeros(n), sy=np.zeros(n), sz=np.ones(n))

    with pytest.raises(UnrepresentableParticleData, match="spin"):
        particle_container_to_particlegroup(beam)


@pytest.mark.slow
def test_runtime_component_refuses_loudly(impactx_session):
    from lume_impactx.utils import (
        UnrepresentableParticleData,
        particle_container_to_particlegroup,
    )

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    _seed(beam)
    beam.add_real_comp("my_tag")

    with pytest.raises(UnrepresentableParticleData, match="my_tag"):
        particle_container_to_particlegroup(beam)


@pytest.mark.slow
def test_refusal_names_the_alternative(impactx_session):
    """The message has to say what to do instead, not just what failed."""
    from lume_impactx.utils import (
        UnrepresentableParticleData,
        particle_container_to_particlegroup,
    )

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    n = 64
    _seed(beam, n=n, sx=np.ones(n), sy=np.zeros(n), sz=np.zeros(n))

    with pytest.raises(UnrepresentableParticleData) as excinfo:
        particle_container_to_particlegroup(beam)
    message = str(excinfo.value)
    assert "ImpactX particle container directly" in message
    assert "moments" in message


# --------------------------------------------------------------------------------------
# Bent reference orbits.
#
# ImpactX beam coordinates are transverse to the reference orbit and normalized by
# |p_ref|. The converter used to divide by ref.pz and add ref.x / ref.px back -- lab
# quantities -- which coincides with the correct form only while the orbit is straight.
# Every test above uses a straight lattice, so the error was invisible.
# --------------------------------------------------------------------------------------


def _track_through_bend(phi_deg, extra_elements=(), npart=4000):
    """Track a bunch through a single bend.

    Returns ``(moments, reference, bunch)`` -- all from the *same* run, because two
    separate runs draw independent Monte-Carlo samples and agree only to ~1e-7.
    """
    import lume_impactx  # noqa: F401  -- MPI bootstrap
    from impactx import ImpactX, distribution, elements

    from lume_impactx.utils import (
        particle_container_to_particlegroup,
        refpart_snapshot,
    )

    sim = ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.init_grids()
    sim.beam.ref.set_species("electron").set_kin_energy_MeV(100.0)
    sim.add_particles(
        1e-12,
        distribution.Waterbag(
            lambdaX=1e-4,
            lambdaY=1e-4,
            lambdaT=1e-3,
            lambdaPx=1e-5,
            lambdaPy=1e-5,
            lambdaPt=1e-3,
        ),
        npart,
    )
    sim.lattice.extend(
        [elements.ExactSbend(name="b", ds=1.0, phi=phi_deg, nslice=8), *extra_elements]
    )
    sim.track_particles()
    moments = dict(sim.beam.beam_moments())
    reference = refpart_snapshot(sim.beam.ref)
    bunch = particle_container_to_particlegroup(sim.beam)
    sim.finalize()
    return moments, reference, bunch


@pytest.mark.slow
@pytest.mark.parametrize("phi", [1e-9, 15.0, 30.0, 60.0])
def test_energy_spread_matches_impactx_through_a_bend(phi):
    """The converted energy spread must equal ImpactX's own, at any bend angle.

    ``pt`` is ``-dgamma / |p_ref|``, so ``sigma_energy == sigma_pt * |p_ref| * mc^2``
    exactly. Dividing by ``ref.pz`` instead scaled the result by ``cos(phi)``: 0.966 at
    15 degrees, 0.866 at 30, 0.500 at 60. Both sides come from one run, so this is an
    identity rather than a comparison of two Monte-Carlo samples.
    """
    moments, reference, bunch = _track_through_bend(phi)
    expected = moments["sigma_pt"] * reference.beta_gamma * reference.mass_eV
    assert bunch["sigma_energy"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.slow
def test_bend_transverse_size_matches_impactx():
    """The converted bunch must report the same sigma_x ImpactX does, after a bend."""
    moments, _reference, bunch = _track_through_bend(30.0)
    assert bunch["sigma_x"] == pytest.approx(moments["sigma_x"], rel=1e-9)
    assert bunch["sigma_y"] == pytest.approx(moments["sigma_y"], rel=1e-9)
    # The bunch is centred on the reference orbit, not offset by the orbit's lab x.
    assert abs(bunch["mean_x"]) < 10.0 * bunch["sigma_x"]


@pytest.mark.slow
def test_species_mismatch_is_refused(impactx_session):
    """Injecting a bunch whose species disagrees with the reference silently rescaled it."""
    from lume_impactx.utils import add_particlegroup

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    n = 64
    protons = ParticleGroup(
        data={
            "x": np.zeros(n),
            "y": np.zeros(n),
            "z": np.zeros(n),
            "px": np.zeros(n),
            "py": np.zeros(n),
            "pz": np.full(n, 1.0e9),
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1e-12),
            "species": "proton",
        }
    )
    with pytest.raises(ValueError, match="Species mismatch"):
        add_particlegroup(beam, protons)


@pytest.mark.slow
def test_matching_species_still_injects(impactx_session, bunch):
    from lume_impactx.utils import add_particlegroup

    beam = impactx_session.beam
    beam.ref.set_species("electron").set_kin_energy_MeV(2.0e3)
    add_particlegroup(beam, bunch)
    assert beam.total_number_of_particles() == len(bunch.x)


def test_the_particle_id_keeps_the_rank_not_just_the_id():
    """Only id-and-rank together identify a particle, so both are kept.

    AMReX packs a per-rank counter into bits 24..62 and the originating rank into bits
    0..23. The counter alone repeats across ranks, so using it as the openPMD id would
    hand back colliding ids for any parallel run. On a single rank the rank field is all
    zero, which is why unpacking looks correct until it is run in parallel.
    """
    import numpy as np

    from lume_impactx.utils import particle_id_from_idcpu

    valid_bit = np.uint64(1) << np.uint64(63)
    # The same AMReX counter (5) as seen from two different ranks.
    rank0 = valid_bit | (np.uint64(5) << np.uint64(24)) | np.uint64(0)
    rank1 = valid_bit | (np.uint64(5) << np.uint64(24)) | np.uint64(1)
    lost = (np.uint64(7) << np.uint64(24)) | np.uint64(0)

    ids, valid = particle_id_from_idcpu(np.array([rank0, rank1, lost], dtype=np.uint64))

    assert ids[0] != ids[1], "two ranks must not collide"
    assert valid.tolist() == [True, True, False]
    # Aliveness lives in status, so only that bit is stripped; everything identifying
    # the particle survives, and AMReX' own counter and rank are recoverable.
    assert (int(ids[0]) >> 24, int(ids[0]) & 0xFFFFFF) == (5, 0)
    assert (int(ids[1]) >> 24, int(ids[1]) & 0xFFFFFF) == (5, 1)


def test_the_particle_id_fits_a_particlegroup():
    """Stripping the validity bit is what makes the value fit int64, which is what
    ParticleGroup stores ids in -- the raw idcpu exceeds its positive range."""
    import numpy as np

    from lume_impactx.utils import particle_id_from_idcpu

    packed = np.array(
        [9223372036871553024, 9223372070409207808, 9223372036888330240],
        dtype=np.uint64,
    )
    assert packed.max() > np.iinfo(np.int64).max
    ids, valid = particle_id_from_idcpu(packed)
    assert valid.all()
    assert ids.dtype == np.int64
    assert (ids > 0).all()

    n = len(packed)
    pg = ParticleGroup(
        data={
            "x": np.zeros(n),
            "y": np.zeros(n),
            "z": np.zeros(n),
            "px": np.zeros(n),
            "py": np.zeros(n),
            "pz": np.full(n, 1e8),
            "t": np.zeros(n),
            "status": np.ones(n, dtype=int),
            "weight": np.full(n, 1e-12),
            "species": "electron",
            "id": ids,
        }
    )
    assert np.array_equal(pg.id, ids)
    # The original idcpu is recovered by putting the validity bit back.
    assert np.array_equal(
        pg.id.astype(np.uint64) | (np.uint64(1) << np.uint64(63)), packed
    )
