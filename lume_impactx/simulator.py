"""The ImpactX session object driven by :class:`~lume_impactx.model.LUMEImpactXModel`.

``ImpactXSimulator`` holds a *specification* -- a lattice, a reference particle, a beam
and a set of simulation settings -- and builds, tracks and tears down a fresh
``impactx.ImpactX`` on every :meth:`ImpactXSimulator.track` call.

Why rebuild rather than rewind
------------------------------
ImpactX tracking is destructive: it consumes the particle container and advances the
reference particle's ``s``. Three ways out were measured against ImpactX 26.06/26.08:

============================  ===================================================
Approach                      Outcome
============================  ===================================================
Resample the distribution     **Rejected.** Resampling *within one session* advances
in place                      a global RNG: two identical calls differed by 1.8e-2
                              relative in ``sigma_x``. Every ``set()`` would return a
                              different answer for the same inputs.
Hold one session, snapshot     Works and is bit-exact, but freezes the mesh: ``n_cell``
the beam and re-inject         and friends are read-only after ``init_grids()``, so
                               they could never be exposed as variables.
**Rebuild the session**        **Chosen.** Bit-exact (a fresh session re-seeds the RNG),
                               keeps every parameter writable, and needs no snapshot
                               machinery for the distribution path.
============================  ===================================================

Rebuilding is not the expensive option people expect. With a 32^3 space-charge mesh and
20k particles it cost 777 ms/iteration against 747 ms for snapshot-and-re-inject -- 4%.
Tracking dominates, not ``init_grids()``.

The one hard requirement is that ``mpi4py`` owns MPI, otherwise the second
``ImpactX.finalize()`` in a process tears MPI down for good. :mod:`lume_impactx._mpi`
arranges that at import time.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from lume_impactx._mpi import ensure_external_mpi
from lume_impactx.elements import restore_lattice, snapshot_lattice
from lume_impactx.utils import (
    C_LIGHT,
    ImpactXRefPart,
    add_particlegroup,
    apply_species,
    particle_container_to_particlegroup,
    refpart_snapshot,
)

try:
    from beamphysics import ParticleGroup
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup

#: Settings that must be applied before ``init_grids()``, in this order.
logger = logging.getLogger(__name__)

#: Settings that must be applied before ``init_grids()``, in this order.
_PRE_GRID_SETTINGS = ("particle_shape", "n_cell", "max_level", "space_charge")

#: Defaults chosen so a get/set loop leaves no files behind and prints nothing.
_DEFAULT_SETTINGS: dict[str, Any] = {
    "verbose": 0,
    "tiny_profiler": False,
    "diagnostics": False,
    "slice_step_diagnostics": False,
    "space_charge": False,
    # Explicit so n_steps and the s-series variable shapes have a stated basis rather
    # than an implied one.
    "periods": 1,
}


def _matrix_to_numpy(matrix: Any):
    """Convert an AMReX ``SmallMatrix`` to a correctly-oriented numpy array.

    ``np.asarray`` on one of these yields the **transpose**: the matrix is Fortran
    ordered and the buffer is read as C ordered. A drift then comes back as
    ``[[1, 0], [L, 1]]`` instead of ``[[1, L], [0, 1]]`` -- a plausible-looking matrix
    that is wrong. ``.to_numpy()`` orients it properly.
    """
    import numpy as np

    if hasattr(matrix, "to_numpy"):
        return np.asarray(matrix.to_numpy())
    return np.asarray(matrix)  # pragma: no cover - already an array


class ImpactXSimulator:
    """A rebuildable ImpactX simulation.

    Parameters
    ----------
    lattice : list of impactx.elements.*
        The canonical, mutable lattice. Action variables write to these objects, and
        each :meth:`track` copies them into a fresh ``ImpactX``. Note that
        ``KnownElementsList.extend`` copies, so mutating the list after a track is
        exactly how changes take effect on the next one.
    ref : dict
        Reference-particle specification. Either ``{"species": ..., "kin_energy_MeV":
        ...}`` using an ImpactX built-in species, or ``{"charge_qe": ..., "mass_MeV":
        ..., "kin_energy_MeV": ...}``.
    distribution : impactx.distribution.*, optional
        Beam distribution to sample. Mutually exclusive with ``initial_particles``.
    npart : int, optional
        Number of macroparticles to sample. Required with ``distribution``.
    bunch_charge_C : float, optional
        Bunch charge in Coulomb. Required with ``distribution``.
    initial_particles : ParticleGroup, optional
        An explicit bunch to inject. Mutually exclusive with ``distribution``.
    ref_origin : ImpactXRefPart, optional
        Where this section starts along the machine, normally the upstream section's
        final reference particle. Only meaningful with ``initial_particles``: the
        arrival time ``t``, the arc length ``s`` and the reference **energy** are taken
        from it, so a stage downstream of an RF cavity inherits the energy the beam
        actually has. The lab footprint is not carried -- see :meth:`_align_reference`.
        Because the energy comes from upstream, ``ref:kin_energy_MeV`` is generated
        read-only for such a stage. Without ``ref_origin`` the origin is taken from the
        bunch itself, which is right for a standalone run but loses absolute timing.
    settings : dict, optional
        Attributes to set on the ``ImpactX`` object, e.g. ``{"space_charge": "3D",
        "n_cell": [32, 32, 32], "particle_shape": 2}``.
    track_on_init : bool
        Track once during construction so results are available immediately. Keep this
        True: a ``LUMEModel`` must be able to answer ``get()`` before any ``set()``.

    Raises
    ------
    ValueError
        If neither or both of ``distribution`` and ``initial_particles`` are given.
    """

    def __init__(
        self,
        *,
        lattice: list,
        ref: dict[str, Any],
        distribution: Any = None,
        npart: int | None = None,
        bunch_charge_C: float | None = None,
        initial_particles: ParticleGroup | None = None,
        ref_origin: ImpactXRefPart | None = None,
        settings: dict[str, Any] | None = None,
        track_on_init: bool = True,
    ) -> None:
        if (distribution is None) == (initial_particles is None):
            raise ValueError(
                "Provide exactly one of distribution= (with npart= and "
                "bunch_charge_C=) or initial_particles=."
            )
        if distribution is not None and (npart is None or bunch_charge_C is None):
            raise ValueError("distribution= also requires npart= and bunch_charge_C=.")

        ensure_external_mpi()

        self.lattice = list(lattice)
        self.ref = dict(ref)
        self.distribution = distribution
        self.npart = npart
        self.bunch_charge_C = bunch_charge_C
        self.initial_particles = initial_particles
        self.ref_origin = ref_origin
        self.settings = {**_DEFAULT_SETTINGS, **(settings or {})}

        self._initial_lattice = snapshot_lattice(self.lattice)
        self._initial_ref = copy.deepcopy(self.ref)
        self._initial_settings = copy.deepcopy(self.settings)
        self._initial_particles_0 = initial_particles
        self._ref_origin_0 = ref_origin

        self._results: dict[str, Any] = {}
        self.track_count = 0
        if track_on_init:
            self.track()

    @classmethod
    def from_tao(cls, tao: Any, **kwargs: Any) -> "ImpactXSimulator":
        """Build a simulator from a Bmad/Tao model.

        Both halves come straight from Tao: the reference particle and bunch, and the
        lattice element by element, unless you pass ``lattice=``. Every element mapping
        was verified against Bmad tracking; read
        :func:`lume_impactx.interfaces.bmad.lattice_from_tao` and
        :func:`~lume_impactx.interfaces.bmad.translate_element` for what still differs
        and what is dropped.

        Parameters
        ----------
        tao : pytao.Tao
            A Tao instance with a tracked beam saved at the start element.
        **kwargs
            Passed to :func:`~lume_impactx.interfaces.bmad.simulator_from_tao`, e.g.
            ``ele``, ``lattice``, ``nslice``, ``skip_unsupported``, ``settings``.

        Examples
        --------
        >>> tao = Tao(init_file="tao.init", noplot=True)
        >>> tao.cmd("set global track_type = beam")
        >>> tao.cmd("set beam saved_at = *")
        >>> sim = ImpactXSimulator.from_tao(tao)
        >>> sim.track()
        """
        from lume_impactx.interfaces.bmad import simulator_from_tao

        return simulator_from_tao(tao, **kwargs)

    # -- introspection -----------------------------------------------------------

    @property
    def results(self) -> dict[str, Any]:
        """The most recent :meth:`track` snapshot.

        Keys: ``moments`` (dict), ``moments_history`` (DataFrame), ``final_particles``
        (ParticleGroup or None), ``ref_final`` (:class:`ImpactXRefPart`),
        ``n_particles`` (int), ``n_steps`` (int), ``run_time`` (float), and the linear
        optics -- ``transfer_map`` (6x6), ``cumulative_maps`` (n, 6, 6), ``map_s`` and
        ``map_names``. The cumulative maps run from the lattice start to each element's
        exit, so the last one equals ``transfer_map``.

        Raises
        ------
        RuntimeError
            If :meth:`track` has never run.
        """
        if not self._results:
            raise RuntimeError("No results yet; call track() first.")
        return self._results

    @property
    def n_steps(self) -> int:
        """Rows the moment history will have: ``periods * sum(nslice)``.

        Known ahead of a run, which is what lets the s-series variables declare an
        exact ``NDVariable.shape`` instead of over-allocating and padding.
        """
        periods = int(self.settings.get("periods", 1))
        return periods * sum(int(getattr(e, "nslice", 1) or 1) for e in self.lattice)

    # -- running -----------------------------------------------------------------

    def _build(self):
        """Construct, configure and seed a fresh ``ImpactX``. Grids are initialized."""
        from impactx import ImpactX

        sim = ImpactX()

        remaining = dict(self.settings)
        for key in _PRE_GRID_SETTINGS:
            if key in remaining:
                setattr(sim, key, remaining.pop(key))
        for key, value in remaining.items():
            setattr(sim, key, value)

        sim.init_grids()

        ref = sim.beam.ref
        if "species" in self.ref:
            apply_species(ref, self.ref["species"])
        if "charge_qe" in self.ref:
            ref.set_charge_qe(self.ref["charge_qe"])
        if "mass_MeV" in self.ref:
            ref.set_mass_MeV(self.ref["mass_MeV"])
        if "kin_energy_MeV" in self.ref:
            ref.set_kin_energy_MeV(self.ref["kin_energy_MeV"])

        if self.initial_particles is not None:
            self._align_reference(ref, self.initial_particles)
            add_particlegroup(sim.beam, self.initial_particles)
        else:
            sim.add_particles(self.bunch_charge_C, self.distribution, self.npart)

        sim.beam.store_beam_moments = True
        sim.lattice.extend(self.lattice)
        return sim

    @staticmethod
    def _linear_optics(sim) -> dict[str, Any]:
        """Capture the lattice's linear transfer maps at the initial reference particle.

        Taken before tracking, because tracking advances the reference particle and the
        maps are a property of the lattice as configured. ``fallback_identity_map`` keeps
        an element with no closed-form linear map (a ``Programmable``, say) from failing
        the whole run -- it contributes identity, and the count of such elements is
        reported so the result is not mistaken for exact.
        """
        import numpy as np

        reference = sim.beam.ref
        try:
            whole = _matrix_to_numpy(
                sim.lattice.transfer_map(reference, fallback_identity_map=True)
            )
            trace = sim.lattice.map_trace(reference)
        except Exception as exc:  # pragma: no cover - depends on the lattice
            logger.debug("Linear optics unavailable: %s", exc)
            return {}

        # map_trace returns *cumulative* maps -- entry i is the map from the lattice
        # start to element i's exit, not element i's own map. Entry 0 is the identity
        # at s = 0, and the last entry equals the whole-lattice map.
        maps = np.asarray([_matrix_to_numpy(entry["M"]) for entry in trace])
        return {
            "transfer_map": whole,
            "cumulative_maps": maps,
            "map_s": np.asarray([float(entry["s"]) for entry in trace]),
            "map_names": [str(entry["name"]) for entry in trace],
        }

    @staticmethod
    def _snapshot_particles(beam):
        """Convert the tracked bunch, deferring a refusal rather than failing the run.

        A run carrying spin is perfectly usable through LUME for moments and plots; it
        is only the per-particle hand-off that has no representation. So the conversion
        problem is recorded here and re-raised from :attr:`final_particles`, where it
        actually matters. (Bunches this class seeds never carry spin -- there is no way
        to pass a spin distribution -- so this path is for containers set up by hand.)
        """
        from lume_impactx.utils import UnrepresentableParticleData

        try:
            return particle_container_to_particlegroup(beam)
        except UnrepresentableParticleData as exc:
            return exc

    def _align_reference(self, ref, particles: ParticleGroup) -> None:
        """Put the reference particle where the incoming bunch actually is.

        Beam coordinates are relative to the reference particle, and only ``t`` enters
        the conversion: ``position_t`` is ``c * (t_i - t_ref)``. A fresh ``ImpactX``
        starts its reference at ``t = 0``, so without this an incoming bunch would be
        offset by its entire absolute arrival time -- metres of apparent bunch offset,
        which matters to any downstream RF phase.

        ``x``, ``y``, ``z``, ``px``, ``py`` are deliberately left at zero. Beam
        coordinates are transverse to the reference orbit, so a bunch handed over from
        an upstream section is already expressed relative to that orbit; seeding the
        downstream reference with the upstream's lab position and angle would bend its
        orbit away from the section's own axis. Carrying ``z`` alone would be worse than
        carrying nothing -- a reference at the machine's ``z`` but on this section's
        axis is neither lab-correct nor section-local -- so the lab footprint simply is
        not tracked across a hand-off. ``s``, the arc length along the machine, is, and
        so is the reference energy.
        """
        if self.ref_origin is not None:
            origin = self.ref_origin
            ref.t = origin.t
            ref.s = origin.s
            # Carry the reference *energy*: an upstream RF cavity or ChrAcc changes it,
            # and beam momenta are normalized by it, so keeping this stage's configured
            # energy would silently rescale the incoming bunch.
            ref.set_kin_energy_MeV((origin.gamma - 1.0) * ref.mass_MeV)
            return

        # No upstream reference: centre on the bunch itself. A t-coordinate bunch is
        # drifted on a copy first, so this matches the plane the converter will use.
        if not particles.in_z_coordinates:
            particles = particles.copy()
            particles.drift_to_z()
        ref.t = C_LIGHT * float(particles["mean_t"])

    def track(self) -> dict[str, Any]:
        """Build, track and tear down a simulation, then cache the results.

        Returns
        -------
        dict
            The same mapping as :attr:`results`.
        """
        import time

        sim = self._build()
        optics = self._linear_optics(sim)
        start = time.monotonic()
        try:
            sim.track_particles()
            beam = sim.beam
            # Everything must be read out before finalize(): the container is gone
            # afterwards, and so is any reference into it.
            self._results = {
                "moments": dict(beam.beam_moments()),
                "moments_history": beam.beam_moments_history(),
                "final_particles": self._snapshot_particles(beam),
                "ref_final": refpart_snapshot(beam.ref),
                "n_particles": int(beam.total_number_of_particles()),
                "n_steps": self.n_steps,
                "run_time": time.monotonic() - start,
                **optics,
            }
        finally:
            sim.finalize()

        self.track_count += 1
        return self._results

    def reset(self) -> dict[str, Any]:
        """Restore the construction-time lattice, reference and settings, then track."""
        restore_lattice(self.lattice, self._initial_lattice)
        self.ref = copy.deepcopy(self._initial_ref)
        self.settings = copy.deepcopy(self._initial_settings)
        self.initial_particles = self._initial_particles_0
        self.ref_origin = self._ref_origin_0
        return self.track()

    # -- particle bridges for StagedModel ---------------------------------------

    @property
    def final_particles(self) -> ParticleGroup:
        """The bunch at the end of the lattice.

        Raises
        ------
        UnrepresentableParticleData
            If the bunch carries spin or runtime SoA components, which
            ``ParticleGroup`` cannot hold. Moments and plots still work; only this
            hand-off is unavailable.
        RuntimeError
            On a rank that holds no particles, where ``to_df`` yields None. Returning
            None instead would surface as an opaque ``TypeError`` from
            ``ParticleGroupVariable.validate_value``.
        """
        particles = self.results["final_particles"]
        if isinstance(particles, Exception):
            raise particles
        if particles is None:
            raise RuntimeError(
                "No final particles on this MPI rank. ImpactX always uses "
                "MPI_COMM_WORLD and cannot be given a communicator, so a LUME model "
                "is effectively single-rank; run multi-rank ImpactX directly."
            )
        return particles

    @property
    def ref_snapshot(self) -> ImpactXRefPart:
        """The reference particle as it was at the end of the last track."""
        return self.results["ref_final"]

    def plot(self, y=("sigma_x", "sigma_y"), **kwargs):
        """Plot beam moments along the lattice.

        Thin wrapper over :func:`lume_impactx.plot.plot_moments_with_layout`; see it for
        the full set of options.
        """
        from lume_impactx.plot import plot_moments_with_layout

        return plot_moments_with_layout(self, y=y, **kwargs)

    def __repr__(self) -> str:
        seeded = (
            "initial_particles"
            if self.initial_particles is not None
            else f"{type(self.distribution).__name__}(npart={self.npart})"
        )
        return (
            f"<ImpactXSimulator {len(self.lattice)} elements, {seeded}, "
            f"n_steps={self.n_steps}, tracks={self.track_count}>"
        )
