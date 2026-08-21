"""ImpactX beam data <-> ParticleGroup.

`ImpactX <https://impactx.readthedocs.io>`_ is an s-based beam dynamics code. Its
particles are held at a common ``s`` with a spread in arrival time, which is
z-coordinates on this side -- all ``z`` equal, ``t`` varying -- so the conversion is a
direct algebraic map, like the Bmad interface and unlike the time-based ones.

Generated from lume-impactx; see https://github.com/lume-science/lume-impactx.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..particles import ParticleGroup
from ..species import charge_of, e_charge, mass_of

C_LIGHT = 299792458.0

__all__ = [
    "ImpactXRefPart",
    "particlegroup_to_impactx",
    "impactx_to_particlegroup_data",
    "pmd_species_of",
    "refpart_from_openpmd",
    "read_beam_monitor",
]


#: ImpactX's built-in species names mapped to openPMD-beamphysics names.
#: ImpactX only knows these four; anything else needs explicit mass and charge.
IMPACTX_TO_PMD_SPECIES = {
    "electron": "electron",
    "positron": "positron",
    "proton": "proton",
    "Hminus": "H-",
}
PMD_TO_IMPACTX_SPECIES = {v: k for k, v in IMPACTX_TO_PMD_SPECIES.items()}

# --------------------------------------------------------------------------------------
# Pure core (upstreamable: numpy + beamphysics only, no impactx import)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactXRefPart:
    """An ImpactX reference particle, detached from any live ``ImpactX`` session.

    Holding this as a plain dataclass rather than wrapping ``impactx.RefPart`` is what
    lets the converters run with no ImpactX object in the process -- which the openPMD
    reader and most of the test suite rely on.

    Attributes
    ----------
    x, y, z : float
        Lab-frame position of the reference particle, in metres.
    t : float
        ``c * t`` of the reference particle, in **metres** (ImpactX convention).
    px, py, pz : float
        Momenta normalized by ``m * c``, i.e. ``beta_i * gamma``. Dimensionless.
    pt : float
        ``-gamma`` of the reference particle. Dimensionless.
    mass_MeV : float
        Rest mass in MeV.
    charge_qe : float
        Charge in units of the elementary charge, e.g. -1 for an electron.
    s : float
        Integrated path length along the reference orbit, in metres.
    """

    x: float
    y: float
    z: float
    t: float
    px: float
    py: float
    pz: float
    pt: float
    mass_MeV: float
    charge_qe: float
    s: float = 0.0

    @property
    def mass_eV(self) -> float:
        """Rest mass in eV."""
        return self.mass_MeV * 1.0e6

    @property
    def gamma(self) -> float:
        """Relativistic gamma of the reference particle."""
        return -self.pt

    @property
    def beta_gamma(self) -> float:
        """Magnitude of the normalized reference momentum."""
        return float(np.sqrt(self.px**2 + self.py**2 + self.pz**2))

    @property
    def qm_eV(self) -> float:
        """Charge over mass in 1/eV, the form ``add_n_particles`` expects."""
        return self.charge_qe / self.mass_eV

    @property
    def qm_SI(self) -> float:
        """Charge over mass in C/kg, the form ``to_df()["qm"]`` reports."""
        return self.qm_eV * C_LIGHT**2


def particlegroup_to_impactx(pg: ParticleGroup, ref: ImpactXRefPart) -> dict:
    """Convert a ``ParticleGroup`` to ImpactX fixed-s beam arrays.

    Parameters
    ----------
    pg : ParticleGroup
        The bunch to convert. Unless it already sits exactly on the reference plane it
        is copied and drifted to ``ref.z``, so the input is never mutated.
    ref : ImpactXRefPart
        The reference particle the ImpactX coordinates are relative to.

    Returns
    -------
    dict
        Keys ``position_x``, ``position_y``, ``position_t``, ``momentum_x``,
        ``momentum_y``, ``momentum_t`` (arrays), ``weighting`` (array, real particles
        per macroparticle), ``qm`` (scalar, 1/eV) and ``species`` (str). These map
        one-to-one onto ``ImpactXParticleContainer.add_n_particles``.
    """
    # The bunch must occupy a single plane. A t-coordinate bunch (spread in z) is
    # drifted to its own mean z on a copy, so the input is never mutated. Note this is
    # deliberately the bunch's own plane, not some lab z: in the local frame the plane
    # *is* the reference particle's location.
    if not pg.in_z_coordinates:
        pg = pg.copy()
        pg.drift_to_z()

    mass_eV = ref.mass_eV
    beta_gamma = ref.beta_gamma

    position_x = pg.x
    position_y = pg.y
    position_t = C_LIGHT * pg.t - ref.t

    momentum_x = pg.px / mass_eV / beta_gamma
    momentum_y = pg.py / mass_eV / beta_gamma

    # gamma - gamma_ref, written to avoid cancellation: both are ~4000 for a 2 GeV beam,
    # and their difference is ~1e-3, so the naive form loses about 3.5 digits.
    p_mc2 = (pg.p / mass_eV) ** 2
    gamma = np.sqrt(1.0 + p_mc2)
    dgamma = (p_mc2 - beta_gamma**2) / (gamma + ref.gamma)
    momentum_t = -dgamma / beta_gamma

    return {
        "position_x": position_x,
        "position_y": position_y,
        "position_t": position_t,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "momentum_t": momentum_t,
        "weighting": pg.weight / abs(charge_of(pg.species)),
        "qm": ref.qm_eV,
        "species": pg.species,
    }


def impactx_to_particlegroup_data(
    data: dict,
    ref: ImpactXRefPart,
    species: str | None = None,
) -> dict:
    """Convert ImpactX fixed-s beam arrays to ``ParticleGroup`` data.

    The inverse of :func:`particlegroup_to_impactx`.

    Parameters
    ----------
    data : dict
        Arrays keyed as ``ImpactXParticleContainer.to_df()`` names them:
        ``position_x/y/t``, ``momentum_x/y/t``, ``weighting``. An ``id`` key is carried
        through if present.
    ref : ImpactXRefPart
        The reference particle the ImpactX coordinates are relative to.
    species : str, optional
        openPMD-beamphysics species name. Inferred from ``ref`` when omitted.

    Returns
    -------
    dict
        Suitable for ``ParticleGroup(data=...)``: ``x``, ``y``, ``z`` in metres,
        ``px``, ``py``, ``pz`` in eV/c, ``t`` in seconds, ``weight`` in Coulomb,
        ``status`` and ``species``. The result is in z-coordinates: every ``z`` is
        ``ref.z``, and the bunch length shows up as a spread in ``t``.
    """
    if species is None:
        species = pmd_species_of(ref)

    mass_eV = ref.mass_eV
    beta_gamma = ref.beta_gamma
    n = len(np.asarray(data["position_x"]))

    # In the local frame the reference particle is the origin: its transverse momentum
    # is zero by construction and its longitudinal momentum is |p_ref|. ref.px / ref.pz
    # are *lab* components and must not be mixed in here.
    px_mc = beta_gamma * np.asarray(data["momentum_x"])
    py_mc = beta_gamma * np.asarray(data["momentum_y"])
    # ref.gamma is -ref.pt, so this is gamma_ref + (gamma - gamma_ref) = gamma.
    gamma = ref.gamma - beta_gamma * np.asarray(data["momentum_t"])
    pz_mc = np.sqrt(gamma**2 - 1.0 - px_mc**2 - py_mc**2)

    pg_data = {
        "x": np.asarray(data["position_x"]),
        "y": np.asarray(data["position_y"]),
        "z": np.zeros(n),
        "px": px_mc * mass_eV,
        "py": py_mc * mass_eV,
        "pz": pz_mc * mass_eV,
        "t": (ref.t + np.asarray(data["position_t"])) / C_LIGHT,
        "weight": np.asarray(data["weighting"]) * abs(charge_of(species)),
        "status": np.ones(n, dtype=int),
        "species": species,
    }
    if "id" in data:
        pg_data["id"] = np.asarray(data["id"])
    return pg_data


def pmd_species_of(ref: ImpactXRefPart, rtol: float = 1e-6) -> str:
    """Infer the openPMD-beamphysics species name from a reference particle.

    Parameters
    ----------
    ref : ImpactXRefPart
        The reference particle.
    rtol : float
        Relative tolerance for the mass match. ImpactX and openPMD-beamphysics carry
        electron masses that differ in the 9th digit, so this cannot be exact.

    Returns
    -------
    str
        A species name such as ``"electron"``.

    Raises
    ------
    ValueError
        If no known species matches; pass ``species=`` explicitly in that case.
    """
    for pmd_name in IMPACTX_TO_PMD_SPECIES.values():
        charge_matches = np.isclose(
            ref.charge_qe, charge_of(pmd_name) / e_charge, rtol=rtol
        )
        mass_matches = np.isclose(ref.mass_eV, mass_of(pmd_name), rtol=rtol)
        if charge_matches and mass_matches:
            return pmd_name
    raise ValueError(
        f"Cannot infer a species from charge_qe={ref.charge_qe} and "
        f"mass_MeV={ref.mass_MeV}. Pass species= explicitly."
    )


#: ImpactX SoA columns that this module maps onto ParticleGroup fields. Anything else a
#: container holds -- ``spin_x/y/z`` with ``sim.spin`` on, or a runtime component added
#: with ``add_real_comp`` -- has no ParticleGroup representation.
_MAPPED_SOA_COLUMNS = frozenset(
    {
        "idcpu",
        "position_x",
        "position_y",
        "position_t",
        "momentum_x",
        "momentum_y",
        "momentum_t",
        "qm",
        "weighting",
    }
)

#: The spin components, which ImpactX always allocates. They stay at exactly zero
#: unless the beam was seeded with a spin distribution -- ``sim.spin = True`` alone is
#: not enough, the gate is the ``spin_distr`` argument to ``add_particles``. So testing
#: for "any non-zero" is exact: zero means there is genuinely nothing to lose.
#:
#: ImpactXSimulator has no way to pass a spin distribution, so a bunch it produces
#: always converts. The guard matters for containers built by hand and for
#: :func:`read_beam_monitor` reading a file from a spin-seeded run.
SPIN_COLUMNS = ("spin_x", "spin_y", "spin_z")


class UnrepresentableParticleData(NotImplementedError):
    """Raised when a bunch carries per-particle data ``ParticleGroup`` cannot hold.

    Converting anyway would return a bunch that looks right and has silently lost
    physics, so the conversion refuses instead.
    """


def _check_representable(columns: dict) -> None:
    """Refuse to convert a bunch whose extra per-particle data would be dropped.

    Parameters
    ----------
    columns : dict
        Per-particle arrays keyed by ImpactX SoA name.

    Raises
    ------
    UnrepresentableParticleData
        If any spin component is non-zero, or any unmapped component is present.
    """
    carries_spin = any(
        name in columns and np.any(np.asarray(columns[name]) != 0.0)
        for name in SPIN_COLUMNS
    )
    runtime = sorted(
        name
        for name in columns
        if name not in _MAPPED_SOA_COLUMNS and name not in SPIN_COLUMNS
    )
    if not carries_spin and not runtime:
        return

    carried = []
    if carries_spin:
        carried.append("spin (spin_x/y/z)")
    if runtime:
        carried.append(f"runtime components {runtime}")
    raise UnrepresentableParticleData(
        f"This bunch carries {' and '.join(carried)}, which openPMD-beamphysics' "
        "ParticleGroup cannot represent, and nothing in LUME analyses, chains or plots "
        "today. Converting would silently drop it. Work with the ImpactX container "
        "directly for these quantities. Spin *moments* are unaffected and are exposed "
        "as ordinary variables when sim.spin is on."
    )


# --------------------------------------------------------------------------------------
# openPMD BeamMonitor reader
#
# ImpactX's elements.BeamMonitor writes standard openPMD with the species always named
# "beam" ("particles_lost" for the loss monitor), records position/{x,y,t},
# momentum/{x,y,t}, weighting, qm and id, all with unitSI == 1, plus the reference
# particle as per-iteration species attributes. Verified against 26.08.
#
# This path needs no live ImpactX object, which is what makes ImpactXRefPart a plain
# dataclass rather than a wrapper around impactx.RefPart.
# --------------------------------------------------------------------------------------

#: Elementary charge in Coulomb, for decoding ``charge_ref`` / ``mass_ref``.
E_CHARGE_C = 1.602176634e-19

#: openPMD records that carry information already mapped onto ParticleGroup fields, or
#: that are bookkeeping rather than per-particle data. Everything else in a species is
#: collected only so the refusal can name it. ``positionOffset`` is required by the
#: openPMD standard and
#: is all-zero for ImpactX, so it must not become an extra.
OPENPMD_STANDARD_RECORDS = frozenset(
    {
        "position",
        "positionOffset",
        "momentum",
        "weighting",
        "qm",
        "id",
        "charge",
        "mass",
        "particleStatus",
        "particlePatches",
        "time",
        "timeOffset",
    }
)


def refpart_from_openpmd(species: Any) -> ImpactXRefPart:
    """Rebuild a reference particle from a BeamMonitor species' attributes.

    Parameters
    ----------
    species : openpmd_api.ParticleSpecies
        A species from a BeamMonitor iteration.

    Returns
    -------
    ImpactXRefPart
        ``mass_ref`` is stored in kg and ``charge_ref`` in Coulomb, so both are
        converted here to the MeV / elementary-charge units the converters use.
    """
    get = species.get_attribute
    return ImpactXRefPart(
        x=get("x_ref"),
        y=get("y_ref"),
        z=get("z_ref"),
        t=get("t_ref"),
        px=get("px_ref"),
        py=get("py_ref"),
        pz=get("pz_ref"),
        pt=get("pt_ref"),
        mass_MeV=get("mass_ref") * C_LIGHT**2 / E_CHARGE_C / 1.0e6,
        charge_qe=get("charge_ref") / E_CHARGE_C,
        s=get("s_ref"),
    )


def read_beam_monitor(
    path: str,
    iteration: int | None = None,
    species_name: str = "beam",
    species: str | None = None,
) -> ParticleGroup:
    """Read an ImpactX ``BeamMonitor`` openPMD file into a ``ParticleGroup``.

    Parameters
    ----------
    path : str
        Path to the file ImpactX wrote, e.g. ``diags/openPMD/monitor.h5``.
    iteration : int, optional
        Which iteration to read. The last one when omitted.
    species_name : str
        openPMD species to read. ImpactX writes ``"beam"``, and ``"particles_lost"``
        for the loss monitor.
    species : str, optional
        openPMD-beamphysics species name; inferred from the reference particle when
        omitted.

    Returns
    -------
    ParticleGroup
        In z-coordinates, matching what
        :func:`particle_container_to_particlegroup` returns for the same step.

    Raises
    ------
    KeyError
        If the requested iteration or species is not in the file.
    UnrepresentableParticleData
        If the monitor recorded spin or runtime components.
    """
    try:
        import openpmd_api as io
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Reading BeamMonitor output needs the openpmd-api Python package. The "
            "conda-forge impactx package pulls it in; with the impactx-noacc wheel "
            "install it explicitly, e.g. pip install 'lume-impactx[impactx]'."
        ) from exc

    series = io.Series(path, io.Access.read_only)
    iterations = list(series.iterations)
    if not iterations:
        raise KeyError(f"No iterations in {path!r}.")
    if iteration is None:
        iteration = iterations[-1]
    elif iteration not in iterations:
        raise KeyError(f"Iteration {iteration} not in {path!r}; have {iterations}.")

    particles = series.iterations[iteration].particles
    if species_name not in particles:
        raise KeyError(
            f"Species {species_name!r} not in {path!r}; have {list(particles)}."
        )
    beam = particles[species_name]

    chunks = {
        "position_x": beam["position"]["x"].load_chunk(),
        "position_y": beam["position"]["y"].load_chunk(),
        "position_t": beam["position"]["t"].load_chunk(),
        "momentum_x": beam["momentum"]["x"].load_chunk(),
        "momentum_y": beam["momentum"]["y"].load_chunk(),
        "momentum_t": beam["momentum"]["t"].load_chunk(),
        "weighting": beam["weighting"][io.Record_Component.SCALAR].load_chunk(),
    }

    # A monitor writes spin when sim.spin is on, and any runtime SoA component too.
    # ParticleGroup cannot hold those, so collect them only to refuse loudly rather
    # than hand back a bunch that has quietly lost them.
    unrepresentable = {}
    for name in beam:
        if name in OPENPMD_STANDARD_RECORDS:
            continue
        for component, values in beam[name].items():
            key = (
                name
                if component == io.Record_Component.SCALAR
                else f"{name}_{component}"
            )
            unrepresentable[key] = values.load_chunk()

    ref = refpart_from_openpmd(beam)
    series.flush()

    _check_representable({k: np.asarray(v) for k, v in unrepresentable.items()})
    data = {key: np.asarray(value) for key, value in chunks.items()}
    return ParticleGroup(data=impactx_to_particlegroup_data(data, ref, species=species))
