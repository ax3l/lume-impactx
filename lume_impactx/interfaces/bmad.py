"""Translate a Bmad/Tao model into an ImpactX simulation.

The **beam** side is faithful: the reference particle and the bunch come straight from
Tao, and the bunch travels as an openPMD-beamphysics ``ParticleGroup``, which pytao
already produces and which :mod:`lume_impactx.utils` converts exactly.

The **lattice** side is a bridge, not a translation, and is deliberately noisy about it.
It routes Bmad -> MAD-X -> ImpactX using each code's own importer/exporter, so it
carries only what all three represent. See :func:`lattice_from_tao` for what is dropped.

Examples
--------
>>> from pytao import Tao
>>> from lume_impactx import ImpactXSimulator
>>> tao = Tao(init_file="tao.init", noplot=True)
>>> tao.cmd("set global track_type = beam")
>>> tao.cmd("set beam saved_at = *")
>>> sim = ImpactXSimulator.from_tao(tao)
>>> sim.track()
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from typing import Any

try:
    from beamphysics import ParticleGroup
    from beamphysics.species import mass_of
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup
    from pmd_beamphysics.species import mass_of

__all__ = [
    "TaoTranslationWarning",
    "particles_from_tao",
    "reference_from_tao",
    "beam_from_tao",
    "lattice_from_tao",
    "simulator_from_tao",
]


class TaoTranslationWarning(UserWarning):
    """Raised when part of a Bmad model cannot be carried into ImpactX."""


# --------------------------------------------------------------------------------------
# Beam
# --------------------------------------------------------------------------------------


def particles_from_tao(tao: Any, ele: str = "BEGINNING") -> ParticleGroup:
    """Read the tracked bunch at one element as a ``ParticleGroup``.

    Parameters
    ----------
    tao : pytao.Tao
        A Tao instance that has already tracked a beam.
    ele : str
        Element to read at, as Tao names it. Must be in ``beam saved_at``.

    Returns
    -------
    ParticleGroup
        In z-coordinates, which is what the ImpactX converter expects.

    Raises
    ------
    RuntimeError
        If Tao has no tracked bunch at ``ele``, with the commands needed to get one.
    """
    try:
        return tao.particles(ele)
    except Exception as exc:
        raise RuntimeError(
            f"Tao has no tracked bunch at {ele!r}. A beam has to be tracked, and the "
            "element has to be saved, before it can be translated:\n"
            "    tao.cmd('set global track_type = beam')\n"
            f"    tao.cmd('set beam saved_at = {ele}')   # or '*' for every element\n"
            "    tao.track_beam()\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc


def reference_from_tao(
    tao: Any, ele: str = "BEGINNING", species: str | None = None
) -> dict[str, Any]:
    """Build the reference-particle specification for :class:`ImpactXSimulator`.

    The energy is Bmad's **design** reference energy ``E_TOT`` at ``ele``, not the
    bunch's mean. That is the faithful choice: ImpactX phase-space coordinates are
    offsets from the reference particle, so using the design reference reproduces
    Bmad's own ``pz`` offsets rather than re-centring the bunch.

    Parameters
    ----------
    tao : pytao.Tao
        The Tao instance.
    ele : str
        Element to take the reference at.
    species : str, optional
        openPMD-beamphysics species name. Taken from the tracked bunch when omitted,
        since Bmad exposes no lattice-level species query through pytao.

    Returns
    -------
    dict
        ``{"species": ..., "kin_energy_MeV": ...}``, ready to pass as ``ref=``.
    """
    if species is None:
        species = str(particles_from_tao(tao, ele).species)

    total_energy_eV = float(tao.ele_gen_attribs(ele)["E_TOT"])
    rest_mass_eV = mass_of(species)
    kinetic_eV = total_energy_eV - rest_mass_eV
    if kinetic_eV <= 0.0:
        raise ValueError(
            f"Bmad reports E_TOT = {total_energy_eV:.6e} eV at {ele!r}, which is below "
            f"the {species} rest mass ({rest_mass_eV:.6e} eV)."
        )
    return {"species": species, "kin_energy_MeV": kinetic_eV / 1.0e6}


def beam_from_tao(
    tao: Any, ele: str = "BEGINNING", species: str | None = None
) -> tuple[dict[str, Any], ParticleGroup]:
    """Return ``(reference_spec, particles)`` for one element.

    Convenience wrapper over :func:`reference_from_tao` and :func:`particles_from_tao`
    that reads the bunch once.
    """
    particles = particles_from_tao(tao, ele)
    reference = reference_from_tao(tao, ele, species=species or str(particles.species))
    return reference, particles


# --------------------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------------------

#: Tao writes a *labelled* beam definition terminated by ";;", which ImpactX's MAD-X
#: parser reads as an empty species and then rejects. Matching it lets us substitute a
#: form the parser accepts.
_TAO_BEAM_DEFINITION = re.compile(r"(?is)^\s*\w+\s*:\s*beam\b.*?;;", re.M)


def _normalize_tao_madx(text: str, species: str, kin_energy_MeV: float) -> str:
    """Rewrite Tao's MAD-X beam definition into the form ImpactX's parser accepts.

    Without this the load fails outright:
    ``ValueError: Unknown MAD-X particle species requires explicit MASS and CHARGE``.
    Dropping the definition does not help -- the parser requires a BEAM command -- so
    one is synthesised from values already taken from Tao.
    """
    total_GeV = (kin_energy_MeV + mass_of(species) / 1.0e6) / 1.0e3
    beam_line = f"beam, particle={species.lower()}, energy={total_GeV!r};"
    replaced, count = _TAO_BEAM_DEFINITION.subn(beam_line, text, count=1)
    if count == 0:
        replaced = beam_line + "\n" + text
    return replaced


def lattice_from_tao(
    tao: Any,
    nslice: int = 1,
    min_model: str = "exact",
    species: str | None = None,
    kin_energy_MeV: float | None = None,
    ele: str = "BEGINNING",
) -> list:
    """Translate the Tao lattice into ImpactX elements, via MAD-X.

    .. warning::

       This is a **bridge, not a translation**, and it goes through two importers that
       are themselves works in progress -- ImpactX warns that its MAD-X parser is "under
       active development and provided as a preview". Check the result against the Bmad
       model before trusting any number that comes out of it.

       What is **not** carried, silently, by one hop or the other:

       * **Numerics control.** ``nslice`` is applied uniformly here; Bmad's per-element
         integrator choice, ``num_steps``, ``ds_step`` and tracking/mat6 methods have no
         MAD-X representation. TODO: map these per element once the direct translator
         exists.
       * Element types MAD-X cannot express: taylor maps, wigglers/undulators,
         ``patch`` elements, ``em_field``, and Bmad's ``overlay``/``group``/``girder``
         control structures.
       * Multipole error tables, aperture definitions, fringe-field models and
         higher-order edge effects.
       * Multi-branch lattices: only the tracked branch is written.

    Parameters
    ----------
    tao : pytao.Tao
        The Tao instance.
    nslice : int
        Slices per thick element, applied uniformly.
    min_model : str
        Lowest ImpactX element-model tier to use -- ``"linear"``, ``"paraxial"`` or
        ``"exact"``. Requires an ImpactX new enough to accept it; older builds always
        use linear models, and a warning says so.
    species, kin_energy_MeV : optional
        Used to synthesise the MAD-X BEAM command. Taken from Tao when omitted.
    ele : str
        Element the reference is taken at, when it has to be looked up.

    Returns
    -------
    list
        ImpactX elements, ready to hand to :class:`ImpactXSimulator`.
    """
    from impactx import elements

    if species is None or kin_energy_MeV is None:
        reference = reference_from_tao(tao, ele, species=species)
        species = species or reference["species"]
        kin_energy_MeV = (
            kin_energy_MeV
            if kin_energy_MeV is not None
            else reference["kin_energy_MeV"]
        )

    warnings.warn(
        "Translating a Bmad lattice through MAD-X. Per-element numerics control "
        "(integrator, num_steps, ds_step, tracking method) is NOT carried, and "
        f"nslice={nslice} is applied uniformly. Verify the result against Bmad.",
        TaoTranslationWarning,
        stacklevel=2,
    )

    with tempfile.TemporaryDirectory() as workdir:
        written = os.path.join(workdir, "tao_export.madx")
        normalized = os.path.join(workdir, "impactx_ready.madx")
        tao.cmd(f"write madx {written}")
        if not os.path.exists(written):
            raise RuntimeError(
                f"Tao did not write {written!r}. 'write madx' is required to bridge the "
                "lattice; check the Tao version and that the path is writable."
            )
        with open(written) as handle:
            text = handle.read()
        with open(normalized, "w") as handle:
            handle.write(_normalize_tao_madx(text, species, kin_energy_MeV))

        known = elements.KnownElementsList()
        try:
            known.load_file(normalized, nslice, min_model=min_model)
        except TypeError:
            # min_model landed after 26.08; older builds always use linear models.
            warnings.warn(
                f"This ImpactX does not accept min_model={min_model!r}, so the lattice "
                "was loaded with linear element models. Exact models need a newer "
                "ImpactX.",
                TaoTranslationWarning,
                stacklevel=2,
            )
            known.load_file(normalized, nslice)
        return list(known)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def simulator_from_tao(
    tao: Any,
    ele: str = "BEGINNING",
    lattice: list | None = None,
    nslice: int = 1,
    min_model: str = "exact",
    species: str | None = None,
    settings: dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Build an :class:`ImpactXSimulator` from a Tao model.

    Parameters
    ----------
    tao : pytao.Tao
        A Tao instance with a tracked beam saved at ``ele``.
    ele : str
        Element to take the beam and reference particle from.
    lattice : list, optional
        ImpactX elements to use. When omitted the Bmad lattice is bridged through
        MAD-X -- read :func:`lattice_from_tao` before relying on that.
    nslice, min_model : see :func:`lattice_from_tao`
    species : str, optional
        Overrides the species taken from the tracked bunch.
    settings : dict, optional
        ImpactX settings for the simulator.
    **kwargs
        Passed to :class:`ImpactXSimulator`.

    Returns
    -------
    ImpactXSimulator
    """
    from lume_impactx.simulator import ImpactXSimulator

    reference, particles = beam_from_tao(tao, ele, species=species)
    if lattice is None:
        lattice = lattice_from_tao(
            tao,
            nslice=nslice,
            min_model=min_model,
            species=reference["species"],
            kin_energy_MeV=reference["kin_energy_MeV"],
            ele=ele,
        )
    return ImpactXSimulator(
        lattice=lattice,
        ref=reference,
        initial_particles=particles,
        settings=settings,
        **kwargs,
    )
