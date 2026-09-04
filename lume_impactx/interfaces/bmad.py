"""Translate a Bmad/Tao model into an ImpactX simulation.

The **beam** side is faithful: the reference particle and the bunch come straight from
Tao, and the bunch travels as an openPMD-beamphysics ``ParticleGroup``, which pytao
already produces and which :mod:`lume_impactx.utils` converts exactly.

The **lattice** side is a direct, element-by-element translation. Every mapping was
verified against Bmad itself -- by comparing Tao's ``ele_mat6`` in ImpactX's basis and
by tracking the same bunch through both codes -- rather than assumed from the two
manuals. ImpactX's *exact* element models are used throughout, because the paraxial ones
disagree with Bmad at the 5e-5 level.

Anything Bmad represents and ImpactX does not is dropped with a
:class:`TaoTranslationWarning` naming the element and the attribute, so nothing is lost
silently. Elements with length and no verified equivalent raise
:class:`UnsupportedElementError`. See :func:`translate_element` for the element table
and :func:`lattice_from_tao` for the caveats.

Examples
--------
>>> from pytao import Tao
>>> from lume_impactx import ImpactXSimulator, LUMEImpactXModel
>>> tao = Tao(init_file="tao.init", noplot=True)
>>> tao.cmd("set global track_type = beam")
>>> tao.cmd("set beam saved_at = *")

A simulation to track directly:

>>> sim = ImpactXSimulator.from_tao(tao)
>>> sim.track()

Or a LUME model with variables generated, ready for ``get``/``set`` or ``lume-pva``:

>>> model = LUMEImpactXModel.from_tao(tao)
>>> model.set({"ele:QF:k": 1.5})
>>> model.get("moment_final:sigma_x")
"""

from __future__ import annotations

import math
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
    "UnsupportedElementError",
    "beam_from_tao",
    "capture_points_from_tao",
    "lattice_from_tao",
    "lattice_species_from_tao",
    "model_from_tao",
    "particles_from_tao",
    "reference_from_tao",
    "simulator_from_tao",
    "translate_element",
]


class TaoTranslationWarning(UserWarning):
    """Raised when part of a Bmad model cannot be carried into ImpactX."""


class UnsupportedElementError(NotImplementedError):
    """A Bmad element has length and no verified ImpactX equivalent."""


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


def lattice_species_from_tao(tao: Any, branch: int = 0) -> str | None:
    """The species the *lattice* is built for, as an openPMD-beamphysics name.

    This is what Bmad tracks, and what every magnet strength in the lattice is
    normalised to. It is not necessarily what a bunch calls itself: ``tao.particles()``
    takes its species from the beam file's own metadata, so loading an electron file
    into a positron lattice yields a bunch labelled ``electron`` that Bmad nonetheless
    tracks as a positron.

    Returns ``None`` when Tao will not answer, so callers can fall back rather than
    fail.
    """
    try:
        # Universes are 1-based, branches 0-based.
        value = dict(tao.branch1(1, branch)).get("param_particle")
    except Exception:  # pragma: no cover - depends on the pytao build
        return None
    return str(value).lower() if value else None


def _species_for(tao: Any, ele: str, branch: int = 0) -> str:
    """Reconcile the lattice's species with the bunch's own label.

    The lattice wins. A Bmad file that sets no ``parameter[particle]`` defaults to
    **positron**, so feeding it an electron bunch flips the sign of every bend,
    quadrupole and kicker relative to what Bmad tracked -- measured 100% wrong, and
    previously silent, because the translator took the species from the bunch.
    """
    from_lattice = lattice_species_from_tao(tao, branch)
    try:
        from_bunch = str(particles_from_tao(tao, ele).species)
    except Exception:  # pragma: no cover - no tracked beam
        from_bunch = None

    if from_lattice is None:
        if from_bunch is None:
            raise ValueError(
                "Could not determine the species from either the lattice or the bunch."
            )
        _warn(
            "Tao would not report the lattice species; using the bunch's label "
            f"{from_bunch!r}. If the lattice is for another species, every magnet "
            "strength is normalised to it and the translation will be wrong."
        )
        return from_bunch

    if from_bunch is not None and from_bunch.lower() != from_lattice.lower():
        # Refuse rather than pick one. Bmad tracks the bunch's species through a lattice
        # whose magnet strengths are normalised to *its* species, and the translation
        # reproduces neither: measured 100% wrong against Bmad with either choice. A
        # Bmad file that sets no parameter[particle] defaults to positron, so this is
        # easy to hit by accident with an electron beam and no warning at all before.
        raise ValueError(
            f"The bunch is {from_bunch!r} but the lattice is for {from_lattice!r}. "
            "Bmad normalises every magnet strength to the lattice species, so tracking "
            "a different one through it is a setup this translator cannot reproduce -- "
            "measured 100% away from Bmad whichever species is used. Note a Bmad file "
            "that sets no parameter[particle] defaults to positron. Set "
            "parameter[particle] to match the beam, or pass species= explicitly if you "
            "know the two are consistent."
        )
    return from_lattice


def reference_from_tao(
    tao: Any,
    ele: str = "BEGINNING",
    species: str | None = None,
    branch: int = 0,
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
        openPMD-beamphysics species name. Taken from the *lattice* when omitted -- what
        Bmad actually tracked -- not from the bunch's own label. See
        :func:`lattice_species_from_tao`.
    branch : int
        Lattice branch the species is read from.

    Returns
    -------
    dict
        ``{"species": ..., "kin_energy_MeV": ...}``, ready to pass as ``ref=``.
    """
    if species is None:
        species = _species_for(tao, ele, branch)

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
    tao: Any,
    ele: str = "BEGINNING",
    species: str | None = None,
    branch: int = 0,
) -> tuple[dict[str, Any], ParticleGroup]:
    """Return ``(reference_spec, particles)`` for one element.

    Convenience wrapper over :func:`reference_from_tao` and :func:`particles_from_tao`
    that reads the bunch once.

    """
    particles = particles_from_tao(tao, ele)
    reference = reference_from_tao(tao, ele, species=species, branch=branch)
    if species is not None and str(particles.species).lower() != str(species).lower():
        # An explicit species overrides both halves. Without this it would set the
        # reference and leave the bunch's own label alone, and the injection check
        # downstream would reject the pair -- an override that cannot actually be used.
        _warn(
            f"Relabelling the bunch from {particles.species!r} to {species!r}, as "
            "asked. Bmad tracked it as the lattice species, so check that is what you "
            "meant."
        )
        particles.species = species
    return reference, particles


# --------------------------------------------------------------------------------------
# Lattice
#
# Every mapping below was verified by comparing the linear transfer map (Tao's
# ``ele_mat6``, taken into ImpactX's basis) and by tracking the same bunch through both
# codes. The quoted agreement is the worst absolute matrix difference, or the worst
# relative coordinate difference for tracking. ``test_bmad.py`` asserts these:
# its ``test_tracking_matches_bmad`` cases track a bunch through both codes on
# every run, rather than reading the translator's own output back to itself.
# --------------------------------------------------------------------------------------

#: Bmad elements that are geometrically a drift. Collimators are drifts plus an
#: aperture, which :func:`_apertures_for` adds separately, from the same ``*_LIMIT``
#: attributes that every other element also carries.
DRIFT_LIKE_KEYS = frozenset(
    {"drift", "pipe", "monitor", "instrument", "ecollimator", "rcollimator"}
)

#: Marker-like elements whose bunch is worth capturing, matching Impact-Z's
#: ``write_beam_eles=("monitor::*", "marker::*")``.
_CAPTURE_KEYS = frozenset({"marker", "beginning_ele", "monitor", "instrument"})

#: Bmad elements that carry no length and no effect on the beam.
MARKER_LIKE_KEYS = frozenset(
    {"marker", "beginning_ele", "fork", "photon_fork", "fiducial", "null_ele"}
)

#: Zero-length elements that still carry a transfer map. The zero-length fallback
#: below turns an unknown element into a marker, which is right for a genuine marker
#: and wrong for these: a taylor element measured 3.2e-1 away from Bmad that way.
_MAP_AT_ZERO_LENGTH_KEYS = frozenset(
    {"taylor", "match", "patch", "ab_multipole", "multipole", "sad_mult"}
)

#: Structural elements that describe control relationships, not beam optics. Bmad
#: already folds their effect into the attributes of the elements they control, so
#: skipping them is correct rather than lossy.
CONTROL_KEYS = frozenset({"overlay", "group", "girder", "ramper", "feedback"})

#: Multipole order -> index into ImpactX's ``k_normal``/``k_skew`` arrays. Verified
#: against Bmad: index 1 reproduces a quadrupole, so the index *is* the order minus one.
_MULTIPOLE_INDEX = {"sextupole": 2, "octupole": 3}
_MULTIPOLE_STRENGTH = {"sextupole": "K2", "octupole": "K3"}

#: Bmad ``fringe_type`` values, from ``bmad_struct.f90:170``.
_FRINGE_TYPES = frozenset(
    {
        "none",
        "soft_edge_only",
        "hard_edge_only",
        "full",
        "sad_full",
        "linear_edge",
        "basic_bend",
    }
)
#: Fringe types that include the hard-edge (pole-face rotation) kick.
_FRINGE_HARD = frozenset(
    {"hard_edge_only", "full", "sad_full", "basic_bend", "linear_edge"}
)
#: Fringe types that include the soft edge, i.e. Bmad's ``fint``/``hgap``. Bmad zeroes
#: ``fint_gap`` only for ``hard_edge_only`` and ``sad_full`` (fringe_mod.f90:280), so
#: everything else -- including ``basic_bend``, the *default* for a bend -- keeps it.
_FRINGE_SOFT = frozenset({"soft_edge_only", "full", "basic_bend", "linear_edge"})

#: Bmad ``fringe_at``/``aperture_at`` values, from ``bmad_struct.f90:387``.
_AT_ENTRY = frozenset({"entrance_end", "both_ends"})
_AT_EXIT = frozenset({"exit_end", "both_ends"})


def _warn(message: str) -> None:
    warnings.warn(message, TaoTranslationWarning, stacklevel=3)


def _get(info: dict, key: str, default: float = 0.0) -> float:
    """Read a Bmad attribute as a float, treating a missing or null value as zero."""
    value = info.get(key, default)
    return default if value is None else float(value)


def _alignment(info: dict, name: str, key: str) -> dict:
    """Translate Bmad misalignments into ImpactX element arguments.

    Verified: ``dx = +X_OFFSET_TOT``, ``dy = +Y_OFFSET_TOT`` and, for straight elements,
    ``rotation = +degrees(TILT_TOT)``. Both signs were checked against the negated
    alternative, which is wrong by O(1). Note this differs from Impact-Z, which needs a
    negated tilt: two codes of the same family are not interchangeable here.

    Bends carry no ``TILT``. They have ``REF_TILT``, which rotates the bend plane and
    the downstream reference frame with it, and ``ROLL``, which rotates only the magnet.
    ImpactX's ``rotation`` reproduces ``REF_TILT`` exactly (verified to 1.8e-15 for a
    vertical bend, ``ref_tilt = pi/2``) but does *not* reproduce ``ROLL``. See
    :func:`_bend_roll` for why, and for what is done instead.
    """
    is_bend = key in ("sbend", "rbend", "rf_bend")
    if is_bend:
        # REF_TILT only. ROLL is a different thing and is handled where the bend is
        # built, because ImpactX has no element that expresses it -- see _bend_roll.
        rotation = math.degrees(_get(info, "REF_TILT_TOT", _get(info, "REF_TILT")))
    else:
        rotation = math.degrees(_get(info, "TILT_TOT", _get(info, "TILT")))

    out = {
        "dx": _get(info, "X_OFFSET_TOT", _get(info, "X_OFFSET")),
        "dy": _get(info, "Y_OFFSET_TOT", _get(info, "Y_OFFSET")),
        "rotation": rotation,
    }
    offset = max(
        abs(_get(info, "X_OFFSET_TOT", _get(info, "X_OFFSET"))),
        abs(_get(info, "Y_OFFSET_TOT", _get(info, "Y_OFFSET"))),
    )
    if is_bend and rotation != 0.0 and offset > 0.0:
        _warn(
            f"{name}: this bend has both REF_TILT and a transverse offset. Bmad applies "
            "the offset as a rigid-body displacement about the bend centre in the "
            "tilted frame (offset_particle.f90:208-226), which ImpactX's element "
            "alignment does not reproduce: measured 1.8e-4. Either alone is exact."
        )
    for attribute in ("X_PITCH_TOT", "Y_PITCH_TOT"):
        if _get(info, attribute, _get(info, attribute[:-4])) != 0.0:
            _warn(
                f"{name}: Bmad {attribute} is dropped. ImpactX elements have transverse "
                "offsets and a roll, but no pitch."
            )
            break
    if _get(info, "Z_OFFSET_TOT", _get(info, "Z_OFFSET")) != 0.0:
        _warn(f"{name}: Bmad Z_OFFSET is dropped; ImpactX has no longitudinal offset.")
    return out


def _apertures_for(info: dict, name: str) -> tuple[list, list]:
    """Build ImpactX Apertures from Bmad's aperture limits.

    Every Bmad element carries ``X1/X2/Y1/Y2_LIMIT``, ``aperture_type`` and
    ``aperture_at``, not just collimators, so this applies to all of them.

    Returns
    -------
    tuple
        ``(entry, exit)`` lists of ImpactX elements to place before and after the body.
    """
    from impactx import elements

    x1, x2 = abs(_get(info, "X1_LIMIT")), abs(_get(info, "X2_LIMIT"))
    y1, y2 = abs(_get(info, "Y1_LIMIT")), abs(_get(info, "Y2_LIMIT"))
    half_x, half_y = max(x1, x2), max(y1, y2)
    if half_x <= 0.0 and half_y <= 0.0:
        return [], []

    at = str(info.get("aperture_at", "Exit_End") or "Exit_End").lower()
    if at in ("no_aperture", "lord_defined"):
        return [], []
    if at in ("continuous", "surface", "wall_transition"):
        _warn(
            f"{name}: Bmad aperture_at={at!r} is applied continuously along the "
            "element. ImpactX apertures are thin, so it is applied at both ends only."
        )
        at = "both_ends"

    if (x1 and x2 and abs(x1 - x2) > 1e-15) or (y1 and y2 and abs(y1 - y2) > 1e-15):
        _warn(
            f"{name}: Bmad's aperture is asymmetric about the axis "
            f"(x1={x1}, x2={x2}, y1={y1}, y2={y2}); ImpactX apertures are centred, so "
            "the larger half-width is used and the aperture is too permissive."
        )
    if _get(info, "X_OFFSET_TOT") != 0.0 or _get(info, "Y_OFFSET_TOT") != 0.0:
        moves = str(info.get("offset_moves_aperture", "")).lower() in ("t", "true", "1")
        if moves:
            _warn(
                f"{name}: Bmad offset_moves_aperture is set, but the translated "
                "aperture stays on axis."
            )

    aperture_type = str(
        info.get("aperture_type", "rectangular") or "rectangular"
    ).lower()
    if aperture_type not in ("rectangular", "elliptical", "auto"):
        # bmad_struct.f90:160 also allows Surface, Wall3D, Custom and Lord_Defined.
        _warn(
            f"{name}: Bmad aperture_type={info.get('aperture_type')!r} has no ImpactX "
            "equivalent; a rectangular aperture is used instead."
        )
    shape = "elliptical" if aperture_type == "elliptical" else "rectangular"

    # Bmad treats a zero limit as *no* limit and otherwise falls back to
    # bmad_com%max_aperture_limit, which defaults to 1000 m. A 1 m stand-in here would
    # be a thousand times tighter than Bmad and could remove particles Bmad keeps.
    unlimited = 1000.0

    def make(suffix: str):
        return elements.Aperture(
            name=f"{name}_aperture{suffix}",
            aperture_x=half_x if half_x > 0.0 else unlimited,
            aperture_y=half_y if half_y > 0.0 else unlimited,
            shape=shape,
        )

    entry = [make("_entry")] if at in _AT_ENTRY else []
    exit_ = [make("_exit")] if at in _AT_EXIT else []
    return entry, exit_


def _check_stray_kicks(info: dict, name: str, key: str) -> None:
    """Warn about ``hkick``/``vkick`` on an element that is not a kicker.

    Bmad lets any element carry a steering kick. ImpactX has no equivalent field on a
    quadrupole or a bend, so the kick is dropped rather than silently mismodelled.
    """
    if key in ("hkicker", "vkicker", "kicker", "ac_kicker"):
        return
    # HKICK and BL_HKICK are the same physical kick in normalised and field-integral
    # units, so only one of each pair is reported.
    for attribute, alternative in (("HKICK", "BL_HKICK"), ("VKICK", "BL_VKICK")):
        value = _get(info, attribute) or _get(info, alternative)
        if value != 0.0:
            _warn(
                f"{name}: Bmad {attribute}={value} on a {key} is dropped. Add an "
                "explicit kicker element to the Bmad lattice to carry it across."
            )


def _quadrupole_edges(
    info: dict, name: str, k1: float, align: dict, body: list
) -> list:
    """Wrap a quadrupole body in ImpactX QuadEdge elements when Bmad applies a fringe.

    ``QuadEdge`` is Bmad's quadrupole fringe, not merely something similarly named:
    ImpactX cites Forest and Milutinovic, NIM A 269, 474 (1988) and uses
    ``a = +-(-k/12)/(1+delta)`` (QuadEdge.H:124), which is bit-for-bit Bmad's
    ``hard_multipole_edge_kick`` at ``n = 1``, whose ``cab = charge_dir*k1/(4*(n+2)*rel_p)``
    is the same ``k1/12/rel_p`` (fringe_mod.f90:770-833).

    Measured on ``quadrupole, l=0.3, k1=2, fringe_type=hard_edge_only``: body alone
    5.0e-7, body with both edges 1.9e-14.

    Bmad's default for a quadrupole is ``none``, so this usually adds nothing.
    """
    from impactx import elements

    fringe_type = str(info.get("FRINGE_TYPE", "None") or "none").lower()
    fringe_at = str(info.get("FRINGE_AT", "Both_Ends") or "both_ends").lower()

    # Bmad's SAD soft quadrupole edge is driven by FQ1/FQ2, which default to zero and
    # make the map a no-op. Warn only when they are actually set.
    if fringe_type in ("soft_edge_only", "full") and (
        _get(info, "FQ1") != 0.0 or _get(info, "FQ2") != 0.0
    ):
        _warn(
            f"{name}: Bmad's soft quadrupole edge (FQ1/FQ2) has no ImpactX equivalent "
            "and is dropped."
        )

    if fringe_type not in ("hard_edge_only", "full"):
        return body

    entry = (
        [elements.QuadEdge(name=f"{name}_f1", k=k1, flag="entry", **align)]
        if fringe_at in _AT_ENTRY
        else []
    )
    exit_ = (
        [elements.QuadEdge(name=f"{name}_f2", k=k1, flag="exit", **align)]
        if fringe_at in _AT_EXIT
        else []
    )
    return entry + body + exit_


def _bend_roll(info: dict, name: str, angle: float, align: dict, bend_body) -> list:
    """Build a Bmad bend's body, carrying ``ROLL`` as a kick at the bend centre.

    ImpactX has no element that expresses Bmad's ``ROLL``, and this was checked rather
    than assumed. Bmad rolls a bend as a rigid-body rotation of the *curved* magnet
    about its centre, leaving the reference orbit on the design trajectory; the whole
    of ImpactX's rotation machinery instead rotates the magnet *and* the orbit with it:

    * ``Alignment(rotation=...)``, the ``rotation`` argument every element takes, only
      transforms particle coordinates in and out at the element boundaries.
    * ``PlaneXYRot`` is that same rotation as a standalone thin element -- measured
      identical to ``rotation`` (0.9497 versus 0.9492 against a rolled bend).
    * ``PRot`` rotates in the *x-z* plane, changing the reference orbit's angle with
      respect to z. That is pole-face geometry, not a transverse roll.

    All three reproduce Bmad's ``REF_TILT`` exactly instead, which is why
    :func:`_alignment` uses ``rotation`` for that and not for this.

    What a roll actually does to a bend is deflect the beam out of the bend plane. An
    on-axis particle leaves a bend rolled by ``psi`` with ``px = ANGLE*(1 - cos psi)``
    and ``py = -ANGLE*sin psi`` -- measured against Bmad, and matching those closed
    forms to 0.25% and 0.08%. Applying exactly that as a thin kick between two half
    bends captures **99.92%** of the roll, measured at rolls of 1e-4, 1e-3, 1e-2 and
    0.1 rad, where dropping the roll captures none of it.
    """
    from impactx import elements

    roll = _get(info, "ROLL_TOT", _get(info, "ROLL"))
    if roll == 0.0:
        return bend_body()

    _warn(
        f"{name}: Bmad ROLL={roll} rad has no ImpactX equivalent -- an element rotation "
        "turns the reference orbit with the magnet, which a roll does not. It is "
        "modelled as a thin kick between two half bends, which reproduces 99.92% of "
        "the roll; the remaining 0.08% is not corrected."
    )
    return (
        bend_body(0.5, "_a")
        + [
            elements.Kicker(
                name=f"{name}_roll",
                xkick=angle * (1.0 - math.cos(roll)),
                ykick=-angle * math.sin(roll),
                **align,
            )
        ]
        + bend_body(0.5, "_b")
    )


#: Tao reports a multipole table in whichever representation the element uses: a thin
#: `multipole` comes back as KnL/Tn, while error multipoles on a magnet come back as
#: An/Bn. Checking only one silently reads a live corrector as empty.
_MULTIPOLE_STRENGTH_KEYS = ("An", "Bn", "KnL", "An (equiv)", "Bn (equiv)")


def _live_multipole_orders(info: dict) -> list:
    """Orders of the multipole terms that are actually non-zero, in order.

    ``lattice_from_tao`` stows ``tao.ele_multipoles(...)["data"]`` here.
    """
    return sorted(
        entry.get("index")
        for entry in (info.get("_multipoles") or [])
        if any(entry.get(key) for key in _MULTIPOLE_STRENGTH_KEYS)
    )


def _has_multipole_content(info: dict) -> bool:
    """Whether a thin multipole element actually carries a moment."""
    return bool(_live_multipole_orders(info))


def _check_multipole_errors(info: dict, name: str) -> None:
    """Warn about Bmad multipole error tables attached to a normal magnet.

    Bmad lets any magnet carry ``A_n``/``B_n`` (or ``K_nL``/``T_n``) multipole errors on
    top of its main field. Nothing in ImpactX's element set carries them, and they are
    invisible to a transfer-matrix check above n=1: a skew-quad error ``a2 = 5`` on a
    quadrupole moves a tracked bunch by 4.0e-3.
    """
    # NOT `has#ab_multipoles` from ele_head: that is True for every quadrupole,
    # meaning only that the structure exists. The live values come from
    # tao.ele_multipoles(...)["data"], which lattice_from_tao stows here.
    orders = _live_multipole_orders(info)
    if orders:
        _warn(
            f"{name}: Bmad multipole error terms of order {orders} (A_n/B_n) are "
            "dropped; ImpactX magnets carry no error multipoles. They are invisible to "
            "a transfer-matrix check above n=1 -- a skew-quad error a2=5 moves a tracked "
            "bunch by 4.0e-3 -- so model them as separate multipole elements in Bmad."
        )


def _bend_edges(info: dict, name: str, rc: float, align: dict) -> tuple[list, list]:
    """Build the ImpactX DipEdge elements for a Bmad bend's pole faces.

    Which edges exist is decided by Bmad, not assumed:

    * ``fringe_type = none`` gives **no** edge focusing at all. Adding a DipEdge anyway
      is wrong by 8.9e-2 on the transfer map.
    * ``basic_bend`` (Bmad's default for a bend) and ``hard_edge_only`` are exactly the
      hard-edge map: ``DipEdge(psi=E1, rc, g=0)`` matches Bmad to 1.9e-15.
    * ``full``/``sad_full`` add Bmad's soft edge, which maps onto ImpactX's gap and
      fringe-field integral as ``g = 2*HGAP`` and ``K2 = FINT`` (``FINTX``/``HGAPX`` at
      the exit). Residual 2.7e-5, which is ImpactX's documented first-order-in-``g/rc``
      truncation of the exact ``tan(psi - psi_correction)``, not a mapping error.
    * ``soft_edge_only`` keeps the gap term and drops the pole-face rotation.

    ``fringe_at`` selects which ends get an edge. The bend's own misalignment is applied
    to the edges as well as to the body -- each ImpactX element transforms into and back
    out of the element frame, so this rotates the edge focusing with the magnet instead
    of double-counting the offset. Omitting it makes a vertical bend wrong by 3.9e-4.

    ImpactX's ``nonlinear`` edge model is used, not the default ``linear`` one: Bmad's
    hard edge carries the second-order terms too, and tracking with ``linear`` is wrong
    by 2.2e-4 where ``nonlinear`` agrees to 2.1e-9. This is also why each edge is given
    an explicit ``location``, which only the nonlinear model reads.
    """
    from impactx import elements

    fringe_type = str(info.get("FRINGE_TYPE", "Basic_Bend") or "none").lower()
    fringe_at = str(info.get("FRINGE_AT", "Both_Ends") or "both_ends").lower()
    if fringe_type not in _FRINGE_TYPES:
        _warn(
            f"{name}: unrecognised Bmad FRINGE_TYPE={fringe_type!r}; treated as the "
            "hard edge."
        )
        fringe_type = "hard_edge_only"

    hard = fringe_type in _FRINGE_HARD
    soft = fringe_type in _FRINGE_SOFT
    if not (hard or soft):
        return [], []

    # hwang_bend_edge_kick also carries k1*tan(e) (fringe_mod.f90:287), the coupling
    # between the bend's quadrupole component and its pole face, and Bmad supports
    # H1/H2 pole-face curvature. DipEdge has neither.
    if hard and _get(info, "K1") != 0.0:
        _warn(
            f"{name}: the K1 x pole-face coupling in Bmad's edge map (its k1*tan(e) "
            "term) has no DipEdge counterpart and is dropped."
        )
    for attribute in ("H1", "H2"):
        if _get(info, attribute) != 0.0:
            _warn(
                f"{name}: Bmad {attribute}={info[attribute]} (pole-face curvature) is "
                "dropped; ImpactX's DipEdge has flat pole faces."
            )

    # Bmad's linear_edge and the SAD soft edge are non-chromatic, which is exactly what
    # ImpactX's linear model is; everything else uses Hwang & Lee, as DipEdge does.
    model = (
        "linear" if fringe_type in ("linear_edge", "soft_edge_only") else "nonlinear"
    )
    # basic_bend (Bmad's default) and linear_edge use the same Hwang & Lee map DipEdge
    # implements, and agree to 2.1e-9 and 1.5e-5. The other types use a genuinely
    # different map in Bmad -- a PTC Lie map for `full`, SAD's for `sad_full` -- so a
    # residual remains that no DipEdge parameter removes (the K0..K6 space was searched).
    _RESIDUAL = {"full": "6.1e-5", "sad_full": "3.3e-4", "soft_edge_only": "7.9e-5"}
    if fringe_type in _RESIDUAL and (
        _get(info, "FINT") != 0.0 or _get(info, "FINTX") != 0.0
    ):
        note = (
            "Bmad zeroes FINT/HGAP for this type, so only the hard edge is translated."
            if not soft
            else "The gap itself maps exactly, as g = 2*HGAP and K2 = FINT."
        )
        _warn(
            f"{name}: Bmad FRINGE_TYPE={info.get('FRINGE_TYPE')!r} uses a different edge "
            f"map than ImpactX's DipEdge, not a truncation of it. {note} A residual of "
            f"about {_RESIDUAL[fringe_type]} remains. Bmad's default 'basic_bend' agrees "
            "to 2.1e-9 if you can use it instead."
        )

    def edge(psi: float, fint: float, hgap: float, location: str, suffix: str):
        psi = psi if hard else 0.0
        gap = 2.0 * hgap if soft else 0.0
        # Not skipped when psi == 0: ImpactX's nonlinear edge map keeps a term
        # loc/(2*rc) that survives a zero pole-face angle, and so does Bmad's. Dropping
        # the edge there makes a bend wrong by 3.9e-4.
        return elements.DipEdge(
            name=f"{name}{suffix}",
            psi=psi,
            rc=rc,
            g=gap,
            K0=0.0,
            # K3 defaults to 1/6 in ImpactX and drives a -4*c12*y^3 term whose 1/g
            # factor *grows* as the gap shrinks; Bmad's hwang_bend_edge_kick has no
            # such term. Measured: leaving the default costs 1.0e-6 where K3=0 gives
            # 2.1e-9. Harmless when there is no gap.
            K3=0.0,
            K2=fint if soft else 1.0,
            location=location,
            model=model,
            **align,
        )

    entry = (
        edge(_get(info, "E1"), _get(info, "FINT"), _get(info, "HGAP"), "entry", "_e1")
        if fringe_at in _AT_ENTRY
        else None
    )
    exit_ = (
        edge(_get(info, "E2"), _get(info, "FINTX"), _get(info, "HGAPX"), "exit", "_e2")
        if fringe_at in _AT_EXIT
        else None
    )
    return ([entry] if entry else []), ([exit_] if exit_ else [])


def translate_element(
    info: dict,
    nslice: int = 8,
    name: str = "",
    mass_eV: float | None = None,
    momentum_scale: float = 1.0,
) -> list:
    """Translate one Tao element into zero or more ImpactX elements.

    Parameters
    ----------
    info : dict
        Merged ``ele_head`` / ``ele_gen_attribs`` output for the element.
    nslice : int
        Slices per thick element.
    name : str
        Element name to give the ImpactX elements.
    mass_eV : float, optional
        Rest mass of the reference species, needed to normalise an RF voltage. Defaults
        to the electron mass.
    momentum_scale : float
        ``p0c_Bmad / p0c_ImpactX`` at this element. Bmad holds ``p0c`` fixed across an
        ``rfcavity`` while ImpactX's reference particle really is accelerated, so every
        strength Bmad normalises to *its* momentum must be rescaled to ImpactX's.
        :func:`lattice_from_tao` computes it; it is exactly 1 without acceleration.

    Returns
    -------
    list
        ImpactX elements, in beam order. Empty for elements with no beam effect.

    Raises
    ------
    UnsupportedElementError
        If the element has length and no faithful ImpactX representation. A zero-length
        unknown element is treated as a marker instead, which is safe.

    Notes
    -----
    The element table. Every row was established against Bmad itself, by comparing Tao's
    ``ele_mat6`` in ImpactX's basis and by tracking the same 64-particle bunch through
    both codes at 100 MeV with a 5e-4 momentum spread. "Agreement" is the worst relative
    coordinate difference measured.

    | Bmad | ImpactX | agreement |
    | --- | --- | --- |
    | ``drift``, ``pipe``, ``monitor``, ``instrument``, collimators | ``ExactDrift`` | 2.8e-15 |
    | ``marker``, zero-length drift-like | ``Marker`` | exact |
    | ``quadrupole`` | ``ChrQuad(k=K1)`` | 1.9e-14 |
    | quadrupole fringe | ``QuadEdge(k=K1)`` at each end | 1.9e-14 |
    | ``sbend``/``rbend`` body | ``ExactSbend(phi=ANGLE)`` | 1.4e-11 |
    | bend pole faces incl. FINT/HGAP | ``DipEdge(psi, rc, g=2*HGAP, K2=FINT, K3=0)`` | 2.2e-9 |
    | ``sbend`` with ``k1``/``k2`` | ``ChrQuad``/``ThinDipole``/``Multipole`` steps | 4.7e-6 at 32 |
    | ``sbend``, ``exact_multipoles=vertically_pure`` | ``ExactCFbend`` | 2.5e-7 |
    | zero-angle ``sbend`` with ``k1`` | ``ChrQuad(k=K1)`` | 2.2e-14 |
    | ``solenoid`` | ``ChrAcc(ez=0, bz=KS*beta_gamma)`` | 2.9e-9 |
    | ``sextupole`` | ``ExactMultipole(k_normal=[0,0,K2])`` | 9.2e-11 (see below) |
    | ``octupole`` | ``ExactMultipole(k_normal=[0,0,0,K3])`` | 1.2e-9 |
    | ``hkicker``/``vkicker``/``kicker`` | ``Kicker(xkick, ykick)`` | 1.3e-14 |
    | ``rfcavity`` | sliced ``ExactDrift`` + ``ShortRF`` | 8.5e-8 |
    | ``lcavity``, travelling wave | sliced ``ExactDrift`` + ``ShortRF`` | 9.6e-6 |
    | ``x_offset``/``y_offset`` | ``dx``/``dy``, same sign | 6.2e-9 |
    | ``tilt`` | ``rotation = +degrees(TILT)`` | 2.2e-14 |
    | bend ``ref_tilt`` | ``rotation`` on body *and* edges | 1.2e-11 |
    | bend ``roll`` | half bends around a centre ``Kicker`` | 99.93% out of plane |
    | ``is_on = F``, straight elements | ``ExactDrift`` of the same length | 6.0e-15 |
    | aperture limits | ``Aperture``, shape from ``aperture_type`` | exact |

    ImpactX's ``Exact*`` models are the more physical maps, but Bmad's ``bmad_standard``
    body for a quadrupole, a solenoid and a combined-function bend is paraxial in (x, y)
    and exact in energy (``track_a_bend.f90:111``), which is what ImpactX's ``Chr*``
    family models -- hence ``ChrQuad`` over ``ExactQuad``, 1.9e-14 against 4.5e-9. Bmad's
    drift and pure bend body really are exact, so ``ExactDrift``/``ExactSbend`` win there.

    The sextupole figure is measured with Bmad's own integrator converged. Bmad defaults
    to a single drift-kick-drift step, where a ``k2 = 25`` sextupole differs by 2.4e-6 --
    that is Bmad's integration error, not this translation's.

    Dropped, each with a warning naming the element and attribute: ``x_pitch``,
    ``y_pitch``, ``z_offset``, ``hkick``/``vkick`` on a non-kicker, multipole error
    tables (``A_n``/``B_n``) on a magnet, a bend's ``DG`` field error, a bend carrying
    both ``REF_TILT`` and a transverse offset, fringe fields other than bend pole faces
    and quadrupole edges, Bmad's soft quadrupole edge when ``FQ1``/``FQ2`` are set, and
    ``PHI0_AUTOSCALE``. ``PHI0_MULTIPASS`` is *not* dropped -- ``track_a_rfcavity.f90:81``
    folds it into the phase, so this does too.

    See :func:`_bend_edges` for the pole-face fringe rules, :func:`_quadrupole_edges` for
    the quadrupole ones, and :func:`_bend_roll` for why a roll cannot be expressed
    directly.
    """
    from impactx import elements

    key = str(info.get("key", "")).lower()
    name = name or str(info.get("name", "") or "")
    length = _get(info, "L")
    mass_eV = mass_eV if mass_eV is not None else mass_of("electron")

    if key in CONTROL_KEYS:
        return []

    align = _alignment(info, name, key)
    entry_aperture, exit_aperture = _apertures_for(info, name)

    if key in MARKER_LIKE_KEYS or (length == 0.0 and key in DRIFT_LIKE_KEYS):
        # Apertures are computed first: a thin collimator is the canonical way to write
        # one, and returning a bare Marker here silently discarded its limits.
        return entry_aperture + [elements.Marker(name=name or key)] + exit_aperture

    _check_stray_kicks(info, name, key)
    _check_multipole_errors(info, name)

    def thick(**kwargs):
        return dict(name=name, ds=length, nslice=nslice, **align, **kwargs)

    def wrap(body: list) -> list:
        return entry_aperture + body + exit_aperture

    # An element Bmad has switched off still occupies its length.
    if info.get("is_on") is False:
        # track1_bmad.f90:51-59 substitutes a drift for most switched-off elements, but
        # *not* for sbend, lcavity or patch: track_a_bend.f90:90-94 zeroes g_tot while
        # keeping g, so the particle goes straight through a still-curved coordinate
        # system. Translating that as a drift is 100% wrong.
        if key in ("sbend", "rbend", "rf_bend"):
            raise UnsupportedElementError(
                f"{name}: a switched-off Bmad bend is not a drift -- it keeps its "
                "curved reference geometry with zero field, which ImpactX cannot "
                "express. Remove the element or set is_on = T."
            )
        _warn(f"{name}: Bmad has this element switched off; translated as a drift.")
        if length == 0.0:
            return wrap([elements.Marker(name=name or key)])
        return wrap([elements.ExactDrift(name=name, ds=length, nslice=nslice)])

    # Fringe fields outside bends have no ImpactX counterpart. Bmad's default for a
    # quadrupole, a solenoid and a sextupole is 'None', so this stays quiet in practice.
    # Bends handle their own fringe in _bend_edges, quadrupoles in _quadrupole_edges,
    # and Bmad's rfcavity/lcavity tracking ignores FRINGE_TYPE altogether -- neither
    # `fringe` nor `apply_element_edge_kick` appears in track_a_rfcavity.f90 -- so
    # warning for those would be pure noise.
    if key not in (
        "sbend",
        "rbend",
        "rf_bend",
        "lcavity",
        "rfcavity",
        "quadrupole",
    ):
        fringe = str(info.get("FRINGE_TYPE", "None") or "none").lower()
        if fringe not in ("none", ""):
            _warn(
                f"{name}: Bmad FRINGE_TYPE={info.get('FRINGE_TYPE')!r} on a {key} is "
                "dropped; ImpactX models fringe fields only at bend and quadrupole "
                "edges."
            )

    if key in DRIFT_LIKE_KEYS:
        return wrap([elements.ExactDrift(**thick())])

    if key == "quadrupole":
        # ChrQuad, not ExactQuad. Bmad's bmad_standard quadrupole body is paraxial in
        # (x, y) but *exact* in energy, which is precisely what ImpactX's Chr* family
        # models. ExactQuad is the more physical map but a worse match for Bmad:
        # measured 4.5e-9 for ExactQuad against 1.9e-14 for ChrQuad. (The fully linear
        # `Quad` is 8.0e-5.)
        k1 = _get(info, "K1") * momentum_scale
        body = [elements.ChrQuad(**thick(k=k1))]
        return wrap(_quadrupole_edges(info, name, k1, align, body))

    if key == "solenoid":
        # ChrAcc, not Sol. ImpactX's `Sol` is a purely linear map with no pt dependence
        # in its transverse terms, so it disagrees with Bmad by 1.4e-4 at dp/p = 5e-4.
        # `ChrAcc` expands the Hamiltonian to second order transversely but keeps the
        # *exact* pt dependence, and with no electric field it is a chromatic solenoid:
        # measured 2.9e-9 on the same bunch.
        #
        # The field strengths are normalised differently. Bmad's KS is charge*Bz/p0c,
        # while ChrAcc's bz is charge*Bz/(m*c), so bz = KS * beta_gamma. Verified: the
        # unscaled and half-scaled alternatives are ~3e4 times worse. beta_gamma comes
        # from this element's own P0C, so it stays right if the lattice ever changes
        # energy upstream.
        beta_gamma = _get(info, "P0C") / mass_eV
        return wrap(
            [elements.ChrAcc(**thick(ez=0.0, bz=_get(info, "KS") * beta_gamma))]
        )

    if key in ("sbend", "rbend", "rf_bend"):
        if key == "rf_bend":
            _warn(f"{name}: rf_bend is translated as a static bend; its RF is dropped.")
        angle = _get(info, "ANGLE")
        if angle == 0.0:
            # A zero-angle sbend is legal Bmad and may still focus. Translating it as a
            # drift silently discards K1 entirely -- measured 100% wrong.
            k1 = _get(info, "K1") * momentum_scale
            if k1 != 0.0:
                return wrap([elements.ChrQuad(**thick(k=k1))])
            return wrap([elements.ExactDrift(**thick())])
        if _get(info, "DG") != 0.0:
            _warn(
                f"{name}: Bmad DG={info['DG']} (a bend field error) is dropped; the "
                "translated bend uses the design angle."
            )
        rc = length / angle
        k1 = _get(info, "K1") * momentum_scale
        k2 = _get(info, "K2") * momentum_scale
        # Bmad's `exact_multipoles` decides whether a bend's multipoles are expanded in
        # Cartesian coordinates (the default, `off`) or curvilinear ones. ImpactX's
        # ExactCFbend is curvilinear, so it only matches the `vertically_pure` setting
        # -- measured 2.5e-7 there against 7.8e-5 for Bmad's default, and that 7.8e-5 is
        # a convention floor, not a convergence one: it is unmoved by Bmad num_steps=400
        # and integrator_order=6, and by ImpactX int_order 2->6, mapsteps 10->400.
        curvilinear = str(info.get("EXACT_MULTIPOLES", "") or "").lower() == (
            "vertically_pure"
        )

        def bend_body(fraction: float = 1.0, suffix: str = "") -> list:
            steps = max(int(nslice * fraction), 1)
            if k1 == 0.0 and k2 == 0.0:
                return [
                    elements.ExactSbend(
                        name=f"{name}{suffix}",
                        ds=length * fraction,
                        nslice=steps,
                        phi=math.degrees(angle * fraction),
                        **align,
                    )
                ]
            if curvilinear:
                # ImpactX's defaults (int_order=2, mapsteps=10) are under-converged
                # here: they cost 7.3e-4 where int_order=4, mapsteps=100 reaches the
                # 2.5e-7 floor and stays there.
                return [
                    elements.ExactCFbend(
                        name=f"{name}{suffix}",
                        ds=length * fraction,
                        k_normal=[1.0 / rc, k1, k2],
                        k_skew=[0.0, 0.0, 0.0],
                        int_order=4,
                        mapsteps=100,
                        nslice=steps,
                        **align,
                    )
                ]
            # Cartesian multipoles: drift-kick-drift in Bmad's own model. ChrQuad is
            # paraxial in (x, y) with exact pt dependence, exactly like Bmad's
            # sbend_body_with_k1_map; ThinDipole supplies the curvature and a thin
            # Multipole the sextupole term. ImpactX's CFbend is non-chromatic instead
            # (1.6e-4 at dp/p = 5e-4, 1.1e-3 at 2e-3).
            out: list = []
            half = length * fraction / (2.0 * steps)
            for index in range(steps):
                arm = dict(ds=half, k=k1, nslice=1, **align)
                out.append(elements.ChrQuad(name=f"{name}{suffix}_{index}a", **arm))
                out.append(
                    elements.ThinDipole(
                        name=f"{name}{suffix}_{index}",
                        theta=math.degrees(angle * fraction / steps),
                        rc=rc,
                        **align,
                    )
                )
                if k2 != 0.0:
                    out.append(
                        elements.Multipole(
                            name=f"{name}{suffix}_{index}s",
                            multipole=3,
                            K_normal=k2 * length * fraction / steps,
                            K_skew=0.0,
                            **align,
                        )
                    )
                out.append(elements.ChrQuad(name=f"{name}{suffix}_{index}b", **arm))
            return out

        if k1 != 0.0 or k2 != 0.0:
            _warn(
                f"{name}: a combined-function bend is translated as "
                + (
                    "an ExactCFbend, which matches Bmad's 'vertically_pure' multipoles "
                    "to 2.5e-7 -- but only once Bmad's own integrator is converged. At "
                    "Bmad's default num_steps the two differ by 6.2e-4, and that is "
                    "Bmad's integration error, not this translation's."
                    if curvilinear
                    else f"{nslice} ChrQuad/ThinDipole steps, because Bmad's own "
                    "combined-function body is paraxial-but-chromatic and no single "
                    "ImpactX element reproduces it with Cartesian multipoles. The "
                    "error converges with nslice: measured 7.5e-5 at 8 steps, 4.7e-6 "
                    "at 32 and 2.9e-7 at 128, for dp/p = 5e-4."
                )
            )
        body = _bend_roll(info, name, angle, align, bend_body)
        entry_edge, exit_edge = _bend_edges(info, name, rc, align)
        return wrap(entry_edge + body + exit_edge)

    if key in _MULTIPOLE_INDEX:
        order = _MULTIPOLE_INDEX[key]
        coefficients = [0.0] * (order + 1)
        # Bmad integrates a sextupole/octupole with `num_steps` drift-kick-drift steps,
        # and defaults to *one*. ImpactX's `nslice` steps are finer, so the two differ
        # by Bmad's integration error -- 2.4e-6 at the Bmad default, falling to 9.2e-11
        # once Bmad's `num_steps` is raised to 200. The translation is not the coarse
        # side here, so this is documented rather than warned about.
        coefficients[order] = _get(info, _MULTIPOLE_STRENGTH[key]) * momentum_scale
        return wrap(
            [
                elements.ExactMultipole(
                    **thick(k_normal=coefficients, k_skew=[0.0] * (order + 1))
                )
            ]
        )

    if key in ("hkicker", "vkicker", "kicker", "ac_kicker"):
        # A steering kick is Bmad's dp/p0c, so it rescales; the geometric deflection a
        # rolled bend produces (in _bend_roll) is an angle and does not.
        if key == "hkicker":
            xkick, ykick = _get(info, "KICK") * momentum_scale, 0.0
        elif key == "vkicker":
            xkick, ykick = 0.0, _get(info, "KICK") * momentum_scale
        else:
            xkick = _get(info, "HKICK") * momentum_scale
            ykick = _get(info, "VKICK") * momentum_scale
        if key == "ac_kicker":
            _warn(
                f"{name}: ac_kicker is translated as a static kicker; its time "
                "dependence is dropped."
            )
        kick = elements.Kicker(name=name, xkick=xkick, ykick=ykick, **align)
        if length == 0.0:
            return wrap([kick])
        # A thick kicker becomes drift-kick-drift.
        _warn(
            f"{name}: a kicker with length {length} m is translated as "
            "drift-kick-drift, with the kick concentrated at the centre."
        )
        half = dict(name=f"{name}_d", ds=length / 2.0, nslice=max(nslice // 2, 1))
        return wrap([elements.ExactDrift(**half), kick, elements.ExactDrift(**half)])

    if key == "rfcavity":
        return wrap(_translate_rfcavity(info, name, length, nslice, align, mass_eV))

    if key == "lcavity":
        return wrap(_translate_lcavity(info, name, length, nslice, align, mass_eV))

    if length > 0.0:
        raise UnsupportedElementError(
            f"{name}: Bmad element type {key!r} has length {length} m and no verified "
            "ImpactX equivalent."
        )

    if key in ("multipole", "ab_multipole") and not _has_multipole_content(info):
        # A corrector set to zero. Every SLAC lattice carries these -- cu_hxr's SQ01 and
        # CQ01 both have an empty multipole table -- and refusing them would mean no
        # production lattice could be translated without skip_unsupported.
        return wrap([elements.Marker(name=name or key)])

    if key in _MAP_AT_ZERO_LENGTH_KEYS:
        raise UnsupportedElementError(
            f"{name}: Bmad element type {key!r} carries a transfer map even at zero "
            "length, so replacing it with a marker would silently drop real physics -- "
            "measured 3.2e-1 away from Bmad for a taylor element. Pass "
            "skip_unsupported=True to accept a marker anyway."
        )

    _warn(
        f"{name}: Bmad element type {key!r} is not translated; it has zero length, so "
        "it is replaced by a marker."
    )
    return wrap([elements.Marker(name=name or key)])


def _translate_rfcavity(
    info: dict, name: str, length: float, nslice: int, align: dict, mass_eV: float
) -> list:
    """A Bmad rfcavity as sliced drift-kick-drift around ImpactX ShortRF elements.

    ImpactX ``ShortRF`` is thin and takes a *normalized* voltage, ``V = max energy gain
    / (m c^2)``. Verified against Bmad for electrons: ``phase_deg = 90 - 360 * PHI0``
    (Bmad's ``phi0`` is in units of 2*pi). The splitting error converges: 2.8e-7 at one
    slice, 1.5e-9 at 64.
    """
    from impactx import elements

    voltage = _get(info, "VOLTAGE")
    frequency = _get(info, "RF_FREQUENCY")
    phi0 = _get(info, "PHI0")
    if frequency == 0.0 or voltage == 0.0:
        return [elements.ExactDrift(name=name, ds=length, nslice=nslice, **align)]

    # track_a_rfcavity.f90:81 adds phi0_multipass into the phase under Bmad's default
    # relative-time tracking, so it belongs in the phase rather than in a warning.
    phi0 += _get(info, "PHI0_MULTIPASS")
    if _get(info, "PHI0_AUTOSCALE") != 0.0:
        _warn(f"{name}: Bmad PHI0_AUTOSCALE={info['PHI0_AUTOSCALE']} is dropped.")
    # CAVITY_TYPE and N_CELL are deliberately not warned about: neither appears in
    # track_a_rfcavity.f90, so Bmad's rfcavity ignores them too.
    _warn(
        f"{name}: rfcavity is translated as {nslice} thin ShortRF kicks inside drifts."
    )

    phase = 90.0 - 360.0 * phi0
    out = []
    half = length / (2.0 * nslice)
    for index in range(nslice):
        out.append(elements.ExactDrift(name=f"{name}_d{index}a", ds=half, nslice=1))
        out.append(
            elements.ShortRF(
                name=f"{name}_{index}",
                V=voltage / mass_eV / nslice,
                freq=frequency,
                phase=phase,
                **align,
            )
        )
        out.append(elements.ExactDrift(name=f"{name}_d{index}b", ds=half, nslice=1))
    return out


def _translate_lcavity(
    info: dict, name: str, length: float, nslice: int, align: dict, mass_eV: float
) -> list:
    """A Bmad lcavity as sliced drift-kick-drift around ImpactX ShortRF elements.

    Unlike an ``rfcavity``, an ``lcavity`` moves the reference energy, and ImpactX can
    follow it: ``ShortRF`` updates the reference particle directly
    (``ShortRF.H:207``, ``refpart.pt = pt - V*cos(phi)`` with ``pt = -gamma``). Bmad's
    ``compute_reference_energy.f90:628-637`` gives
    ``E_tot(exit) = E_tot(in) + VOLTAGE*cos(2*pi*(PHI0 + PHI0_MULTIPASS))`` -- note
    **cos** here where the rfcavity phase convention uses sin -- so the ImpactX phase is
    ``360*(PHI0 + PHI0_MULTIPASS)`` degrees. The reference energy then tracks Bmad's to
    about 3e-12.

    What is *not* modelled is Bmad's Rosenzweig-Serafini edge focusing
    (``track_a_lcavity.f90:288``) and, for a standing-wave cavity, the ponderomotive
    focusing (``:424``). That is why the accuracy depends so strongly on
    ``CAVITY_TYPE``, and the warning says which case you are in.
    """
    from impactx import elements

    voltage = _get(info, "VOLTAGE")
    if voltage == 0.0:
        voltage = _get(info, "GRADIENT") * (_get(info, "L_ACTIVE") or length)
    frequency = _get(info, "RF_FREQUENCY")
    phi0 = _get(info, "PHI0") + _get(info, "PHI0_MULTIPASS")
    if frequency == 0.0 or voltage == 0.0:
        return [elements.ExactDrift(name=name, ds=length, nslice=nslice, **align)]

    cavity_type = str(info.get("CAVITY_TYPE", "standing_wave") or "").lower()
    if cavity_type == "traveling_wave":
        _warn(
            f"{name}: lcavity is translated as {nslice} thin ShortRF kicks inside "
            "drifts, carrying the reference-energy change. Bmad's Rosenzweig-Serafini "
            "edge focusing is not modelled: measured 9.8e-6 on a travelling-wave "
            "cavity."
        )
    else:
        _warn(
            f"{name}: lcavity is translated as {nslice} thin ShortRF kicks inside "
            "drifts. The reference-energy change is carried to 3e-12, but Bmad's "
            "Rosenzweig-Serafini edge focusing and the standing-wave ponderomotive "
            "focusing are not modelled, and they are large: measured 9.1e-2 against "
            "Bmad's defaults. Treat a standing-wave lcavity as indicative only."
        )

    phase = 360.0 * phi0
    out: list = []
    half = length / (2.0 * nslice)
    for index in range(nslice):
        out.append(elements.ExactDrift(name=f"{name}_d{index}a", ds=half, nslice=1))
        out.append(
            elements.ShortRF(
                name=f"{name}_{index}",
                V=voltage / mass_eV / nslice,
                freq=frequency,
                phase=phase,
                **align,
            )
        )
        out.append(elements.ExactDrift(name=f"{name}_d{index}b", ds=half, nslice=1))
    return out


def _reference_mass_eV(tao: Any, branch: int = 0) -> float:
    """Rest mass of the lattice's reference species, from Tao's energy and momentum.

    Uses ``E_TOT`` and ``P0C`` rather than a species name, so it works for any species
    Bmad supports without a lookup table.

    Raises rather than guessing. The mass is load-bearing -- it sets ``ChrAcc.bz`` for
    every solenoid and ``ShortRF.V`` for every cavity -- so falling back to the electron
    mass in a proton lattice would silently scale a solenoid by 1836.
    """
    try:
        attribs = dict(tao.ele_gen_attribs(f"{branch}>>0"))
        e_tot, p0c = float(attribs["E_TOT"]), float(attribs["P0C"])
    except Exception as exc:
        raise ValueError(
            f"Could not read the reference energy of branch {branch} from Tao ({exc}). "
            "The reference mass sets every solenoid and cavity strength, so this "
            "cannot be guessed."
        ) from exc
    mass_squared = e_tot * e_tot - p0c * p0c
    if mass_squared <= 0.0:
        raise ValueError(
            f"Tao reports E_TOT={e_tot} and P0C={p0c} at the start of branch {branch}, "
            "which gives no real rest mass."
        )
    return math.sqrt(mass_squared)


def _element_indices(tao: Any, branch: int) -> list[int]:
    """Indices of the elements actually tracked through, in beam order.

    Two Tao subtleties, both of which bite silently:

    * ``-no_slaves`` excludes super/multipass *slaves* but keeps *lords*
      (``tao_pipe_cmd.f90:4986``). ``-track_only`` is the flag that drops lords. Using
      the first and then addressing elements by a bare running index walks the tracking
      branch and the lord region as one list, so a superposed lattice is silently
      truncated (measured 1.6 m of a 2.6 m lattice) or has a super-lord translated twice
      (3.2 m).
    * Without ``-index_order`` the list comes back in s-order, not index order.
    """
    raw = tao.lat_list(
        f"{branch}>>*", "ele.ix_ele", flags="-array_out -track_only -index_order"
    )
    return [int(value) for value in raw]


def _restrict_to_range(
    tao: Any,
    branch: int,
    indices: list[int],
    track_start: str | None,
    track_end: str | None,
) -> list[int]:
    """Narrow the element list to a named range, inclusive of the end element.

    ImpactX has no notion of starting partway through a lattice -- it tracks whatever
    is in ``sim.lattice`` -- so a partial model is made by translating only part of it.
    That is a slice of a Python list, which is why this needs nothing from ImpactX.

    ``track_start`` names the element the beam is taken *at*, so translation begins with
    the element **after** it: the bunch has already been through it. ``track_end`` is
    included, matching Tao, where ``track_end = "END"`` means the whole lattice.
    """
    if track_start is None and track_end is None:
        return indices

    names = {}
    for position, index in enumerate(indices):
        try:
            name = str(dict(tao.ele_head(f"{branch}>>{index}")).get("name", "") or "")
        except Exception:  # pragma: no cover - already fatal in the main walk
            continue
        # First occurrence wins, as Tao's own bare-name lookup does.
        names.setdefault(name.lower(), position)

    def position_of(label: str, what: str) -> int:
        found = names.get(str(label).lower())
        if found is None:
            raise ValueError(
                f"{what}={label!r} is not an element of branch {branch}. "
                f"The lattice runs {list(names)[:1]} .. {list(names)[-1:]}."
            )
        return found

    first = 0 if track_start is None else position_of(track_start, "track_start") + 1
    last = (
        len(indices) - 1 if track_end is None else position_of(track_end, "track_end")
    )
    if first > last:
        raise ValueError(
            f"track_start={track_start!r} is at or after track_end={track_end!r}; "
            "there would be nothing to track."
        )
    return indices[first : last + 1]


def lattice_from_tao(
    tao: Any,
    nslice: int = 8,
    skip_unsupported: bool = False,
    branch: int = 0,
    track_start: str | None = None,
    track_end: str | None = None,
) -> list:
    """Translate a Tao lattice into ImpactX elements, element by element.

    Every mapping is verified against Bmad; see :func:`translate_element` and the module
    tests for the measured agreement of each element type.

    Parameters
    ----------
    tao : pytao.Tao
        The Tao instance.
    nslice : int
        Slices per thick element. Bmad's per-element ``num_steps``/``ds_step`` are *not*
        read, because they control a different integrator.
    skip_unsupported : bool
        Replace an untranslatable element with a marker and warn, instead of raising.
    branch : int
        Lattice branch to translate. Only one branch is translated.
    track_start : str, optional
        Translate from *after* this element, the one the beam is taken at. The bunch has
        already been through it, so including it would apply it twice.
    track_end : str, optional
        Translate up to and including this element.

    Returns
    -------
    list
        ImpactX elements in beam order.

    Raises
    ------
    UnsupportedElementError
        For an element with length and no verified equivalent, unless
        ``skip_unsupported``.
    ValueError
        If the branch is empty, if an element cannot be read, or if the translation
        produces nothing.
    """
    from impactx import elements

    indices = _restrict_to_range(
        tao, branch, _element_indices(tao, branch), track_start, track_end
    )
    if not indices:
        raise ValueError(f"Tao reported no tracked elements in branch {branch}.")
    mass_eV = _reference_mass_eV(tao, branch)

    # ImpactX's reference particle really is accelerated by a ShortRF, while Bmad holds
    # p0c fixed across an rfcavity. Track ImpactX's gamma so downstream strengths can be
    # renormalised; ShortRF.H:207 does pt -= V*cos(phase), and pt = -gamma.
    gamma = None
    lattice: list = []

    for index in indices:
        identifier = f"{branch}>>{index}"
        try:
            head = dict(tao.ele_head(identifier))
            attribs = dict(tao.ele_gen_attribs(identifier))
        except Exception as exc:
            # Skipping would remove the element's length from the lattice and quietly
            # change the optics, so this is fatal rather than a warning.
            raise ValueError(
                f"Could not read element {identifier} from Tao ({exc}). Refusing to "
                "translate a lattice with a hole in it."
            ) from exc

        info = {**attribs, **head}
        try:
            info["_multipoles"] = dict(tao.ele_multipoles(identifier)).get("data") or []
        except Exception:  # pragma: no cover - element types without a multipole table
            info["_multipoles"] = []
        element_name = str(head.get("name", "") or "")
        p0c_bmad = _get(info, "P0C")
        if gamma is None and p0c_bmad > 0.0:
            gamma = math.hypot(p0c_bmad, mass_eV) / mass_eV

        momentum_scale = 1.0
        if gamma is not None and gamma > 1.0 and p0c_bmad > 0.0:
            p0c_impactx = mass_eV * math.sqrt(gamma * gamma - 1.0)
            if p0c_impactx > 0.0:
                momentum_scale = p0c_bmad / p0c_impactx
                if abs(momentum_scale - 1.0) < 1e-12:
                    momentum_scale = 1.0

        try:
            translated = translate_element(
                info,
                nslice=nslice,
                name=element_name,
                mass_eV=mass_eV,
                momentum_scale=momentum_scale,
            )
        except UnsupportedElementError as exc:
            if not skip_unsupported:
                raise
            # A drift, not a marker, whenever the element has length. Replacing a 1.7 m
            # undulator with a zero-length marker moves every element downstream of it:
            # measured on LCLS cu_hxr, skipping its 98 wigglers and 2 patches lost
            # 109.4 m of a 1750.9 m lattice -- 6.2% -- silently.
            length = _get(info, "L")
            if length > 0.0:
                _warn(
                    f"{exc} Replaced by a drift of its {length} m, so the elements "
                    "after it stay at the right s -- but what it does to the beam is "
                    "still lost, and that is not always small. Measured against Bmad on "
                    "cu_hxr's own undulators, a drift is 7.0e-3 out for an HXR segment "
                    "and 2.5e-1 out for the laser-heater undulator."
                )
                translated = [
                    elements.ExactDrift(
                        name=element_name or "skipped", ds=length, nslice=nslice
                    )
                ]
            else:
                _warn(f"{exc} Replaced by a marker.")
                translated = [elements.Marker(name=element_name or "skipped")]

        if momentum_scale != 1.0:
            _warn(
                f"{element_name}: an upstream cavity left ImpactX's reference momentum "
                f"differing from Bmad's, so this element's normalised strengths were "
                f"rescaled by {momentum_scale:.6g}. Without it the element would be "
                "referenced to the wrong rigidity.\n"
                "Across an rfcavity the two disagree by construction: Bmad holds p0c "
                "fixed while ImpactX accelerates its reference. Inside an lcavity they "
                "disagree only transiently -- a thin-kick model steps the reference "
                "where Bmad ramps it -- and reconverge by the end of the structure."
            )

        for element in translated:
            if type(element).__name__ == "ShortRF":
                values = element.to_dict()
                gamma += values["V"] * math.cos(math.radians(values["phase"]))
        lattice.extend(translated)

    if not lattice:
        raise ValueError(
            f"Translated {len(indices)} Tao elements into an empty ImpactX lattice. "
            "Nothing would be tracked; this is a translation failure, not a valid "
            "result."
        )
    return lattice


# --------------------------------------------------------------------------------------


def capture_points_from_tao(
    tao: Any,
    branch: int = 0,
    track_start: str | None = None,
    track_end: str | None = None,
) -> list[str]:
    """Names of the Bmad elements whose bunch is worth keeping.

    Markers, monitors and instruments. Impact-Z's default is narrower --
    ``write_beam_eles=("monitor::*", "marker::*")`` -- so this is a superset of it,
    not a mirror.

    These are names for :attr:`~lume_impactx.simulator.ImpactXSimulator.capture_at`,
    not lattice elements: ImpactX captures through ``sim.hook``, so nothing is inserted
    into the beamline.

    Two ``lat_list`` calls rather than one ``ele_head`` per element -- the per-element
    form duplicated the whole walk that :func:`lattice_from_tao` already does, adding
    18% to the translation of a 302-element lattice.

    Repeated names are returned once. The simulator disambiguates the occurrences when
    it captures them, as ``NAME``, ``NAME##2`` and so on.
    """
    flags = "-array_out -track_only -index_order"
    keys = list(tao.lat_list(f"{branch}>>*", "ele.key", flags=flags))
    names = list(tao.lat_list(f"{branch}>>*", "ele.name", flags=flags))
    if track_start is not None or track_end is not None:
        # Same range as the lattice, or capture_at would name elements that were never
        # translated and warn about every one of them after the run.
        indices = _element_indices(tao, branch)
        kept = set(_restrict_to_range(tao, branch, indices, track_start, track_end))
        pairs = [
            (key, name)
            for index, key, name in zip(indices, keys, names)
            if index in kept
        ]
        keys = [key for key, _ in pairs]
        names = [name for _, name in pairs]
    seen: list[str] = []
    for key, name in zip(keys, names):
        if str(key).lower() in _CAPTURE_KEYS and name and name not in seen:
            seen.append(str(name))
    return seen


def simulator_from_tao(
    tao: Any,
    ele: str | None = None,
    lattice: list | None = None,
    nslice: int = 8,
    species: str | None = None,
    settings: dict[str, Any] | None = None,
    skip_unsupported: bool = False,
    branch: int = 0,
    capture: bool = True,
    track_start: str | None = None,
    track_end: str | None = None,
    **kwargs: Any,
):
    """Build an :class:`~lume_impactx.simulator.ImpactXSimulator` from a Tao model.

    Parameters
    ----------
    tao : pytao.Tao
        A Tao instance with a tracked beam saved at ``ele``.
    ele : str, optional
        Element to take the beam and reference particle from. Defaults to
        ``track_start`` when given, otherwise the start of ``branch``.
    lattice : list, optional
        ImpactX elements to use. Translated from Tao when omitted.
    nslice : int
        Slices per thick element.
    species : str, optional
        Overrides the species taken from the tracked bunch.
    settings : dict, optional
        ImpactX settings for the simulator.
    skip_unsupported : bool
        Replace untranslatable elements with markers instead of raising.
    branch : int
        Lattice branch to translate.
    capture : bool
        Capture the bunch at every Bmad marker, monitor and instrument, so it appears
        in ``simulator.particles`` under that element's name.
    track_start : str, optional
        Model only the lattice downstream of this element, starting from the bunch Tao
        has there. The default beam element follows it, so one argument moves both the
        lattice and the beam and they cannot disagree.
    track_end : str, optional
        Model up to and including this element.
    **kwargs
        Passed to the simulator, e.g. ``track_on_init``.

    Returns
    -------
    ImpactXSimulator
    """
    from lume_impactx.simulator import ImpactXSimulator

    if ele is None:
        # track_start is where the beam is taken *and* where the lattice begins, so the
        # two cannot be set inconsistently by accident.
        if track_start is not None:
            ele = track_start
        else:
            ele = "BEGINNING" if branch == 0 else f"{branch}>>0"
    # Only from the translated lattice: names taken from Tao would not match a
    # user-supplied one, and would capture nothing without saying so.
    capture_at = (
        capture_points_from_tao(
            tao, branch=branch, track_start=track_start, track_end=track_end
        )
        if capture and lattice is None
        else []
    )
    reference, particles = beam_from_tao(tao, ele, species=species, branch=branch)
    if lattice is None:
        lattice = lattice_from_tao(
            tao,
            nslice=nslice,
            skip_unsupported=skip_unsupported,
            branch=branch,
            track_start=track_start,
            track_end=track_end,
        )
    return ImpactXSimulator(
        lattice=lattice,
        ref=reference,
        initial_particles=particles,
        settings=settings,
        capture_at=capture_at,
        **kwargs,
    )


def model_from_tao(
    tao: Any,
    config: Any = None,
    dummy_run: bool = False,
    **kwargs: Any,
):
    """Build a :class:`~lume_impactx.model.LUMEImpactXModel` from a Tao model.

    This is the one-step Tao to LUME path: it translates the beam and the lattice,
    builds the simulator, tracks once, and generates the action variables, so the
    result can be driven by ``get()``/``set()`` or served over EPICS by ``lume-pva``
    without any further wiring.

    Parameters
    ----------
    tao : pytao.Tao
        A Tao instance with a tracked beam saved at the start element.
    config : VariableMappingConfig, optional
        Controls which variables are generated and how they are named. Defaults to
        :class:`~lume_impactx.config.VariableMappingConfig`.
    dummy_run : bool
        Skip re-tracking on ``set()``, to batch several writes into one run.
    **kwargs
        Passed to :func:`simulator_from_tao`, e.g. ``ele``, ``lattice``, ``nslice``,
        ``species``, ``settings``, ``skip_unsupported``. ``capture`` defaults to False
        here, because no generated variable reads the captured bunches and the model
        re-tracks on every ``set()``.

    Returns
    -------
    LUMEImpactXModel

    Examples
    --------
    >>> tao = Tao(init_file="tao.init", noplot=True)
    >>> tao.cmd("set global track_type = beam")
    >>> tao.cmd("set beam saved_at = *")
    >>> model = model_from_tao(tao, nslice=16)
    >>> model.set({"ele:qf:k": 1.3})
    >>> model.get("moment_final:sigma_x")
    """
    from lume_impactx.model import LUMEImpactXModel

    # capture defaults off here, unlike simulator_from_tao. A LUMEModel re-tracks on
    # every set(), and lume_impactx.config generates no variable from the captures --
    # only particles:initial_particles and particles:final_particles -- so the model
    # would pay a measured ~9x tracking cost, and hold a ParticleGroup per marker, for
    # data nothing exposes. Pass capture=True to opt in.
    kwargs.setdefault("capture", False)
    simulator = simulator_from_tao(tao, **kwargs)
    return LUMEImpactXModel.from_simulator(
        simulator, config=config, dummy_run=dummy_run
    )
