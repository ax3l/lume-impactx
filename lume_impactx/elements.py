"""Introspection of ImpactX lattice elements.

Both :mod:`lume_impactx.simulator` (to snapshot and restore a lattice) and
:mod:`lume_impactx.config` (to generate variables) need to know which attributes an
element has and which of them can be written. This module is the single answer.

Why properties rather than ``to_dict()``
----------------------------------------
``to_dict()`` looks like the obvious schema source, but it is not one:

* It **omits ``nslice`` when it is at its default**. ``Quad(ds=0.3, k=2.0)`` has no
  ``nslice`` key while ``Quad(ds=0.3, k=2.0, nslice=4)`` does, so two structurally
  identical lattices would produce different variable sets.
* It carries **no read/write information**, and writability is genuinely per-type:
  ``Drift.aperture_x`` is read-only while ``Aperture.aperture_x`` is writable, and
  ``ThinDipole.ds``/``nslice`` are read-only while ``Drift``'s are not.
* It reports **radians for angles the constructors take in degrees** on
  :data:`DEGREE_ELEMENTS` (ImpactX emits a ``RuntimeWarning`` about it).

pybind11 exposes every element attribute as a real ``property``, so ``fset is None``
maps exactly onto ``Variable.read_only`` -- and does so correctly for element types that
do not exist yet.
"""

from __future__ import annotations

from typing import Any

#: Element types whose angle attributes read back in radians although their
#: constructors take degrees. Mirrors ``impactx.extensions.KnownElementsList``.
DEGREE_ELEMENTS = ("ExactSbend", "PlaneXYRot", "PRot", "ThinDipole")

#: Attributes that exist as properties but are not element inputs: naming helpers,
#: cached transfer maps, and the callable hooks on ``Programmable``.
DEFAULT_DENY_ATTRIBUTES = frozenset(
    {
        "has_name",
        "map",
        "spin_coupling",
        "symplectic",
        "threadsafe",
        "push",
        "beam_particles",
        "ref_particle",
        # BeamMonitor lazily-computed knobs; reading them before a run raises
        "alpha",
        "beta",
        "cn",
        "tn",
        "nonlinear_lens_invariants",
    }
)


def element_type(element: Any) -> str:
    """Return an element's ImpactX type name, e.g. ``"Quad"``."""
    return type(element).__name__


def element_attribute_schema(
    element: Any,
    deny: frozenset[str] = DEFAULT_DENY_ATTRIBUTES,
) -> dict[str, bool]:
    """Map each settable/gettable attribute of an element to whether it is writable.

    Parameters
    ----------
    element : impactx.elements.*
        Any ImpactX lattice element.
    deny : frozenset of str
        Attribute names to skip.

    Returns
    -------
    dict
        ``{attribute_name: writable}``. Attributes that raise on read are skipped, so
        the result is always safe to ``getattr``.
    """
    cls = type(element)
    schema: dict[str, bool] = {}
    for name in dir(cls):
        if name.startswith("_") or name in deny:
            continue
        descriptor = getattr(cls, name, None)
        if not isinstance(descriptor, property):
            continue
        try:
            getattr(element, name)
        except Exception:
            # e.g. BeamMonitor.alpha before a run: "m.alpha is not set yet"
            continue
        schema[name] = descriptor.fset is not None
    return schema


def snapshot_element(element: Any) -> dict[str, Any]:
    """Capture an element's writable attribute values.

    Used to restore a lattice on ``reset()``. Deliberately avoids ``to_dict()`` /
    ``from_dicts()`` -- rebuilding elements would hit the degrees/radians bug on
    :data:`DEGREE_ELEMENTS`, whereas reading and writing the same attribute is
    self-consistent whatever unit it is in.
    """
    snapshot = {}
    for name, writable in element_attribute_schema(element).items():
        if not writable:
            continue
        value = getattr(element, name)
        if value is None:
            # An unnamed element reads back name=None, but the setter demands a str,
            # so there is nothing to restore and writing it back would raise.
            continue
        snapshot[name] = value
    return snapshot


def restore_element(element: Any, snapshot: dict[str, Any]) -> None:
    """Write a :func:`snapshot_element` result back onto an element."""
    for name, value in snapshot.items():
        setattr(element, name, value)


def snapshot_lattice(lattice: list) -> list[dict[str, Any]]:
    """Capture the writable state of every element in a lattice."""
    return [snapshot_element(element) for element in lattice]


def restore_lattice(lattice: list, snapshots: list[dict[str, Any]]) -> None:
    """Restore a lattice captured by :func:`snapshot_lattice`.

    Raises
    ------
    ValueError
        If the lattice length no longer matches the snapshot.
    """
    if len(lattice) != len(snapshots):
        raise ValueError(
            f"Lattice has {len(lattice)} elements but the snapshot has "
            f"{len(snapshots)}; elements cannot be added or removed after the "
            "simulator is built."
        )
    for element, snapshot in zip(lattice, snapshots):
        restore_element(element, snapshot)


def beam_capture(name: str):
    """A zero-length element that snapshots the bunch passing through it.

    This is how :attr:`lume_impactx.simulator.ImpactXSimulator.particles` gets a beam
    at a named point without writing files: an ``elements.Programmable`` whose push hook
    converts the live container to a ``ParticleGroup`` and stashes it on the element,
    where :meth:`~lume_impactx.simulator.ImpactXSimulator.track` harvests it before the
    container is torn down.

    It is the in-memory counterpart of ImpactX's ``BeamMonitor``, which writes openPMD
    to disk; a get/set loop would otherwise leave a trail of files.

    Parameters
    ----------
    name : str
        Key the captured bunch appears under.

    Returns
    -------
    impactx.elements.Programmable
        Ready to append to a lattice.
    """
    from impactx import elements

    from lume_impactx.utils import particle_container_to_particlegroup

    element = elements.Programmable(name=name, ds=0.0, nslice=1)
    capture: dict = {"name": name, "particles": None}

    def push(particle_container, step, period):
        try:
            capture["particles"] = particle_container_to_particlegroup(
                particle_container
            )
        except Exception as exc:  # pragma: no cover - e.g. a bunch carrying spin
            capture["particles"] = exc

    def advance_reference(reference_particle):
        """No-op: a zero-length probe does not move the reference particle.

        Registered explicitly because ImpactX otherwise prints "Programmable element -
        ref particles: NO HOOK", which reads like a mistake.
        """

    element.push = push
    element.ref_particle = advance_reference
    # Keep the closures alive for as long as the element is: pybind stores them without
    # owning them, and a garbage-collected callback is a crash rather than an error.
    element._lume_impactx_hooks = (push, advance_reference)
    element._lume_impactx_capture = capture
    return element
