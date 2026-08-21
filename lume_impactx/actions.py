"""Action variables bridging LUME's ``get``/``set`` API to an :class:`ImpactXSimulator`.

Each class mixes one of ``lume.actions``' action mixins into one of
``lume.variables``' :class:`~lume.variables.Variable` subclasses, exactly as
``impact/model/actions.py`` and ``lume_cheetah/actions.py`` do.

Read-only variables come in two flavours here, deliberately:

* Attributes that ImpactX itself refuses to write (``Drift.aperture_x``,
  ``ThinDipole.ds``) use the **writable** mixin with ``read_only=True``.
  ``LUMEModel.set`` raises :class:`~lume.exceptions.ReadOnlyError` before ``_set`` is
  ever reached, so the guarantee holds, and one class then covers both cases for a
  given value type.
* Quantities that are outputs by nature -- beam moments, the post-track reference
  particle -- use :class:`~lume.actions.ReadOnlyActionMixin`, whose validator makes
  ``read_only=False`` impossible to construct.

Every ``_get``/``_set`` re-resolves ``simulator.lattice[index]`` rather than caching an
element handle. ImpactX's ``KnownElementsList.extend`` copies elements, and a stale
handle into a rebuilt lattice is a dangling reference, not an exception.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import (
    BoolVariable,
    EnumVariable,
    IntVariable,
    NDVariable,
    ParticleGroupVariable,
    ScalarVariable,
    StrVariable,
)
from pydantic import model_validator

from lume_impactx.simulator import ImpactXSimulator

__all__ = [
    "EleScalarAction",
    "EleIntAction",
    "EleBoolAction",
    "EleStrAction",
    "EleEnumAction",
    "EleNDAction",
    "RefAction",
    "RefEnumAction",
    "RefFinalAction",
    "SimScalarAction",
    "SimIntAction",
    "SimBoolAction",
    "SimEnumAction",
    "MomentAction",
    "MomentHistoryAction",
    "RunInfoAction",
    "OpticsAction",
    "ParticleGroupAction",
]


def _check(simulator: Any) -> ImpactXSimulator:
    if not isinstance(simulator, ImpactXSimulator):
        raise TypeError(
            f"Expected an ImpactXSimulator, got {type(simulator).__name__!r}."
        )
    return simulator


class _ElementAccessMixin:
    """Resolve an element by index and read or write one of its attributes.

    The index is validated on every access. Variables are generated against a specific
    lattice, so inserting or removing an element shifts every later index -- and because
    the elements are ordinary Python objects in a public list, nothing stops a user
    doing that. Left unchecked, ``ele:q1:k`` would quietly read and write a *different*
    magnet. Both the element type and its name are checked, because a swap or an
    in-place replacement keeps the lattice length and the type identical and would
    otherwise be completely silent.
    """

    ele_index: int
    ele_name: str = ""
    attribute: str
    #: Element type recorded when the variable was generated, "" when unknown.
    ele_type: str = ""
    #: Lattice length when the variable was generated, 0 when unknown.
    lattice_size: int = 0

    def _element(self, simulator: Any):
        sim = _check(simulator)
        lattice = sim.lattice

        if self.lattice_size and len(lattice) != self.lattice_size:
            raise RuntimeError(
                f"Variable {self.name!r} was generated for a lattice of "
                f"{self.lattice_size} elements, but it now has {len(lattice)}. "
                "Element indices have shifted, so this variable may address the wrong "
                "element. Rebuild the model with "
                "LUMEImpactXModel.from_simulator(simulator) after changing the lattice."
            )
        try:
            element = lattice[self.ele_index]
        except IndexError as exc:
            raise IndexError(
                f"Variable {self.name!r} points at lattice index {self.ele_index}, "
                f"but the lattice has {len(lattice)} elements."
            ) from exc

        actual_type = type(element).__name__
        actual_name = getattr(element, "name", None) or ""
        if (self.ele_type and actual_type != self.ele_type) or (
            self.ele_name and actual_name != self.ele_name
        ):
            raise RuntimeError(
                f"Variable {self.name!r} expects {self.ele_type or '?'} "
                f"{self.ele_name or '(unnamed)'!r} at lattice index {self.ele_index}, "
                f"but found {actual_type} {actual_name or '(unnamed)'!r}. The lattice "
                "changed after the model was built; rebuild it with "
                "LUMEImpactXModel.from_simulator(simulator)."
            )
        return element

    def _get(self, simulator: Any) -> Any:
        return getattr(self._element(simulator), self.attribute)

    def _set(self, simulator: Any, value: Any) -> None:
        setattr(self._element(simulator), self.attribute, value)


class EleScalarAction(
    _ElementAccessMixin, WritableActionMixin[ImpactXSimulator], ScalarVariable
):
    """A float element attribute, e.g. ``Quad.k``."""


class EleIntAction(
    _ElementAccessMixin, WritableActionMixin[ImpactXSimulator], IntVariable
):
    """An integer element attribute, e.g. ``nslice`` or ``mapsteps``."""


class EleBoolAction(
    _ElementAccessMixin, WritableActionMixin[ImpactXSimulator], BoolVariable
):
    """A boolean element attribute, e.g. ``Aperture.shift_odd_x``."""


class EleStrAction(
    _ElementAccessMixin, WritableActionMixin[ImpactXSimulator], StrVariable
):
    """A free-form string element attribute."""


class EleEnumAction(
    _ElementAccessMixin, WritableActionMixin[ImpactXSimulator], EnumVariable
):
    """A string element attribute with a fixed option set, e.g. ``Aperture.shape``."""


class EleNDAction(
    _ElementAccessMixin, WritableActionMixin[ImpactXSimulator], NDVariable
):
    """A sequence element attribute, e.g. ``ExactMultipole.k_normal``.

    ImpactX wants a Python sequence back, so the setter converts from ndarray and
    refuses a length change -- the declared ``shape`` would no longer hold.
    """

    def _get(self, simulator: Any) -> Any:
        value = getattr(self._element(simulator), self.attribute)
        return np.asarray(value, dtype=self.dtype)

    def _set(self, simulator: Any, value: Any) -> None:
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.shape:
            raise ValueError(
                f"{self.name!r} expects shape {self.shape}, got {array.shape}."
            )
        setattr(self._element(simulator), self.attribute, array.tolist())


class _RefAccessMixin:
    """Read or write one key of the simulator's reference-particle specification.

    This addresses the *input* reference particle, not the live one: the live one is
    rebuilt from this spec on every track, so writing to it would be discarded.
    """

    key: str

    def _get(self, simulator: Any) -> Any:
        return _check(simulator).ref[self.key]

    def _set(self, simulator: Any, value: Any) -> None:
        _check(simulator).ref[self.key] = value


class RefAction(_RefAccessMixin, WritableActionMixin[ImpactXSimulator], ScalarVariable):
    """A numeric reference-particle input, e.g. ``kin_energy_MeV``."""


class RefEnumAction(
    _RefAccessMixin, WritableActionMixin[ImpactXSimulator], EnumVariable
):
    """The reference-particle species."""


class RefFinalAction(ReadOnlyActionMixin[ImpactXSimulator], ScalarVariable):
    """A reference-particle quantity after tracking, e.g. final ``s`` or energy."""

    key: str

    def _get(self, simulator: Any) -> Any:
        return getattr(_check(simulator).results["ref_final"], self.key)


class _SimAccessMixin:
    """Read or write one ImpactX simulation setting."""

    key: str

    def _get(self, simulator: Any) -> Any:
        return _check(simulator).settings.get(self.key)

    def _set(self, simulator: Any, value: Any) -> None:
        _check(simulator).settings[self.key] = value


class SimScalarAction(
    _SimAccessMixin, WritableActionMixin[ImpactXSimulator], ScalarVariable
):
    """A float simulation setting, e.g. ``mlmg_relative_tolerance``."""


class SimIntAction(_SimAccessMixin, WritableActionMixin[ImpactXSimulator], IntVariable):
    """An integer simulation setting, e.g. ``particle_shape`` or ``periods``."""


class SimBoolAction(
    _SimAccessMixin, WritableActionMixin[ImpactXSimulator], BoolVariable
):
    """A boolean simulation setting, e.g. ``csr`` or ``isr``."""


class SimEnumAction(
    _SimAccessMixin, WritableActionMixin[ImpactXSimulator], EnumVariable
):
    """A simulation setting with a fixed option set, e.g. ``space_charge``."""


class MomentAction(ReadOnlyActionMixin[ImpactXSimulator], ScalarVariable):
    """One beam moment at the end of the lattice."""

    moment_name: str

    def _get(self, simulator: Any) -> Any:
        return _check(simulator).results["moments"][self.moment_name]


class MomentHistoryAction(ReadOnlyActionMixin[ImpactXSimulator], NDVariable):
    """One beam moment as a function of ``s``.

    ``shape`` is set from ``simulator.n_steps``, which equals ``periods * sum(nslice)``
    exactly, so a mismatch means the lattice changed after the model was built. This
    used to pad or truncate to the declared shape; truncation silently discarded
    s-points with no NaN to signal it, and left ``run_info:n_steps`` contradicting the
    array length. It now raises.
    """

    moment_name: str

    def _get(self, simulator: Any) -> Any:
        history = _check(simulator).results["moments_history"]
        values = np.asarray(history[self.moment_name], dtype=self.dtype)
        expected = self.shape[0]
        if values.shape[0] != expected:
            raise RuntimeError(
                f"Variable {self.name!r} declares {expected} s-points but the run "
                f"produced {values.shape[0]}. n_steps is periods * sum(nslice), so the "
                "lattice or the period count changed after the model was built. "
                "Rebuild it with LUMEImpactXModel.from_simulator(simulator)."
            )
        return values


class RunInfoAction(ReadOnlyActionMixin[ImpactXSimulator], ScalarVariable):
    """A scalar fact about the last run, e.g. ``run_time`` or ``n_particles``."""

    key: str

    def _get(self, simulator: Any) -> Any:
        return _check(simulator).results[self.key]


class OpticsAction(ReadOnlyActionMixin[ImpactXSimulator], NDVariable):
    """A linear-optics array from the last run: a transfer map or its s grid.

    ImpactX computes these from the lattice at the initial reference particle, before
    tracking, so they describe the lattice as configured rather than the tracked beam.
    """

    key: str

    def _get(self, simulator: Any) -> Any:
        results = _check(simulator).results
        if self.key not in results:
            raise RuntimeError(
                f"Variable {self.name!r} needs {self.key!r}, which the last run did not "
                "produce. Linear optics are unavailable for this lattice."
            )
        return np.asarray(results[self.key], dtype=self.dtype)


class ParticleGroupAction(WritableActionMixin[ImpactXSimulator], ParticleGroupVariable):
    """A bunch exposed as a ``ParticleGroup``.

    Only ``initial_particles`` may be written; every other bunch is an output.
    """

    tool_name: str

    @model_validator(mode="after")
    def _only_initial_particles_is_writable(self) -> "ParticleGroupAction":
        if self.tool_name != "initial_particles" and not self.read_only:
            raise ValueError(
                f"Particle group {self.tool_name!r} is an output; construct it with "
                "read_only=True."
            )
        return self

    def _get(self, simulator: Any) -> Any:
        sim = _check(simulator)
        if self.tool_name == "initial_particles":
            return sim.initial_particles
        return sim.final_particles

    def _set(self, simulator: Any, value: Any) -> None:
        _check(simulator).initial_particles = value
