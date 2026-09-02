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
from lume.exceptions import ReadOnlyError
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
    "SimNDAction",
    "MomentAction",
    "MomentHistoryAction",
    "RunInfoAction",
    "OpticsAction",
    "ParticleGroupAction",
    "_ElementByNameMixin",
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


class SimNDAction(_SimAccessMixin, WritableActionMixin[ImpactXSimulator], NDVariable):
    """An array-valued simulation setting, e.g. ``n_cell`` or ``prob_relative``.

    Writable because the simulator rebuilds the session on every track. ImpactX makes
    ``n_cell`` read-only once ``init_grids()`` has run, so a model holding one live
    session could not offer this at all.
    """

    def _get(self, simulator: Any) -> Any:
        return np.asarray(_check(simulator).settings[self.key], dtype=self.dtype)

    def _set(self, simulator: Any, value: Any) -> None:
        array = np.asarray(value, dtype=self.dtype)
        if array.shape != self.shape:
            raise ValueError(
                f"{self.name!r} expects shape {self.shape}, got {array.shape}."
            )
        # ImpactX wants a plain Python list of ints for the mesh settings.
        as_list = array.tolist()
        if np.issubdtype(self.dtype, np.integer):
            as_list = [int(v) for v in as_list]
        _check(simulator).settings[self.key] = as_list


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


# --------------------------------------------------------------------------------------
# Addressing by name
#
# The generated variables above address elements by lattice *index*, which is right for
# config.py: it walks the lattice, so the index is known and authoritative, and a shifted
# index is caught rather than silently reading the wrong magnet.
#
# Hand-written variables are the other case. A virtual accelerator knows "QF01", not
# "index 37", and its conversion logic -- kG to k1 through the magnetic rigidity, say --
# belongs in the facility's own repository, not here. These are the bases it subclasses,
# shaped like lume-cheetah's so code written against that contract ports across:
#
#     class QuadrupoleBCTRL(ImpactXWritableScalarVariable):
#         unit: str = "kG"
#         def _get(self, simulator):
#             element, energy = self._resolve_element_and_energy(simulator, self.element_name)
#             return element.k * element.ds * 33.356 * energy / 1e9
# --------------------------------------------------------------------------------------


class _ElementByNameMixin:
    """Resolve an element by name, and the reference energy where it sits.

    ImpactX element names are not unique -- a lattice may use one element twice, and
    :func:`~lume_impactx.interfaces.bmad.lattice_from_tao` splits a single Bmad element
    into several -- so repeats are addressed as ``NAME##2``, ``NAME##3`` in beam order,
    the convention :attr:`~lume_impactx.simulator.ImpactXSimulator.ele` and the captured
    bunches share. Lookup folds case, because Tao returns names upper case.

    No lattice-shape check is needed here, unlike :class:`_ElementAccessMixin`: a name
    still finds its element after the lattice is edited, which is the whole point of
    addressing this way.
    """

    element_name: str

    @staticmethod
    def _resolve_element_and_energy(simulator: Any, element_name: str):
        """The element and the reference total energy in eV at it.

        Returns both because almost every engineering-unit conversion needs the second:
        a quadrupole's kG is its gradient times the magnetic rigidity, which is set by
        the reference momentum *at that element* -- and that differs either side of an
        accelerating cavity. The signature matches lume-cheetah's
        ``_resolve_element_and_energy`` deliberately.
        """
        sim = _check(simulator)
        element = sim.ele[element_name]
        try:
            energy = sim.reference_energy_at(element_name)
        except KeyError:  # pragma: no cover - ele[] would already have raised
            energy = None
        return element, energy

    def _element(self, simulator: Any):
        element, _ = self._resolve_element_and_energy(simulator, self.element_name)
        return element


class ImpactXWritableActionMixin(
    _ElementByNameMixin, WritableActionMixin[ImpactXSimulator]
):
    """Read and write one attribute of a named element.

    Subclasses that need a unit conversion override ``_get``/``_set``; those that just
    want the raw attribute set :attr:`attribute` and inherit these.
    """

    #: Attribute to read and write. Subclasses overriding _get/_set may leave it unset.
    attribute: str = ""

    def _get(self, simulator: Any) -> Any:
        return getattr(self._element(simulator), self.attribute)

    def _set(self, simulator: Any, value: Any) -> None:
        setattr(self._element(simulator), self.attribute, value)


class ImpactXReadOnlyActionMixin(
    ImpactXWritableActionMixin, ReadOnlyActionMixin[ImpactXSimulator]
):
    """A readback: the same ``_get`` as its writable counterpart, and no ``_set``.

    Inheriting from the writable mixin is what lets a facility write ``BACT`` as a
    one-line subclass of its ``BCTRL``, reusing the conversion rather than repeating it.
    """

    read_only: bool = True

    def _set(self, simulator: Any, value: Any) -> None:
        raise ReadOnlyError(f"{self.name} is read-only.")


class ImpactXWritableScalarVariable(ImpactXWritableActionMixin, ScalarVariable):
    """A writable float on a named element."""


class ImpactXReadOnlyScalarVariable(ImpactXReadOnlyActionMixin, ScalarVariable):
    """A read-only float on a named element."""


class ImpactXWritableIntVariable(ImpactXWritableActionMixin, IntVariable):
    """A writable integer on a named element, e.g. ``nslice``."""


class ImpactXWritableNDVariable(ImpactXWritableActionMixin, NDVariable):
    """A writable array on a named element, e.g. ``k_normal``."""


class ImpactXReadOnlyNDVariable(ImpactXReadOnlyActionMixin, NDVariable):
    """A read-only array derived from a named element."""


class ImpactXReadOnlyEnumVariable(ImpactXReadOnlyActionMixin, EnumVariable):
    """A read-only enumerated readback, e.g. a control state."""


class ImpactXBunchAtElementVariable(
    ReadOnlyActionMixin[ImpactXSimulator], ParticleGroupVariable
):
    """The captured bunch at a named element.

    The element must be in the simulator's
    :attr:`~lume_impactx.simulator.ImpactXSimulator.capture_at`, or there is nothing to
    return; :meth:`~lume_impactx.simulator.ImpactXSimulator.from_tao` fills that in from
    the Bmad markers and monitors. This is the variable a screen image is built from.
    """

    element_name: str

    def _get(self, simulator: Any) -> Any:
        return _check(simulator).particles[self.element_name]
