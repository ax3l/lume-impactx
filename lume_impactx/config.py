"""Generation of LUME action variables from an :class:`ImpactXSimulator`.

``make_actions(simulator, config)`` walks the lattice, the reference particle, the
simulation settings and the last run's results, and returns one
:class:`~lume.actions.Action` per exposed quantity.

Element variables are generated **generically**, from pybind property introspection
(see :mod:`lume_impactx.elements`), rather than from a hand-written config class per
element type as ``impact/model/config.py`` does for Impact-T. ImpactX has 38 element
types; enumerating their attributes by hand would be ~400 fields that go stale on every
release, and ``property.fset`` already carries writability more accurately than a
hand-maintained table would.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from lume.actions import Action
from pydantic import BaseModel

from lume_impactx.actions import (
    EleBoolAction,
    EleEnumAction,
    EleIntAction,
    EleNDAction,
    EleScalarAction,
    EleStrAction,
    MomentAction,
    MomentHistoryAction,
    OpticsAction,
    ParticleGroupAction,
    RefAction,
    RefEnumAction,
    RefFinalAction,
    RunInfoAction,
    SimBoolAction,
    SimEnumAction,
    SimIntAction,
    SimScalarAction,
)
from lume_impactx.elements import element_attribute_schema, element_type
from lume_impactx.simulator import ImpactXSimulator
from lume_impactx.units import (
    MOMENT_ALIASES,
    MOMENT_UNITS,
    REF_UNITS,
    element_unit,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AttributeConfig",
    "ElementsConfig",
    "MomentsConfig",
    "RefConfig",
    "SimConfig",
    "ParticlesConfig",
    "RunInfoConfig",
    "OpticsConfig",
    "VariableMappingConfig",
    "make_actions",
]

#: String element attributes with a fixed option set. Verified empirically against
#: ImpactX 26.08 by probing which values the setter accepts -- anything not listed here
#: becomes a plain ``StrVariable``, which is the safe default.
ELEMENT_ENUMS: dict[tuple[str, str], list[str]] = {
    ("Aperture", "shape"): ["rectangular", "elliptical"],
    ("Aperture", "action"): ["transmit", "absorb"],
    ("PolygonAperture", "action"): ["transmit", "absorb"],
    ("DipEdge", "model"): ["linear", "nonlinear"],
    ("DipEdge", "location"): ["entry", "exit"],
}

#: Beam moments that only mean something with the matching ImpactX setting enabled.
#: ``beam_moments()`` reports the spin moments unconditionally -- as exact zeros when
#: ``sim.spin`` is off -- so without this gate a plain run grows twelve variables that
#: are structurally present and physically meaningless.
CONDITIONAL_MOMENTS: dict[str, tuple[str, ...]] = {
    "spin": ("mean_sx", "mean_sy", "mean_sz", "sigma_sx", "sigma_sy", "sigma_sz"),
    "eigenemittances": ("emittance_1", "emittance_2", "emittance_3"),
}

#: Element attributes that must stay read-only regardless of what ImpactX allows,
#: because writing them would invalidate variables already generated from them. Each
#: entry is annotated with what it would break.
STRUCTURAL_ELEMENT_ATTRIBUTES = frozenset(
    {
        "nslice",  # sets n_steps, and so every s-series NDVariable.shape
        "name",  # part of the variable names themselves
        "unit",  # selects the unit of k / bscale, which is baked into their labels
    }
)

#: Simulation settings exposed as variables, with their option sets where they are
#: enumerated. ``periods`` is read-only for the same reason ``nslice`` is.
SIM_ENUMS: dict[str, list[Any]] = {
    "space_charge": [False, "2D", "2p5D", "3D", "Gauss3D", "Gauss2p5D"],
    "poisson_solver": ["multigrid", "fft"],
    "particle_bc": ["open", "periodic", "absorbing", "reflecting"],
}
SIM_BOOLS = (
    "csr",
    "isr",
    "isr_on_ref_part",
    "spin",
    "eigenemittances",
    "slice_step_diagnostics",
    "diagnostics",
)
SIM_INTS = ("particle_shape", "csr_bins", "isr_order", "periods", "max_level")
SIM_FLOATS = (
    "mlmg_relative_tolerance",
    "mlmg_absolute_tolerance",
    "space_charge_gauss_taylor_delta",
    "space_charge_gauss_long_scale",
)
STRUCTURAL_SIM_SETTINGS = frozenset({"periods"})

#: Reference-particle inputs that can be written.
REF_INPUT_KEYS = ("kin_energy_MeV", "charge_qe", "mass_MeV")
#: Reference-particle quantities read back after tracking.
REF_FINAL_KEYS = (
    "s",
    "t",
    "x",
    "y",
    "z",
    "px",
    "py",
    "pz",
    "pt",
    "gamma",
    "beta_gamma",
    "mass_MeV",
    "charge_qe",
)
#: Species ImpactX knows natively.
SPECIES_OPTIONS = ["electron", "positron", "proton", "H-"]

#: Scalar facts about the last run.
RUN_INFO_UNITS: dict[str, str | None] = {
    "n_particles": None,
    "n_steps": None,
    "run_time": "s",
}


class AttributeConfig(BaseModel):
    """Per-attribute overrides.

    Attributes
    ----------
    read_only : bool or None
        None means infer from the element's property setter.
    unit : str or None
        None means look the unit up in :mod:`lume_impactx.units`.
    value_range : tuple of float or None
        Optional validation range, also surfaced as PV control limits by ``lume-pva``.
    """

    read_only: bool | None = None
    unit: str | None = None
    value_range: tuple[float, float] | None = None


class ElementsConfig(BaseModel):
    """How lattice element attributes become variables."""

    pattern: str = "ele:{name}:{attrib}"
    duplicate_pattern: str = "ele:{name}#{occurrence}:{attrib}"
    unnamed_pattern: str = "ele:{type}#{index}:{attrib}"
    include_unnamed: bool = False
    include_sequences: bool = False
    include_kinds: list[str] | None = None
    exclude_kinds: list[str] = []
    exclude_attributes: list[str] = []
    attributes: dict[str, AttributeConfig] = {}
    control_to_tool_name: dict[str, str] | None = None


class MomentsConfig(BaseModel):
    """How beam moments become variables.

    Moments gated by an ImpactX setting -- the spin moments, the eigenemittances -- are
    only generated when that setting is on; see :data:`CONDITIONAL_MOMENTS`. Pass an
    explicit ``include`` list to bypass the gate.
    """

    pattern: str = "moment:{name}"
    final_pattern: str = "moment_final:{name}"
    include_history: bool = True
    include_final: bool = True
    drop_legacy_aliases: bool = True
    include: list[str] | None = None


class RefConfig(BaseModel):
    """How the reference particle becomes variables."""

    pattern: str = "ref:{key}"
    final_pattern: str = "ref_final:{key}"
    include_species: bool = True
    include_final: bool = True


class SimConfig(BaseModel):
    """How ImpactX simulation settings become variables.

    Only settings the simulator actually carries are exposed; see
    :func:`_make_sim_actions` for why an unset ImpactX parameter cannot be one.
    """

    pattern: str = "sim:{key}"
    include: list[str] | None = None


class ParticlesConfig(BaseModel):
    """How bunches become ``ParticleGroup`` variables."""

    pattern: str = "particles:{name}"


class RunInfoConfig(BaseModel):
    """How per-run facts become variables."""

    pattern: str = "run_info:{key}"


class OpticsConfig(BaseModel):
    """How the lattice's linear transfer maps become variables.

    The ImpactX counterpart of lume-bmad's ``mat6`` output. ``cumulative_maps`` runs
    from the lattice start to each element's exit, so its last entry is the
    whole-lattice map.
    """

    pattern: str = "optics:{key}"
    include: list[str] | None = None


class VariableMappingConfig(BaseModel):
    """The whole mapping. Set any section to None to skip that category.

    Attributes
    ----------
    prefix : str
        Prepended to every generated variable name. Needed when composing several
        ImpactX sections into a ``StagedModel``: two sections usually reuse element
        names, and ``StagedModel`` refuses duplicate variable names across stages.
        For example ``prefix="linac:"`` yields ``linac:ele:quad1:k``.
    """

    prefix: str = ""
    elements: ElementsConfig | None = ElementsConfig()
    moments: MomentsConfig | None = MomentsConfig()
    ref: RefConfig | None = RefConfig()
    sim: SimConfig | None = SimConfig()
    particles: ParticlesConfig | None = ParticlesConfig()
    run_info: RunInfoConfig | None = RunInfoConfig()
    optics: OpticsConfig | None = OpticsConfig()


# --------------------------------------------------------------------------------------
# Element variables
# --------------------------------------------------------------------------------------


def _element_labels(lattice: list, config: ElementsConfig) -> list[tuple[str, dict]]:
    """Work out one naming pattern and its fields per lattice element.

    Element names are not unique -- ImpactX's own ``test_xopt.py`` uses ``"quad1"``
    twice -- so repeats get an occurrence suffix.
    """
    counts: dict[str, int] = {}
    for element in lattice:
        name = getattr(element, "name", None)
        if name:
            counts[name] = counts.get(name, 0) + 1

    seen: dict[str, int] = {}
    labels = []
    for index, element in enumerate(lattice):
        name = getattr(element, "name", None)
        etype = element_type(element)
        if not name:
            labels.append((config.unnamed_pattern, {"type": etype, "index": index}))
            continue
        if config.control_to_tool_name:
            reverse = {v: k for k, v in config.control_to_tool_name.items()}
            name = reverse.get(name, name)
        if counts.get(name, 0) > 1:
            seen[name] = seen.get(name, 0) + 1
            labels.append(
                (
                    config.duplicate_pattern,
                    {
                        "name": name,
                        "occurrence": seen[name],
                        "type": etype,
                        "index": index,
                    },
                )
            )
        else:
            labels.append(
                (config.pattern, {"name": name, "type": etype, "index": index})
            )
    return labels


def _make_element_action(
    *,
    var_name: str,
    index: int,
    ele_name: str,
    etype: str,
    attribute: str,
    value: Any,
    read_only: bool,
    config: ElementsConfig,
    unit_flag: int | None = None,
    lattice_size: int = 0,
) -> Action | None:
    """Dispatch one element attribute onto the right Variable subclass."""
    override = config.attributes.get(attribute, AttributeConfig())
    if override.read_only is not None:
        read_only = override.read_only
    if override.unit is not None:
        unit = override.unit
    else:
        unit = element_unit(etype, attribute, unit_flag=unit_flag)
    common = dict(
        name=var_name,
        ele_index=index,
        ele_name=ele_name,
        ele_type=etype,
        lattice_size=lattice_size,
        attribute=attribute,
        read_only=read_only,
    )

    if isinstance(value, bool):
        return EleBoolAction(**common, default_value=value)
    if isinstance(value, (int, np.integer)):
        return EleIntAction(**common, default_value=int(value), unit=unit)
    if isinstance(value, (float, np.floating)):
        return EleScalarAction(
            **common,
            default_value=float(value),
            unit=unit,
            value_range=override.value_range,
        )
    if isinstance(value, str):
        options = ELEMENT_ENUMS.get((etype, attribute))
        if options is not None:
            return EleEnumAction(**common, default_value=value, options=options)
        return EleStrAction(**common, default_value=value)
    if isinstance(value, (list, tuple, np.ndarray)):
        if not config.include_sequences:
            return None
        array = np.asarray(value, dtype=np.float64)
        return EleNDAction(
            **common,
            shape=array.shape,
            dtype=np.dtype(np.float64),
            default_value=array,
            unit=unit,
        )
    logger.debug(
        "Skipping %s.%s: unsupported value type %s", etype, attribute, type(value)
    )
    return None


def _make_element_actions(
    simulator: ImpactXSimulator, config: ElementsConfig
) -> list[Action]:
    actions: list[Action] = []
    labels = _element_labels(simulator.lattice, config)

    for index, (element, (pattern, fields)) in enumerate(
        zip(simulator.lattice, labels)
    ):
        etype = element_type(element)
        if config.include_kinds is not None and etype not in config.include_kinds:
            continue
        if etype in config.exclude_kinds:
            continue
        if "name" not in fields and not config.include_unnamed:
            continue

        for attribute, writable in element_attribute_schema(element).items():
            if attribute in config.exclude_attributes:
                continue
            if attribute in config.attributes and config.attributes[attribute] is None:
                continue
            read_only = not writable or attribute in STRUCTURAL_ELEMENT_ATTRIBUTES
            try:
                value = getattr(element, attribute)
            except Exception:  # pragma: no cover - guarded in element_attribute_schema
                continue
            if value is None:
                continue
            action = _make_element_action(
                unit_flag=getattr(element, "unit", None),
                lattice_size=len(simulator.lattice),
                var_name=pattern.format(attrib=attribute, **fields),
                index=index,
                ele_name=str(fields.get("name", "")),
                etype=etype,
                attribute=attribute,
                value=value,
                read_only=read_only,
                config=config,
            )
            if action is not None:
                actions.append(action)
    return actions


# --------------------------------------------------------------------------------------
# Moment, reference, settings, particle and run-info variables
# --------------------------------------------------------------------------------------


def _make_moment_actions(
    simulator: ImpactXSimulator, config: MomentsConfig
) -> list[Action]:
    results = simulator.results
    moments = results["moments"]
    history = results["moments_history"]
    n_steps = results["n_steps"]

    names = config.include
    if names is None:
        names = sorted(moments)
        if config.drop_legacy_aliases:
            # beam_moments() reports both mean_x and x_mean, sigma_x and sig_x.
            names = [n for n in names if n not in MOMENT_ALIASES]
        for setting, gated in CONDITIONAL_MOMENTS.items():
            if not simulator.settings.get(setting):
                names = [n for n in names if n not in gated]

    actions: list[Action] = []
    for name in names:
        unit = MOMENT_UNITS.get(name)
        if config.include_final and name in moments:
            actions.append(
                MomentAction(
                    name=config.final_pattern.format(name=name),
                    moment_name=name,
                    unit=unit,
                    read_only=True,
                )
            )
        if config.include_history and history is not None and name in history:
            actions.append(
                MomentHistoryAction(
                    name=config.pattern.format(name=name),
                    moment_name=name,
                    shape=(n_steps,),
                    dtype=np.dtype(np.float64),
                    unit=unit,
                    read_only=True,
                )
            )
    return actions


def _make_ref_actions(simulator: ImpactXSimulator, config: RefConfig) -> list[Action]:
    # A stage seeded from an upstream section takes its energy from that section, so a
    # write here would be silently discarded on the next track. Expose it read-only
    # rather than accepting a value and ignoring it.
    energy_from_upstream = simulator.ref_origin is not None

    actions: list[Action] = []
    for key in REF_INPUT_KEYS:
        if key in simulator.ref:
            actions.append(
                RefAction(
                    name=config.pattern.format(key=key),
                    key=key,
                    default_value=float(simulator.ref[key]),
                    unit=REF_UNITS.get(key),
                    read_only=energy_from_upstream and key == "kin_energy_MeV",
                )
            )
    if config.include_species and "species" in simulator.ref:
        actions.append(
            RefEnumAction(
                name=config.pattern.format(key="species"),
                key="species",
                default_value=simulator.ref["species"],
                options=list(SPECIES_OPTIONS),
            )
        )
    if config.include_final:
        for key in REF_FINAL_KEYS:
            actions.append(
                RefFinalAction(
                    name=config.final_pattern.format(key=key),
                    key=key,
                    unit=REF_UNITS.get(key),
                    read_only=True,
                )
            )
    return actions


def _make_sim_actions(simulator: ImpactXSimulator, config: SimConfig) -> list[Action]:
    known = list(SIM_ENUMS) + list(SIM_BOOLS) + list(SIM_INTS) + list(SIM_FLOATS)
    keys = known if config.include is None else config.include

    # Only expose settings that actually have a value. ImpactX has no readable default
    # for an unset parameter -- reading one raises "algo.csr is not set yet" -- so a
    # variable for a setting the simulator will not apply would have nothing to report,
    # and `get()` would fail validation on None. Add the setting at construction
    # (settings={"csr": True}) to get a variable for it.
    actions: list[Action] = []
    for key in keys:
        if key not in simulator.settings:
            if config.include is not None:
                logger.debug(
                    "Skipping sim variable %r: not in the simulator's settings.", key
                )
            continue
        read_only = key in STRUCTURAL_SIM_SETTINGS
        var_name = config.pattern.format(key=key)
        current = simulator.settings[key]
        if key in SIM_ENUMS:
            options = list(SIM_ENUMS[key])
            if current is not None and current not in options:
                options.append(current)
            actions.append(
                SimEnumAction(
                    name=var_name,
                    key=key,
                    options=options,
                    default_value=current,
                    read_only=read_only,
                )
            )
        elif key in SIM_BOOLS:
            actions.append(
                SimBoolAction(
                    name=var_name,
                    key=key,
                    default_value=bool(current) if current is not None else None,
                    read_only=read_only,
                )
            )
        elif key in SIM_INTS:
            actions.append(
                SimIntAction(
                    name=var_name,
                    key=key,
                    default_value=int(current) if current is not None else None,
                    read_only=read_only,
                )
            )
        elif key in SIM_FLOATS:
            actions.append(
                SimScalarAction(
                    name=var_name,
                    key=key,
                    default_value=float(current) if current is not None else None,
                    read_only=read_only,
                )
            )
    return actions


def _make_particle_actions(
    simulator: ImpactXSimulator, config: ParticlesConfig
) -> list[Action]:
    actions: list[Action] = [
        ParticleGroupAction(
            name=config.pattern.format(name="final_particles"),
            tool_name="final_particles",
            read_only=True,
        )
    ]
    if simulator.initial_particles is not None:
        actions.append(
            ParticleGroupAction(
                name=config.pattern.format(name="initial_particles"),
                tool_name="initial_particles",
            )
        )
    return actions


def _make_run_info_actions(
    simulator: ImpactXSimulator, config: RunInfoConfig
) -> list[Action]:
    return [
        RunInfoAction(
            name=config.pattern.format(key=key),
            key=key,
            unit=unit,
            read_only=True,
        )
        for key, unit in RUN_INFO_UNITS.items()
    ]


#: Linear-optics arrays and their units.
OPTICS_KEYS: dict[str, str | None] = {
    "transfer_map": None,
    "cumulative_maps": None,
    "map_s": "m",
}


def _make_optics_actions(
    simulator: ImpactXSimulator, config: OpticsConfig
) -> list[Action]:
    results = simulator.results
    keys = config.include if config.include is not None else list(OPTICS_KEYS)

    actions: list[Action] = []
    for key in keys:
        if key not in results:
            continue
        array = np.asarray(results[key], dtype=np.float64)
        actions.append(
            OpticsAction(
                name=config.pattern.format(key=key),
                key=key,
                shape=array.shape,
                dtype=np.dtype(np.float64),
                unit=OPTICS_KEYS.get(key),
                read_only=True,
            )
        )
    return actions


def make_actions(
    simulator: ImpactXSimulator,
    config: VariableMappingConfig | None = None,
) -> list[Action]:
    """Generate the action variables for a simulator.

    Parameters
    ----------
    simulator : ImpactXSimulator
        A simulator that has already tracked once, so moments and their sizes are known.
    config : VariableMappingConfig, optional
        Which categories to generate and how to name them.

    Returns
    -------
    list of Action
        Ready to hand to :class:`~lume_impactx.model.LUMEImpactXModel`.

    Raises
    ------
    ValueError
        If two variables would end up with the same name.
    """
    config = config or VariableMappingConfig()
    actions: list[Action] = []

    if config.elements is not None:
        actions += _make_element_actions(simulator, config.elements)
    if config.moments is not None:
        actions += _make_moment_actions(simulator, config.moments)
    if config.ref is not None:
        actions += _make_ref_actions(simulator, config.ref)
    if config.sim is not None:
        actions += _make_sim_actions(simulator, config.sim)
    if config.particles is not None:
        actions += _make_particle_actions(simulator, config.particles)
    if config.run_info is not None:
        actions += _make_run_info_actions(simulator, config.run_info)
    if config.optics is not None:
        actions += _make_optics_actions(simulator, config.optics)

    if config.prefix:
        actions = [
            a.model_copy(update={"name": config.prefix + a.name}) for a in actions
        ]

    seen: set[str] = set()
    duplicates = {a.name for a in actions if a.name in seen or seen.add(a.name)}
    if duplicates:
        raise ValueError(
            f"Duplicate variable names generated: {sorted(duplicates)}. Adjust the "
            "naming patterns in VariableMappingConfig."
        )
    return actions
