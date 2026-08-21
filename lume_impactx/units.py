"""Unit tables for ImpactX quantities.

Action classes never hard-code a unit; they look one up here, the way
``impact/model/config.py`` uses ``ELE_UNITS`` / ``STAT_UNITS``. Units are plain
strings, which is what ``lume.variables.Variable.unit`` takes and what ``lume-pva``
puts into a PV's ``display.units``.

Sources
-------
Beam moments follow ImpactX's reduced-diagnostics documentation
(``docs/source/dataanalysis/dataanalysis.rst``): positions, sigmas, emittances,
Twiss beta and dispersion in metres; momenta, Twiss alpha and momentum dispersion
dimensionless; ``charge_C`` in Coulomb.

Phase-space momenta ``px``/``py``/``pt`` are normalized by the reference momentum
and are therefore dimensionless, *not* eV/c. See
``docs/source/theory/coordinates_units.rst``.
"""

from __future__ import annotations

#: Unit for every key that ``ImpactXParticleContainer.beam_moments()`` can return.
MOMENT_UNITS: dict[str, str | None] = {
    "step": None,
    "period": None,
    "s": "m",
    "charge_C": "C",
}
for _plane in ("x", "y", "t"):
    MOMENT_UNITS[f"mean_{_plane}"] = "m"
    MOMENT_UNITS[f"min_{_plane}"] = "m"
    MOMENT_UNITS[f"max_{_plane}"] = "m"
    MOMENT_UNITS[f"sigma_{_plane}"] = "m"
    MOMENT_UNITS[f"emittance_{_plane}"] = "m"
    MOMENT_UNITS[f"emittance_{_plane}n"] = "m"
    MOMENT_UNITS[f"beta_{_plane}"] = "m"
    MOMENT_UNITS[f"alpha_{_plane}"] = None
    # momenta are normalized by the reference momentum -> dimensionless
    MOMENT_UNITS[f"mean_p{_plane}"] = None
    MOMENT_UNITS[f"min_p{_plane}"] = None
    MOMENT_UNITS[f"max_p{_plane}"] = None
    MOMENT_UNITS[f"sigma_p{_plane}"] = None
for _plane in ("x", "y"):
    MOMENT_UNITS[f"dispersion_{_plane}"] = "m"
    MOMENT_UNITS[f"dispersion_p{_plane}"] = None
# with sim.eigenemittances = True
for _i in (1, 2, 3):
    MOMENT_UNITS[f"emittance_{_i}"] = "m"
# with sim.spin = True
for _axis in ("sx", "sy", "sz"):
    MOMENT_UNITS[f"mean_{_axis}"] = None
    MOMENT_UNITS[f"sigma_{_axis}"] = None
del _plane, _i, _axis

#: Legacy spellings that ``beam_moments()`` returns alongside the canonical names,
#: e.g. it reports both ``mean_x`` and ``x_mean``. Mapped legacy -> canonical so the
#: duplicates can be dropped when generating variables.
MOMENT_ALIASES: dict[str, str] = {}
for _canonical in list(MOMENT_UNITS):
    for _prefix in ("mean_", "min_", "max_"):
        if _canonical.startswith(_prefix):
            _rest = _canonical[len(_prefix) :]
            MOMENT_ALIASES[f"{_rest}_{_prefix[:-1]}"] = _canonical
    if _canonical.startswith("sigma_"):
        MOMENT_ALIASES[f"sig_{_canonical[len('sigma_') :]}"] = _canonical
del _canonical, _prefix, _rest

#: Reference-particle attribute units. ``t`` and ``z`` are lengths because ImpactX
#: carries ``c * time`` rather than time.
REF_UNITS: dict[str, str | None] = {
    "x": "m",
    "y": "m",
    "z": "m",
    "t": "m",
    "px": None,
    "py": None,
    "pz": None,
    "pt": None,
    "s": "m",
    "sedge": "m",
    "mass": "kg",
    "charge": "C",
    "mass_MeV": "MeV",
    "charge_qe": None,
    "kin_energy_MeV": "MeV",
    "beta": None,
    "gamma": None,
    "beta_gamma": None,
    "rigidity_Tm": "T*m",
    "qm_ratio_SI": "C/kg",
    "gyromagnetic_anomaly": None,
}

#: Element attribute units that mean the same thing for every element type.
ELE_UNITS: dict[str, str | None] = {
    "ds": "m",
    "dx": "m",
    "dy": "m",
    "rotation": "deg",
    "aperture_x": "m",
    "aperture_y": "m",
    "repeat_x": "m",
    "repeat_y": "m",
    "nslice": None,
    "mapsteps": None,
    "int_order": None,
    "name": None,
    "shift_odd_x": None,
    "min_radius2": "m^2",
    "vertices_x": "m",
    "vertices_y": "m",
}

#: Element attribute units that depend on the element type. Consulted before
#: :data:`ELE_UNITS`. Keys are ``(element_type, attribute)``.
ELE_UNITS_BY_TYPE: dict[tuple[str, str], str | None] = {
    ("Quad", "k"): "1/m^2",
    ("ChrQuad", "k"): "1/m^2",
    ("ExactQuad", "k"): "1/m^2",
    ("QuadEdge", "k"): "1/m^2",
    ("SoftQuadrupole", "gscale"): "1/m^2",
    ("CFbend", "k"): "1/m^2",
    ("ConstF", "kx"): "1/m",
    ("ConstF", "ky"): "1/m",
    ("ConstF", "kt"): "1/m",
    ("Sol", "ks"): "1/m",
    ("SoftSolenoid", "bscale"): "1/m",
    ("ChrPlasmaLens", "k"): "1/m^2",
    ("TaperedPL", "k"): "1/m",
    ("TaperedPL", "taper"): "1/m",
    ("Sbend", "rc"): "m",
    ("ExactSbend", "phi"): "rad",
    ("ExactSbend", "B"): "T",
    ("CFbend", "rc"): "m",
    ("DipEdge", "psi"): "rad",
    ("DipEdge", "rc"): "m",
    ("DipEdge", "g"): "m",
    ("ThinDipole", "theta"): "rad",
    ("ThinDipole", "rc"): "m",
    ("RFCavity", "escale"): "1/m",
    ("RFCavity", "freq"): "Hz",
    ("RFCavity", "phase"): "deg",
    ("ShortRF", "V"): None,
    ("ShortRF", "freq"): "Hz",
    ("ShortRF", "phase"): "deg",
    ("Buncher", "V"): None,
    ("Buncher", "k"): "1/m",
    ("ChrAcc", "ez"): "1/m",
    ("ChrAcc", "bz"): "1/m",
    ("PRot", "phi_in"): "rad",
    ("PRot", "phi_out"): "rad",
    ("PlaneXYRot", "angle"): "rad",
    ("NonlinearLens", "knll"): "m",
    ("NonlinearLens", "cnll"): "m",
}


#: Attributes whose unit depends on the element's sibling ``unit`` flag:
#: ``(unit=0, unit=1)``. From ImpactX's ``docs/source/usage/python.rst``, e.g.
#: "Quadrupole strength in m^(-2) ... OR Quadrupole strength in T/m (if unit = 1)".
#: The label is resolved when variables are generated, which is why ``unit`` itself is
#: exposed read-only -- otherwise flipping it would leave every strength mislabelled.
UNIT_FLAG_DEPENDENT: dict[tuple[str, str], tuple[str, str]] = {
    ("ChrQuad", "k"): ("1/m^2", "T/m"),
    ("ExactQuad", "k"): ("1/m^2", "T/m"),
    ("QuadEdge", "k"): ("1/m^2", "T/m"),
    ("ChrPlasmaLens", "k"): ("1/m^2", "T/m"),
    ("SoftSolenoid", "bscale"): ("1/m", "T"),
    ("TaperedPL", "k"): ("1/m", "T"),
}

#: Attributes deliberately left unlabelled, with the reason. Recording them stops the
#: same "why is this None?" investigation happening twice.
UNLABELLED_ATTRIBUTES: dict[tuple[str, str], str] = {
    # Arrays of per-order coefficients: "meter^(-m) OR T/meter^(m-1) for m=1,2,3,..",
    # so no single unit string applies to the array.
    ("ExactCFbend", "k_normal"): "per-order multipole coefficients",
    ("ExactCFbend", "k_skew"): "per-order multipole coefficients",
    ("ExactMultipole", "k_normal"): "per-order multipole coefficients",
    ("ExactMultipole", "k_skew"): "per-order multipole coefficients",
    ("Multipole", "K_normal"): "per-order multipole coefficient",
    ("Multipole", "K_skew"): "per-order multipole coefficient",
    # ImpactX 26.08 exposes xkick/ykick but not the sibling `unit`, so the label
    # ("dimensionless" OR "T-m") cannot be resolved from the element.
    ("Kicker", "xkick"): "unit flag not readable in 26.08",
    ("Kicker", "ykick"): "unit flag not readable in 26.08",
    # Dimensionless or model-dependent edge-field coefficients.
    ("DipEdge", "K0"): "dimensionless fringe-field coefficient",
}


def element_unit(
    element_type: str, attribute: str, unit_flag: int | None = None
) -> str | None:
    """Look up the unit for one element attribute.

    Parameters
    ----------
    element_type : str
        The element's ``to_dict()["type"]``, e.g. ``"Quad"``.
    attribute : str
        The attribute name, e.g. ``"k"``.
    unit_flag : int, optional
        The element's ``unit`` attribute, for the strengths whose unit depends on it
        (see :data:`UNIT_FLAG_DEPENDENT`). Treated as 0 when omitted.

    Returns
    -------
    str or None
        The unit string, or None when the quantity is dimensionless or unknown.
    """
    dependent = UNIT_FLAG_DEPENDENT.get((element_type, attribute))
    if dependent is not None:
        return dependent[1] if unit_flag else dependent[0]
    if (element_type, attribute) in ELE_UNITS_BY_TYPE:
        return ELE_UNITS_BY_TYPE[(element_type, attribute)]
    return ELE_UNITS.get(attribute)


def canonical_moment_name(key: str) -> str:
    """Map a legacy ``beam_moments()`` spelling onto its canonical name.

    ``beam_moments()`` reports both ``mean_x`` and ``x_mean``; this collapses the
    two so only one variable is generated.
    """
    return MOMENT_ALIASES.get(key, key)
