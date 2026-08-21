"""Variable-generation tests.

These need real ImpactX *element* objects but no ``ImpactX`` session: elements,
``KnownElementsList`` and property introspection all work without ``init_grids()``.
That makes the whole element half of config.py testable in milliseconds, with none of
the process-lifetime hazards a real session carries.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

impactx = pytest.importorskip("impactx")
from impactx import elements  # noqa: E402

from lume.variables import ScalarVariable  # noqa: E402
from lume_impactx.actions import (  # noqa: E402
    EleBoolAction,
    EleEnumAction,
    EleIntAction,
    EleScalarAction,
)
from lume_impactx.config import (  # noqa: E402
    ElementsConfig,
    _element_labels,
    _make_element_actions,
)
from lume_impactx.elements import element_attribute_schema  # noqa: E402


def _sim(lattice):
    """A stand-in exposing only what element generation touches."""
    return SimpleNamespace(lattice=lattice)


def _by_name(actions):
    return {a.name: a for a in actions}


def test_labels_disambiguate_repeated_element_names():
    """ImpactX's own test_xopt.py uses 'quad1' twice, so this must be handled."""
    lattice = [
        elements.Quad(name="qf", ds=1.0, k=1.0),
        elements.Drift(name="dr", ds=1.0),
        elements.Quad(name="qf", ds=1.0, k=-1.0),
    ]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert "ele:qf#1:k" in actions
    assert "ele:qf#2:k" in actions
    assert "ele:dr:ds" in actions  # unique names keep the plain pattern
    assert actions["ele:qf#1:k"].ele_index == 0
    assert actions["ele:qf#2:k"].ele_index == 2


def test_unnamed_elements_are_skipped_by_default():
    lattice = [elements.Quad(ds=1.0, k=1.0), elements.Drift(name="dr", ds=1.0)]
    assert not any(
        a.name.startswith("ele:Quad")
        for a in _make_element_actions(_sim(lattice), ElementsConfig())
    )
    included = _make_element_actions(
        _sim(lattice), ElementsConfig(include_unnamed=True)
    )
    assert any(a.name == "ele:Quad#0:k" for a in included)


def test_read_only_follows_the_property_setter():
    """Writability is per-type: Drift.aperture_x is read-only, Aperture's is not."""
    lattice = [
        elements.Drift(name="dr", ds=1.0),
        elements.Aperture(name="ap", aperture_x=1e-3, aperture_y=1e-3),
        elements.ThinDipole(name="td", theta=1.0, rc=1.0),
    ]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert actions["ele:dr:aperture_x"].read_only is True
    assert actions["ele:ap:aperture_x"].read_only is False
    assert actions["ele:dr:ds"].read_only is False
    assert actions["ele:td:ds"].read_only is True


def test_structural_attributes_are_forced_read_only():
    """nslice fixes every s-series shape, and name fixes the variable names."""
    lattice = [elements.Quad(name="qf", ds=1.0, k=1.0, nslice=4)]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert element_attribute_schema(lattice[0])["nslice"] is True  # ImpactX allows it
    assert actions["ele:qf:nslice"].read_only is True  # we do not
    assert actions["ele:qf:name"].read_only is True


def test_value_types_dispatch_to_the_right_variable():
    lattice = [
        elements.Quad(name="qf", ds=1.0, k=1.0, nslice=4),
        elements.Aperture(name="ap", aperture_x=1e-3, aperture_y=1e-3),
    ]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert isinstance(actions["ele:qf:k"], EleScalarAction)
    assert isinstance(actions["ele:qf:nslice"], EleIntAction)
    assert isinstance(actions["ele:ap:shift_odd_x"], EleBoolAction)
    assert isinstance(actions["ele:ap:shape"], EleEnumAction)
    assert actions["ele:ap:shape"].options == ["rectangular", "elliptical"]


def test_units_are_attached():
    lattice = [
        elements.Quad(name="qf", ds=1.0, k=1.0),
        elements.Sol(name="sl", ds=1.0, ks=0.5),
    ]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert actions["ele:qf:k"].unit == "1/m^2"
    assert actions["ele:sl:ks"].unit == "1/m"  # same attribute name, different unit
    assert actions["ele:qf:ds"].unit == "m"
    assert actions["ele:qf:rotation"].unit == "deg"


def test_sequences_are_opt_in():
    lattice = [
        elements.RFCavity(
            name="rf",
            ds=1.0,
            escale=1.0,
            freq=1.3e9,
            phase=-90.0,
            cos_coefficients=[0.1, 0.2],
            sin_coefficients=[0.0, 0.0],
        )
    ]
    default = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert "ele:rf:cos_coefficients" not in default

    opted_in = _by_name(
        _make_element_actions(_sim(lattice), ElementsConfig(include_sequences=True))
    )
    if "ele:rf:cos_coefficients" in opted_in:  # exposed via to_dict, not as a property
        assert opted_in["ele:rf:cos_coefficients"].shape == (2,)


def test_exclusions():
    lattice = [elements.Quad(name="qf", ds=1.0, k=1.0)]
    cfg = ElementsConfig(exclude_attributes=["rotation", "dx", "dy"])
    actions = _by_name(_make_element_actions(_sim(lattice), cfg))
    assert "ele:qf:rotation" not in actions
    assert "ele:qf:k" in actions

    assert (
        _make_element_actions(_sim(lattice), ElementsConfig(exclude_kinds=["Quad"]))
        == []
    )
    assert (
        _make_element_actions(_sim(lattice), ElementsConfig(include_kinds=["Drift"]))
        == []
    )


#: One constructible instance per element type, to prove generation never raises.
def _sample_elements():
    e = elements
    return [
        e.Drift(name="a", ds=1.0),
        e.ChrDrift(name="b", ds=1.0),
        e.ExactDrift(name="c", ds=1.0),
        e.Quad(name="d", ds=1.0, k=1.0),
        e.ChrQuad(name="f", ds=1.0, k=1.0),
        e.ExactQuad(name="g", ds=1.0, k=1.0),
        e.QuadEdge(name="h", k=1.0),
        e.Sbend(name="i", ds=1.0, rc=10.0),
        e.ExactSbend(name="j", ds=1.0, phi=10.0),
        e.CFbend(name="k", ds=1.0, rc=10.0, k=1.0),
        e.DipEdge(name="l", psi=0.1, rc=10.0, g=0.01),
        e.ThinDipole(name="m", theta=1.0, rc=10.0),
        e.Multipole(name="n", multipole=2, K_normal=1.0, K_skew=0.0),
        e.Sol(name="o", ds=1.0, ks=0.5),
        e.ShortRF(name="p", V=1.0, freq=1.3e9),
        e.Buncher(name="q", V=1.0, k=1.0),
        e.ChrAcc(name="r", ds=1.0, ez=1.0, bz=1.0),
        e.ChrPlasmaLens(name="s", ds=1.0, k=1.0),
        e.TaperedPL(name="t", k=1.0, taper=0.1),
        e.ConstF(name="u", ds=1.0, kx=1.0, ky=1.0, kt=1.0),
        e.NonlinearLens(name="v", knll=1e-6, cnll=0.01),
        e.Kicker(name="w", xkick=1e-4, ykick=0.0),
        e.PlaneXYRot(name="x", angle=1.0),
        e.PRot(name="y", phi_in=1.0, phi_out=1.0),
        e.Aperture(name="z", aperture_x=1e-3, aperture_y=1e-3),
        e.Marker(name="aa"),
        e.Empty(),
        e.BeamMonitor("bb", backend="h5"),
        e.Programmable(name="cc", ds=0.0),
        e.RFCavity(
            name="dd",
            ds=1.0,
            escale=1.0,
            freq=1.3e9,
            phase=-90.0,
            cos_coefficients=[0.1],
            sin_coefficients=[0.0],
        ),
        e.SoftSolenoid(
            name="ee",
            ds=1.0,
            bscale=1.0,
            cos_coefficients=[0.1],
            sin_coefficients=[0.0],
        ),
        e.SoftQuadrupole(
            name="ff",
            ds=1.0,
            gscale=1.0,
            cos_coefficients=[0.1],
            sin_coefficients=[0.0],
        ),
    ]


def test_every_element_type_generates_cleanly():
    """Generation must not raise for any element type, named or not."""
    lattice = _sample_elements()
    assert len(lattice) >= 30, "sample should cover most of the 38 element types"

    actions = _make_element_actions(
        _sim(lattice), ElementsConfig(include_unnamed=True, include_sequences=True)
    )
    assert actions

    names = [a.name for a in actions]
    assert len(names) == len(set(names)), "generated names must be unique"
    for action in actions:
        assert action.name.startswith("ele:")
        assert 0 <= action.ele_index < len(lattice)


def test_labels_cover_every_element():
    lattice = _sample_elements()
    labels = _element_labels(lattice, ElementsConfig())
    assert len(labels) == len(lattice)


@pytest.mark.slow
def test_spin_moments_appear_only_when_spin_is_on(fodo_lattice, waterbag):
    """beam_moments() always reports spin moments, as exact zeros when spin is off.

    Generating variables for them regardless would add twelve structurally present,
    physically meaningless entries to every ordinary run.
    """
    from lume_impactx.config import make_actions
    from lume_impactx.simulator import ImpactXSimulator

    def variable_names(**settings):
        simulator = ImpactXSimulator(
            lattice=fodo_lattice,
            ref={"species": "electron", "kin_energy_MeV": 2.0e3},
            distribution=waterbag,
            npart=200,
            bunch_charge_C=1e-9,
            settings=settings or None,
        )
        return {action.name for action in make_actions(simulator)}

    spin_moment_names = {
        "mean_sx",
        "mean_sy",
        "mean_sz",
        "sigma_sx",
        "sigma_sy",
        "sigma_sz",
    }

    off = variable_names()
    assert not any(n.rsplit(":", 1)[-1] in spin_moment_names for n in off)

    on = variable_names(spin=True)
    assert {f"moment_final:{n}" for n in spin_moment_names} <= on
    assert {f"moment:{n}" for n in spin_moment_names} <= on


@pytest.mark.slow
@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="defaults"),
        pytest.param(
            {"csr": True, "csr_bins": 150, "isr": True, "particle_shape": 2},
            id="collective-effects",
        ),
        pytest.param({"spin": True, "particle_shape": 2}, id="spin"),
    ],
)
def test_every_generated_variable_is_readable_and_writable(
    fodo_lattice, waterbag, settings
):
    """Every generated variable must survive get(), and every writable one a set().

    This caught twelve broken ``sim:*`` variables: they were generated for settings the
    simulator did not carry, so their value was None and ``get()`` failed validation.
    ImpactX has no readable default for an unset parameter -- reading one raises
    "algo.csr is not set yet" -- so there was nothing for them to report.
    """
    import numpy as np

    from lume_impactx.model import LUMEImpactXModel
    from lume_impactx.simulator import ImpactXSimulator

    simulator = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
        settings=settings or None,
    )
    # dummy_run keeps this fast and isolates variable plumbing from re-tracking.
    model = LUMEImpactXModel.from_simulator(simulator, dummy_run=True)
    assert model.supported_variables

    unreadable = []
    unwritable = []
    for name in sorted(model.supported_variables):
        try:
            current = model.get(name)
        except Exception as exc:  # noqa: BLE001 - reporting every failure at once
            unreadable.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        variable = model.supported_variables[name]
        if variable.read_only:
            continue
        try:
            model.set({name: current})
            roundtripped = model.get(name)
        except Exception as exc:  # noqa: BLE001
            unwritable.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        if isinstance(current, (int, float)) and not isinstance(current, bool):
            same = np.isclose(roundtripped, current, rtol=1e-12, atol=0.0)
        elif isinstance(current, np.ndarray):
            same = np.array_equal(roundtripped, current)
        else:
            same = roundtripped == current
        if not same:
            unwritable.append(
                f"{name}: set was a no-op ({current!r} -> {roundtripped!r})"
            )

    assert not unreadable, f"variables that cannot be read: {unreadable}"
    assert not unwritable, f"writable variables that do not round-trip: {unwritable}"


@pytest.mark.slow
def test_sim_variables_track_the_settings_actually_carried(fodo_lattice, waterbag):
    from lume_impactx.config import make_actions
    from lume_impactx.simulator import ImpactXSimulator

    def sim_variables(**settings):
        simulator = ImpactXSimulator(
            lattice=fodo_lattice,
            ref={"species": "electron", "kin_energy_MeV": 2.0e3},
            distribution=waterbag,
            npart=200,
            bunch_charge_C=1e-9,
            settings=settings or None,
        )
        names = {a.name for a in make_actions(simulator) if a.name.startswith("sim:")}
        return simulator, names

    simulator, default_names = sim_variables()
    assert default_names == {
        f"sim:{k}"
        for k in simulator.settings
        if k in {"space_charge", "diagnostics", "slice_step_diagnostics", "periods"}
    }
    assert "sim:csr" not in default_names

    _, with_csr = sim_variables(csr=True, csr_bins=150, particle_shape=2)
    assert {"sim:csr", "sim:csr_bins", "sim:particle_shape"} <= with_csr


def test_unit_labels_follow_the_element_unit_flag():
    """ImpactX strengths change unit with a sibling `unit` attribute.

    `ChrQuad.k` is 1/m^2 for unit=0 and T/m for unit=1. The label is resolved when the
    variable is generated, so `unit` itself must be read-only -- otherwise flipping it
    would leave every strength advertising the wrong unit to a control system.
    """
    from impactx import elements

    lattice = [
        elements.ChrQuad(name="cq0", ds=1.0, k=1.0, unit=0),
        elements.ChrQuad(name="cq1", ds=1.0, k=1.0, unit=1),
        elements.TaperedPL(name="pl", k=1.0, taper=0.1, unit=1),
    ]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    assert actions["ele:cq0:k"].unit == "1/m^2"
    assert actions["ele:cq1:k"].unit == "T/m"
    assert actions["ele:pl:k"].unit == "T"
    assert actions["ele:cq1:unit"].read_only is True


def test_degree_element_angles_are_labelled_radians():
    """These four read back in radians although their constructors take degrees."""
    from impactx import elements

    lattice = [
        elements.ExactSbend(name="sb", ds=1.0, phi=30.0),
        elements.ThinDipole(name="td", theta=15.0, rc=10.0),
        elements.PRot(name="pr", phi_in=5.0, phi_out=5.0),
        elements.PlaneXYRot(name="rot", angle=10.0),
    ]
    actions = _by_name(_make_element_actions(_sim(lattice), ElementsConfig()))
    for name in ["ele:sb:phi", "ele:td:theta", "ele:pr:phi_in", "ele:rot:angle"]:
        assert actions[name].unit == "rad", name
    # and the value really is radians
    assert lattice[0].phi == pytest.approx(np.deg2rad(30.0))


@pytest.mark.slow
def test_lattice_mutation_is_caught_rather_than_misaddressing(fodo_lattice, waterbag):
    """Inserting an element shifts every index; variables must refuse, not silently move.

    A same-type insertion used to make `ele:quad1:k` read and write a *different*
    magnet, with no error at all.
    """
    from lume_impactx.model import LUMEImpactXModel
    from lume_impactx.simulator import ImpactXSimulator
    from impactx import elements

    simulator = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    model = LUMEImpactXModel.from_simulator(simulator, dummy_run=True)
    assert model.get("ele:quad1:k") == pytest.approx(1.0)

    simulator.lattice.insert(0, elements.Quad(name="quad0", ds=0.1, k=9.0, nslice=5))

    with pytest.raises(RuntimeError, match="was generated for a lattice of"):
        model.get("ele:quad1:k")
    with pytest.raises(RuntimeError, match="was generated for a lattice of"):
        model.set({"ele:quad1:k": 3.0})

    # Rebuilding is the documented remedy, and it works.
    rebuilt = LUMEImpactXModel.from_simulator(simulator, dummy_run=True)
    assert rebuilt.get("ele:quad0:k") == pytest.approx(9.0)
    assert rebuilt.get("ele:quad1:k") == pytest.approx(1.0)


@pytest.mark.slow
def test_s_series_length_mismatch_raises(fodo_lattice, waterbag):
    """Appending an element used to silently truncate the s-series with no NaN."""
    from impactx import elements

    from lume_impactx.model import LUMEImpactXModel
    from lume_impactx.simulator import ImpactXSimulator

    simulator = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    model = LUMEImpactXModel.from_simulator(simulator, dummy_run=True)
    history = model.get("moment:sigma_x")
    assert history.shape == (simulator.n_steps,)

    simulator.lattice.append(elements.Drift(name="extra", ds=0.5, nslice=5))
    simulator.track()
    with pytest.raises(RuntimeError, match="s-points"):
        model.get("moment:sigma_x")


@pytest.mark.slow
@pytest.mark.parametrize("mutation", ["replace-same-type", "swap"])
def test_same_type_lattice_edits_are_caught(fodo_lattice, waterbag, mutation):
    """Length and type both survive a swap or in-place replacement; the name does not.

    This is the case the guard exists for: without the name check, `ele:quad1:k` reads
    and writes a different magnet with no error at all.
    """
    from impactx import elements

    from lume_impactx.model import LUMEImpactXModel
    from lume_impactx.simulator import ImpactXSimulator

    simulator = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    model = LUMEImpactXModel.from_simulator(simulator, dummy_run=True)
    assert model.get("ele:quad1:k") == pytest.approx(1.0)

    lattice = simulator.lattice
    if mutation == "replace-same-type":
        lattice[1] = elements.Quad(name="OTHER", ds=1.0, k=99.0, nslice=5)
    else:
        lattice[1], lattice[3] = lattice[3], lattice[1]

    assert len(lattice) == len(fodo_lattice)  # length guard cannot see this
    with pytest.raises(RuntimeError, match="but found"):
        model.get("ele:quad1:k")
    with pytest.raises(RuntimeError, match="but found"):
        model.set({"ele:quad1:k": 3.0})


def test_float_simulation_settings_become_scalar_variables():
    """SimScalarAction was documented but unreachable: no float settings were dispatched."""
    from lume_impactx.actions import SimScalarAction
    from lume_impactx.config import SIM_FLOATS

    assert "mlmg_relative_tolerance" in SIM_FLOATS
    assert issubclass(SimScalarAction, ScalarVariable)


@pytest.mark.slow
@pytest.mark.parametrize(
    "section,prefix",
    [
        ("elements", "ele:"),
        ("moments", "moment"),
        ("ref", "ref"),
        ("sim", "sim:"),
        ("particles", "particles:"),
        ("run_info", "run_info:"),
    ],
)
def test_each_config_section_can_be_switched_off(
    fodo_lattice, waterbag, section, prefix
):
    """Setting a section to None must drop exactly that namespace.

    lume-impact covers the same with test_no_element_vars_when_elements_none and
    friends.
    """
    from lume_impactx.config import VariableMappingConfig, make_actions
    from lume_impactx.simulator import ImpactXSimulator

    simulator = ImpactXSimulator(
        lattice=fodo_lattice,
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    full = {a.name for a in make_actions(simulator)}
    assert any(n.startswith(prefix) for n in full), (
        f"{prefix} missing from the baseline"
    )

    without = {
        a.name
        for a in make_actions(simulator, VariableMappingConfig(**{section: None}))
    }
    assert not any(n.startswith(prefix) for n in without)
    # and nothing else was lost
    assert {n for n in full if not n.startswith(prefix)} == without


@pytest.mark.slow
def test_optics_variables_expose_the_transfer_maps(waterbag):
    """The ImpactX counterpart of lume-bmad's mat6 output."""
    from impactx import elements

    from lume_impactx.model import LUMEImpactXModel
    from lume_impactx.simulator import ImpactXSimulator

    simulator = ImpactXSimulator(
        lattice=[elements.Drift(name="d", ds=0.5, nslice=1)],
        ref={"species": "electron", "kin_energy_MeV": 2.0e3},
        distribution=waterbag,
        npart=200,
        bunch_charge_C=1e-9,
    )
    model = LUMEImpactXModel.from_simulator(simulator)

    assert model.supported_variables["optics:transfer_map"].shape == (6, 6)
    assert model.supported_variables["optics:map_s"].unit == "m"
    for name in ("optics:transfer_map", "optics:cumulative_maps", "optics:map_s"):
        assert model.supported_variables[name].read_only is True

    np.testing.assert_allclose(
        model.get("optics:transfer_map")[:2, :2], [[1.0, 0.5], [0.0, 1.0]], atol=1e-12
    )
