"""The generated openPMD-beamphysics interface must stay in step with utils.py.

``upstream/impactx.py`` is a candidate contribution to openPMD-beamphysics, generated
from ``lume_impactx/utils.py``. These tests make sure the committed file is current and
that it behaves identically to the implementation this package actually uses, so the
contribution cannot quietly drift.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "make_upstream_interface.py"
GENERATED = ROOT / "upstream" / "impactx.py"


def test_generated_file_is_up_to_date():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _load_generated():
    """Import the generated module with its relative imports satisfied.

    It is written for openPMD-beamphysics' package layout (``from ..species import``),
    so it is loaded under a synthetic package whose parent resolves to ``beamphysics``.
    """
    source = GENERATED.read_text().replace("from ..", "from beamphysics.")
    module_name = "_generated_impactx_interface"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations via sys.modules[cls.__module__], so the module has
    # to be registered before the class body runs.
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(GENERATED), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_generated_module_imports_and_exports():
    module = _load_generated()
    for name in [
        "ImpactXRefPart",
        "particlegroup_to_impactx",
        "impactx_to_particlegroup_data",
        "pmd_species_of",
        "refpart_from_openpmd",
        "read_beam_monitor",
    ]:
        assert hasattr(module, name), name
    assert set(module.__all__) == {
        "ImpactXRefPart",
        "particlegroup_to_impactx",
        "impactx_to_particlegroup_data",
        "pmd_species_of",
        "refpart_from_openpmd",
        "read_beam_monitor",
    }


def test_generated_module_has_no_downstream_imports():
    """openPMD-beamphysics must depend on neither ImpactX nor lume-impactx.

    The reader is the place this could regress: it builds a ``ParticleGroup`` and
    refuses bunches carrying spin, and both of those must resolve without reaching back
    into lume-impactx.
    """
    forbidden = ("lume_impactx", "lume.", "import impactx", "from impactx")
    for line in GENERATED.read_text().splitlines():
        stripped = line.split("#")[0].strip()
        if not stripped.startswith(("import ", "from ")):
            continue
        for token in forbidden:
            assert token not in stripped, (
                f"upstream file must not import {token!r}: {stripped}"
            )


def test_generated_module_carries_the_refusal_guard():
    """The reader refuses to drop spin, so its guard must travel with it.

    ``read_beam_monitor`` calls ``_check_representable``; if that helper stayed behind
    in lume-impactx the generated file would import fine and NameError at runtime.
    """
    module = _load_generated()
    assert hasattr(module, "_check_representable")
    assert hasattr(module, "UnrepresentableParticleData")

    module._check_representable({"position_x": np.zeros(4), "spin_x": np.zeros(4)})
    with pytest.raises(module.UnrepresentableParticleData, match="spin"):
        module._check_representable({"spin_z": np.ones(4)})


def test_generated_module_matches_lume_impactx(bunch, electron_ref):
    """Identical numbers from both copies, or the contribution has drifted."""
    from lume_impactx import utils

    module = _load_generated()

    theirs = module.particlegroup_to_impactx(bunch, electron_ref)
    ours = utils.particlegroup_to_impactx(bunch, electron_ref)
    assert set(theirs) == set(ours)
    for key, value in ours.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(theirs[key], value, err_msg=key)
        else:
            assert theirs[key] == value, key

    back_theirs = module.impactx_to_particlegroup_data(theirs, electron_ref)
    back_ours = utils.impactx_to_particlegroup_data(ours, electron_ref)
    for key, value in back_ours.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(back_theirs[key], value, err_msg=key)
        else:
            assert back_theirs[key] == value, key


def test_generated_species_inference_matches(electron_ref):
    module = _load_generated()
    from lume_impactx import utils

    assert module.pmd_species_of(electron_ref) == utils.pmd_species_of(electron_ref)
    with pytest.raises(ValueError):
        module.pmd_species_of(
            module.ImpactXRefPart(**{**electron_ref.__dict__, "mass_MeV": 42.0})
        )
