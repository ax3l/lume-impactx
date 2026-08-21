"""Packaging checks.

The dependency test exists because a real bug slipped through: ``pandas`` is required
by ImpactX's own ``to_df()`` and ``beam_moments_history()``, which this package calls on
every track, but neither the conda-forge package nor the ``impactx-noacc`` wheel
declares it -- so ``pip install lume-impactx[impactx]`` produced an installation that
failed at import of the first simulation. The conda dev environment happened to have
pandas, so nothing local caught it.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "lume_impactx"
PYPROJECT = ROOT / "pyproject.toml"

#: Third-party modules that are deliberately not hard dependencies.
KNOWN_OPTIONAL = {
    # ImpactX: conda-forge for accelerated builds, the `impactx` extra for the wheel.
    "impactx": "declared in the [impactx] extra, not a hard dependency",
    # Only imported when Config.have_mpi; raises a clear message when absent.
    "mpi4py": "only needed for MPI-enabled ImpactX builds",
    # Only needed by read_beam_monitor; declared in the [impactx] extra.
    "openpmd_api": "declared in the [impactx] extra",
    # Import fallback for openpmd-beamphysics < 0.15, which is already the floor.
    "pmd_beamphysics": "compatibility fallback for openpmd-beamphysics < 0.15",
}

#: Module name -> distribution name, where they differ.
MODULE_TO_DISTRIBUTION = {
    "beamphysics": "openpmd-beamphysics",
    "lume": "lume-base",
    "yaml": "pyyaml",
}


def _declared_distributions() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    requirements = data["project"]["dependencies"]
    names = set()
    for requirement in requirements:
        name = requirement.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _imported_top_level_modules() -> set[str]:
    """Top-level module names imported anywhere in the package, tests excluded."""
    modules: set[str] = set()
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    modules.add(node.module.split(".")[0])
    return modules


def _is_stdlib(name: str) -> bool:
    import sys

    return name in sys.stdlib_module_names


def test_every_imported_third_party_module_is_declared():
    declared = _declared_distributions()
    undeclared = []
    for module in sorted(_imported_top_level_modules()):
        if module == "lume_impactx" or _is_stdlib(module):
            continue
        if module in KNOWN_OPTIONAL:
            continue
        distribution = MODULE_TO_DISTRIBUTION.get(module, module).lower()
        if distribution not in declared:
            undeclared.append(f"{module} (distribution {distribution!r})")

    assert not undeclared, (
        "These modules are imported by lume_impactx but are not in "
        f"[project.dependencies]: {undeclared}. Either declare them, or add them to "
        "KNOWN_OPTIONAL with the reason they are optional."
    )


@pytest.mark.parametrize("module", sorted(KNOWN_OPTIONAL))
def test_optional_modules_are_not_hard_dependencies(module):
    """Guard the other direction: an optional module must stay out of the hard deps."""
    distribution = MODULE_TO_DISTRIBUTION.get(module, module).lower()
    assert distribution not in _declared_distributions(), (
        f"{module} is listed in KNOWN_OPTIONAL as {KNOWN_OPTIONAL[module]!r}, but it "
        "is also a hard dependency. One of the two is wrong."
    )


def test_declared_extras_are_the_documented_ones():
    data = tomllib.loads(PYPROJECT.read_text())
    extras = set(data["project"]["optional-dependencies"])
    assert {"impactx", "dev", "docs"} <= extras


def test_pandas_is_declared():
    """ImpactX needs it for to_df(); nothing in the ImpactX install chain declares it."""
    assert "pandas" in _declared_distributions()
