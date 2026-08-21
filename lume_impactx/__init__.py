"""ImpactX-specific implementation of LUMEModel classes for virtual accelerators."""

from lume_impactx._mpi import ensure_external_mpi

# Must run before the first ImpactX() is constructed anywhere in the process.
# No-op for serial builds; see lume_impactx._mpi for why this exists.
ensure_external_mpi()

from lume_impactx.archive import archive, load_archive  # noqa: E402
from lume_impactx.config import VariableMappingConfig, make_actions  # noqa: E402
from lume_impactx.model import LUMEImpactXModel  # noqa: E402
from lume_impactx.plot import (  # noqa: E402
    plot_lattice_layout,
    plot_moments_with_layout,
)
from lume_impactx.simulator import ImpactXSimulator  # noqa: E402
from lume_impactx.staged import StagedImpactXModel  # noqa: E402
from lume_impactx.utils import (  # noqa: E402
    ImpactXRefPart,
    impactx_to_particlegroup_data,
    particlegroup_to_impactx,
    read_beam_monitor,
)

try:
    from lume_impactx._version import __version__
except ImportError:  # pragma: no cover - source checkout without setuptools-scm
    __version__ = "0.0.0.dev0"

__all__ = [
    # models and simulation
    "ImpactXSimulator",
    "LUMEImpactXModel",
    "StagedImpactXModel",
    # variable generation
    "VariableMappingConfig",
    "make_actions",
    # particle conversion
    "ImpactXRefPart",
    "particlegroup_to_impactx",
    "impactx_to_particlegroup_data",
    "read_beam_monitor",
    # persistence and plotting
    "archive",
    "load_archive",
    "plot_lattice_layout",
    "plot_moments_with_layout",
    # misc
    "ensure_external_mpi",
    "__version__",
]
