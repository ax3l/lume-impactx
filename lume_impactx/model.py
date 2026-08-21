"""The LUME model wrapping an ImpactX simulation."""

from __future__ import annotations

from typing import Any

from lume.actions import Action, ActionModel
from lume.staged_model import FinalParticlesMixIn, InitialParticlesMixIn

from lume_impactx.simulator import ImpactXSimulator

try:
    from beamphysics import ParticleGroup
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup

__all__ = ["LUMEImpactXModel"]


class LUMEImpactXModel(
    InitialParticlesMixIn, FinalParticlesMixIn, ActionModel[ImpactXSimulator]
):
    """A ``LUMEModel`` over an :class:`~lume_impactx.simulator.ImpactXSimulator`.

    ``set()`` writes the requested variables and then re-tracks, so a following
    ``get()`` reflects them. Reads come from the simulator's cached results rather than
    from live ImpactX state, because the container is torn down at the end of each
    track.

    Parameters
    ----------
    simulator : ImpactXSimulator
        The simulation to drive.
    actions : list of Action
        The action variables to expose. Usually from
        :func:`~lume_impactx.config.make_actions` via :meth:`from_simulator`.
    dummy_run : bool
        Skip re-tracking on ``set()``. Useful for staging several writes into one run:
        set with ``dummy_run=True``, then call ``simulator.track()`` once. A full
        simulation per ``set()`` is expensive when space charge is on.

    Examples
    --------
    >>> model = LUMEImpactXModel.from_simulator(simulator)
    >>> model.set({"ele:q1:k": 1.2})
    >>> model.get("moment_final:sigma_x")
    6.46e-05
    """

    def __init__(
        self,
        simulator: ImpactXSimulator,
        actions: list[Action],
        dummy_run: bool = False,
    ) -> None:
        super().__init__(simulator=simulator, action_variables=actions)
        self.dummy_run = dummy_run

    @classmethod
    def from_simulator(
        cls,
        simulator: ImpactXSimulator,
        config: Any = None,
        **kwargs: Any,
    ) -> "LUMEImpactXModel":
        """Build a model with variables generated from the simulator.

        Parameters
        ----------
        simulator : ImpactXSimulator
            The simulation to introspect and drive.
        config : VariableMappingConfig, optional
            Controls which variables are generated and how they are named. Defaults to
            :class:`~lume_impactx.config.VariableMappingConfig`.
        **kwargs
            Passed to :class:`LUMEImpactXModel`, e.g. ``dummy_run``.
        """
        from lume_impactx.config import make_actions

        return cls(simulator, make_actions(simulator, config), **kwargs)

    def _set(self, values: dict[str, Any]) -> None:
        super()._set(values)
        if not self.dummy_run:
            self.simulator.track()

    def reset(self) -> None:
        """Restore the simulator's construction-time state and re-track.

        Overrides ``ActionModel.reset``, which would only write each writable variable
        back to its ``default_value``. The simulator can restore its lattice, settings
        and reference particle exactly, which is both stronger and cheaper.
        """
        self.simulator.reset()

    @property
    def initial_particles(self) -> ParticleGroup:
        """The bunch injected at the start of the lattice.

        Raises
        ------
        RuntimeError
            If the simulator samples a distribution rather than taking an explicit
            bunch, so there is no ``ParticleGroup`` to hand back.
        """
        particles = self.simulator.initial_particles
        if particles is None:
            raise RuntimeError(
                "This simulator seeds its beam from a distribution, so there are no "
                "initial particles. Construct ImpactXSimulator with "
                "initial_particles= to use it as a staged-model target."
            )
        return particles

    @initial_particles.setter
    def initial_particles(self, value: ParticleGroup) -> None:
        self.simulator.initial_particles = value

    @property
    def final_particles(self) -> ParticleGroup:
        """The bunch at the end of the lattice, from the last track."""
        return self.simulator.final_particles

    def plot(self, y=("sigma_x", "sigma_y"), **kwargs):
        """Plot beam moments along the lattice, from the last track."""
        return self.simulator.plot(y=y, **kwargs)
