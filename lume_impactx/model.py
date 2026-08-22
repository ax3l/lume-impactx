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

    @classmethod
    def from_tao(
        cls,
        tao: Any,
        config: Any = None,
        dummy_run: bool = False,
        **kwargs: Any,
    ) -> "LUMEImpactXModel":
        """Build a model straight from a Bmad/Tao model.

        Equivalent to ``from_simulator(ImpactXSimulator.from_tao(tao, ...))``, and the
        one-step path from a Tao session to a LUME model with generated variables.
        Every element mapping is verified against Bmad tracking; read
        :func:`lume_impactx.interfaces.bmad.translate_element` for what differs and
        what is dropped.

        Parameters
        ----------
        tao : pytao.Tao
            A Tao instance with a tracked beam saved at the start element.
        config : VariableMappingConfig, optional
            Controls which variables are generated and how they are named.
        dummy_run : bool
            Skip re-tracking on ``set()``, to batch several writes into one run.
        **kwargs
            Passed to :func:`~lume_impactx.interfaces.bmad.simulator_from_tao`, e.g.
            ``ele``, ``lattice``, ``nslice``, ``settings``, ``skip_unsupported``.

        Examples
        --------
        >>> model = LUMEImpactXModel.from_tao(tao, nslice=16)
        >>> model.set({"ele:qf:k": 1.3})
        >>> model.get("moment_final:sigma_x")
        """
        from lume_impactx.interfaces.bmad import model_from_tao

        return model_from_tao(tao, config=config, dummy_run=dummy_run, **kwargs)

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
