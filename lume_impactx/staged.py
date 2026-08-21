"""Chaining several ImpactX sections into one LUME model.

``lume.staged_model.StagedModel`` already composes models by handing each stage's final
particles to the next. :class:`StagedImpactXModel` subclasses it to fix one behaviour
that matters here, and to make the ImpactX-to-ImpactX case convenient to build.

The upstream ``_set`` runs::

    if model_values:
        model.set(model_values)

so a downstream stage that receives new particles but has **no variables in this
particular ``set()`` call** never re-runs -- and then reports stale ``final_particles``.
That is fine for a generator feeding a tracker (the tracker usually has variables of its
own), but wrong the moment you write only an upstream variable, which is the common case
when scanning an injector setting. :meth:`StagedImpactXModel._set` re-runs any stage
whose incoming particles changed.

Because ``StagedModel`` refuses duplicate variable names across stages, give each stage
its own ``VariableMappingConfig(prefix=...)``.
"""

from __future__ import annotations

from typing import Any

from lume.model import LUMEModel
from lume.staged_model import FinalParticlesMixIn, InitialParticlesMixIn, StagedModel

from lume_impactx.config import VariableMappingConfig
from lume_impactx.model import LUMEImpactXModel
from lume_impactx.simulator import ImpactXSimulator

__all__ = ["StagedImpactXModel"]


class StagedImpactXModel(StagedModel):
    """Several models in series, passing particles downstream.

    Parameters
    ----------
    models : list of LUMEModel
        Ordered stages. Every stage but the last must expose ``final_particles``, and
        every stage but the first must accept ``initial_particles``.

    Examples
    --------
    >>> staged = StagedImpactXModel.from_simulators(
    ...     [injector_sim, linac_sim], prefixes=["inj:", "linac:"]
    ... )
    >>> staged.set({"inj:ele:quad1:k": 1.2})
    >>> staged.get("linac:moment_final:sigma_x")
    """

    def __init__(self, models: list[LUMEModel]) -> None:
        super().__init__(models)

    @classmethod
    def from_simulators(
        cls,
        simulators: list[ImpactXSimulator],
        prefixes: list[str] | None = None,
        configs: list[VariableMappingConfig] | None = None,
        **kwargs: Any,
    ) -> "StagedImpactXModel":
        """Build a staged model from ImpactX simulators.

        Parameters
        ----------
        simulators : list of ImpactXSimulator
            The sections, in beam order. Every one after the first must have been
            constructed with ``initial_particles=`` so it can be re-seeded.
        prefixes : list of str, optional
            One variable-name prefix per stage. Defaults to ``"stage0:"``,
            ``"stage1:"``, ... which keeps names unique without any thought.
        configs : list of VariableMappingConfig, optional
            Per-stage configs. When given, ``prefixes`` is ignored -- set
            ``VariableMappingConfig.prefix`` on each instead.
        **kwargs
            Passed to each :class:`~lume_impactx.model.LUMEImpactXModel`.

        Raises
        ------
        ValueError
            If ``prefixes`` or ``configs`` does not match the number of simulators.
        """
        if configs is None:
            if prefixes is None:
                prefixes = [f"stage{i}:" for i in range(len(simulators))]
            if len(prefixes) != len(simulators):
                raise ValueError(
                    f"Got {len(prefixes)} prefixes for {len(simulators)} simulators."
                )
            configs = [VariableMappingConfig(prefix=p) for p in prefixes]
        if len(configs) != len(simulators):
            raise ValueError(
                f"Got {len(configs)} configs for {len(simulators)} simulators."
            )

        models = [
            LUMEImpactXModel.from_simulator(simulator, config, **kwargs)
            for simulator, config in zip(simulators, configs)
        ]
        return cls(models)

    def _set(self, values: dict[str, Any]) -> None:
        """Set values stage by stage, re-running any stage whose input changed.

        Overrides ``StagedModel._set``, which skips a stage that has no values in this
        call even when it has just been handed new particles.
        """
        incoming = None
        for i, model in enumerate(self.lume_model_instances):
            model_values = {
                k: v for k, v in values.items() if k in model.supported_variables
            }

            reseeded = False
            if (
                i > 0
                and incoming is not None
                and isinstance(model, InitialParticlesMixIn)
            ):
                model.initial_particles = incoming
                self._carry_origin(self.lume_model_instances[i - 1], model)
                reseeded = True

            if model_values:
                model.set(model_values)
            elif reseeded:
                # New particles but nothing else to write: still needs to re-run, or
                # final_particles below would be from the previous bunch.
                self._rerun(model)

            if isinstance(model, FinalParticlesMixIn):
                incoming = model.final_particles

    @staticmethod
    def _carry_origin(upstream: LUMEModel, downstream: LUMEModel) -> None:
        """Tell a stage where along the machine its incoming bunch arrived.

        This carries the arrival time, the arc length and the reference energy. The
        energy is the load-bearing part: an upstream cavity changes it, and beam momenta
        are normalized by it. See ``ImpactXSimulator._align_reference``.
        """
        up = getattr(upstream, "simulator", None)
        down = getattr(downstream, "simulator", None)
        if up is None or down is None:
            return
        try:
            down.ref_origin = up.results["ref_final"]
        except (RuntimeError, KeyError):  # pragma: no cover - upstream never tracked
            pass

    @staticmethod
    def _rerun(model: LUMEModel) -> None:
        """Re-run a stage that was re-seeded but had no variables written."""
        simulator = getattr(model, "simulator", None)
        track = getattr(simulator, "track", None)
        if callable(track):
            track()
        else:  # pragma: no cover - non-ImpactX stages
            model.set({})
