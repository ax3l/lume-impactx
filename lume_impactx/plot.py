"""Plotting beam moments against ``s``, with a lattice layout strip.

Modelled on ``impact/plot.py::plot_stats_with_layout`` in lume-impact, which is what
users of the LUME ecosystem expect a beam-dynamics wrapper to give them.

The layout is drawn here rather than via ``KnownElementsList.plot_survey`` because the
simulator keeps its lattice as a plain Python list -- the list is the canonical, mutable
thing that action variables write to, and only gets copied into a ``KnownElementsList``
inside :meth:`~lume_impactx.simulator.ImpactXSimulator.track`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from lume_impactx.elements import element_type
from lume_impactx.units import MOMENT_UNITS

__all__ = ["plot_lattice_layout", "plot_moments_with_layout"]

#: Face colour per element kind. Anything unlisted falls back to ``_DEFAULT_COLOR``.
ELEMENT_COLORS: dict[str, str] = {
    "Drift": "#d9d9d9",
    "ChrDrift": "#d9d9d9",
    "ExactDrift": "#d9d9d9",
    "Quad": "#d62728",
    "ChrQuad": "#d62728",
    "ExactQuad": "#d62728",
    "SoftQuadrupole": "#d62728",
    "QuadEdge": "#ff9896",
    "Sbend": "#2ca02c",
    "ExactSbend": "#2ca02c",
    "CFbend": "#2ca02c",
    "ExactCFbend": "#2ca02c",
    "DipEdge": "#98df8a",
    "ThinDipole": "#98df8a",
    "Sol": "#9467bd",
    "SoftSolenoid": "#9467bd",
    "RFCavity": "#ff7f0e",
    "ShortRF": "#ff7f0e",
    "Buncher": "#ff7f0e",
    "ChrAcc": "#ff7f0e",
    "Multipole": "#8c564b",
    "ExactMultipole": "#8c564b",
    "Kicker": "#e377c2",
    "Aperture": "#7f7f7f",
    "PolygonAperture": "#7f7f7f",
    "BeamMonitor": "#17becf",
    "Marker": "#17becf",
}
_DEFAULT_COLOR = "#bcbd22"

#: Elements drawn as a vertical marker line rather than a box.
_ZERO_LENGTH_HEIGHT = 0.9


def _nice_label(key: str) -> str:
    """Turn a moment key into an axis label with its unit."""
    unit = MOMENT_UNITS.get(key)
    return f"{key} [{unit}]" if unit else key


def plot_lattice_layout(
    lattice: Sequence[Any],
    ax: Any = None,
    include_labels: bool = False,
    s_start: float = 0.0,
) -> Any:
    """Draw a lattice as coloured boxes against ``s``.

    Parameters
    ----------
    lattice : sequence of impactx.elements.*
        The lattice to draw, in beam order.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created when omitted.
    include_labels : bool
        Annotate each named element with its name.
    s_start : float
        ``s`` of the first element, for drawing one section of a staged model.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 1.4))

    s = s_start
    for element in lattice:
        ds = float(getattr(element, "ds", 0.0) or 0.0)
        kind = element_type(element)
        color = ELEMENT_COLORS.get(kind, _DEFAULT_COLOR)
        if ds > 0.0:
            ax.add_patch(
                Rectangle(
                    (s, -_ZERO_LENGTH_HEIGHT / 2),
                    ds,
                    _ZERO_LENGTH_HEIGHT,
                    facecolor=color,
                    edgecolor="none",
                )
            )
        else:
            ax.axvline(s, color=color, linewidth=1.5)
        if include_labels:
            name = getattr(element, "name", None)
            if name:
                ax.annotate(
                    name,
                    (s + ds / 2, _ZERO_LENGTH_HEIGHT / 2),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        s += ds

    ax.set_xlim(s_start, s)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel("s [m]")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    return ax


def plot_moments_with_layout(
    simulator: Any,
    y: Sequence[str] = ("sigma_x", "sigma_y"),
    y2: Sequence[str] = (),
    x: str = "s",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylim2: tuple[float, float] | None = None,
    include_layout: bool = True,
    include_labels: bool = False,
    include_legend: bool = True,
    figsize: tuple[float, float] = (10, 5),
    return_figure: bool = True,
    **kwargs: Any,
) -> Any:
    """Plot beam moments along the lattice.

    Parameters
    ----------
    simulator : ImpactXSimulator
        A simulator that has tracked at least once.
    y : sequence of str
        Moment keys on the left axis, e.g. ``("sigma_x", "sigma_y")``.
    y2 : sequence of str
        Moment keys on a twinned right axis, e.g. ``("mean_pt",)``.
    x : str
        Key for the horizontal axis; ``"s"`` in practice.
    xlim, ylim, ylim2 : tuple of float, optional
        Axis limits.
    include_layout : bool
        Draw the lattice layout strip underneath.
    include_labels : bool
        Annotate the layout with element names.
    include_legend : bool
        Draw a legend.
    figsize : tuple of float
        Figure size in inches.
    return_figure : bool
        Return the figure. When False, returns None after drawing.
    **kwargs
        Forwarded to every ``Axes.plot`` call.

    Returns
    -------
    matplotlib.figure.Figure or None

    Raises
    ------
    KeyError
        If a requested key is not in the moment history.
    """
    import matplotlib.pyplot as plt

    history = simulator.results["moments_history"]
    missing = [k for k in (x, *y, *y2) if k not in history]
    if missing:
        raise KeyError(
            f"Not in the moment history: {missing}. "
            f"Available: {sorted(history.columns)[:12]}..."
        )

    if include_layout:
        fig, (ax, ax_layout) = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
            sharex=True,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_layout = None

    xs = np.asarray(history[x])
    for key in y:
        ax.plot(xs, np.asarray(history[key]), label=_nice_label(key), **kwargs)
    ax.set_ylabel(", ".join(_nice_label(k) for k in y))

    ax_right = None
    if y2:
        ax_right = ax.twinx()
        for key in y2:
            ax_right.plot(
                xs,
                np.asarray(history[key]),
                linestyle="--",
                label=_nice_label(key),
                **kwargs,
            )
        ax_right.set_ylabel(", ".join(_nice_label(k) for k in y2))
        if ylim2 is not None:
            ax_right.set_ylim(*ylim2)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)

    if include_legend:
        handles, labels = ax.get_legend_handles_labels()
        if ax_right is not None:
            extra = ax_right.get_legend_handles_labels()
            handles, labels = handles + extra[0], labels + extra[1]
        ax.legend(handles, labels, loc="best", fontsize=8)

    if ax_layout is not None:
        plot_lattice_layout(
            simulator.lattice, ax=ax_layout, include_labels=include_labels
        )
    else:
        ax.set_xlabel(_nice_label(x))

    return fig if return_figure else None
