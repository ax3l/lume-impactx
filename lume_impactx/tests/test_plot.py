"""Plotting tests. Headless: the Agg backend is forced before pyplot is imported."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from lume_impactx.plot import plot_lattice_layout, plot_moments_with_layout  # noqa: E402


def test_layout_draws_one_patch_per_thick_element(fodo_lattice):
    ax = plot_lattice_layout(fodo_lattice, include_labels=True)
    # five thick elements -> five rectangles, and the axis spans the cell
    assert len(ax.patches) == 5
    assert ax.get_xlim() == pytest.approx((0.0, 3.0))


def test_layout_marks_zero_length_elements():
    from impactx import elements

    lattice = [
        elements.Drift(name="d", ds=1.0),
        elements.Marker(name="m"),
        elements.Drift(name="d2", ds=1.0),
    ]
    ax = plot_lattice_layout(lattice)
    assert len(ax.patches) == 2  # the marker is a line, not a box
    assert ax.get_xlim() == pytest.approx((0.0, 2.0))


@pytest.mark.slow
def test_moment_plot_has_layout_and_curves(fodo_simulator):
    fig = plot_moments_with_layout(
        fodo_simulator, y=("sigma_x", "sigma_y"), y2=("mean_pt",)
    )
    main, layout = fig.axes[0], fig.axes[1]
    assert len(main.lines) == 2
    assert len(layout.patches) == 5
    assert main.get_legend() is not None


@pytest.mark.slow
def test_moment_plot_without_layout(fodo_simulator):
    fig = plot_moments_with_layout(fodo_simulator, include_layout=False)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel().startswith("s")


@pytest.mark.slow
def test_plot_rejects_unknown_keys(fodo_simulator):
    with pytest.raises(KeyError, match="Not in the moment history"):
        plot_moments_with_layout(fodo_simulator, y=("not_a_moment",))


@pytest.mark.slow
def test_model_and_simulator_expose_plot(fodo_simulator):
    from lume_impactx.model import LUMEImpactXModel

    assert fodo_simulator.plot().axes
    assert LUMEImpactXModel.from_simulator(fodo_simulator).plot().axes
