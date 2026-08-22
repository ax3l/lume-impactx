r"""Short-range wakefields, applied as a thin kick.

ImpactX has no wakefield lattice element. It ships the numerical primitives --
``impactx.wakeconvolution.deposit_charge`` / ``derivative_charge`` / ``w_l_csr`` /
``convolve_fft`` -- and leaves the assembly to the user. This module assembles one
useful case, the **resistive-wall** wake of a round pipe, into an element you can drop
into a lattice.

The wake is applied as a *thin* kick: an ``elements.Programmable`` of zero length whose
push hook bins the bunch longitudinally, convolves the line charge density with the wake
function, and applies the resulting energy change. Put it immediately after the drift or
pipe whose length it represents.

Model
-----
The longitudinal wake per unit length of a round pipe of radius ``b`` and conductivity
``sigma_c``, in the Bane-Sands short-range form:

$$ s_0 = \left(\frac{2 b^2}{Z_0 \sigma_c}\right)^{1/3} $$

$$ W_L(s) = \frac{Z_0 c}{\pi b^2}\left[\frac{1}{3}e^{-s/s_0}\cos(\sqrt{3}s/s_0)
   - \frac{\sqrt{2}}{\pi}\int_0^\infty
   \frac{x^2 e^{-x^2 s/s_0}}{x^6+8}\,dx\right] $$

for $s \ge 0$, and zero for $s < 0$ (causality). At $s = 0$ the integral is analytic
and the whole expression collapses to $W_L(0) = Z_0 c / (4\pi b^2)$, which
:func:`resistive_wall_wake` is tested against.

This is the classic short-range approximation: it assumes a round, thick, dc-conducting
wall and a relativistic beam, and it ignores ac conductivity and surface roughness. It
is a demonstration of the mechanism rather than a validated engineering model -- check
it against a dedicated wake code before using it for a real machine.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Z0",
    "csr_validity_margin",
    "C_LIGHT",
    "resistive_wall_s0",
    "resistive_wall_wake",
    "apply_longitudinal_wake",
    "ResistiveWallWake",
]

#: Impedance of free space, ohm.
Z0 = 376.730313412
#: Speed of light, m/s.
C_LIGHT = 299792458.0


def csr_validity_margin(
    bend_radius: float, sigma_z: float, sigma_transverse: float
) -> float:
    r"""How comfortably a bunch satisfies the 1D CSR validity condition.

    The 1D CSR model that ImpactX, Bmad and elegant all use projects the bunch onto a
    line, which is only justified when the bunch is transversely thin compared with the
    CSR formation width:

    $$ \sigma_\perp \ll R\left(\frac{\sigma_z}{R}\right)^{2/3} $$

    (Sagan and Mayes, *Coherent Synchrotron Radiation Simulations for Off-Axis Beams
    Using the Bmad Toolkit*, IPAC2017 THPAB076, following Saldin et al.,
    NIM A **398**, 373 (1997).)

    Parameters
    ----------
    bend_radius : float
        Bending radius ``R``, metres.
    sigma_z : float
        RMS bunch length, metres.
    sigma_transverse : float
        RMS transverse size, metres.

    Returns
    -------
    float
        The ratio of the limit to ``sigma_transverse``. Much greater than 1 means the
        model is on solid ground; approaching 1 means the 1D treatment is breaking down
        and a 3D space-charge calculation is the honest answer.

    Examples
    --------
    >>> round(csr_validity_margin(bend_radius=5.73, sigma_z=1e-4, sigma_transverse=1e-4))
    39
    """
    if bend_radius <= 0.0 or sigma_z <= 0.0 or sigma_transverse <= 0.0:
        raise ValueError("bend_radius, sigma_z and sigma_transverse must be positive.")
    limit = bend_radius * (sigma_z / bend_radius) ** (2.0 / 3.0)
    return float(limit / sigma_transverse)


def resistive_wall_s0(pipe_radius: float, conductivity: float) -> float:
    """Characteristic length of the resistive-wall wake.

    Parameters
    ----------
    pipe_radius : float
        Beam-pipe radius ``b``, metres.
    conductivity : float
        Wall dc conductivity, S/m (copper is 5.96e7).

    Returns
    -------
    float
        ``s0 = (2 b^2 / (Z0 sigma))^(1/3)``, metres.
    """
    if pipe_radius <= 0.0:
        raise ValueError(f"pipe_radius must be positive, got {pipe_radius}.")
    if conductivity <= 0.0:
        raise ValueError(f"conductivity must be positive, got {conductivity}.")
    return float((2.0 * pipe_radius**2 / (Z0 * conductivity)) ** (1.0 / 3.0))


def _bane_sands_integral(u: np.ndarray, n_nodes: int = 400) -> np.ndarray:
    """Evaluate ``int_0^inf x^2 exp(-u x^2)/(x^6+8) dx`` for an array of ``u >= 0``.

    The infinite range is mapped to ``[0, 1)`` with ``x = t/(1-t)`` and integrated with
    Gauss-Legendre, which converges quickly and needs no SciPy.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * (nodes + 1.0)  # [-1, 1] -> [0, 1]
    w = 0.5 * weights
    x = t / (1.0 - t)
    jacobian = 1.0 / (1.0 - t) ** 2

    u = np.asarray(u, dtype=float)[..., None]
    integrand = x**2 * np.exp(-u * x**2) / (x**6 + 8.0)
    return np.sum(integrand * w * jacobian, axis=-1)


def resistive_wall_wake(
    s: np.ndarray, pipe_radius: float, conductivity: float
) -> np.ndarray:
    """Longitudinal resistive-wall wake per unit length.

    Parameters
    ----------
    s : array_like
        Distance behind the source particle, metres. Negative values give zero, by
        causality.
    pipe_radius : float
        Beam-pipe radius, metres.
    conductivity : float
        Wall dc conductivity, S/m.

    Returns
    -------
    numpy.ndarray
        ``W_L(s)`` in V/(C m). Positive means an energy *loss* for a trailing particle.

    Examples
    --------
    >>> float(resistive_wall_wake(0.0, 0.01, 5.96e7))  # doctest: +ELLIPSIS
    89873...
    """
    s = np.asarray(s, dtype=float)
    s0 = resistive_wall_s0(pipe_radius, conductivity)
    prefactor = Z0 * C_LIGHT / (np.pi * pipe_radius**2)

    u = np.clip(s, 0.0, None) / s0
    oscillatory = (1.0 / 3.0) * np.exp(-u) * np.cos(np.sqrt(3.0) * u)
    diffusive = (np.sqrt(2.0) / np.pi) * _bane_sands_integral(u)

    wake = prefactor * (oscillatory - diffusive)
    return np.where(s >= 0.0, wake, 0.0)


def apply_longitudinal_wake(
    particle_container,
    wake_function,
    length: float,
    num_bins: int = 128,
) -> None:
    """Apply a longitudinal wake to a bunch as a single energy kick.

    Bins the bunch by arrival time, convolves the binned charge with ``wake_function``,
    and adds the resulting energy change to every particle's ``momentum_t``.

    Parameters
    ----------
    particle_container : impactx.ImpactXParticleContainer
        The bunch, e.g. ``sim.beam``.
    wake_function : callable
        ``wake_function(s)`` -> wake per unit length in V/(C m), for an array of
        non-negative distances behind the source.
    length : float
        Length over which the wake acts, metres.
    num_bins : int
        Longitudinal bins. Too few smears the wake; too many makes each bin noisy.

    Notes
    -----
    Rank-local: on a multi-rank run each rank sees only its own particles, so the
    convolution would be wrong. ImpactX models are single-rank here anyway; see the MPI
    note in the README.
    """
    from impactx import ImpactXParIter

    e_charge = 1.602176634e-19

    # ImpactX position_t is c*(t_i - t_ref) in metres, and *increasing* t means the
    # particle arrives later, i.e. sits further back in the bunch.
    tiles = []
    for level in range(particle_container.finest_level + 1):
        for tile in ImpactXParIter(particle_container, level=level):
            soa = tile.soa().to_xp()
            tiles.append(soa)
    if not tiles:
        return

    position_t = np.concatenate([np.asarray(soa.real["position_t"]) for soa in tiles])
    weighting = np.concatenate([np.asarray(soa.real["weighting"]) for soa in tiles])
    if position_t.size == 0:
        return

    lo, hi = float(position_t.min()), float(position_t.max())
    if not np.isfinite(lo) or hi <= lo:
        return
    bin_size = (hi - lo) / num_bins
    edges = lo + bin_size * np.arange(num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    reference = particle_container.ref

    # `weighting` counts real particles, so the charge in a bin is the count times the
    # charge of one real particle.
    charge_per_particle = abs(reference.charge_qe) * e_charge
    binned_charge, _ = np.histogram(
        position_t, bins=edges, weights=weighting * charge_per_particle
    )

    # Convolve: a particle in bin i feels the wake of all the charge ahead of it. This
    # is a sum over discrete bin charges, not over a density, so there is no bin_size
    # factor. Bin i's own charge contributes W_L(0), i.e. the full self-wake -- a
    # half-bin correction would be the refinement here.
    lags = centers - centers[0]
    kernel = np.asarray(wake_function(lags), dtype=float)
    voltage = length * np.convolve(binned_charge, kernel)[:num_bins]

    # A positive wake means the trailing particle loses energy.
    energy_change_eV = -voltage

    beta_gamma = float(np.sqrt(reference.px**2 + reference.py**2 + reference.pz**2))
    mass_eV = reference.mass_MeV * 1.0e6

    # Interpolate between bin centres rather than taking the bin's value wholesale.
    # A nearest-bin kick quantises the energy profile into visible steps -- an artefact
    # of the binning, not physics -- and the step height does not shrink with more
    # particles, only with more bins.
    per_particle_eV = np.interp(position_t, centers, energy_change_eV)

    # pt = -dgamma / |p_ref|, so an energy gain lowers pt.
    delta_pt = -(per_particle_eV / mass_eV) / beta_gamma

    offset = 0
    for soa in tiles:
        count = len(soa.real["position_t"])
        soa.real["momentum_t"][:] = (
            np.asarray(soa.real["momentum_t"]) + delta_pt[offset : offset + count]
        )
        offset += count


def ResistiveWallWake(
    name: str,
    length: float,
    pipe_radius: float,
    conductivity: float = 5.96e7,
    num_bins: int = 128,
):
    """A thin element applying the resistive-wall wake of a pipe.

    Parameters
    ----------
    name : str
        Element name.
    length : float
        Length of pipe the wake represents, metres. The element itself has zero length;
        place it directly after the drift it stands for.
    pipe_radius : float
        Beam-pipe radius, metres.
    conductivity : float
        Wall dc conductivity, S/m. Defaults to copper.
    num_bins : int
        Longitudinal bins for the convolution.

    Returns
    -------
    impactx.elements.Programmable
        Ready to append to a lattice.

    Examples
    --------
    >>> wake = ResistiveWallWake("rw", length=10.0, pipe_radius=0.01)
    >>> lattice = [Drift(name="pipe", ds=10.0, nslice=20), wake]
    """
    from impactx import elements

    element = elements.Programmable(name=name, ds=0.0, nslice=1)

    def push(particle_container, step, period):
        apply_longitudinal_wake(
            particle_container,
            lambda s: resistive_wall_wake(s, pipe_radius, conductivity),
            length=length,
            num_bins=num_bins,
        )

    def advance_reference(reference_particle):
        """No-op: a thin wake kick does not move the reference particle.

        Registered explicitly because ImpactX prints "Programmable element - ref
        particles: NO HOOK" otherwise, which reads like a mistake rather than a
        deliberate zero-length element.
        """

    element.push = push
    element.ref_particle = advance_reference
    # Keep the closures alive for as long as the element is: pybind stores them without
    # owning them, and a garbage-collected callback is a crash rather than an error.
    element._lume_impactx_hooks = (push, advance_reference)
    return element
