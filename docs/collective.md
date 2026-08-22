# Collective effects

Space charge, CSR and wakefields, and where each one lives.

| effect | how ImpactX does it | model variables |
|---|---|---|
| Space charge | built-in solver | `sim:space_charge`, `sim:poisson_solver`, `sim:n_cell`, `sim:prob_relative`, `sim:particle_shape`, `sim:dynamic_size` |
| CSR | built-in, per-step | `sim:csr`, `sim:csr_bins` |
| Wakefields | **no element** — primitives only | assembled by `lume_impactx.wakes` |

Runnable, self-checking scripts for all three are in [`examples/`](https://github.com/ax3l/lume-impactx/tree/main/examples).

!!! warning "`particle_shape` is required"
    ImpactX refuses `init_grids()` with *"particle_shape is not set, cannot initialize
    grids with guard cells for collective effects"* whenever space charge or CSR is on.
    Pass it in `settings`.

## Space charge

```python
simulator = ImpactXSimulator(
    lattice=[elements.Drift(name="drift", ds=6.0, nslice=40)],
    ref={"species": "electron", "kin_energy_MeV": 250.0},
    distribution=beam, npart=10_000, bunch_charge_C=1.0e-9,
    settings={
        "space_charge": "3D",          # or "2D", "2p5D", "Gauss3D", "Gauss2p5D"
        "poisson_solver": "multigrid", # or "fft"
        "particle_shape": 2,
        "n_cell": [32, 32, 40],
        "prob_relative": [3.0],
        "dynamic_size": True,
    },
)
```

The mesh is writable through the model, which is worth noting: ImpactX makes `n_cell`
read-only once `init_grids()` has run, so a model holding one live session could not
offer it. Because the simulator rebuilds the session on every track, it can.

```python
model.set({"sim:n_cell": np.array([16, 16, 24])})   # NDVariable wants an ndarray
```

`examples/space_charge_expansion.py` reproduces the standard benchmark — a cold bunch
coasting down a drift expands to exactly twice its initial size — and asserts it.

## CSR

Two settings, `sim:csr` and `sim:csr_bins`. CSR acts inside bends: radiation emitted on
the curved orbit catches up with the bunch, costing mean energy and adding correlated
spread.

```python
settings={"csr": True, "csr_bins": 150, "particle_shape": 2}
```

### When the 1D model applies

ImpactX, Bmad and elegant all use the same 1D CSR treatment, following Saldin *et al.*
It projects the bunch onto a line, which is only justified while the bunch is
transversely thin compared with the CSR formation width:

$$\sigma_\perp \ll R\left(\frac{\sigma_z}{R}\right)^{2/3}$$

`csr_validity_margin()` reports the ratio, and `examples/csr_chicane.py` asserts it:

```python
from lume_impactx.wakes import csr_validity_margin

margin = csr_validity_margin(bend_radius=5.73, sigma_z=1e-4, sigma_transverse=1e-4)
# 38.6 -- comfortably valid; approaching 1 means the 1D treatment is breaking down
```

!!! warning "Off-axis beams"
    The standard 1D implementation also assumes the beam centroid stays **near the
    lattice reference orbit**. Where it does not — an FFAG passing several beams of
    different energy through one magnet, or any strongly off-axis trajectory — the
    calculation degrades, and at a kinked reference orbit the kick formally diverges.
    Bmad has since reformulated this around a centroid-based CSR reference trajectory
    (Sagan & Mayes, IPAC2017 [THPAB076](https://accelconf.web.cern.ch/ipac2017/papers/thpab076.pdf)).
    ImpactX's CSR carries the original restriction, so treat a strongly off-axis result
    with suspicion.

!!! note "Check the bin convergence"
    The CSR model bins the bunch longitudinally, so the answer depends on `csr_bins`.
    `examples/csr_chicane.py` scans it. In the moderate regime there (250 MeV, 1 nC,
    100 µm) the result moves ~3% between 50 and 300 bins. Push into a shorter, denser
    bunch and it stops converging altogether while CSR removes several percent of the
    beam energy — check before quoting a number.

## Wakefields

ImpactX has no wakefield element. It exposes `impactx.wakeconvolution`
(`deposit_charge`, `derivative_charge`, `w_l_csr`, `w_l_rf`, `w_t_rf`, `convolve_fft`)
and leaves the assembly to you. Note the shape of those primitives: CSR in the 1D
formalism is itself a convolution of the line-density *derivative* $d\lambda/ds$ with a
kernel, which is why `derivative_charge` sits next to `w_l_csr`. A resistive wall
convolves the density itself, so the module below uses only the binning half. `lume_impactx.wakes` assembles the **resistive-wall**
case into a thin element:

```python
from lume_impactx.wakes import ResistiveWallWake

lattice = [
    elements.Drift(name="pipe", ds=20.0, nslice=40),
    ResistiveWallWake("rw", length=20.0, pipe_radius=0.005, conductivity=5.96e7),
]
```

The wake model is **single-rank only**: it bins and convolves the particles on one
rank, and nothing reduces across ranks, so it refuses to run under MPI rather than
return a quietly wrong line charge density.

It is a zero-length `Programmable` whose push hook bins the bunch by arrival time,
convolves the line charge with the wake function, and applies the energy change. Place it
directly after the drift whose length it stands for.

### The model

The longitudinal wake per unit length of a round pipe, in the Bane–Sands short-range
form, with $s_0 = (2b^2/Z_0\sigma)^{1/3}$:

$$
W_L(s) = \frac{Z_0 c}{\pi b^2}\left[\frac{1}{3}e^{-s/s_0}\cos\!\left(\frac{\sqrt{3}s}{s_0}\right)
         - \frac{\sqrt{2}}{\pi}\int_0^\infty \frac{x^2 e^{-x^2 s/s_0}}{x^6+8}\,dx\right]
$$

and zero for $s < 0$. At $s = 0$ the integral is analytic — $\pi/(12\sqrt2)$ — so the
bracket is exactly $1/4$ and $W_L(0) = Z_0 c/(4\pi b^2)$. The numerical quadrature is
tested against that closed form, and the applied kick is tested to be exactly linear in
bunch charge and in pipe length.

!!! warning "What this model is not"
    It assumes a round, thick, dc-conducting wall and a relativistic beam, and ignores ac
    conductivity and surface roughness. Two implementation limits: the convolution is
    **rank-local**, so it is wrong under MPI; and a bin's own charge contributes the full
    $W_L(0)$ self-wake, where a half-bin correction would be the refinement. Treat it as
    a demonstration of the mechanism, not a validated engineering model — check against a
    dedicated wake code before designing hardware with it.

Writing your own wake is a one-liner — `apply_longitudinal_wake` takes any callable:

```python
from lume_impactx.wakes import apply_longitudinal_wake

apply_longitudinal_wake(sim.beam, my_wake_function, length=10.0, num_bins=128)
```
