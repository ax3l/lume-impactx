<div align="center">

# 🚧 ⚠️ STATUS WARNING ⚠️ 🚧

### **This repo is a first vibe-coded draft and has seen no manual validation yet.**

</div>

> [!CAUTION]
> 🤖 **Written end-to-end by an AI agent.** Nothing here has been checked by a human
> against an independent reference.
>
> 🧪 The test suite is green and several results are pinned to closed-form values —
> but **a passing test only proves the code agrees with itself**, not with physics.
>
> 🔬 Treat every number as unverified. Cross-check against ImpactX, Bmad or Impact-Z
> directly before using this for anything that matters.
>
> 🩹 Known-shaky areas, in rough order: the **resistive-wall wake model**
> (`lume_impactx/wakes.py` — invented for a demo, not validated), the **Bmad lattice
> translation** (each element mapping is verified against Bmad tracking, but the soft
> bend edge, a bend roll and Bmad's own coarse multipole integrator differ in known
> ways), and anything touching **MPI** (untested — no MPI build was available).

---

# lume-impactx

ImpactX-specific implementation of LUMEModel classes for virtual accelerators.

[ImpactX](https://impactx.readthedocs.io) is a beam-dynamics code with a native Python
API. This package exposes it through the [LUME](https://www.lume.science) model
interface, so an ImpactX lattice can be driven by the same `get()` / `set()` calls used
by `lume-cheetah` and `lume-bmad`, served over EPICS with `lume-pva`, or optimized with
Xopt.

## What this package provides

- `ImpactXSimulator`: holds a lattice, reference particle and beam, and builds and
  tracks a simulation on demand.
- `LUMEImpactXModel`: a LUME action model whose variables are generated from the
  lattice, reference particle, simulation settings and beam moments.
- `StagedImpactXModel`: chains sections, passing particles downstream.
- Converters between openPMD-beamphysics `ParticleGroup` and ImpactX beam data, in
  memory or from a `BeamMonitor` openPMD file.
- `archive` / `load_archive` for HDF5 persistence, and
  `plot_moments_with_layout` for beam moments against `s`.

## Installation

ImpactX is not a hard dependency, because the accelerated builds (OpenMP, MPI, GPU)
ship through conda-forge rather than PyPI:

```bash
conda create -n lume-impactx -c conda-forge python=3.12 impactx
conda activate lume-impactx
pip install -e .
```

For a quick serial start, or in CI, the sequential-CPU PyPI wheel works too:

```bash
pip install -e ".[impactx]"     # pulls impactx-noacc
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

Quick import check:

```bash
python -c "import impactx, lume_impactx; print('ok', lume_impactx.__version__)"
```

## Example

```python
from impactx import distribution, elements
from lume_impactx import ImpactXSimulator, LUMEImpactXModel

ns = 5
lattice = [
    elements.Drift(name="drift1", ds=0.25, nslice=ns),
    elements.Quad(name="quad1", ds=1.0, k=1.0, nslice=ns),
    elements.Drift(name="drift2", ds=0.5, nslice=ns),
    elements.Quad(name="quad2", ds=1.0, k=-1.0, nslice=ns),
    elements.Drift(name="drift3", ds=0.25, nslice=ns),
]
waterbag = distribution.Waterbag(
    lambdaX=3.9984884770e-5, lambdaY=3.9984884770e-5, lambdaT=1.0e-3,
    lambdaPx=2.6623538760e-5, lambdaPy=2.6623538760e-5, lambdaPt=2.0e-3,
    muxpx=-0.846574929020762, muypy=0.846574929020762, mutpt=0.0,
)

simulator = ImpactXSimulator(
    lattice=lattice,
    ref={"species": "electron", "kin_energy_MeV": 2.0e3},
    distribution=waterbag,
    npart=10_000,
    bunch_charge_C=1.0e-9,
)

model = LUMEImpactXModel.from_simulator(simulator)
model.get("moment_final:sigma_x")     # 7.57e-05
model.set({"ele:quad1:k": 1.2})       # writes, then re-tracks
model.get("moment_final:sigma_x")     # 6.46e-05  (npart=10_000)
model.reset()                         # back to the construction-time state, exactly

fig = model.plot(y=("sigma_x", "sigma_y"), include_labels=True)
```

Variable names follow `ele:{name}:{attrib}`, `moment:{name}` (an s-series),
`moment_final:{name}`, `ref:{key}`, `ref_final:{key}`, `sim:{key}`, `particles:{name}`
and `run_info:{key}`. See the [Variables](docs/variables.md) page.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

The [design notes](docs/design.md) record the non-obvious decisions and the
measurements behind them.

## A note on MPI

Importing `lume_impactx` imports `mpi4py` when ImpactX was built with MPI. This is
deliberate and load-bearing: `ImpactX.finalize()` calls `MPI_Finalize()` if AMReX owns
MPI, which would make the *second* simulation in a process abort. Letting `mpi4py` own
MPI keeps `finalize()` repeatable. `scripts/spike_amrex_cycles.py` checks this holds for
your build.

ImpactX always uses `MPI_COMM_WORLD` and cannot be handed a custom communicator, so a
`LUMEImpactXModel` is effectively single-rank. Run multi-rank ImpactX directly instead.

## Contributing upstream

`upstream/impactx.py` is a candidate `beamphysics/interfaces/impactx.py` for
[openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics), which
has no ImpactX interface today. It is generated from `lume_impactx/utils.py` and tested
for drift; see [`upstream/README.md`](upstream/README.md). It has not been submitted.

## Testing

```bash
pytest                  # everything
pytest -m "not slow"    # only the tests that need no ImpactX run
```
