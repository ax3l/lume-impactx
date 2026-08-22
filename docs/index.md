<div align="center" markdown>

# 🚧 ⚠️ STATUS WARNING ⚠️ 🚧

### **This repo is a first vibe-coded draft and has seen no manual validation yet.**

</div>

!!! danger "🤖 Written end-to-end by an AI agent"

    Nothing here has been checked by a human against an independent reference.

    🧪 The test suite is green and several results are pinned to closed-form values —
    but **a passing test only proves the code agrees with itself**, not with physics.

    🔬 Treat every number on this site as unverified. Cross-check against ImpactX, Bmad
    or Impact-Z directly before using this for anything that matters.

!!! warning "🩹 Known-shaky areas, in rough order"

    - The **resistive-wall wake model** ([`lume_impactx.wakes`](collective.md#wakefields))
      — invented for a demonstration, not validated against a wake code.
    - The **Bmad lattice translation** ([`from_tao`](bmad.md)) — each element mapping is
      verified against Bmad tracking, but the soft bend edge, a bend roll and Bmad's own
      coarse multipole integrator differ in known ways.
    - Anything touching **MPI** — untested, because no MPI-enabled build was available.

---

# LUME-ImpactX

Tools for using [ImpactX](https://impactx.readthedocs.io) in
[LUME](https://www.lume.science).

ImpactX is a beam-dynamics code with a native Python API. This package exposes it
through the LUME model interface, so an ImpactX lattice can be driven by the same
`get()` / `set()` calls used by `lume-cheetah` and `lume-bmad`, served over EPICS with
`lume-pva`, or optimized with Xopt.

## What it provides

| Piece | Purpose |
|---|---|
| `ImpactXSimulator` | Holds a lattice, reference particle and beam; builds and tracks a simulation on demand. |
| `LUMEImpactXModel` | A LUME action model whose variables are generated from the lattice, reference particle, settings and beam moments. |
| `StagedImpactXModel` | Chains sections, passing particles downstream. |
| `particlegroup_to_impactx` / `read_beam_monitor` | Conversion between openPMD-beamphysics `ParticleGroup` and ImpactX beam data, in memory or from a `BeamMonitor` file. |
| `archive` / `load_archive` | HDF5 persistence of a whole simulation. |
| `plot_moments_with_layout` | Beam moments against `s`, with a lattice layout strip. |
| `lume_impactx.wakes` | Resistive-wall wakefield as a drop-in lattice element. |
| `ImpactXSimulator.from_tao` | Build a simulation from a Bmad/Tao model — beam and lattice, element by element. |
| `LUMEImpactXModel.from_tao` | The same, straight to a LUME model with generated variables. |
| `sim.run()` / `sim.particles["end"]` | The lume-impact three-verb shape: `.from_tao()`, `.run()`, `.particles[...]` keyed by Bmad element name. |

## Installation

ImpactX is not a hard dependency, because the accelerated builds (OpenMP, MPI, GPU)
ship through conda-forge rather than PyPI:

```bash
conda create -n lume-impactx -c conda-forge python=3.12 impactx
conda activate lume-impactx
pip install lume-impactx
```

For a serial start, or in CI, the sequential-CPU PyPI wheel works too:

```bash
pip install "lume-impactx[impactx]"     # pulls impactx-noacc
```

!!! warning "MPI builds"
    Importing `lume_impactx` imports `mpi4py` when ImpactX was built with MPI. This is
    load-bearing: `ImpactX.finalize()` calls `MPI_Finalize()` if AMReX owns MPI, which
    makes the *second* simulation in a process abort. Letting `mpi4py` own MPI keeps
    `finalize()` repeatable. `scripts/spike_amrex_cycles.py` checks this for your build.

    ImpactX always uses `MPI_COMM_WORLD` and cannot be handed a communicator, so a
    `LUMEImpactXModel` is effectively single-rank. Run multi-rank ImpactX directly.
