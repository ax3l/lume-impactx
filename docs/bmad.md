# From Bmad / Tao

`ImpactXSimulator.from_tao()` builds a simulation from a running
[Tao](https://www.classe.cornell.edu/bmad/) model.

```python
from pytao import Tao
from lume_impactx import ImpactXSimulator

tao = Tao(init_file="tao.init", noplot=True)
tao.cmd("set global track_type = beam")
tao.cmd("set beam saved_at = *")
tao.track_beam()

sim = ImpactXSimulator.from_tao(tao)
sim.track()
```

Bmad and pytao are **not dependencies** — install them from conda-forge:

```bash
conda install -c conda-forge bmad pytao
```

## The beam is translated faithfully; the lattice is bridged

These two halves have very different standing, and it matters.

**Beam — exact.** The bunch comes from `tao.particles(ele)`, which pytao already returns
as an openPMD-beamphysics `ParticleGroup`, and that is precisely what ImpactX's converter
consumes. Nothing resamples or re-centres it. Against a Bmad FODO cell the energy
distribution comes through at machine precision:

| quantity | agreement with Bmad |
|---|---|
| `mean_energy`, `sigma_energy` | 2e-15 |
| charge, particle count | exact |
| `sigma_x` | 4e-7 |
| `norm_emit_x` | 3e-5 |

The transverse residuals are **not** from the beam hand-off — they are the lattice.

**Lattice — a bridge.** There is no direct Bmad→ImpactX translator. `from_tao` routes
Bmad → MAD-X → ImpactX using each code's own exporter and importer, so it carries only
what all three represent. ImpactX itself warns that its MAD-X parser is "under active
development and provided as a preview".

!!! warning "What the bridge drops, silently"

    - **Numerics control.** `nslice` is applied uniformly; Bmad's per-element integrator
      choice, `num_steps`, `ds_step` and tracking/mat6 methods have no MAD-X
      representation. *TODO: map these per element once a direct translator exists.*
    - Element types MAD-X cannot express: taylor maps, wigglers/undulators, `patch`
      elements, `em_field`, and Bmad's `overlay` / `group` / `girder` control structures.
    - Multipole error tables, aperture definitions, fringe-field models and higher-order
      edge effects.
    - Multi-branch lattices — only the tracked branch is written.

    Check the result against the Bmad model before trusting a number from it.

If you already have an ImpactX lattice you trust, pass it and use `from_tao` only for the
beam:

```python
sim = ImpactXSimulator.from_tao(tao, lattice=my_impactx_elements)
```

## Element models

`min_model` picks the lowest ImpactX element-model tier the importer may use:

| `min_model` | elements produced |
|---|---|
| `"linear"` | `Quad`, `Drift` |
| `"paraxial"` | `ChrQuad`, `ChrDrift` |
| `"exact"` (default here) | `ExactQuad`, `ExactDrift` |

`from_tao` defaults to `"exact"`, since a Bmad user is usually after fidelity rather than
speed. The parameter landed after ImpactX 26.08 — on an older build the lattice loads
with linear models and a `TaoTranslationWarning` says so, rather than failing.

## A Tao export quirk worth knowing

Tao writes a *labelled* MAD-X beam definition terminated by `;;`:

```
beam_def: Beam, Particle = Electron, Energy = 6.0E-003, Npart = 0.0E+000;;
```

ImpactX's parser reads that as an **empty** species and aborts the entire load with
`Unknown MAD-X particle species requires explicit MASS and CHARGE in the BEAM command`.
Deleting the line does not help either — the parser requires a BEAM command. So the
bridge rewrites it into the bare lowercase form, using the species and energy already
taken from Tao. Worth fixing on one side or the other upstream.

## Using the pieces separately

```python
from lume_impactx.interfaces.bmad import beam_from_tao, lattice_from_tao

reference, particles = beam_from_tao(tao, ele="BEGINNING")
lattice = lattice_from_tao(tao, nslice=10, min_model="exact")
```

`reference_from_tao` uses Bmad's **design** reference energy `E_TOT`, not the bunch mean.
That is the faithful choice: ImpactX phase-space coordinates are offsets from the
reference particle, so the design reference reproduces Bmad's own `pz` offsets instead of
re-centring the bunch.
