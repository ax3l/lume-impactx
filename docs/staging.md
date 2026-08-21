# Staging

`StagedImpactXModel` chains sections, passing each stage's final bunch to the next.

```python
from lume_impactx import ImpactXSimulator, StagedImpactXModel

upstream = ImpactXSimulator(
    lattice=first_half, ref=ref, distribution=waterbag,
    npart=10_000, bunch_charge_C=1e-9,
)
downstream = ImpactXSimulator(
    lattice=second_half, ref=ref,
    initial_particles=upstream.final_particles,
)

staged = StagedImpactXModel.from_simulators(
    [upstream, downstream], prefixes=["a:", "b:"]
)
staged.set({"a:ele:quad1:k": 1.3})
staged.get("b:moment_final:sigma_x")
```

## Prefixes are not optional

`StagedModel` refuses duplicate variable names across stages, and two sections of a real
machine usually reuse element names. `from_simulators` defaults to `stage0:`, `stage1:`,
… ; pass `prefixes=` or per-stage `VariableMappingConfig(prefix=...)` for something
more readable.

## Two behaviours this class fixes

**Re-seeded stages re-run.** `lume.staged_model.StagedModel._set` only calls
`model.set(...)` for a stage that has variables in *this* call:

```python
if model_values:
    model.set(model_values)
```

So writing only an upstream variable — the common case when scanning an injector
setting — hands the downstream stage a new bunch but never re-tracks it, and its
`final_particles` go stale. `StagedImpactXModel` re-runs any stage whose input changed.

**Sections inherit the beam's energy and timing.** A fresh `ImpactX` starts its
reference particle at `t = s = 0` and at the energy this stage was configured with. But
an upstream RF cavity or `ChrAcc` changes the beam energy, and beam momenta are
normalized by the reference momentum — so a stage holding its own configured energy
silently rescales the incoming bunch. Across an accelerating cavity that put the staged
result 100% away from the one-shot lattice.

`StagedImpactXModel` carries the upstream reference particle forward as `ref_origin`,
from which the downstream stage takes the arrival time `t`, the arc length `s` and the
reference energy. Because the energy comes from upstream, `ref:kin_energy_MeV` is
generated **read-only** for such a stage.

The lab footprint (`x`, `y`, `z`, `px`, `py`) is deliberately not carried: beam
coordinates are transverse to the reference orbit, so the bunch is already expressed
relative to it, and seeding the downstream reference with the upstream's lab position
and angle would bend its orbit off the axis its own lattice describes. Beam widths are
identical with or without `ref_origin`; what it changes is absolute timing and energy.

Building a downstream simulator by hand, pass it yourself:

```python
downstream = ImpactXSimulator(
    lattice=second_half, ref=ref,
    initial_particles=upstream.final_particles,
    ref_origin=upstream.results["ref_final"],
)
```

Without `ref_origin` the origin is taken from the bunch itself, which is right for a
standalone run but only as good as the bunch centroid for absolute arrival time.
