# Variables

`LUMEImpactXModel.from_simulator()` generates variables automatically. A five-element
FODO cell yields 149 of them.

## Namespaces

| Pattern | Kind | Writable | Example |
|---|---|---|---|
| `ele:{name}:{attrib}` | scalar / int / bool / str / enum | depends on the element | `ele:quad1:k` |
| `moment:{name}` | `NDVariable`, shape `(n_steps,)` | no | `moment:sigma_x` |
| `moment_final:{name}` | scalar at the lattice end | no | `moment_final:sigma_x` |
| `ref:{key}` | reference-particle input | yes | `ref:kin_energy_MeV` |
| `ref_final:{key}` | reference particle after tracking | no | `ref_final:s` |
| `sim:{key}` | ImpactX simulation setting the simulator carries | mostly | `sim:space_charge` |
| `particles:{name}` | `ParticleGroup` | only `initial_particles` | `particles:final_particles` |
| `run_info:{key}` | facts about the last run | no | `run_info:run_time` |
| `optics:{key}` | linear transfer maps of the lattice | no | `optics:transfer_map` |

## Element attributes

Attributes are discovered from ImpactX's pybind properties rather than from
`to_dict()`, for three reasons:

- `to_dict()` **omits `nslice` when it is at its default**, so `Quad(ds=0.3, k=2.0)`
  reports no `nslice` while `Quad(ds=0.3, k=2.0, nslice=4)` does — two structurally
  identical lattices would get different variable sets.
- `to_dict()` carries **no read/write information**, and writability is genuinely
  per-type: `Drift.aperture_x` is read-only while `Aperture.aperture_x` is writable, and
  `ThinDipole.ds` / `nslice` are read-only while `Drift`'s are not.
- `property.fset is None` maps exactly onto `Variable.read_only`, and stays correct for
  element types that do not exist yet.

Three attributes are forced read-only beyond what ImpactX allows: `nslice`, because it
sets `n_steps` and therefore the shape of every s-series variable; `name`, because it is
part of the variable names themselves; and `unit`, because it selects which unit `k` or
`bscale` is in and that label is baked into the variable when it is generated.

## Duplicate element names

Element names are not unique — ImpactX's own `test_xopt.py` uses `quad1` twice — so
repeats get an occurrence suffix:

```
ele:qf#1:k
ele:qf#2:k
```

Unnamed elements are skipped unless `ElementsConfig(include_unnamed=True)`, in which
case they are addressed as `ele:{type}#{index}:{attrib}`.

## s-series sizing

`moment:*` variables declare an exact shape, because the row count is known before the
run: `n_steps == periods * sum(nslice)`. There is no over-allocation and no NaN padding.
This is why `ele:*:nslice` and `sim:periods` are read-only by default.

## Simulation settings

`sim:{key}` variables are generated only for settings the simulator actually carries —
the defaults it applies, plus anything passed as `settings=`. ImpactX has no readable
default for a parameter that was never set (reading one raises `algo.csr is not set
yet`), so a variable for an unapplied setting would have nothing to report.

To expose one, set it at construction:

```python
simulator = ImpactXSimulator(..., settings={"csr": True, "csr_bins": 150, "particle_shape": 2})
# -> sim:csr, sim:csr_bins, sim:particle_shape now exist
```

Note ImpactX requires `particle_shape` whenever a collective effect (space charge, CSR)
is on, and refuses `init_grids()` without it.

## Linear optics

`optics:transfer_map` is the lattice's 6x6 linear map, and `optics:cumulative_maps` is
the map from the lattice start to each element's exit — its last entry equals
`optics:transfer_map`, and `optics:map_s` gives the matching `s` values. These are the
ImpactX counterpart of lume-bmad's `mat6` output.

They are computed from the lattice at the *initial* reference particle, before tracking,
so they describe the lattice as configured rather than the tracked beam. Elements with no
closed-form linear map contribute identity.

!!! note "Matrix orientation"
    ImpactX returns these as AMReX `SmallMatrix`, which is Fortran ordered. Calling
    `np.asarray()` on one yields the **transpose** — a drift comes back as
    `[[1, 0], [L, 1]]` instead of `[[1, L], [0, 1]]`. The variables here are already
    oriented correctly.

## Conditional moments

`beam_moments()` reports the spin moments (`mean_sx`, `sigma_sx`, …) whether or not
`sim.spin` is on, as exact zeros when it is off. Variables for those are generated only
when `sim.spin` is enabled, so an ordinary run is not padded with meaningless entries.

The eigenemittances (`emittance_1/2/3`) behave differently: ImpactX omits them from
`beam_moments()` entirely unless `sim.eigenemittances` is on, so they are self-gating.
They are listed in `CONDITIONAL_MOMENTS` for symmetry, but that entry is a no-op.

Pass an explicit `MomentsConfig(include=[...])` to bypass the gate.

Per-particle spin is a different matter: `ParticleGroup` cannot represent it and nothing
in LUME analyses it, so `particles:*` variables raise
`UnrepresentableParticleData` for a bunch that actually carries spin rather than handing
back a zeroed one. See the [design notes](design.md).

## Units

Every variable carries a unit where one exists, because `lume-pva` turns them into a
PV's `display.units`. Units come from tables in `lume_impactx.units`, never from the
action classes, so the same attribute name can mean different things on different
elements — `Quad.k` is `1/m^2` while `Sol.ks` is `1/m`.

!!! note "Angles"
    `ExactSbend.phi`, `PRot.phi_in`/`phi_out`, `PlaneXYRot.angle` and `ThinDipole.theta`
    read back in **radians** although their constructors take **degrees**. The unit
    tables label them `rad` to match what you actually get.

## Customising

```python
from lume_impactx import LUMEImpactXModel, VariableMappingConfig
from lume_impactx.config import ElementsConfig, MomentsConfig

config = VariableMappingConfig(
    elements=ElementsConfig(
        include_kinds=["Quad"],                      # quadrupoles only
        exclude_attributes=["dx", "dy", "rotation"],
    ),
    moments=MomentsConfig(include=["sigma_x", "sigma_y", "emittance_x"]),
    sim=None,                                        # skip settings entirely
)
model = LUMEImpactXModel.from_simulator(simulator, config)
```
