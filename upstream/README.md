# Upstream contribution: `beamphysics/interfaces/impactx.py`

`impactx.py` here is a candidate for
[openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics), to sit
alongside the existing `astra`, `bmad`, `elegant`, `gpt` and `impact` interfaces. There
is currently no ImpactX interface there, which is the gap this fills.

It is **generated**, not hand-maintained:

```bash
python scripts/make_upstream_interface.py          # regenerate
python scripts/make_upstream_interface.py --check  # verify it is current
```

The generator slices the ImpactX-free core out of `lume_impactx/utils.py` and rewrites
the imports to openPMD-beamphysics' relative style (`from ..species import ...`).
`lume_impactx/tests/test_upstream.py` regenerates and compares, and separately checks
that the generated module produces numerically identical results to the one this
package uses — so the contribution cannot drift from the tested implementation.

## What it contains

| Name | Purpose |
|---|---|
| `ImpactXRefPart` | An ImpactX reference particle as a plain dataclass, so the conversion needs no live ImpactX object. |
| `particlegroup_to_impactx` | `ParticleGroup` -> ImpactX fixed-s arrays. |
| `impactx_to_particlegroup_data` | The inverse. |
| `pmd_species_of` | Infer a species from a reference particle. |
| `particle_id_from_idcpu` | Split AMReX' packed `idcpu` into a unique `ParticleGroup` id and a validity flag. |
| `refpart_from_openpmd` | Rebuild a reference particle from a `BeamMonitor` file's species attributes. |
| `beam_monitor_iterations` | List the iterations in a `BeamMonitor` file. |
| `read_beam_monitor` / `read_beam_monitor_data` | Read an ImpactX `BeamMonitor` openPMD file into a `ParticleGroup`, or into the data dict `ParticleGroup(data=...)` takes. |
| `UnrepresentableParticleData` | Raised instead of silently dropping spin or runtime per-particle components. |

It imports only numpy and openPMD-beamphysics itself; `openpmd_api` is imported lazily
inside the reader, matching how other optional readers are handled.

Downstream, `beamphysics/particles.py` gains a thin `ParticleGroup.from_impactx()`
classmethod, mirroring `from_bmad` / `from_genesis4`.

## Notes for the reviewer

- ImpactX is s-based with momenta normalized by the reference momentum, so the closest
  existing analogue is `bmad.py`, and the signatures follow it.
- Particles come out in **z-coordinates** (all `z` equal, spread in `t`), matching
  `bmad_to_particlegroup_data`'s `"z": np.zeros(len(p))  # Zero by definition`.
- Frames: the transverse coordinates stay in the local frame relative to the reference
  particle (adding `x_ref` would be wrong on a bent orbit), while `t` is absolute lab
  time -- which is exactly what openPMD's `position/t + positionOffset/t` means in
  ImpactX output. The module docstring says so. Note `positionOffset` is *not* zero in
  ImpactX output; it holds `(x_ref, y_ref, t_ref)`.
- Two ImpactX quirks are handled and documented in the module: `t` is `c*t` in metres
  there, and `qm` is reported in different units depending on how the particles were
  inserted, so it is never trusted on read.
- ImpactX carries `spin_x/y/z` and arbitrary runtime per-particle components, which
  `ParticleGroup` has no place for. This module does not attempt to carry them; it
  raises rather than returning a silently zeroed bunch, and `strict=False` opts out.
  Note the same gap exists in `bmad.py`, whose `write_bmad` carries a `TODO: Spin`.
  **If openPMD-beamphysics wants a general extra-per-particle-array mechanism on
  `ParticleGroup`, that would close it for every s-based code at once** -- worth a
  conversation, and this interface could then carry them instead of refusing.

## Two ImpactX findings this exposed

Both verified against ImpactX 26.08:

1. **`particles_lost` output carries a zeroed reference particle.** `track_particles`
   constructs the loss monitor from a default-constructed `RefPart`, so the file has
   `mass_ref = 0` and `gamma_ref = 0` and its normalized coordinates cannot be
   converted at all. `read_beam_monitor` detects this and asks for an explicit `ref=`.
   Worth filing upstream in ImpactX.
2. **The species in `particles_lost.*` is still named `"beam"`.** Only the file name
   differs. Not a bug, but the earlier draft of this module documented it wrongly.

## Status

**Not yet submitted.** Opening the PR is a deliberate, outward-facing step for the
maintainer of this repository to take.
