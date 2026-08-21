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
| `particlegroup_to_impactx` | `ParticleGroup` → ImpactX fixed-s arrays. |
| `impactx_to_particlegroup_data` | The inverse. |
| `pmd_species_of` | Infer a species from a reference particle. |
| `refpart_from_openpmd` | Rebuild a reference particle from a `BeamMonitor` file's species attributes. |
| `read_beam_monitor` | Read an ImpactX `BeamMonitor` openPMD file into a `ParticleGroup`. |

It imports only numpy and openPMD-beamphysics itself; `openpmd_api` is imported lazily
inside `read_beam_monitor`, matching how other optional readers are handled.

`read_beam_monitor` raises `UnrepresentableParticleData` when a monitor recorded spin or
a runtime per-particle component, rather than returning a bunch that has quietly lost
them. The guard travels with the module, and a test asserts it does — leaving it behind
would make the reader `NameError` at runtime.

## Notes for the reviewer

- ImpactX is s-based with momenta normalized by the reference momentum, so the closest
  existing analogue is `bmad.py`, and the signatures follow it.
- Particles come out in **z-coordinates** (all `z` equal, spread in `t`), matching
  `bmad_to_particlegroup_data`'s `"z": np.zeros(len(p))  # Zero by definition`.
- Two ImpactX quirks are handled and documented in the module: `t` is `c·t` in metres
  there, and `qm` is reported in different units depending on how the particles were
  inserted, so it is never trusted on read.
- ImpactX carries `spin_x/y/z` and arbitrary runtime per-particle components, which
  `ParticleGroup` has no place for. This module does not attempt to carry them; the
  lume-impactx wrappers around it raise rather than return a silently zeroed bunch.
  Note the same gap exists in `bmad.py`, whose `write_bmad` carries a `TODO: Spin`.
  **If openPMD-beamphysics wants a general extra-per-particle-array mechanism on
  `ParticleGroup`, that would close it for every s-based code at once** — worth a
  conversation, and this interface could then carry them instead of refusing.

## Status

**Not yet submitted.** Opening the PR is a deliberate, outward-facing step for the
maintainer of this repository to take.
