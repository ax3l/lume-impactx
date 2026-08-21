# Design notes

Decisions here that are not obvious, and the measurements behind them.

## Rebuild the session on every track

ImpactX tracking is destructive: it consumes the particle container and advances the
reference particle's `s`. Three ways out were measured against ImpactX 26.06/26.08.

| Approach | Outcome |
|---|---|
| Resample the distribution in place | **Rejected.** Resampling within one session advances a global RNG: two identical calls differed by 1.8e-2 relative in `sigma_x`. Every `set()` would return a different answer for the same inputs. |
| Hold one session, snapshot the beam and re-inject | Works and is bit-exact, but freezes the mesh — `n_cell` and friends are read-only after `init_grids()`, so they could never be exposed as variables. |
| **Rebuild the session** | **Chosen.** Bit-exact, because a fresh session re-seeds the RNG. Keeps every parameter writable. |

Rebuilding is not the expensive option it sounds like. With a 32³ space-charge mesh and
20k particles it cost **777 ms/iteration against 747 ms** for snapshot-and-re-inject — a
4% difference. Tracking dominates, not `init_grids()`.

## mpi4py must own MPI

`ImpactX.finalize()` calls `amrex::Finalize()`, which calls `MPI_Finalize()` when AMReX
initialized MPI. `MPI_Finalize` is terminal, so the *second* `ImpactX()` in a process
aborts:

```
Attempting to use an MPI routine (internal_Comm_dup) before initializing or after finalizing MPICH
```

It is an abort, not an exception — buffered stdout is lost with it. Importing `mpi4py`
first makes mpi4py the owner, AMReX leaves MPI alone, and `finalize()` becomes
repeatable. This is the same trick ImpactX uses in its own `tests/python/conftest.py`.
`lume_impactx/_mpi.py` does it at import time; `scripts/spike_amrex_cycles.py` verifies
it (200 cycles at ~6 ms each, bit-identical).

The alternative — `amr.initialize(...)` once and never `finalize()` — was also tried and
is worse: an external initialize bypasses ImpactX's `overwrite_amrex_parser_defaults`,
leaving `particles.do_tiling` unset, and OpenMP tracking then asserts.

## Particles are exchanged in z-coordinates

ImpactX holds particles at a common `s` with a spread in arrival time. In
`ParticleGroup` terms that *is* z-coordinates — all `z` equal, `t` varying — which is
the convention `beamphysics/interfaces/bmad.py` already uses for s-based codes. So the
conversion is a direct algebraic map with no drifting.

### The local frame

Beam coordinates are **transverse to the reference orbit**, and the momenta are
normalized by the *magnitude* of the reference momentum, `|p_ref| = beta_gamma`. The
reference particle's own `x`, `y`, `px`, `py` are **lab** quantities and must not be
mixed in. Getting this wrong is invisible on a straight beamline — there `ref.px = 0`
and `ref.pz == |p_ref|`, so the correct and incorrect forms agree exactly — and wrong
the moment the orbit bends:

| bend angle | `sigma_energy` reported, dividing by `ref.pz` |
|---|---|
| 0° | correct |
| 15° | ×0.966 |
| 30° | ×0.866 |
| 60° | ×0.500 |

The ratio is `cos(phi) = ref.pz / |p_ref|`, while ImpactX's own `sigma_pt` stays
invariant. The converted bunch is therefore in the local frame at the reference
particle's `s`: transverse coordinates are displacements from the orbit, `z` is zero for
every particle, and the orbit's lab position lives in the reference snapshot instead.
This matches `beamphysics/interfaces/bmad.py`, which does the same for Bmad.

It also settles what a staged hand-off must carry. The downstream section's reference
particle stays on **its own** axis — seeding it with the upstream's lab position and
angle would bend its orbit away from the section it describes. Only the arrival time
`t` has to be carried, so `position_t` stays a small offset rather than the bunch's
absolute time of flight — along with the arc length `s`, and the reference **energy**,
which an upstream RF cavity changes and which the beam momenta are normalized by.
The lab footprint (`x`, `y`, `z`, `px`, `py`) is deliberately not carried at all:
carrying `z` alone would put the reference at the machine's longitudinal position but on
this section's transverse axis, which is neither lab-correct nor section-local.

With that, two chained sections across a 30° bend reproduce the one-shot lattice to
~1e-13 — beam-width quantities to 4e-16, and the centroid, which is a difference of
large numbers, to 2e-13. Across an accelerating cavity the agreement is 4e-14; without
carrying the reference energy it is 100% wrong.

Two unit traps are worth knowing:

- **`qm` has no single unit.** `add_n_particles` is documented as, and behaves as, 1/eV
  (`-1.957e-6` for electrons) and stores that value verbatim. `ImpactX.add_particles` —
  the distribution path — stores SI C/kg (`-1.759e11`). So `to_df()["qm"]` reports
  whichever unit the particles were inserted with. The readers here never trust it;
  species comes from the reference particle instead.
- **`t` is seconds in `ParticleGroup` but `c·t` in metres in ImpactX.**

`ParticleGroup.drift_to_z()` defaults to the bunch's *mean* z. That is what the
converter wants — in the local frame the bunch's own plane *is* the reference plane — so
it calls `drift_to_z()` with no argument. Passing a lab `z` instead drifts the bunch
bodily backwards; doing that by accident cost a factor of 4.8 in `sigma_x` for a bunch
handed over from 5 m downstream, silently.

## Archiving works around a degrees/radians bug

`to_dict()` returns radians for the angle of every type in `DEGREE_ELEMENTS`
(`ExactSbend`, `PlaneXYRot`, `PRot`, `ThinDipole`) while `from_dicts()` expects degrees,
so a naive round-trip divides those angles by 57.3 — an `ExactSbend` built with `phi=30`
comes back as `0.0091 rad` instead of `0.5236 rad`. `to_dict(in_degrees=True)` fixes it,
and that keyword exists *only* on the affected types, so it is applied conditionally.

## Spin and other per-particle records

ImpactX bunches can carry more than `ParticleGroup` models: `spin_x/y/z` when
`sim.spin` is on, plus any runtime component added with `add_real_comp`.
`ParticleGroup` has no place for those — and nothing downstream analyses, chains or
plots spin today. Checked against openPMD-beamphysics 0.16.0 and lume-base 0.5.0:
`ParticleGroup` has no spin support in `particles.py`, `statistics.py`, `plot.py` or
`units.py`; `readers.py` knows `spin` only as an openPMD record name; and lume-base,
lume-cheetah, lume-bmad and lume-impact mention spin nowhere.

So this package does not carry it. What it does not do is drop it quietly:
`particle_container_to_particlegroup` and `read_beam_monitor` raise
`UnrepresentableParticleData` when a bunch actually carries spin or a runtime
component, rather than returning a silently zeroed bunch that would look plausible and
be wrong.

The test is exact rather than heuristic: ImpactX always allocates `spin_x/y/z`, and
leaves them at *identically zero* unless the beam was seeded with a spin distribution.
Note `sim.spin = True` alone is not the gate — the `spin_distr` argument to
`add_particles` is. So "any non-zero component" has no false positives: zero means there
is genuinely nothing to lose.

`ImpactXSimulator` has no way to pass a spin distribution, so a bunch it produces always
converts. The guard is for containers assembled by hand and for `read_beam_monitor`
reading a file from a spin-seeded run — that path is reachable and tested.

The refusal is deferred, not raised during tracking. A run with `sim.spin` on is
perfectly usable through LUME for moments and plots; only the per-particle hand-off has
no representation. `ImpactXSimulator.track()` records the problem and
`final_particles` re-raises it, so the failure lands where it actually matters.

Spin **moments** are unaffected — they are ordinary scalars. They are exposed as
variables when `sim.spin` is on, and gated off when it is not, because
`beam_moments()` reports them either way and twelve always-zero variables on every
ordinary run is just noise.

## Known limitations

- `add_n_particles` assigns its own ids, so `pg.id` does not survive injection.
- Matrix-valued element attributes (`LinearMap.R`, `SpinMap.A`) cannot be archived:
  `archive()` raises `TypeError` rather than dropping them silently.
- A model is effectively single-rank: ImpactX always uses `MPI_COMM_WORLD`.
