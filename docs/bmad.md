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

## `.from_tao()`, `.run()`, `.particles[...]`

The same three-verb shape lume-impact uses, so a Tao model can be driven the way you
would drive Impact-T or Impact-Z:

```python
sim = ImpactXSimulator.from_tao(tao)
sim.run()                       # an alias for .track()

sim.particles["end"]            # the bunch at the end of the lattice
sim.particles["BEGINNING"]      # and at the start
```

`particles` is keyed by **Bmad element name**, and lookup is case-insensitive because
Tao returns names upper case. Every Bmad `marker`, `monitor` and `instrument` is
captured — a superset of Impact-Z's `write_beam_eles=("monitor::*", "marker::*")` — so a
mid-lattice monitor called `SCREEN1` shows up as `sim.particles["screen1"]`.

Names are not unique, and the mapping does not pretend otherwise:

- A line that **uses one element twice** gives two elements with one name. Occurrences
  after the first become `SCREEN1##2`, `SCREEN1##3`, in beam order.
- Tracking **several periods** fires the hook once per element per turn. Turns after the
  first append `@2`, `@3`.

`initial`, `beginning`, `final` and `end` are reserved: they always mean the run's own
endpoints, so a captured element cannot shadow them. lume-impact's own spellings,
`initial_particles` and `final_particles`, work too.

Capture is in memory and does **not** modify the lattice. It uses ImpactX's
`sim.hook["after_element"]`, reading `sim.beam.to_df()` at the element's exit — the
mechanism ImpactX's documentation prescribes for in-situ analysis:

> The `Programmable` element is for *replacing* a beamline element's particle push. For
> in-situ **analysis** of the beam, use `sim.hook` callbacks with `sim.beam` instead.

A `Programmable` probe would also be wrong mechanically: its `push` hook fires once per
**slice**, not once per element (measured — `nslice=4` gives four calls), so it behaves
as a probe only by accident of being built with `nslice=1`. ImpactX's own `BeamMonitor`
is the other alternative, but it writes openPMD to disk, which would leave a trail of
files behind a get/set loop.

!!! warning "Capture is not free"
    Each capture copies a whole `ParticleGroup`. Measured on 50 000 particles with 42
    capture points, tracking went from 289 ms to 2590 ms — a **9× slowdown** — and held
    151 MB. Pass `capture=False` to switch it off; the ends still resolve, from the
    run's own initial and final bunches.

    `LUMEImpactXModel.from_tao` therefore defaults to `capture=False`: a LUME model
    re-tracks on every `set()`, and no generated variable reads the captured bunches.
    Pass `capture=True` to opt in.

`capture_at` on the simulator takes the same list of element names directly, matched
case-insensitively. Names that never match an element warn after the run rather than
silently capturing nothing.

## Driving it as a LUME model

`ImpactXSimulator.from_tao` gives you a simulation. `LUMEImpactXModel.from_tao` goes one
step further and hands back a LUME model: it translates, tracks once, and generates the
action variables, so the result can be driven by `get()`/`set()` or served over EPICS by
`lume-pva` without any further wiring.

```python
from lume_impactx import LUMEImpactXModel

model = LUMEImpactXModel.from_tao(tao, nslice=16)

model.get("moment_final:sigma_x")      # 1.578686e-03
model.set({"ele:QF:k": 1.5})           # writes, then re-tracks
model.get("moment_final:sigma_x")      # 1.356999e-03
model.reset()                          # back to the translated lattice
```

It is exactly `LUMEImpactXModel.from_simulator(ImpactXSimulator.from_tao(tao, ...))`, and
takes the same translator keywords — `ele`, `lattice`, `nslice`, `species`, `settings`,
`skip_unsupported` — plus `config` and `dummy_run`.

Variable names use Bmad's element names as they come out of Tao, which are **upper case**
— `ele:QF:k`, not `ele:qf:k`. Repeated names are disambiguated by index: a FODO cell whose
two drifts are both called `D` generates `ele:D#1:ds` and `ele:D#2:ds`.

!!! tip "Batch writes when tracking is expensive"
    Every `set()` re-runs the whole simulation, which with space charge on can take
    minutes. Pass `dummy_run=True` to write several variables first and track once:

    ```python
    model = LUMEImpactXModel.from_tao(tao, dummy_run=True)
    model.set({"ele:QF:k": 1.5, "ele:QD:k": -1.5})
    model.simulator.track()
    ```

## Both halves are direct translations

**Beam — exact.** The bunch comes from `tao.particles(ele)`, which pytao already returns
as an openPMD-beamphysics `ParticleGroup`, and that is precisely what ImpactX's converter
takes. The reference particle uses the *design* energy `E_TOT` at that element, not the
bunch mean, so a mis-centred bunch stays mis-centred instead of being silently re-centred.

**Lattice — element by element.** Every element is translated directly into ImpactX
elements. No intermediate format is involved. ImpactX's *exact* models are used
throughout (`ExactDrift`, `ExactQuad`, `ExactSbend`, `ExactMultipole`), because the
paraxial ones disagree with Bmad at the 5e-5 level.

Every mapping below was established against Bmad itself — by comparing Tao's `ele_mat6`
taken into ImpactX's basis, and by tracking the same 64-particle bunch through both codes
at 100 MeV with a 5e-4 momentum spread. The agreement column is the worst relative
coordinate difference measured, and `lume_impactx/tests/test_bmad.py` asserts it: the
`test_tracking_matches_bmad` cases re-run the comparison against Bmad on every test run.

| Bmad | ImpactX | agreement |
| --- | --- | --- |
| `drift`, `pipe`, `monitor`, `instrument`, collimators | `ExactDrift` | 2.8e-15 |
| `marker`, zero-length drift-like | `Marker` | exact |
| `quadrupole` | `ChrQuad(k=K1)` | 1.9e-14 |
| quadrupole fringe | `QuadEdge(k=K1)` at each end | 1.9e-14 |
| `sbend` / `rbend` body | `ExactSbend(phi=ANGLE)` | 1.4e-11 |
| bend pole faces, incl. `FINT`/`HGAP` | `DipEdge(psi, rc, g=2·HGAP, K2=FINT, K3=0)` | 2.2e-9 |
| `sbend` with `k1`/`k2`, Cartesian multipoles | `ChrQuad`+`ThinDipole`(+`Multipole`) steps | 4.7e-6 at 32 steps |
| `sbend` with `k1`, `exact_multipoles=vertically_pure` | `ExactCFbend` | 2.5e-7 |
| zero-angle `sbend` with `k1` | `ChrQuad(k=K1)` | 2.2e-14 |
| `solenoid` | `ChrAcc(ez=0, bz=KS·βγ)` | 2.9e-9 |
| `sextupole` | `ExactMultipole(k_normal=[0,0,K2])` | 9.2e-11 (see note) |
| `octupole` | `ExactMultipole(k_normal=[0,0,0,K3])` | 1.2e-9 |
| `hkicker`/`vkicker`/`kicker` | `Kicker(xkick, ykick)` | 1.3e-14 |
| `rfcavity` | sliced `ExactDrift` + `ShortRF` | 8.5e-8 |
| `lcavity`, travelling wave | sliced `ExactDrift` + `ShortRF` | 9.6e-6 |
| `x_offset`/`y_offset` | `dx`/`dy` (same sign) | 6.2e-9 |
| `tilt` | `rotation = +degrees(TILT)` | 2.2e-14 |
| bend `ref_tilt` | `rotation` on body *and* edges | 1.2e-11 |
| bend `roll` | half bends around a centre `Kicker` | 99.93%, see below |
| `patch`, tilt only | `PlaneXYRot(-degrees(TILT))` | 5.6e-16 |
| `wiggler`, periodic planar/helical | `LinearMap` (analytic Bmad map) | 4.2e-15 vs Bmad's linear map; 2.4e-5 vs full `bmad_standard` |
| zeroed thin `multipole` | `Marker` | exact |
| `is_on = F` (straight elements) | `ExactDrift` of the same length | 6.0e-15 |
| aperture limits | `Aperture`, shape from `aperture_type` | exact |

The sextupole figure is measured with Bmad's own integrator converged. At Bmad's default
of a single step the two differ by 2.4e-6 — that is Bmad's integration error, not the
translation's.

!!! note "Exact is not always the right model"
    ImpactX's `Exact*` elements are the more physical maps, but Bmad's `bmad_standard`
    body for a quadrupole, a solenoid and a combined-function bend is **paraxial in
    (x, y) and exact in energy** — `track_a_bend.f90:111` says so outright. That is
    precisely what ImpactX's `Chr*` family models, which is why `ChrQuad` beats
    `ExactQuad` by 250,000× here (1.9e-14 against 4.5e-9). Bmad's *drift* and *pure bend
    body* really are exact, so `ExactDrift` and `ExactSbend` win there. The choice is
    made per element, from what Bmad actually does.

!!! note "Sign conventions are ImpactX's, not Impact-Z's"
    The tilt sign is `+degrees(TILT)` here, where lume-impact's Impact-Z translator needs
    a *negated* tilt. Two codes from the same family are not interchangeable, and each
    sign above was checked against its negated alternative, which is wrong by O(1).

## What differs, and by how much

These are model differences, not mapping errors, and each emits a
`TaoTranslationWarning` naming the element — except the multipole integrator, where the
coarse side is Bmad rather than the translation.

**Some bend fringe types use a different map.** `basic_bend` (Bmad's default) and
`linear_edge` use the Hwang & Lee map that ImpactX's `DipEdge` implements, and agree to
2.2e-9 and 1.5e-5 *including* `FINT`/`HGAP`, which map exactly onto `g = 2·HGAP`,
`K2 = FINT`. But `full` uses a PTC Lie map in Bmad and `sad_full` uses SAD's, which no
`DipEdge` parameter reproduces: residuals of 6.1e-5, 3.3e-4 and 7.9e-5 for `full`,
`sad_full` and `soft_edge_only`.

**Bmad integrates multipoles coarsely by default.** A `sextupole` or `octupole` is
tracked with `num_steps` drift-kick-drift steps, and Bmad defaults to *one*. ImpactX's
`nslice` steps are finer, so the two differ by Bmad's integration error, which grows with
strength: a `k2 = 25` sextupole differs by 2.4e-6 at Bmad's default, falling to 9.2e-11
once `num_steps` is raised to 200, while a `k3 = 80` octupole is already converged at
1.2e-9. ImpactX is the more accurate side here, so this is documented rather than warned
about — raise `num_steps` in Bmad to close it.

**A combined-function bend depends on Bmad's `exact_multipoles`.** ImpactX's
`ExactCFbend` expands multipoles in curvilinear coordinates, so it matches Bmad's
`vertically_pure` setting (2.5e-7) but not its default `off`, which is Cartesian. That
gap is a *convention*, not convergence: it is unmoved by Bmad `num_steps=400` and
`integrator_order=6`, and by ImpactX `int_order` 2→6 and `mapsteps` 10→400. For Bmad's
default the bend is instead split into `ChrQuad`/`ThinDipole` steps in Bmad's own model,
converging as 7.5e-5 at 8 steps, 4.7e-6 at 32 and 2.9e-7 at 128. (ImpactX's own defaults
of `int_order=2, mapsteps=10` are under-converged for `ExactCFbend` — worth 10× — so
`int_order=4, mapsteps=100` is used.)

**A bend `roll` has no ImpactX equivalent and is carried as its effect.** This was
checked rather than assumed: `Alignment(rotation=...)`, `PlaneXYRot` and `PRot` all turn
the reference orbit with the magnet, which is `REF_TILT`, not `ROLL`. `PlaneXYRot`
measures identical to `rotation` (0.9497 versus 0.9492 against a rolled bend), and `PRot`
rotates in the *x-z* plane, which is pole-face geometry. So the roll is applied as what
it physically does: an on-axis particle leaves a bend rolled by `psi` with
`px = ANGLE·(1 − cos psi)` and `py = −ANGLE·sin psi`, placed as a thin kick between two
half bends. That captures **99.93%** of the dominant out-of-plane kick at every roll from
1e-4 to 0.1 rad. The much smaller in-plane component is captured less well — 23% at
roll 1e-4, 75% at 1e-3, 97% at 1e-2 — but it is 3700× smaller in absolute terms at the
small-roll end. Dropping the roll, as a translator without this would, captures none of
either.

**A standing-wave `lcavity` is indicative only.** The reference-energy change itself is
carried to 3e-12 — `ShortRF` updates ImpactX's reference particle directly — and a
travelling-wave cavity tracks to 9.6e-6. But Bmad's Rosenzweig–Serafini edge focusing and
the standing-wave ponderomotive focusing are not modelled, and for the standing-wave
default that costs **9.1e-2**. The warning says which case you are in.

## Skipping what cannot be translated

`skip_unsupported=True` replaces an element with no verified equivalent by a **drift of
its own length**, not by a marker, so everything downstream stays at the right `s`. That
matters more than it sounds: an element replaced by a marker moves everything after it.

The drift is not offered as an equivalent. Measured against Bmad on `cu_hxr`'s own
undulators, a drift was **7.0e-3** out for an HXR segment and **2.5e-1** out for the
laser-heater undulator, which is why wigglers are now translated properly instead — see
[Wigglers and undulators](#wigglers-and-undulators). LCLS `cu_hxr` translates in strict
mode: 9907 ImpactX elements over its full 1750.883 m.

## Wigglers and undulators

A wiggler becomes a **`LinearMap`** carrying an analytic copy of Bmad's own averaged
wiggler map, built from the element's `B_MAX`, `L_PERIOD`, `KX` and `P0C` rather than
scraped from Tao, so it follows a change of energy or field instead of going stale.

That a 6x6 suffices is a fact about Bmad, not a concession. `bmad_standard` wiggler
tracking is **not** a symplectic field integrator: it is the averaged (ponderomotive)
model, and `track_a_wiggler.f90:59` collapses it to a *single* step when the element
carries no multipoles. All 98 of `cu_hxr`'s wigglers are `tracking_method =
Bmad_Standard` with an empty multipole table, so `NUM_STEPS` (6, 20 or 100) is consumed
only by PTC and never by tracking.

Verified against Bmad's own `mat6` for all 98 `cu_hxr` wigglers, in the ImpactX basis
and the lab frame: **1.045e-14**. Against Bmad tracking a bunch with `tracking_method =
linear`, covering planar, helical, and tilts of 0, pi/2 and 0.3 rad: **4e-15 to
8.5e-14**.

Three pieces make up the map:

- Transverse: `quad_mat2_calc` per plane with `k1x = kfoc*(kx/kz)**2` and
  `k1y = -kfoc*(kz**2 + kx**2)/kz**2`, `kfoc = 0.5*g_max**2`, `g_max = c*B_MAX/P0C`.
  With `kx = 0` — every `cu_hxr` wiggler — one plane is a **pure drift** and the other
  focuses. Helical instead gets `k1x = k1y = -kfoc`, focusing equally in both.
- `low_energy_z_correction.f90` at `pz = 0`, giving `L*(m/E_tot)**2`.
- The undulation path lengthening, `L*(kz*OSC_AMPLITUDE)**2/4 * (beta**3/gamma**2 + 2)`.

The last two combine to `R56 = (L/gamma**2)*(1 + K**2/2)`, so the undulation term
**dominates**: it is 3.00x, 5.12x and 1.96x the plain drift `R56` for `cu_hxr`'s three
wiggler families (K = 2.000, 2.872, 1.385). Any element carrying only a drift `R56` is
wrong by that factor.

Element `TILT` is passed through as ImpactX `rotation`, not baked into the matrix. This
matters for `cu_hxr`: its 64 HXR undulator segments carry `TILT = pi/2`, so their
focusing is **horizontal**, while the phase shifters and the laser-heater undulator
(`TILT = 0`) focus vertically.

### What a linear map cannot hold

Bmad's only wiggler nonlinearity is an octupole-like kick, `py += k3l*(1+delta)*kz**2 *
y**3/3` (and the same in `px`, helical only). **No ImpactX element can express it**, and
that is not a gap in ImpactX. Both `Multipole` and `ExactMultipole` build their kick as
`dpx - i*dpy = -sum(alpha_m * zeta**m)/m!` with `zeta = x + i*y` (`Multipole.H:302`,
`ExactMultipole.H:343-366`) — an analytic function of `zeta`, hence a vacuum field.
Bmad's term needs `dpx = 0` with `dpy` proportional to `y**3`, which requires
`zeta`-conjugate and is not analytic. With `kx = 0` the undulator field is
x-independent, so the ponderomotive potential goes as `cosh(kz*y)**2` and is not
harmonic: it is an averaged effective Hamiltonian, second order in `B/p`, not a field.
The `Exact` and `Chr` axes do not help — they change the drift between kicks and the
`pt` dependence, not the harmonic constraint — and there is no `ChrMultipole`.

Measured against the real beam divergence at each element (`beta_y` from the lattice,
0.4 um normalised emittance), the dropped octupole is `dpy/sigma_py` of 1.8e-7 for an
HXR segment, 1.2e-9 for a phase shifter and 2.4e-4 for the laser heater. The 6x6 also
freezes `k1/(1+delta)**2` at `delta = 0` over a phase advance of at most 6.7 degrees.

End to end against full `bmad_standard`, at each element's own energy and beam size:
**2.4e-5** for an HXR segment and **1.1e-3** for the laser-heater undulator — roughly
300x and 230x better than the drift they replace.

A wiggler with `B_MAX = 0` or no period becomes an exact drift of its length. A
`fieldmap` or custom-field wiggler is refused rather than silently given the periodic
model.

## What is dropped

Each of these warns with the element name and attribute:

- `ROLL`'s in-plane component, and a bend carrying **both** `REF_TILT` and a transverse
  offset — Bmad displaces the magnet about the bend centre in the tilted frame, which
  ImpactX's element alignment does not reproduce (1.8e-4). Either one alone is exact.
- `x_pitch`/`y_pitch` and `z_offset` — ImpactX elements have transverse offsets and a
  roll, but no pitch and no longitudinal offset.
- `hkick`/`vkick` on an element that is not a kicker.
- Multipole error tables (`A_n`/`B_n`) attached to a magnet — a skew-quad error `a2 = 5`
  on a quadrupole moves a tracked bunch by 4.0e-3, invisible to a transfer-matrix check.
- `DG`, a bend field error — the translated bend uses the design angle.
- Fringe fields other than bend pole faces and quadrupole edges. Bmad's default is `None`
  for quadrupoles, solenoids and sextupoles, and its `rfcavity`/`lcavity` tracking
  ignores `FRINGE_TYPE` entirely, so this stays quiet in practice.
- Bmad's soft quadrupole edge, but only when `FQ1`/`FQ2` are actually set — they default
  to zero, which makes that map a no-op.
- `PHI0_AUTOSCALE`. `PHI0_MULTIPASS` is **not** dropped: `track_a_rfcavity.f90:81` adds
  it into the phase, so the translation does too.

A switched-off **bend** raises rather than warning: `track_a_bend.f90:90-94` zeroes the
field but keeps the curved geometry, so it is not a drift and ImpactX cannot express it.
So does any element with length and no verified equivalent — `taylor`, `sol_quad`,
`match`, `elseparator`, and a `patch` that displaces or re-times the frame. Pass `skip_unsupported=True` to replace them
with markers and warn instead:

```python
lattice = lattice_from_tao(tao, nslice=10, skip_unsupported=True)
```

## An independent third opinion

Every other comparison here is against Bmad. That is the right reference, but it is a
single one: a shared misreading of a Bmad convention would look like agreement. So
`lume_impactx/tests/test_impactz.py` tracks the same bunch through **three** codes —
Bmad, Impact-Z via lume-impact, and ImpactX via this translator — where the two
translators share no implementation.

| case | Bmad ↔ ImpactX | Bmad ↔ Impact-Z |
| --- | --- | --- |
| drift | 1.0e-15 | 2.0e-16 |
| quadrupole | 1.9e-14 | 1.7e-14 |
| **sbend** | **4.3e-11** | 1.2e-7 |
| four FODO cells | 3.2e-14 | 1.0e-13 |

The bend is the interesting row: this translation lands about 2800× closer to Bmad than
the sibling package's does, which is an independent check that the exact-sector-bend and
nonlinear-`DipEdge` work was worth doing.

It needs a toolchain the other tests do not, all of it on conda-forge, and skips when
any of it is missing:

```bash
micromamba create -n lume-impactx-z -c conda-forge python=3.12 impactx bmad pytao \
    impact-z distgen pytest
pip install lume-impact
```

## Species

The species comes from the **lattice** — what Bmad actually tracked, and what every
magnet strength in the lattice is normalised to — not from the bunch's own label.

This matters more than it sounds. A Bmad file that sets no `parameter[particle]`
defaults to **positron**, and `tao.particles()` takes its species from the beam file's
metadata, so an electron beam file in a defaulted lattice yields a bunch labelled
`electron` that Bmad nonetheless tracks against positron-normalised magnets. Trusting
the bunch there tracked **100% away from Bmad, silently**.

The two disagreeing is a setup this translator cannot reproduce — measured 100% off
whichever species is chosen — so it raises rather than picking one. Set
`parameter[particle]` to match the beam, or pass `species=` to override both the
reference and the bunch label if you know they are consistent.

## Reference energy

Bmad holds `p0c` fixed across an `rfcavity` while ImpactX's reference particle really is
accelerated. Every strength Bmad normalises to *its* momentum — quadrupole `k`, multipole
coefficients, steering kicks — is therefore rescaled by `p0c_Bmad / p0c_ImpactX` at each
element, and geometric quantities are not. Without it, a quadrupole after a 5 MV cavity
at `phi0 = 0.25` is referenced to the wrong rigidity and tracks 4.9e-2 away from Bmad;
with it, 8.5e-8. The rescale is announced with a warning naming the element and factor,
and is exactly 1 in a lattice with no acceleration.

## Controlling the numerics

`nslice` sets the number of steps per thick element (default 8). Bmad's own `num_steps`
and `ds_step` are deliberately *not* read across, because they drive a different
integrator:

```python
sim = ImpactXSimulator.from_tao(tao, nslice=20)
```

Raise it where the translation is a splitting rather than a single element — a
combined-function bend, an `rfcavity` or an `lcavity` — since those converge with
`nslice`.

Bends default to `fringe_type = Basic_Bend`, which maps exactly, `FINT`/`HGAP` included.
To compare the body alone, turn the fringe off in Bmad and the translator will emit no
edges:

```python
tao.cmd("set ele * fringe_type = none")
```

## Modelling part of a lattice

`track_start` and `track_end` translate only a range, starting from the bunch Tao has
there:

```python
sim = ImpactXSimulator.from_tao(tao, track_start="MID", track_end="SCR")
```

`track_start` names the element the beam is taken *at*, so translation begins with the
element after it — the bunch has already been through it. `track_end` is inclusive, as
in Tao. One argument moves both the lattice and the beam, so they cannot disagree, and
the capture points are restricted to the same range.

Verified against Tao: starting at `MID` reaches the same `END` as tracking the whole
lattice, to 1e-14.

ImpactX itself has no notion of starting partway through — it tracks whatever is in
`sim.lattice` — so this is a slice of the translated element list, not an ImpactX
feature.

## Branches

`branch` selects which lattice branch to translate, and reaches the beam, the reference
mass and the lattice together:

```python
sim = ImpactXSimulator.from_tao(tao, branch=1)
```

Elements are enumerated with Tao's `-track_only -index_order` flags. `-no_slaves`, the
obvious-looking choice, keeps *lords* while bare indices address the tracking branch —
mixing the two silently truncates a superposed lattice or translates a super-lord twice.
