# Quickstart

## Build a simulator

`ImpactXSimulator` holds the specification. The lattice is an ordinary Python list of
ImpactX elements, and it is the canonical, mutable thing that variables write to.

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
```

The simulator tracks once during construction, so results are available immediately.

## Drive it as a LUME model

```python
model = LUMEImpactXModel.from_simulator(simulator)

model.get("moment_final:sigma_x")     # 7.570128803674508e-05
model.set({"ele:quad1:k": 1.2})       # writes, then re-tracks
model.get("moment_final:sigma_x")     # 6.456590780202415e-05
model.reset()                         # back to the construction-time state, exactly
```

`set()` re-runs the whole simulation. With space charge on that can take minutes, so to
change several things at once, defer tracking:

```python
model = LUMEImpactXModel.from_simulator(simulator, dummy_run=True)
model.set({"ele:quad1:k": 1.2, "ele:quad2:k": -1.2})
simulator.track()                     # one run instead of two
```

## Plot

```python
fig = model.plot(y=("sigma_x", "sigma_y"), y2=("mean_pt",), include_labels=True)
```

## Archive

```python
from lume_impactx import archive, load_archive

archive(simulator, "fodo.h5")
restored = load_archive("fodo.h5")        # no ImpactX run needed
restored.results["moments"]["sigma_x"]
```
