# Examples

Each script is self-checking — it asserts the physics it is demonstrating — and writes a
PNG next to itself. Run from anywhere:

```bash
python examples/space_charge_expansion.py
python examples/csr_chicane.py
python examples/resistive_wall_wake.py
```

| example | shows | check it makes |
|---|---|---|
| `space_charge_expansion.py` | 3D space charge on a coasting bunch | the bunch expands to 2× its initial size, the analytically known result |
| `csr_chicane.py` | CSR through a four-bend chicane | CSR removes energy and grows the spread; also scans `sim:csr_bins` to show the resolution dependence |
| `resistive_wall_wake.py` | resistive-wall wakefield in a copper pipe | the bunch is decelerated, transverse dynamics untouched |

The first two mirror Bmad's `tao_examples/space_charge` and `tao_examples/csr_beam_tracking`,
so the same physics can be compared across the two codes. The third has no Bmad
counterpart — ImpactX has no wakefield element, so `lume_impactx.wakes` assembles one
from its convolution primitives.

`space_charge_expansion.py` needs a few seconds; the other two run in about a second at
their default particle counts.
