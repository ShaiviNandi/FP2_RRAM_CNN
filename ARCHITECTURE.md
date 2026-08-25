# The architecture, defined

The single reference for what is being proposed. Any comparison run against a
configuration other than the one defined here is measuring something else.

---

## One paragraph

A fully weight-stationary analog compute-in-memory accelerator in which every
CNN weight is stored as a **2-bit block-floating-point (FP2-E1M0) value in one
differential pair of three-state ReRAM cells**. Multiply-accumulate happens by
Ohm's law and Kirchhoff's current law in the array. Each bitline is sensed
passively through a shared resistor, digitised by a 6-bit SAR ADC, and scaled
by one compile-time constant per column that removes the sense-resistor loading
error exactly.

---

## 1. The number format

| Property | Value |
|---|---|
| Format | FP2-E1M0 (1 exponent bit, 0 mantissa, plus sign) |
| Levels | $\{-1, -0.5, 0, +0.5, +1\}$ — five signed |
| Distinct magnitudes | **three** (1, 0.5, 0) |
| Block size $B$ | **32** (fixed by the format specification) |
| Shared scale | one 8-bit E8M0 exponent per block |
| Storage cost | 2.25 bits/weight |
| Training | QAT with straight-through estimator, FP32 master weights |
| First/last layer | left in full precision (standard convention) |

**Why five levels need only three states:** the differential pair supplies the
sign. A positive weight puts the magnitude on the $+$ bitline and HRS on the
$-$; a negative weight reverses it. Three physical states express five signed
values.

---

## 2. The cell

| Property | Value |
|---|---|
| Structure | **2T2R** — two 1T1R cells, one per polarity |
| Cells per weight | **2** (one differential pair) |
| States per cell | **3** |
| $R_{\mathrm{LRS}}$ | 542.8 Ω |
| $R_{\mathrm{MID}}$ | 1099.8 Ω |
| $R_{\mathrm{HRS}}$ | 218 587.2 Ω |
| On/off ratio | 403 |

Mapping:

| Weight | $G^{p}$ cell | $G^{n}$ cell |
|---|---|---|
| $+1.0$ | LRS | HRS |
| $+0.5$ | MID | HRS |
| $0.0$ | HRS | HRS |
| $-0.5$ | HRS | MID |
| $-1.0$ | HRS | LRS |

**There is no multi-cell weight decomposition and no shift-add tree.** One
weight, one differential pair, one read. This is the property that most
distinguishes the design from a conventional multi-bit CIM mapping, and it is
the one that was silently discarded in the NeuroSim runs.

---

## 3. The tile

| Property | Value |
|---|---|
| Tile height $M$ | **128** (proposed) |
| Tile width $K$ | 16 columns |
| Constraint | $\mathbf{B = M}$ — block size *is* tile height |
| Mapping | fully weight-stationary, all weights resident |
| Network | ResNet-18, 11.16 M weights, 5464 tiles |

**Why $B = M$:** a column produces one current, which can carry only one scale
factor. A tile spanning two blocks would need two scales applied to a sum
already formed by Kirchhoff's current law. The format fixes $B=32$; ADC
amortisation wants $M \geq 128$. That collision is the subject of the work.

---

## 4. Readout

| Property | Value |
|---|---|
| Sensing | **passive shared sense resistor**, not a transimpedance amplifier |
| $R_{\mathrm{sense}}$ | 20 Ω |
| $V_{\mathrm{read}}$ | 0.1 V |
| Bitline settling | ~1 ps |
| ADC | 6-bit SAR |
| **Converters per column** | **2 — one per bitline** |
| Column mux | 8:1 |
| Input drive | 4-bit parallel DAC per row (not bit-serial) |

**Two converters per column is required, not optional.** Each bitline must be
scaled by its own constant *before* the differential subtraction. A single
differential sense amplifier has already subtracted them, which makes the
correction impossible. This is a real cost of the method and the model charges
for it.

---

## 5. The correction

Compute once, at compile time, from the programmed conductances:

$$c_j = 1 + R_{\mathrm{sense}} \sum_i G_{ij}$$

Apply $c_j$ to bitline $j$ **before** subtracting the two polarities. The
algebra collapses to $(G_p - G_n)^{\!\top}V$, the ideal product.

| Property | Value |
|---|---|
| Constants | one per bitline |
| Computed from | nominal programmed conductances — **no read-back** |
| Calibration data | **none required** |
| Hardware cost | folds into the block-scale multiply already in the datapath |
| Residual | machine precision (0.0002% in the mesh self-test) |
| Refresh | periodic, ~10 logarithmically spaced over product life (drift) |

---

## 6. Mapping onto NeuroSim

The configuration that represents this architecture, against NeuroSim's
defaults.

| NeuroSim parameter | Default | **Must be** | Why |
|---|---|---|---|
| `memcelltype` | 1 (SRAM) | **2** (analog eNVM) | otherwise the device parameters are inert |
| `wl_weight` | 8 | **2** | 8-bit weights add 4 cells/weight and a shift-add tree that FP2 does not have |
| `cellBit` | 2 | 2 | with `wl_weight=2` this gives exactly one cell per weight |
| `ADCprecision` | — | 6 | measured requirement, flat in $B$ |
| `subArray` | — | 128 | the proposed tile height |
| `resistanceOn` | 6e3 | 542.8 | the device |
| `resistanceOff` | — | 218587.2 | the device |
| `onoffratio` (wrapper) | 10 | 403 | separate from the C++ parameter; both must be set |
| `widthInFeatureSize1T1R` | 12 | ≥128 | a 542.8 Ω LRS needs a ~127F select transistor |

**Two traps, both hit already.** `resistanceOn`/`resistanceOff` are ignored
unless `memcelltype = 2`; and the wrapper's `--onoffratio` is a *different*
parameter from the C++ `resistanceOn`/`resistanceOff` pair, so setting one
leaves the other stale.

---

## 7. The device parameter question

$R_{\mathrm{LRS}} = 542.8\,\Omega$ sits at an awkward point and the tension
should be stated rather than hidden:

- **Low $R_{\mathrm{LRS}}$** → large read current → wide select transistor →
  large cell; **and** large $R_s G_{\mathrm{col}}$ → large loading error, which
  is exactly what the calibration removes.
- **High $R_{\mathrm{LRS}}$** → small transistor, compact cell, small loading
  error → but less signal current for the ADC to resolve.

The calibration corrects the *measurement*, not the *current*. The current
still flows, still dissipates power, and still sets the transistor width.
Algebra does not undo Ohm's law.

That is why the contribution is best framed as removing a constraint: without
the correction, a designer is pushed toward high-resistance cells for accuracy
reasons; with it, the resistance can be chosen on area and signal grounds
alone.
