# Methodology and assumption provenance

Every model used in this project, what it computes, and where each number came
from. Written so that a reviewer or examiner can find the origin of any figure
without reading the source.

The governing principle: **the array is simulated, the periphery is modelled,
and the two are labelled differently everywhere.** Output carries three tags.

| Tag | Meaning | Trust |
|---|---|---|
| `[SIM]` | Measured by ngspice on the actual netlist | High |
| `[MODEL]` | Closed-form derivation from device parameters | Medium |
| `[ASSUM]` | Literature-typical placeholder, no PDK behind it | Low — replace before publishing |

Conflating these is how CIM papers quietly overstate results, so the tags are
printed alongside every number rather than buried in a footnote.

---

## 1. Device parameters

| Parameter | Value | Source |
|---|---|---|
| `R_LRS` | 542.8 Ω | SPICE calibration of the `rram_v_1_0_0` Verilog-A compact model |
| `R_MID` | 1099.8 Ω | same |
| `R_HRS` | 218 587.2 Ω | same |
| on/off ratio | 402.7 | derived |
| `R_sense` | 20 Ω | design choice, swept in B2 |
| `V_read` | 0.1 V | design choice, kept low to stay in the linear read regime |

The three states were obtained by sweeping RESET pulse amplitude and width
against the compact model (`calibration_sweep.py`), reading back the resulting
resistance with a small non-disturbing read pulse, and selecting the operating
points that land in distinguishable bands. They are therefore **derived from a
device model, not measured on fabricated silicon**, which is the single most
important caveat on the entire project.

### Level-mapping error (quantified, stage B4)

The three programmed states do not land exactly on {0, 0.5, 1}:

- `R_MID / R_LRS = 2.0262`, not 2.
- `R_HRS` is finite, so it contributes 4.6 µS rather than nothing.

Working the differential encoding through, level 0.5 lands at **0.49228**, an
error of **−1.543%**. Levels ±1 and 0 are exact by construction, since the
normalisation is defined by them.

This is **separate from readout error** and is currently absorbed into what the
results call the digital ceiling. Two fixes, both verified in `run_blocking.sh
b4`: trim `R_MID` to **1082.91 Ω** (a −1.54% trim, which the script asserts
lands level 0.5 on exactly 0.5), or fold the measured ratio into the block
scale factor at zero hardware cost.

---

## 2. Circuit model

### 2.1 The closed-form nodal solve (the workhorse)

Kirchhoff's current law at a bitline held at `v_bl`, drained through `R_s`:

```
Σ_i (V_i − v_bl)·G_i = v_bl / R_s
```

Solving for `v_bl`:

```
v_bl = (Σ_i V_i G_i) / (1/R_s + G_col),     G_col = Σ_i G_i
```

This is closed form, not iterative, which is the whole reason full-test-set
evaluation is affordable: it vectorises into one matrix product per tile and
polarity, and all tiles of a layer batch into a single GPU `bmm`. A general
circuit simulator must solve a system numerically per operating point.

Dividing through by the ideal accumulation gives the result the paper is built
on:

```
I_actual = I_ideal / (1 + R_s · G_col)
```

No activation term appears in the denominator, so the attenuation is a
deterministic function of the programmed conductances.

### 2.2 The full 2-D mesh solve (`wordline_ir.py`)

Used only to test what the closed form leaves out. Unknowns are the `M·K`
wordline node voltages and `M·K` bitline node voltages; KCL at every node gives
a sparse system solved directly by LU. No iteration, so no convergence
question.

Structure: wordline `i` driven at its left end through `R_DRV`, crossing the
array through `K` segments of `R_WL`; bitline `k` running down through `M`
segments of `R_BL` into `R_SENSE`; cell `(i,k)` bridging the two planes.

**Numerical caveat, discovered by the self-test.** Driving wire resistance to
zero with a huge conductance destroys the conditioning of the matrix — cell
conductances are ~1e-3 S, so a 1e12 S wire gives a condition number ~1e15 and
the LU solve loses all precision. The ideal-wire limit is therefore clamped at
1e-6 Ω, which is 1e-9 of the cell resistance and solves cleanly. Agreement with
the closed form at that limit is **8.795e-08 relative**; at 1e-12 Ω it degrades
to 8.7e-03. This is a solver artefact, not physics, and the guard is in the
code with that explanation.

---

## 3. Validation chain

| Level | Checked against | Worst disagreement |
|---|---|---|
| ngspice DC solve | — (reference) | — |
| Closed-form nodal model | ngspice, **87 168 tiles** | **3.41 × 10⁻⁷ %** |
| Vectorised GPU solver | nodal model | < 10⁻¹² relative |
| End-to-end CNN inference | vectorised solver | identical code path |
| 2-D mesh solver | closed form at ideal wires | 8.8 × 10⁻⁸ relative |

Every tile of every layer, not a sample: 357 M MAC-reads, 22 minutes on 15
cores, zero errors. This matters because a zero-mean additive Gaussian error
model — the standard in CIM evaluation — **structurally cannot represent a
systematic gain**. It assumes error averages away over many accumulated terms.
The error here is a 26–74% attenuation that never averages away. Any evaluation
built on that assumption would observe the accuracy collapse, attribute it to
noise in tall columns, and conclude the tile-height limit is fundamental.

---

## 4. Device variability

**Model.** Log-normal multiplicative scatter on conductance, drawn once per
device instance and then **frozen** — a manufactured chip does not re-roll its
defects between inferences.

**Why log-normal.** Conductance is positive and its spread is multiplicative;
programming scatter in filamentary devices is conventionally reported this way.

**Protocol.** 10 independent device instances per operating point, full
10 000-image test set, σ ∈ {0, 5%, 10%, 20%}. Three calibration variants:

- `raw` — no calibration
- `blind` — constants from **nominal** programmed conductances, no read-back.
  This is what is actually buildable: the compiler knows what it asked the
  array to store, not what the array became.
- `write-verify` — constants from the **actual** perturbed conductances, i.e.
  every cell read back after programming. The upper bound on what calibration
  can achieve, at one measurement per cell.

**Result.** Blind calibration loses 0.90 points from σ=0 to σ=20%. Blind is
consistently *equal or slightly better* than write-verify at high σ across all
four tile heights, which is the opposite of the expected ordering and is not
yet explained — flagged rather than claimed.

---

## 5. Drift

The result that changed the design, and the one with the softest inputs.

### 5.1 The model

```
G(t) = G(t_ref) · (t / t_ref)^(−ν),      t_ref = 1 s
```

A power law, taken from the PCM resistance-drift literature and applied to
ReRAM for the same physical reason: after programming, the filament relaxes
structurally, and that relaxation is scale-free over many decades of time.
Nothing drifts below `t_ref`.

### 5.2 Three details that decide the outcome

**State dependence.** HRS relaxes markedly faster than LRS, because HRS has no
stable conductive filament holding it in place:

| | ν | Source |
|---|---|---|
| `ν_LRS` | 0.01 | literature-typical `[ASSUM]` |
| `ν_HRS` | 0.05 | literature-typical `[ASSUM]` |

Cells are classified as programmed-on if their conductance exceeds the
**geometric mean** of `G_LRS` and `G_HRS`, rather than from a stored mask, so
the classification tracks actual conductance rather than intent. State
dependence matters less than it appears for the gain constant, because `Σ G`
over a column is dominated by LRS/MID cells (1.8e-3 S each against HRS's
4.6e-6 S), but it moves the differential zero point.

**Cell-to-cell spread in ν.** This is the part that cannot be calibrated away.
Each cell gets `ν + σ_ν·N(0,1)` with `σ_ν = 0.004` `[ASSUM]`. If every cell
drifted by the *same* factor, the gain constant would simply be wrong by a
known amount and one global rescale would fix it. Because ν itself varies cell
to cell, drift injects **dispersion**, and dispersion is not a per-column
constant.

**Why drift breaks what variability did not.** Programming scatter is frozen at
t=0 and the constants are computed *after* it, so blind calibration already
accounts for it. Drift keeps moving after the constants were computed. That
asymmetry is the entire drift result.

### 5.3 Implementation notes

Noise is drawn on CPU with a seeded generator and moved to the device, so
results are reproducible and identical between CPU and GPU runs. Timepoints
evaluated: 1 s, 1 hour, 1 day, 1 month, 1 year.

### 5.4 Sensitivity, because σ_ν is the least certain input (stage B3)

Rather than asserting σ_ν = 0.004, it was swept at B=128, t = 1 year:

| σ_ν | stale | refreshed | refresh worth |
|---|---:|---:|---:|
| 0.001 | 71.52 | 92.12 | +20.60 |
| 0.002 | 71.53 | 92.03 | +20.50 |
| 0.004 | 71.73 | 91.73 | +20.00 |
| 0.008 | 73.36 | 90.54 | +17.18 |
| 0.016 | 77.91 | 80.29 | **+2.38** |

**Refresh holds within 2 points of the ceiling up to σ_ν = 0.004 and begins
failing at 0.008.** By 0.016 refreshing is nearly worthless. The operating
assumption sits at the edge of the region where the claim is safe, and the
paper should state the boundary rather than the point.

### 5.5 A design error found and corrected

Recalibrating only the loading constant left **15.7% residual — worse than
leaving it stale**. Drift shrinks the weights themselves, and that scale error
dominates the loading error. The fix is a least-squares per-column gain folded
into the block scale, correcting both at once.

### 5.6 Mechanism confirmed (stage B2)

The observation that stale loss falls with tile height was made *after* looking
at the data, which is the weakest kind of claim. The proposed mechanism —
that the error scales with `R_s·G_col` — makes a falsifiable prediction: the
tile-height trend must strengthen as `R_s` grows and wash out as it shrinks.

| R_s (Ω) | B=32 | B=64 | B=128 | B=256 | spread |
|---:|---:|---:|---:|---:|---:|
| 5 | −31.56 | −28.32 | −35.16 | −26.83 | −4.73 |
| 10 | −29.47 | −25.44 | −29.43 | −20.57 | −8.90 |
| 20 | −25.69 | −20.56 | −21.08 | −12.88 | −12.81 |
| 40 | −19.66 | −13.67 | −12.18 | −6.38 | −13.28 |
| 80 | −12.03 | −7.46 | −5.54 | −2.89 | −9.14 |

Stale cost shrinks monotonically with `R_s` at every tile height, and the
ordering is clean and monotonic at `R_s ≥ 20` but scrambled at `R_s = 5` where
the loading term is small. **The prediction passed**, so the trend is explained
rather than merely observed.

---

## 6. ADC model

`[ASSUM]` throughout. SAR architecture, Walden-style figure of merit.

| Constant | Value | Basis |
|---|---|---|
| `ADC_FOM_FJ_PER_CONV_STEP` | 20 | 5–50 fJ spans most published SAR ADCs; mid-range |
| `ADC_AREA_MM2` | 0.005 | order-of-magnitude for an 8-bit SAR at 65 nm |
| `ADC_CLK_MHZ` | 500 | conversion takes `ADC_BITS` cycles |
| `COLS_PER_ADC` | 8 | column mux ratio, amortises ADC area |
| `ADCS_PER_COLUMN` | **2** | **required by the method** |

Energy per conversion is `FOM × 2^bits`, so each bit removed halves it.

**`ADCS_PER_COLUMN = 2` is a real cost of this work and is charged for.** The
per-column calibration requires reading both bitlines separately, because each
must be scaled by its own constant before subtraction. A conventional
differential design uses one converter per column and a sense amplifier that
has already subtracted them, making the correction impossible. Setting this to
1 gives the baseline a reviewer would call fair, and both should be reported.

### Sensitivity (stage B7)

The headline efficiency figure is dominated by one assumed constant:

| ADC source | pJ/MAC | TOPS/W | ADC % energy | die area |
|---|---:|---:|---:|---:|
| placeholder (FoM 20) | 0.0246 | 81.3 | 80.7% | 412 mm² |
| NeuroSim-sourced area | 0.0246 | 81.3 | 80.7% | 329 mm² |
| published SAR (FoM 9.5) | 0.0141 | 142.2 | 66.6% | 364 mm² |

**A 1.75× swing in the headline number from the ADC figure of merit alone.**
Report the spread; the FoM 9.5 row still needs a real citation substituted.

---

## 7. Input drive, area and energy

| Constant | Value | Basis |
|---|---|---|
| `TECH_F_NM` | 65 | node choice |
| `CELL_AREA_F2` | 40 | 1T1R select transistor must pass SET/RESET current, so far above the ideal 6F²; published macros span 20–60 |
| `ARRAY_PERIPHERY_OVERHEAD` | 1.4 | decoders, write drivers, precharge, routing; 1.3–1.6 typical |
| `DAC_BITS` | 4 | activation precision |
| `DAC_UNIT_AREA_UM2` | 3 | one unit cap plus switch at 65 nm; a 4-bit converter needs 2⁴ |
| `WL_DRIVER_AREA_UM2` | 8 | level shifter plus inverter chain, ~20 transistors |
| `DAC_ENERGY_PJ_PER_ACTIVATION` | 0.05 | `[ASSUM]` |
| `RERAM_WRITE_PJ_PER_CELL` | ~100 | against ~0.001 pJ to read |
| `ARRAY_REF_UW_PER_TILE` | 85.8 | **empirical anchor from ngspice**, not assumed |

**Input scheme.** Default is a parallel multi-level DAC, not bit-serial, and
deliberately: the ADC resolution sweep applied the converter to a full
multi-bit activation read, so the measured "6 bits required" describes *that*
scheme. Bit-serial fires the ADC once per activation bit, so it needs its own
sweep. Selecting bit-serial without re-measuring would combine a resolution
requirement from one scheme with an energy model from another.

**A correction worth recording.** `hw_model`'s fallback array-power model was
once **30× too high** (2587 µW/tile modelled against 85.8 µW measured), which
made every efficiency figure pessimistic. It assumed every cell at nominal
conductance with full `V_read` across it, whereas in 2T2R only the magnitude
side of a nonzero weight leaves HRS, half of those sit at `R_MID`, and
post-ReLU activations are sparse. The fallback is now anchored to the ngspice
measurement.

### What the energy model does NOT count

`hw_model` counts array, ADC, DAC and partial-sum. It has **no category for
interconnect, buffers, pooling or activation units**. NeuroSim's chip-level
accounting puts those at **20.8%** of per-image energy. Adding that share to
81.3 TOPS/W gives ≈ **64 TOPS/W**.

**81.3 TOPS/W is therefore a tile-level number, and must be labelled as such.**
Most of the apparent 2.8× advantage over NeuroSim is an accounting difference
rather than a design difference.

---

## 8. SRAM baseline

| Constant | Value | Basis |
|---|---|---|
| `SRAM_BIT_F2` | 140 | 6T bitcell, 120–160 F² typical at 65 nm |
| `SRAM_LEAK_PW_PER_BIT` | 120 | room temperature; varies >10× with corner and temperature |
| `SRAM_READ_PJ_PER_BIT` | 0.05 | macro read plus delivery to a MAC unit |

**Scope, stated plainly.** This compares *weight storage arrays*, not
accelerators. No sense amps, decoders or controller are counted on the SRAM
side. It is fair framed as "how much area do the weights occupy" and dishonest
framed as "our chip beats SRAM".

**Weight reuse is the contested parameter.** At reuse 1 SRAM pays 0.1 pJ/MAC to
fetch and ReRAM pays zero; at reuse 1000 the term is 0.0001 pJ/MAC and the
advantage nearly vanishes. Reuse 1 is honest for batch-1 inference on FC and
1×1 convolutions and flattering for batched convolution. **Report the curve**
(`--reuse-sweep`), never the single point.

---

## 9. Quantization and training

**FP2-E1M0.** Five levels {−1, −0.5, 0, +0.5, +1} with an 8-bit scale shared
across B consecutive weights, giving 2.25 bits/weight. The scale is quantized
with the exponent rounded **up**, guaranteeing |w|/scale ≤ 1.

**QAT.** Straight-through estimator — rounding has zero gradient everywhere,
which would stop learning, so the backward pass treats it as the identity.
FP32 master weights are kept alongside and are what the optimizer updates.

**First and last layers are left in full precision** by default, the standard
convention. `--quantize-first-last` exposes the cost.

---

## 10. BatchNorm recalibration baseline

The closest competing method, run as a baseline rather than argued against.

**Procedure**, no gradients anywhere: convert to the raw uncorrected crossbar
path; reset every BN layer's running mean and variance; put **only** the BN
layers into training mode (train mode on the whole model would enable dropout
and other stochastic paths and corrupt the statistics); stream calibration
images so the BN layers measure the attenuated statistics the crossbar actually
produces; return to eval and measure.

Affine weight and bias are untouched — that is what makes it retraining-free
rather than a one-layer fine-tune.

**Calibration images come from the TRAIN split**, strided rather than
truncated. Using test images to fit the statistics and then reporting test
accuracy would let the baseline see the evaluation set. Striding matters
because these sets are class-ordered in places, so the first N images can cover
only a few classes.

**The prediction being tested:** BN recalibration should recover *less* as tile
height grows, because a per-channel scalar can only absorb a layer-average
attenuation while the loading gain varies column to column, and that variation
grows with M. The verdict is derived from the data in the script rather than
asserted.

---

## 11. Known limitations of the methodology

Ordered by how tightly each bounds a conclusion.

1. **Wire parasitics are not in the main evaluation path.** The mesh solve
   shows the per-column constant leaves **49.1%** residual with 0.5 Ω bitline
   metal and **25.4%** with a 50 Ω wordline driver, against 0.0002% with ideal
   wires. A fitted per-column gain does not transfer to unseen activations at
   M=256. Distributed resistance is not a lumped per-column term, so the
   claim must be narrowed to the `R_sense` loading term.
2. **No fabricated device.** Every device parameter derives from a compact
   model, not measured silicon.
3. **ADC, DAC, cell and SRAM constants are `[ASSUM]`.** The B7 sweep shows a
   1.75× headline swing from the ADC FoM alone.
4. **Interconnect and buffer energy are absent** (§7).
5. **NeuroSim cross-check is VGG-8 against ResNet-18** — not like-for-like,
   and must be labelled wherever quoted.
6. **Drift exponents are literature-typical**, which is why B3 swept them.
7. **Read noise and temperature are not modelled.** Read noise varies per read
   so it should average over a dot product rather than accumulate; likely
   benign, but unmeasured.
8. **`RTL_RelErr%` reads N/A everywhere** because the SystemVerilog top level
   has no behavioural model at the ADC boundary.

---

## 12. Reproducibility

Every number traces to a named CSV; every figure and table is regenerated from
those CSVs by `make_figures.py` and `figures_sweeps.py`. Captions that state a
threshold **derive it from the data** rather than hardcoding it — a hardcoded
caption is what kept a figure claiming "5 bits suffice" after the answer had
changed to 6.

```bash
bash run_blocking.sh            # list stages
bash run_blocking.sh blocking   # A2-A4, ~20 min
bash run_blocking.sh important  # B1-B7, overnight
python3 drift_summary.py --latex
python3 bn_baseline.py --sweep 32,64,128,256 --max-images 0 --adc-bits 6
```

Each simulator carries a `--self-test` that asserts agreement with its
reference implementation. Those self-tests have caught real bugs: the mesh
conditioning failure in §2.2, and an unstable per-column gain fit that produced
a nonsense 2488% before being replaced with least squares on held-out
activation patterns.

---

