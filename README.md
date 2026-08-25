# FP2 on a 2T2R ReRAM Crossbar

Circuit-exact simulation of a 2-bit block-floating-point (FP2-E1M0) CNN
accelerator on a 2T2R resistive-RAM crossbar, with per-column loading-gain
calibration. Every convolution is executed through the resistive-divider
equations; no additive-noise surrogate is used anywhere.

CIFAR-10 Top-1, ResNet-18, full 10,000-image test set:

| B = M | FP2 digital | crossbar, raw | crossbar, calibrated |
|------:|------------:|--------------:|---------------------:|
| 32    | 92.42%      | 77.66%        | **92.42%** |
| 64    | 92.47%      | 57.54%        | **92.47%** |
| 128   | 92.81%      | 11.60%        | **92.81%** |
| 256   | 92.43%      | 10.00%        | **92.43%** |

---

## The problem

A block-floating-point format shares one scale across `B` weights. A crossbar
column produces one current, which can carry one scale. **The block size and
the tile height are therefore the same physical parameter.** FP2 is used with
`B = 32`; amortising the ADC wants `M ≥ 128`.

Built naively, the collision is fatal: accuracy falls to chance by `M = 128`.

## The cause

A shared sense resistor holds the bitline slightly above ground, reducing the
drive across every cell on it:

```
I_actual = I_ideal / (1 + R_s · G_col)
```

The denominator contains no activation term. It is a **deterministic gain set
by the programmed conductances**, not noise — which is why an additive-noise
model cannot see it, and why it grows with `M` instead of averaging away.

## The fix

Per bitline `j`, one compile-time constant `α_j = 1 + R_s · Σ G_ij`. Applied to
each branch **before** the differential subtraction, the algebra collapses back
to the exact product:

```
α_p·I_p − α_n·I_n = (G_p − G_n)ᵀ V
```

Exact to machine precision. One multiply per column, folded into the
block-scale multiply already in the datapath. No calibration data, no
read-back, no retraining.

---

## Architecture

| | |
|---|---|
| Format | FP2-E1M0, levels `{0, ±0.5, ±1}`, E8M0 scale shared over `B = 32` |
| Cell | 2T2R differential pair; `w ∝ G_p − G_n` |
| States | `R_LRS` 542.8 Ω, `R_MID` 1099.8 Ω, `R_HRS` 218.6 kΩ (contrast 403) |
| Tile | `M` rows × `K` columns, `M = B`, fully weight-stationary |
| Readout | Passive shared sense resistor, `R_s = 20 Ω`, `V_read = 0.1 V` |
| Converters | 6-bit SAR, **two per column** — both bitlines sensed separately |
| Correction | One constant per bitline, applied before subtraction |

Two converters per column is a requirement of the method, not an
implementation choice: the two bitlines of a pair carry different `G_col` and
must be corrected independently.

`ARCHITECTURE.md` gives the full definition, including what is deliberately
out of scope.

## Methodology

Four levels, each validated against the one above:

| Level | Checked against | Worst disagreement |
|---|---|---|
| ngspice DC solve | — (reference) | — |
| Closed-form nodal model | ngspice, all 87,168 tiles | 3.41 × 10⁻⁷ % |
| Vectorised GPU solver | nodal model | < 10⁻¹² relative |
| End-to-end inference | vectorised solver | same code path |

Validation is exhaustive rather than sampled: every tile of every layer,
357 M MAC reads. The closed form needs no iteration, so it vectorises into one
matrix multiply per layer — the difference between minutes and weeks for a
full-test-set sweep.

Device variability, conductance drift, ADC resolution and wire parasitics are
swept rather than fixed at a single point. Every model constant is a named,
overridable value at the top of the module that uses it, with its source in
the comment beside it.

---

## Layout

```
*.py                        simulation, hardware model, figures
ARCHITECTURE.md             the design, defined precisely
results/{csv,figures,logs}  generated outputs
rtl/                        Verilog periphery and testbenches
spice/                      ngspice decks and the Verilog-A device model
run_all.sh                  full pipeline, staged
sync_results.sh             copy generated artefacts into results/
```

| File | Role |
|---|---|
| `analog_eval.py` | Inference through the crossbar equations; ADC, variability and drift models |
| `crossbar_array_test.py` | Scalar reference solver and ngspice netlist generation |
| `ngspice_full_sweep.py` | Exhaustive per-tile validation, parallel and resumable |
| `hw_model.py` | Area, energy, delay; every assumption a named constant |
| `qat_finetune_fp2.py` | FP2 quantization-aware training, straight-through estimator |
| `format_sweep.py` | Read margin and storage cost across weight formats |
| `wordline_ir.py` | Full 2-D resistive mesh, for the wire-parasitic bounds |
| `vonneumann_baseline.py` | Digital fetch-and-multiply baseline |
| `bn_baseline.py` | BatchNorm recalibration baseline |
| `neurosim_compare.py` | Chip-level cross-check against NeuroSim |
| `make_figures.py`, `figures_sweeps.py` | Figures and LaTeX tables from the CSVs |

---

## Requirements

Python 3.10+, PyTorch 2.0+, NumPy, Matplotlib. `ngspice` is needed only to
re-run the SPICE validation. A CUDA GPU is optional; the accuracy sweeps run on
CPU at roughly 20× the wall time.

```bash
pip install -r requirements.txt
```

---

## Scope

Results are ResNet-18 on CIFAR-10 and CIFAR-100. Device parameters come from
SPICE calibration of a Verilog-A compact model, not fabricated silicon; the
structure of the argument does not depend on the particular values, but every
downstream number does.

Three bounds are measured and reported rather than omitted:

- **Wire parasitics.** Bitline metal leaves a 49.1% residual after correction.
  A per-column scalar cannot capture a distributed drop, so the claim is
  restricted to the shared-sense-resistor term.
- **Drift.** A constant computed once at manufacture decays; periodic
  recomputation from the drifted array holds accuracy flat.
- **Area denominators.** Density figures are array-level. At 2-bit weights the
  weight array is under 1% of a chip-level floorplan.

## License

MIT. See `LICENSE`.
