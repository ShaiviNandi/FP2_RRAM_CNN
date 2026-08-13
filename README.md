# FP2 on a 2T2R ReRAM Crossbar

Circuit-exact simulation of a 2-bit block-floating-point (FP2-E1M0) CNN
accelerator built on a 2T2R resistive-RAM crossbar, with per-column
loading-gain calibration.

Every convolution is executed through the actual resistive-divider equations.
No additive-Gaussian noise surrogate is used anywhere.

---

## The result in one table

CIFAR-10 Top-1, ResNet-18, full 10,000-image test set.

| B = M | FP2 digital | crossbar, raw | crossbar, calibrated |
|------:|------------:|--------------:|---------------------:|
| 32    | 92.42%      | 77.66%        | **92.42%**           |
| 64    | 92.47%      | 57.54%        | **92.47%**           |
| 128   | 92.81%      | 11.60%        | **92.81%**           |
| 256   | 92.43%      | 10.00%        | **92.43%**           |

Uncalibrated readout reaches chance (10 classes) by B=128. Calibration
restores the digital ceiling exactly at every tile height.

---

## The problem

FP2-E1M0 stores each weight as one of {-1, -0.5, 0, +0.5, +1}, with B=32
consecutive weights sharing an 8-bit scale factor.

A crossbar column produces one current, which can carry only one scale factor.
A tile spanning two blocks would require two scales applied to a sum already
formed by Kirchhoff's current law. Block size and tile height are therefore the
same physical parameter:

    B = M

The format specifies 32. ADC amortisation wants 128 or more. That collision is
what this repository characterises.

## The cause

A shared sense resistor R_s lifts the bitline off ground, reducing the voltage
across every cell in the column. Nodal analysis at the bitline gives:

    v_bl = (sum_i V_i G_i) / (1/R_s + G_col),    G_col = sum_i G_i

    I_actual = I_ideal / (1 + R_s * G_col)

No activation term appears in the denominator. The error is a deterministic
function of the programmed conductances, not noise. Additive zero-mean models
cannot represent it, which is why the failure mode is invisible to the standard
evaluation methodology.

Fraction of ideal current reaching the ADC:

| M | 32 | 128 | 256 |
|---|---:|----:|----:|
| retained | 73.9% | 41.5% | 26.0% |

## The fix

One constant per column, computed at compile time:

    c_j = 1 + R_s * sum_i G_ij

2T2R reads both bitlines separately, so each is scaled by its own constant
*before* the differential subtraction. Correcting after subtraction does not
cancel, because the two lines carry different total conductances. The algebra
collapses to the ideal product:

    (G_p - G_n)^T V

Cost: one multiply per column, foldable into the block-scale multiply already
present in the digital path.

---

## Validation

| Level | Checked against | Worst disagreement |
|---|---|---|
| ngspice DC solve | reference | -- |
| closed-form nodal model | ngspice, 87,168 tiles | 3.41e-7 % |
| vectorised GPU solver | nodal model | < 1e-12 relative |
| end-to-end inference | vectorised solver | same code path |

Every tile of every layer, 357 M MAC-reads, zero errors.

Device parameters, from SPICE calibration of the `rram_v_1_0_0` Verilog-A
model: R_LRS = 542.8 ohm, R_MID = 1099.8 ohm, R_HRS = 218587.2 ohm
(on/off ratio 403), R_sense = 20 ohm, V_read = 0.1 V.

---

## Layout

```
*.py                    simulation, evaluation, hardware model, figures
rtl/                    SystemVerilog/Verilog periphery and testbenches
spice/                  ngspice decks and the Verilog-A ReRAM model
docs/                   report, explainer (PDF + LaTeX source), open items
index.html, server.py   browser dashboard
run_all.sh              full pipeline, staged
run_blocking.sh         pre-submission checklist, staged
```

Key modules:

| File | Role |
|---|---|
| `analog_eval.py` | End-to-end inference through the crossbar equations. Holds the ADC, variability and drift models. |
| `crossbar_array_test.py` | Scalar reference solver and ngspice netlist generation. |
| `ngspice_full_sweep.py` | Exhaustive per-tile ngspice validation, parallel and resumable. |
| `benchmark_resnet18.py` | Per-layer SNR and readout error against a golden model. |
| `qat_finetune_fp2.py` | FP2 quantization-aware training with a straight-through estimator. |
| `hw_model.py` | Area, energy and delay. Every assumption is a named, overridable constant. |
| `codesign_sweep.py` | Block-size Pareto sweep and the SRAM baseline. |
| `make_figures.py`, `figures_sweeps.py` | All figures and LaTeX tables, generated from the CSVs. |

---

## Requirements

```bash
pip install -r requirements.txt
```

ngspice is required only for the SPICE validation paths:

```bash
sudo apt install ngspice
```

A CUDA GPU is optional. Full-test-set sweeps take minutes on a GPU and hours
on CPU.

## Quick start

```bash
# verify the solver against its scalar reference
python3 analog_eval.py --self-test

# the headline sweep: raw vs calibrated at four tile heights
python3 analog_eval.py --sweep 32,64,128,256 --data-dir ./data \
  --max-images 0 --adc-bits 6 --out-csv analog_accuracy.csv

# figures and LaTeX tables
python3 make_figures.py --outdir paper/figures
python3 figures_sweeps.py --outdir paper/figures

# dashboard
python3 server.py --fetch-vendor && python3 server.py --port 5057
```

`run_all.sh` runs any stage or all of them, with live output and per-stage
logs. `run_all.sh` with no argument lists the stages.

---

## Scope and limitations

Stated plainly, because several of these bound the claim.

- **Drift.** ReRAM conductance relaxes as a power law. A calibration computed
  once at manufacture loses 27 points after a simulated year. Recomputing it
  from the drifted array holds accuracy within 0.8 points indefinitely. The
  supported claim is calibration *with periodic refresh*, not compile-time
  calibration. Roughly ten refreshes over a product lifetime suffice, since
  damage accumulates with the logarithm of time.
- **Die area.** At B=128, fully weight-stationary ResNet-18 models to 412 mm2
  at 65 nm. The argument depends on scaled nodes.
- **Write cost.** Reprogramming is ~100 pJ/cell against ~0.001 pJ/cell to read.
  Break-even against time-multiplexing is ~2200 images, so only fully
  weight-stationary operation is viable.
- **Wordline IR drop** is not modelled. Unlike bitline loading it is
  activation-dependent, so it would *not* reduce to a compile-time constant.
  This is the most likely remaining threat to the contribution.
- **Level mapping.** R_MID/R_LRS = 2.0262 rather than exactly 2, and finite HRS
  leakage shifts every level. Level 0.5 lands at 0.492. Separate from readout
  error, currently absorbed into the reported digital ceiling.
- **RTL.** `rtl/resnet18_reram_top.sv` has no behavioural model at the ADC
  boundary, so the RTL cross-check reports N/A. The synthesis flow is blocked
  on this.
- **NeuroSim comparison** is VGG-8 against ResNet-18 and is not like-for-like.

`docs/TODO.md` tracks open items, including two retracted claims and the
reasoning behind each retraction.

## Reproducing the figures

Every number traces to a named CSV, and every figure and table is regenerated
from those CSVs by `make_figures.py` and `figures_sweeps.py`. Captions that
state a threshold derive it from the data rather than hardcoding it.

## Citation

```bibtex
@misc{fp2reram,
  title  = {Block Size Is Tile Height: Loading-Gain Calibration for
            2-bit Floating-Point Compute-in-Memory},
  author = {Nandi, Shaivi},
  year   = {2026},
  note   = {https://github.com/<user>/fp2-reram-crossbar}
}
```

## License

MIT. See `LICENSE`.
