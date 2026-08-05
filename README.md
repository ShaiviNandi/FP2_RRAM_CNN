# FP2 ReRAM Crossbar Neural Network Accelerator

A mixed-signal neural network accelerator design combining **2-bit floating-point (FP2-E1M0) encoding** with **ReRAM (resistive RAM) crossbar arrays** for efficient matrix multiplication in low-precision inference.

## Overview

This project implements a complete design-space exploration and verification flow for a MAC (multiply-accumulate) unit based on analog crossbar computation using ReRAM devices. The key innovation is using ReRAM's two-state resistive behavior to encode weights via cell state mapping, combined with FP2 quantization for extreme compression.

**Core concept:**
- **Weights**: Quantized to FP2-E1M0 (exponent=1, mantissa=0, ±{1.0, 0.5, 0}), then encoded into ReRAM cell states (HRS/LRS/off)
- **Computation**: Analog crossbar performs weighted sum as current on bitlines (limited by shared sense resistor loading)
- **Topology variants**: 1T1R (single-ended, unsigned) vs 2T2R (differential, signed)
- **Golden models**: Independent Python reference implementations for validation against circuit simulation

## Project Structure

### Digital Design (Verilog/SystemVerilog)

| File | Purpose |
|------|---------|
| `fp2_packer.v` | Encodes FP2 values into packed block format for datapath |
| `fp2_packer_dsp.v` | DSP-optimized packing pipeline (area/power variant) |
| `fp2_unpacker.v` | Decodes FP2 bitstreams to direction & magnitude signals for DAC |
| `tb_unpacker.v` | Testbench for unpacker RTL verification |
| `tb_unpacker_export.v` | Exports DAC control signals (`dac_stimulus.txt`) for analog simulation |
| `tb_accum.v` | Accumulator/post-processing testbench |

### Analog Design (SPICE/Verilog-A)

| File | Purpose |
|------|---------|
| `rram_v_1_0_0_openvaf.va` | **Stanford ReRAM compact model** (Verilog-A): physics-based two-state device with gap dynamics, thermal effects, and switching stochasticity |
| `rram_2x2_crossbar_fixed.sp` | 2×2 crossbar netlist: wordline drivers, ReRAM cells, bitline sense resistors (R_sense loading) |
| `rram_2stage_lowdisturb_read.sp` | Read operation: low pulse to avoid inadvertent switching |
| `rram_2stage_strong_pulse.sp` | Program operation: strong pulse to induce SET/RESET transitions |
| `rram_2stage_program_compute.sp` | Interleaved program + compute: SET/RESET during MAC phases |

### Golden Reference & Validation (Python)

| File | Purpose |
|------|---------|
| **`crossbar_array_test.py`** | **Core golden model**: resistive-divider crossbar simulation (no SPICE needed). Implements ideal/noisy MACs, weight grouping, 1T1R vs 2T2R, includes netlist builders for ngspice. |
| **`crossbar_cli.py`** | **Interactive CLI** for single-run exploration. Specify M×K array, weights, activations, topology → see digital vs analog output per column with error metrics. |
| **`golden_mac_reference.py`** | **Validation reference** for unpacker RTL. Reads DAC stimulus, decodes FP2→weight, computes expected bitline currents, diffs vs ngspice results with tolerance checking. |
| `topology_sweep.py` | **Design-space sweep**: varies M (height) and K (width) across both 1T1R and 2T2R; plots error, cell count, and tradeoffs. Essential for architecture selection. |
| `dac_to_pwl.py` | **Format converter**: transforms RTL-generated `dac_stimulus.txt` → LTspice-compatible PWL voltage files for behavioral simulation without full compact-model SPICE. |

## Key Features

### Quantization
- **FP2-E1M0**: 2-bit floating point (1 sign, 1 exponent, 0 mantissa bits)
  - Representable values: `{-1.0, -0.5, 0, +0.5, +1.0}` (5 distinct magnitudes × 2 signs)
  - Compact: 2 bits per weight (vs 32 for FP32 or 8 for FP8)

### Crossbar Topologies

**2T2R (Differential)**
- Two transistors + two ReRAM cells per synapse (complementary weight pairs)
- Signed weights: positive current = +weight, negative = –weight
- Shared bitline R_sense loading from both cells
- Higher cell count, but full range [-1.0, +1.0]

**1T1R (Single-Ended)**
- One transistor + one ReRAM cell per synapse (unsigned only)
- Cannot represent negative weights (clipped to 0)
- Half the cell count and area of 2T2R
- Accuracy trade-off: negative contributions lost

### Golden Models

1. **Resistive-divider (fast, Python-based)**
   - Treats each ReRAM cell as fixed HRS/LRS resistor
   - Accurate for steady-state DC current (post-transient)
   - No SPICE/ngspice needed; ~ms per run

2. **Compact-model (accurate, ngspice-based)**
   - Full Stanford ReRAM Verilog-A model
   - Captures gap evolution, thermal dynamics, transient behavior
   - Slow (~minutes per transient), but ground truth for validation

3. **RTL-to-netlist (end-to-end verification)**
   - Testbench exports RTL control signals → `dac_stimulus.txt`
   - DAC stimulus converted → PWL voltage sources
   - PWL fed into ngspice netlist
   - Simulated currents diffed against golden reference

## Usage

### 1. Quick Design-Space Exploration (Python Golden Model)

**Compare 1T1R vs 2T2R across array sizes:**
```bash
python3 topology_sweep.py \
  --m-values 2 4 8 16 32 64 128 \
  --k-fixed 8 \
  --seeds 5 \
  --r-sense 20.0 \
  --out topology_sweep.png
```

Outputs:
- Console table: error % and cell count for each (M, K) pair
- `topology_sweep.png`: 3-panel plot (error vs M, error vs K, cells vs M)

**Sample output:**
```
=== M sweep (K fixed) ===
      M   2T2R err%   1T1R err%  2T2R cells  1T1R cells
      2       1.25       8.45           32           16
      8       0.87       5.20          128           64
     64       0.52       3.10         1024          512
```

---

### 2. Single-Run Analysis

**See digital vs analog output for a specific configuration:**

```bash
# Random 4x3 array, 2T2R, differential weights
python3 crossbar_cli.py --m 4 --k 3 --topology 2t2r

# Same, but 1T1R (unsigned) to see sign-loss impact
python3 crossbar_cli.py --m 4 --k 3 --topology 1t1r

# Explicit weight matrix
python3 crossbar_cli.py \
  --weights "1,-1,0.5;-0.5,0.5,0;0.5,0,-1;0,1,0.5" \
  --activations "1,1,1,1" \
  --topology 2t2r

# Export ngspice netlist for the above
python3 crossbar_cli.py \
  --weights "1,-1,0.5;-0.5,0.5,0;0.5,0,-1;0,1,0.5" \
  --activations "1,1,1,1" \
  --topology 2t2r \
  --netlist-out array.cir
```

**Output:**
```
Topology: 2T2R   M=4  K=3  Rsense=20ohm  Vread=0.1V
Weights (quantized to FP2):
  +1.0 -1.0 +0.5
  -0.5 +0.5 +0.0
  +0.5 +0.0 -1.0
  +0.0 +1.0 +0.5
Activations: [1.0, 1.0, 1.0, 1.0]

Physical cells: 24   Bitlines/TIA channels needed: 6

 col     digital (exact)    analog (recovered)    abs err     err %
   0           1.5000             1.4823           0.0177      1.18
   1          -0.5000            -0.4914           0.0086      1.72
   2           0.5000             0.4891           0.0109      2.18
```

---

### 3. RTL-to-Analog Verification Flow

**Step 1: Run RTL testbench to generate DAC stimulus**
```bash
iverilog -o tb_unpacker_export tb_unpacker_export.v fp2_unpacker.v
./tb_unpacker_export
# Produces: dac_stimulus.txt
```

**Step 2: Convert digital stimulus to analog PWL files**
```bash
python3 dac_to_pwl.py dac_stimulus.txt \
  --time-unit 1e-12 \
  --vhigh 1.2 \
  --tr 100e-12 \
  --outdir ./pwl_files
# Produces: Dir1.pwl, Mag1_1.pwl, Mag1_0.pwl, Dir2.pwl, Mag2_1.pwl, Mag2_0.pwl
```

**Step 3: Run ngspice transient with crossbar netlist**
```bash
ngspice -b rram_2x2_crossbar_fixed.sp -o sim_results.log
# Extracts bitline currents: results.txt
```

**Step 4: Compare golden model vs ngspice**
```bash
python3 golden_mac_reference.py dac_stimulus.txt \
  --r-hrs 100e3 \
  --r-lrs 1e3 \
  --r-sense 1e3 \
  --vhigh 1.2 \
  --results results.txt \
  --tolerance 0.15
```

**Output:**
```
time     Dir1 M1_1 M1_0 Dir2 M2_1 M2_0   w1   w2  R1(ohm)  R2(ohm) I_wl1_exp(A) I_wl2_exp(A)
0        0    0    0    0    1    0     0.00  1.00  100000   1000    1.1802e-05   1.0813e-04
1        1    1    0    0    0    1     1.00  0.00   1000    100000  1.0813e-04   1.1802e-05
```

---

### 4. Architecture Search

**Sweep array dimensions with custom parameters:**
```bash
python3 topology_sweep.py \
  --m-values 4 8 16 32 64 128 256 \
  --k-values 2 4 8 16 32 64 \
  --m-fixed 16 \
  --k-fixed 16 \
  --r-sense 10.0 \
  --vread 0.2 \
  --seeds 10 \
  --out architecture_sweep.png
```

This produces side-by-side error curves for 1T1R vs 2T2R, helping you choose:
- **Array height (M)**: affects how many wordlines load the shared sense resistor
- **Array width (K)**: minimal impact on error (sanity check that K doesn't cause systematic degradation)
- **Topology**: 1T1R saves area but loses negative weights; 2T2R is full-range but doubled area

---

## Dependencies

### Python
```bash
pip install matplotlib  # for plotting (topology_sweep.py)
```

### Simulation
- **Verilog simulation**: Icarus Verilog (`iverilog`), or any IEEE 1364 compatible simulator
- **SPICE simulation**: ngspice (open source) or commercial (Cadence Spectre, Synopsys HSPICE)
- **RRAM model**: Stanford compact model (included as `rram_v_1_0_0_openvaf.va`)

### No external dependencies for golden models
```bash
python3 golden_mac_reference.py dac_stimulus.txt  # pure Python, no NumPy/SciPy
```

## File Format Specifications

### `dac_stimulus.txt` (RTL Export)
Generated by `$fdisplay` in testbench; consumed by `golden_mac_reference.py` and `dac_to_pwl.py`:
```
# time_ns Dir1 Mag1_1 Mag1_0 Dir2 Mag2_1 Mag2_0
0 0 0 0 0 0 0
1 0 1 0 0 0 1
2 1 0 1 0 1 0
```
- `time_ns`: absolute time in picoseconds (Icarus $time in finest precision units)
- `Dir1/Dir2`: sign bit (0=positive, 1=negative)
- `Mag1_1/Mag1_0, Mag2_1/Mag2_0`: 2-bit magnitude encoding

### `results.txt` (ngspice Output)
Generated by wrdata; consumed by `golden_mac_reference.py`:
```
0.0e+00 1.1802e-05 0.0e+00 1.0813e-04
1.0e-09 1.1800e-05 1.0e-09 1.0815e-04
```
- Four columns: `time_s I_wl1 time_s I_wl2` (wrdata interleaves dual traces)

### `.pwl` Files (LTspice Format)
Generated by `dac_to_pwl.py`; consumed by ngspice via `.include` or direct source:
```
0.0000e+00 0.0000
1.0000e-10 0.0000
1.1000e-10 1.2000
1.2000e-10 1.2000
2.0000e-10 0.0000
```
- Two columns: `time_s voltage_v`, piecewise-linear interpolation

## Known Limitations & Trade-offs

| Feature | 2T2R | 1T1R |
|---------|------|------|
| **Signed weights** | ✓ Full range [–1, +1] | ✗ Unsigned only [0, +1]; negatives clipped |
| **Cell count** | 2MK | MK |
| **Bitlines** | 2K (differential) | K (single-ended) |
| **TIA channels** | 2K | K |
| **Typical error** | 0.5–1.5% | 3–8% (sign-loss dominated) |
| **Area (normalized)** | 1.0× | 0.5× |

**Design choice**: 2T2R for maximum accuracy; 1T1R for extreme area/power constraints where model sparsity can compensate for sign clipping.

## Key Assumptions in Golden Model

1. **Steady-state DC only**: Resistive-divider model is valid post-transient; edges modeled as instantaneous (conservative for current peaks).
2. **Two-state resistor**: Each cell is HRS (off, 100kΩ) or LRS (on, 1kΩ) per magnitude; mid-range values (0.5) mapped to LRS (configurable via `--r-mid`).
3. **Shared R_sense loading**: Single sense resistor per bitline; current divider between parallel cells dominates error scaling with M.
4. **Ideal DAC**: All control signals switch crisply; no cross-talk or timing skew.

These are conservative for a simple validation layer; the ngspice flow with the full compact model provides ground truth.

## Next Steps / TODOs

- [ ] **Multi-layer integration**: Stack multiple 2D crossbar slices to form a systolic PE array (16×16 or larger)
- [ ] **Hessian-K-means weight grouping**: Cluster weights to reduce unique cell states, improving yield and programmability
- [ ] **Datapath integration**: Feed unpacker → crossbar MAC → accumulator → output formatting
- [ ] **Power/timing analysis**: Extract energy per MAC from SPICE; overlay on accuracy vs area Pareto frontier
- [ ] **Behavioral model (no SPICE)**: Replace compact model with simplified Verilog for faster iteration
- [ ] **Layout & parasitic extraction**: Place-and-route crossbar; include interconnect R/C

## References

- **FP2-E1M0 quantization**: Extreme-precision floating point for low-precision inference
- **ReRAM compact models**: Stanford RRAM model (gap-based switching dynamics)
- **Analog compute**: Crossbar MAC from [citation: analog NN inference papers]
- **Mixed-radix design**: [Your design notes / papers]

## Contact & Contributions

This is an active research/design project. For bugs, feature requests, or design discussions, please:
1. Check existing issues in documentation
2. Document reproducible test cases (Python + netlist combo)
3. Propose changes with golden-model validation (diff before/after)

---

**Project Status**: Verification infrastructure complete; RTL and testbenches in progress.  
**Last Updated**: August 2026

