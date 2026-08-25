# Every number, where it came from, and how sure we are

One table per category. Three provenance tiers:

| Tier | Meaning |
|---|---|
| **CITED** | taken from a named publication, quote in the notes |
| **SIM** | measured by ngspice or by our own solver |
| **ASSUM** | assumed value, bounded by a sweep or by cited limits |

---

## 1. Number format

| Quantity | Value | Tier | Source |
|---|---|---|---|
| Block size *k* | **32** | CITED | Rouhani et al., *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537. Table 1: every concrete MX format (MXFP8/6/4, MXINT8) uses block size 32. |
| Shared scale format | **E8M0**, 8 bit | CITED | Same, Table 1: "All concrete MX formats use E8M0 (an 8-bit exponent) as the format for the shared scale." |
| Element levels | {−1, −0.5, 0, +0.5, +1} | — | FP2-E1M0, this work |
| Bits per weight | 2.25 | — | derived: 2 bits/element + 8 bits/32 elements |

**This closes the B=32 citation.** It is an Open Compute Project standard rather
than a single paper, which is stronger. The framing becomes: *MX fixes k=32 for
digital packing reasons; on a crossbar that number becomes the tile height.*

---

## 2. ReRAM device

| Quantity | Value | Tier | Source |
|---|---|---|---|
| `R_LRS` | 542.8 Ω | SIM | SPICE calibration of the `rram_v_1_0_0` Verilog-A compact model |
| `R_MID` | 1099.8 Ω | SIM | same |
| `R_HRS` | 218 587.2 Ω | SIM | same |
| On/off ratio | 402.7 | — | derived |
| `R_MID / R_LRS` | 2.0262 | — | derived; should be exactly 2, hence the level-mapping error |
| `R_sense` | 20 Ω | — | design choice, swept in B2 |
| `V_read` | 0.1 V | — | design choice |

**Largest caveat in the project:** these come from a compact *model*, not
fabricated silicon.

### Level-mapping error (derived, verified in `run_blocking.sh b4`)

| Level | Ideal | Actual | Error |
|---|---|---|---|
| +1.0 | 1.0000 | 1.00000 | 0.000% |
| +0.5 | 0.5000 | 0.49228 | **−1.543%** |
| 0.0 | 0.0000 | 0.00000 | 0.000% |

Fix: trim `R_MID` to **1082.91 Ω** (−1.54%), asserted exact in the script.

---

## 3. ADC

| Quantity | Value | Tier | Source |
|---|---|---|---|
| Figure of merit | **20 fJ/conv-step** | **CITED** | Firlej 2024, see below |
| Area | 0.005 mm² | ASSUM, **bounded** | see below |
| Resolution | 6 bit | SIM | measured requirement, flat in B, full test set |
| Clock | 500 MHz | ASSUM | conversion takes `ADC_BITS` cycles → 12 ns |
| Converters per column | **2** | — | required by the method; both bitlines sensed separately |
| Column mux | 8:1 | ASSUM | |

**Figure of merit — matched, not bracketed.**

M. Firlej et al., "Ultra-low power 10-bit 50–90 MSps SAR ADCs in 65 nm CMOS
for multi-channel ASICs", *JINST* **19** P01029 (2024). Same node, same
architecture, same speed class, measured silicon. FoM computed from the
reported power, ENOB and sampling rate as `P / (2^ENOB · Fs)`:

| Firlej variant | Power | ENOB | Fs | FoM |
|---|---|---|---|---|
| standard | 1000 µW | 9.0 | 80 MSps | **24.4 fJ/conv-step** |
| standard | 500 µW | 9.2 | 40 MSps | **21.3 fJ/conv-step** |
| low-power | 400 µW | 9.3 | 40 MSps | **15.9 fJ/conv-step** |
| **this work** | — | — | ~83 MS/s | **20 fJ**, 6 b, 65 nm |

The assumed 20 fJ sits between their 40 and 80 MSps points. Because 6 bits
requires fewer conversion steps than their 10, the assumption is if anything
conservative. Firlej is the primary reference; the wider bracket below is
retained for context.

| Wider bracket | FoM | Conditions |
|---|---|---|
| Tai et al., ISSCC 2014, 11.2 | 0.85 fJ/conv-step | 10 b, 200 kS/s, 40 nm |
| 65 nm 1.2 V 12 b 30 MS/s ADC | 410 fJ | 12 b, 30 MS/s, 18 mW |

**Area bounds:** Tai's 10-bit core is 0.0065 mm² at 40 nm.

| Basis | Value at 65 nm |
|---|---|
| scaled by (65/40)², no bit scaling | **0.0172 mm²** (pessimistic) |
| plus 2^(6−10) capacitor-array scaling | **0.0011 mm²** (optimistic) |
| this work assumes | 0.005 mm² (between the two) |

The answer depends on how 10-bit area is scaled to 6-bit, so **report the
range**.

---

## 4. DRAM and memory hierarchy

| Quantity | Value | Tier | Source |
|---|---|---|---|
| DRAM access energy | **10 pJ/bit** | CITED | Horowitz, *Computing's Energy Problem (and what we can do about it)*, ISSCC 2014, pp. 10–14: "even when the I/O is improved, the energy cost of a DRAM access will still be large (10pJ/bit, 0.6nJ/8B)" |
| DRAM, contemporary I/O | **20 pJ/bit** | CITED | Same: DRAM I/O "takes over 20pJ/bit" |
| SRAM bitcell | 140 F² | **CITED bound** | 150–200 F² measured — see §5 |
| SRAM leakage | 120 pW/bit | **CITED** | Verma & Chandrakasan JSSC 2008, 65 nm — see below |
| SRAM read | 0.05 pJ/bit | ASSUM | macro read plus delivery |

10 pJ/bit is Horowitz's **optimistic floor**, used deliberately so the digital
baseline is not strawmanned.

**SRAM retention leakage — CITED at the right node.**

N. Verma and A. P. Chandrakasan, "A 256 kb 65 nm 8T Subthreshold SRAM
Employing Sense-Amplifier Redundancy", *IEEE JSSC* **43**(1), pp. 141–149,
Jan. 2008. **65 nm CMOS — no node scaling required.**

| Measured | Value |
|---|---|
| Total leakage, 256 kb macro at V_min = 350 mV | **2.2 µW** |
| Per bit | 2.2 µW / 262144 = **8.39 pW/bit at 0.35 V** |

The same paper supplies the voltage dependence: *"in a 65 nm process, the
leakage current reduction from a V_DD of 1 V to 0.3 V due to DIBL is over 4×,
and the leakage power savings is over 10×."* Applying that:

| V_DD | Leakage/bit at 65 nm |
|---|---:|
| 0.35 V (measured) | 8.39 pW |
| 1.0 V (paper's own ≥10× factor) | **≳84 pW** |
| 1.2 V (further DIBL + higher V in P = IV) | **~125–170 pW** |

**The assumed 120 pW/bit sits inside that range.** It is defensible, and the
non-volatility argument stands. Two notes: the cell is 8T, so two extra
transistors leak per bit and 8.39 pW/bit slightly *over*-states a 6T cell; and
the ≥10× factor is a lower bound, so 120 pW/bit is a conservative reading.

Note also the 22 nm part's 10 pW/bit *shutdown* figure is not usable in any
case: shutdown collapses the array and loses the weights.

**6T cell sizing at 65 nm, measured.** *A 0.9 V 64 Mb 6T SRAM cell with read
and write assist schemes in 65 nm LSTP technology*: PU/PD/PG widths
**120 / 220 / 140 nm**, gate length **90 nm**, cell ratio 1.57, V_min 0.9 V
with assist. Use this rather than a generic 6T when the reviewer asks what
SRAM cell is being compared against.

---

## 5. Cell and array area

| Quantity | Value | Tier | Status |
|---|---|---|---|
| `CELL_AREA_F2` | 40 F² | **CITED bracket** | 31–84 F² measured, see below |
| 1T1R select transistor width | 127.28 F at R_LRS=542.8 | SIM | NeuroSim's own sizing |
| Technology node | 65 nm | — | |
| Array periphery overhead | 1.4× | ASSUM | 1.3–1.6 typical |

**Measured 1T1R bitcell areas, both from silicon:**

| Source | Node | Bitcell | In F² | Note |
|---|---|---|---|---|
| W. Shim, X. Sun, J.-S. Seo, S. Yu, "2-Bit-Per-Cell RRAM-Based In-Memory Computing for Area-, Energy-Efficient Deep Learning", *IEEE Solid-State Circuits Letters*, 2020 | 90 nm CMOS + HfO₂ | 0.5 × 0.5 µm | **≈31 F²** | authors' own figure; IMC macro |
| *A 1 Mb RRAM Macro with Bipolar Forming …* | 40 nm | 0.135 µm² | **≈84 F²** | dense NVM macro |

`CELL_AREA_F2 = 40` therefore sits inside a range set by two fabricated parts,
nearer the aggressive end. **This closes the dispute on the number itself.**

**But the resistance caveat stands.** NeuroSim sizes the select transistor to
carry the LRS read current and demands 127.28 F against a 12 F default. Width
scales as 1/`R_LRS`. Backing out the resistance that fits a 12 F cell gives
**5757 Ω**, and NeuroSim's own commented-out default on that line is `6e3` —
they agree. So 40 F² is valid at R_LRS ≳ 6 kΩ, **not** at 542.8 Ω. Both cited
parts above run at high LRS, which is consistent.

**6T SRAM cell area, from the same 90 nm IMC paper:** two SRAM cells occupy
"300–400 F²", i.e. **150–200 F² per 6T cell**. The model's
`SRAM_BIT_F2 = 140` is therefore *below* the measured range — generous to
SRAM, which is the safe direction for the comparison and should be stated as
such.

**Write energy.** E. Perez et al., "Variability and Energy Consumption
Tradeoffs in Multilevel Programming of RRAM Arrays", *IEEE TED* **68**(6),
pp. 2693–2698, 2021. 4-kbit HfO₂ 1T1R array. Their Eq. (1) gives the energy of
a set operation as `Σᵢ (V_pr,i · I_pr,i · T_pr + V_rd · I_rd,i · T_rd)` with
V_pr = 1.2 V, T_pr = T_rd = 1 µs, and compliance currents I_trg of 10/20/30/40
µA. One programming pulse is therefore **12–48 pJ**, and a multi-pulse
write-verify set runs to several hundred pJ. This brackets the ~100 pJ/cell
figure used in the model. Read the exact per-level averages off their Fig. 10
before quoting a single number — the text gives the equation, not the values.

---

## 6. Digital MAC (von Neumann baseline)

| Quantity | Value | Tier |
|---|---|---|
Horowitz ISSCC 2014 **Figure 1.1.9**, "Rough energy costs for various
operations in 45 nm 0.9 V", supplies both the MAC and the SRAM-read terms.
Values read off the figure:

| | 8 bit | 32 bit | | 16 bit | 32 bit |
|---|---:|---:|---|---:|---:|
| Int Add | 0.03 pJ | 0.1 pJ | FAdd | 0.4 pJ | 0.9 pJ |
| Int Mult | 0.2 pJ | 3.1 pJ | FMult | 1.1 pJ | 3.7 pJ |

| Memory (64-bit access) | 8 KB | 32 KB | 1 MB | DRAM |
|---|---:|---:|---:|---:|
| Cache | 10 pJ | 20 pJ | 100 pJ | 1.3–2.6 nJ |

**Both legacy constants were optimistic, and correcting them helps the
result.** Run `vonneumann_baseline.py --horowitz` to use the cited set.

| Quantity | Legacy | Cited | Derivation |
|---|---:|---:|---|
| INT8 MAC | 0.30 pJ | **0.591 pJ** | (0.2 mult + 0.03 add) × 2.57 for 45 nm 0.9 V → 65 nm 1.2 V |
| FP2 MAC | 0.03 pJ | **0.077 pJ** | 8-bit add scaled; a 2-bit multiply is a select, so the accumulator dominates |
| SRAM read | 0.05 pJ/bit | **0.3125 pJ/bit** | 32 KB cache, 20 pJ per 64-bit access — see caveat |
| DRAM | 10 pJ/bit | consistent | 1.3–2.6 nJ / 64 b = 20–41 pJ/bit; 10 is the text's improved-I/O floor |

**SRAM read — settled by three independent sources, two of them macros.**
Horowitz's cache figures alone do not decide this: a *processor cache* carries
tag lookup, tag comparison, way selection, decode and the return path to the
core, none of which a directly-addressed scratchpad pays. His 1 MB figure is
an upper bound against a 4 MB scratchpad, not the answer. Putting three
sources on the same footing — advanced nodes scaled to 65 nm 1.2 V by (65/F)
and (1.2/V)² — they converge:

| Source | Node, supply | As reported | At 65 nm 1.2 V |
|---|---|---:|---:|
| Horowitz 32 KB cache | 45 nm, 0.9 V | 20 pJ / 64 b | **0.313** (left unscaled) |
| 22 nm SRAM, *JSSC* 2025 | 22 nm, 0.5 V | ~20.5 fJ/bit † | **0.349** |
| 14 nm FinFET FVS SRAM | 14 nm, 0.5 V | 24.6 fJ/bit | **0.658** |

† **Derived, not quoted.** That paper headlines *10 aJ/total-bit*, which is
macro energy divided by the full 256 Kb capacity, explicitly chosen "to
reflect the leakage effect of un-accessed banks" — **not** energy per bit
read. Its array is 8 banks × 2 × (128 × 128), so a 128-bit access implies
10 aJ × 262144 / 128 = 20.5 fJ per accessed bit. The access width is inferred
from the geometry; treat the row as approximate.

**Converged range at 65 nm 1.2 V: 0.3–0.7 pJ/bit.** The legacy 0.05 pJ/bit is
therefore **6–13× optimistic** — not the 31× a naive read of the 1 MB cache
figure suggests. The model uses 0.3125, the optimistic edge of the range.

**Which constant actually decides the comparison.** As reuse grows the fetch
term vanishes and TOPS/W → 2/`mac_pj`, independent of SRAM read energy:

| FP2 MAC | asymptotic TOPS/W | ReRAM CIM (47.09) |
|---|---:|---:|
| 0.030 pJ (legacy) | 66.67 | **0.71× — CIM loses** |
| 0.077 pJ (scaled) | 25.97 | **1.81× — CIM wins** |

The load-bearing correction is the **MAC term**, and it is pure node scaling:
0.03 pJ is the 45 nm value used unscaled at 65 nm. SRAM read energy only moves
the low-reuse end, where CIM wins across the whole plausible range:

| SRAM read pJ/bit | source | von Neumann @ reuse 1 | ReRAM CIM advantage |
|---:|---|---:|---:|
| 0.05 | legacy assumption | 11.30 | 5.8× |
| 0.156 | Horowitz 8 KB | 5.14 | 12.8× |
| 0.3125 | **Horowitz 32 KB, used** | 2.85 | **23.0×** |
| 1.5625 | Horowitz 1 MB | 0.62 | 105.0× |

Report the sweep, not a point. Cache energy is left unscaled at 45 nm; scaling
to 65 nm would raise it a further 2.57× and is not claimed.

Still uncited: MAC unit area (1800 / 220 µm²) and SRAM leakage pW/bit.
Control logic, instruction fetch and clock distribution are **not** modelled,
so the baseline remains optimistic and every advantage quoted against it is a
lower bound.

Control logic, instruction fetch and clock distribution are **not** modelled,
so the baseline is optimistic and every advantage quoted against it is a lower
bound.

---

## 7. Drift

| Quantity | Value | Tier | Source |
|---|---|---|---|
| Model | `G(t) = G₀(t/t_ref)^(−ν)` | CITED | Pries et al., *Resistance Drift Convergence and Inversion in Amorphous Phase Change Materials*, Adv. Funct. Mater. 2022 |
| `ν_LRS` | 0.01 | ASSUM | |
| `ν_HRS` | 0.05 | ASSUM | HRS relaxes faster — no stable filament |
| `σ_ν` | 0.004 | ASSUM, **swept** | least certain input; boundary measured at 0.008 |
| `t_ref` | 1 s | — | |

---

## 8. Simulation tools

| Tool | Used for | Source |
|---|---|---|
| **ngspice** | validating the closed-form nodal model, 87 168 tiles | ngspice.sourceforge.io |
| **DNN+NeuroSim V1.4** | chip area, energy, latency | Peng, Huang, Luo, Sun, Yu, *DNN+NeuroSim*, IEDM 2019. V1.5: arXiv:2505.02314 |
| `analog_eval.py` | **all accuracy** — FP2 mapping, crossbar solve, calibration | this work |
| `hw_model.py` | tile-level area/energy/delay | this work |

**Division of labour, stated in methodology:** accuracy comes from our
circuit-exact simulator, which models the FP2 three-state differential mapping;
NeuroSim supplies area, energy and latency, which depend on operand width and
array geometry rather than the level mapping. **NeuroSim accuracy figures are
never quoted.**

---

## 9. Measured results

### Validation
| Level | Reference | Worst disagreement |
|---|---|---|
| Closed-form nodal model | ngspice, **87 168 tiles**, 357 M MAC-reads | **3.41×10⁻⁷ %** |
| Vectorised GPU solver | nodal model | <10⁻¹² relative |
| 2-D mesh solver | closed form, ideal wires | 8.8×10⁻⁸ relative |

### The collision — CIFAR-10 Top-1, full 10 000-image test set
| B=M | FP2 digital | raw crossbar | **calibrated** |
|---:|---:|---:|---:|
| 32 | 92.42 | 77.66 | **92.42** |
| 64 | 92.47 | 57.54 | **92.47** |
| 128 | 92.81 | 11.60 | **92.81** |
| 256 | 92.43 | 10.00 | **92.43** |

### Device sweep — on/off ratio held at 403, 2000 images
| R_LRS | loading err | raw | **calibrated** |
|---:|---:|---:|---:|
| 543 | 58.6% | 12.35 | **91.95** |
| 6 000 | 11.4% | 91.00 | **91.70** |
| 100 000 | 0.8% | 91.75 | **91.95** |

**Calibrated flat within 0.40 pts over a 184× range; raw varies 79.6 pts.**

### BatchNorm baseline — full test set, 6-bit ADC
| B=M | BN gap | **per-column gap** |
|---:|---:|---:|
| 32 | −1.67 | **−0.07** |
| 256 | −2.19 | **−0.55** |

Worst case 4.3× better. The predicted tile-height trend was **tested and did
not hold** — dropped.

### Variability — 10 seeds, full test set
Blind calibration loses **0.90 pts** from σ=0 to σ=20%.

### Drift — B=128, 1 year
| σ_ν | stale | refreshed |
|---:|---:|---:|
| 0.004 | 71.73 | **91.73** |
| 0.008 | 73.36 | 90.54 |
| 0.016 | 77.91 | **80.29 — fails** |

Refresh holds to σ_ν ≈ 0.004, begins failing at 0.008.

### Wire parasitics — M=128, residual after per-column calibration
| Case | Residual |
|---|---:|
| ideal wires | **0.0002%** |
| wordline metal 0.5 Ω | 1.82% |
| wordline driver 50 Ω | 25.4% |
| bitline metal 0.5 Ω | **49.1%** |

Distributed wire resistance is **not** compile-time correctable. Claim narrowed
to the R_sense term.

### Technology comparison — NeuroSim, 2-bit weights, VGG-8

**6-bit hardware ADC.** NeuroSim's hardware converter resolution is set by
`levelOutput` in `Param.cpp`, not by the Python `--ADCprecision` flag;
`levelOutput = 64` is required for a 6-bit converter.

| | chip area | latency/img | TOPS/W | TOPS/mm² |
|---|---:|---:|---:|---:|
| SRAM CIM | 23.93 mm² | 149.28 µs | 67.99 | 0.3449 |
| ReRAM CIM (6 kΩ) | 120.41 mm² | 178.18 µs | 47.09 | 0.0574 |
| ReRAM CIM (20 kΩ) | 109.31 mm² | 149.97 µs | 58.54 | 0.0751 |
| ReRAM CIM (50 kΩ) | 107.59 mm² | 134.44 µs | 65.09 | 0.0852 |
| MRAM-like (ratio 3) | 88.49 mm² | 216.95 µs | 37.90 | 0.0642 |

### Tile height — the area and efficiency argument, 6-bit ADC
| rows | chip area | named subtotal | ADC area | TOPS/W |
|---:|---:|---:|---:|---:|
| 32 | 342.51 mm² | 75.41 mm² | 13.51 mm² | 21.23 |
| 64 | 206.80 mm² | 46.46 mm² | 7.88 mm² | 33.17 |
| **128** | **120.41 mm²** | **33.71 mm²** | **4.04 mm²** | **47.09** |

Quote the **named subtotal** ratio of 2.24×, not the chip-total 2.84×: the
chip total tracks floorplan geometry, and 65.6% of it sits inside per-tile
bounding boxes that no component line reports.

**Caveat:** reported categories sum to ~30% of ReRAM chip area against ~96%
for SRAM. The remainder sits inside NeuroSim's own `ChipArea` computation.

### MRAM — FP2 is unrepresentable at low on/off ratio
| Level | should read | reads at ratio 3 |
|---:|---:|---:|
| +1.0 | 1.000 | **0.667** |
| +0.5 | 0.500 | **0.160** |

Accuracy at ratio 3: **10.70% (chance)**, against 91.70% at ratio 403.
**FP2 requires ReRAM's high on/off ratio; MRAM cannot hold the format.**

### Against von Neumann — FP2 weights, 4 MB buffer
| reuse | von Neumann | SRAM CIM | ReRAM CIM |
|---:|---:|---:|---:|
| 1 | 2.85 | 67.99 | 47.09 |
| 1000 | 25.75 | 67.99 | 47.09 |

ReRAM CIM is **16.5×** von Neumann at reuse 1 and **1.8×** at reuse 1000; SRAM
CIM is 23.9× and 2.6×. Both use the cited Horowitz constants and the 6-bit
converter. The CIM columns are flat because
they never fetch a weight — that flatness is the O(1)-MAC advantage.

### FP2 vs INT8 in the digital baseline
| | model size | DRAM miss | TOPS/W @ reuse 1 |
|---|---:|---:|---:|
| INT8 | 11.16 MB | 0.641 | 0.04 |
| **FP2** | **3.14 MB** | **0.000** | **15.38** |

**384× better**, purely because FP2 fits a 4 MB on-chip buffer. Uses the cited
Horowitz DRAM figure.

---

## 10. What still needs a citation

| Item | Status |
|---|---|
| ADC figure of merit | ✅ **closed** — Firlej, JINST 2024, 65 nm SAR, matched node and speed |
| 1T1R cell area | ✅ **closed** — 31 F² (Shim 2020, 90 nm) to 84 F² (1 Mb macro, 40 nm) |
| ReRAM write energy | ✅ **closed** — Perez, TED 2021, Eq. (1); 12–48 pJ per pulse |
| 6T SRAM cell area | ✅ **closed** — 150–200 F², Shim 2020 |
| 6T SRAM sizing at 65 nm | ✅ **closed** — 64 Mb LSTP paper, PU/PD/PG = 120/220/140 nm |
| Silicon CIM macro for the table | ✅ **closed** — four parts, §12 |
| ReRAM drift ν, σ_ν | ✅ **closed enough** — Joshi, *Nat. Commun.* 2020 (CIM-specific) supersedes Pries (PCM material) |
| SRAM read pJ/bit | ✅ **closed** — Horowitz Fig. 1.1.9, 1 MB cache, 1.5625 pJ/bit |
| INT8 MAC energy | ✅ **closed** — Horowitz Fig. 1.1.9, 0.23 pJ @45 nm → 0.591 pJ @65 nm |
| SRAM leakage pW/bit at 65 nm | ✅ **closed** — Verma & Chandrakasan JSSC 2008, 8.39 pW/bit @0.35 V, ≥10× to 1.0 V per the same paper. See §4. |
| Digital MAC unit **area** (1800 / 220 µm²) | ⚠ **no citable source** — see below |

**On MAC unit area.** The uploaded MAC review (*International Innovative
Research Journal of Engineering and Technology*, IIRJET, ISSN 2456-1983)
reports **1083–1252 µm²** for MAC units at TSMC 65 nm, which is the same order
as the assumed 1800 µm², but IIRJET is not an indexed venue
with recognised peer review, the text is poorly edited throughout, and the
numbers are secondhand from papers it summarises. A DATE reviewer will notice.

Two acceptable options: chase the primary sources behind that review's tables
and cite those, or **delete the MAC-area column entirely**. Nothing in the
argument depends on it — area enters only the von Neumann row, and the buffer
dominates it (18.93 mm² of 19.15 mm²). Deleting is the cheaper fix.

---

## 12. Silicon CIM macros — the comparison table

All measured parts, for the related-work table. Note the units differ: some
quote **MAC-level** efficiency (the array only), some **full-network** (the
whole chip including buffers and data movement). Mixing them is the most
common error in this literature.

| Macro | Node | Precision | MAC-level | Full-network | Area |
|---|---|---|---:|---:|---|
| FACIM, fully analog CIM | TSMC 0.18 µm | INT8 | 918 TOPS/W | **41.1 TOPS/W** | 20.98 mm², 635 kb |
| FACIM, high-speed point | TSMC 0.18 µm | INT8 | 1880 TOPS/W | **117 TOPS/W** | 97.76 GOPS/mm² |
| TDC-CiM, resonant SRAM CIM | TSMC 28 nm | INT8 | — | **38.46 TOPS/W** | 320 GOPS |
| SRAM-LUT IMC, dense | 28 nm | INT8 | — | **2.32 TOPS/W** | 0.172 mm² |
| SRAM-LUT IMC, sparse | 28 nm | INT8 | — | **72.93 TOPS/W** @0.56 V | |
| SRAM-LUT IMC, sparse | 28 nm | FP8 | — | 31.42 TFLOPS/W | 3.56 TFLOPS/mm² |
| C2IM, 10T unit | TSMC 65 nm | 1 b weight | 166.67 TOPS/W | — | unit-level, 200 MOPS |
| 9 kb SRAM CIM, segmented charge sharing | 28 nm | multibit MAC | — | **30.74 TOPS/W** | 1.15 TOPS/mm²; 9.52 fJ/bit best corner → 105.04 TOPS/W |
| InFP, reconfigurable SRAM CIM | TSMC 45 nm | BF16 | — | **17.97 TFLOPS/W** | 0.081 mm², 0.395 TFLOPS/mm² |
| SRAM digital CIM, linear interpolation | — | 8 b | — | **11.13–17.72 TOPS/W** | 4.61–8.36 for 19-b outputs |

**The FACIM row is the one to cite in defence of your own numbers.** The same
chip reports 918 TOPS/W for MAC operations and 41.1 TOPS/W for full-network
inference — a **22× gap on one piece of silicon**, from peripheral and data
movement cost the array-level figure excludes. This is direct published
evidence for why a chip-level TOPS/W is far below a tile-level one, and it
pre-empts the reviewer who compares your chip-level result against someone
else's array-level headline.

---

## 11. Re-runs needed

| Run | Needed? |
|---|---|
| `analog_eval` accuracy sweeps | **No** — current |
| ngspice validation | **No** — one-time, current |
| variability, drift, ADC sweeps | **No** — current |
| BN baseline | **No** — current |
| wire parasitics | **No** — current |
| `rlrs_tradeoff --accuracy` | **Yes** — first working run only covered 2000 images; re-run at `--max-images 0` for the paper |
| NeuroSim comparisons | **Only** if the ADC area bound changes |
| Figures | **Yes** after the above — `make_all_figures.py` |
