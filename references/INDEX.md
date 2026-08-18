# Reference index

Every uploaded paper, the folder it sits in, and the exact number taken from
it. Values marked **derived** were computed here from what the paper reports,
not quoted — the derivation is given so it can be checked.

Cross-reference: `CITATIONS.md` in the project root.

---

## 01_number_format

| File | Cited for |
|---|---|
| `2310.10537v3.pdf` — Rouhani et al., *Microscaling Data Formats for Deep Learning* | Block size **B = 32**, shared **E8M0** 8-bit scale. The specification behind FP2-E1M0. |

---

## 02_reram_device

| File | Cited for |
|---|---|
| `2-Bit-Per-Cell_RRAM-Based_In-Memory_Computing…` — Shim, Sun, Seo, Yu, *IEEE SSC-L* 2020 | 1T1R bitcell **0.5 × 0.5 µm ≈ 31 F²** at 90 nm + HfO₂. Also **6T SRAM at 150–200 F²** ("two SRAM cells with 300–400 F²"). Settles `CELL_AREA_F2` and `SRAM_BIT_F2`. |
| `A_1Mb_RRAM_Macro_with_Bipolar_Forming…` | Bitcell **0.135 µm² at 40 nm ≈ 84 F²**. Upper end of the cell-area bracket. |
| `Variability_and_Energy_Consumption_Tradeoffs…` — Perez et al., *IEEE TED* 68(6) 2021 | Write energy. Eq. (1) with V_pr = 1.2 V, T = 1 µs, I_trg = 10–40 µA → **12–48 pJ per programming pulse**. **Derived**; per-level averages are in their Fig. 10, read them before quoting a point value. |
| `Resistive_Random_Access_Memory_RRAM_an_Overview_of.pdf` — Zahoor et al. | General device background. |
| `An_Energy-Efficient_Current-Controlled_Write_and_Read_Scheme…` | Write/read scheme context. No pJ figures in text. |
| `On_the_write_energy_of_non-volatile_resistive_crossbar_arrays_with_selectors.pdf` | Analytical write-energy model for arrays with selectors. |
| `Forming_Kinetics_in_hboxHfO_2_-Based_RRAM_Cells.pdf` | Device area **0.125 µm²**, range 0.125–2.544 µm². Forming kinetics. |
| `Fabrication_and_Characterization_of_Nanoscale_NiO…` | Confined-filament device characterisation. |

---

## 03_adc

| File | Cited for |
|---|---|
| `Firlej_2024_J._Inst._19_P01029.pdf` — **the ADC citation** | 10-bit SAR, **65 nm CMOS, 50–90 MSps**. FoM **derived** as `P / (2^ENOB · Fs)`: 24.4 fJ @80 MSps, 21.3 @40 MSps, 15.9 @40 MSps low-power. The assumed **20 fJ/conv-step** sits between their operating points, at matching node, architecture and speed class. |
| `11.2_A_0.85fJ_conversion-step_10b_200kS_s…` — Tai, ISSCC 2014 | Superseded FoM bracket (0.85 fJ at 200 kS/s). Still the source for **ADC area**: 0.0065 mm² 10-bit core at 40 nm → 0.0172 mm² pessimistic / 0.0011 mm² optimistic at 65 nm. |
| `A_65nm_CMOS_1.2V_12b_30MS_s_ADC…` | Superseded upper bracket, 410 fJ/conv-step. |
| `A_3bit_36GS_s_flash_ADC_in_65nm…` | Not used. Flash at 3 bit / 36 GS/s is the wrong architecture and speed class. |

---

## 04_memory_energy

| File | Cited for |
|---|---|
| `1.1_Computings_energy_problem…` — **Horowitz, ISSCC 2014** | Four separate numbers. **Text:** DRAM I/O "takes over 20pJ/bit"; improved I/O "will still be large (10pJ/bit, 0.6nJ/8B)" — the optimistic floor used in the model. **Figure 1.1.9** (45 nm, 0.9 V): Int Add 8b 0.03 pJ / 32b 0.1 pJ; Int Mult 8b 0.2 pJ / 32b 3.1 pJ; Cache 64-bit access 8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ; DRAM 1.3–2.6 nJ. |

Derived from Fig. 1.1.9: INT8 MAC = 0.2 + 0.03 = **0.23 pJ at 45 nm 0.9 V**,
scaled ×(65/45)×(1.2/0.9)² = 2.57 → **0.591 pJ at 65 nm 1.2 V**.
SRAM read from the 32 KB cache = 20/64 = **0.3125 pJ/bit**.

Caveat kept in `CITATIONS.md`: a processor cache carries tag lookup, tag
compare, way select and the return path to the core, none of which an
accelerator scratchpad pays. The 1 MB figure is an upper bound, not the answer.

---

## 05_sram

| File | Cited for |
|---|---|
| `A_256_kb_65_nm_8T_Subthreshold_SRAM…` — **Verma & Chandrakasan, JSSC 43(1) 2008** | **The leakage citation. 65 nm — no node scaling.** Measured: 256 kb macro draws **2.2 µW at V_min = 350 mV** → **8.39 pW/bit** (derived). Same paper gives the voltage dependence: 1 V → 0.3 V cuts leakage power "over 10×", so ≳84 pW/bit at 1.0 V and ~125–170 pW/bit at 1.2 V. The assumed **120 pW/bit** sits inside that range. Cell is 8T, so this slightly over-states a 6T cell. |
| `A_0.9V_64Mb_6T_SRAM_cell…65nm_LSTP` | 6T sizing at 65 nm: PU/PD/PG widths **120 / 220 / 140 nm**, L = 90 nm, cell ratio 1.57, V_min 0.9 V with assist. Use when asked what SRAM cell is being compared against. |
| `A_0.5-V_125-MHz_256-Kb_22-nm_SRAM…` | Access energy reference point. Headline is **10 aJ/total-bit**, which is macro energy over full 256 Kb capacity ("to reflect the leakage effect of un-accessed banks") — **not** energy per bit read. Array is 8 banks × 2 × (128×128), so a 128-bit access implies **~20.5 fJ/accessed-bit** (derived, access width inferred). Also 12 pW/bit deep-sleep, 10 pW/bit shutdown — both at 0.5 V, **not comparable to 65 nm 1.2 V**. |
| `A_High-Density_Low-Leakage…Fully_Voltage-Stacked_SRAM` | 14 nm FinFET macro: **24.6 fJ/bit access energy** and **5.34 pW/bit leakage** at 0.5 V. Clean per-accessed-bit definition. Advanced-node reference point only. |
| `Comparative_analysis_of_SRAM_cells_in_sub-threshold_region_in_65nm.pdf` | 65 nm sub-threshold 6T/10T comparison. Background. |
| `Improved_write_margin_6T-SRAM…` | 65 nm write-margin work. Background. |

---

## 06_cim_macros

Silicon parts for the related-work table. **Units differ — some quote
MAC-level (array only), some full-network (whole chip). Mixing them is the
most common error in this literature.**

| File | Node | Precision | Figure |
|---|---|---|---|
| `A_Fully_Analog_Computing-in-Memory_Macro…INT8-MAC` (FACIM) | TSMC 0.18 µm | INT8 | **918 TOPS/W MAC-level vs 41.1 TOPS/W full-network**; at speed, 1880 vs 117. 20.98 mm², 635 kb, 97.76 GOPS/mm². |
| `TDC-CiM…` | TSMC 28 nm | INT8 | 320 GOPS, **38.46 TOPS/W** |
| `A_High-SNR_SRAM-LUT-Based…` | 28 nm | INT8 / FP8 | 2.32 TOPS/W dense, **72.93 TOPS/W** sparse @0.56 V; FP8 1.87 / 31.42 TFLOPS/W. Macro 0.172 mm². |
| `A_28-nm_9-kb_SRAM_CIM…Segmented_Charge_Sharing` | 28 nm | multibit MAC | **30.74 TOPS/W**, 1.15 TOPS/mm²; 9.52 fJ/bit best corner → 105.04 TOPS/W |
| `InFP_A_17.97_TFLOPS_W…` | TSMC 45 nm | BF16 | **17.97 TFLOPS/W**, 0.081 mm², 0.395 TFLOPS/mm² |
| `C2IM_A_Compact…6T_SRAM` | TSMC 65 nm | 1 b weight | 166.67 TOPS/W, unit-level, 200 MOPS |
| `SRAM-Based_Digital_CIM_Macro_for_Linear_Interpolation_and_MAC` | — | 8 b | 11.13–17.72 TOPS/W; 4.61–8.36 for 19-b outputs |
| `8-b_Precision_8-Mb_ReRAM_CIM_Macro…` | — | 8 b ReRAM | The ReRAM silicon comparison point |
| `An_RRAM_Digital_CIM_Macro…` | — | — | Digital-CIM RRAM contrast |

**The FACIM row is the one to cite in defence of your own numbers.** One chip,
918 TOPS/W at MAC level and 41.1 TOPS/W full-network: a **22× gap on a single
piece of silicon**. Published evidence for why a chip-level TOPS/W sits far
below a tile-level one, and it pre-empts the reviewer comparing your
chip-level result against someone else's array headline.

---

## 07_drift

| File | Cited for |
|---|---|
| `s41467-020-16108-9.pdf` — Joshi et al., *Nature Communications* 2020 | **Preferred drift citation.** Computational PCM for DNN inference — the use case matches. |
| `Adv Funct Materials 2022 — Pries…` | Drift physics in amorphous phase-change material. Superseded by Joshi for this application, kept for the mechanism. |

Both are PCM. A ReRAM-specific ν and σ_ν would be better still.

---

## 08_uncited_or_unmined

| File | Status |
|---|---|
| `document.pdf` — MAC architectures review, IIRJET | **DO NOT CITE.** Not an indexed venue with recognised peer review, poorly edited, numbers secondhand from papers it summarises. Reports 1083–1252 µm² MAC area at TSMC 65 nm — same order as the assumed 1800 µm², but chase its primary sources or drop the area column instead. |
| `Analysis_of_an_Efficient_CIM-SRAM_for_VLSI_Applications.pdf` | Not mined. Contains third-party TOPS/W figures only. |
| `ReRAM Crossbar Non-Ideality Mitigation.pdf` | Not mined — relevant to the calibration novelty argument, worth a pass. |
| `s41378-026-01335-9.pdf` — Tsiamis et al., *Microsystems & Nanoeng.* 2026 | Not mined. |
| `fnins-10-00056.pdf` | Not mined. |
| `pxc3897431.pdf` | Not mined. |
| `2505.02314v1.pdf` | Not mined. |

---

## Still open

| Item | Note |
|---|---|
| Digital MAC unit area, 65 nm | No citable source. Recommended fix: delete the column. Area enters only the von Neumann row, where the buffer is 18.93 mm² of 19.15 mm². |
| ReRAM-specific drift exponents | Both drift sources are PCM. |
