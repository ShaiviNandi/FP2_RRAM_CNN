# What's left

Ordered by whether it blocks submission. Effort is my estimate of a session's
work, not calendar time.

---

## A. Blocking — the paper cannot go out without these

| # | Item | Who | Effort |
|---|---|---|---|
| A1 | **Novelty search.** "IR drop compensation ReRAM crossbar", "conductance-dependent output scaling CIM", "sneak-path calibration in-memory computing". Establishes whether the intro says "first to" or "unlike prior work". | **You** | 1 h |
| A2 | **Re-run the Pareto and `hw_final` at 6-bit.** Every efficiency figure currently quoted is either 5-bit or rescaled by hand. `ADC_BITS=6 bash run_all.sh pareto` then `hw_model --adc-bits 6 --power-csv ngspice_b128_summary.csv`. | run | 10 min |
| A3 | **Regenerate figures 1–7.** They still carry n=2000 numbers; the CSVs are current. `python3 make_figures.py --outdir paper/figures`. | run | 1 min |
| A4 | **Delete or regenerate `RESULTS.md`.** It is stale; `REPORT.md` superseded it. Two documents disagreeing is worse than one. | 5 min |
| A5 | **Cite the FP2 spec for B=32.** The entire framing rests on the format fixing the block size. Quote the paper. | **You** | 15 min |

---

## B. Important — a reviewer will ask

| # | Item | Notes | Effort |
|---|---|---|---|
| B1 | **Error bars.** 10 seeds exist for variability but tables print means only. The SDs are already computed — just surface them. | code | 30 min |
| B2 | **Confirm the drift-vs-tile-height trend.** Stale loss falls 27.3 → 11.9 pts from B=32 to B=256. Large and monotonic, unlike the variability trend I retracted, but it was found *after* looking at the data. Verify with 10 seeds and an `--r-sense` sweep, since the proposed mechanism predicts the trend tracks `R_s·G`. | run + 1 session | 3 h |
| B3 | **Sweep `--nu-sigma`.** The refresh conclusion rests on σ_ν = 0.004, my least-certain input. Report the value at which stale calibration becomes viable, rather than asserting one number. | run | 2 h |
| B4 | **Level-mapping error.** Found while building the LLaMA path: the three programmed states do not land on {0, 0.5, 1}. `R_MID/R_LRS = 2.0262`, and finite HRS subtracts 0.25% from every level, so level 0.5 is really 0.491 (−1.79%). This is *separate* from readout error, is currently absorbed into what we call the digital ceiling, and is fixable by trimming R_MID at programming time or folding the measured ratio into the block scale. Quantify it, then decide whether to fix or report. | 1 session | — |
| B5 | **Reprogramming energy in the Pareto.** `hw_model` costs it now, but `codesign_sweep` runs resident-only, so the Pareto never shows it. Either state the assumption or add the column. | 30 min | — |
| B6 | **NeuroSim comparison is VGG8 vs our ResNet-18.** Not like-for-like. Either say so prominently or port ResNet-18 into NeuroSim. | 15 min or 1 session | — |

---

## C. Strengthening — makes it a better paper

| # | Item | Notes |
|---|---|---|
| C1 | **LLaMA results.** `llama_analog_eval.py` is built and self-tested; it has not been *run*. Start with `JackFram/llama-160m` (~0.9 GB of conductance tensors). TinyLlama-1.1B needs ~11.8 GB in fp32, so it needs `--layers 0-5` or fp16 storage on an 8 GB card. |
| C2 | **CIFAR-100 through the analog path.** The checkpoint exists (72.35% QAT); `analog_eval --dataset cifar100` now supports it. Never run. |
| C3 | **ImageNet.** The strongest single credibility upgrade for an architecture venue. Needs the dataset on disk. |
| C4 | **Read noise.** Distinct from programming scatter and drift: it varies per read, so it *averages* over a dot product rather than accumulating. Likely benign, which is worth showing. |
| C5 | **Wordline IR drop.** Currently the wordline is an ideal voltage source. Real drivers have finite resistance and the drop is activation-dependent, so unlike the bitline loading it is **not** a compile-time constant. This is the most likely thing to weaken the contribution and nobody has looked at it. |
| C6 | **Temperature.** Drift and conductance are both strongly temperature-dependent. |

---

## D. Deferred — real work, not needed for this paper

| # | Item | Blocker |
|---|---|---|
| D1 | **RTL / synthesis flow** (Vivado, Yosys, OpenROAD). | `resnet18_reram_top.sv` has no behavioural model at the ADC boundary, which is why `RTL_RelErr%` has read N/A in every run since the start. Fix that first. Even then it characterises the digital periphery, not the array. |
| D2 | **Tapeout-grade area/delay.** | Needs a real PDK. Everything is `[ASSUM]` today. |
| D3 | **Silicon.** | — |

---

## Fastest path to submission

1. A1 (yours, 1 h) — decides how the intro is written.
2. A2 + A3 + A4 (15 min of running) — makes every number in the paper consistent.
3. B1 + B3 (overnight) — error bars and the σ_ν sensitivity.
4. C1 (one run) — LLaMA perplexity, which is what makes a reviewer sit up.

That is roughly a week, most of it compute. B2 and B4 are the two I would add
if there is time, because both are cases where the data is telling us something
we have not fully explained.

---

## Things I got wrong, for the record

Worth keeping visible so the same mistakes are not repeated.

- **"Taller columns are more variability-tolerant."** Retracted. It came from
  1000 images and 3 seeds, where the CI is ±1.9 points and the claimed effect
  was a few tenths. The full test set reversed the sign.
- **"Five ADC bits suffice."** Wrong for the same reason — six are needed.
- **`hw_model`'s array-power fallback was 30× too high**, which made every
  efficiency figure pessimistic until `--power-csv` was wired in.
- **`fig10`'s title was hardcoded** and kept claiming "5 bits" after the answer
  changed to 6. Now derived from the data.
- **Recalibrating the loading constant alone made drift worse, not better** —
  the weight shrinkage dominates, and the fix is a per-column gain folded into
  the block scale.

The pattern in all five: a number that looked settled at small sample size, or
a caption that stopped tracking its data. Re-run anything you intend to quote.
