# What's left

> **Status, current.** Every runnable experiment is done: A2, A3, A4 and
> B1-B7. The novelty search is done and written up in `NOVELTY.md`; it
> narrowed the claim and added two required experiments. What remains is
> writing, plus the two baselines named below.

> Methodology, assumption provenance and sensitivity sweeps are in
> `METHODOLOGY.md`. The BN baseline (N2) is implemented in
> `bn_baseline.py` and needs running.

## Added by the novelty search (see NOVELTY.md)

| # | Item | Why | Effort |
|---|---|---|---|
| N1 | **Address the transimpedance-amplifier objection in the introduction.** A TIA holds the bitline at virtual ground, so the loading term never arises. Without this, the work reads as solving a self-inflicted problem. Argue passive sensing on area, static power and bandwidth. | blocking | 1 h |
| N2 | **Run BatchNorm recalibration as a baseline.** It is retraining-free and reported within 0.16% of FP, so it is the closest competitor. The claim that a per-layer scalar cannot capture per-column variation in `G_col` is currently an argument, not a result. | blocking | 1 session |
| N3 | Add a TIA cost row to the efficiency table. | 30 min |
| N4 | Read the six primary sources, not the survey. Check specifically whether any prior work corrects **per column** rather than per layer. | **You** | 2 h |

## Resolved since the last revision

- **A1 novelty search** -- done, `NOVELTY.md`.
- **A2** -- Pareto and hw_model re-run at 6 bit. **81.3 TOPS/W**, not the
  hand-rescaled 84.4. Efficiency gain is **13.1x**, not 13.6x.
- **A3** -- all figures and tables regenerated.
- **A4** -- stale RESULTS.md archived.
- **B1** -- variability at 10 seeds, full test set. Blind calibration loses
  0.90 pts at sigma=20%. Blind is now consistently *better* than write-verify
  at high sigma across all four tile heights, which strengthens the no-read-back
  claim. Worth understanding why before publishing it.
- **B2** -- **the drift-vs-tile-height mechanism is confirmed.** Stale cost
  shrinks monotonically with R_s at every tile height, and the tile-height
  spread grows from -4.73 pts at R_s=5 to -12.81 at R_s=20. That is what an
  `R_s*G_col` mechanism predicts, so the trend is explained rather than merely
  observed and survives the "found after looking at the data" objection.
- **B3** -- **refresh holds within 2 pts up to sigma_nu = 0.004, and fails at
  0.008** (90.54%, -2.27 vs ceiling; refreshing worth only +2.4 pts at 0.016).
  Report that boundary rather than asserting one value.
- **B4** -- level-mapping error quantified: level 0.5 lands at 0.49228
  (-1.543%). Fix is a -1.54% trim of R_MID to 1082.91 ohm, verified exact.
- **B5** -- wire parasitics isolated. Wordline metal is mild (1.8% at 0.5 ohm);
  **wordline driver (25.4% at 50 ohm) and bitline metal (49.1% at 0.5 ohm) are
  not corrected** by the per-column constant. A fitted per-column gain does not
  transfer to unseen activations at M=256.
- **B6** -- NeuroSim re-run at on/off 403. 30.3 -> **29.29 TOPS/W**, ~3% worse,
  so the half-conversion caveat did not matter. Conclusion unchanged.
- **B7** -- ADC and cell area sourced three ways. TOPS/W spans **81.3 to 142.2**
  depending only on the ADC figure of merit. Report the spread. Weight-reuse
  curve implemented; quote the curve, not reuse=1.

## Two claims that must be narrowed before submission

1. **"The analog error is a compile-time constant"** -> true of the `R_sense`
   term only. Distributed wire parasitics are not compile-time correctable.
2. **"81.3 TOPS/W"** -> tile-level. Chip-level, adding NeuroSim's 20.8%
   interconnect/buffer/pooling share, is nearer 64. A fully integrated
   neuromorphic chip in the literature reports 78.4 TOPS/W at chip level.

---


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
