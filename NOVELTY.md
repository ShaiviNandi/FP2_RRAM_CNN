# Novelty positioning

## The lead claim: FP2 and three-state ReRAM are a matched pair

This is the contribution to lead with, and it is the one that holds up.

**What is NOT novel:** using ReRAM for compute-in-memory. That is a large,
mature field with silicon results. Any sentence resembling "we use ReRAM for
CIM" will be read as background, not contribution, and stating it as novelty
invites a reviewer to dismiss the rest.

**What IS novel:** the specific claim that FP2-E1M0 is the number format ReRAM
can actually hold, and that this is a structural match rather than a
convenient choice.

The argument, in three steps:

1. **Multi-level ReRAM is reliable at roughly three states.** Programming a
   filament to land repeatably on eight distinguishable conductance levels is
   beyond what the device does; three to four is the practical ceiling. This
   is the binding device constraint on any analog CIM design.
2. **FP2-E1M0's five signed levels need only three conductance magnitudes.**
   The levels are {-1, -0.5, 0, +0.5, +1}. Three magnitudes -- 1, 0.5, 0 --
   and 2T2R differential encoding supplies the sign for free by putting the
   magnitude on whichever bitline carries the polarity.
3. **Therefore FP2 lands exactly on what the device supports.** Not
   approximately, exactly. INT8 would need 256 levels, or eight cells per
   weight. INT4 needs sixteen. Binary throws away the dynamic range that block
   floating point exists to provide.

The format was designed for digital packing efficiency, with no reference to
resistive memory. That it lands precisely on the three-state limit of a ReRAM
cell is the observation this work is built on.

**Why this matters for framing:** a device constraint and a format constraint
that coincide is a co-design result, not an implementation detail. It is also
what makes the rest of the paper inevitable rather than arbitrary --
Section~\ref{sec:collision} follows directly, because once FP2 is the format,
its block size B=32 is fixed, and B=M forces that onto the crossbar.

**How to say it in one sentence:** *"Two-bit block floating point is not one
quantization choice among many for ReRAM CIM -- it is the format the device can
physically hold, because five signed levels require only three conductance
magnitudes once a differential pair supplies the sign."*

**The one caution.** Check that no prior work has already mapped a 2-bit
floating-point or similar five-level format onto three-state 2T2R. Search
"FP2 ReRAM crossbar", "block floating point in-memory computing",
"ternary/five-level ReRAM mapping". If someone has, the novelty moves entirely
to B=M and the calibration, both of which still stand.

---

## How the two novelty claims fit together

They are one story, told in order:

    FP2 is what a three-state ReRAM cell can hold        <- device/format match
      -> so the format is fixed, and with it B = 32      <- format constraint
        -> B = M, so the crossbar tile height is fixed   <- the collision
          -> which forces a tile too short to amortise the ADC
            -> the loading gain is what makes tall tiles fail
              -> and it is exactly correctable            <- the fix

Leading with the format-device match makes every later step follow from a
physical constraint rather than a design preference. That is a much harder
paper to argue with than one that opens on calibration.

---


## The landscape

| Approach | Level | Mechanism | Reported recovery | Cost |
|---|---|---|---|---|
| Weight-centric prediction (WPM) / non-ideality-aware training | algorithmic | pre-distorts the weight matrix offline so hardware output matches the ideal | baseline restored on CIFAR-10, ImageNet | requires retraining |
| **BN layer recalibration** | software post-processing | recomputes BatchNorm running mean/variance to absorb column-current attenuation | **within 0.16% of FP** | retraining-free; needs a validation set |
| **Op-amp virtual ground / TIA sensing** | circuit | holds the bitline at true virtual ground | loading error does not occur | one op-amp per column; area, bandwidth |
| DTCO peripheral compensation | mixed-signal | injects cancellation current at sense nodes | 71% → 2% distortion; 89% → 95% | op-amps / current mirrors |
| BinSparX | algorithmic | sparsification to cut total array current | — | accuracy/sparsity trade |
| Conductance compensation (CC) | circuit | differential op-amp feedback without inverters | — | resistor matching |

Two of these are direct threats.

---

## Threat 1 — the transimpedance amplifier

**The objection.** A TIA holds the bitline at true virtual ground. With
`v_bl = 0` the loading term `1 + R_s·G_col` never arises. The error this work
corrects is a consequence of choosing passive resistive sensing, so a reviewer
can reasonably ask why not simply use the standard circuit fix.

**This is the single most dangerous question and it cannot be deflected.**
Left unaddressed, the work reads as an elaborate solution to a self-inflicted
problem.

**The defensible answer.** Passive sensing is chosen deliberately, and the
trade is quantifiable:

- A TIA is an op-amp per column, with area and static power that scale with
  column count. At `M=128, K=16` the model already puts the ADC at 26.5% of
  tile area with `ADCS_PER_COLUMN = 2`; adding an op-amp per column is a third
  peripheral block competing for the same budget.
- Op-amp bandwidth bounds the read cycle. The passive path settles in
  `tau ≈ 1 ps` against a 12 ns ADC conversion, so sensing is free in the
  latency budget; a TIA would not be.
- The calibration costs **one multiply per column**, folded into the
  block-scale multiply already present in the datapath. Zero additional analog
  hardware.

So the claim is not "loading cannot be fixed in the circuit". It is
**"loading need not be fixed in the circuit"** — the cheapest possible readout
can be used, and exactness recovered digitally for free. That is a design-space
argument, which is appropriate for DATE.

**Required action:** state the TIA alternative in the introduction, cite it,
and give the area/power/bandwidth argument for preferring passive sensing.
Ideally add a row to the efficiency table showing what a per-column op-amp
would cost.

---

## Threat 2 — BatchNorm recalibration

**The objection.** BN recalibration is retraining-free, absorbs systematic
column-current attenuation, and lands within 0.16% of floating point. That is
close to what this work claims, by a much simpler route.

### MEASURED (2026-08, `bn_baseline.csv`, full test set, 6-bit ADC)

| B=M | ideal | raw | BN-recal | per-column | BN gap | per-column gap |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 92.42 | 78.90 | 90.75 | **92.35** | −1.67 | **−0.07** |
| 64 | 92.47 | 57.47 | 90.09 | **92.09** | −2.38 | **−0.38** |
| 128 | 92.81 | 11.53 | 90.62 | **92.42** | −2.19 | **−0.39** |
| 256 | 92.43 | 10.00 | 90.24 | **91.88** | −2.19 | **−0.55** |

**Prediction 2 FAILED and must be dropped.** The argument was that BN
recalibration would degrade with tile height, because a per-channel scalar
cannot capture per-column variation in `G_col` and that variation grows with
M. It does not: the BN gap averages 2.02 pts at B<128 and 2.19 pts at B>=128,
which is flat within noise. Delete this argument from the paper.

**What the measurement gives instead, which is better than the argument:**

1. **Per-column wins decisively at every tile height.** Worst gap 0.55 pts
   against BN's 2.38 -- a factor of 4.3. At B=32 it is 0.07 vs 1.67, a factor
   of 24. This is a cleaner result than a trend would have been, because it
   holds everywhere rather than only at large M.
2. **BN recalibration is a strong baseline and should be reported as one.** It
   lifts B=256 from 10.00% (chance) to 90.24%. Presenting it honestly makes
   the paper more credible, not less -- and it demonstrates that the error
   really is systematic, since a statistical fit can absorb most of it.
3. **The literature figure does not reproduce here.** Prior work reports BN
   recalibration within 0.16% of floating point; this setup gets 1.67-2.38
   pts. The difference is worth a sentence: their attenuation is IR-drop
   induced and roughly uniform, whereas the shared-R_sense gain varies with
   each column's programmed conductance, so a per-channel affine has less to
   grip. State the discrepancy rather than quietly using the favourable number.
4. **Exactness, no calibration data, and no BatchNorm requirement all stand
   untouched.** Those were always the stronger three.

**One detail worth explaining before a reviewer asks.** The per-column gap
grows slightly with M (0.07 -> 0.55 pts). That is *not* calibration residual,
which is exact -- it is 6-bit ADC quantization, whose relative cost grows as
the partial-sum range widens with tile height. The ADC sweep in the main
results shows the same effect.

**Required action: DONE.** The baseline is run and reported above. The
remaining write-up task is to include the table and to state plainly that the
tile-height prediction was tested and did not hold.

---

## Threat 3 — three more compensation families, from the non-ideality survey

**First, on the source.** `references/08_uncited_or_unmined/ReRAM Crossbar
Non-Ideality Mitigation.pdf` is **not a citable paper.** It has no author, no
venue, and its bibliography is ResearchGate and aggregator links. It is a
generated literature synthesis. Its value is as a **map of prior art**: chase
the primary sources it points at and cite those. Do not cite the document.

Three families it surfaces that were not previously on the threat list. Two of
them are closer to this work than BatchNorm was.

### 3a. SW-2T2R common-mode cancellation — the nearest architectural neighbour

The Sign-Weighted 2T2R architecture uses two differential ReRAM cells per
weight with sign-dependent activation control. When the positive and negative
conductances are matched, the net differential source-line current approaches
zero; this **common-mode reduction suppresses the cumulative voltage drop along
array lines**, and the chip reports 78.4 TOPS/W.

**Why this is the sharpest threat yet.** It is 2T2R, it is differential, and it
reduces a loading-type error by exploiting the differential structure — three
properties this work also claims. A reviewer who knows this chip will ask what
is left.

**The distinction to draw, and it is real.** SW-2T2R reduces the *magnitude* of
the common-mode current so the residual drop is small. This work does not
reduce the current at all; it measures the resulting per-column gain
`c_j = 1 + R_s·ΣG_ij` and divides it out **before** the differential
subtraction, which is exact rather than approximate. The two are complementary,
not competing: SW-2T2R shrinks the error, per-column calibration removes what
remains. Say so explicitly and cite them.

**Action: find and cite the primary source** — "A Fully Integrated Analog ReRAM
Based 78.4 TOPS/W Compute-In-Memory Chip with Fully Parallel MAC Computing",
ISSCC 2020, paper 33.2. **Blocking.**

### 3b. Weight pre-distortion (WPM / S-WPM)

Non-ideality-aware training that transforms target weights into a pre-distorted
matrix, so the physical crossbar output under non-idealities matches the ideal
result. Reported to accelerate offline training by orders of magnitude against
SPICE-in-the-loop.

**Distinction.** WPM folds the correction into the weights at training time and
therefore needs retraining and a fixed assumed non-ideality. Per-column
calibration is computed from the *programmed conductances after write*, needs
no retraining, and tracks the device as programmed. That difference matters
most under variability and drift, where the training-time assumption goes
stale — which is exactly the regime this paper measures.

**Action: cite as related work.** Primary source is a KAUST paper, "Mitigating
the Impact of ReRAM I-V Nonlinearity and IR Drop via Fast Offline Network
Training". Not blocking, but strengthens positioning.

### 3c. Circuit-level current injection (DTCO)

Peripheral circuits that model the linear relation between column current and
parasitic drop, then inject a cancelling current at the sensing node. Reported
to cut current distortion from 71% to 2% and restore accuracy from 89% to 95%.

**Distinction.** Same insight — the error is a predictable function of column
current — implemented in analog hardware rather than in the digital domain.
This work's version costs one multiply per column in digital and no added
analog circuitry. That is the trade to state.

### Also worth reading before submission

- **On-chip ADC reference tuning** and the 40 nm MLC-RRAM CIM macro with
  temperature-independent ADC references — relevant to the 6-bit ADC argument.

---

## Threat 4 — Shanbhag's SNR limits, and the experiment it demands

S. K. Roy, A. Patil, N. R. Shanbhag, "Fundamental Limits on the Computational
Accuracy of Resistive Crossbar-based In-memory Architectures", **ISCAS 2022**,
pp. 384–388. Analytical compute-SNR framework for ReRAM, MRAM and FeFET
crossbars, validated in a commercial 22 nm node, benchmarked on ResNet-20 /
CIFAR-10. **Read it in full before submitting.**

### Two of its results corroborate this work

**1. It explains the flat R_LRS sweep from theory.** Shanbhag: *"SNRmax only
depends on the resistive contrast and not on the absolute value of R_on, as
the effect of increasing R_on affects both signal and noise equally."* This
work swept R_LRS over 184× with the on/off ratio **held constant at 403** and
measured calibrated accuracy flat within 0.40 pts. That is exactly what the
theory predicts. **Claim the agreement** — an empirical result matching an
independent analytical prediction is much stronger than either alone.

**2. It corroborates the MRAM failure.** They report resistive contrast of
**2 for MRAM at R_on = 3 kΩ**, and find SNRmax saturates only above a contrast
of **12–15**, below which the device is the limiter. This work tested
R_on = 3 kΩ at ratio 3 and measured chance accuracy. Same device corner, same
conclusion, arrived at independently.

### One result is a direct threat: R_sense

Their **SNR-optimal sensing resistance for ReRAM is R_s\* = 835 Ω** (with
681 Ω and 562 Ω for later layers). This work uses **R_sense = 20 Ω**, and the
sweep in `b2` only covered 5–80 Ω. **A reviewer holding this paper will ask
why the design sits 40× below the SNR optimum.**

**The answer is likely a result rather than a defence, and it has not been
run.** Without calibration a large R_s is unusable: loading error is already
58.6% at R_s = 20 Ω, M = 128, R_LRS = 543 Ω, and it grows with R_s. Per-column
calibration removes that term *exactly*, so the constraint that forces R_s low
should disappear — leaving more signal into the ADC and higher SNR. If that
holds, the framing becomes:

> Per-column calibration does not merely recover accuracy at a fixed operating
> point. It unlocks the SNR-optimal sensing resistance identified by
> independent analysis, which an uncalibrated crossbar cannot use.

That is a stronger claim than anything currently in the paper. **It is also
falsifiable** — the ADC may clip, or quantisation may eat the gain. Run it
before deciding how to frame it.

### Positioning note

Their introduction dismisses prior work as *"either empirical design approaches
or simulation-based and therefore unable to pinpoint the precise limits."*
This work is simulation-based. Do not compete on their ground: cite them as
the analytical framework, and position this work as a **correction method**
evaluated against it, not as a limits analysis.

### Comparison point worth quoting

Their crossbar mapping without retraining lands **within 2%** of an 84.94%
fixed-point ResNet-20 baseline. This work lands within **0.07–0.55 pts** of a
92.42–92.81% baseline. Different network and weight precision, so not a
head-to-head — but it is a fair indication of where the calibrated result sits
relative to published practice.

### Summary of what changed

| | Before | After |
|---|---|---|
| Nearest prior art | BatchNorm recalibration | **SW-2T2R common-mode cancellation** |
| Blocking citations | FP2 spec (closed) | **ISSCC 2020 33.2** |
| Claim to soften | — | "differential structure suppresses loading" — prior art does this too |
| Claim that strengthens | — | exact vs approximate; post-write vs training-time |

None of this breaks the contribution. It narrows the framing from "we exploit
the differential structure" to **"we remove the residual exactly, after write,
without retraining or calibration data"** — which is what the measurements
actually support.

---

## What survives as the contribution

Stated at the width the evidence supports.

1. **The `B = M` coupling.** Block-floating-point block size and CIM tile
   height are one physical parameter. No prior work found treats them jointly,
   and the format-imposed `B=32` is what makes the collision concrete rather
   than hypothetical. **This is the most defensible novelty and should lead.**
2. **An exact, data-free, per-column correction for the shared-sense-resistor
   loading term**, requiring no read-back and folding into an existing
   multiply. Narrower than "IR drop compensation", which is well covered.
3. **Circuit-exact evaluation validated exhaustively against SPICE** — every
   tile of every layer, 87,168 tiles, worst disagreement `3.41×10⁻⁷ %`. The
   literature's additive-Gaussian error models structurally cannot represent a
   systematic gain, which is why this failure mode is invisible to them. The
   survey confirms fast IR-drop models exist (fixed-point iteration, sparse LU
   + Anderson) but they target *modelling* speed, not this coupling.
4. **Transfer to transformers** without BatchNorm, where activation
   statistics-based fixes are unavailable.

---

## What must be dropped or narrowed

- **"First to compensate crossbar loading."** False. Concede immediately.
- **"The analog error is a compile-time constant."** True only of the
  `R_sense` term. The mesh solve shows bitline *wire* resistance leaves 49%
  residual after per-column calibration, and a fitted per-column gain does not
  transfer to unseen activations at `M=256`. Narrow to: *the shared-sense-
  resistor loading term is exactly correctable at compile time; distributed
  wire parasitics are not, and are reported separately.*
- **Efficiency framing.** 81.3 TOPS/W is tile-level. The survey cites a fully
  integrated neuromorphic chip at 78.4 TOPS/W — real silicon, chip-level, and
  essentially the same number. Quoting 81.3 against that without saying it
  excludes interconnect and buffers would not survive review.

---

## Introduction, as it should now read

Not "first to compensate crossbar loading". Instead:

> Crossbar non-ideality compensation is well studied, spanning offline
> non-ideality-aware training, retraining-free BatchNorm recalibration,
> peripheral current injection, and virtual-ground transimpedance sensing.
> This work addresses a coupling those approaches do not: in a
> block-floating-point format the block size and the crossbar tile height are
> the same physical parameter, because a column produces one current that can
> carry only one scale factor. For FP2-E1M0, which fixes B=32, this places the
> format directly in conflict with the tile height that amortises the ADC. We
> show the resulting error is a deterministic function of the programmed
> conductances alone, correctable exactly by one constant per bitline applied
> before the differential subtraction, requiring no read-back, no validation
> data, and no additional analog hardware.

---

## Open items

- [ ] Read the six primary sources rather than the survey. Two matter most:
      *Modeling and Compensation of IR Drop in Crosspoint Accelerators of
      Neural Networks*, and the KAUST WPM/NAT work.
- [ ] Check whether any prior work applies a **per-column** rather than
      per-layer correction. This is the precise boundary of claim 2.
- [ ] Run BN recalibration as a baseline (Threat 2).
- [ ] Add the TIA cost row to the efficiency table (Threat 1).
- [ ] Cite the FP2 specification for `B=32`.
