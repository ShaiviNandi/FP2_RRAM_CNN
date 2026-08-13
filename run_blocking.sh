#!/usr/bin/env bash
# =============================================================================
# run_blocking.sh -- the "before submission" runbook.
#
# Every item from TODO.md sections A (blocking) and B (important) that can be
# executed. A1 (novelty search) is yours and is not here.
#
#   bash run_blocking.sh            list the stages
#   bash run_blocking.sh a2         run one stage
#   bash run_blocking.sh blocking   A2 + A3 + A4       (~20 min)
#   bash run_blocking.sh important  B1..B6             (overnight)
#   bash run_blocking.sh all        everything
#
# Live output: every stage streams to the terminal AND to logs/<stage>.log.
# =============================================================================
set -u
cd "$(dirname "$0")"

ADC_BITS="${ADC_BITS:-6}"          # 6, not 5 -- see TODO A2
DATA="${DATA:-./data}"
CKPT32="${CKPT32:-resnet18_cifar10_fp32.pth}"
POWER_CSV="${POWER_CSV:-ngspice_b128_summary.csv}"
FIGDIR="${FIGDIR:-paper/figures}"
SEEDS="${SEEDS:-10}"
mkdir -p logs "$FIGDIR"

# sed -u keeps the pipe unbuffered; without it output appears only at the end
if echo x | sed -u 's/x/x/' >/dev/null 2>&1; then SED_U="sed -u"; else SED_U="sed"; fi

run() {                       # run <stage> <description> <command...>
  local tag="$1" desc="$2"; shift 2
  echo
  echo "==============================================================="
  echo ">>> $tag : $desc"
  echo ">>> $*"
  echo "==============================================================="
  local t0=$SECONDS
  PYTHONUNBUFFERED=1 "$@" 2>&1 | $SED_U 's/^/   | /' | tee "logs/$tag.log"
  local rc=${PIPESTATUS[0]} dt=$((SECONDS - t0))
  if [ "$rc" -ne 0 ]; then
    echo "!!! $tag FAILED (exit $rc) after ${dt}s -- see logs/$tag.log"
    FAILED="${FAILED:-} $tag"
  else
    echo "<<< $tag done in ${dt}s"
  fi
}

# -----------------------------------------------------------------------------
# A2  Re-run the Pareto and the hardware model at 6-bit.
#     Every efficiency figure currently quoted is either 5-bit or was rescaled
#     by hand using the SAR 2^N law. This measures it directly.
# -----------------------------------------------------------------------------
stage_a2() {
  run a2_pareto "Pareto sweep at ${ADC_BITS}-bit ADC" \
    python3 codesign_sweep.py --blocks 32,64,128,256 \
      --fp32-checkpoint "$CKPT32" --bf16 \
      --power-csv "$POWER_CSV" --adc-bits "$ADC_BITS" \
      --skip-qat --skip-bench \
      --out-csv codesign_pareto.csv

  run a2_hw "hardware model at ${ADC_BITS}-bit ADC" \
    python3 hw_model.py --layers-csv bench_b128.csv --tile-m 128 \
      --adc-bits "$ADC_BITS" --power-csv "$POWER_CSV" \
      --report hw_final.json
}

# -----------------------------------------------------------------------------
# A3  Regenerate figures 1-7 (and 8-13) from the current CSVs.
#     They still carry n=2000 numbers; the CSVs underneath are current.
# -----------------------------------------------------------------------------
stage_a3() {
  run a3_fig17 "figures 1-7 and tables 1-4" \
    python3 make_figures.py --outdir "$FIGDIR"

  run a3_fig813 "figures 8-13 and tables 5-8" \
    python3 figures_sweeps.py --outdir "$FIGDIR"
}

# -----------------------------------------------------------------------------
# A4  RESULTS.md is stale and contradicts REPORT.md. Two documents that
#     disagree are worse than one. Archive it rather than delete, so nothing
#     is lost if a number in it turns out to be the only surviving copy.
# -----------------------------------------------------------------------------
stage_a4() {
  echo
  echo ">>> a4 : retire the stale RESULTS.md"
  if [ -f RESULTS.md ]; then
    mkdir -p archive
    mv RESULTS.md "archive/RESULTS.stale.$(date +%Y%m%d).md"
    echo "   | moved RESULTS.md -> archive/  (REPORT.md is now the single source)"
  else
    echo "   | RESULTS.md already gone -- nothing to do"
  fi
}

# -----------------------------------------------------------------------------
# B1  Error bars. The per-seed SDs were ALREADY being written to the CSV and
#     silently dropped by the table/plot generators; that is now fixed in
#     figures_sweeps.py. This re-runs variability at 10 seeds so the SDs are
#     meaningful, then regenerates. If variability.csv already has 10 seeds,
#     skip straight to stage_a3.
# -----------------------------------------------------------------------------
stage_b1() {
  run b1_variability "variability, ${SEEDS} seeds, full test set" \
    python3 analog_eval.py --sweep 32,64,128,256 \
      --variability 0,0.05,0.1,0.2 --variability-seeds "$SEEDS" \
      --data-dir "$DATA" --max-images 0 --adc-bits "$ADC_BITS" \
      --out-csv variability.csv

  run b1_figs "regenerate with error bars" \
    python3 figures_sweeps.py --outdir "$FIGDIR"
}

# -----------------------------------------------------------------------------
# B3  Sweep the drift exponent spread.
#     The refresh conclusion rests on sigma_nu = 0.004, the least certain
#     input in the whole model. Rather than asserting one value, report the
#     value at which stale calibration stops being viable.
# -----------------------------------------------------------------------------
stage_b3() {
  for NS in 0.001 0.002 0.004 0.008 0.016; do
    run "b3_nusigma_$NS" "drift at nu-sigma=$NS" \
      python3 analog_eval.py --sweep 128 \
        --drift 1,3600,86400,2592000,31536000 --nu-sigma "$NS" \
        --data-dir "$DATA" --max-images 0 --adc-bits "$ADC_BITS" \
        --out-csv "drift_nusigma_$NS.csv"
  done
  echo
  echo ">>> b3 summary: stale-calibration cost against nu-sigma"
  python3 - <<'PY'
import csv, glob, re
print(f"{'nu_sigma':>10}{'t':>8}{'raw':>9}{'stale':>9}{'recalib':>9}{'stale cost':>12}")
print("-" * 57)
for f in sorted(glob.glob("drift_nusigma_*.csv"),
                key=lambda p: float(re.search(r"_([0-9.]+)\.csv", p).group(1))):
    ns = re.search(r"_([0-9.]+)\.csv", f).group(1)
    for r in csv.DictReader(open(f)):
        if r.get("t_label") in ("1y", "1yr", "31536000.0"):
            print(f"{ns:>10}{r.get('t_label',''):>8}{float(r['raw']):>9.2f}"
                  f"{float(r['stale']):>9.2f}{float(r['recalib']):>9.2f}"
                  f"{float(r['stale_cost']):>+12.2f}")
PY
}

# -----------------------------------------------------------------------------
# B2  Confirm the drift-vs-tile-height trend with an R_sense sweep.
#     Stale loss falls 27.3 -> 11.9 pts from B=32 to B=256. The proposed
#     mechanism predicts the trend tracks R_s*G, so sweeping R_s is the test:
#     if the trend is real it should scale with R_s, and if it is an artefact
#     it will not move.
# -----------------------------------------------------------------------------
stage_b2() {
  for RS in 5 10 20 40 80; do
    run "b2_rsense_$RS" "drift vs tile height at R_s=${RS} ohm" \
      python3 analog_eval.py --sweep 32,64,128,256 \
        --drift 31536000 --r-sense "$RS" \
        --data-dir "$DATA" --max-images 0 --adc-bits "$ADC_BITS" \
        --out-csv "drift_rsense_$RS.csv"
  done
}

# -----------------------------------------------------------------------------
# B4  Quantify the level-mapping error.
#     The three programmed states do not land on {0, 0.5, 1}:
#     R_MID/R_LRS = 2.0262 (not 2), and finite HRS subtracts ~0.25% from every
#     level. This is SEPARATE from the readout error the paper fixes, and is
#     currently absorbed into the reported digital ceiling. Pure analysis,
#     no GPU needed.
# -----------------------------------------------------------------------------
stage_b4() {
  echo
  echo ">>> b4 : level-mapping error"
  PYTHONUNBUFFERED=1 python3 - 2>&1 | $SED_U 's/^/   | /' | tee logs/b4_levelmap.log <<'PY'
import analog_eval as ae
R_L, R_M, R_H = ae.R_LRS, ae.R_MID, ae.R_HRS
G_L, G_M, G_H = 1/R_L, 1/R_M, 1/R_H
full = G_L - G_H                      # the +1 level, differentially
print(f"R_LRS={R_L}  R_MID={R_M}  R_HRS={R_H}")
print(f"on/off ratio        {R_H/R_L:.1f}")
print(f"R_MID / R_LRS       {R_M/R_L:.4f}   (should be exactly 2)")
print()
print(f"{'level':>7}{'ideal':>10}{'actual':>12}{'error %':>11}")
print("-" * 40)
worst = 0.0
for name, ideal, g in (("+1.0", 1.0, G_L - G_H),
                       ("+0.5", 0.5, G_M - G_H),
                       (" 0.0", 0.0, G_H - G_H)):
    act = g / full
    err = 100 * (act - ideal) / (ideal if ideal else 1)
    worst = max(worst, abs(err) if ideal else 0.0)
    print(f"{name:>7}{ideal:>10.4f}{act:>12.5f}{err:>+11.3f}")
print("-" * 40)
print(f"worst level error   {worst:.3f}%")
print()
# Want (G_M - G_H)/(G_L - G_H) = 0.5 exactly, so G_M = G_H + 0.5*(G_L - G_H).
G_M_target = G_H + 0.5 * (G_L - G_H)
R_M_target = 1.0 / G_M_target
print(f"Fix A: trim R_MID at programming time.")
print(f"        target R_MID = {R_M_target:.2f} ohm   (currently {R_M})")
print(f"        i.e. a {100*(R_M_target-R_M)/R_M:+.2f}% trim")
# verify the trim actually removes the error
chk = (G_M_target - G_H) / (G_L - G_H)
assert abs(chk - 0.5) < 1e-12, chk
print(f"        verified: trimmed level 0.5 lands at {chk:.12f}")
print("Fix B: fold the measured ratio into the block scale -- zero hardware cost,")
print("       and it also absorbs any per-chip trim error.")
PY
}

# -----------------------------------------------------------------------------
# B6  NeuroSim: re-run at the correct on/off ratio, and label the workload.
#     The C++ side already carries the device parameters; the wrapper has its OWN
#     --onoffratio which was still at its default of 10 against the actual 403. That
#     is why TOPS/W barely moved (29.3 -> 30.3) after the "conversion".
# -----------------------------------------------------------------------------
stage_b6() {
  local NS="${NEUROSIM_DIR:-../DNN_NeuroSim_V1.4}"
  if [ ! -d "$NS" ]; then
    echo "!!! b6 skipped: NeuroSim not found at $NS"
    echo "    set NEUROSIM_DIR=/path/to/DNN_NeuroSim_V1.4 and re-run"
    return
  fi
  echo
  echo ">>> b6 : verify the C++ device parameters actually took"
  grep -n "resistanceOn\s*=\|resistanceOff\s*=" "$NS"/Inference_pytorch/NeuroSIM/Param.cpp \
    | $SED_U 's/^/   | /'
  echo "   | ^^ both must read 542.8 and 218587.2 with NO trailing multiplier"

  run b6_neurosim "NeuroSim at on/off ratio 403" \
    python3 "$NS"/Inference_pytorch/inference.py \
      --dataset cifar10 --model VGG8 --mode WAGE \
      --cellBit 2 --subArray 128 --ADCprecision "$ADC_BITS" \
      --onoffratio 403
  echo "   | NOTE: this is VGG8, the reference numbers are ResNet-18 -- NOT like-for-like."
  echo "   | Label it as such in the paper or port ResNet-18 into NeuroSim."
}

# -----------------------------------------------------------------------------
usage() {
  cat <<'EOF'
Stages:
  a2   Pareto + hardware model at 6-bit ADC          ~15 min   BLOCKING
  a3   Regenerate all figures and tables             ~1 min    BLOCKING
  a4   Retire the stale RESULTS.md                   instant   BLOCKING
  b1   Variability at 10 seeds + error bars          ~2 h      important
  b2   Drift vs tile height, R_sense sweep           ~4 h      important
  b3   Drift exponent spread sweep                   ~3 h      important
  b4   Level-mapping error (analysis only)           instant   important
  b6   NeuroSim at the correct on/off ratio          ~30 min   important

Groups:
  blocking   = a2 a3 a4
  important  = b4 b1 b3 b2 b6      (cheapest first)
  all        = blocking + important

Not scriptable:
  a1   Novelty search  -- yours. Search these three phrases:
         "IR drop compensation ReRAM crossbar"
         "conductance-dependent output scaling CIM"
         "sneak-path calibration in-memory computing"
       It decides whether the intro says "first to" or "unlike prior work".
  a5   Cite the FP2 spec for B=32 -- the whole framing rests on the format
       fixing the block size, so quote the paper directly.

Environment overrides:
  ADC_BITS=6  DATA=./data  SEEDS=10  CKPT32=...  POWER_CSV=...
  FIGDIR=paper/figures  NEUROSIM_DIR=../DNN_NeuroSim_V1.4
EOF
}

case "${1:-}" in
  a2) stage_a2 ;;
  a3) stage_a3 ;;
  a4) stage_a4 ;;
  b1) stage_b1 ;;
  b2) stage_b2 ;;
  b3) stage_b3 ;;
  b4) stage_b4 ;;
  b6) stage_b6 ;;
  blocking)  stage_a2; stage_a3; stage_a4 ;;
  important) stage_b4; stage_b1; stage_b3; stage_b2; stage_b6 ;;
  all)       stage_a2; stage_a3; stage_a4
             stage_b4; stage_b1; stage_b3; stage_b2; stage_b6 ;;
  *) usage; exit 0 ;;
esac

echo
if [ -n "${FAILED:-}" ]; then
  echo "==============================================================="
  echo "FAILED STAGES:${FAILED}"
  echo "logs are in logs/"
  echo "==============================================================="
  exit 1
fi
echo "All requested stages completed."
