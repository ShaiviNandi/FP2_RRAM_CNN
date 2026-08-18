#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- regenerate every result in the paper, one stage at a time.
#
# Pasting multi-line commands into a terminal is fragile: a dropped backslash
# silently splits one command into several, and the first fragment often runs
# happily with default arguments instead of the intended ones. That is how a
# sweep ends up at 2000 images with an 8-bit ADC when the request was the full
# test set at 5 bits. Running a file removes that failure mode entirely.
#
# USAGE
#   bash run_all.sh                 # list stages, run nothing
#   bash run_all.sh check           # self-tests only, ~1 min
#   bash run_all.sh pareto          # corrected co-design sweep
#   bash run_all.sh quick           # check + pareto + figures  (~5 min)
#   bash run_all.sh overnight       # accuracy + variability + adc (hours)
#   bash run_all.sh all             # everything, in dependency order
#
# Every stage logs to logs/<stage>.log and prints its own wall time.
# =============================================================================
set -uo pipefail

DATA=${DATA:-./data}
BLOCKS=${BLOCKS:-32,64,128,256}
ADC_BITS=${ADC_BITS:-5}
FP32=${FP32:-resnet18_cifar10_fp32.pth}
POWER_CSV=${POWER_CSV:-ngspice_full_summary.csv}
WORKERS=${WORKERS:-15}
FIGDIR=${FIGDIR:-paper/figures}
PY=${PY:-python3}
SRC=${SRC:-/mnt/c/Users/shaiv/OneDrive - iiit-b/Desktop/IIITB/Projects/FP2_RRAM_CNN}
NEUROSIM=${NEUROSIM:-$HOME/DNN_NeuroSim_V1.4/Inference_pytorch}

mkdir -p logs "$FIGDIR"

# `sed -u` is GNU/BusyBox; BSD sed (macOS) spells it -l and older seds have
# neither. Probe once rather than assume, because falling back silently to a
# buffering sed is exactly the failure this is here to prevent.
if echo x | sed -u 's/x/y/' >/dev/null 2>&1;   then SED_U="sed -u"
elif echo x | sed -l 's/x/y/' >/dev/null 2>&1; then SED_U="sed -l"
else SED_U="cat"; printf '!! sed has no unbuffered mode; output will appear in bursts\n"'; fi

STAGE_N=0
STAGE_TOTAL=0
say()  {
  STAGE_N=$((STAGE_N+1))
  printf '\n\033[1;36m== [%d/%d] %s\033[0m  \033[2m(%s elapsed, started %s)\033[0m\n' \
         "$STAGE_N" "$STAGE_TOTAL" "$*" "$(fmt_dur $((SECONDS-T0)))" "$(date +%H:%M:%S)"
}
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }

# Number of say() calls each stage makes, so the [n/N] counter is honest.
stage_weight() {
  case $1 in
    ngspice) echo 2 ;;   # sweep + final hardware model
    figures) echo 1 ;;
    *)       echo 1 ;;
  esac
}

run() {                      # run <logname> <command...>
  local name=$1; shift
  local t0=$SECONDS
  printf '   $ %s\n' "$*"
  printf '   ---------------- live output (also -> logs/%s.log) ----------------\n' "$name"

  # tee, not redirect: the previous version sent everything to a file, so a
  # six-hour sweep looked like a hung terminal.
  #
  # Every element of this pipeline buffers by default when its output is not a
  # terminal, and any one of them re-hides the progress:
  #   * Python block-buffers to a pipe        -> PYTHONUNBUFFERED=1
  #   * GNU sed block-buffers to a pipe       -> sed -u   (measured: without
  #     it, five lines emitted over 6 s all appeared at once at the end)
  # tee itself is unbuffered, so it needs nothing.
  #
  # PIPESTATUS[0] is the python exit code; $? would be tee's, which is
  # essentially always 0 and would mask every failure.
  PYTHONUNBUFFERED=1 "$@" 2>&1 | $SED_U 's/^/   | /' | tee "logs/$name.log"
  local rc=${PIPESTATUS[0]}

  if [[ $rc -eq 0 ]]; then
    printf '   ok  (%s)  -> logs/%s.log\n' "$(fmt_dur $((SECONDS-t0)))" "$name"
  else
    warn "FAILED after $(fmt_dur $((SECONDS-t0))) (exit $rc) -- full log: logs/$name.log"
    return 1
  fi
}

fmt_dur() {                  # seconds -> "3m 07s"
  local s=$1
  if   (( s < 60 ));   then printf '%ds' "$s"
  elif (( s < 3600 )); then printf '%dm %02ds' $((s/60)) $((s%60))
  else                      printf '%dh %02dm' $((s/3600)) $(((s%3600)/60))
  fi
}

need_file() {
  [[ -f $1 ]] || { warn "missing $1 -- $2"; return 1; }
}

# -----------------------------------------------------------------------------
# Pull the current scripts out of the synced OneDrive folder. Kept as a stage
# so "which version produced this result" has a single answer: whatever `sync`
# last copied. SRC is overridable for anyone whose folder lives elsewhere.
stage_sync() {
  say "Sync scripts from $SRC"
  [[ -d $SRC ]] || die "SRC not found: $SRC  (override with SRC=... bash run_all.sh sync)"
  local n=0
  for f in hw_model.py codesign_sweep.py analog_eval.py benchmark_resnet18.py \
           qat_finetune_fp2.py ngspice_full_sweep.py figures_sweeps.py \
           make_figures.py server.py index.html crossbar_array_test.py \
           column_mac_test.py ngspice_bridge.py run_all.sh \
           REPORT.md RESULTS.md PAPER_DRAFT.md; do
    if [[ -f "$SRC/$f" ]]; then
      cp "$SRC/$f" . && n=$((n+1))
    else
      warn "not in SRC: $f"
    fi
  done
  echo "   copied $n file(s)"
  warn "run_all.sh may have just been replaced -- re-invoke if this run behaves oddly"
}

# -----------------------------------------------------------------------------
stage_check() {
  say "Self-tests"
  run selftest_hw       $PY hw_model.py       --self-test || return 1
  run selftest_codesign $PY codesign_sweep.py --self-test || return 1
  run selftest_analog   $PY analog_eval.py    --self-test || return 1
  run selftest_figs     $PY figures_sweeps.py --self-test || return 1
  if command -v ngspice >/dev/null; then
    run selftest_ngspice $PY ngspice_full_sweep.py --self-test || return 1
  else
    warn "ngspice not on PATH -- skipping its self-test"
  fi
  say "All self-tests passed"
}

# -----------------------------------------------------------------------------
# Corrected co-design sweep. --power-csv and --adc-bits are the two flags whose
# absence silently produced the pessimistic numbers in the first submission
# draft, so they are wired in here rather than left to the caller.
stage_pareto() {
  say "Co-design sweep, ADC ${ADC_BITS}b, measured array power"
  local extra=()
  if need_file "$POWER_CSV" "array power will fall back to an empirical estimate"; then
    extra+=(--power-csv "$POWER_CSV")
  fi
  local skip=()
  if [[ -f qat_b32.pth && -f bench_b32.csv ]]; then
    skip=(--skip-qat --skip-bench)
    say "  reusing existing checkpoints and benchmark CSVs (seconds, not ~1 h)"
  fi
  run pareto $PY codesign_sweep.py --blocks "$BLOCKS" \
      --fp32-checkpoint "$FP32" --bf16 --adc-bits "$ADC_BITS" \
      "${extra[@]}" "${skip[@]}" --out-csv codesign_pareto.csv
}

# -----------------------------------------------------------------------------
# The credibility item. At n=1000 the 95% CI is about +-1.9 points and several
# reported gaps are smaller than that; --max-images 0 uses all 10000.
stage_accuracy() {
  say "End-to-end analog accuracy, FULL test set"
  run accuracy $PY analog_eval.py --sweep "$BLOCKS" --data-dir "$DATA" \
      --max-images 0 --out-csv analog_accuracy.csv
}

stage_variability() {
  say "Device variability, full test set, 10 seeds"
  run variability $PY analog_eval.py --sweep "$BLOCKS" \
      --variability 0,0.05,0.1,0.2 --variability-seeds 10 \
      --data-dir "$DATA" --max-images 0 --out-csv variability.csv
}

stage_drift() {
  say "Conductance drift, full test set"
  run drift $PY analog_eval.py --sweep "$BLOCKS" \
      --drift 1,3600,86400,2592000,31536000 --drift-seeds 3 \
      --data-dir "$DATA" --max-images 0 --out-csv drift.csv
}

stage_adc() {
  say "ADC resolution sweep, full test set"
  run adc $PY analog_eval.py --sweep "$BLOCKS" --adc-sweep 4,5,6,8,10 \
      --data-dir "$DATA" --max-images 0 --out-csv adc_sweep.csv
}

# -----------------------------------------------------------------------------
# ngspice at the tile height actually proposed. --cifar-arch and --num-classes
# are mandatory: without them the checkpoint's 10-way head hits a 1000-way
# model and torch raises a size mismatch that strict=False does NOT suppress.
stage_ngspice() {
  command -v ngspice >/dev/null || die "ngspice not on PATH (sudo apt-get install -y ngspice)"
  need_file qat_b128.pth "run stage 'pareto' first" || return 1
  say "Exhaustive ngspice at tile_m=128"
  run ngspice $PY ngspice_full_sweep.py --checkpoint qat_b128.pth \
      --cifar-arch --num-classes 10 --calib-dataset cifar10 --calib-dir "$DATA" \
      --skip-first-last --tile-m 128 --max-positions 32 --workers "$WORKERS" \
      --out-jsonl ngspice_b128.jsonl --out-csv ngspice_b128_summary.csv

  say "Final hardware model on the measured array power"
  run hw_final $PY hw_model.py --layers-csv bench_b128.csv --tile-m 128 \
      --adc-bits "$ADC_BITS" --power-csv ngspice_b128_summary.csv \
      --physical-tiles 256 --report hw_final.json
}

# -----------------------------------------------------------------------------
stage_figures() {
  say "Figures and LaTeX tables -> $FIGDIR"
  run figures_main   $PY make_figures.py    --outdir "$FIGDIR"
  run figures_sweeps $PY figures_sweeps.py  --outdir "$FIGDIR"
  echo
  ls -1 "$FIGDIR" | sed 's/^/   /'
}

# -----------------------------------------------------------------------------
stage_cifar100() {
  say "CIFAR-100 breadth"
  run c100_qat $PY qat_finetune_fp2.py --dataset cifar100 --data-dir "$DATA" \
      --download --pretrain-epochs 30 --epochs 10 --batch-size 256 --bf16 \
      --block-size 128 --save-fp32-checkpoint resnet18_c100_fp32.pth \
      --out-checkpoint qat_c100_b128.pth --log-csv qat_c100_history.csv
  warn "analog_eval's loader is CIFAR-10 only; CIFAR-100 analog eval needs a patch"
}

# -----------------------------------------------------------------------------
# NeuroSim at this device's parameters. It ships with on/off ratio 10 against
# 403 (218587.2 / 542.8), and that ratio governs how much a nominally-off cell
# leaks into the bitline -- the single most likely source of disagreement
# between the two models. The sed patterns are verified against the file before
# rebuilding, because a silently-missed substitution would hand back the
# default-parameter numbers with no indication anything went wrong.
stage_neurosim() {
  local NS=${NEUROSIM:-$HOME/DNN_NeuroSim_V1.4/Inference_pytorch}
  [[ -d $NS/NeuroSIM ]] || die "NeuroSim not at $NS (override with NEUROSIM=...)"
  say "NeuroSim at R_on=542.8, R_off=218587.2 (on/off 403)"

  local P="$NS/NeuroSIM/Param.cpp"
  cp "$P" "$P.bak.$(date +%s)"
  # Consume everything up to the ';', not just a numeric literal. NeuroSim
  # ships `resistanceOff = 6000*10;` -- an expression -- so a pattern that
  # matched only digits would rewrite the 6000 and leave the *10 behind,
  # silently producing a value 10x too large.
  sed -i 's/resistanceOn[[:space:]]*=[^;]*/resistanceOn = 542.8/' "$P"
  sed -i 's/resistanceOff[[:space:]]*=[^;]*/resistanceOff = 218587.2/' "$P"

  echo "   verifying both substitutions actually landed:"
  grep -nE "resistanceOn|resistanceOff" "$P" | head -4 | sed 's/^/      /'
  grep -q "resistanceOn = 542.8;"     "$P" || die "resistanceOn not set -- edit $P by hand"
  grep -q "resistanceOff = 218587.2;" "$P" || die "resistanceOff not set -- edit $P by hand"
  echo "   on/off ratio now $(python3 -c 'print(round(218587.2/542.8))')"

  run neurosim_build make -C "$NS/NeuroSIM" -j"$(nproc)" || return 1
  # --onoffratio is a SEPARATE parameter from Param.cpp's resistanceOn/Off.
  # The C++ side drives the circuit model; this Python flag drives the
  # wrapper's own weight-mapping and noise model. Editing only Param.cpp
  # leaves the run half-converted -- which is what happened first time, and
  # why TOPS/W barely moved (29.3 -> 30.3) despite a 40x change in the ratio.
  local ONOFF
  ONOFF=$(python3 -c 'print(round(218587.2/542.8))')
  ( cd "$NS" && run neurosim_infer $PY inference.py --dataset cifar10 \
      --model VGG8 --mode WAGE --inference 1 --cellBit 1 \
      --subArray 128 --ADCprecision "$ADC_BITS" --onoffratio "$ONOFF" )
  warn "NeuroSim runs VGG8, not ResNet-18 -- it is a cross-check on the "
  warn "peripheral models, not a like-for-like comparison. Say so in the paper."
}

# -----------------------------------------------------------------------------
usage() {
  cat <<'EOF'
stages (in dependency order):
  sync         copy current scripts out of the OneDrive folder    seconds
  check        self-tests for every script                        ~1 min
  pareto       corrected co-design sweep (reuses checkpoints)     ~1 min / ~1 h
  accuracy     end-to-end analog accuracy, FULL test set          ~1.5 h
  variability  device variability, full set, 10 seeds             ~6 h
  adc          ADC resolution sweep, full set                     ~2 h
  drift        conductance drift over 1s..1y, full set           ~3 h
  ngspice      exhaustive ngspice at tile_m=128 + final hw model  ~30 min
  figures      all figures and LaTeX tables                       seconds
  cifar100     CIFAR-100 training (analog eval not yet supported) ~30 min
  neurosim     rebuild NeuroSim at on/off=403 and run it          ~5 min

groups:
  quick        sync check pareto figures
  overnight    accuracy variability adc drift
  all          everything above, in dependency order

env overrides:
  DATA BLOCKS ADC_BITS FP32 POWER_CSV WORKERS FIGDIR PY SRC NEUROSIM
EOF
}

[[ $# -eq 0 ]] && { usage; exit 0; }

T0=$SECONDS

# Expand groups so the [n/N] counter knows how many stages are coming.
expand() {
  case $1 in
    quick)     echo sync check pareto figures ;;
    overnight) echo accuracy variability adc drift ;;
    all)       echo sync check pareto accuracy variability adc drift ngspice figures \
                    cifar100 neurosim ;;
    *)         echo "$1" ;;
  esac
}
PLAN=()
for arg in "$@"; do PLAN+=($(expand "$arg")); done
for st in "${PLAN[@]}"; do STAGE_TOTAL=$((STAGE_TOTAL + $(stage_weight "$st"))); done

printf '\033[1mPlan:\033[0m %s\n' "${PLAN[*]}"
printf 'Logs: ./logs/    Figures: %s/    Started %s\n' "$FIGDIR" "$(date)"

for arg in "$@"; do
  case $arg in
    sync|check|pareto|accuracy|variability|adc|drift|ngspice|figures|cifar100|neurosim)
               "stage_$arg" ;;
    quick)     stage_sync && stage_check && stage_pareto && stage_figures ;;
    overnight) stage_accuracy; stage_variability; stage_adc; stage_drift ;;
    # Independent stages are separated by ';' not '&&': cifar100 and neurosim
    # are additive breadth, and a missing NeuroSim checkout should not discard
    # the hours of sweeps that already succeeded.
    all)       stage_sync && stage_check && stage_pareto && stage_accuracy && \
               stage_variability && stage_adc && stage_drift && \
               stage_ngspice && stage_figures
               stage_cifar100
               stage_neurosim ;;
    *) die "unknown stage '$arg'" ;;
  esac
done
printf '\n\033[1;32m== done: %d stage(s) in %s\033[0m  (finished %s)\n' \
       "$STAGE_N" "$(fmt_dur $((SECONDS-T0)))" "$(date +%H:%M:%S)"
printf 'Logs in ./logs/  ·  figures in %s/\n' "$FIGDIR"
