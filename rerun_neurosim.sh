#!/usr/bin/env bash
# =============================================================================
# rerun_neurosim.sh
# =============================================================================
# Regenerate every artefact that depends on NeuroSim's hardware model.
#
# SCOPE
# -----
# NeuroSim's hardware ADC resolution comes from `levelOutput` in Param.cpp, not
# from the Python --ADCprecision flag. Changing levelOutput invalidates every
# NeuroSim-derived number and nothing else.
#
# Unaffected, because they use the local simulator and its own ADC model:
#   accuracy sweeps, ADC-bit sweep, variability, drift, wire parasitics,
#   BatchNorm baseline, R_sense sweep, R_LRS accuracy columns, hw_model.py,
#   vonneumann_baseline.py
#
# Affected:
#   cmp_fp2.csv, cmp_mram.csv, rram_ron_*.csv, the NeuroSim columns of
#   rlrs_tradeoff.csv, threeway*.csv (derived from cmp_fp2.csv), and the
#   tile-height logs
#
# USAGE
#   bash rerun_neurosim.sh --check     # one run, confirm levelOutput is live
#   bash rerun_neurosim.sh             # full regeneration
# =============================================================================
set -u
cd "$(dirname "$0")" 2>/dev/null || true
NS="${NEUROSIM_DIR:-$HOME/DNN_NeuroSim_V1.4}"
PARAM="$NS/Inference_pytorch/NeuroSIM/Param.cpp"
mkdir -p logs

lvl=$(grep -oP 'levelOutput\s*=\s*\K[0-9]+' "$PARAM" 2>/dev/null | head -1)
if [ -z "${lvl:-}" ]; then
    echo "Could not read levelOutput from $PARAM"
    exit 1
fi
bits=$(python3 -c "import math;print(int(math.log2($lvl)))")
echo "levelOutput = $lvl  ->  ${bits}-bit hardware ADC"
echo

if [ "${1:-}" = "--check" ]; then
    # Confirm the parameter is live before spending an hour on the rest.
    # If ADC area does not move between two levelOutput values, the hardware
    # resolution is set somewhere else and the rest of this script is pointless.
    echo "Single run at 128 rows, compared against the 5-bit reference of"
    echo "2.054 mm2 of ADC area."
    python3 neurosim_compare.py --cells rram --subarray 128 --adc-bits "$bits" \
        --resistance-on 6000 --cell-width 0 --wl-weight 2 \
        2>&1 | tee logs/levelcheck.log
    adc=$(grep -oP 'ADC / sense \(mm2\)\s+\K[0-9.]+' logs/levelcheck.log | tail -1)
    echo
    if [ -z "${adc:-}" ]; then
        echo "VERDICT: could not read ADC area. Inspect logs/levelcheck.log."
        exit 1
    fi
    same=$(python3 -c "print(abs($adc-2.054) < 0.01)")
    if [ "$same" = "True" ]; then
        echo "VERDICT: ADC area still $adc mm2. levelOutput is NOT the live"
        echo "parameter. Do not re-run anything until the live one is found."
        exit 1
    fi
    echo "VERDICT: ADC area $adc mm2 against 2.054 at 5 bits. levelOutput is"
    echo "live. Re-run without --check."
    exit 0
fi

run () {          # $1 = label, rest = args to neurosim_compare
    local tag="$1"; shift
    echo "=== $tag ==="
    python3 neurosim_compare.py "$@" 2>&1 | tee "logs/rerun_${tag}.log"
    [ -f neurosim_rram.log ] && cp neurosim_rram.log "logs/raw_${tag}.log"
    [ -f neurosim_sram.log ] && cp neurosim_sram.log "logs/raw_${tag}_sram.log"
}

# 1. Technology comparison -- feeds the three-way table
run cmp_fp2 --cells sram,rram --resistance-on 6000 --cell-width 0 \
    --wl-weight 2 --out-csv cmp_fp2.csv

# 2. Tile-height sweep -- the area argument
for rows in 32 64 128; do
    run "b${rows}" --cells rram --subarray "$rows" --adc-bits "$bits" \
        --resistance-on 6000 --cell-width 0 --wl-weight 2 \
        --out-csv "cmp_b${rows}.csv"
done

# 3. Low-contrast device
run mram --cells rram --resistance-on 3000 --resistance-off 9000 \
    --onoffratio 3 --wl-weight 2 --cell-width 24 --out-csv cmp_mram.csv

# 4. Device resistance sweep, hardware only
for R in 6000 20000 50000 100000; do
    run "ron_${R}" --cells rram --resistance-on "$R" --cell-width 0 \
        --wl-weight 2 --out-csv "rram_ron_${R}.csv"
done

# 5. Derived tables. No simulation, seconds.
python3 vonneumann_baseline.py --horowitz --reuse-sweep 1,2,5,10,100,1000 \
    --neurosim-csv cmp_fp2.csv --out-csv threeway_cited.csv \
    2>&1 | tee logs/rerun_threeway.log

python3 neurosim_area_breakdown.py \
    --log logs/raw_b32.log --log logs/raw_b64.log --log logs/raw_b128.log \
    --names "32 rows,64 rows,128 rows" 2>&1 | tee logs/rerun_area.log

echo
echo "============================================================"
echo "Regenerated at ${bits}-bit. Still to redo by hand:"
echo "  rlrs_tradeoff.py --accuracy --neurosim   (hardware columns only;"
echo "                                            accuracy columns are local"
echo "                                            and remain valid)"
echo "  make_all_figures.py                      (reads the CSVs above)"
echo "============================================================"
