# Every command, in the order you would run them

Copy-paste safe. Each block is self-contained and sets its own variables, so a
fresh terminal works without remembering anything from earlier.

---

## 0. Setup — run once per new terminal

```bash
cd ~/fp2_reram/python
export S="/mnt/c/Users/shaiv/OneDrive - iiit-b/Desktop/IIITB/Projects/FP2_RRAM_CNN"
export NEUROSIM_DIR=~/DNN_NeuroSim_V1.4
```

Make it permanent so you never chase an empty `$S` again:

```bash
echo 'export S="/mnt/c/Users/shaiv/OneDrive - iiit-b/Desktop/IIITB/Projects/FP2_RRAM_CNN"' >> ~/.bashrc
echo 'export NEUROSIM_DIR=~/DNN_NeuroSim_V1.4' >> ~/.bashrc
```

Sync every script from the shared folder:

```bash
cd ~/fp2_reram/python
cp "$S"/*.py "$S"/*.sh . 2>/dev/null
ls -la *.py | tail -20
```

---

## 1. Sanity — run these first, they take seconds

Every simulator asserts agreement with its own reference. If any fails, stop:
something downstream is wrong.

```bash
python3 analog_eval.py --self-test
python3 wordline_ir.py --self-test
python3 bn_baseline.py --self-test
python3 vonneumann_baseline.py --self-test
python3 rlrs_tradeoff.py --self-test
python3 make_all_figures.py --self-test
python3 figures_sweeps.py --self-test
```

---

## 2. The one re-run that is actually needed

The device sweep is the paper's headline figure and was last run at 2000
images. Everything else is current at full test set.

```bash
cd ~/fp2_reram/python
mkdir -p logs
nohup python3 -u rlrs_tradeoff.py --accuracy --neurosim --max-images 0 \
  --out-csv rlrs_tradeoff.csv > logs/rlrs_full.log 2>&1 &
sleep 3; ls -la logs/ ; tail -f logs/rlrs_full.log
```

`-u` is not optional. Python buffers stdout when it is a file rather than a
terminal, so without it the log stays empty for minutes and the run looks
dead. `sleep 3` before `tail` avoids racing the redirect.

Expect a few hours. Watch for the guard message — if the digital baseline
lands at chance and the on/off ratio is healthy, a constant is stale and the
point must be discarded rather than reported.

---

## 2b. The one experiment that is NOT yet run

Shanbhag et al. (ISCAS 2022) put the SNR-optimal sensing resistance for ReRAM
at **835 Ω**. This design uses **20 Ω**, and the `b2` sweep only reached 80 Ω.
The claim to test: per-column calibration removes the loading term exactly, so
the constraint forcing R_sense low disappears and the SNR-optimal value becomes
reachable.

```bash
cd ~/fp2_reram/python && mkdir -p logs
for RS in 20 100 200 400 600 835 1200; do
  echo "=== R_sense = $RS ohm ==="
  python3 analog_eval.py --sweep 128 --data-dir ./data --max-images 0 \
    --adc-bits 6 --r-sense $RS --out-csv rsense_hi_$RS.csv
done 2>&1 | tee logs/rsense_snr_optimum.log
```

Read the **calibrated** column. Three outcomes, all publishable:

- rises toward 835 Ω → calibration unlocks the SNR optimum. Strongest result
  in the paper; reframe accordingly.
- flat → calibration makes accuracy independent of R_sense, which is itself
  the exactness claim demonstrated on a second axis.
- falls → the ADC is clipping. Re-run with `--adc-bits 8` to separate
  calibration residual from converter range before drawing a conclusion.

---

## 3. Open questions worth answering

### Where NeuroSim's missing 70% of chip area goes

```bash
grep -n "chipArea\s*=\|chipArea +=" \
  $NEUROSIM_DIR/Inference_pytorch/NeuroSIM/Chip.cpp | head -30
```

If terms appear that are not printed in the summary, name them in the paper
instead of calling it a tool limitation.

### Confirm the MRAM result in your own simulator

NeuroSim's accuracy uses its own quantiser and must never be quoted. This uses
the FP2 mapping.

```bash
python3 -c "
import sys, analog_eval as ae
ae.R_LRS=3000; ae.R_MID=3000*1099.8/542.8; ae.R_HRS=9000; ae.G_LRS=1/3000
sys.argv=['analog_eval','--sweep','128','--data-dir','./data',
          '--max-images','0','--adc-bits','6']
ae.main()" 2>&1 | tail -12
```

Expect chance. At on/off ratio 3 a weight of +1.0 reads as 0.667 and +0.5 as
0.160, so the format is unrepresentable. That collapse IS the result.

---

## 4. Regenerate everything from the CSVs

No simulation, seconds to run. Do this after any sweep.

```bash
cd ~/fp2_reram/python
python3 make_figures.py --outdir paper/figures        # fig1-7,  tab1-4
python3 figures_sweeps.py --outdir paper/figures      # fig8-13, tab5-8
python3 make_all_figures.py --outdir paper/figures    # figA-F, the comparisons
python3 drift_summary.py --latex                      # tab9, tab10
ls paper/figures/*.pdf | wc -l
```

---

## 5. Full pipeline, if you ever need it from scratch

```bash
bash run_blocking.sh                # lists stages, runs nothing
bash run_blocking.sh blocking       # a2 a3 a4        ~20 min
bash run_blocking.sh important      # b1-b7           overnight
```

Individual stages: `a2 a3 a4 b1 b2 b3 b4 b5 b6 b7`.

---

## 6. Comparison runs

### Technology comparison, FP2 precision

```bash
python3 neurosim_compare.py --cells sram,rram \
  --resistance-on 6000 --cell-width 0 --wl-weight 2 \
  --out-csv cmp_fp2.csv
```

`--wl-weight 2` is essential. NeuroSim defaults to 8-bit weights, which adds
four cells per weight and a shift-add tree that FP2 does not have.

### MRAM-like device

```bash
python3 neurosim_compare.py --cells rram \
  --resistance-on 3000 --resistance-off 9000 --onoffratio 3 \
  --wl-weight 2 --cell-width 24 --out-csv cmp_mram.csv
```

`--cell-width 24` is required: at 3 kΩ the select transistor needs 23.03 F and
NeuroSim refuses at the 12 F default.

### Device resistance sweep, hardware only

```bash
for R in 6000 20000 50000 100000; do
  python3 neurosim_compare.py --cells rram --resistance-on $R \
    --cell-width 0 --wl-weight 2 --out-csv rram_ron_$R.csv
done
```

### Von Neumann baseline and the three-way comparison

Run both. `--horowitz` is the version to put in the paper; the default is the
optimistic floor and is worth reporting as "even under these assumptions".

```bash
# cited constants -- Horowitz ISSCC 2014 Fig. 1.1.9
python3 vonneumann_baseline.py --horowitz --reuse-sweep 1,2,5,10,100,1000 \
  --neurosim-csv cmp_fp2.csv --out-csv threeway_cited.csv

# legacy optimistic floor, for the sensitivity statement
python3 vonneumann_baseline.py --reuse-sweep 1,2,5,10,100,1000 \
  --neurosim-csv cmp_fp2.csv --out-csv threeway.csv
```

### FP2 versus INT8 on the memory wall

```bash
python3 vonneumann_baseline.py --bits 8 --buffer-mb 4 --reuse-sweep 1,10,100
python3 vonneumann_baseline.py --bits 2 --buffer-mb 4 --reuse-sweep 1,10,100
```

INT8 misses the buffer 64% of the time; FP2 fits entirely. Horowitz's
contemporary DRAM figure doubles the gap:

```bash
python3 vonneumann_baseline.py --bits 8 --dram-pj-per-bit 20 --reuse-sweep 1
```

---

## 7. Substituting the cited ADC numbers

The figure of merit is bracketed by real silicon and can stay at 20 fJ. The
area is the uncertain one, so run both bounds and report the range.

```bash
# pessimistic: Tai ISSCC 2014 scaled 40nm -> 65nm, no bit scaling
python3 hw_model.py --layers-csv bench_b128.csv --tile-m 128 --adc-bits 6 \
  --power-csv ngspice_b128_summary.csv \
  --set ADC_AREA_MM2=0.0172 --report hw_adc_pessimistic.json

# optimistic: same, plus 2^(6-10) capacitor-array scaling
python3 hw_model.py --layers-csv bench_b128.csv --tile-m 128 --adc-bits 6 \
  --power-csv ngspice_b128_summary.csv \
  --set ADC_AREA_MM2=0.0011 --report hw_adc_optimistic.json
```

---

## 8. Individual experiments

```bash
# headline accuracy sweep
python3 analog_eval.py --sweep 32,64,128,256 --data-dir ./data \
  --max-images 0 --adc-bits 6 --out-csv analog_accuracy.csv

# BatchNorm baseline
python3 bn_baseline.py --sweep 32,64,128,256 --data-dir ./data \
  --max-images 0 --adc-bits 6 --out-csv bn_baseline.csv

# wire parasitics, isolated one at a time
python3 wordline_ir.py --breakdown --tile-m 128 --tile-k 16 \
  --out-csv wire_parasitics.csv

# device variability, 10 seeds
python3 analog_eval.py --sweep 32,64,128,256 \
  --variability 0,0.05,0.1,0.2 --variability-seeds 10 \
  --data-dir ./data --max-images 0 --adc-bits 6 --out-csv variability.csv

# drift
python3 analog_eval.py --sweep 32,64,128,256 \
  --drift 1,3600,86400,2592000,31536000 \
  --data-dir ./data --max-images 0 --adc-bits 6 --out-csv drift.csv

# ADC resolution
python3 analog_eval.py --sweep 32,64,128,256 --adc-sweep 4,5,6,8 \
  --data-dir ./data --max-images 0 --out-csv adc_sweep.csv

# level-mapping error, instant, no GPU
bash run_blocking.sh b4
```

---

## 9. Documents

```bash
cd "$S"
pdflatex -interaction=nonstopmode FP2_ReRAM_Explained.tex   # run 3x for TOC
pdflatex -interaction=nonstopmode date_paper.tex            # run 3x for refs
grep -c "^!" FP2_ReRAM_Explained.log date_paper.log         # want 0 and 0
```

---

## 10. Before submitting — the checks that catch stale numbers

```bash
cd "$S"
# numbers that were corrected; all should return 0
grep -rn "84\.4\|13\.6×\|13\.6x" *.md *.tex docs/*.md docs/*.tex 2>/dev/null | wc -l

# NeuroSim accuracy must never appear as a result
grep -rn "NeuroSim.*accurac\|accurac.*NeuroSim" *.md docs/*.md 2>/dev/null

# every TOPS/W should say whether it is tile-level or chip-level
grep -rn "TOPS/W" *.md docs/*.md 2>/dev/null | grep -v "tile-level\|chip-level"
```

Still yours, not scriptable — two items left. See `CITATIONS.md` §10.

- SRAM leakage in pW/bit at 65 nm
- digital MAC unit area (the 1800 / 220 µm² pair)

Both feed only the von Neumann baseline, which is already declared optimistic.
Neither touches the accuracy results.
