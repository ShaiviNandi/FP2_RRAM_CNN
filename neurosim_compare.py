#!/usr/bin/env python3
"""
neurosim_compare.py
Runs NeuroSim twice on the same network -- once with SRAM cells, once with the
ReRAM device -- and tabulates area, energy, latency and efficiency from a single
tool.

Why run the baseline inside neurosim:
The SRAM comparison in codesign_sweep.py is analytical: it multiplies a 6T
bitcell area by a bit count and prices a fetch. That is defensible for "how
much area do the weights occupy" and indefensible as an accelerator comparison,
because it counts no sense amplifiers, decoders, buffers or interconnect on the
SRAM side.

NeuroSim models both cell types with the same peripheral circuitry, the same
floorplanner and the same energy accounting. Running it twice removes the
methodology difference and leaves only the device difference, which is the
comparison the paper actually wants to make.

It also produces the numbers that are otherwise assumed:
transistor-level ADC area, buffer and interconnect energy, and chip-level
latency.

How the cell type is switched:
NeuroSim selects the cell in C++, not from the Python wrapper: Param.cpp
carries `memcelltype` with 1 = SRAM, 2 = analog eNVM (the case here), 3 = digital
eNVM. Switching therefore means editing Param.cpp and rebuilding, which this
script does with a backup and an automatic restore, so an interrupted run
cannot leave the tree in the SRAM configuration.

What to do with the result:
Report NeuroSim SRAM and NeuroSim ReRAM side by side as the like-for-like
comparison, and keeps the tile-level model as the third column with its
scope stated. Three columns from two tools is more honest than one number.

Usage:
    python3 neurosim_compare.py --dry-run          # show what would run
    python3 neurosim_compare.py --out-csv neurosim_compare.csv

    NEUROSIM_DIR=~/DNN_NeuroSim_V1.4 python3 neurosim_compare.py
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

CELL_TYPES = {"sram": 1, "rram": 2}

# NeuroSim sizes the 1T1R select transistor to carry the current the LRS
# demands, then checks it against the cell width assigned in the layout. A low
# LRS resistance needs a wide transistor: at R_ON = 542.8 ohm NeuroSim asks for
# 127.28F against a default assignment of 12F and refuses to continue.
#
# That is not a tool bug. It is the device saying an aggressive LRS costs cell
# area, which the CELL_AREA_F2 = 40 assumption in hw_model.py does not capture.
# Raising the assigned width lets the run finish AND reports the honest cell
# size, which is the number the paper needs.
WIDTH_PARAM = "widthInFeatureSize1T1R"

# Parsed out of NeuroSim's summary block. Keys are what the paper needs.
PATTERNS = {
    "chip_area_um2":      r"ChipArea\s*:\s*([\d.e+]+)um\^2",
    "cim_array_um2":      r"Chip total CIM array\s*:\s*([\d.e+]+)um\^2",
    "ic_area_um2":        r"Total IC Area on chip.*?:\s*([\d.e+]+)um\^2",
    "adc_area_um2":       r"Total ADC[^:]*Area on chip\s*:\s*([\d.e+]+)um\^2",
    "accum_area_um2":     r"Total Accumulation Circuits.*?on chip\s*:\s*([\d.e+]+)um\^2",
    "other_area_um2":     r"Other Peripheries.*?:\s*([\d.e+]+)um\^2",
    "latency_ns":         r"pipeline-system-clock-cycle \(per image\) is:\s*([\d.e+]+)ns",
    "dyn_energy_pj":      r"pipeline-system readDynamicEnergy \(per image\) is:\s*([\d.e+]+)pJ",
    "leak_energy_pj":     r"pipeline-system leakage Energy \(per image\) is:\s*([\d.e+]+)pJ",
    "leak_power_uw":      r"pipeline-system leakage Power \(per image\) is:\s*([\d.e+]+)uW",
    "buffer_energy_pj":   r"buffer readDynamicEnergy \(per image\) is:\s*([\d.e+]+)pJ",
    "ic_energy_pj":       r"ic readDynamicEnergy \(per image\) is:\s*([\d.e+]+)pJ",
    "tops_per_w":         r"Energy Efficiency TOPS/W[^:]*:\s*([\d.e+]+)",
    "tops":               r"Throughput TOPS[^:]*:\s*([\d.e+]+)",
    "fps":                r"Throughput FPS[^:]*:\s*([\d.e+]+)",
    "tops_per_mm2":       r"Compute efficiency TOPS/mm\^2[^:]*:\s*([\d.e+]+)",
    "accuracy_pct":       r"Accuracy:\s*\d+/\d+\s*\((\d+)%\)",
    "req_tx_width_f":     r"Transistor width of 1T1R=([\d.]+)F is larger",
}

# Energy split, needed to show what the tile-level model omits.
SPLIT = {
    "adc_energy_pj":   r"ADC.*?readDynamicEnergy is\s*:\s*([\d.e+]+)pJ",
    "accum_energy_pj": r"Accumulation Circuits.*?readDynamicEnergy is\s*:\s*([\d.e+]+)pJ",
    "other_energy_pj": r"Other Peripheries.*?readDynamicEnergy is\s*:\s*([\d.e+]+)pJ",
}


# =============================================================================
def find_neurosim():
    for c in (os.environ.get("NEUROSIM_DIR", ""), "../DNN_NeuroSim_V1.4",
              os.path.expanduser("~/DNN_NeuroSim_V1.4"),
              "/opt/DNN_NeuroSim_V1.4"):
        if c and os.path.isdir(c):
            return c
    raise SystemExit("NeuroSim not found. Set NEUROSIM_DIR=/path/to/"
                     "DNN_NeuroSim_V1.4 on the same line as the command.")


def set_param(param_cpp, name, value):
    """Rewrite a single named assignment in Param.cpp and verify the result.

    Matched up to the semicolon rather than to digits, so trailing arithmetic
    such as `= 6e3*17;` is replaced rather than left in place. A digits-only
    pattern is how an earlier device conversion in this project ended up ten
    times wrong.
    """
    src = open(param_cpp).read()
    m = re.search(rf"({re.escape(name)}\s*=\s*)([^;]+)(;)", src)
    if not m:
        return None                      # parameter absent in this version
    prev = m.group(2).strip()
    open(param_cpp, "w").write(src[:m.start(2)] + str(value) + src[m.end(2):])
    chk = re.search(rf"{re.escape(name)}\s*=\s*([^;]+);", open(param_cpp).read())
    if chk.group(1).strip() != str(value):
        raise SystemExit(f"{name} verification failed")
    return prev


def set_cell_type(param_cpp, value, dry=False):
    """Rewrite memcelltype in Param.cpp, returning the previous value.

    The assignment is matched up to the semicolon rather than to digits alone.
    A digits-only pattern silently leaves trailing arithmetic in place, which is
    how an earlier device conversion in this project ended up ten times wrong.
    """
    src = open(param_cpp).read()
    m = re.search(r"(memcelltype\s*=\s*)([^;]+)(;)", src)
    if not m:
        raise SystemExit(f"could not find memcelltype in {param_cpp}")
    prev = m.group(2).strip()
    if dry:
        return prev
    open(param_cpp, "w").write(
        src[:m.start(2)] + str(value) + src[m.end(2):])
    # Read back and verify rather than trusting the write.
    check = re.search(r"memcelltype\s*=\s*([^;]+);", open(param_cpp).read())
    got = check.group(1).strip()
    if got != str(value):
        raise SystemExit(f"memcelltype verification failed: wanted {value}, "
                         f"file now reads {got!r}")
    return prev


def parse(text):
    out = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    # The per-layer split repeats; the final occurrence is the chip summary.
    for key, pat in SPLIT.items():
        ms = re.findall(pat, text)
        out[key] = float(ms[-1]) if ms else None
    return out


def run_one(ns, cell, args):
    """Set the cell type, rebuild, run, restore. Restore happens even on error."""
    inf = os.path.join(ns, "Inference_pytorch")
    param = os.path.join(inf, "NeuroSIM", "Param.cpp")
    ckpt = os.path.join(inf, "log", f"{args.model}.pth")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"missing checkpoint {ckpt}. NeuroSim ships it via a "
                         f"link in Inference_pytorch/README.")

    backup = param + ".fp2backup"
    shutil.copy2(param, backup)
    try:
        prev = set_cell_type(param, CELL_TYPES[cell])
        print(f"  memcelltype {prev} -> {CELL_TYPES[cell]} ({cell})", flush=True)

        if cell == "rram" and args.resistance_on:
            was = set_param(param, "resistanceOn", args.resistance_on)
            print(f"  resistanceOn {was} -> {args.resistance_on}", flush=True)
        if cell == "rram" and args.resistance_off:
            was = set_param(param, "resistanceOff", args.resistance_off)
            print(f"  resistanceOff {was} -> {args.resistance_off}", flush=True)

        if cell == "rram" and args.cell_width:
            was = set_param(param, WIDTH_PARAM, args.cell_width)
            if was is None:
                print(f"  [warn] {WIDTH_PARAM} not found; if the run aborts on "
                      f"transistor width, find the equivalent name in Param.cpp")
            else:
                print(f"  {WIDTH_PARAM} {was} -> {args.cell_width}", flush=True)
                print(f"  NOTE: this is the honest cell size for R_LRS=542.8. "
                      f"Compare it against CELL_AREA_F2=40 in hw_model.py.",
                      flush=True)

        print("  rebuilding NeuroSIM ...", flush=True)
        mk = subprocess.run(["make", "-j4"], cwd=os.path.join(inf, "NeuroSIM"),
                            capture_output=True, text=True)
        if mk.returncode != 0:
            print(mk.stdout[-2000:]); print(mk.stderr[-2000:])
            raise SystemExit(f"NeuroSIM build failed for {cell}")

        cmd = [sys.executable, "inference.py",
               "--dataset", args.dataset, "--model", args.model,
               "--mode", "WAGE", "--cellBit", str(args.cell_bit),
               "--subArray", str(args.subarray),
               "--ADCprecision", str(args.adc_bits)]
        # parallelRead is the number of rows sensed simultaneously and cannot
        # exceed the subarray height. NeuroSim's default is 128; leaving it
        # there while shrinking --subArray asks for more rows than exist and
        # every latency and energy figure comes back -nan, with the summary
        # section never written. Tied to the subarray unless overridden.
        pr = args.parallel_read or args.subarray
        if pr > args.subarray:
            print(f"  [warn] parallelRead {pr} exceeds subArray "
                  f"{args.subarray}; results will be nan")
        cmd += ["--parallelRead", str(pr)]
        if args.wl_weight is not None:
            cmd += ["--wl_weight", str(args.wl_weight)]
        if args.wl_activate is not None:
            cmd += ["--wl_activate", str(args.wl_activate)]
        if cell == "rram":
            cmd += ["--onoffratio", str(args.onoffratio)]
        print(f"  $ {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=inf, capture_output=True, text=True)
        text = r.stdout + r.stderr
        open(f"neurosim_{cell}.log", "w").write(text)
        if r.returncode != 0:
            print(text[-3000:])
            raise SystemExit(f"NeuroSim run failed for {cell}")
        return parse(text)
    finally:
        shutil.move(backup, param)
        print(f"  Param.cpp restored", flush=True)


# =============================================================================
def report(rows):
    def g(r, k, scale=1.0, nd=2):
        v = r.get(k)
        return f"{v*scale:,.{nd}f}" if v is not None else "--"

    print("\n" + "=" * 78)
    print("NEUROSIM, SAME TOOL, SAME NETWORK, DIFFERENT CELL")
    print("=" * 78)
    cfg = rows[0].get("_cfg", "")
    if cfg:
        print(cfg)
        print("-" * 78)
    names = [r["cell"] for r in rows]
    print(f"{'metric':<34}" + "".join(f"{n:>14}" for n in names))
    print("-" * 78)
    for label, key, scale, nd in [
        ("chip area (mm2)",        "chip_area_um2", 1e-6, 3),
        ("  CIM array (mm2)",      "cim_array_um2", 1e-6, 3),
        ("  ADC / sense (mm2)",    "adc_area_um2", 1e-6, 3),
        ("  interconnect (mm2)",   "ic_area_um2", 1e-6, 3),
        ("  accumulation (mm2)",   "accum_area_um2", 1e-6, 3),
        ("  other periphery (mm2)","other_area_um2", 1e-6, 3),
        ("latency/image (us)",     "latency_ns", 1e-3, 2),
        ("dynamic energy/img (uJ)","dyn_energy_pj", 1e-6, 3),
        ("leakage power (uW)",     "leak_power_uw", 1.0, 1),
        ("  ADC energy (uJ)",      "adc_energy_pj", 1e-6, 3),
        ("  accum energy (uJ)",    "accum_energy_pj", 1e-6, 3),
        ("  other energy (uJ)",    "other_energy_pj", 1e-6, 3),
        ("TOPS",                   "tops", 1.0, 2),
        ("TOPS/W",                 "tops_per_w", 1.0, 2),
        ("TOPS/mm2",               "tops_per_mm2", 1.0, 4),
        ("FPS",                    "fps", 1.0, 0),
        ("accuracy (%)",           "accuracy_pct", 1.0, 0),
        ("1T1R tx width needed (F)","req_tx_width_f", 1.0, 2),
    ]:
        print(f"{label:<34}" + "".join(f"{g(r,key,scale,nd):>14}" for r in rows))
    print("-" * 78)

    d = {r["cell"]: r for r in rows}
    if "sram" in d and "rram" in d:
        s_, m_ = d["sram"], d["rram"]

        def cmp(label, key, higher_is_better):
            a, b = s_.get(key), m_.get(key)
            if not a or not b:
                return
            # State the direction in words. A bare ratio like "0.26x" is
            # ambiguous about which way it cuts and has already caused one
            # misreading of these results.
            if higher_is_better:
                if b >= a:
                    print(f"  {label}: ReRAM BETTER by {b/a:.2f}x")
                else:
                    print(f"  {label}: ReRAM WORSE by {a/b:.2f}x")
            else:
                if b <= a:
                    print(f"  {label}: ReRAM BETTER by {a/b:.2f}x")
                else:
                    print(f"  {label}: ReRAM WORSE by {b/a:.2f}x")

        for r in rows:
            parts = sum(r.get(k) or 0 for k in
                        ("cim_array_um2", "adc_area_um2", "ic_area_um2",
                         "accum_area_um2", "other_area_um2"))
            tot = r.get("chip_area_um2") or 0
            if tot and abs(parts - tot) / tot > 0.05:
                print(f"\n  [warn] {r['cell']}: area components sum to "
                      f"{parts*1e-6:.1f} mm2 but chip is {tot*1e-6:.1f} mm2 "
                      f"({100*(tot-parts)/tot:.0f}% unaccounted). A component "
                      f"is not being parsed -- do not trust the breakdown.")

        print("\nDIRECTION OF EACH COMPARISON")
        cmp("chip area", "chip_area_um2", False)
        cmp("latency/image", "latency_ns", False)
        cmp("dynamic energy", "dyn_energy_pj", False)
        cmp("leakage power", "leak_power_uw", False)
        cmp("TOPS/W", "tops_per_w", True)
        cmp("TOPS/mm2", "tops_per_mm2", True)
        cmp("throughput FPS", "fps", True)

        print()
        print("Both columns come from one tool, one floorplanner and one energy")
        print("model, so the difference is the device rather than the")
        print("methodology. That is what the analytical SRAM baseline could not")
        print("claim -- and it cuts both ways.")

        w = m_.get("req_tx_width_f")
        print()
        print("READ THIS BEFORE QUOTING ANYTHING ABOVE")
        print("  These ReRAM numbers are for R_LRS = 542.8 ohm, which forces a")
        print("  ~127F-wide 1T1R select transistor to carry the read current.")
        print("  That cell dominates the area result and, through it, TOPS/mm2.")
        print("  A higher LRS resistance needs a narrower transistor and a")
        print("  smaller cell, at the cost of less signal current. Sweep it:")
        print("    for R in 542.8 2000 6000 20000; do \\")
        print("      NEUROSIM_DIR=... python3 neurosim_compare.py \\")
        print("        --cells rram --resistance-on $R --cell-width 0 \\")
        print("        --out-csv rram_ron_$R.csv ; done")
        print("  The cell width NeuroSim demands scales roughly as 1/R_LRS, so")
        print("  the sweep is the design result rather than the single point.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="VGG8")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--subarray", type=int, default=128)
    ap.add_argument("--adc-bits", type=int, default=6)
    ap.add_argument("--parallel-read", type=int, default=None,
                    help="Rows sensed in parallel. Defaults to --subarray. "
                         "Values above the subarray height produce nan.")
    ap.add_argument("--cell-bit", type=int, default=2)
    ap.add_argument("--wl-weight", type=int, default=None,
                    help="Weight precision in bits. NeuroSim defaults to 8, "
                         "which stores each weight across 8/cellBit cells and "
                         "adds a shift-add tree to recombine them -- hardware "
                         "FP2 does not need. Set 2 to match FP2, where one "
                         "2-bit cell holds a whole weight and the shift-add "
                         "disappears.")
    ap.add_argument("--wl-activate", type=int, default=None,
                    help="Activation precision in bits. Left at NeuroSim's "
                         "default unless set, so weight and activation "
                         "precision can be varied independently.")
    ap.add_argument("--onoffratio", type=float, default=403)
    ap.add_argument("--resistance-on", type=float, default=None,
                    help="Override resistanceOn (LRS, ohm) in Param.cpp. The "
                         "select-transistor width, and hence cell area, scales "
                         "roughly as 1/resistanceOn.")
    ap.add_argument("--resistance-off", type=float, default=None,
                    help="Override resistanceOff (HRS, ohm).")
    ap.add_argument("--cell-width", type=float, default=128.0,
                    help="Assigned 1T1R cell width in feature sizes. NeuroSim "
                         "refuses to run when the transistor it sizes for the "
                         "LRS current exceeds this. 0 leaves Param.cpp alone.")
    ap.add_argument("--cells", default="sram,rram")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ns = find_neurosim()
    print(f"NeuroSim at {ns}")
    param = os.path.join(ns, "Inference_pytorch", "NeuroSIM", "Param.cpp")
    print(f"memcelltype currently: {set_cell_type(param, 0, dry=True)}")
    if args.dry_run:
        for c in args.cells.split(","):
            print(f"  would set memcelltype={CELL_TYPES[c]} ({c}), rebuild, run")
        return

    rows = []
    for c in args.cells.split(","):
        print(f"\n=== {c.upper()} ===", flush=True)
        r = run_one(ns, c, args)
        r["cell"] = c
        wlw = args.wl_weight if args.wl_weight is not None else 8
        cells_per_w = max(1, -(-wlw // args.cell_bit))     # ceil division
        r["_cfg"] = (f"weight {wlw}b, cell {args.cell_bit}b -> "
                     f"{cells_per_w} cell(s) per weight"
                     + ("  [FP2: one cell, no shift-add tree]"
                        if cells_per_w == 1 else
                        f"  [needs a shift-add tree over {cells_per_w} cells]"))
        rows.append(r)
    report(rows)

    if args.out_csv and rows:
        keys = ["cell"] + [k for k in rows[0] if k != "cell"]
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
