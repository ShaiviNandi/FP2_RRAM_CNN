#!/usr/bin/env python3
"""
codesign_sweep.py
The paper's core experiment, plus the digital baseline that motivates it.

The thesis this script exists to test:
In a block-floating-point format, B = the number of weights sharing one scale.
In an analog CIM macro, M = the number of rows sharing one bitline and one ADC.

For an output-side-rescaled crossbar these are the same PHYSICAL PARAMETER.
benchmark_resnet18.weight_scale_factor already documents why: the crossbar
rescales a whole tile's output current by one scale, so a tile may not span
two quantization blocks with different scales. Hence B == M, necessarily.

FP2 (Dang et al., TCAS-I 2026) specifies B = 32, chosen for digital packing.
Analog CIM wants M >= 128, because the ADC's per-column cost amortizes over M
rows and dominates everything below that. Two communities, one parameter,
opposite directions, and no published measurement of what the collision costs
-- because measuring it needs QAT accuracy AND circuit-exact analog error on
the same weights, which is what this repo now does.

This script sweeps B, and at each B collects:

    accuracy   from qat_finetune_fp2.py --block-size B      (FP32/PTQ/QAT)
    SNR        from benchmark_resnet18.py --tile-m B        (digital + analog)
    hardware   from hw_model.py --tile-m B                  (pJ/MAC, TOPS/W, mm2)

One row per B. The resulting CSV is the paper's Pareto curve.

It also computes the DIGITAL BASELINE (Section: motivation) -- ReRAM vs 6T SRAM
on the metrics ReRAM actually wins, which are all memory metrics, not compute
metrics. See digital_baseline() for the arithmetic and its assumptions.

Why the FP32 PRE-TRAIN is hoisted out:
The FP32 starting point does not depend on B, so pre-training it once and
reusing it across all B is both faster and necessary for the comparison
to be clean: if each B started from its own independently-trained FP32 model,
differences between B would be confounded with run-to-run training variance
(repeat runs vary by ~0.3 pts at fixed settings). --fp32-checkpoint enforces
the shared start.

Usage:
    python3 codesign_sweep.py --self-test        # no GPU, no data needed

    # See the commands without running them
    python3 codesign_sweep.py --blocks 32,64,128,256 --dry-run

    # 0) one shared FP32 start (~4 min on a 4070)
    python3 qat_finetune_fp2.py --dataset cifar10 --data-dir ./data         --pretrain-epochs 30 --epochs 0 --bf16 --batch-size 256         --save-fp32-checkpoint resnet18_cifar10_fp32.pth         --out-checkpoint /tmp/discard.pth

    # 1) the sweep (~4 min per B: 10 QAT epochs + benchmark + model)
    python3 codesign_sweep.py --blocks 32,64,128,256         --fp32-checkpoint resnet18_cifar10_fp32.pth         --data-dir ./data --out-csv codesign_pareto.csv

    # 2) just the digital-baseline table (instant)
    python3 codesign_sweep.py --baseline-only --weights 11157504
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

# =============================================================================
# DIGITAL BASELINE ASSUMPTIONS
# Same epistemic rules as hw_model.py: these are literature-typical values for
# a technology this script cannot see. Override with --set NAME=VALUE.
# =============================================================================
BASE = {
    "TECH_F_NM": 65.0,
    # 1T1R ReRAM footprint. Matches hw_model.CELL_AREA_F2 -- keep them equal or
    # the two scripts describe different chips.
    "RERAM_CELL_F2": 40.0,
    # 6T SRAM bitcell. 120-160 F^2 is the usual 65nm band.
    "SRAM_BIT_F2": 140.0,
    # Static leakage per SRAM bit at room temperature. This is the number the
    # non-volatility argument rests on and it varies by more than 10x with
    # process corner, Vdd and temperature -- treat the standby comparison as
    # order-of-magnitude until replaced with silicon data.
    "SRAM_LEAK_PW_PER_BIT": 120.0,
    # Energy to read one bit out of an SRAM macro and deliver it to a MAC unit.
    # This is the term in-memory compute eliminates entirely.
    "SRAM_READ_PJ_PER_BIT": 0.05,
    # ReRAM write cost. Excluded from hw_model entirely; included here because
    # any honest ReRAM-vs-SRAM comparison has to price the thing ReRAM is bad
    # at. Reported per weight (2 cells).
    "RERAM_WRITE_PJ_PER_CELL": 100.0,
    "RERAM_WRITE_NS_PER_CELL": 100.0,
    "FP2_BITS_PER_WEIGHT": 2.0,
    "FP32_BITS_PER_WEIGHT": 32.0,
}


def digital_baseline(n_weights, assumptions=None, weight_reuse=1.0):
    """ReRAM vs SRAM on the metrics ReRAM actually wins.

    weight_reuse = how many MACs each fetched weight serves in the digital
    design. This is the single most important and most contested number in any
    CIM-vs-digital comparison: a well-blocked dataflow on a large-batch conv
    reuses each weight thousands of times and the weight-fetch term nearly
    vanishes, while batch-1 inference on 1x1 convs and FC layers reuses each
    weight ONCE and the fetch term dominates. Reporting a single reuse figure
    is how CIM papers inflate their advantage; report the curve instead
    (--reuse-sweep does).
    """
    A = dict(BASE)
    if assumptions:
        A.update(assumptions)
    f_um = A["TECH_F_NM"] / 1000.0
    um2_per_f2 = f_um * f_um

    fp2_bits = A["FP2_BITS_PER_WEIGHT"]
    # 2T2R: two cells per signed FP2 weight (one per polarity)
    reram_um2_per_w = 2.0 * A["RERAM_CELL_F2"] * um2_per_f2
    sram_fp2_um2_per_w = fp2_bits * A["SRAM_BIT_F2"] * um2_per_f2
    sram_fp32_um2_per_w = A["FP32_BITS_PER_WEIGHT"] * A["SRAM_BIT_F2"] * um2_per_f2

    area = dict(
        reram_mm2=n_weights * reram_um2_per_w / 1e6,
        sram_fp2_mm2=n_weights * sram_fp2_um2_per_w / 1e6,
        sram_fp32_mm2=n_weights * sram_fp32_um2_per_w / 1e6,
        density_gain_vs_sram_fp2=sram_fp2_um2_per_w / reram_um2_per_w,
        density_gain_vs_sram_fp32=sram_fp32_um2_per_w / reram_um2_per_w,
        um2_per_weight_reram=reram_um2_per_w,
        um2_per_weight_sram_fp2=sram_fp2_um2_per_w,
    )

    bits = n_weights * fp2_bits
    standby = dict(
        sram_leak_mw=bits * A["SRAM_LEAK_PW_PER_BIT"] * 1e-12 * 1e3,
        reram_leak_mw=0.0,
        bits_held=bits,
    )

    # Weight-delivery energy per MAC. CIM's is zero by construction: the weight
    # never leaves the cell it is stored in.
    fetch_pj_per_mac_sram = (fp2_bits * A["SRAM_READ_PJ_PER_BIT"]) / max(weight_reuse, 1e-9)
    movement = dict(
        weight_reuse=weight_reuse,
        sram_fetch_pj_per_mac=fetch_pj_per_mac_sram,
        reram_fetch_pj_per_mac=0.0,
    )

    write = dict(
        reram_full_program_uj=n_weights * 2 * A["RERAM_WRITE_PJ_PER_CELL"] * 1e-6,
        reram_full_program_ms=n_weights * 2 * A["RERAM_WRITE_NS_PER_CELL"] * 1e-6,
    )
    return dict(area=area, standby=standby, movement=movement, write=write,
                assumptions=A, n_weights=n_weights)


def duty_cycle_crossover(base, active_energy_j_per_inference, inferences_per_sec):
    """At what duty cycle does SRAM standby leakage exceed the entire active
    energy budget? Below that point non-volatility, not compute efficiency, is
    the dominant term -- which is the regime edge FP2 targets."""
    leak_w = base["standby"]["sram_leak_mw"] * 1e-3
    active_w = active_energy_j_per_inference * inferences_per_sec
    if leak_w <= 0:
        return float("inf")
    return active_w / leak_w  # duty cycle at which the two are equal


def print_baseline(base):
    A, ar, st, mv, wr = (base["assumptions"], base["area"], base["standby"],
                         base["movement"], base["write"])
    n = base["n_weights"]
    print("\n" + "=" * 78)
    print(f"DIGITAL BASELINE -- 2T2R ReRAM vs 6T SRAM, {n:,} weights, "
          f"{A['TECH_F_NM']:.0f} nm")
    print("=" * 78)
    print("\nWEIGHT STORAGE AREA  [MODEL on ASSUM]")
    print(f"  per weight   ReRAM 2T2R      {ar['um2_per_weight_reram']:.4f} um2")
    print(f"               SRAM  FP2 (2b)  {ar['um2_per_weight_sram_fp2']:.4f} um2  "
          f"-> ReRAM {ar['density_gain_vs_sram_fp2']:.1f}x denser")
    print(f"  whole net    ReRAM           {ar['reram_mm2']:.2f} mm2")
    print(f"               SRAM  FP2       {ar['sram_fp2_mm2']:.2f} mm2")
    print(f"               SRAM  FP32      {ar['sram_fp32_mm2']:.1f} mm2  "
          f"-> ReRAM {ar['density_gain_vs_sram_fp32']:.0f}x smaller")
    print("\nSTANDBY  [MODEL on ASSUM -- leakage varies >10x with corner/temp]")
    print(f"  SRAM holding {st['bits_held']/8e6:.2f} MB of FP2 weights: "
          f"{st['sram_leak_mw']:.2f} mW, continuously, forever")
    print(f"  ReRAM: {st['reram_leak_mw']:.1f} mW. Retention is a materials "
          f"property, not a power draw.")
    print("\nWEIGHT MOVEMENT  [MODEL]")
    print(f"  at reuse={mv['weight_reuse']:.0f}x  SRAM fetch "
          f"{mv['sram_fetch_pj_per_mac']:.4f} pJ/MAC   ReRAM 0 pJ/MAC")
    print(f"  (in-memory compute eliminates this term by construction; it is "
          f"the whole point)")
    print("\nTHE COST SIDE -- what ReRAM is bad at  [MODEL on ASSUM]")
    print(f"  programming all {2*n:,} cells once: {wr['reram_full_program_uj']:.1f} uJ, "
          f"{wr['reram_full_program_ms']:.1f} ms")
    print(f"  A design that reloads tiles often pays this repeatedly. Any "
          f"time-multiplexed\n  mapping (hw_model --physical-tiles) must "
          f"amortize it or the comparison is dishonest.")
    print("=" * 78)


# =============================================================================
# Sweep orchestration
# =============================================================================
def run(cmd, dry=False, log=None):
    printable = " ".join(cmd)
    print(f"    $ {printable}")
    if dry:
        return 0, ""
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    if log:
        with open(log, "a") as f:
            f.write(f"\n\n===== {printable} =====\n{out}")
    print(f"      -> exit {p.returncode} in {time.time()-t0:.0f}s"
          + (f", log appended to {log}" if log else ""))
    if p.returncode != 0:
        print(f"      !! FAILED. Last lines:\n"
              + "\n".join("      " + l for l in out.strip().splitlines()[-8:]))
    return p.returncode, out


def parse_qat_meta(path):
    """qat_finetune_fp2.py writes <checkpoint>_meta.json with the accuracies."""
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        m = json.load(f)
    return dict(acc_fp32=m.get("acc_fp32"), acc_ptq=m.get("acc_fp2_ptq"),
                acc_qat=m.get("acc_fp2_qat"),
                util_qat=m.get("cell_utilization_qat"),
                resid_qat=m.get("quant_residual_qat"))


def parse_bench_csv(path):
    """Mean SNR over layers from benchmark_resnet18.py's CSV."""
    if not os.path.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {}
    def mean(k):
        vals = [float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")]
        return sum(vals) / len(vals) if vals else float("nan")
    dig, ana = mean("snr_digital_db"), mean("snr_analog_db")
    return dict(snr_digital_db=dig, snr_analog_db=ana, snr_gap_db=dig - ana,
                relerr_analog_pct=mean("relerr_analog_pct"),
                cell_util_pct=mean("cell_utilization_pct"), n_layers=len(rows))


def parse_hw_report(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    R = d.get("result", {})
    pl = R.get("per_layer", [])
    adc = (sum(r["adc_energy_frac"] for r in pl) / len(pl)) if pl else float("nan")
    return dict(pj_per_mac=R.get("pj_per_mac"), tops_per_w=R.get("tops_per_w"),
                area_mm2=(R.get("area") or {}).get("total_mm2"),
                adc_energy_frac=adc, total_tiles=R.get("total_tiles"))


def sweep(args):
    blocks = [int(b) for b in args.blocks.split(",")]
    rows = []
    log = args.log
    if log and os.path.exists(log) and not args.dry_run:
        os.remove(log)

    for B in blocks:
        print(f"\n{'='*78}\nBLOCK / TILE HEIGHT B = M = {B}\n{'='*78}")
        ckpt = f"qat_b{B}.pth"
        bench_csv = f"bench_b{B}.csv"
        hw_json = f"hw_b{B}.json"

        if args.skip_qat and os.path.exists(ckpt) and not args.dry_run:
            print(f"  [1/3] QAT skipped, reusing {ckpt}")
        else:
            print(f"  [1/3] QAT at block={B}")
            qat = [sys.executable, "qat_finetune_fp2.py",
                   "--dataset", args.dataset, "--data-dir", args.data_dir,
                   "--block-size", str(B), "--scale-mode", args.scale_mode,
                   "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                   "--lr", str(args.lr), "--out-checkpoint", ckpt]
            if args.fp32_checkpoint:
                qat += ["--checkpoint", args.fp32_checkpoint]
            else:
                qat += ["--pretrain-epochs", str(args.pretrain_epochs)]
            if args.bf16:
                qat += ["--bf16"]
            if args.workers:
                qat += ["--workers", str(args.workers)]
            rc, _ = run(qat, args.dry_run, log)
            if rc and not args.dry_run:
                print(f"  skipping B={B}: QAT failed")
                continue

        if args.skip_bench and os.path.exists(bench_csv) and not args.dry_run:
            print(f"  [2/3] Benchmark skipped, reusing {bench_csv}")
        else:
            print(f"  [2/3] Crossbar fidelity at tile_m={B}")
            bench = [sys.executable, "benchmark_resnet18.py",
                     "--checkpoint", ckpt, "--cifar-arch",
                     "--num-classes", str(args.num_classes),
                     "--tile-m", str(B), "--tile-k", str(args.tile_k),
                     "--scale-mode", args.scale_mode,
                     "--calib-dataset", args.calib_dataset,
                     "--calib-dir", args.data_dir,
                     "--skip-first-last", "--max-positions", str(args.max_positions),
                     "--out-csv", bench_csv]
            rc, _ = run(bench, args.dry_run, log)

        print(f"  [3/3] Hardware model at tile_m={B}")
        hw = [sys.executable, "hw_model.py", "--layers-csv", bench_csv,
              "--tile-m", str(B), "--tile-k", str(args.tile_k),
              "--report", hw_json]
        if args.power_csv:
            hw += ["--power-csv", args.power_csv]
        if args.adc_bits is not None:
            hw += ["--adc-bits", str(args.adc_bits)]
        if args.physical_tiles:
            hw += ["--physical-tiles", str(args.physical_tiles)]
        rc, _ = run(hw, args.dry_run, log)

        if args.dry_run:
            continue

        row = dict(block_size=B, tile_m=B, tile_k=args.tile_k)
        row.update(parse_qat_meta(ckpt))
        row.update(parse_bench_csv(bench_csv))
        row.update(parse_hw_report(hw_json))
        rows.append(row)
        print(f"  => acc_qat={row.get('acc_qat')}  "
              f"snr_analog={row.get('snr_analog_db')}  "
              f"TOPS/W={row.get('tops_per_w')}")

    if args.dry_run or not rows:
        return rows

    cols = ["block_size", "tile_m", "tile_k", "acc_fp32", "acc_ptq", "acc_qat",
            "util_qat", "resid_qat", "snr_digital_db", "snr_analog_db",
            "snr_gap_db", "relerr_analog_pct", "cell_util_pct",
            "pj_per_mac", "tops_per_w", "area_mm2", "adc_energy_frac",
            "total_tiles", "n_layers"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*88}")
    print("CO-DESIGN PARETO -- block size B is simultaneously the FP2 block "
          "and the crossbar tile height")
    print("=" * 88)
    print(f"{'B=M':>5}{'acc QAT%':>10}{'vs FP32':>9}{'SNR dig':>9}{'SNR ana':>9}"
          f"{'gap':>7}{'pJ/MAC':>9}{'TOPS/W':>9}{'ADC%E':>8}{'area mm2':>10}")
    print("-" * 88)
    for r in rows:
        d = (r.get("acc_qat") or 0) - (r.get("acc_fp32") or 0)
        print(f"{r['block_size']:>5}{r.get('acc_qat') or float('nan'):>10.2f}"
              f"{d:>+9.2f}{r.get('snr_digital_db') or float('nan'):>9.2f}"
              f"{r.get('snr_analog_db') or float('nan'):>9.2f}"
              f"{r.get('snr_gap_db') or float('nan'):>7.2f}"
              f"{r.get('pj_per_mac') or float('nan'):>9.4f}"
              f"{r.get('tops_per_w') or float('nan'):>9.1f}"
              f"{100*(r.get('adc_energy_frac') or float('nan')):>8.1f}"
              f"{r.get('area_mm2') or float('nan'):>10.3f}")
    print("-" * 88)
    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        dacc = (hi.get("acc_qat") or 0) - (lo.get("acc_qat") or 0)
        dsnr = (hi.get("snr_analog_db") or 0) - (lo.get("snr_analog_db") or 0)
        de = ((hi.get("tops_per_w") or 1) / (lo.get("tops_per_w") or 1))
        print(f"B {lo['block_size']} -> {hi['block_size']}:  "
              f"accuracy {dacc:+.2f} pts,  analog SNR {dsnr:+.2f} dB,  "
              f"efficiency x{de:.1f}")
        print("That trade is the paper. FP2 fixes B=32 for reasons that have "
              "nothing to do with\nanalog readout; this table prices that choice "
              "in accuracy, SNR and TOPS/W at once.")
    print(f"\nWrote {args.out_csv}")
    return rows


# =============================================================================
def self_test():
    print("=== codesign_sweep self-test ===\n")
    ok = True

    print("[1] Baseline area arithmetic is dimensionally right")
    b = digital_baseline(11_157_504)
    ar = b["area"]
    # hand check: 2 cells * 40 F^2 * (0.065 um)^2
    manual = 2 * 40 * (0.065 ** 2)
    print(f"    ReRAM per weight {ar['um2_per_weight_reram']:.6f} um2 "
          f"(hand {manual:.6f}) -> {abs(ar['um2_per_weight_reram']-manual)<1e-12}")
    print(f"    density vs SRAM-FP2  {ar['density_gain_vs_sram_fp2']:.2f}x   "
          f"vs SRAM-FP32 {ar['density_gain_vs_sram_fp32']:.1f}x")
    ok &= abs(ar["um2_per_weight_reram"] - manual) < 1e-12
    # ReRAM must be denser than SRAM at equal precision, or an assumption is wrong
    ok &= ar["density_gain_vs_sram_fp2"] > 1.0
    # and the FP32 gain must be exactly 16x the FP2 gain (32 bits vs 2 bits)
    ratio = ar["density_gain_vs_sram_fp32"] / ar["density_gain_vs_sram_fp2"]
    print(f"    FP32/FP2 density-gain ratio {ratio:.2f} (must be 16.0)")
    ok &= abs(ratio - 16.0) < 1e-9

    print("\n[2] Area scales linearly with weight count")
    b2 = digital_baseline(2 * 11_157_504)
    lin = abs(b2["area"]["reram_mm2"] - 2 * b["area"]["reram_mm2"]) < 1e-9
    print(f"    doubling weights doubles area: {lin}")
    ok &= lin

    print("\n[3] Weight-movement term falls as 1/reuse, ReRAM stays at zero")
    prev = None
    for reuse in (1, 10, 100, 1000):
        m = digital_baseline(1000, weight_reuse=reuse)["movement"]
        print(f"    reuse {reuse:5d}x -> SRAM {m['sram_fetch_pj_per_mac']:.6f} pJ/MAC, "
              f"ReRAM {m['reram_fetch_pj_per_mac']:.1f}")
        if prev is not None:
            ok &= m["sram_fetch_pj_per_mac"] < prev
        prev = m["sram_fetch_pj_per_mac"]
        ok &= m["reram_fetch_pj_per_mac"] == 0.0
    print("    monotonic decrease and ReRAM identically zero:", ok)

    print("\n[4] Duty-cycle crossover behaves sensibly")
    # 3642.7 nJ per inference is the measured number from hw_report.json
    for rate in (1, 10, 100):
        d = duty_cycle_crossover(b, 3642.7e-9, rate)
        print(f"    at {rate:4d} inferences/s, SRAM leakage equals active power "
              f"at duty cycle {100*d:.4f}%")
    d1 = duty_cycle_crossover(b, 3642.7e-9, 1)
    d2 = duty_cycle_crossover(b, 3642.7e-9, 100)
    print(f"    higher inference rate raises the crossover: {d2 > d1}")
    ok &= d2 > d1

    print("\n[5] CSV parsers tolerate missing files rather than crashing")
    for fn, name in ((parse_qat_meta, "parse_qat_meta"),
                     (parse_bench_csv, "parse_bench_csv"),
                     (parse_hw_report, "parse_hw_report")):
        r = fn("definitely_not_a_real_file_12345.csv")
        print(f"    {name}(missing) -> {r} (empty dict expected)")
        ok &= r == {}

    print("\n[6] Baseline report renders")
    print_baseline(b)

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 60)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Print the ReRAM-vs-SRAM table and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands the sweep would run, then stop")
    ap.add_argument("--weights", type=int, default=11_157_504,
                    help="Weight count for the baseline table (default: the "
                         "crossbar-mapped weights of CIFAR ResNet-18)")
    ap.add_argument("--reuse-sweep", default=None, metavar="LIST",
                    help="Report the weight-fetch term across a range of reuse "
                         "factors, e.g. 1,10,100,1000. Reuse=1 (batch-1 "
                         "inference on FC and 1x1 convs) maximally flatters "
                         "in-memory compute; a well-blocked batched conv "
                         "reuses each weight thousands of times and the term "
                         "nearly vanishes. Reporting the curve rather than a "
                         "single point is the honest presentation.")
    ap.add_argument("--reuse", type=float, default=1.0,
                    help="Weight reuse factor for the digital baseline. 1 = "
                         "batch-1 inference on 1x1/FC layers, the regime that "
                         "favours CIM most. Be explicit about this in the paper.")

    ap.add_argument("--blocks", default="32,64,128,256")
    ap.add_argument("--fp32-checkpoint", default=None,
                    help="Shared FP32 start for every B. Strongly recommended: "
                         "without it each B retrains from scratch and B-to-B "
                         "differences get confounded with training variance.")
    ap.add_argument("--pretrain-epochs", type=int, default=30,
                    help="Only used when --fp32-checkpoint is absent")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--calib-dataset", default="cifar10")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tile-k", type=int, default=16)
    ap.add_argument("--scale-mode", default="e8m0")
    ap.add_argument("--max-positions", type=int, default=32)
    ap.add_argument("--power-csv", default=None,
                    help="ngspice_full_sweep.py summary. WITHOUT this, hw_model "
                         "falls back to an empirical array-power estimate and "
                         "the efficiency column is pessimistic -- pass it.")
    ap.add_argument("--adc-bits", type=float, default=None,
                    help="ADC resolution for the hardware model. Default 8 is "
                         "over-provisioned: analog_eval --adc-sweep measured 5 "
                         "bits as sufficient at every block size.")
    ap.add_argument("--skip-qat", action="store_true",
                    help="Reuse existing qat_b<B>.pth instead of retraining. Use "
                         "when only the benchmark or hardware-model stages need "
                         "to be redone -- saves ~3 min per block size.")
    ap.add_argument("--skip-bench", action="store_true",
                    help="Reuse existing bench_b<B>.csv too, i.e. re-run only "
                         "the hardware model. Seconds instead of minutes.")
    ap.add_argument("--physical-tiles", type=int, default=0)
    ap.add_argument("--out-csv", default="codesign_pareto.csv")
    ap.add_argument("--log", default="codesign_sweep.log")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args()

    for kv in args.set:
        name, _, val = kv.partition("=")
        if name not in BASE:
            raise SystemExit(f"Unknown assumption {name!r}. Known: {sorted(BASE)}")
        BASE[name] = float(val)

    if args.self_test:
        raise SystemExit(self_test())

    if args.reuse_sweep:
        reuses = [float(x) for x in args.reuse_sweep.split(",")]
        print("\n" + "=" * 78)
        print("WEIGHT-FETCH TERM vs REUSE  -- the single most contested number")
        print("=" * 78)
        print(f"{'reuse':>10}{'SRAM fetch pJ/MAC':>22}{'ReRAM':>10}"
              f"{'SRAM total adv.':>18}")
        print("-" * 78)
        for r in reuses:
            b = digital_baseline(args.weights, weight_reuse=r)
            mv = b["movement"]
            print(f"{r:>10.0f}{mv['sram_fetch_pj_per_mac']:>22.6f}"
                  f"{0.0:>10.1f}{mv['sram_fetch_pj_per_mac']:>18.6f}")
        print("-" * 78)
        print("The fetch advantage falls as 1/reuse. Quoting reuse=1 alone")
        print("overstates the in-memory-compute case; quote the curve.")
        print()
    print_baseline(digital_baseline(args.weights, weight_reuse=args.reuse))
    if args.baseline_only:
        return
    sweep(args)


if __name__ == "__main__":
    main()
