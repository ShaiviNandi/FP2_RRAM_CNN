#!/usr/bin/env python3
"""
rlrs_tradeoff.py
================================================================================
Where should R_LRS sit? One table combining what the device costs in silicon
against what it buys in signal.

THE TENSION
-----------
R_LRS pulls two ways at once, which is why a single operating point cannot be
defended and a sweep is the actual result.

  LOW  R_LRS -> large read current
               -> wide 1T1R select transistor -> large cell, large chip
               -> large R_s * G_col           -> large loading error
               -> but strong signal for the ADC to resolve

  HIGH R_LRS -> small current, compact cell, small loading error
               -> but weak signal, so ADC quantisation and read noise matter
                  more

The per-column calibration removes the loading ERROR but not the CURRENT. The
current still flows, still dissipates, still sets the transistor width. So the
calibration does not fix R_LRS: it removes accuracy from the list of reasons
to pick a high one, leaving area and signal.

WHAT THIS SCRIPT REPORTS AT EACH R_LRS
--------------------------------------
  from first principles   read current per cell, column current, the select
                          transistor width NeuroSim would demand, and the
                          uncorrected loading attenuation
  from analog_eval        end-to-end CIFAR-10 accuracy, raw and calibrated,
                          plus readout SNR             (--accuracy)
  from NeuroSim           chip area, energy, TOPS/W, TOPS/mm2   (--neurosim)

The first block always runs and needs nothing installed. The other two are
opt-in because they are slow.

USAGE
-----
    python3 rlrs_tradeoff.py --self-test
    python3 rlrs_tradeoff.py                      # analysis only, instant
    python3 rlrs_tradeoff.py --accuracy --max-images 2000
    NEUROSIM_DIR=~/DNN_NeuroSim_V1.4 python3 rlrs_tradeoff.py \\
        --accuracy --neurosim --out-csv rlrs_tradeoff.csv
================================================================================
"""
import argparse
import csv
import math
import os
import subprocess
import sys

# Device and circuit constants, matching ARCHITECTURE.md.
R_MID_RATIO = 1099.8 / 542.8      # MID sits at this multiple of LRS
R_HRS_DEFAULT = 218587.2
# On/off ratio of the measured device. Sweeping R_LRS with R_HRS FIXED silently
# destroys the format: at R_LRS = 50k the ratio falls to 4.4 and the three FP2
# levels stop being distinguishable, so even the digital baseline collapses to
# chance. That is a real device failure, not a simulation bug, but it is not the
# experiment intended. Holding the RATIO constant sweeps the device the way a
# process engineer would actually move it.
ONOFF_RATIO = 218587.2 / 542.8
R_SENSE, VREAD = 20.0, 0.1

# NeuroSim's own sizing anchor: at R_LRS = 542.8 ohm it demands a 127.28F
# select transistor against a 12F default. Width scales with the current the
# cell must pass, hence inversely with resistance.
ANCHOR_R, ANCHOR_W = 542.8, 127.28
DEFAULT_CELL_W = 12.0


# =============================================================================
def analytic(r_lrs, m_rows, r_hrs=R_HRS_DEFAULT, r_sense=R_SENSE, vread=VREAD,
             frac_on=0.4):
    """First-principles consequences of choosing this R_LRS.

    frac_on is the fraction of cells in a column sitting in a conducting state.
    Post-ReLU sparsity and the FP2 level distribution put this near 0.4; it is
    a parameter rather than a constant because the answer depends on it.
    """
    g_lrs, g_mid, g_hrs = 1.0 / r_lrs, 1.0 / (r_lrs * R_MID_RATIO), 1.0 / r_hrs
    i_cell = vread * g_lrs

    # Column conductance: conducting cells split between LRS and MID, the rest
    # at HRS. This is the term the calibration divides out.
    n_on = m_rows * frac_on
    g_col = 0.5 * n_on * g_lrs + 0.5 * n_on * g_mid + (m_rows - n_on) * g_hrs
    attenuation = 1.0 / (1.0 + r_sense * g_col)

    # Select transistor width, scaled from NeuroSim's own anchor point.
    tx_width_f = ANCHOR_W * (ANCHOR_R / r_lrs)

    return dict(
        r_lrs=r_lrs,
        i_cell_ua=i_cell * 1e6,
        i_col_ma=n_on * i_cell * 1e3,
        g_col_ms=g_col * 1e3,
        retained_pct=100.0 * attenuation,
        loading_err_pct=100.0 * (1.0 - attenuation),
        tx_width_f=tx_width_f,
        cell_fits_default=tx_width_f <= DEFAULT_CELL_W,
        # Relative cell area, taking width as the driver at fixed height.
        rel_cell_area=tx_width_f / DEFAULT_CELL_W,
    )


# =============================================================================
def run_accuracy(r_lrs, args):
    """End-to-end accuracy at this R_LRS via analog_eval.

    analog_eval derives its resistances from module constants, so they are
    overridden in a subprocess rather than mutated in-process. Keeping each
    point in its own process also means one failure cannot corrupt the rest.
    """
    # G_LRS is derived from R_LRS at import time and is used to convert
    # bitline current back into a weight value:
    #     out = i / (vread * G_LRS) * block_scale
    # Setting R_LRS alone leaves G_LRS stale, rescaling every output by the
    # ratio of old to new resistance and driving all modes to chance, including
    # the digital baseline, which contains no crossbar. Both must be set.
    code = (
        "import sys, analog_eval as ae;"
        f"ae.R_LRS={r_lrs}; ae.R_MID={r_lrs * R_MID_RATIO}; "
        f"ae.R_HRS={getattr(args, '_r_hrs_now', args.r_hrs or R_HRS_DEFAULT)}; "
        f"ae.G_LRS={1.0 / r_lrs};"
        "sys.argv=['analog_eval','--sweep',str(%d),'--data-dir',%r,"
        "'--max-images',str(%d),'--adc-bits',str(%d),'--r-sense',str(%f)];"
        "ae.main()" % (args.block, args.data_dir, args.max_images,
                       args.adc_bits, args.r_sense)
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    open(f"rlrs_acc_{r_lrs:g}.log", "w").write(out)
    import re
    acc = {}
    # Match analog_eval's actual printed lines, which read e.g.
    #   "FP2 digital-exact (baseline)          :  92.81%  (4s)"
    #   "ANALOG raw readout                    :  11.53%  (9s)"
    #   "ANALOG + per-column gain calibration  :  92.42%  (7s)"
    for key, pat in (("ideal", r"digital-exact[^:]*:\s*([\d.]+)%"),
                     ("raw", r"ANALOG raw readout[^:]*:\s*([\d.]+)%"),
                     ("calibrated",
                      r"ANALOG \+ per-column[^:]*:\s*([\d.]+)%")):
        m = re.findall(pat, out)
        acc[key] = float(m[-1]) if m else None
    if not any(acc.values()):
        print(f"    [warn] no accuracy parsed at R={r_lrs:g}; "
              f"see rlrs_acc_{r_lrs:g}.log")
    elif acc.get("ideal") is not None and acc["ideal"] < 20.0:
        # The digital baseline contains no crossbar, so it cannot legitimately
        # sit at chance. If it does, a derived constant is stale and every
        # number from this point is meaningless.
        print(f"    [ERROR] digital baseline at {acc['ideal']:.2f}% at "
              f"R={r_lrs:g} -- chance. If the on/off ratio here is small "
              f"the DEVICE has failed and that is a real result; if the ratio "
              f"is healthy a derived constant is stale. Check the log.")
        acc = {k: None for k in acc}
    return acc


def run_neurosim(r_lrs, args):
    """NeuroSim area and energy at this R_LRS, via neurosim_compare."""
    if not os.path.isfile("neurosim_compare.py"):
        print("    [warn] neurosim_compare.py not found; skipping")
        return {}
    csv_out = f"rlrs_ns_{r_lrs:g}.csv"
    cmd = [sys.executable, "neurosim_compare.py", "--cells", "rram",
           "--resistance-on", str(r_lrs), "--cell-width", "0",
           "--wl-weight", "2", "--adc-bits", str(args.adc_bits),
           "--out-csv", csv_out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    open(f"rlrs_ns_{r_lrs:g}.log", "w").write(r.stdout + r.stderr)
    if not os.path.isfile(csv_out):
        print(f"    [warn] NeuroSim produced no CSV at R={r_lrs:g}")
        return {}
    rows = list(csv.DictReader(open(csv_out)))
    return rows[0] if rows else {}


# =============================================================================
def self_test():
    print("=" * 64)
    print("SELF-TEST")
    print("=" * 64)
    lo, hi = analytic(542.8, 128), analytic(20000.0, 128)

    print(f"  [1] current falls with resistance: "
          f"{lo['i_cell_ua']:.1f} -> {hi['i_cell_ua']:.1f} uA")
    assert lo["i_cell_ua"] > hi["i_cell_ua"]

    print(f"  [2] transistor narrows with resistance: "
          f"{lo['tx_width_f']:.1f}F -> {hi['tx_width_f']:.1f}F")
    assert lo["tx_width_f"] > hi["tx_width_f"]

    print(f"  [3] loading error shrinks with resistance: "
          f"{lo['loading_err_pct']:.1f}% -> {hi['loading_err_pct']:.1f}%")
    assert lo["loading_err_pct"] > hi["loading_err_pct"]

    # The anchor must reproduce NeuroSim's own reported number.
    a = analytic(ANCHOR_R, 128)
    print(f"  [4] anchor reproduces NeuroSim: {a['tx_width_f']:.2f}F "
          f"(NeuroSim reported {ANCHOR_W}F)")
    assert abs(a["tx_width_f"] - ANCHOR_W) < 0.01

    print(f"  [5] loading error grows with tile height: "
          f"M=32 {analytic(542.8,32)['loading_err_pct']:.1f}%  "
          f"M=256 {analytic(542.8,256)['loading_err_pct']:.1f}%")
    assert analytic(542.8, 256)["loading_err_pct"] > \
           analytic(542.8, 32)["loading_err_pct"]

    print("\n" + "=" * 64)
    print("SELF-TEST PASSED")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="542.8,1000,2000,6000,20000,50000,100000")
    ap.add_argument("--block", type=int, default=128, help="tile height M")
    ap.add_argument("--r-hrs", type=float, default=None,
                    help="Fix R_HRS at this value. Default is to scale it with "
                         "R_LRS so the on/off ratio stays at the measured "
                         "403 -- otherwise high R_LRS collapses the ratio and "
                         "the format stops working for reasons unrelated to "
                         "the loading error under study.")
    ap.add_argument("--r-sense", type=float, default=R_SENSE)
    ap.add_argument("--frac-on", type=float, default=0.4)
    ap.add_argument("--adc-bits", type=int, default=6)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--accuracy", action="store_true",
                    help="Run analog_eval at each point (slow).")
    ap.add_argument("--neurosim", action="store_true",
                    help="Run NeuroSim at each point (slow, rebuilds).")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    rows = []
    for r in [float(x) for x in args.sweep.split(",")]:
        # Scale HRS with LRS unless pinned, holding the on/off ratio fixed.
        r_hrs = args.r_hrs if args.r_hrs else r * ONOFF_RATIO
        args._r_hrs_now = r_hrs
        d = analytic(r, args.block, r_hrs, args.r_sense, frac_on=args.frac_on)
        d["r_hrs"] = r_hrs
        d["onoff"] = r_hrs / r
        if args.accuracy:
            print(f"  accuracy at R_LRS={r:g} ...", flush=True)
            d.update({f"acc_{k}": v for k, v in run_accuracy(r, args).items()})
        if args.neurosim:
            print(f"  NeuroSim at R_LRS={r:g} ...", flush=True)
            ns = run_neurosim(r, args)
            for k in ("chip_area_um2", "dyn_energy_pj", "tops_per_w",
                      "tops_per_mm2", "latency_ns"):
                v = ns.get(k)
                d[k] = float(v) if v else None
        rows.append(d)

    print("\n" + "=" * 96)
    print(f"R_LRS TRADE-OFF   M={args.block}, R_sense={args.r_sense:g} ohm, "
          f"{100*args.frac_on:.0f}% of cells conducting")
    print("=" * 96)
    hdr = (f"{'R_LRS':>9}{'I_cell':>9}{'I_col':>9}{'tx w':>8}{'cell':>7}"
           f"{'loading':>10}")
    if args.accuracy:
        hdr += f"{'raw %':>9}{'calib %':>9}"
    if args.neurosim:
        hdr += f"{'area mm2':>11}{'TOPS/W':>9}{'TOPS/mm2':>10}"
    print(hdr)
    print(f"{'ohm':>9}{'uA':>9}{'mA':>9}{'F':>8}{'x12F':>7}{'err %':>10}")
    print("-" * 96)
    for d in rows:
        line = (f"{d['r_lrs']:>9.0f}{d['i_cell_ua']:>9.1f}{d['i_col_ma']:>9.2f}"
                f"{d['tx_width_f']:>8.1f}{d['rel_cell_area']:>7.1f}"
                f"{d['loading_err_pct']:>10.1f}")
        if args.accuracy:
            line += (f"{d.get('acc_raw') or float('nan'):>9.2f}"
                     f"{d.get('acc_calibrated') or float('nan'):>9.2f}")
        if args.neurosim:
            a = d.get("chip_area_um2")
            line += (f"{(a*1e-6 if a else float('nan')):>11.2f}"
                     f"{d.get('tops_per_w') or float('nan'):>9.2f}"
                     f"{d.get('tops_per_mm2') or float('nan'):>10.4f}")
        print(line)
    print("-" * 96)

    fits = [d for d in rows if d["cell_fits_default"]]
    if fits:
        r_fit = min(d["r_lrs"] for d in fits)
        print(f"Smallest R_LRS that fits the default 12F cell: {r_fit:,.0f} ohm.")
    print("Below that the select transistor, not the resistor, sets cell area.")
    print()
    print("HOW TO READ THIS")
    print("  Moving right buys area and loses signal. The calibration makes")
    print("  the loading column irrelevant to ACCURACY -- calibrated accuracy")
    print("  should stay flat across the sweep -- so R_LRS can be chosen on")
    print("  area and ADC-resolvability alone. That freedom is the")
    print("  contribution; without the correction the raw column forces a")
    print("  high resistance for accuracy reasons.")

    if args.out_csv and rows:
        keys = sorted({k for d in rows for k in d})
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
