#!/usr/bin/env python3
"""
ReRAM RESET-pulse calibration sweep -- WITH the 1T1R select switch in series.

WHY THIS EXISTS: the original calibration_sweep.py characterized each pulse
(width, voltage) against a bare 20-ohm write-path resistor. column_mac_test.py
later confirmed (2026-07 session) that with the select switch (Ron=1ohm) in
the loop, the previously-calibrated 143ns/-1.5V "intermediate" pulse instead
saturates the cell to near-HRS (~218.6kohm, same as the 200ns pulse) instead
of landing at the intended ~1095.6ohm. Root cause: this device's switching
kinetics are exponentially sensitive to the voltage actually reaching the
cell (sinh() drive term), and the 143ns point already sat right at a steep
threshold in the bare-resistor sweep (185ns->46.8kohm, 190ns->218.6kohm) --
adding even 1 extra ohm of series impedance (Ron) was enough to push it
over that cliff.

WHAT'S DIFFERENT HERE: identical sweep methodology to calibration_sweep.py,
except the write/read path now goes cell -> select switch (Ron=1, Roff=1e12,
same .model selmod SW(...) used in column_mac_test.py, always held closed
via Vsel) -> series resistor -> ground. This makes the calibration table
match the ACTUAL circuit topology used during real column programming, not
an idealized bare-resistor stand-in.

WHAT THIS DOES NOT MODEL: same caveats as the original script -- no thermal
accumulation across repeated pulses, no automatic write-verify search. Also
does not model neighbor-cell sneak paths (that's a separate, still-open
characterization step); this isolates just the switch-Ron effect on a
single cell.

Usage:
    python3 calibration_sweep_with_switch.py --osdi rram_v_1_0_0.osdi \\
        --widths 100 110 120 130 143 150 160 170 180 \\
        --voltages -1.5

Requires: ngspice (built with OSDI support) on PATH, or pass --ngspice-bin.
"""
import argparse
import subprocess
import sys
import os
import tempfile
import shutil


NETLIST_TEMPLATE = """\
* Auto-generated single-cell RESET calibration test, WITH select switch (Ron={r_on}ohm)
* width={width_ns}ns  amplitude={voltage}V  r_program={r_program}ohm  Ron={r_on}ohm

.control
pre_osdi {osdi_path}
.endc

.model selmod SW(Ron={r_on} Roff=1e12 Vt=0.5 Vh=0.1)

* Select switch held closed for the whole run -- this script isolates the
* steady-state Ron effect on calibration, not switch timing (already
* confirmed in the 2026-07 session that settle time is not the cause of
* the discrepancy; only Ron itself matters).
Vsel sel 0 DC 1.0

* Program phase: 0 to {prog_end}n, RESET pulse
* Compute/read phase: {prog_end}n to {total_end}n, small non-disturb read
Vprog wl 0 PWL(0 0 0.05n {voltage} {prog_minus}n {voltage} {prog_end}n 0 {read_start}n {vread} {total_end}n {vread})

N1 wl celltop rram_model
S1 celltop bl sel 0 selmod
Rprog bl 0 {r_program}

.model rram_model rram_v_1_0_0

.tran {tstep}p {total_end}n
.control
run
wrdata {results_file} i(Vprog) v(wl) v(bl)
.endc

.end
"""


def build_netlist(osdi_path, width_ns, voltage, r_program, r_on, vread, results_file):
    prog_end = width_ns
    prog_minus = max(width_ns - 0.05, 0.001)
    read_start = width_ns + 0.05
    total_end = width_ns + 2.0
    tstep = max(1, width_ns * 1000 // 2000)
    return NETLIST_TEMPLATE.format(
        width_ns=width_ns, voltage=voltage, r_program=r_program, r_on=r_on,
        prog_end=prog_end, prog_minus=prog_minus, read_start=read_start,
        total_end=total_end, vread=vread, tstep=tstep,
        osdi_path=osdi_path, results_file=results_file,
    )


def run_ngspice(ngspice_bin, netlist_path, workdir):
    try:
        result = subprocess.run(
            [ngspice_bin, "-b", netlist_path],
            cwd=workdir, capture_output=True, text=True, timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"ngspice binary not found: {ngspice_bin}"
    except subprocess.TimeoutExpired:
        return -2, "", "ngspice run timed out (60s)"


def parse_last_row(results_path):
    if not os.path.exists(results_path):
        return None
    last = None
    with open(results_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                last = parts
    if last is None:
        return None
    t, i, _, vwl, _, vbl = last[:6]
    return float(t), float(i), float(vwl), float(vbl)


def classify(r_ohm, r_lrs_ref, r_hrs_threshold):
    if r_ohm <= 2 * r_lrs_ref:
        return "near-LRS (~weight 1.0)"
    elif r_ohm >= r_hrs_threshold:
        return "near-HRS (~weight 0.0)"
    else:
        return "intermediate (~weight 0.5 candidate)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osdi", required=True, help="Path to compiled rram_v_1_0_0.osdi")
    ap.add_argument("--widths", type=float, nargs="+", default=[100, 110, 120, 130, 143, 150, 160, 170, 180],
                     help="RESET pulse widths to sweep, in ns -- default zooms in around the old "
                          "bare-resistor 143ns point, since that's the region that flipped to HRS "
                          "once Ron was added")
    ap.add_argument("--voltages", type=float, nargs="+", default=[-1.5],
                     help="RESET pulse amplitudes to sweep, in V (default: -1.5, the confirmed-working "
                          "amplitude from the previous session)")
    ap.add_argument("--r-program", type=float, default=20.0,
                     help="Write-path series resistance in ohms, in series AFTER the select switch "
                          "(default 20, matching column_mac_test.py's Rsense)")
    ap.add_argument("--r-on", type=float, default=1.0,
                     help="Select switch Ron in ohms (default 1, matching selmod in column_mac_test.py)")
    ap.add_argument("--vread", type=float, default=0.1,
                     help="Read voltage for the compute-phase measurement (default 0.1V)")
    ap.add_argument("--ngspice-bin", default="ngspice", help="Path to ngspice binary (default: 'ngspice' on PATH)")
    ap.add_argument("--r-lrs-ref", type=float, default=None,
                     help="Reference LRS resistance for classification (default: auto, from the 0V/no-pulse baseline run)")
    ap.add_argument("--r-hrs-threshold", type=float, default=50000.0,
                     help="Resistance above which a result is classified near-HRS (default 50kOhm)")
    ap.add_argument("--keep-files", action="store_true", help="Keep generated netlists/results instead of deleting the temp dir")
    args = ap.parse_args()

    osdi_path = os.path.abspath(args.osdi)
    if not os.path.exists(osdi_path):
        print(f"ERROR: {osdi_path} not found", file=sys.stderr)
        sys.exit(1)

    workdir = tempfile.mkdtemp(prefix="rram_sweep_switch_")
    print(f"Working directory: {workdir}")
    print(f"OSDI model: {osdi_path}")
    print(f"Select switch: Ron={args.r_on} ohm (Roff=1e12, held closed throughout)")
    print(f"Write-path resistance (after switch): {args.r_program} ohm  |  Read voltage: {args.vread} V")
    print(f"Total series impedance during read: ~{args.r_on + args.r_program} ohm "
          f"(vs. {args.r_program} ohm in the original bare-resistor calibration)")
    print()

    print("Running baseline (no RESET pulse) to establish reference LRS resistance...")
    baseline_results = "baseline.txt"
    baseline_netlist = build_netlist(osdi_path, 0.05, 0.0, args.r_program, args.r_on, args.vread, baseline_results)
    netlist_path = os.path.join(workdir, "baseline.sp")
    with open(netlist_path, "w") as f:
        f.write(baseline_netlist)
    rc, out, err = run_ngspice(args.ngspice_bin, netlist_path, workdir)
    r_lrs_ref = args.r_lrs_ref
    if rc == 0:
        row = parse_last_row(os.path.join(workdir, baseline_results))
        if row:
            t, i, vwl, vbl = row
            if abs(i) > 1e-15:
                r_lrs_ref = abs(args.vread / i) - args.r_program - args.r_on
    if r_lrs_ref is None:
        print("  WARNING: baseline run failed or gave near-zero current; falling back to 500 ohm assumed reference.")
        print(f"  ngspice stderr: {err.strip()[:300]}")
        r_lrs_ref = 500.0
    print(f"  Reference LRS resistance (no RESET applied): {r_lrs_ref:.1f} ohm")
    print()

    print("=" * 100)
    print(f"{'Width(ns)':>10} {'Volt(V)':>8} {'I_read(A)':>12} {'R_cell(ohm)':>12} {'R/R_LRS':>9}  Classification")
    print("=" * 100)

    results = []
    for width in args.widths:
        for voltage in args.voltages:
            results_file = f"w{width}_v{abs(voltage)}.txt".replace(".", "p")
            netlist = build_netlist(osdi_path, width, voltage, args.r_program, args.r_on, args.vread, results_file)
            netlist_path = os.path.join(workdir, f"net_{results_file}.sp")
            with open(netlist_path, "w") as f:
                f.write(netlist)

            rc, out, err = run_ngspice(args.ngspice_bin, netlist_path, workdir)
            if rc != 0:
                print(f"{width:10.1f} {voltage:8.2f}  ERROR running ngspice: {err.strip()[:60]}")
                continue

            row = parse_last_row(os.path.join(workdir, results_file))
            if row is None:
                print(f"{width:10.1f} {voltage:8.2f}  ERROR: no output data parsed")
                continue

            t, i_read, vwl, vbl = row
            if abs(i_read) < 1e-15:
                print(f"{width:10.1f} {voltage:8.2f}  WARNING: read current ~0, cell may be in extreme HRS or sim diverged")
                continue

            # Subtract BOTH the series resistor AND the switch Ron to
            # isolate the cell's own resistance -- this is the key
            # difference vs. the bare-resistor script.
            r_cell = abs(args.vread / i_read) - args.r_program - args.r_on

            ratio = r_cell / r_lrs_ref
            cls = classify(r_cell, r_lrs_ref, args.r_hrs_threshold)
            results.append((width, voltage, r_cell, ratio, cls))
            print(f"{width:10.1f} {voltage:8.2f} {i_read:12.3e} {r_cell:12.1f} {ratio:9.2f}  {cls}")

    print("=" * 100)
    print()

    if results:
        print("SUMMARY / INTERPRETATION")
        print("-" * 100)
        r_values = [r[2] for r in results]
        print(f"Resistance range observed across the sweep: {min(r_values):.1f} ohm to {max(r_values):.1f} ohm")
        near_lrs = [r for r in results if "near-LRS" in r[4]]
        near_hrs = [r for r in results if "near-HRS" in r[4]]
        intermediate = [r for r in results if "intermediate" in r[4]]
        print(f"  near-LRS points:      {len(near_lrs)}")
        print(f"  intermediate points:  {len(intermediate)}")
        print(f"  near-HRS points:      {len(near_hrs)}")
        print()
        if intermediate:
            print("Candidate pulse settings for an intermediate ('0.5 magnitude') target,")
            print("NOW VALID for use with the real select-switch-equipped column circuit:")
            for w, v, r, ratio, cls in sorted(intermediate, key=lambda x: abs(x[2] - (r_lrs_ref + args.r_hrs_threshold) / 2)):
                print(f"    width={w}ns, amplitude={v}V  ->  R={r:.1f} ohm ({ratio:.2f}x LRS reference)")
            print()
            print("After selecting a candidate above, update WIDTH_FOR_MAGNITUDE and")
            print("R_FOR_MAGNITUDE in column_mac_test.py (the 0.5 entries) to these new")
            print("values -- the old 143ns/1095.6ohm entries were calibrated WITHOUT the")
            print("switch and are now confirmed invalid for that circuit.")
        else:
            print("No sweep points landed in the 'intermediate' band with the current")
            print(f"--r-hrs-threshold ({args.r_hrs_threshold} ohm). Given how sharp the")
            print("cliff was in the bare-resistor sweep, a finer width step may be needed")
            print("(e.g. 1ns increments between 100-140ns) to find a new stable landing")
            print("point with Ron now in the loop -- the cliff itself likely shifted to")
            print("an earlier width, not just changed height.")

    if not args.keep_files:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"\nNetlists and results kept in: {workdir}")


if __name__ == "__main__":
    main()
