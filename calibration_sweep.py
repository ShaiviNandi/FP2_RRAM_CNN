#!/usr/bin/env python3
"""
ReRAM RESET-pulse calibration sweep.

Rather than hand-picking one pulse amplitude/width at a time, this script
drives ngspice through a grid of (amplitude, width) combinations for a
single rram_v_1_0_0 cell, and for each one:
  1. Applies the RESET pulse (program phase) through a LOW-IMPEDANCE write
     path (confirmed necessary -- see the session notes below).
  2. Applies a small, brief READ pulse (compute phase) to measure the
     resulting cell resistance without disturbing it further.
  3. Reports the resulting resistance and a rough magnitude-level
     classification (near-LRS / intermediate / near-HRS).

WHY A LOW-IMPEDANCE WRITE PATH: this model's cell current is exponential
in Vtb (Itb = I0*exp(-gap/g0)*sinh(Vtb/V0), V0=0.25V), so the cell's own
terminal voltage self-limits, diode-like, no matter how hard you drive the
supply, UNLESS the series (sense/write) resistance is small enough that it
doesn't eat the applied voltage. Confirmed experimentally in this session:
1kOhm sense resistor -> no observable RESET even at -8V supply. 20Ohm
write-path resistor -> clear ~390x resistance change at just -3V. This
script defaults to a 20Ohm write-path resistor for that reason; the READ
path in a final design would typically use a separate, larger sense
resistor for good read margin -- this script isn't trying to design that
read circuit, just characterize the cell's write response.

WHAT THIS DOES NOT MODEL: thermal accumulation across repeated pulses
(each sweep point starts from a fresh gap_ini), and it does not attempt to
find a true "3-level" write-verify recipe automatically -- it reports the
calibration DATA so you can pick pulse parameters for HRS / mid / LRS
targets, that decision is left to you (and should probably be re-checked
against your actual target R_HRS/R_LRS spec, which this script doesn't
know).

Usage:
    python3 calibration_sweep.py --osdi rram_v_1_0_0.osdi \
        --widths 5 10 20 40 80 160 --voltages -1.0 -1.5 -2.0 -2.5 -3.0

Requires: ngspice (built with OSDI support, per this session's build) on
PATH, or pass --ngspice-bin /path/to/ngspice.
"""
import argparse
import subprocess
import sys
import os
import tempfile
import shutil


NETLIST_TEMPLATE = """\
* Auto-generated single-cell RESET calibration test
* width={width_ns}ns  amplitude={voltage}V  r_program={r_program}ohm

.control
pre_osdi {osdi_path}
.endc

* Program phase: 0 to {prog_end}n, RESET pulse
* Compute/read phase: {prog_end}n to {total_end}n, small non-disturb read
Vprog wl 0 PWL(0 0 0.05n {voltage} {prog_minus}n {voltage} {prog_end}n 0 {read_start}n {vread} {total_end}n {vread})

Rprog bl 0 {r_program}

N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran {tstep}p {total_end}n
.control
run
wrdata {results_file} i(Vprog) v(wl) v(bl)
.endc

.end
"""


def build_netlist(osdi_path, width_ns, voltage, r_program, vread, results_file):
    prog_end = width_ns
    prog_minus = max(width_ns - 0.05, 0.001)
    read_start = width_ns + 0.05
    total_end = width_ns + 2.0  # 2ns read window after programming
    tstep = max(1, width_ns * 1000 // 2000)  # coarser step for long pulses, ps units
    return NETLIST_TEMPLATE.format(
        width_ns=width_ns, voltage=voltage, r_program=r_program,
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
    """Return (time, i_vprog, v_wl, v_bl) from the last line of wrdata output."""
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
    ap.add_argument("--widths", type=float, nargs="+", default=[5, 10, 20, 40, 80, 160],
                     help="RESET pulse widths to sweep, in ns (default: 5 10 20 40 80 160)")
    ap.add_argument("--voltages", type=float, nargs="+", default=[-1.0, -1.5, -2.0, -2.5, -3.0],
                     help="RESET pulse amplitudes to sweep, in V (default: -1.0 -1.5 -2.0 -2.5 -3.0)")
    ap.add_argument("--r-program", type=float, default=20.0,
                     help="Write-path series resistance in ohms (default 20, confirmed necessary this session -- 1k showed no effect)")
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

    workdir = tempfile.mkdtemp(prefix="rram_sweep_")
    print(f"Working directory: {workdir}")
    print(f"OSDI model: {osdi_path}")
    print(f"Write-path resistance: {args.r_program} ohm  |  Read voltage: {args.vread} V")
    print()

    # Baseline run: width=0 (no programming pulse at all) to establish the
    # reference LRS resistance from this model's default gap_ini state.
    print("Running baseline (no RESET pulse) to establish reference LRS resistance...")
    baseline_results = "baseline.txt"
    baseline_netlist = build_netlist(osdi_path, 0.05, 0.0, args.r_program, args.vread, baseline_results)
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
                r_lrs_ref = abs(args.vread / i) - args.r_program
    if r_lrs_ref is None:
        print("  WARNING: baseline run failed or gave near-zero current; falling back to 500 ohm assumed reference.")
        print(f"  ngspice stderr: {err.strip()[:300]}")
        r_lrs_ref = 500.0
    print(f"  Reference LRS resistance (no RESET applied): {r_lrs_ref:.1f} ohm")
    print()

    print("=" * 100)
    print(f"{'Width(ns)':>10} {'Volt(V)':>8} {'ProgV_cell(V)':>14} {'I_read(A)':>12} {'R_cell(ohm)':>12} {'R/R_LRS':>9}  Classification")
    print("=" * 100)

    results = []
    for width in args.widths:
        for voltage in args.voltages:
            results_file = f"w{width}_v{abs(voltage)}.txt".replace(".", "p")
            netlist = build_netlist(osdi_path, width, voltage, args.r_program, args.vread, results_file)
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

            r_cell = abs(args.vread / i_read) - args.r_program
            # Estimate the voltage that actually landed on the cell during
            # programming (approximation using the cell's LRS-state
            # resistance at the start of the program pulse, since the exact
            # dynamic op point during programming isn't in this row -- this
            # is a rough indicator for interpreting the sweep, not a
            # precise per-run measurement).
            prog_v_est = voltage * r_lrs_ref / (r_lrs_ref + args.r_program)

            ratio = r_cell / r_lrs_ref
            cls = classify(r_cell, r_lrs_ref, args.r_hrs_threshold)
            results.append((width, voltage, r_cell, ratio, cls))
            print(f"{width:10.1f} {voltage:8.2f} {prog_v_est:14.3f} {i_read:12.3e} {r_cell:12.1f} {ratio:9.2f}  {cls}")

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
            print("Candidate pulse settings for an intermediate ('0.5 magnitude') target:")
            for w, v, r, ratio, cls in sorted(intermediate, key=lambda x: abs(x[2] - (r_lrs_ref + args.r_hrs_threshold) / 2)):
                print(f"    width={w}ns, amplitude={v}V  ->  R={r:.1f} ohm ({ratio:.2f}x LRS reference)")
        else:
            print("No sweep points landed in the 'intermediate' band with the current")
            print(f"--r-hrs-threshold ({args.r_hrs_threshold} ohm). Either narrow the sweep")
            print("around the transition (the jump from near-LRS to near-HRS may be sharp")
            print("given how steep this model's switching kinetics are), or adjust")
            print("--r-hrs-threshold if your actual target intermediate resistance differs.")
        print()
        print("NOTE: 'ProgV_cell(V)' above is an ESTIMATE using the LRS-state divider ratio,")
        print("not a direct per-run measurement (the compute-phase read row doesn't carry")
        print("the programming-phase operating point). Treat it as directional, not exact --")
        print("if you need the exact voltage trajectory during programming, add a wrdata")
        print("point at t=width_ns/2 to the netlist template instead of only the final row.")

    if not args.keep_files:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"\nNetlists and results kept in: {workdir}")


if __name__ == "__main__":
    main()
