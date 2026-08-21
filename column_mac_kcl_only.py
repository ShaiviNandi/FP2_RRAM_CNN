#!/usr/bin/env python3
"""
KCL summation validation using FIXED resistors at the already-calibrated
values, instead of re-simulating the dynamic program-then-read sequence.

WHY this PIVOT: the previous version tried to validate two different things
at once -- (A) does KCL correctly sum N cells' currents, and (B) does
sequential programming through a shared bitline correctly reach the
intended resistance for each cell. (B) turned out to have real, still
unresolved issues (a 1T1R select-switch that corrupts programming for
reasons not yet diagnosed). Chasing (B) was blocking any progress on (A),
which is what this step actually needs. This script isolates (A) cleanly:
it uses plain SPICE resistors set to the validated calibration values
(542.8 / 1095.6 / 218587.1 ohm for weight magnitudes 1.0 / 0.5 / 0.0), and
checks whether ngspice's linear solve matches the Python nodal-analysis
golden model for the shared-bitline circuit. This is a much lower-risk
check (ngspice solving a linear resistor network is about as
well-established as SPICE gets), but it is still worth confirming the
Python math agrees with an independent solver before trusting it further.

(B) -- whether real sequential programming through a shared bitline
actually reaches these resistances -- remains a real open item, to be
revisited deliberately (likely as part of the sneak-path/IR-drop
characterization step), not smuggled into this one.

Usage:
    python3 column_mac_kcl_only.py         --weights 1.0 -1.0 0.5 -0.5 0.0 1.0 -0.5 0.5         --activations 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
"""
import argparse
import subprocess
import sys
import os
import tempfile
import shutil

R_FOR_MAGNITUDE = {
    1.0: 542.8,
    0.5: 1095.6,
    0.0: 218587.1,
}


def build_netlist(weights, activations, r_sense, vread_base, results_file):
    n = len(weights)
    lines = ["* KCL validation: fixed calibrated resistors, shared bitline", ""]
    for i, (w, a) in enumerate(zip(weights, activations), start=1):
        mag = abs(w)
        if mag not in R_FOR_MAGNITUDE:
            raise ValueError(f"weight magnitude {mag} not in calibration table")
        r = R_FOR_MAGNITUDE[mag]
        sign = -1.0 if w < 0 else 1.0
        vread = vread_base * sign * a
        lines.append(f"* Row {i}: weight={w}, activation={a}, R={r} ohm")
        lines.append(f"Vrow{i} wl{i} 0 DC {vread}")
        lines.append(f"Rcell{i} wl{i} bl {r}")
        lines.append("")
    lines.append(f"Rsense bl 0 {r_sense}")
    lines.append("")
    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"i(Vrow{i})" for i in range(1, n + 1))
    lines.append(f"wrdata {results_file} v(bl) {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines)


def run_ngspice(ngspice_bin, netlist_path, workdir):
    try:
        result = subprocess.run([ngspice_bin, "-b", netlist_path],
                                 cwd=workdir, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"ngspice binary not found: {ngspice_bin}"
    except subprocess.TimeoutExpired:
        return -2, "", "timed out"


def parse_op_result(results_path, n_cells):
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        lines = [l for l in f if l.strip()]
    if not lines:
        return None
    last = lines[-1].split()
    v_bl = float(last[1])
    currents = []
    idx = 2
    for _ in range(n_cells):
        currents.append(float(last[idx + 1]))
        idx += 2
    return v_bl, currents


def golden_nodal_solve(weights, activations, r_sense, vread_base):
    sum_g = 1.0 / r_sense
    sum_v_over_r = 0.0
    r_list, v_list = [], []
    for w, a in zip(weights, activations):
        r = R_FOR_MAGNITUDE[abs(w)]
        sign = -1.0 if w < 0 else 1.0
        v = vread_base * sign * a
        r_list.append(r)
        v_list.append(v)
        sum_g += 1.0 / r
        sum_v_over_r += v / r
    v_bl = sum_v_over_r / sum_g
    i_total = v_bl / r_sense
    per_cell_physical = [(v_list[i] - v_bl) / r_list[i] for i in range(len(weights))]
    return v_bl, i_total, per_cell_physical


def ideal_target_current(weights, activations, vread_base):
    g_lrs = 1.0 / R_FOR_MAGNITUDE[1.0]
    weighted_sum = sum(w * a for w, a in zip(weights, activations))
    return vread_base * g_lrs * weighted_sum, weighted_sum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=float, nargs="+", required=True)
    ap.add_argument("--activations", type=float, nargs="+", required=True)
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--ngspice-bin", default="ngspice")
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()

    if len(args.weights) != len(args.activations):
        print("ERROR: --weights and --activations must match in length", file=sys.stderr)
        sys.exit(1)

    n = len(args.weights)
    print(f"KCL-only validation: N={n} cells, fixed calibrated resistors, Rsense={args.r_sense} ohm")
    print(f"Weights:     {args.weights}")
    print(f"Activations: {args.activations}")
    print()

    i_ideal, weighted_sum = ideal_target_current(args.weights, args.activations, args.vread)
    v_bl_exact, i_exact, per_cell_exact = golden_nodal_solve(args.weights, args.activations, args.r_sense, args.vread)

    print(f"Target MAC value Sum(weight*activation) = {weighted_sum:.4f}")
    print(f"Ideal (Rsense->0) target current:  {i_ideal:.6e} A")
    print(f"Circuit-exact golden current:      {i_exact:.6e} A  (V_bl={v_bl_exact:.6f} V)")
    print()

    workdir = tempfile.mkdtemp(prefix="rram_kcl_")
    results_file = "kcl_results.txt"
    netlist = build_netlist(args.weights, args.activations, args.r_sense, args.vread, results_file)
    netlist_path = os.path.join(workdir, "kcl.sp")
    with open(netlist_path, "w") as f:
        f.write(netlist)
    if args.keep_files:
        print(f"Netlist: {netlist_path}")

    rc, out, err = run_ngspice(args.ngspice_bin, netlist_path, workdir)
    if rc != 0:
        print(f"ERROR running ngspice: {err.strip()[:1000]}")
        sys.exit(1)

    row = parse_op_result(os.path.join(workdir, results_file), n)
    if row is None:
        print("ERROR: no data parsed")
        sys.exit(1)
    v_bl_sim, currents_sim = row
    # NOTE: ngspice reports source current with the passive sign convention
    # (negative when the source is delivering power to the circuit) -- flip
    # sign here to compare against the golden model's "physical current
    # into the cell" convention. This is the exact bug found this session
    # in the previous script's comparison logic.
    i_total_sim = -sum(currents_sim)

    print(f"ngspice V_bl: {v_bl_sim:.6f} V  (golden: {v_bl_exact:.6f} V)")
    print(f"ngspice I_total (sign-corrected): {i_total_sim:.6e} A  (golden circuit-exact: {i_exact:.6e} A)")
    err_pct = 100.0 * abs(i_total_sim - i_exact) / abs(i_exact) if i_exact != 0 else float('nan')
    print(f"Error vs circuit-exact golden model: {err_pct:.3f}%")
    if err_pct < 1.0:
        print("PASS: KCL summation across N cells matches the exact nodal-analysis")
        print("golden model to within 1%. This confirms ngspice and the Python golden")
        print("model agree on the shared-bitline circuit's physics.")
    else:
        print("WARNING: still a meaningful discrepancy -- worth checking per-cell values")
        print("individually before trusting this further.")

    if not args.keep_files:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
