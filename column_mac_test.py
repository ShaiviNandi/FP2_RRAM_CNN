#!/usr/bin/env python3
"""
N-cell ReRAM crossbar COLUMN MAC validation.

Unlike the earlier 2-cell calibration tests (each cell had its own private
bitline+resistor, i.e. they never actually summed anything), this builds a
real crossbar column: all N cells share ONE bitline node and ONE sense
resistor, with each cell's other terminal driven independently. This is
what makes KCL actually do the accumulate.

WHY the GOLDEN MODEL ISN'T A NAIVE SUM: because all cells share one finite
sense resistor (not an ideal virtual-ground TIA), the shared bitline node
voltage depends on ALL cells simultaneously -- a coupled linear circuit,
not N independent dividers. This script computes THREE numbers so the
coupling error introduced by that finite resistor is visible, not hidden:
  1. "target" MAC value: the pure Sum(weight_i * activation_i), scaled by
     what current that should produce if the crossbar were ideal (ideal
     virtual-ground readout, Rsense -> 0).
  2. "circuit-exact" current: solved via real nodal analysis for this
     circuit's actual finite Rsense -- this is the correct ground truth
     for what ngspice should report, and will differ from (1) by an
     amount that grows with Rsense and with how many/how conductive the
     cells are (this IS a real, quantifiable non-ideality, related to but
     not identical to classic multi-row/column sneak-path -- this is
     single-column shared-sense-resistor loading).
  3. ngspice's actual simulated result -- compared against (2), not (1),
     since (2) is the correct target for this specific circuit topology.

WEIGHT -> RESISTANCE MAPPING: recalibrated 2026-07 session, WITH the 1T1R
select switch (Ron=1ohm) included in the calibration loop -- the original
bare-resistor calibration (143ns/200ns) was confirmed invalid once the
switch was added: the 143ns pulse over-RESET the cell to near-HRS instead
of landing at the intended intermediate resistance (Ron=1ohm was enough to
push this device's steep, exponentially voltage-sensitive switching past
its threshold). Recalibrated via calibration_sweep_with_switch.py, which
includes the same selmod Ron=1ohm switch used here:
    weight magnitude 1.0 -> width=0ns    (R ~ 542.8 ohm, LRS default state, unchanged)
    weight magnitude 0.5 -> width=161ns  (R ~ 1099.8 ohm, measured WITH switch)
    weight magnitude 0.0 -> width=220ns  (R ~ 218587.2 ohm, saturated HRS, WITH switch --
                                          numerically identical to the old bare-resistor
                                          value since at full saturation the tiny read
                                          current makes the extra 1ohm negligible; only
                                          the WIDTH needed to reach it changed, not R)
Sign is realized by flipping the COMPUTE-PHASE READ VOLTAGE polarity per
weight (not stored in the device) -- consistent with the sign-realization
decision flagged in the session summary document.

Usage:
    python3 column_mac_test.py --osdi rram_v_1_0_0.osdi         --weights 1.0 -1.0 0.5 -0.5 0.0 1.0 -0.5 0.5         --activations 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0
"""
import argparse
import subprocess
import sys
import os
import tempfile
import shutil

# CRITICAL FIX (2026-07): the RRAM compact model (or its OSDI compilation)
# has a pathological special case triggered when the driver voltage is held
# at LITERALLY 0.0V for an extended (multi-timestep) DC-like hold -- not
# present for a driver that merely touches 0V at a single instant during a
# continuous ramp. Confirmed via a fully isolated single-cell, no-switch
# test: a driver held at exact 0V before a RESET pulse causes the cell to
# saturate to near-HRS regardless of pulse width/timing (matching the long-
# standing "1T1R corrupts intermediate-state programming" symptom from
# earlier sessions); the same pulse preceded by a driver held at any
# nonzero voltage -- confirmed amplitude-independent from 1e-6V to 1e-3V --
# avoids the pathology entirely and lands close to the calibrated target.
# Fix: every idle "0V" hold in the driver PWL uses this tiny epsilon
# instead of literal 0, which is physically/electrically inert (cell
# current at this bias is negligible) but avoids the exact-zero special
# case. NOTE: even with this fix, a real multi-ns pre-pulse dwell (as
# happens for every row after the first in a real column) still measured
# ~1945ohm instead of the near-zero-dwell calibration's 1099.8ohm for the
# 0.5-magnitude target -- this may be a real dwell-dependent effect that
# needs its own recalibration pass (calibrate WITH a representative dwell,
# not just switch Ron), tracked separately from this fix.
DRIVER_IDLE_V = -1e-6

# Calibrated width lookup, RECALIBRATED 2026-07 with the select switch
# (Ron=1ohm) included in the loop -- see calibration_sweep_with_switch.py.
# The old 143ns/200ns bare-resistor values are confirmed invalid for this
# circuit (the 143ns pulse over-RESETs to near-HRS once Ron is present).
WIDTH_FOR_MAGNITUDE = {
    1.0: 0.0,      # no pulse -- default LRS state (unchanged)
    0.5: 161.0,    # measured R ~ 1099.8 ohm, WITH switch Ron=1ohm in loop
    0.0: 220.0,    # measured R ~ 218587.2 ohm (saturated), WITH switch
}
# Measured resistances at those calibration points (ohms) -- used for the
# golden model so it reflects reality, not an idealized guess. The 0.5
# value shifted slightly (1095.6 -> 1099.8); the 0.0 value is numerically
# unchanged from the bare-resistor calibration (saturation swamps the 1ohm
# difference), only its width changed.
R_FOR_MAGNITUDE = {
    1.0: 542.8,
    0.5: 1099.8,
    0.0: 218587.2,
}
PROGRAM_VOLTAGE = -1.5  # V, confirmed working amplitude this session


def build_column_netlist(osdi_path, weights, activations, r_program, r_sense,
                          vread_base, results_file, slot_duration=240.0, use_switch=True,
                          switch_settle_ns=0.04):
    """IMPORTANT: rows are programmed SEQUENTIALLY, with a per-row SELECT
    SWITCH (1T1R-style isolation) that is only closed during that row's own
    program slot and during the final simultaneous read. Two prior versions
    of this script were tried and found insufficient:
      v1: simultaneous programming of all rows -- massive sneak-path
          corruption (confirmed: the weight=0.0 cell showed current
          comparable to weight=1.0 cells).
      v2: sequential programming, but unselected rows only held at 0V (no
          isolation) -- STILL insufficient. Confirmed this session: even
          with correct sequencing, unselected LRS-state neighbor cells
          remain electrically tied to the shared bitline, forming parallel
          load paths that lower the effective series impedance seen by the
          actively-programming cell (quantified: ~1.26x lower than the
          isolated single-cell calibration assumed, for 7 LRS neighbors).
          Because this model's switching kinetics are exponentially
          sensitive to voltage (the sinh() drive term) and the calibrated
          143ns pulse already sits right at a steep threshold (185ns->46.8k
          ohm, 190ns->218.6k ohm per the calibration sweep), that small
          impedance shift was enough to push intermediate-target cells into
          full runaway saturation -- confirmed experimentally: ALL weight=
          +/-0.5 cells collapsed to HRS-like current in the v2 run.
    v3 (this version) adds an explicit per-row select switch (ngspice S
    device) between each cell and the shared bitline, open (very high
    impedance) except during that row's own program slot or the final read
    -- this is what a real 1T1R array's access transistor does, and is the
    standard reason ReRAM crossbars use 1T1R rather than bare 0T1R
    crosspoints for exactly this failure mode."""
    n = len(weights)
    lines = []
    lines.append(f"* {n}-cell ReRAM crossbar column: shared bitline, 1T1R-style select switches")
    lines.append("* Rows programmed SEQUENTIALLY with per-row isolation, read SIMULTANEOUSLY.")
    lines.append(".control")
    lines.append(f"pre_osdi {osdi_path}")
    lines.append(".endc")
    lines.append("")
    if use_switch:
        lines.append(".model selmod SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)")
        lines.append("")

    slot_duration = float(slot_duration)  # ns per row's program slot (now caller-configurable)
    total_program_time = n * slot_duration
    read_start = total_program_time + 0.05
    total_end = total_program_time + 2.0
    v_sel_on = 1.0   # closes the switch (> Vt=0.5)
    v_sel_off = 0.0  # opens the switch

    row_data = []
    for i, (w, a) in enumerate(zip(weights, activations)):
        mag = abs(w)
        if mag not in WIDTH_FOR_MAGNITUDE:
            raise ValueError(f"weight magnitude {mag} has no calibration entry; "
                              f"only {{0.0, 0.5, 1.0}} are calibrated this session")
        width = WIDTH_FOR_MAGNITUDE[mag]
        sign = -1.0 if w < 0 else 1.0
        vread = vread_base * sign * a

        slot_start = i * slot_duration

        # Driver PWL: stays at 0V through the whole program phase (its own
        # RESET pulse happens on its own timeline below), switches to the
        # read voltage only for the final simultaneous read.
        drv_points = [(0, DRIVER_IDLE_V), (total_program_time, DRIVER_IDLE_V), (read_start, vread), (total_end, vread)]
        drv_pwl = " ".join(f"{t}n {v}" for t, v in drv_points)

        # Select-switch control PWL: HIGH only during this row's own program
        # slot, and HIGH again during the final read (all rows reconnect
        # for simultaneous parallel read); LOW (isolated) everywhere else.
        #
        # CRITICAL: the switch must never be electrically open at exactly
        # t=0. Root-caused 2026-07: ngspice computes an initial transient
        # operating-point solve at t=0 before any PWL source starts moving;
        # if the switch is open at that instant, the RRAM device's internal
        # state gets initialized from a near-floating DC condition instead
        # of a normal closed-circuit one, and the compact model never
        # recovers from that bad initial state for the rest of the run --
        # no RESET pulse of any width/timing/voltage can move it, because
        # it isn't starting from a physically normal state to begin with.
        # This was confirmed by forcing Vsel to DC-closed and reproducing
        # the exact golden-model current; it's very likely the real cause
        # of the long-standing "1T1R select-switch corrupts intermediate-
        # state programming" issue from earlier sessions, not switch Ron
        # or pulse calibration. Fix: every row's switch starts CLOSED at
        # t=0, and only opens (if isolation is needed) shortly afterward --
        # at t=0.02ns, still well before the earliest possible pulse onset
        # at t=0.05ns on any row, so isolation is still fully intact by the
        # time any real programming voltage appears.
        sel_points = [(0, v_sel_on)]
        if slot_start > 0:
            sel_points.append((0.02, v_sel_off))
            sel_points.append((slot_start, v_sel_off))
        sel_points.append((slot_start + 0.01, v_sel_on))
        slot_end = slot_start + slot_duration
        if slot_end < total_program_time - 0.001:
            # More rows still need to program after this one -- isolate
            # this cell from their activity by opening the switch until
            # the final read.
            sel_points.append((slot_end - 0.01, v_sel_on))
            sel_points.append((slot_end, v_sel_off))
            sel_points.append((total_program_time, v_sel_off))
            sel_points.append((total_program_time + 0.01, v_sel_on))
        else:
            # This is the last row's program slot -- nothing left to
            # isolate from, so just stay closed straight through to the
            # read (no spurious open/close blip right at the boundary).
            sel_points.append((total_program_time, v_sel_on))
        sel_points.append((total_end, v_sel_on))
        sel_pwl = " ".join(f"{t}n {v}" for t, v in sel_points)

        # Reset-pulse PWL that only fires while this row's own select switch
        # is closed (during its own slot) -- riding on top of the driver.
        # The switch closes at slot_start+0.01ns (see sel_points above); the
        # stimulus then waits `switch_settle_ns` before the RESET pulse actually
        # starts ramping, instead of the previous hardcoded 0.04ns gap, so
        # this can be swept to test whether the switch's own Ron transition
        # needs more time to fully settle before the pulse hits the cell.
        pulse_start = slot_start + 0.01 + switch_settle_ns
        if width > 0.001:
            drv_points = [(0, DRIVER_IDLE_V)]
            if slot_start > 0:
                drv_points.append((slot_start, DRIVER_IDLE_V))
            drv_points.append((pulse_start, PROGRAM_VOLTAGE))
            drv_points.append((pulse_start + width - 0.05, PROGRAM_VOLTAGE))
            drv_points.append((pulse_start + width, DRIVER_IDLE_V))
            drv_points.append((slot_start + slot_duration, DRIVER_IDLE_V))
            drv_points.append((total_program_time, DRIVER_IDLE_V))
            drv_points.append((read_start, vread))
            drv_points.append((total_end, vread))
            drv_pwl = " ".join(f"{t}n {v}" for t, v in drv_points)

        row_data.append((i + 1, w, a, width, slot_start, drv_pwl, sel_pwl))

    for i, w, a, width, slot_start, drv_pwl, sel_pwl in row_data:
        lines.append(f"* Row {i}: target weight={w}, activation={a}, "
                      f"RESET width={width}ns, program slot starts at t={slot_start}ns")
        lines.append(f"Vrow{i} wl{i} 0 PWL({drv_pwl})")
        if use_switch:
            lines.append(f"Vsel{i} sel{i} 0 PWL({sel_pwl})")
            lines.append(f"N{i} wl{i} celltop{i} rram_model")
            lines.append(f"S{i} celltop{i} bl sel{i} 0 selmod")
        else:
            lines.append(f"* (switch bypassed for diagnostic -- direct connection)")
            lines.append(f"N{i} wl{i} bl rram_model")
        lines.append("")

    lines.append(f"Rsense bl 0 {r_sense}")
    lines.append(".model rram_model rram_v_1_0_0")
    lines.append("")
    lines.append(f".tran 50p {total_end}n")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"i(Vrow{i})" for i in range(1, n + 1))
    lines.append(f"wrdata {results_file} v(bl) {wrdata_vecs}")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")
    return "\n".join(lines)


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


def parse_last_row(results_path, n_cells):
    if not os.path.exists(results_path):
        return None
    last = None
    with open(results_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                last = parts
    if last is None:
        return None
    # columns: time v(bl) [time i(Vrow_k)]*n -- wrdata repeats time per vector
    v_bl = float(last[1])
    currents = []
    idx = 2
    for _ in range(n_cells):
        currents.append(float(last[idx + 1]))
        idx += 2
    return v_bl, currents


def golden_nodal_solve(weights, activations, r_sense, vread_base):
    """Exact linear nodal analysis for the shared-bitline circuit:
    V_bl * (1/Rsense + sum(1/R_i)) = sum(V_row_i / R_i)
    Returns (v_bl, i_total, per_cell_currents)."""
    sum_g = 1.0 / r_sense
    sum_v_over_r = 0.0
    r_list = []
    v_list = []
    for w, a in zip(weights, activations):
        mag = abs(w)
        r = R_FOR_MAGNITUDE[mag]
        sign = -1.0 if w < 0 else 1.0
        v = vread_base * sign * a
        r_list.append(r)
        v_list.append(v)
        sum_g += 1.0 / r
        sum_v_over_r += v / r
    v_bl = sum_v_over_r / sum_g
    i_total = v_bl / r_sense
    per_cell = [(v_list[i] - v_bl) / r_list[i] for i in range(len(weights))]
    return v_bl, i_total, per_cell


def ideal_target_current(weights, activations, vread_base):
    """Ideal virtual-ground (Rsense->0) target: I = vread_base * G_LRS *
    Sum(weight_i * activation_i). This is what a perfect TIA-based readout
    would give -- the 'textbook' MAC result, uncorrupted by shared-resistor
    loading."""
    g_lrs = 1.0 / R_FOR_MAGNITUDE[1.0]
    weighted_sum = sum(w * a for w, a in zip(weights, activations))
    i_ideal = vread_base * g_lrs * weighted_sum
    return i_ideal, weighted_sum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osdi", required=True)
    ap.add_argument("--weights", type=float, nargs="+", required=True,
                     help="Weight values, must each be in {0.0, 0.5, 1.0, -0.5, -1.0}")
    ap.add_argument("--activations", type=float, nargs="+", required=True,
                     help="Per-row activation values, same length as --weights")
    ap.add_argument("--r-program", type=float, default=20.0, help="Per-row write-path resistance during programming (ohm, unused once shared bitline exists -- kept for reference)")
    ap.add_argument("--r-sense", type=float, default=20.0, help="SHARED sense resistor for the whole column (ohm, default 20 -- same low-impedance philosophy as this session's confirmed write path)")
    ap.add_argument("--vread", type=float, default=0.1, help="Base read voltage magnitude (V, default 0.1)")
    ap.add_argument("--ngspice-bin", default="ngspice")
    ap.add_argument("--slot-duration", type=float, default=240.0,
                     help="ns per row's sequential program slot (default 240, margin above the "
                          "recalibrated 220ns HRS pulse width -- was 210/200ns before the "
                          "switch-in-loop recalibration). Lower this to test whether idle "
                          "dead-time before read matters.")
    ap.add_argument("--no-switch", action="store_true",
                     help="Bypass the 1T1R select switch entirely (direct cell-to-bitline connection). Diagnostic flag to isolate whether the switch itself is the cause of a discrepancy.")
    ap.add_argument("--switch-settle-ns", type=float, default=0.04,
                     help="Time (ns) to wait after the select switch closes before the RESET pulse "
                          "starts ramping (default 0.04, matching the original hardcoded gap that "
                          "produced the 100%% MAC discrepancy). Raise this substantially (e.g. 5, 20, "
                          "50) to test whether the switch's Ron transition needs more settling time "
                          "before the cell sees the full pulse.")
    ap.add_argument("--keep-files", action="store_true")
    args = ap.parse_args()

    if len(args.weights) != len(args.activations):
        print("ERROR: --weights and --activations must be the same length", file=sys.stderr)
        sys.exit(1)

    osdi_path = os.path.abspath(args.osdi)
    if not os.path.exists(osdi_path):
        print(f"ERROR: {osdi_path} not found", file=sys.stderr)
        sys.exit(1)

    n = len(args.weights)
    print(f"Column MAC test: N={n} cells, shared Rsense={args.r_sense} ohm, Vread_base={args.vread} V")
    print(f"Weights:     {args.weights}")
    print(f"Activations: {args.activations}")
    print(f"Switch-close-to-pulse-start settle time: {args.switch_settle_ns} ns "
          f"(switch closes at slot_start+0.01ns, pulse starts at slot_start+0.01+{args.switch_settle_ns}ns)")
    per_row_width = max(WIDTH_FOR_MAGNITUDE.get(abs(w), 0.0) for w in args.weights)
    margin = args.slot_duration - (0.01 + args.switch_settle_ns + per_row_width)
    if margin < 5.0:
        print(f"  WARNING: only {margin:.2f}ns of slot margin left after settle+pulse -- "
              f"raise --slot-duration when --switch-settle-ns is pushed higher.")
    print()

    # --- Golden computations ---
    i_ideal, weighted_sum = ideal_target_current(args.weights, args.activations, args.vread)
    v_bl_exact, i_exact, per_cell_exact = golden_nodal_solve(
        args.weights, args.activations, args.r_sense, args.vread)

    print("=" * 90)
    print("GOLDEN MODEL (computed independently, before running ngspice)")
    print("-" * 90)
    print(f"  Target MAC value  Sum(weight_i * activation_i) = {weighted_sum:.4f}")
    print(f"  (1) IDEAL target current (virtual-ground readout, Rsense->0): {i_ideal:.6e} A")
    print(f"  (2) CIRCUIT-EXACT current (real nodal solve, finite Rsense={args.r_sense}ohm): {i_exact:.6e} A")
    coupling_error_pct = 100.0 * abs(i_exact - i_ideal) / abs(i_ideal) if i_ideal != 0 else float('nan')
    print(f"      -> shared-Rsense loading error vs ideal: {coupling_error_pct:.2f}%")
    print(f"  V_bl (shared bitline node voltage, exact solve): {v_bl_exact:.6f} V")
    print("=" * 90)
    print()

    # --- Build and run the real ngspice simulation ---
    workdir = tempfile.mkdtemp(prefix="rram_column_")
    results_file = "column_results.txt"
    netlist = build_column_netlist(osdi_path, args.weights, args.activations,
                                    args.r_program, args.r_sense, args.vread, results_file,
                                    slot_duration=args.slot_duration, use_switch=not args.no_switch,
                                    switch_settle_ns=args.switch_settle_ns)
    netlist_path = os.path.join(workdir, "column_mac.sp")
    with open(netlist_path, "w") as f:
        f.write(netlist)
    if args.keep_files:
        print(f"Netlist written to: {netlist_path}")

    rc, out, err = run_ngspice(args.ngspice_bin, netlist_path, workdir)
    if rc != 0:
        print(f"ERROR running ngspice (rc={rc}):")
        print(err.strip()[:2000])
        sys.exit(1)

    row = parse_last_row(os.path.join(workdir, results_file), n)
    if row is None:
        print("ERROR: no output data parsed from ngspice run")
        sys.exit(1)
    v_bl_sim, currents_sim = row
    # NOTE: ngspice reports source current with the passive sign convention
    # (negative when the source is delivering power to the circuit) -- flip
    # sign here to compare against the golden model's "physical current into
    # the cell" convention. Same fix already applied in column_mac_kcl_only.py;
    # this script was missing it, so the --no-switch run above
    # showed ~200% "error" despite matching the golden model almost exactly.
    i_total_sim = -sum(currents_sim)

    print("NGSPICE SIMULATION RESULT (settled compute-phase values)")
    print("-" * 90)
    print(f"  V_bl (simulated): {v_bl_sim:.6f} V   (golden exact-solve predicted: {v_bl_exact:.6f} V)")
    print(f"  Per-cell currents (A, sign-corrected to physical convention): "
          f"{['%.4e' % (-c) for c in currents_sim]}")
    print(f"  Total bitline current I_total = Sum(per-cell currents) = {i_total_sim:.6e} A")
    print()
    err_vs_exact = 100.0 * abs(i_total_sim - i_exact) / abs(i_exact) if i_exact != 0 else float('nan')
    err_vs_ideal = 100.0 * abs(i_total_sim - i_ideal) / abs(i_ideal) if i_ideal != 0 else float('nan')
    print("=" * 90)
    print("COMPARISON")
    print("-" * 90)
    print(f"  ngspice vs CIRCUIT-EXACT golden model (correct target for this topology): {err_vs_exact:.2f}% error")
    print(f"  ngspice vs IDEAL virtual-ground target (textbook MAC, no loading):        {err_vs_ideal:.2f}% error")
    print()
    if err_vs_exact < 5.0:
        print("  PASS: ngspice matches the circuit-exact nodal solve within 5%. KCL summation")
        print("  across cells is behaving as this circuit's physics predicts.")
    else:
        print("  WARNING: ngspice deviates from the circuit-exact solve by more than 5%.")
        print("  This suggests either a modeling assumption here doesn't hold (e.g. R_i")
        print("  drifting during the brief read due to residual switching dynamics), or")
        print("  the per-cell resistances at read time differ from the calibration table")
        print("  values used in the golden model. Worth checking per-cell currents above")
        print("  against R_FOR_MAGNITUDE-derived expectations individually.")
    print()
    print(f"  Shared-resistor loading error (ideal vs exact, i.e. the crossbar non-ideality")
    print(f"  itself, independent of simulation accuracy): {coupling_error_pct:.2f}%")
    print("  This number grows with N and with how large a fraction of cells are in LRS")
    print("  (low resistance, closer to Rsense's own scale) -- this is the quantity to")
    print("  track as N scales up, ahead of the full sneak-path characterization.")

    if not args.keep_files:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
