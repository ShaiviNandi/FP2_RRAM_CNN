#!/usr/bin/env python3
"""
M x K differential (2T2R) ReRAM crossbar array: matrix-vector multiply,
and (via im2col) convolution -- built directly on the validated single-
column MAC unit in column_mac_test.py / column_mac_kcl_only.py.

SIGNED WEIGHTS -- WHY DIFFERENTIAL (2T2R) NOW
    The single-cell tests this session got signed weights by flipping the
    READ voltage's polarity per cell. That only works in isolation -- on a
    real shared wordline, every cell on that row sees the SAME physical
    voltage at the same instant, so +V cannot be applied to one cell and -V
    to its neighbor. Real signed crossbars (and this project's own name,
    EvoCIM-2T2R) use a differential pair per weight instead: R+ and R-.
    Whichever side represents the sign gets the calibrated magnitude
    resistance; the OTHER side is driven fully HRS ("off", magnitude 0).
    Reading BOTH bitline currents and subtracting (I+ - I-) reproduces the
    signed weighted sum using only one shared positive read voltage per
    row -- exactly as a real 2T2R array works.

DATAFLOW -- WEIGHT-STATIONARY (same two-phase split as this session)
    Phase 1, PROGRAM (once): decompose every weight into (R+, R-), then
    sequentially program every physical cell using the exact calibrated
    pulse recipe validated in column_mac_test.py (WIDTH_FOR_MAGNITUDE,
    R_FOR_MAGNITUDE, the DRIVER_IDLE_V epsilon fix, switch-closed-at-t=0).
    THIS FILE DOES NOT YET GENERATE THAT DYNAMIC PROGRAMMING NETLIST for
    the full array -- see the note in build_array_read_netlist() below for
    why, and what the next step looks like.

    Phase 2, READ (many times, cheap): once cells are programmed, reuse
    them for every new activation vector -- e.g. once per output pixel's
    receptive-field patch in a convolution -- WITHOUT reprogramming. This
    file implements Phase 2 as a fixed-resistor netlist (mirroring
    column_mac_kcl_only.py exactly, just generalized to K columns and
    differential pairs), because validating the ARRAY MATH first, before
    paying for full dynamic-programming simulation cost across many cells,
    is the same order of operations this whole session followed for one
    column.

Usage (matmul demo):
    python3 crossbar_array_test.py --demo-matmul

Usage (convolution demo):
    python3 crossbar_array_test.py --demo-conv
"""
import argparse
import sys

import column_mac_test as col

# FP2-E1M0 magnitude set this project quantizes weights to.
FP2_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def quantize_to_fp2(w):
    """Snap a float weight to the nearest representable FP2-E1M0 level."""
    return min(FP2_LEVELS, key=lambda lvl: abs(lvl - w))


def decompose_differential(w):
    """Return (width_p, width_n, r_p, r_n) for one signed weight using
    2T2R differential encoding. The magnitude side uses the calibrated
    pulse width; the OFF side always uses the 0.0-magnitude (full HRS)
    width, since an unused polarity must be explicitly driven off, not
    left at its arbitrary default state."""
    w_q = quantize_to_fp2(w)
    mag = abs(w_q)
    off_width = col.WIDTH_FOR_MAGNITUDE[0.0]
    off_r = col.R_FOR_MAGNITUDE[0.0]
    mag_width = col.WIDTH_FOR_MAGNITUDE[mag]
    mag_r = col.R_FOR_MAGNITUDE[mag]
    if w_q >= 0:
        return mag_width, off_width, mag_r, off_r
    else:
        return off_width, mag_width, off_r, mag_r


def build_array_read_netlist(W, activations, r_sense=20.0, vread_base=0.1,
                              results_file="array_results.txt"):
    """Fixed-resistor READ-ONLY netlist for an M x K differential crossbar,
    given a weight matrix W (list of M rows, each a list of K weights) and
    an activation vector (length M). Mirrors column_mac_kcl_only.py's
    approach exactly, generalized: one shared wordline per row feeds ALL
    K columns' differential cell pairs in parallel (this IS the real
    crossbar topology -- no per-column driver duplication), and each
    column has its own positive/negative bitline with its own Rsense.

    NOT YET IMPLEMENTED: the dynamic PROGRAMMING netlist that actually
    RESETs each of the 2*M*K physical cells to reach these resistances.
    That's a direct generalization of column_mac_test.py's per-row program
    slots to per-(row, column, polarity) slots (2*M*K sequential program
    operations instead of M) -- happy to build that next once this array
    MATH is confirmed correct, same order this session validated the
    single column: KCL-only first (this), dynamic programming second.
    """
    M = len(activations)
    K = len(W[0])
    lines = ["* M x K differential (2T2R) crossbar -- fixed-resistor READ-ONLY", ""]
    for i in range(M):
        v = vread_base * activations[i]
        lines.append(f"Vrow{i} wl{i} 0 DC {v}")
        for k in range(K):
            _, _, r_p, r_n = decompose_differential(W[i][k])
            lines.append(f"Rcell{i}_{k}p wl{i} blp{k} {r_p}")
            lines.append(f"Rcell{i}_{k}n wl{i} bln{k} {r_n}")
        lines.append("")
    for k in range(K):
        lines.append(f"Rsensep{k} blp{k} 0 {r_sense}")
        lines.append(f"Rsensen{k} bln{k} 0 {r_sense}")
    lines.append("")
    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"v(blp{k}) v(bln{k})" for k in range(K))
    lines.append(f"wrdata {results_file} {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines)


def golden_array_matmul(W, activations, r_sense=20.0, vread_base=0.1):
    """Exact nodal-analysis golden model for the same M x K differential
    crossbar (circuit-exact, includes shared-Rsense loading -- same
    caveat column_mac_kcl_only.py documented: this loading error grows
    with M and with how many cells are near-LRS). Columns are electrically
    independent given a shared wordline voltage (no cross-column coupling
    in this topology), so each column is solved separately.
    Returns a list of K differential currents (I+ - I-), one per output."""
    M = len(activations)
    K = len(W[0])
    results = []
    for k in range(K):
        sum_g_p = 1.0 / r_sense
        sum_v_over_r_p = 0.0
        sum_g_n = 1.0 / r_sense
        sum_v_over_r_n = 0.0
        for i in range(M):
            _, _, r_p, r_n = decompose_differential(W[i][k])
            v = vread_base * activations[i]
            sum_g_p += 1.0 / r_p
            sum_v_over_r_p += v / r_p
            sum_g_n += 1.0 / r_n
            sum_v_over_r_n += v / r_n
        v_blp = sum_v_over_r_p / sum_g_p
        v_bln = sum_v_over_r_n / sum_g_n
        i_diff = (v_blp - v_bln) / r_sense
        results.append(i_diff)
    return results


def ideal_target_currents(W, activations, vread_base=0.1):
    """Ideal (Rsense->0) textbook matrix-vector product, scaled into
    current units by the LRS conductance -- the same 'target MAC value'
    concept as column_mac_kcl_only.py, generalized to K outputs."""
    g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
    M = len(activations)
    K = len(W[0])
    out = []
    for k in range(K):
        s = sum(activations[i] * quantize_to_fp2(W[i][k]) for i in range(M))
        out.append(vread_base * g_lrs * s)
    return out


# ---------------------------------------------------------------------------
# Convolution via im2col: reduces conv2d to repeated matmul calls against
# the SAME programmed weight matrix (weight-stationary reuse).
# ---------------------------------------------------------------------------

def im2col_patches(input_fmap, kernel_h, kernel_w, stride=1):
    """input_fmap: nested list, shape (C_in, H, W).
    Returns (patches, out_h, out_w) where each patch is a flat list of
    length C_in*kernel_h*kernel_w, flattened in (channel, row, col) order
    -- this order MUST match kernels_to_weight_matrix()'s flatten order."""
    C_in = len(input_fmap)
    H = len(input_fmap[0])
    Wd = len(input_fmap[0][0])
    out_h = (H - kernel_h) // stride + 1
    out_w = (Wd - kernel_w) // stride + 1
    patches = []
    for oy in range(out_h):
        for ox in range(out_w):
            patch = []
            for c in range(C_in):
                for ky in range(kernel_h):
                    for kx in range(kernel_w):
                        patch.append(input_fmap[c][oy * stride + ky][ox * stride + kx])
            patches.append(patch)
    return patches, out_h, out_w


def kernels_to_weight_matrix(kernels):
    """kernels: list of C_out kernels, each shape (C_in, kh, kw).
    Returns W as a list of M rows x C_out columns (M = C_in*kh*kw) -- each
    COLUMN is one flattened output-channel kernel, flattened in the SAME
    (channel, row, col) order as im2col_patches()."""
    flat_kernels = []
    for kern in kernels:
        flat = [v for c in kern for row in c for v in row]
        flat_kernels.append(flat)
    M = len(flat_kernels[0])
    C_out = len(kernels)
    W = [[flat_kernels[k][i] for k in range(C_out)] for i in range(M)]
    return W


def conv2d_via_crossbar(input_fmap, kernels, stride=1, r_sense=20.0, vread_base=0.1):
    """Maps a full conv2d onto the crossbar via im2col + weight-stationary
    reuse: kernels load ONCE as the K = C_out columns; every output
    pixel's receptive-field patch streams through as a new activation
    vector against the SAME programmed weights -- no reprogramming per
    pixel, matching real CiM weight-stationary dataflow.
    Returns (outputs, out_h, out_w) where outputs[p] is a length-C_out
    list of golden differential currents for output pixel p."""
    kh = len(kernels[0][0])
    kw = len(kernels[0][0][0])
    W = kernels_to_weight_matrix(kernels)
    patches, out_h, out_w = im2col_patches(input_fmap, kh, kw, stride)
    outputs = [golden_array_matmul(W, patch, r_sense, vread_base) for patch in patches]
    return outputs, out_h, out_w


def build_array_program_and_read_netlist(osdi_path, W, activations, r_sense=20.0,
                                          vread_base=0.1, slot_duration=240.0,
                                          switch_settle_ns=0.04,
                                          results_file="array_dynamic_results.txt"):
    """Full dynamic program-then-read netlist for the M x K x 2 (differential)
    crossbar. Generalizes column_mac_test.py's per-row program-then-read
    pattern to 2*M*K sequential program slots (one per row/column/polarity
    triple), with every fix validated this session baked in from the start:
      - switches start CLOSED at t=0 (never open at t=0 -- the root cause
        of the "1T1R corrupts intermediate-state programming" bug),
      - idle driver holds use DRIVER_IDLE_V (tiny nonzero epsilon), never
        literal 0V (the second root cause found this session),
      - pulse widths/resistances come from column_mac_test.py's
        switch-in-loop recalibrated WIDTH_FOR_MAGNITUDE / R_FOR_MAGNITUDE.

    KEY STRUCTURAL POINT: the driver is genuinely shared per physical row
    (one wordline drives all K columns), but each row's driver PWL must
    now carry 2K separate pulses in sequence -- one per cell that row
    owns -- instead of the single-column case's at-most-one pulse.

    Programming order is fully sequential (row-major: for each row, for
    each column, for each polarity) -- 2*M*K total program slots. Real
    hardware would parallelize column programming within a row-select
    window; this sticks to strict sequential slots for a direct,
    unambiguous generalization of the already-validated single-column
    approach. After all programming, every row's driver switches to its
    activation-scaled READ voltage and every switch closes simultaneously
    for one parallel read across all M rows and K columns."""
    M = len(activations)
    K = len(W[0])
    PROGRAM_VOLTAGE = col.PROGRAM_VOLTAGE
    DRIVER_IDLE_V = col.DRIVER_IDLE_V

    cell_order = []
    for i in range(M):
        for k in range(K):
            w_p, w_n, _, _ = decompose_differential(W[i][k])
            cell_order.append((i, k, "p", w_p))
            cell_order.append((i, k, "n", w_n))

    total_cells = len(cell_order)
    total_program_time = total_cells * slot_duration
    read_start = total_program_time + 0.05
    total_end = total_program_time + 2.0

    slots_by_row = {i: [] for i in range(M)}
    for idx, (i, k, pol, width) in enumerate(cell_order):
        slots_by_row[i].append((idx * slot_duration, k, pol, width))

    lines = [
        "* M x K x 2 differential (2T2R) crossbar -- full dynamic program-then-read",
        f"* {M} rows x {K} columns x 2 polarities = {total_cells} physical cells",
        f"* {total_cells} sequential program slots of {slot_duration}ns each"
        f" ({total_program_time}ns total program time)",
        ".control",
        f"pre_osdi {osdi_path}",
        ".endc",
        "",
        ".model selmod SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)",
        ".model rram_model rram_v_1_0_0",
        "",
    ]

    # --- Per-row shared driver PWL: idle epsilon except during THIS row's
    # own program slots, then a single read pulse at the very end. ---
    for i in range(M):
        pts = [(0, DRIVER_IDLE_V)]
        for slot_start, k, pol, width in slots_by_row[i]:
            if width > 0.001:
                pulse_start = slot_start + 0.01 + switch_settle_ns
                if slot_start > 0:
                    pts.append((slot_start, DRIVER_IDLE_V))
                pts.append((pulse_start, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width - 0.05, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width, DRIVER_IDLE_V))
            # width ~ 0 (default LRS state, e.g. magnitude 1.0): no pulse
            # needed, idle carries straight through this slot.
        pts.append((total_program_time, DRIVER_IDLE_V))
        v = vread_base * activations[i]
        pts.append((read_start, v))
        pts.append((total_end, v))
        pwl = " ".join(f"{t}n {vv}" for t, vv in pts)
        lines.append(f"Vrow{i} wl{i} 0 PWL({pwl})")
    lines.append("")

    # --- Per-cell device + select switch: closed at t=0 (avoids the
    # initial-operating-point bug), opens shortly after if isolation is
    # needed before its own slot, closes for its own pulse, then isolates
    # again until the final simultaneous read. ---
    for idx, (i, k, pol, width) in enumerate(cell_order):
        slot_start = idx * slot_duration
        slot_end = slot_start + slot_duration
        bl_node = f"blp{k}" if pol == "p" else f"bln{k}"
        cid = f"{i}_{k}{pol}"

        sel_pts = [(0, 1.0)]
        if slot_start > 0:
            sel_pts.append((0.02, 0.0))
            sel_pts.append((slot_start, 0.0))
        sel_pts.append((slot_start + 0.01, 1.0))
        if slot_end < total_program_time - 0.001:
            sel_pts.append((slot_end - 0.01, 1.0))
            sel_pts.append((slot_end, 0.0))
            sel_pts.append((total_program_time, 0.0))
            sel_pts.append((total_program_time + 0.01, 1.0))
        else:
            sel_pts.append((total_program_time, 1.0))
        sel_pts.append((total_end, 1.0))
        sel_pwl = " ".join(f"{t}n {v}" for t, v in sel_pts)

        lines.append(f"N{cid} wl{i} celltop{cid} rram_model")
        lines.append(f"S{cid} celltop{cid} {bl_node} sel{cid} 0 selmod")
        lines.append(f"Vsel{cid} sel{cid} 0 PWL({sel_pwl})")
        lines.append("")

    for k in range(K):
        lines.append(f"Rsensep{k} blp{k} 0 {r_sense}")
        lines.append(f"Rsensen{k} bln{k} 0 {r_sense}")
    lines.append("")
    lines.append(f".tran 50p {total_end}n")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"v(blp{k}) v(bln{k})" for k in range(K))
    lines.append(f"wrdata {results_file} {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines)


def build_array_program_and_read_netlist_1t1r(osdi_path, W, activations, r_sense=20.0,
                                               vread_base=0.1, slot_duration=240.0,
                                               switch_settle_ns=0.04,
                                               results_file="array_1t1r_dynamic_results.txt"):
    """1T1R (single-ended) counterpart to build_array_program_and_read_netlist:
    M*K sequential program slots instead of 2*M*K (no polarity dimension,
    no differential pair), one bitline per column instead of two. Same
    fixes baked in (switch closed at t=0, DRIVER_IDLE_V epsilon holds,
    switch-in-loop calibration). Negative weights are clipped to 0 by
    decompose_single_ended -- this netlist will faithfully reproduce that
    real limitation, not paper over it."""
    M = len(activations)
    K = len(W[0])
    PROGRAM_VOLTAGE = col.PROGRAM_VOLTAGE
    DRIVER_IDLE_V = col.DRIVER_IDLE_V

    cell_order = []
    for i in range(M):
        for k in range(K):
            width, _ = decompose_single_ended(W[i][k])
            cell_order.append((i, k, width))

    total_cells = len(cell_order)
    total_program_time = total_cells * slot_duration
    read_start = total_program_time + 0.05
    total_end = total_program_time + 2.0

    slots_by_row = {i: [] for i in range(M)}
    for idx, (i, k, width) in enumerate(cell_order):
        slots_by_row[i].append((idx * slot_duration, k, width))

    lines = [
        "* M x K 1T1R (single-ended, unsigned) crossbar -- full dynamic program-then-read",
        f"* {M} rows x {K} columns = {total_cells} physical cells (half of the 2T2R equivalent)",
        ".control",
        f"pre_osdi {osdi_path}",
        ".endc",
        "",
        ".model selmod SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)",
        ".model rram_model rram_v_1_0_0",
        "",
    ]

    for i in range(M):
        pts = [(0, DRIVER_IDLE_V)]
        for slot_start, k, width in slots_by_row[i]:
            if width > 0.001:
                pulse_start = slot_start + 0.01 + switch_settle_ns
                if slot_start > 0:
                    pts.append((slot_start, DRIVER_IDLE_V))
                pts.append((pulse_start, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width - 0.05, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width, DRIVER_IDLE_V))
        pts.append((total_program_time, DRIVER_IDLE_V))
        v = vread_base * activations[i]
        pts.append((read_start, v))
        pts.append((total_end, v))
        pwl = " ".join(f"{t}n {vv}" for t, vv in pts)
        lines.append(f"Vrow{i} wl{i} 0 PWL({pwl})")
    lines.append("")

    for idx, (i, k, width) in enumerate(cell_order):
        slot_start = idx * slot_duration
        slot_end = slot_start + slot_duration
        cid = f"{i}_{k}"

        sel_pts = [(0, 1.0)]
        if slot_start > 0:
            sel_pts.append((0.02, 0.0))
            sel_pts.append((slot_start, 0.0))
        sel_pts.append((slot_start + 0.01, 1.0))
        if slot_end < total_program_time - 0.001:
            sel_pts.append((slot_end - 0.01, 1.0))
            sel_pts.append((slot_end, 0.0))
            sel_pts.append((total_program_time, 0.0))
            sel_pts.append((total_program_time + 0.01, 1.0))
        else:
            sel_pts.append((total_program_time, 1.0))
        sel_pts.append((total_end, 1.0))
        sel_pwl = " ".join(f"{t}n {v}" for t, v in sel_pts)

        lines.append(f"N{cid} wl{i} celltop{cid} rram_model")
        lines.append(f"S{cid} celltop{cid} bl{k} sel{cid} 0 selmod")
        lines.append(f"Vsel{cid} sel{cid} 0 PWL({sel_pwl})")
        lines.append("")

    for k in range(K):
        lines.append(f"Rsense{k} bl{k} 0 {r_sense}")
    lines.append("")
    lines.append(f".tran 50p {total_end}n")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"v(bl{k})" for k in range(K))
    lines.append(f"wrdata {results_file} {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines)



def build_conv_dynamic_netlist(osdi_path, kernels, input_fmap, stride=1, r_sense=20.0,
                                vread_base=0.1, slot_duration=240.0, switch_settle_ns=0.04,
                                read_window_ns=2.0, results_file="conv_dynamic_results.txt"):
    """Full dynamic program-then-MULTI-read netlist for convolution via
    im2col + weight-stationary reuse. This is ONE continuous ngspice
    transient (device state only persists within a single simulation):
    kernels are programmed ONCE using the same 2*M*K sequential program
    slots as build_array_program_and_read_netlist(), then EVERY output
    pixel's receptive-field patch gets its own short READ window
    immediately after -- switches stay closed throughout all read windows
    (no reprogramming between patches), row driver voltages step to a
    new patch's activation values at each window boundary.

    Returns (netlist_text, read_window_times, out_h, out_w, K) where
    read_window_times is [(patch_index, window_start_ns, window_end_ns), ...]
    for locating each patch's settled output in the results file afterward."""
    kh = len(kernels[0][0])
    kw = len(kernels[0][0][0])
    W = kernels_to_weight_matrix(kernels)
    patches, out_h, out_w = im2col_patches(input_fmap, kh, kw, stride)
    M = len(patches[0])
    K = len(kernels)
    n_patches = len(patches)

    PROGRAM_VOLTAGE = col.PROGRAM_VOLTAGE
    DRIVER_IDLE_V = col.DRIVER_IDLE_V

    cell_order = []
    for i in range(M):
        for k in range(K):
            w_p, w_n, _, _ = decompose_differential(W[i][k])
            cell_order.append((i, k, "p", w_p))
            cell_order.append((i, k, "n", w_n))
    total_cells = len(cell_order)
    total_program_time = total_cells * slot_duration

    slots_by_row = {i: [] for i in range(M)}
    for idx, (i, k, pol, width) in enumerate(cell_order):
        slots_by_row[i].append((idx * slot_duration, k, pol, width))

    read_window_times = []
    t = total_program_time + 0.05
    for p in range(n_patches):
        w_start = t
        w_end = t + read_window_ns
        read_window_times.append((p, w_start, w_end))
        # Tiny gap (not w_end itself) before the next window starts --
        # otherwise consecutive windows share an identical PWL timestamp
        # whenever the voltage changes between them, which ngspice warns
        # about ("non-increasing PWL time points") and handles undefined.
        t = w_end + 0.01
    total_end = t + 1.0

    lines = [
        f"* Conv2d via crossbar: {M} rows x {K} cols x 2 pol = {total_cells} cells,"
        f" {n_patches} patches ({out_h}x{out_w} output), weight-stationary",
        ".control",
        f"pre_osdi {osdi_path}",
        ".endc",
        "",
        ".model selmod SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)",
        ".model rram_model rram_v_1_0_0",
        "",
    ]

    # --- Per-row driver: program pulses first (same as matmul version),
    # then step through one read voltage per patch, back to back. ---
    for i in range(M):
        pts = [(0, DRIVER_IDLE_V)]
        for slot_start, k, pol, width in slots_by_row[i]:
            if width > 0.001:
                pulse_start = slot_start + 0.01 + switch_settle_ns
                if slot_start > 0:
                    pts.append((slot_start, DRIVER_IDLE_V))
                pts.append((pulse_start, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width - 0.05, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width, DRIVER_IDLE_V))
        pts.append((total_program_time, DRIVER_IDLE_V))
        for p, w_start, w_end in read_window_times:
            v = vread_base * patches[p][i]
            pts.append((w_start, v))
            pts.append((w_end, v))
        pwl = " ".join(f"{t_}n {vv}" for t_, vv in pts)
        lines.append(f"Vrow{i} wl{i} 0 PWL({pwl})")
    lines.append("")

    # --- Per-cell device + select switch: identical logic to
    # build_array_program_and_read_netlist -- closed at t=0, isolates
    # until its own program slot, then stays closed straight through
    # ALL read windows once programming is done (no reprogramming). ---
    for idx, (i, k, pol, width) in enumerate(cell_order):
        slot_start = idx * slot_duration
        slot_end = slot_start + slot_duration
        bl_node = f"blp{k}" if pol == "p" else f"bln{k}"
        cid = f"{i}_{k}{pol}"

        sel_pts = [(0, 1.0)]
        if slot_start > 0:
            sel_pts.append((0.02, 0.0))
            sel_pts.append((slot_start, 0.0))
        sel_pts.append((slot_start + 0.01, 1.0))
        if slot_end < total_program_time - 0.001:
            sel_pts.append((slot_end - 0.01, 1.0))
            sel_pts.append((slot_end, 0.0))
            sel_pts.append((total_program_time, 0.0))
            sel_pts.append((total_program_time + 0.01, 1.0))
        else:
            sel_pts.append((total_program_time, 1.0))
        sel_pts.append((total_end, 1.0))
        sel_pwl = " ".join(f"{t}n {v}" for t, v in sel_pts)

        lines.append(f"N{cid} wl{i} celltop{cid} rram_model")
        lines.append(f"S{cid} celltop{cid} {bl_node} sel{cid} 0 selmod")
        lines.append(f"Vsel{cid} sel{cid} 0 PWL({sel_pwl})")
        lines.append("")

    for k in range(K):
        lines.append(f"Rsensep{k} blp{k} 0 {r_sense}")
        lines.append(f"Rsensen{k} bln{k} 0 {r_sense}")
    lines.append("")
    lines.append(f".tran 50p {total_end}n")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"v(blp{k}) v(bln{k})" for k in range(K))
    lines.append(f"wrdata {results_file} {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines), read_window_times, out_h, out_w, K


def build_conv_dynamic_netlist_1t1r(osdi_path, kernels, input_fmap, stride=1, r_sense=20.0,
                                     vread_base=0.1, slot_duration=240.0, switch_settle_ns=0.04,
                                     read_window_ns=2.0, results_file="conv_1t1r_dynamic_results.txt"):
    """1T1R (single-ended) counterpart to build_conv_dynamic_netlist: M*K
    program slots instead of 2*M*K, one bitline per column. Same
    weight-stationary multi-read-window structure. Negative kernel
    weights are clipped to 0 (decompose_single_ended), reproducing the
    real 1T1R sign limitation in the actual simulated netlist."""
    kh = len(kernels[0][0])
    kw = len(kernels[0][0][0])
    W = kernels_to_weight_matrix(kernels)
    patches, out_h, out_w = im2col_patches(input_fmap, kh, kw, stride)
    M = len(patches[0])
    K = len(kernels)
    n_patches = len(patches)

    PROGRAM_VOLTAGE = col.PROGRAM_VOLTAGE
    DRIVER_IDLE_V = col.DRIVER_IDLE_V

    cell_order = []
    for i in range(M):
        for k in range(K):
            width, _ = decompose_single_ended(W[i][k])
            cell_order.append((i, k, width))
    total_cells = len(cell_order)
    total_program_time = total_cells * slot_duration

    slots_by_row = {i: [] for i in range(M)}
    for idx, (i, k, width) in enumerate(cell_order):
        slots_by_row[i].append((idx * slot_duration, k, width))

    read_window_times = []
    t = total_program_time + 0.05
    for p in range(n_patches):
        w_start = t
        w_end = t + read_window_ns
        read_window_times.append((p, w_start, w_end))
        t = w_end + 0.01
    total_end = t + 1.0

    lines = [
        f"* Conv2d via crossbar (1T1R): {M} rows x {K} cols = {total_cells} cells,"
        f" {n_patches} patches ({out_h}x{out_w} output), weight-stationary",
        ".control",
        f"pre_osdi {osdi_path}",
        ".endc",
        "",
        ".model selmod SW(Ron=1 Roff=1e12 Vt=0.5 Vh=0.1)",
        ".model rram_model rram_v_1_0_0",
        "",
    ]

    for i in range(M):
        pts = [(0, DRIVER_IDLE_V)]
        for slot_start, k, width in slots_by_row[i]:
            if width > 0.001:
                pulse_start = slot_start + 0.01 + switch_settle_ns
                if slot_start > 0:
                    pts.append((slot_start, DRIVER_IDLE_V))
                pts.append((pulse_start, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width - 0.05, PROGRAM_VOLTAGE))
                pts.append((pulse_start + width, DRIVER_IDLE_V))
        pts.append((total_program_time, DRIVER_IDLE_V))
        for p, w_start, w_end in read_window_times:
            v = vread_base * patches[p][i]
            pts.append((w_start, v))
            pts.append((w_end, v))
        pwl = " ".join(f"{t_}n {vv}" for t_, vv in pts)
        lines.append(f"Vrow{i} wl{i} 0 PWL({pwl})")
    lines.append("")

    for idx, (i, k, width) in enumerate(cell_order):
        slot_start = idx * slot_duration
        slot_end = slot_start + slot_duration
        cid = f"{i}_{k}"

        sel_pts = [(0, 1.0)]
        if slot_start > 0:
            sel_pts.append((0.02, 0.0))
            sel_pts.append((slot_start, 0.0))
        sel_pts.append((slot_start + 0.01, 1.0))
        if slot_end < total_program_time - 0.001:
            sel_pts.append((slot_end - 0.01, 1.0))
            sel_pts.append((slot_end, 0.0))
            sel_pts.append((total_program_time, 0.0))
            sel_pts.append((total_program_time + 0.01, 1.0))
        else:
            sel_pts.append((total_program_time, 1.0))
        sel_pts.append((total_end, 1.0))
        sel_pwl = " ".join(f"{t}n {v}" for t, v in sel_pts)

        lines.append(f"N{cid} wl{i} celltop{cid} rram_model")
        lines.append(f"S{cid} celltop{cid} bl{k} sel{cid} 0 selmod")
        lines.append(f"Vsel{cid} sel{cid} 0 PWL({sel_pwl})")
        lines.append("")

    for k in range(K):
        lines.append(f"Rsense{k} bl{k} 0 {r_sense}")
    lines.append("")
    lines.append(f".tran 50p {total_end}n")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"v(bl{k})" for k in range(K))
    lines.append(f"wrdata {results_file} {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines), read_window_times, out_h, out_w, K


def parse_conv_dynamic_results_generic(results_path, read_window_times, K, r_sense=20.0, differential=True):
    """Generalized version of parse_conv_dynamic_results: works for both
    2T2R (differential=True, 2 vectors/col: blp,bln) and 1T1R
    (differential=False, 1 vector/col: bl)."""
    vecs_per_col = 2 if differential else 1
    last_in_window = {p: None for p, _, _ in read_window_times}
    with open(results_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 * (vecs_per_col * K):
                continue
            try:
                t_ns = float(parts[0]) * 1e9
            except ValueError:
                continue
            vals = [float(parts[j]) for j in range(1, len(parts), 2)]
            for p, w_start, w_end in read_window_times:
                if w_start <= t_ns <= w_end:
                    last_in_window[p] = vals
    outputs = []
    for p, _, _ in read_window_times:
        vals = last_in_window[p]
        if vals is None:
            outputs.append(None)
            continue
        if differential:
            diffs = [(vals[2 * k] - vals[2 * k + 1]) / r_sense for k in range(K)]
        else:
            diffs = [vals[k] / r_sense for k in range(K)]
        outputs.append(diffs)
    return outputs


def parse_conv_dynamic_results(results_path, read_window_times, K, r_sense=20.0):
    """Backward-compatible name for the 2T2R (differential) case --
    thin wrapper around parse_conv_dynamic_results_generic."""
    return parse_conv_dynamic_results_generic(results_path, read_window_times, K, r_sense, differential=True)


def decompose_single_ended(w):
    """1T1R version: ONE physical cell per weight, magnitude only. A real
    1T1R cell has no way to encode sign -- any negative weight is simply
    NOT representable and is clipped to 0 here (documented, not hidden).
    Returns (width, r) for the single cell."""
    w_q = quantize_to_fp2(w)
    mag = max(w_q, 0.0)  # negative weights lost -- this IS the 1T1R cost
    return col.WIDTH_FOR_MAGNITUDE[mag], col.R_FOR_MAGNITUDE[mag]


def build_array_read_netlist_1t1r(W, activations, r_sense=20.0, vread_base=0.1,
                                   results_file="array_1t1r_results.txt"):
    """Fixed-resistor READ-ONLY netlist, 1T1R (single-ended) version: ONE
    bitline per column instead of a differential pair, M cells per column
    instead of 2M. Half the physical cells and half the bitlines/Rsense/
    TIA channels of the 2T2R version -- at the cost of losing all negative
    weight contributions (clipped to 0 in decompose_single_ended)."""
    M = len(activations)
    K = len(W[0])
    lines = ["* M x K 1T1R (single-ended, unsigned) crossbar -- fixed-resistor READ-ONLY", ""]
    for i in range(M):
        v = vread_base * activations[i]
        lines.append(f"Vrow{i} wl{i} 0 DC {v}")
        for k in range(K):
            _, r = decompose_single_ended(W[i][k])
            lines.append(f"Rcell{i}_{k} wl{i} bl{k} {r}")
        lines.append("")
    for k in range(K):
        lines.append(f"Rsense{k} bl{k} 0 {r_sense}")
    lines.append("")
    lines.append(".op")
    lines.append(".control")
    lines.append("run")
    wrdata_vecs = " ".join(f"v(bl{k})" for k in range(K))
    lines.append(f"wrdata {results_file} {wrdata_vecs}")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines)


def golden_array_matmul_1t1r(W, activations, r_sense=20.0, vread_base=0.1):
    """Exact nodal-analysis golden model for the 1T1R (single-ended)
    crossbar -- same per-column independent nodal solve as one branch of
    golden_array_matmul, just without the differential subtraction (and
    without a negative-side cell to subtract in the first place)."""
    M = len(activations)
    K = len(W[0])
    results = []
    for k in range(K):
        sum_g = 1.0 / r_sense
        sum_v_over_r = 0.0
        for i in range(M):
            _, r = decompose_single_ended(W[i][k])
            v = vread_base * activations[i]
            sum_g += 1.0 / r
            sum_v_over_r += v / r
        v_bl = sum_v_over_r / sum_g
        results.append(v_bl / r_sense)
    return results


def ideal_target_currents_1t1r(W, activations, vread_base=0.1):
    """Ideal (Rsense->0) target for the 1T1R case -- weights clipped to
    non-negative, matching decompose_single_ended's real limitation."""
    g_lrs = 1.0 / col.R_FOR_MAGNITUDE[1.0]
    M = len(activations)
    K = len(W[0])
    out = []
    for k in range(K):
        s = sum(activations[i] * max(quantize_to_fp2(W[i][k]), 0.0) for i in range(M))
        out.append(vread_base * g_lrs * s)
    return out


def random_fp2_matrix(M, K, seed=0, sparsity=0.15):
    """Generate an M x K weight matrix of random FP2-E1M0 levels, with a
    given fraction forced to exactly 0.0 (typical of trained/pruned
    weights, and relevant here since 0.0 cells sit at near-HRS and behave
    very differently in loading-error terms than LRS-heavy matrices)."""
    import random
    rng = random.Random(seed)
    W = []
    for _ in range(M):
        row = []
        for _ in range(K):
            if rng.random() < sparsity:
                row.append(0.0)
            else:
                row.append(rng.choice(FP2_LEVELS))
        W.append(row)
    return W


def loading_error_sweep(K=8, r_sense=20.0, vread_base=0.1, m_values=(4, 8, 16, 32, 64, 128), seed=0):
    """Sweep M (holding K, r_sense fixed) and report the shared-Rsense
    loading error at each size -- this number determines how
    big a single physical tile can be before passive Rsense readout stops
    being good enough, requiring either a smaller tile (partial sums
    combined digitally across tiles) or an active virtual-ground TIA
    instead of a passive sense resistor."""
    activations = None
    results = []
    for M in m_values:
        W = random_fp2_matrix(M, K, seed=seed)
        a = [1.0] * M
        golden = golden_array_matmul(W, a, r_sense, vread_base)
        ideal = ideal_target_currents(W, a, vread_base)
        errs = [100.0 * abs(g - i) / abs(i) for g, i in zip(golden, ideal) if abs(i) > 1e-12]
        avg_err = sum(errs) / len(errs) if errs else float("nan")
        max_err = max(errs) if errs else float("nan")
        results.append((M, avg_err, max_err))
    return results


def _print_loading_error_sweep(K, r_sense, vread_base, m_values, seed):
    results = loading_error_sweep(K, r_sense, vread_base, m_values, seed)
    print(f"Shared-Rsense={r_sense}ohm loading error vs. array height M (K={K} columns, random FP2 weights)")
    print(f"{'M':>6} {'avg err %':>10} {'max err %':>10}")
    for M, avg_err, max_err in results:
        flag = "  <-- likely needs tiling or active TIA" if avg_err > 20.0 else ""
        print(f"{M:6d} {avg_err:10.2f} {max_err:10.2f}{flag}")
    print()
    print("Rule of thumb from this sweep: once avg loading error crosses ~20%, either")
    print("(a) split into multiple smaller tiles and sum partial results digitally after")
    print("    the ADC (the standard real-hardware answer, matches the 'tile' concept),")
    print("(b) replace the passive Rsense readout with an active virtual-ground TIA")
    print("    (op-amp keeps the bitline pinned near 0V regardless of M, eliminating this")
    print("    error at the cost of real analog design work), or")
    print("(c) lower Rsense further (trades loading error for reduced read-voltage margin")
    print("    and worse SNR at the ADC -- diminishing returns, not a free fix).")


def _print_comparison(W, activations, r_sense, vread_base):
    golden = golden_array_matmul(W, activations, r_sense, vread_base)
    ideal = ideal_target_currents(W, activations, vread_base)
    M, K = len(activations), len(W[0])
    print(f"M={M} inputs, K={K} outputs")
    print(f"{'col':>4} {'ideal I (A)':>14} {'golden I (A)':>14} {'loading err%':>12}")
    for k in range(K):
        err = 100.0 * abs(golden[k] - ideal[k]) / abs(ideal[k]) if ideal[k] != 0 else float("nan")
        print(f"{k:4d} {ideal[k]:14.6e} {golden[k]:14.6e} {err:12.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-matmul", action="store_true", help="Run a small 4x3 matmul demo (golden model + netlist)")
    ap.add_argument("--demo-dynamic-matmul", action="store_true",
                     help="Same 4x3 matmul demo, but generate the FULL dynamic program-then-read netlist "
                          "(2*M*K sequential program slots) instead of fixed resistors")
    ap.add_argument("--demo-conv", action="store_true", help="Run a small 2-input-channel, 2-output-channel conv demo")
    ap.add_argument("--demo-dynamic-conv", action="store_true",
                     help="Small 1-channel conv demo with a FULL dynamic program-then-multi-read netlist "
                          "(kernels programmed once, every output pixel gets its own read window)")
    ap.add_argument("--scale-sweep", action="store_true",
                     help="Sweep array height M and report shared-Rsense loading error at each size "
                          "(fast, no ngspice -- reports how big a tile can be before it needs "
                          "tiling or an active TIA)")
    ap.add_argument("--demo-large-matmul", action="store_true",
                     help="Generate a large fixed-resistor read-only matmul netlist (--array-m x --array-k, "
                          "random FP2 weights) -- this is the fast path that actually scales; use "
                          "--demo-dynamic-matmul for real programming validation, only at small sizes")
    ap.add_argument("--array-m", type=int, default=32, help="Array height (inputs) for --demo-large-matmul")
    ap.add_argument("--array-k", type=int, default=16, help="Array width (outputs) for --demo-large-matmul")
    ap.add_argument("--read-window-ns", type=float, default=2.0)
    ap.add_argument("--osdi", default="rram_v_1_0_0.osdi", help="Path to compiled OSDI model (for --demo-dynamic-matmul)")
    ap.add_argument("--slot-duration", type=float, default=240.0, help="ns per program slot (default 240, matches column_mac_test.py)")
    ap.add_argument("--switch-settle-ns", type=float, default=0.04)
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--netlist-out", default="array_read.sp", help="Where to write the generated ngspice netlist")
    args = ap.parse_args()

    if args.demo_matmul:
        # M=4 inputs, K=3 outputs
        W = [
            [1.0, -1.0, 0.5],
            [-0.5, 0.5, 0.0],
            [0.5, 0.0, -1.0],
            [0.0, 1.0, 0.5],
        ]
        activations = [1.0, 1.0, 1.0, 1.0]
        print("=== Matmul demo: W (4x3) . a (4,) ===")
        _print_comparison(W, activations, args.r_sense, args.vread)
        netlist = build_array_read_netlist(W, activations, args.r_sense, args.vread)
        with open(args.netlist_out, "w") as f:
            f.write(netlist)
        print(f"\nNetlist written to {args.netlist_out}")
        print("Run it yourself with:")
        print(f"  /usr/local/bin/ngspice -b {args.netlist_out}")
        print("  tail -5 array_results.txt   # columns: v(blp0) v(bln0) v(blp1) v(bln1) v(blp2) v(bln2)")

    if args.demo_conv:
        # 2 input channels, 4x4 feature map; 2 output channels, 3x3 kernels
        input_fmap = [
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
            [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        ]
        kernels = [
            [[[1, 0, -1], [0, 1, 0], [-1, 0, 1]], [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]],
            [[[0, 1, 0], [1, -1, 1], [0, 1, 0]], [[-0.5, 0, 0.5], [0, 0, 0], [0.5, 0, -0.5]]],
        ]
        print("=== Conv2d demo: 2-channel 4x4 input, 2 output channels, 3x3 kernels ===")
        outputs, out_h, out_w = conv2d_via_crossbar(input_fmap, kernels, stride=1,
                                                      r_sense=args.r_sense, vread_base=args.vread)
        print(f"Output feature map: {out_h}x{out_w}, {len(outputs[0])} channels")
        for idx, out_vec in enumerate(outputs):
            oy, ox = divmod(idx, out_w)
            print(f"  pixel ({oy},{ox}): {['%.4e' % v for v in out_vec]}")

    if args.demo_dynamic_matmul:
        W = [
            [1.0, -1.0, 0.5],
            [-0.5, 0.5, 0.0],
            [0.5, 0.0, -1.0],
            [0.0, 1.0, 0.5],
        ]
        activations = [1.0, 1.0, 1.0, 1.0]
        print("=== Dynamic program-then-read demo: W (4x3), 2*4*3=24 physical cells ===")
        _print_comparison(W, activations, args.r_sense, args.vread)
        netlist = build_array_program_and_read_netlist(
            args.osdi, W, activations, args.r_sense, args.vread,
            slot_duration=args.slot_duration, switch_settle_ns=args.switch_settle_ns,
        )
        with open(args.netlist_out, "w") as f:
            f.write(netlist)
        M, K = len(activations), len(W[0])
        total_cells = 2 * M * K
        total_time_ns = total_cells * args.slot_duration + 2.0
        print(f"\n{total_cells} cells, {args.slot_duration}ns/slot -> ~{total_time_ns:.0f}ns total simulated time")
        print(f"Netlist written to {args.netlist_out} (this one WILL take noticeably longer to run than the read-only demo)")
        print("Run it yourself with:")
        print(f"  /usr/local/bin/ngspice -b {args.netlist_out}")
        print("  tail -5 array_dynamic_results.txt")
        print("Compare each v(blp_k)-v(bln_k) pair against the golden model printed above --")
        print("if they match, real sequential programming reached the intended per-cell")
        print("resistances across the full array, not just a fixed-resistor approximation.")

    if args.demo_dynamic_conv:
        # Deliberately small: 1 input channel, 3x3 input, 2x2 kernels,
        # 2 output channels -> M=4, K=2, 16 physical cells, 4 read windows
        # (2x2 output). Keeps total simulated time comparable to the
        # matmul demos rather than scaling straight up from --demo-conv's
        # 2-channel example (which would need 72 cells).
        input_fmap = [[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]
        kernels = [
            [[[1, -1], [0.5, 0]]],
            [[[0, 0.5], [-1, 1]]],
        ]
        print("=== Dynamic conv demo: 1-channel 3x3 input, 2 output channels, 2x2 kernels ===")
        golden_outputs, out_h, out_w = conv2d_via_crossbar(input_fmap, kernels, stride=1,
                                                             r_sense=args.r_sense, vread_base=args.vread)
        print(f"Golden (fixed-resistor) outputs, {out_h}x{out_w} output, {len(golden_outputs[0])} channels:")
        for idx, vec in enumerate(golden_outputs):
            oy, ox = divmod(idx, out_w)
            print(f"  pixel ({oy},{ox}): {['%.4e' % v for v in vec]}")

        netlist, read_window_times, out_h2, out_w2, K = build_conv_dynamic_netlist(
            args.osdi, kernels, input_fmap, stride=1, r_sense=args.r_sense, vread_base=args.vread,
            slot_duration=args.slot_duration, switch_settle_ns=args.switch_settle_ns,
            read_window_ns=args.read_window_ns,
        )
        with open(args.netlist_out, "w") as f:
            f.write(netlist)
        M = 4
        total_cells = 2 * M * K
        print(f"\n{total_cells} physical cells, {len(read_window_times)} read windows "
              f"({args.read_window_ns}ns each) after programming")
        print(f"Netlist written to {args.netlist_out}")
        print("Run it yourself with:")
        print(f"  /usr/local/bin/ngspice -b {args.netlist_out}")
        print("Then parse the settled output per patch with:")
        print("  python3 -c \"")
        print("import crossbar_array_test as cb")
        print(f"rwt = {read_window_times}")
        print(f"outs = cb.parse_conv_dynamic_results('conv_dynamic_results.txt', rwt, {K}, {args.r_sense})")
        print("for i, o in enumerate(outs): print(i, o)")
        print("  \"")

    if args.scale_sweep:
        _print_loading_error_sweep(K=8, r_sense=args.r_sense, vread_base=args.vread,
                                    m_values=(4, 8, 16, 32, 64, 128, 256), seed=0)

    if args.demo_large_matmul:
        M, K = args.array_m, args.array_k
        W = random_fp2_matrix(M, K, seed=0)
        activations = [1.0] * M
        print(f"=== Large matmul demo: random FP2 W ({M}x{K}), fixed-resistor READ-ONLY ===")
        _print_comparison(W, activations, args.r_sense, args.vread)
        netlist = build_array_read_netlist(W, activations, args.r_sense, args.vread)
        with open(args.netlist_out, "w") as f:
            f.write(netlist)
        print(f"\nNetlist written to {args.netlist_out} ({M} shared wordlines, {2*K} bitlines, {2*M*K} resistors)")
        print("Run it yourself with:")
        print(f"  /usr/local/bin/ngspice -b {args.netlist_out}")
        print("This is fixed-resistor only -- it validates the ARRAY MATH at this size cheaply.")
        dyn_slots = 2 * M * K
        dyn_time_ns = dyn_slots * args.slot_duration
        print(f"\nFor reference: the equivalent FULL DYNAMIC (real programming) netlist would need")
        print(f"{dyn_slots} sequential program slots -> ~{dyn_time_ns/1000:.1f}us of simulated transient time.")
        print("That's almost certainly impractical to actually run at this size with the current")
        print("fully-sequential programming scheme -- real hardware (and a future version of this")
        print("script) would program all K columns of a row in parallel per row-select cycle,")
        print(f"cutting slots from {dyn_slots} down to {M}. Not implemented yet.")

    if (not args.demo_matmul and not args.demo_dynamic_matmul and not args.demo_conv
            and not args.demo_dynamic_conv and not args.scale_sweep and not args.demo_large_matmul):
        print("Pass --demo-matmul and/or --demo-conv. See --help.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
