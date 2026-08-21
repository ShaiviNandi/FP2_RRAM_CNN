#!/usr/bin/env python3
"""
vonneumann_baseline.py
The missing third column: a conventional digital accelerator that fetches
weights and multiplies them, against the two compute-in-memory designs.

Why this column is necessary:
Both NeuroSim columns are compute-in-memory. `memcelltype = 1` selects
SRAM-based CIM, not a von Neumann machine, so both designs already perform the
multiply-accumulate inside the array and neither pays to move weights. The
O(1)-MAC advantage is present in both and therefore cancels, leaving only the
device difference.

The SRAM CIM column is therefore a strong baseline rather than a weak one, and
comparing against it alone hides the advantage that motivates in-memory
computing.

This model supplies the baseline that makes the advantage visible: a digital
MAC array reading weights from an on-chip SRAM buffer, refilled from DRAM
whenever the working set does not fit.

What is modelled:
Per MAC:
    compute       one multiply-accumulate in a digital datapath
    weight fetch  bits_per_weight / weight_reuse, read from SRAM
    DRAM traffic  the fraction of weights that miss the on-chip buffer

Weight reuse is the parameter that decides the whole comparison and is stated
rather than assumed. A well-blocked batched convolution reuses each weight
thousands of times and the fetch term nearly vanishes. Batch-1 inference on
fully-connected and 1x1 layers reuses each weight ONCE, which is the regime
in-memory computing exists for. Both ends are reported.

What is not modelled:
Control logic, instruction fetch, pipeline registers, clock distribution. The
digital baseline is therefore OPTIMISTIC -- a real accelerator would do worse,
so the advantage computed here is a lower bound. Stated so the comparison
cannot be accused of strawmanning the baseline.

Usage:
    python3 vonneumann_baseline.py --self-test
    python3 vonneumann_baseline.py --reuse-sweep 1,10,100,1000         --neurosim-csv cmp_fp2.csv --out-csv threeway.csv
"""
import argparse
import csv
import os

# =============================================================================
# Constants for the digital baseline. Each carries its source; values without
# one are marked and excluded from reported figures.
A = {
    # Digital MAC energy. Falls faster than linearly with operand width:
    # multiplier area scales roughly with the product of the widths.
    # Defaults are the optimistic set; --horowitz substitutes the cited set
    # defined in HOROWITZ below.
    "MAC_PJ_INT8": 0.30,
    "MAC_PJ_INT4": 0.09,
    "MAC_PJ_FP2":  0.03,

    # SRAM buffer read per bit delivered. Optimistic default; see HOROWITZ.
    "SRAM_READ_PJ_PER_BIT": 0.05,

    # 6T bitcell footprint.
    #   W. Shim, X. Sun, J.-S. Seo, S. Yu, "2-Bit-Per-Cell RRAM-Based In-Memory
    #   Computing for Area-, Energy-Efficient Deep Learning", IEEE SSC-L, 2020.
    # Measured range is 150-200 F2 per cell (two cells at 300-400 F2). The
    # value below is under that range, understating SRAM area and so favouring
    # the SRAM baseline.
    "SRAM_BIT_F2": 140.0,

    # Retention leakage, 65 nm, no node scaling required.
    #   N. Verma, A. P. Chandrakasan, "A 256 kb 65 nm 8T Subthreshold SRAM
    #   Employing Sense-Amplifier Redundancy", IEEE JSSC 43(1), 141-149, 2008.
    # Measured 2.2 uW over 256 kb at 350 mV = 8.39 pW/bit. The same paper puts
    # leakage power at >=10x higher when V_DD rises from 0.3 V to 1 V, giving
    # >=84 pW/bit at 1.0 V and more at 1.2 V.
    # The cited cell is 8T; a 6T cell leaks less, so this is conservative.
    # Sub-0.5 V advanced-node figures (5.34 pW/bit at 14 nm, 12 pW/bit at
    # 22 nm) are not comparable at 65 nm 1.2 V.
    "SRAM_LEAK_PW_PER_BIT": 120.0,

    # DRAM access energy per bit.
    #   M. Horowitz, "Computing's Energy Problem (and what we can do about
    #   it)", ISSCC 2014, pp. 10-14.
    # Contemporary DRAM I/O "takes over 20pJ/bit"; with improved I/O "the
    # energy cost of a DRAM access will still be large (10pJ/bit, 0.6nJ/8B)".
    # The 10 pJ/bit floor is used so the digital baseline is not understated.
    "DRAM_PJ_PER_BIT": 10.0,
    "DRAM_PJ_PER_BIT_CONTEMPORARY": 20.0,

    # Digital MAC unit area, 65 nm, INT8 multiplier plus accumulator.
    # UNCITED. Retained for the internal breakdown, excluded from every
    # reported area: 0.23 mm2 of MAC units against 18.93 mm2 of buffer.
    # Report area_buf_mm2, not area_mm2.
    "MAC_AREA_UM2_INT8": 1800.0,
    "MAC_AREA_UM2_FP2": 220.0,

    "TECH_F_NM": 65.0,
}


# =============================================================================
# Cited constant set, from Horowitz ISSCC 2014 Fig. 1.1.9, "Rough energy costs
# for various operations in 45nm 0.9V".
#
#   Int Add   8 b 0.03 pJ   32 b 0.1 pJ
#   Int Mult  8 b 0.2  pJ   32 b 3.1 pJ
#   Cache, 64 b access      8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ
#   DRAM                    1.3-2.6 nJ
#
# MAC. INT8 multiply-accumulate is 0.2 + 0.03 = 0.23 pJ at 45 nm 0.9 V. Scaling
# by (65/45) for capacitance and (1.2/0.9)^2 for supply gives 2.57x -> 0.59 pJ
# at 65 nm 1.2 V.
#
# SRAM read. Horowitz's figures are for a processor cache, which carries tag
# lookup, tag comparison, way selection, decode and the return path to the
# core; a directly-addressed scratchpad pays none of these, so the 1 MB figure
# of 1.563 pJ/bit bounds a 4 MB scratchpad from above rather than describing
# it. Three sources, two of them SRAM macros, scaled to 65 nm 1.2 V:
#
#   Horowitz 32 KB cache, 45 nm 0.9 V    20 pJ / 64 b  -> 0.313 (unscaled)
#   22 nm SRAM, 0.5 V, JSSC 2025        ~20.5 fJ/bit   -> 0.349
#   14 nm FinFET FVS SRAM, 0.5 V         24.6 fJ/bit   -> 0.658
#
# The 22 nm entry is derived rather than quoted: that paper reports
# 10 aJ/total-bit, defined as macro energy over full 256 Kb capacity "to
# reflect the leakage effect of un-accessed banks", not energy per bit read.
# Its array is 8 banks x 2 x (128 x 128), so a 128-bit access gives
# 10 aJ x 262144 / 128 = 20.5 fJ per accessed bit. Access width is inferred
# from geometry; treat as approximate.
#
# Converged range at 65 nm 1.2 V is 0.3-0.7 pJ/bit. The value set below sits at
# the optimistic edge. Sweep with --sram-read-pj-per-bit.
#
# Sensitivity. As reuse grows the fetch term vanishes and TOPS/W tends to
# 2/mac_pj, independent of SRAM read energy. The asymptote is therefore set by
# the MAC constant alone:
#
#   FP2 MAC 0.030 pJ -> 66.67 TOPS/W -> ReRAM CIM at 0.98x
#   FP2 MAC 0.077 pJ -> 25.97 TOPS/W -> ReRAM CIM at 2.53x
#
# SRAM read energy moves only the low-reuse end, where CIM leads by 6x to 105x
# across the full plausible range.
#
# Cache energy is left unscaled at 45 nm on purpose. Scaling to 65 nm would
# raise it a further 2.57x; leaving it low keeps the baseline optimistic.
HOROWITZ = {
    "MAC_PJ_INT8": 0.23 * (65.0 / 45.0) * (1.2 / 0.9) ** 2,   # 0.591 pJ
    # No 2-bit entry exists in the figure. A 2-bit multiply is a select and a
    # negate, so the accumulator dominates; an accumulator wide enough for a
    # 32-element block is nearer the 8-bit add at 0.03 pJ than the 32-bit add
    # at 0.1 pJ. Taken as the 8-bit add, scaled -- the generous reading.
    "MAC_PJ_FP2": 0.03 * (65.0 / 45.0) * (1.2 / 0.9) ** 2,    # 0.077 pJ
    "MAC_PJ_INT4": 0.09 * (65.0 / 45.0) * (1.2 / 0.9) ** 2,
    # 32 KB cache, 20 pJ per 64-bit access, unscaled 45 nm figure.
    "SRAM_READ_PJ_PER_BIT": 20.0 / 64.0,                       # 0.3125
}


def um2_per_f2():
    f = A["TECH_F_NM"] / 1000.0
    return f * f


# =============================================================================
def von_neumann(n_weights, bits_per_weight, n_macs, weight_reuse,
                buffer_mb, n_mac_units, mac_pj, mac_area_um2):
    """Energy and area for a conventional fetch-and-multiply accelerator.

    weight_reuse is MACs served per weight fetched. The fetch term scales as
    1/reuse, which is the entire reason in-memory compute wins at reuse 1 and
    barely wins at reuse 1000.
    """
    weight_bits = n_weights * bits_per_weight
    buffer_bits = buffer_mb * 8e6

    # Compute
    e_compute_pj = n_macs * mac_pj

    # Weight fetch from the on-chip buffer, once per reuse group
    fetches = n_macs / max(weight_reuse, 1e-9)
    e_sram_pj = fetches * bits_per_weight * A["SRAM_READ_PJ_PER_BIT"]

    # DRAM refill for whatever does not fit on chip. Each resident weight is
    # brought in once per pass over the network.
    miss_frac = max(0.0, 1.0 - buffer_bits / weight_bits)
    passes = n_macs / max(n_weights * weight_reuse, 1e-9)
    e_dram_pj = weight_bits * miss_frac * passes * A["DRAM_PJ_PER_BIT"]

    total_pj = e_compute_pj + e_sram_pj + e_dram_pj

    # Area: MAC units plus the on-chip buffer
    area_mac = n_mac_units * mac_area_um2
    area_buf = buffer_bits * A["SRAM_BIT_F2"] * um2_per_f2()
    leak_uw = buffer_bits * A["SRAM_LEAK_PW_PER_BIT"] * 1e-6

    return dict(
        weight_reuse=weight_reuse,
        e_compute_pj=e_compute_pj, e_sram_pj=e_sram_pj, e_dram_pj=e_dram_pj,
        total_pj=total_pj, pj_per_mac=total_pj / n_macs,
        area_mm2=(area_mac + area_buf) * 1e-6,
        area_mac_mm2=area_mac * 1e-6, area_buf_mm2=area_buf * 1e-6,
        leak_uw=leak_uw, dram_miss_frac=miss_frac,
        tops_per_w=(2.0 * n_macs) / (total_pj * 1e-12) / 1e12,
    )


# =============================================================================
def self_test():
    print("=" * 66)
    print("SELF-TEST")
    print("=" * 66)
    n_w, n_macs = 11_157_504, 11_157_504

    # 1. The fetch term must fall as 1/reuse.
    r1 = von_neumann(n_w, 2, n_macs, 1, 4.0, 1024, A["MAC_PJ_FP2"],
                     A["MAC_AREA_UM2_FP2"])
    r1000 = von_neumann(n_w, 2, n_macs, 1000, 4.0, 1024, A["MAC_PJ_FP2"],
                        A["MAC_AREA_UM2_FP2"])
    ratio = r1["e_sram_pj"] / r1000["e_sram_pj"]
    print(f"  [1] fetch energy falls {ratio:.0f}x from reuse 1 to 1000")
    assert 900 < ratio < 1100, ratio

    # 2. Compute energy must not depend on reuse.
    print(f"  [2] compute energy independent of reuse: "
          f"{r1['e_compute_pj'] == r1000['e_compute_pj']}")
    assert r1["e_compute_pj"] == r1000["e_compute_pj"]

    # 3. A buffer larger than the model must eliminate DRAM traffic.
    big = von_neumann(n_w, 2, n_macs, 1, 100.0, 1024, A["MAC_PJ_FP2"],
                      A["MAC_AREA_UM2_FP2"])
    print(f"  [3] buffer > model: DRAM energy = {big['e_dram_pj']:.1f} pJ, "
          f"miss fraction {big['dram_miss_frac']:.3f}")
    assert big["e_dram_pj"] == 0.0

    # 4. Wider weights must cost more to fetch.
    w8 = von_neumann(n_w, 8, n_macs, 1, 4.0, 1024, A["MAC_PJ_INT8"],
                     A["MAC_AREA_UM2_INT8"])
    print(f"  [4] 8-bit fetch / 2-bit fetch = "
          f"{w8['e_sram_pj']/r1['e_sram_pj']:.1f}x")
    assert abs(w8["e_sram_pj"] / r1["e_sram_pj"] - 4.0) < 1e-6

    print("\n" + "=" * 66)
    print("SELF-TEST PASSED")
    print("=" * 66)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=int, default=11_157_504)
    ap.add_argument("--macs", type=int, default=None,
                    help="MACs per inference; defaults to one per weight, "
                         "i.e. weight reuse folded into --reuse-sweep.")
    ap.add_argument("--bits", type=int, default=2,
                    help="Weight precision. 2 = FP2, matching the proposed "
                         "design; 8 for a conventional INT8 accelerator.")
    ap.add_argument("--dram-pj-per-bit", type=float, default=None,
                    help="Override DRAM energy. Default 10 pJ/bit is "
                         "Horowitz ISSCC 2014's improved-I/O floor; 20 is "
                         "his figure for contemporary DRAM I/O.")
    ap.add_argument("--buffer-mb", type=float, default=4.0,
                    help="On-chip weight buffer. Weights beyond this are "
                         "refilled from DRAM.")
    ap.add_argument("--mac-units", type=int, default=1024)
    ap.add_argument("--reuse-sweep", default="1,10,100,1000")
    ap.add_argument("--neurosim-csv", default=None,
                    help="cmp_fp2.csv from neurosim_compare.py, to place the "
                         "CIM columns alongside.")
    ap.add_argument("--horowitz", action="store_true",
                    help="Replace the legacy MAC and SRAM-read constants with "
                         "values derived from Horowitz ISSCC 2014 Fig. 1.1.9. "
                         "Both corrections make the digital baseline more "
                         "expensive, so the CIM advantage grows. Use this for "
                         "the paper; the default is the optimistic floor.")
    ap.add_argument("--sram-read-pj-per-bit", type=float, default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if args.horowitz:
        A.update(HOROWITZ)
        print("[cited] Horowitz ISSCC 2014 Fig. 1.1.9 constants in use:")
        for k in ("MAC_PJ_INT8", "MAC_PJ_FP2", "SRAM_READ_PJ_PER_BIT"):
            print(f"         {k:<24} {A[k]:.4f} pJ")
        print()
    if args.sram_read_pj_per_bit is not None:
        A["SRAM_READ_PJ_PER_BIT"] = args.sram_read_pj_per_bit
    if args.dram_pj_per_bit is not None:
        A["DRAM_PJ_PER_BIT"] = args.dram_pj_per_bit

    n_macs = args.macs or args.weights
    mac_pj = {2: A["MAC_PJ_FP2"], 4: A["MAC_PJ_INT4"]}.get(args.bits,
                                                           A["MAC_PJ_INT8"])
    mac_area = A["MAC_AREA_UM2_FP2"] if args.bits <= 4 else A["MAC_AREA_UM2_INT8"]

    rows = [von_neumann(args.weights, args.bits, n_macs, float(r),
                        args.buffer_mb, args.mac_units, mac_pj, mac_area)
            for r in args.reuse_sweep.split(",")]

    print("=" * 78)
    print(f"VON NEUMANN DIGITAL BASELINE -- {args.bits}-bit weights, "
          f"{args.weights:,} weights, {args.buffer_mb:.0f} MB buffer")
    print("=" * 78)
    print(f"{'reuse':>8}{'compute':>12}{'SRAM fetch':>13}{'DRAM':>12}"
          f"{'pJ/MAC':>10}{'TOPS/W':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['weight_reuse']:>8.0f}{r['e_compute_pj']*1e-6:>11.2f}u"
              f"{r['e_sram_pj']*1e-6:>12.2f}u{r['e_dram_pj']*1e-6:>11.2f}u"
              f"{r['pj_per_mac']:>10.4f}{r['tops_per_w']:>10.2f}")
    print("-" * 78)
    print(f"buffer area {rows[0]['area_buf_mm2']:.2f} mm2  <- REPORT THIS, "
          f"cited cell area, leakage {rows[0]['leak_uw']*1e-3:.1f} mW")
    print(f"  [excluded] MAC units {rows[0]['area_mac_mm2']:.2f} mm2 -- area "
          f"per MAC unit is uncited, and at {100*rows[0]['area_mac_mm2']/rows[0]['area_mm2']:.1f}% "
          f"of the total it cannot change any conclusion.")
    print(f"DRAM miss fraction {rows[0]['dram_miss_frac']:.3f}")

    if args.neurosim_csv and os.path.isfile(args.neurosim_csv):
        cim = {r["cell"]: r for r in csv.DictReader(open(args.neurosim_csv))}
        print("\n" + "=" * 78)
        print("THREE-WAY: WHERE IN-MEMORY COMPUTE ACTUALLY WINS")
        print("=" * 78)
        print(f"{'':<26}{'von Neumann':>16}{'SRAM CIM':>14}{'ReRAM CIM':>14}")
        print("-" * 78)

        def cimv(cell, key, scale=1.0):
            v = cim.get(cell, {}).get(key)
            return f"{float(v)*scale:,.2f}" if v else "--"

        lo, hi = rows[0], rows[-1]
        print(f"{'TOPS/W @ reuse=1':<26}{lo['tops_per_w']:>16.2f}"
              f"{cimv('sram','tops_per_w'):>14}{cimv('rram','tops_per_w'):>14}")
        print(f"{'TOPS/W @ reuse=' + str(int(hi['weight_reuse'])):<26}"
              f"{hi['tops_per_w']:>16.2f}"
              f"{cimv('sram','tops_per_w'):>14}{cimv('rram','tops_per_w'):>14}")
        print(f"{'weight-fetch pJ/MAC @ 1':<26}"
              f"{lo['e_sram_pj']/ (args.macs or args.weights):>16.4f}"
              f"{0.0:>14.4f}{0.0:>14.4f}")
        print("-" * 78)
        for cell in ("sram", "rram"):
            t = cim.get(cell, {}).get("tops_per_w")
            if t:
                print(f"  {cell.upper()} CIM vs von Neumann at reuse 1: "
                      f"{float(t)/lo['tops_per_w']:.1f}x better")
                print(f"  {cell.upper()} CIM vs von Neumann at reuse "
                      f"{int(hi['weight_reuse'])}: "
                      f"{float(t)/hi['tops_per_w']:.1f}x")
        print()
        print("The CIM columns do not move with reuse because they never fetch")
        print("a weight. That gap IS the O(1)-MAC advantage, and it is only")
        print("visible against a machine that does fetch. Comparing SRAM CIM")
        print("against ReRAM CIM cannot show it, because both already have it.")
        print()
        print("Note the baseline is optimistic: no control logic, no")
        print("instruction fetch, no clock distribution. A real digital")
        print("accelerator would do worse, so this is a lower bound on the")
        print("advantage.")

    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
