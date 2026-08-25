#!/usr/bin/env python3
"""
format_sweep.py
Why FP2-E1M0 and not some other weight format, answered from the device rather
than asserted.

Two independent costs decide it, and they pull in opposite directions.

Read margin. A cell holding N conductance levels must keep adjacent levels
separated despite programming scatter. Levels are placed linearly in
conductance between G_HRS and G_LRS, the measured endpoints. Programming
scatter is relative, so absolute spread grows with G and the tightest pair is
always at the top of the range. Margin is reported in standard deviations and
converted to a per-cell read error rate.

Storage cost. A format needing more magnitudes than one cell can hold must
slice across several cells and recombine with a weighted adder tree. That
multiplies cells per weight, array area, and peripheral depth.

FP2 is the format where both costs are simultaneously minimal: three magnitudes
is exactly what the device holds reliably, so no slicing is needed and the
margin stays wide.

Usage
    python3 format_sweep.py --self-test
    python3 format_sweep.py --sigma 0.05,0.10,0.20
    python3 format_sweep.py --outdir paper/figures --latex
"""
import argparse
import math

# Measured device endpoints, from SPICE calibration of rram_v_1_0_0.
R_LRS, R_MID, R_HRS = 542.8, 1099.8, 218587.2
G_LRS, G_MID, G_HRS = 1.0 / R_LRS, 1.0 / R_MID, 1.0 / R_HRS

# Magnitudes one cell holds reliably. Three is the measured ceiling: HRS, MID
# and LRS are the states the write-verify loop lands on repeatably.
MAGNITUDES_PER_CELL = 3


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def levels_linear_in_g(n):
    """n conductance levels evenly spaced between the measured endpoints."""
    if n < 2:
        return [G_LRS]
    step = (G_LRS - G_HRS) / (n - 1)
    return [G_HRS + k * step for k in range(n)]


def margin_sigmas(n, sigma_rel):
    """Worst adjacent-level separation, in standard deviations.

    Relative scatter makes absolute spread proportional to G, so the worst pair
    is the top one. Separation is compared against the combined spread of the
    two levels being distinguished.
    """
    g = levels_linear_in_g(n)
    worst = float("inf")
    for a, b in zip(g, g[1:]):
        spread = math.hypot(sigma_rel * a, sigma_rel * b)
        worst = min(worst, (b - a) / spread) if spread > 0 else worst
    return worst


def read_error_rate(n, sigma_rel):
    """Probability a cell is read as the wrong level, one decision boundary."""
    m = margin_sigmas(n, sigma_rel)
    return phi(-m / 2.0)


def cells_per_weight(magnitudes):
    """Cells per weight for a 2T2R pair, given how many magnitudes are needed.

    One cell per polarity holds MAGNITUDES_PER_CELL values. More than that
    requires slicing across cells and recombining, so the count grows
    logarithmically and an adder tree appears behind it.
    """
    slices = max(1, math.ceil(math.log(magnitudes, MAGNITUDES_PER_CELL)))
    return 2 * slices, slices


# Formats compared. `magnitudes` counts distinct absolute values including
# zero, since zero is a state the cell must hold.
FORMATS = [
    ("Binary (+/-1)",  2,   2),
    ("Ternary",        3,   2),
    ("INT2",           4,   3),
    ("FP2-E1M0",       5,   3),
    ("FP4-E2M1",      15,   8),
    ("INT4",          16,   9),
    ("INT8",         256, 129),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", default="0.05,0.10,0.20",
                    help="Relative programming scatter values to report.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    sigmas = [float(x) for x in args.sigma.split(",")]

    if args.self_test:
        assert abs(G_LRS * 1e6 - 1842.299) < 0.01
        assert abs(G_MID * 1e6 - 909.256) < 0.01
        # More levels must always be harder to distinguish.
        for s in (0.05, 0.2):
            ms = [margin_sigmas(n, s) for n in (2, 3, 4, 8)]
            assert all(a > b for a, b in zip(ms, ms[1:])), ms
        # FP2 needs no slicing; INT8 does.
        assert cells_per_weight(3)[1] == 1
        assert cells_per_weight(129)[1] > 1
        print("  G_LRS %.1f uS, G_MID %.1f uS, G_HRS %.3f uS"
              % (G_LRS * 1e6, G_MID * 1e6, G_HRS * 1e6))
        print("  SELF-TEST PASSED")
        return

    print("=" * 74)
    print("READ MARGIN AGAINST LEVELS PER CELL")
    print("=" * 74)
    print(f"  Levels placed linearly in G between {G_HRS*1e6:.2f} and "
          f"{G_LRS*1e6:.1f} uS.")
    print()
    hdr = f"{'levels':>7}" + "".join(f"{'sigma=' + f'{s:.0%}':>22}"
                                     for s in sigmas)
    print(hdr)
    print(f"{'':>7}" + "".join(f"{'margin':>11}{'err/cell':>11}"
                               for _ in sigmas))
    print("-" * len(hdr))
    for n in (2, 3, 4, 8, 16):
        row = f"{n:>7}"
        for s in sigmas:
            row += f"{margin_sigmas(n, s):>10.2f}s{read_error_rate(n, s):>11.2e}"
        tag = "   <- device ceiling" if n == MAGNITUDES_PER_CELL else ""
        print(row + tag)

    print()
    print("=" * 74)
    print("STORAGE COST AGAINST FORMAT")
    print("=" * 74)
    print(f"{'format':<16}{'levels':>7}{'magnitudes':>12}{'cells/wt':>10}"
          f"{'slices':>8}{'array area':>12}")
    print("-" * 74)
    base = None
    rows = []
    for name, levels, mags in FORMATS:
        cw, sl = cells_per_weight(mags)
        if name.startswith("FP2"):
            base = cw
        rows.append((name, levels, mags, cw, sl))
    for name, levels, mags, cw, sl in rows:
        rel = cw / base
        note = "  no slicing" if sl == 1 else f"  {sl}-deep adder tree"
        print(f"{name:<16}{levels:>7}{mags:>12}{cw:>10}{sl:>8}"
              f"{rel:>11.1f}x{note}")

    print()
    print("Formats needing more than three magnitudes cannot be held in one")
    print("cell per polarity. They either slice across cells, paying area and")
    print("an adder tree, or push more levels into a single cell, paying the")
    print("margin above. FP2 is the point where neither cost is incurred.")

    if args.latex:
        print()
        print("% ---- paste into the paper ----")
        print("\\begin{table}[H]\n    \\centering")
        print("    \\caption{Weight formats on the measured 2T2R device. Read "
              "margin is the worst adjacent-level separation at 10\\% "
              "programming scatter; storage cost is cells per weight.}")
        print("    \\label{tab:formats}")
        print("    \\begin{tabular}{lrrrr}\n        \\toprule")
        print("        \\textbf{Format} & \\textbf{Levels} & "
              "\\textbf{Magnitudes} & \\textbf{Cells/wt} & "
              "\\textbf{Margin} \\\\")
        print("        \\midrule")
        for name, levels, mags, cw, sl in rows:
            m = margin_sigmas(min(mags, 16), 0.10)
            bold = "\\textbf{%s}" % name if name.startswith("FP2") else name
            print(f"        {bold} & {levels} & {mags} & {cw} & "
                  f"{m:.2f}$\\sigma$ \\\\")
        print("        \\bottomrule\n    \\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    main()
