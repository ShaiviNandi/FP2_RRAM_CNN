#!/usr/bin/env python3
"""
make_memory_area.py
Weight-array silicon footprint for one network held fully resident, across
three storage technologies.

Areas are cell counts times cell area, nothing else. Periphery, converters,
interconnect and global buffer are excluded, so these are ARRAY areas and not
chip areas. The distinction matters: at 2-bit weights the array is a small
fraction of a finished die, and a cell-level density ratio does not carry over
to a chip-level one. The figure is annotated to that effect.

Cell areas:
  1T1R ReRAM  40 F2  -- bracketed by 31 F2 at 90 nm (Shim, SSC-L 2020) and
                        84 F2 at 40 nm (1 Mb macro), valid at R_LRS >~ 6 kOhm
  6T SRAM    140 F2  -- below the 150-200 F2 measured range in Shim 2020, so
                        the SRAM columns are understated and the comparison is
                        conservative

Usage
    python3 make_memory_area.py --outdir paper/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_RERAM = "#1f77b4"
C_SRAM = "#d62728"
C_SRAM2 = "#ff7f0e"

CELL_F2_RERAM = 40.0
CELL_F2_SRAM = 140.0
F_NM = 65.0


def um2_per_f2(f_nm=F_NM):
    f = f_nm / 1000.0
    return f * f


def areas(n_weights):
    a = um2_per_f2()
    return {
        # FP32 in SRAM: 32 bits per weight.
        "SRAM, FP32": (n_weights * 32 * CELL_F2_SRAM * a * 1e-6, C_SRAM),
        # FP2 in SRAM: 2 bits per weight in the array. The shared E8M0 scale
        # adds 0.25 bit per weight and is held separately.
        "SRAM, FP2": (n_weights * 2 * CELL_F2_SRAM * a * 1e-6, C_SRAM2),
        # 2T2R: two cells per weight, one per polarity.
        "ReRAM 2T2R, FP2": (n_weights * 2 * CELL_F2_RERAM * a * 1e-6, C_RERAM),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=int, default=11_157_504)
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    A = areas(args.weights)

    if args.self_test:
        r = A["ReRAM 2T2R, FP2"][0]
        print(f"  ReRAM 2T2R FP2   {r:8.2f} mm2")
        print(f"  SRAM FP2         {A['SRAM, FP2'][0]:8.2f} mm2  "
              f"({A['SRAM, FP2'][0]/r:.2f}x)")
        print(f"  SRAM FP32        {A['SRAM, FP32'][0]:8.2f} mm2  "
              f"({A['SRAM, FP32'][0]/r:.2f}x)")
        assert abs(r - 3.77) < 0.02, r
        assert abs(A["SRAM, FP2"][0] - 13.20) < 0.05
        assert abs(A["SRAM, FP32"][0] - 211.20) < 0.5
        print("  SELF-TEST PASSED")
        return

    os.makedirs(args.outdir, exist_ok=True)
    labels = list(A.keys())[::-1]
    vals = [A[k][0] for k in labels]
    cols = [A[k][1] for k in labels]

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    bars = ax.barh(labels, vals, color=cols, height=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1, vals[-1] * 4 if vals[-1] > max(vals) else max(vals) * 4)
    ax.set_xlabel("weight-array active area (mm$^2$, 65 nm, log scale)")

    base = A["ReRAM 2T2R, FP2"][0]
    for b, v in zip(bars, vals):
        txt = f"{v:.2f} mm$^2$" + ("" if abs(v - base) < 1e-9
                                   else f"   ({v/base:.1f}$\\times$)")
        ax.text(v * 1.12, b.get_y() + b.get_height() / 2, txt,
                va="center", fontsize=8)

    ax.set_title(f"{args.weights/1e6:.2f} M weights held resident",
                 fontsize=9, pad=6)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"fig9_memory_area.{ext}"))
    plt.close(fig)
    print(f"  wrote fig9_memory_area.pdf / .png to {args.outdir}")


if __name__ == "__main__":
    main()
