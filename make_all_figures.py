#!/usr/bin/env python3
"""
make_all_figures.py
Every figure from the comparison work, generated from the CSVs rather than
typed. Complements make_figures.py and figures_sweeps.py, which cover the
original accuracy, variability, ADC and drift results.

Figures produced:
  figA_rlrs_accuracy      calibrated accuracy flat while raw swings with R_LRS
  figA2_rlrs_device       cell area and loading error against R_LRS
  figB_threeway           von Neumann vs SRAM CIM vs ReRAM CIM against reuse
  figC_fp2_vs_int8        why FP2 fits on chip and INT8 does not
  figD_bn_baseline        per-column calibration against BatchNorm recalibration
  figE_wire_parasitics    what the per-column constant does and does not fix
  figF_area_breakdown     where chip area goes, per cell type

Every figure is skipped with a message if its CSV is missing, so a partial run
produces whatever it can rather than failing.

CAPTIONS ARE derived FROM the DATA, never hardcoded. A caption that stopped
tracking its numbers is how a figure in this project kept claiming "5 bits
suffice" after the answer had become 6.

Usage:
    python3 make_all_figures.py --self-test
    python3 make_all_figures.py --outdir paper/figures
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_RAW, C_CAL, C_IDEAL = "#B7410E", "#1B7A43", "#1F4E79"
C_VN, C_SRAM, C_RRAM = "#666666", "#1F4E79", "#B7410E"


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def num(r, k):
    v = r.get(k)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def save(fig, outdir, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# =============================================================================
def fig_rlrs_accuracy(rows, outdir):
    """The headline of the device sweep: calibration flattens the curve."""
    pts = [(num(r, "r_lrs"), num(r, "acc_raw"), num(r, "acc_calibrated"))
           for r in rows]
    pts = [p for p in pts if p[0] and (p[1] is not None or p[2] is not None)]
    if not pts:
        print("  skip figA: no accuracy in rlrs_tradeoff.csv")
        return
    x = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogx([p[0] for p in pts if p[1] is not None],
                [p[1] for p in pts if p[1] is not None],
                "s--", color=C_RAW, label="raw readout")
    ax.semilogx([p[0] for p in pts if p[2] is not None],
                [p[2] for p in pts if p[2] is not None],
                "o-", color=C_CAL, lw=2, label="per-column calibrated")
    ax.set_xlabel(r"$R_{LRS}$ ($\Omega$)")
    ax.set_ylabel("CIFAR-10 Top-1 (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center")

    cal = [p[2] for p in pts if p[2] is not None]
    raw = [p[1] for p in pts if p[1] is not None]
    # Caption derived, not asserted.
    ax.set_title(f"Calibrated accuracy varies {max(cal)-min(cal):.1f} pts "
                 f"across the sweep;\nraw varies {max(raw)-min(raw):.1f} pts",
                 fontsize=10)
    save(fig, outdir, "figA_rlrs_accuracy")


def fig_rlrs_device(rows, outdir):
    """What the device costs in silicon as R_LRS moves."""
    x = [num(r, "r_lrs") for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogx(x, [num(r, "rel_cell_area") for r in rows], "o-",
                color=C_RAW, label=r"cell area ($\times$ default 12F)")
    ax.semilogx(x, [num(r, "loading_err_pct") for r in rows], "s--",
                color=C_IDEAL, label="uncorrected loading error (%)")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$R_{LRS}$ ($\Omega$)")
    ax.set_ylabel("relative cell area  /  loading error %")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fits = [num(r, "r_lrs") for r in rows if num(r, "rel_cell_area") <= 1.0]
    if fits:
        ax.axvline(min(fits), color=C_CAL, ls="-.", lw=1.2)
        ax.set_title(f"Both fall with $R_{{LRS}}$; cell fits the default "
                     f"layout above {min(fits):,.0f} $\\Omega$", fontsize=10)
    save(fig, outdir, "figA2_rlrs_device")


def fig_threeway(vn_rows, ns_rows, outdir):
    """The O(1)-MAC advantage, visible only against a machine that fetches."""
    if not vn_rows:
        print("  skip figB: no threeway.csv")
        return
    x = [num(r, "weight_reuse") for r in vn_rows]
    y = [num(r, "tops_per_w") for r in vn_rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.semilogx(x, y, "o-", color=C_VN, lw=2, label="von Neumann digital")
    cim = {r["cell"]: num(r, "tops_per_w") for r in (ns_rows or [])}
    for cell, col, lbl in (("sram", C_SRAM, "SRAM CIM"),
                           ("rram", C_RRAM, "ReRAM CIM")):
        if cim.get(cell):
            ax.axhline(cim[cell], color=col, ls="--", lw=2, label=lbl)
    ax.set_xlabel("weight reuse (MACs per weight fetched)")
    ax.set_ylabel("TOPS/W")
    ax.grid(alpha=0.3)
    ax.legend()
    if cim.get("rram") and y:
        ax.set_title(f"ReRAM CIM is {cim['rram']/y[0]:.1f}$\\times$ von Neumann "
                     f"at reuse 1, {cim['rram']/y[-1]:.1f}$\\times$ at reuse "
                     f"{x[-1]:.0f}", fontsize=10)
    save(fig, outdir, "figB_threeway")


def fig_fp2_vs_int8(outdir, weights=11_157_504, buffer_mb=4.0):
    """Why 2 bits fits on chip and 8 does not."""
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bits = [2, 4, 8, 16, 32]
    mb = [weights * b / 8e6 for b in bits]
    cols = [C_CAL if m <= buffer_mb else C_RAW for m in mb]
    ax.bar([str(b) for b in bits], mb, color=cols)
    ax.axhline(buffer_mb, color=C_IDEAL, ls="--", lw=1.5,
               label=f"{buffer_mb:.0f} MB on-chip buffer")
    ax.set_xlabel("weight precision (bits)")
    ax.set_ylabel("model size (MB)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fits = [b for b, m in zip(bits, mb) if m <= buffer_mb]
    ax.set_title(f"ResNet-18 fits a {buffer_mb:.0f} MB buffer only at "
                 f"{max(fits) if fits else 0} bits or fewer", fontsize=10)
    save(fig, outdir, "figC_fp2_vs_int8")


def fig_bn(rows, outdir):
    """Per-column calibration against the closest prior art."""
    if not rows:
        print("  skip figD: no bn_baseline.csv")
        return
    b = [num(r, "block_size") for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    w = 0.35
    xs = range(len(b))
    ax.bar([i - w/2 for i in xs], [-num(r, "bn_gap") for r in rows], w,
           color=C_RAW, label="BatchNorm recalibration")
    ax.bar([i + w/2 for i in xs], [-num(r, "corrected_gap") for r in rows], w,
           color=C_CAL, label="per-column calibration")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{int(v)}" for v in b])
    ax.set_xlabel("$B = M$ (tile height)")
    ax.set_ylabel("points below FP2 digital ceiling")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    wb = max(num(r, "bn_gap") for r in rows)
    wc = max(num(r, "corrected_gap") for r in rows)
    ax.set_title(f"Worst gap {wb:.2f} pts vs {wc:.2f} pts "
                 f"({wb/wc:.1f}$\\times$ better)", fontsize=10)
    save(fig, outdir, "figD_bn_baseline")


def fig_wire(rows, outdir):
    """What the per-column constant fixes, and what it does not."""
    if not rows:
        print("  skip figE: no wire_parasitics.csv")
        return
    lab = [r["case"] for r in rows]
    val = [num(r, "col_calib_pct") for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    cols = [C_CAL if v is not None and v < 1.0 else C_RAW for v in val]
    ax.barh(range(len(lab)), val, color=cols)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("residual error after per-column calibration (%)")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()
    ax.set_title("Green: removed exactly.  Orange: not modelled by the "
                 "per-column constant", fontsize=10)
    save(fig, outdir, "figE_wire_parasitics")


def fig_area(rows, outdir):
    """Where chip area goes, and how much is unaccounted."""
    if not rows:
        print("  skip figF: no neurosim CSV")
        return
    keys = [("cim_array_um2", "CIM array"), ("adc_area_um2", "ADC"),
            ("ic_area_um2", "interconnect"), ("accum_area_um2", "accumulation"),
            ("other_area_um2", "other periphery")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bottom = [0.0] * len(rows)
    for k, lbl in keys:
        vals = [(num(r, k) or 0) * 1e-6 for r in rows]
        ax.bar([r["cell"].upper() for r in rows], vals, 0.5, bottom=bottom,
               label=lbl)
        bottom = [a + b for a, b in zip(bottom, vals)]
    # The gap between the parts and the whole is itself a result.
    tot = [(num(r, "chip_area_um2") or 0) * 1e-6 for r in rows]
    gap = [t - b for t, b in zip(tot, bottom)]
    if any(g > 0.05 * t for g, t in zip(gap, tot) if t):
        ax.bar([r["cell"].upper() for r in rows], gap, 0.5, bottom=bottom,
               color="lightgray", hatch="//", label="unaccounted by the tool")
    ax.set_ylabel("chip area (mm$^2$)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    frac = [100 * g / t for g, t in zip(gap, tot) if t]
    ax.set_title("Reported categories do not sum to chip area: "
                 + ", ".join(f"{r['cell']} {f:.0f}% unaccounted"
                             for r, f in zip(rows, frac)), fontsize=9)
    save(fig, outdir, "figF_area_breakdown")


# =============================================================================
def self_test():
    print("=" * 60); print("SELF-TEST"); print("=" * 60)
    import tempfile
    d = tempfile.mkdtemp()
    rows = [dict(r_lrs="542.8", acc_raw="12.35", acc_calibrated="91.95",
                 rel_cell_area="10.6", loading_err_pct="58.6"),
            dict(r_lrs="6000", acc_raw="85.05", acc_calibrated="89.55",
                 rel_cell_area="1.0", loading_err_pct="11.9")]
    fig_rlrs_accuracy(rows, d); fig_rlrs_device(rows, d)
    fig_fp2_vs_int8(d)
    n = len([f for f in os.listdir(d) if f.endswith(".pdf")])
    print(f"  [1] produced {n} PDFs from synthetic rows")
    assert n == 3
    print("  [2] missing CSVs are skipped, not fatal")
    fig_bn(None, d); fig_wire(None, d)
    print("\n" + "=" * 60); print("SELF-TEST PASSED"); print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Writing to {args.outdir}/\n")

    rl = load("rlrs_tradeoff.csv")
    if rl:
        fig_rlrs_accuracy(rl, args.outdir)
        fig_rlrs_device(rl, args.outdir)
    else:
        print("  skip figA/figA2: no rlrs_tradeoff.csv")

    fig_threeway(load("threeway.csv"), load("cmp_fp2.csv"), args.outdir)
    fig_fp2_vs_int8(args.outdir)
    fig_bn(load("bn_baseline.csv"), args.outdir)
    fig_wire(load("wire_parasitics.csv"), args.outdir)
    fig_area(load("cmp_fp2.csv"), args.outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
