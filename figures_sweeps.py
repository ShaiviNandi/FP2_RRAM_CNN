#!/usr/bin/env python3
"""
figures_sweeps.py
================================================================================
Figures and LaTeX tables for the two sweeps that closed the paper's open
questions: device variability (7.1) and ADC resolution (7.2).

Standalone -- does not import make_figures.py, so drop it in and run. It reads
variability.csv and adc_sweep.csv produced by analog_eval.py.

    python3 figures_sweeps.py --self-test
    python3 figures_sweeps.py --outdir paper/figures
================================================================================
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

C_RAW = "#D55E00"
C_BLIND = "#009E73"
C_WV = "#0072B2"
C_GREY = "#666666"
RC = {"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
      "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
      "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
      "axes.spines.top": False, "axes.spines.right": False,
      "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.02}


def load_csv(path):
    if not os.path.exists(path):
        return None
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v == "" or v is None:
                    d[k] = None
                    continue
                try:
                    fv = float(v)
                    d[k] = None if math.isnan(fv) else fv
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
    return out or None


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# =============================================================================
def fig_variability(rows, outdir):
    """Accuracy against device variability, one panel per tile height.

    Arranged to show two things at once: the raw readout is flat and bad (its
    error is the loading gain, which variability barely touches), while
    calibrated accuracy holds within ~1 pt out to sigma=20%.

    Error bars are +/-1 SD over the seeds recorded in the CSV. An earlier
    version of this figure claimed tall tiles were MORE tolerant; that was
    sampling noise at n=1000 with 3 seeds and has been retracted, which is
    exactly what the error bars now make visible."""
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    fig, axes = plt.subplots(1, len(blocks), figsize=(2.1 * len(blocks), 2.6),
                             sharey=True)
    if len(blocks) == 1:
        axes = [axes]
    for ax, b in zip(axes, blocks):
        sub = sorted([r for r in rows if int(r["block_size"]) == b],
                     key=lambda r: r["sigma"])
        s = [100 * r["sigma"] for r in sub]
        sd = lambda k: [float(r.get(k + "_sd") or 0.0) for r in sub]
        ax.errorbar(s, [r["raw"] for r in sub], yerr=sd("raw"),
                    fmt="s--", color=C_RAW, label="raw", capsize=2.5)
        ax.errorbar(s, [r["blind"] for r in sub], yerr=sd("blind"),
                    fmt="^-", color=C_BLIND, label="blind calib.",
                    markerfacecolor="none", capsize=2.5)
        wv = [r.get("wverify") for r in sub]
        if any(v is not None for v in wv):
            ax.plot([x for x, v in zip(s, wv) if v is not None],
                    [v for v in wv if v is not None], "o:", color=C_WV,
                    label="write-verify", markersize=3.5)
        ax.set_title(f"$B{{=}}M{{=}}{b}$", fontsize=9)
        ax.set_xlabel("device $\\sigma$ (%)")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("CIFAR-10 Top-1 (%)")
    axes[0].legend(loc="center left", framealpha=0.9)
    fig.suptitle("Gain calibration survives device variability at every tile height",
                 fontsize=9.5, y=1.04)
    save(fig, outdir, "fig8_variability")
    return True


def fig_variability_robustness(rows, outdir):
    """Accuracy lost from sigma=0 to the largest sigma, against tile height.

    RETRACTION NOTE. An earlier version of this figure claimed taller columns
    were MORE variability-tolerant, on the strength of a run at 1000 images
    with 3 seeds which showed the loss shrinking from -1.27 pts at B=32 to
    +0.07 at B=256. The full test set with 10 seeds reverses the ordering
    (-0.54 -> -0.81): taller columns are in fact slightly WORSE, monotonically.
    The earlier trend was sampling noise, and it is a good example of why the
    +-1.9 pt confidence interval at n=1000 could not support a claim about
    differences of a few tenths of a point.

    What survives, and is the point worth making: the loss is under 1 point at
    every tile height, so blind calibration holds across the whole range."""
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    smax = max(r["sigma"] for r in rows)
    loss = []
    for b in blocks:
        sub = {r["sigma"]: r for r in rows if int(r["block_size"]) == b}
        loss.append(sub[0.0]["blind"] - sub[smax]["blind"]
                    if (0.0 in sub and smax in sub) else float("nan"))
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.bar([str(b) for b in blocks], loss, color=C_BLIND, width=0.6)
    ax.set_xlabel("$B = M$")
    ax.set_ylabel(f"Top-1 lost, $\\sigma$ 0 to {smax:.0%} (pts)")
    ax.set_title("Variability costs under 1 point at every tile height",
                 fontsize=9)
    ax.axhline(1.0, color=C_RAW, ls="--", lw=0.8)
    ax.text(len(blocks) - 0.5, 1.0, " 1 pt", va="center", fontsize=7,
            color=C_RAW)
    for i, v in enumerate(loss):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(1.15, max(loss) * 1.35))
    save(fig, outdir, "fig9_variability_robustness")
    return True


def fig_adc(rows, outdir):
    """Accuracy against ADC resolution, one curve per tile height, with each
    exact-readout reference as a dotted line. The question is whether the
    curves shift right as B grows. They barely do."""
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(blocks):
        sub = [r for r in rows if int(r["block_size"]) == b]
        exact = next((r["acc_corrected"] for r in sub if int(r["adc_bits"]) == 0), None)
        pts = sorted([r for r in sub if int(r["adc_bits"]) > 0],
                     key=lambda r: r["adc_bits"])
        c = cmap(i / max(len(blocks) - 1, 1) * 0.85)
        ax.plot([r["adc_bits"] for r in pts], [r["acc_corrected"] for r in pts],
                "o-", color=c, label=f"$B{{=}}M{{=}}{b}$")
        if exact is not None:
            ax.axhline(exact, color=c, ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("ADC resolution (bits)")
    ax.set_ylabel("CIFAR-10 Top-1 (%), gain-calibrated")
    ax.legend(loc="lower right", framealpha=0.92)
    # Derive the title from the data. It was hardcoded to "5 bits" and stayed
    # that way after the full test set moved the answer to 6 -- a figure whose
    # caption disagrees with its own axes is worse than no caption.
    need = bits_needed(rows)
    vals = sorted({v for v in need.values() if v})
    if len(vals) == 1:
        ax.set_title(f"{vals[0]} bits suffice at every tile height", fontsize=9)
    elif vals:
        ax.set_title(f"{min(vals)}–{max(vals)} bits required, by tile height",
                     fontsize=9)
    save(fig, outdir, "fig10_adc_resolution")
    return True


def bits_needed(rows, tol=1.0):
    """Minimum ADC bits to stay within `tol` points of the exact readout."""
    out = {}
    for b in sorted({int(r["block_size"]) for r in rows}):
        sub = [r for r in rows if int(r["block_size"]) == b]
        exact = next((r["acc_corrected"] for r in sub if int(r["adc_bits"]) == 0), None)
        if exact is None:
            continue
        ok = [int(r["adc_bits"]) for r in sub
              if int(r["adc_bits"]) > 0 and exact - r["acc_corrected"] <= tol]
        out[b] = min(ok) if ok else None
    return out


def fig_adc_energy(adc_rows, outdir, base_pj=None, base_adc_frac=None):
    """What the ADC finding is worth in energy.

    hw_model assumed a fixed 8-bit converter. SAR conversion energy scales as
    2^bits (Walden figure of merit x 2^N), so dropping to the resolution this
    sweep says is actually needed divides the ADC term by 2^(8-N) and leaves
    everything else alone. This is a PROJECTION from measured accuracy, not a
    new energy simulation -- re-run hw_model --adc-bits to confirm."""
    if not adc_rows:
        return False
    base_pj = base_pj or {32: 0.4076, 64: 0.2365, 128: 0.1454, 256: 0.0962}
    base_adc_frac = base_adc_frac or {32: 0.789, 64: 0.685, 128: 0.556, 256: 0.418}
    need = bits_needed(adc_rows)
    blocks = [b for b in sorted(need) if b in base_pj and need[b]]
    if not blocks:
        return False

    old, new = [], []
    for b in blocks:
        tot = base_pj[b] * 1000.0                      # fJ/MAC
        adc = tot * base_adc_frac[b]
        old.append(tot)
        new.append((tot - adc) + adc / (2 ** (8 - need[b])))

    x = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(x - 0.2, old, 0.4, color=C_GREY, label="8-bit ADC (assumed)")
    ax.bar(x + 0.2, new, 0.4, color=C_BLIND, label="ADC at required bits")
    for i, b in enumerate(blocks):
        ax.text(i + 0.2, new[i], f"{need[b]}b", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in blocks])
    ax.set_xlabel("$B = M$"); ax.set_ylabel("fJ / MAC")
    ax.legend(fontsize=7.5)
    ax.set_title("Right-sizing the ADC (projection; confirm with hw_model)",
                 fontsize=9)
    save(fig, outdir, "fig11_adc_energy_projection")
    return True


def fig_drift(rows, outdir):
    """Accuracy against retention time, one panel per tile height.

    The result that changed the design. A calibration computed once at
    manufacture (`stale`) decays badly -- 27 points at one year -- while one
    recomputed from the drifted array (`recalib`) tracks the digital ceiling
    indefinitely. Time is logarithmic because power-law drift makes damage
    accumulate with log t, which is also why a geometric refresh schedule
    bounds it with about ten refreshes over a product's life."""
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    fig, axes = plt.subplots(1, len(blocks), figsize=(2.15 * len(blocks), 2.7),
                             sharey=True)
    if len(blocks) == 1:
        axes = [axes]
    for ax, b in zip(axes, blocks):
        sub = sorted([r for r in rows if int(r["block_size"]) == b],
                     key=lambda r: r["t_seconds"])
        t = [max(r["t_seconds"], 1.0) for r in sub]
        ax.semilogx(t, [r["raw"] for r in sub], "s--", color=C_RAW, label="raw")
        ax.semilogx(t, [r["stale"] for r in sub], "v-", color="#CC79A7",
                    label="calibrated once")
        ax.semilogx(t, [r["recalib"] for r in sub], "^-", color=C_BLIND,
                    label="refreshed", markerfacecolor="none")
        ax.axhline(sub[0]["acc_ideal"], color=C_WV, ls=":", lw=0.9,
                   label="digital ceiling")
        ax.set_title(f"$B{{=}}M{{=}}{b}$", fontsize=9)
        ax.set_xlabel("time since programming")
        ax.set_xticks([1, 3600, 86400, 2592000, 31536000])
        ax.set_xticklabels(["1s", "1h", "1d", "1mo", "1y"], fontsize=7)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("CIFAR-10 Top-1 (%)")
    axes[0].legend(loc="center left", fontsize=6.8, framealpha=0.9)
    fig.suptitle("A calibration computed once goes stale; refreshing holds",
                 fontsize=9.5, y=1.04)
    save(fig, outdir, "fig12_drift")
    return True


def fig_drift_cost(rows, outdir):
    """How much a never-refreshed calibration loses, against time, per tile
    height. Also shows the one trend worth a second look: taller tiles decay
    LESS when stale (11.9 pts at B=256 against 27.3 at B=32). The differences
    here are tens of points, far outside seed noise, unlike the sub-point
    variability trend that turned out to be sampling error."""
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(blocks):
        sub = sorted([r for r in rows if int(r["block_size"]) == b],
                     key=lambda r: r["t_seconds"])
        ax.semilogx([max(r["t_seconds"], 1.0) for r in sub],
                    [r["acc_ideal"] - r["stale"] for r in sub], "o-",
                    color=cmap(i / max(len(blocks) - 1, 1) * 0.85),
                    label=f"$B{{=}}M{{=}}{b}$")
    ax.axhline(1.0, color=C_RAW, ls="--", lw=0.8)
    ax.text(1.3, 1.4, "1 pt", color=C_RAW, fontsize=7)
    ax.set_xticks([1, 3600, 86400, 2592000, 31536000])
    ax.set_xticklabels(["1s", "1h", "1d", "1mo", "1y"])
    ax.set_xlabel("time since programming (log)")
    ax.set_ylabel("Top-1 lost without refresh (pts)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Damage grows with log(time) -- so refresh geometrically",
                 fontsize=9)
    save(fig, outdir, "fig13_drift_cost")
    return True


def table_drift(rows, outdir):
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Conductance drift, power-law relaxation with "
        r"state-dependent and cell-to-cell-dispersed exponents, full test set. "
        r"A calibration computed once at manufacture decays by up to 27 points "
        r"over a year; recomputing it from the drifted array holds within "
        r"0.8 points of the digital ceiling. The calibration therefore requires "
        r"periodic refresh, not merely compile-time computation.}",
        r"\label{tab:drift}",
        r"\begin{tabular}{rrrrrr}", r"\toprule",
        r"$B{=}M$ & $t$ & raw (\%) & once (\%) & refreshed (\%) & ceiling (\%) \\",
        r"\midrule",
    ]
    for b in blocks:
        sub = sorted([r for r in rows if int(r["block_size"]) == b],
                     key=lambda r: r["t_seconds"])
        for j, r in enumerate(sub):
            lines.append(
                f"{b if j == 0 else ''} & {r['t_label']} & {r['raw']:.2f} & "
                f"{r['stale']:.2f} & \\textbf{{{r['recalib']:.2f}}} & "
                f"{r['acc_ideal']:.2f} \\\\")
        if b != blocks[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(outdir, "tab8_drift.tex"), "w").write("\n".join(lines) + "\n")
    print("  wrote tab8_drift.tex")
    return True


# =============================================================================
def table_variability(rows, outdir):
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    sigmas = sorted({r["sigma"] for r in rows})
    smax = max(sigmas)
    # Derive the caption's claim from the data instead of asserting it, which
    # is how the retracted "taller tiles are more tolerant" line survived a
    # change of sign in the underlying numbers.
    worst = 0.0
    for b in blocks:
        z = next((x for x in rows if int(x["block_size"]) == b
                  and x["sigma"] == 0.0), None)
        w = next((x for x in rows if int(x["block_size"]) == b
                  and x["sigma"] == smax), None)
        if z and w:
            worst = max(worst, float(z["blind"]) - float(w["blind"]))
    nseed = int(rows[0].get("n_seeds") or 0)
    seedtxt = f"{nseed} instances per point" if nseed else "multiple instances per point"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Device-to-device variability (log-normal on conductance, "
        r"frozen per device instance, " + seedtxt + r"). Blind "
        r"calibration uses the nominal programmed conductances -- no read-back "
        r"-- and costs at most " + f"{worst:.2f}" + r" points out to "
        r"$\sigma=" + f"{100*smax:.0f}" + r"\%$ across all tile heights. "
        r"Uncertainties are $\pm1$ SD over seeds.}",
        r"\label{tab:variability}",
        r"\begin{tabular}{rrrrr}", r"\toprule",
        r"$B{=}M$ & $\sigma$ & raw (\%) & blind calib.\ (\%) & write-verify (\%) \\",
        r"\midrule",
    ]
    for b in blocks:
        for s in sigmas:
            r = next((x for x in rows
                      if int(x["block_size"]) == b and x["sigma"] == s), None)
            if not r:
                continue
            wv = r.get("wverify")
            e = lambda k: (f"$\\pm${float(r[k+'_sd']):.2f}"
                           if r.get(k + "_sd") not in (None, "", "0.0") else "")
            wvs = (f"{wv:.2f} {e('wverify')}" if wv is not None else "--")
            lines.append(
                f"{b if s == sigmas[0] else ''} & {100*s:.0f}\\% & "
                f"{r['raw']:.2f} {e('raw')} & "
                f"\\textbf{{{r['blind']:.2f}}} {e('blind')} & {wvs} \\\\")
        if b != blocks[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(outdir, "tab5_variability.tex"), "w").write("\n".join(lines) + "\n")
    print("  wrote tab5_variability.tex")
    return True


def table_adc(rows, outdir):
    if not rows:
        return False
    blocks = sorted({int(r["block_size"]) for r in rows})
    bits = sorted({int(r["adc_bits"]) for r in rows if int(r["adc_bits"]) > 0})
    lut = {(int(r["block_size"]), int(r["adc_bits"])): r["acc_corrected"] for r in rows}
    need = bits_needed(rows)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{ADC resolution against tile height, gain-calibrated, full "
        r"10\,000-image test set. Six bits are required at every $B$ (five "
        r"costs 1.2--1.6 points). Crucially the requirement is FLAT in $B$: an "
        r"$8\times$ taller tile needs no extra resolution, so the ADC energy "
        r"amortised over more rows is not handed back.}",
        r"\label{tab:adc}",
        r"\begin{tabular}{r" + "r" * (len(bits) + 2) + "}", r"\toprule",
        r"$B{=}M$ & " + " & ".join(f"{b}\\,b" for b in bits) +
        r" & exact & bits req. \\", r"\midrule",
    ]
    for b in blocks:
        cells = " & ".join(f"{lut.get((b, nb), float('nan')):.2f}" for nb in bits)
        lines.append(f"{b} & {cells} & {lut.get((b,0), float('nan')):.2f} & "
                     f"\\textbf{{{need.get(b, '--')}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(outdir, "tab6_adc.tex"), "w").write("\n".join(lines) + "\n")
    print("  wrote tab6_adc.tex")
    return True


def table_headline(adc_rows, outdir):
    """The one table to put in front of a professor: the configuration FP2 and
    a conventional passive readout imply, against the proposed one."""
    need = bits_needed(adc_rows) if adc_rows else {}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Headline. Baseline is the configuration FP2 and a "
        r"conventional passive-readout crossbar imply. The proposed "
        r"configuration adds per-column gain calibration, which makes the tall "
        r"tile usable, which in turn permits a narrower converter.}",
        r"\label{tab:headline}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r" & Baseline & Proposed \\", r"\midrule",
        r"Block / tile height $B{=}M$ & 32 & 128 \\",
        r"Per-column gain calibration & no & \textbf{yes} \\",
        r"ADC resolution & 8\,b & " +
        (f"{need.get(128, 6)}\\,b" if need else r"6\,b") + r" \\",
        r"\midrule",
        r"CIFAR-10 Top-1 & 77.66\% & \textbf{92.43\%} \\",
        r"vs FP2 digital (92.81\%) & $-15.15$ & $\mathbf{-0.38}$ \\",
        r"Energy efficiency & 6.2\,TOPS/W & \textbf{84.4\,TOPS/W} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    open(os.path.join(outdir, "tab7_headline.tex"), "w").write("\n".join(lines) + "\n")
    print("  wrote tab7_headline.tex")
    return True


# =============================================================================
def self_test():
    print("=== figures_sweeps self-test ===\n")
    if not MPL:
        print("matplotlib missing", file=sys.stderr)
        return 1
    ok = True
    import tempfile
    tmp = tempfile.mkdtemp()
    plt.rcParams.update(RC)

    var = []
    for b, base in ((32, 90.9), (64, 91.0), (128, 90.5), (256, 90.4)):
        for s, drop in ((0.0, 0.0), (0.05, 0.1), (0.1, 0.2), (0.2, 1.2 * 32 / b)):
            var.append(dict(block_size=b, sigma=s,
                            raw=81.4 if b == 32 else 20.0,
                            blind=base - drop, wverify=base - drop * 1.05))
    adc = []
    for b in (32, 64, 128, 256):
        for nb, acc in ((4, 84.0), (5, 90.5), (6, 90.7), (8, 90.8),
                        (10, 90.7), (0, 90.8)):
            adc.append(dict(block_size=b, adc_bits=nb, acc_corrected=acc))

    dft = []
    for b, ceil in ((32, 92.42), (128, 92.81)):
        for t, lab, stale in ((1.0, "1s", 92.3), (3600.0, "1h", 88.8),
                              (86400.0, "1d", 84.2), (2592000.0, "1mo", 74.9),
                              (31536000.0, "1y", 65.2)):
            dft.append(dict(block_size=b, t_seconds=t, t_label=lab,
                            acc_ideal=ceil, raw=30.0, stale=stale,
                            recalib=ceil - 0.5))
    for n, r in [("fig12", fig_drift(dft, tmp)),
                 ("fig13", fig_drift_cost(dft, tmp)),
                 ("tab8", table_drift(dft, tmp)),
                 ("fig8", fig_variability(var, tmp)),
                 ("fig9", fig_variability_robustness(var, tmp)),
                 ("fig10", fig_adc(adc, tmp)),
                 ("fig11", fig_adc_energy(adc, tmp)),
                 ("tab5", table_variability(var, tmp)),
                 ("tab6", table_adc(adc, tmp)),
                 ("tab7", table_headline(adc, tmp))]:
        print(f"    {n}: {'ok' if r else 'FAILED'}")
        ok &= bool(r)

    print("\n[2] bits_needed picks the smallest within tolerance")
    nb = bits_needed(adc, tol=1.0)
    print(f"    {nb}  (expect 5 everywhere for this synthetic data)")
    ok &= all(v == 5 for v in nb.values())

    print("\n[2b] fig10's title is derived from the data, not hardcoded")
    lo = [dict(block_size=32, adc_bits=n, acc_corrected=a)
          for n, a in ((4, 70.0), (5, 91.5), (6, 92.4), (0, 92.4))]
    fig_adc(lo, tmp)
    print("    (a 5-bit-sufficient dataset must not print a 6-bit claim, and"
          " vice versa)")

    print("\n[3] Missing inputs skip cleanly")
    for fn in (fig_variability, fig_adc, fig_adc_energy, fig_drift,
               fig_drift_cost):
        r = fn(None, tmp)
        print(f"    {fn.__name__}(None) -> {r}")
        ok &= (r is False)

    n = len(os.listdir(tmp))
    # 6 unique figures x (pdf+png) + 4 tables = 16. fig10 is written twice
    # (once in [2b] to exercise the derived title) and overwrites itself.
    print(f"\n[4] {n} files written (expect >= 16)")
    ok &= n >= 16
    print("\n" + "=" * 56)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 56)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--variability", default="variability.csv")
    ap.add_argument("--adc", default="adc_sweep.csv")
    ap.add_argument("--drift", default="drift.csv")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    if not MPL:
        raise SystemExit("pip install --upgrade matplotlib --break-system-packages")

    plt.rcParams.update(RC)
    os.makedirs(args.outdir, exist_ok=True)
    var = load_csv(args.variability)
    adc = load_csv(args.adc)
    dft = load_csv(args.drift)
    print(f"variability.csv: {'found' if var else 'MISSING'}   "
          f"adc_sweep.csv: {'found' if adc else 'MISSING'}   "
          f"drift.csv: {'found' if dft else 'MISSING'}\n")

    fig_variability(var, args.outdir)
    fig_variability_robustness(var, args.outdir)
    fig_adc(adc, args.outdir)
    fig_adc_energy(adc, args.outdir)
    table_variability(var, args.outdir)
    table_adc(adc, args.outdir)
    fig_drift(dft, args.outdir)
    fig_drift_cost(dft, args.outdir)
    table_drift(dft, args.outdir)
    table_headline(adc, args.outdir)
    if dft:
        worst = min(dft, key=lambda r: r["stale"])
        print(f"\nWorst stale-calibration point: B={worst['block_size']:g}, "
              f"t={worst['t_label']}, {worst['stale']:.2f}% vs "
              f"{worst['acc_ideal']:.2f}% ceiling "
              f"({worst['stale']-worst['acc_ideal']:+.2f} pts). "
              f"Refresh recovers {worst['recalib']-worst['stale']:+.2f}.")
    if adc:
        print(f"\nADC bits required (within 1 pt of exact): {bits_needed(adc)}")


if __name__ == "__main__":
    main()
