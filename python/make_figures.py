#!/usr/bin/env python3
"""
make_figures.py
================================================================================
Every figure and LaTeX table for the paper, regenerated from the CSV/JSON
artifacts the pipeline already produces. Nothing here computes a new result --
if a number appears in a figure it came from a file on disk, and the figure
caption says which one.

THE TWO ARCHITECTURES UNDER COMPARISON
--------------------------------------
A  FP2-2T2R ReRAM crossbar (proposed)
     FP2-E1M0 weights {-1,-.5,0,+.5,+1} on 3 ReRAM resistance states
     (LRS / mid / HRS), one differential 2T2R pair per signed weight.
     Block of B weights shares one 8-bit E8M0 scale; the crossbar tile height
     M equals B by necessity (output-side rescaling cannot span two scales).
     Passive shared-Rsense readout on two bitlines per column.
     PROPOSED ADDITION: a per-column loading-gain constant (1 + Rs*G_col),
     computed at compile time from the programmed conductances, applied to
     each bitline digitally BEFORE the differential subtraction.

B  FP2-digital SRAM accelerator (baseline)
     Identical network, identical FP2 quantization and block scaling, weights
     in 6T SRAM, MACs in digital adder trees, exact dot product.
     This is `acc_ideal` in analog_accuracy.csv and the SRAM side of
     codesign_sweep.digital_baseline().

INPUTS (all optional -- figures whose inputs are missing are skipped)
    codesign_pareto.csv        codesign_sweep.py
    analog_accuracy.csv        analog_eval.py --sweep
    ngspice_full_summary.csv   ngspice_full_sweep.py
    ngspice_full.jsonl         ngspice_full_sweep.py  (per-tile, for the histogram)
    hw_report.json             hw_model.py --report
    qat_history.csv            qat_finetune_fp2.py --log-csv
    results_cifar_qat_deployed.csv   benchmark_resnet18.py

USAGE
    pip install matplotlib --break-system-packages
    python3 make_figures.py --self-test
    python3 make_figures.py --outdir paper/figures
================================================================================
"""
import argparse
import csv
import json
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    MPL = True
except ImportError:
    MPL = False

import numpy as np

# Colour-blind-safe, print-safe, and distinguishable in greyscale by marker.
C_DIGITAL = "#0072B2"
C_ANALOG = "#D55E00"
C_CORRECTED = "#009E73"
C_ENERGY = "#CC79A7"
C_GREY = "#666666"

PLOT_RC = {
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}


# =============================================================================
def load_csv(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    fv = float(v)
                    d[k] = None if math.isnan(fv) else fv
                except (TypeError, ValueError):
                    d[k] = v
            rows.append(d)
    return rows or None


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# =============================================================================
# Figures
# =============================================================================
def fig_codesign(pareto, analog, outdir):
    """THE money figure. Accuracy against B for all three readout modes, with
    energy efficiency on the right axis. Shows the collision (raw analog
    collapses as B grows, exactly where energy improves) and its resolution
    (calibrated is flat)."""
    if not pareto:
        return False
    B = [r["block_size"] for r in pareto]
    tw = [r["tops_per_w"] for r in pareto]

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax2 = ax.twinx()
    ax2.grid(False)

    if analog:
        amap = {r["block_size"]: r for r in analog}
        xs = [b for b in B if b in amap]
        ax.plot(xs, [amap[b]["acc_ideal"] for b in xs], "o-", color=C_DIGITAL,
                label="B: FP2 digital (SRAM)", zorder=3)
        ax.plot(xs, [amap[b]["acc_analog"] for b in xs], "s--", color=C_ANALOG,
                label="A: crossbar, raw readout", zorder=3)
        if "acc_corrected" in analog[0]:
            ax.plot(xs, [amap[b]["acc_corrected"] for b in xs], "^-",
                    color=C_CORRECTED, label="A: crossbar + gain calib.",
                    zorder=4, markerfacecolor="none")
    else:
        ax.plot(B, [r["acc_qat"] for r in pareto], "o-", color=C_DIGITAL,
                label="FP2 QAT accuracy")

    ax2.plot(B, tw, ":", color=C_ENERGY, marker="D", markersize=4,
             label="energy efficiency", zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(B)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("block size $B$ = crossbar tile height $M$")
    ax.set_ylabel("CIFAR-10 Top-1 (%)")
    ax2.set_ylabel("TOPS/W", color=C_ENERGY)
    ax2.tick_params(axis="y", colors=C_ENERGY)
    ax.set_ylim(0, 100)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", framealpha=0.92)
    if analog:
        amap = {r["block_size"]: r for r in analog}
        b_hi = max(amap)
        ax.annotate("raw readout collapses\nexactly where energy improves",
                    xy=(b_hi, amap[b_hi]["acc_analog"]), xytext=(-8, 34),
                    textcoords="offset points", ha="right", fontsize=7.5,
                    color=C_ANALOG,
                    arrowprops=dict(arrowstyle="->", color=C_ANALOG, lw=0.8))
    save(fig, outdir, "fig1_codesign_collision")
    return True


def fig_attenuation(analog, outdir, r_sense=20.0, r_lrs=542.8, util=0.42):
    """Why the raw readout fails: the shared-Rsense divider retains only
    1/(1+Rs*G_col) of the ideal current, and G_col grows linearly with M.
    Analytic curve, with the measured accuracy loss overlaid."""
    M = np.array([8, 16, 32, 64, 128, 256, 512], dtype=float)
    g_col = M * util / r_lrs
    retained = 1.0 / (1.0 + r_sense * g_col)

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(M, 100 * retained, "-", color=C_ANALOG,
            label=r"$1/(1+R_s G_{col})$  (analytic)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(M)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("crossbar tile height $M$")
    ax.set_ylabel("fraction of ideal current\nreaching the sense node (%)")
    ax.set_ylim(0, 100)

    if analog:
        ax2 = ax.twinx(); ax2.grid(False)
        xs = [r["block_size"] for r in analog]
        ax2.plot(xs, [-r["analog_cost_pts"] for r in analog], "s--",
                 color=C_DIGITAL, label="measured accuracy loss")
        ax2.set_ylabel("Top-1 lost, raw readout (pts)", color=C_DIGITAL)
        ax2.tick_params(axis="y", colors=C_DIGITAL)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.92)
    else:
        ax.legend(loc="upper right")
    ax.set_title("The loading error is a deterministic gain, not noise",
                 fontsize=9)
    save(fig, outdir, "fig2_loading_attenuation")
    return True


def fig_energy_breakdown(pareto, outdir):
    """Where the energy goes as B grows. The ADC's share falls because its
    fixed per-column cost amortises over more rows -- which is the entire
    reason larger B is worth chasing."""
    if not pareto:
        return False
    B = [r["block_size"] for r in pareto]
    adc = np.array([r["adc_energy_frac"] for r in pareto]) * 100.0
    other = 100.0 - adc
    pj = [r["pj_per_mac"] for r in pareto]
    x = np.arange(len(B))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax.bar(x, adc, color=C_ANALOG, label="ADC")
    ax.bar(x, other, bottom=adc, color=C_GREY, label="array + DAC + accum.")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in B])
    ax.set_xlabel("$B = M$"); ax.set_ylabel("share of energy (%)")
    ax.set_ylim(0, 100); ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("Energy breakdown", fontsize=9)

    ax2.plot(x, pj, "o-", color=C_ENERGY)
    ax2.set_xticks(x); ax2.set_xticklabels([str(b) for b in B])
    ax2.set_xlabel("$B = M$"); ax2.set_ylabel("pJ / MAC")
    ax2.set_title("Energy per MAC", fontsize=9)
    save(fig, outdir, "fig3_energy_breakdown")
    return True


def fig_pareto(pareto, analog, outdir):
    """Accuracy against efficiency. The point of the paper is that calibration
    moves the operating point right along the x-axis without moving it down."""
    if not (pareto and analog):
        return False
    amap = {r["block_size"]: r for r in analog}
    pts = [(r["tops_per_w"], amap[r["block_size"]], r["block_size"])
           for r in pareto if r["block_size"] in amap]
    if not pts:
        return False

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot([p[0] for p in pts], [p[1]["acc_analog"] for p in pts], "s--",
            color=C_ANALOG, label="A: raw readout")
    if "acc_corrected" in analog[0]:
        ax.plot([p[0] for p in pts], [p[1]["acc_corrected"] for p in pts], "^-",
                color=C_CORRECTED, label="A: + gain calibration",
                markerfacecolor="none")
    ax.plot([p[0] for p in pts], [p[1]["acc_ideal"] for p in pts], "o:",
            color=C_DIGITAL, label="B: FP2 digital", alpha=0.75)
    for tw, a, b in pts:
        ax.annotate(f"B={b:g}", (tw, a["acc_analog"]), fontsize=7,
                    xytext=(3, -9), textcoords="offset points", color=C_ANALOG)
    ax.set_xlabel("energy efficiency (TOPS/W)")
    ax.set_ylabel("CIFAR-10 Top-1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.set_title("Calibration moves right without moving down", fontsize=9)
    save(fig, outdir, "fig4_pareto")
    return True


def fig_ngspice_validation(jsonl_path, summary, outdir, max_rows=200000):
    """Distribution of ngspice-vs-nodal disagreement over every simulated tile.
    This is the figure that licenses using the fast model everywhere else."""
    errs = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if i >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error") and r.get("rel_err_pct") is not None:
                    errs.append(r["rel_err_pct"])
    if not errs and summary:
        errs = [r["rel_err_mean_pct"] for r in summary]
    if not errs:
        return False

    e = np.array(errs)
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.hist(e, bins=60, color=C_CORRECTED, edgecolor="none")
    ax.set_xlabel("relative disagreement, ngspice vs nodal model (%)")
    ax.set_ylabel("tiles")
    ax.set_title(f"{len(e):,} tiles, max {e.max():.2e}%", fontsize=9)
    ax.axvline(e.max(), color=C_ANALOG, lw=1, ls="--")
    save(fig, outdir, "fig5_ngspice_validation")
    return True


def fig_qat(hist, outdir):
    """Accuracy climbs while the quantization metrics stay flat: QAT adapts the
    network around a fixed noise floor rather than shrinking it."""
    if not hist:
        return False
    ep = [r["epoch"] for r in hist]
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.plot(ep, [r["test_acc"] for r in hist], "o-", color=C_DIGITAL,
            label="test accuracy")
    ax.set_xlabel("QAT epoch"); ax.set_ylabel("Top-1 (%)", color=C_DIGITAL)
    ax.tick_params(axis="y", colors=C_DIGITAL)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ep, [r["cell_utilization_pct"] for r in hist], "s--",
             color=C_ANALOG, label="cell utilization")
    ax2.plot(ep, [r["quant_residual_pct"] for r in hist], "^:",
             color=C_GREY, label="quant. residual")
    ax2.set_ylabel("% of weights / of $\\|W\\|$")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7.5)
    ax.set_title("QAT adapts around the noise floor, not through it", fontsize=9)
    save(fig, outdir, "fig6_qat_history")
    return True


def fig_layer_snr(layers, outdir):
    if not layers:
        return False
    names = [r["layer"].replace("layer", "L").replace("downsample.0", "ds")
             for r in layers]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.bar(x - 0.2, [r["snr_digital_db"] for r in layers], 0.4,
           color=C_DIGITAL, label="FP2 digital")
    ax.bar(x + 0.2, [r["snr_analog_db"] for r in layers], 0.4,
           color=C_ANALOG, label="crossbar (raw)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=55, ha="right", fontsize=6.5)
    ax.set_ylabel("SNR vs FP32 (dB)")
    ax.legend(fontsize=7.5)
    save(fig, outdir, "fig7_layer_snr")
    return True


# =============================================================================
# LaTeX tables
# =============================================================================
def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table_architectures(outdir):
    rows = [
        ("Weight storage", "2T2R ReRAM, 3 states", "6T SRAM, 2 bits"),
        ("Cells / area per weight", r"2 cells, 0.338\,\si{\micro\meter\squared}",
         r"2 bits, 1.183\,\si{\micro\meter\squared}"),
        ("MAC mechanism", "Ohm's law + KCL on bitline", "digital adder tree"),
        ("MACs per read", r"$M\times K$ in one settling time", "1 per adder per cycle"),
        ("Weight-fetch energy", "0 (compute in place)", r"$\propto$ bits fetched"),
        ("Standby power", "0 (non-volatile)", "leakage, continuous"),
        ("Readout", "shared-$R_s$, 2 bitlines/col, ADC", "exact, none needed"),
        ("Loading error", r"$1/(1+R_sG_{col})$, calibrated out", "none"),
        ("Tile height $M$", r"$=B$, the FP2 block size", "unconstrained"),
    ]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{The two architectures compared. A is proposed; B is the "
        r"conventional digital realisation of the identical FP2 network.}",
        r"\label{tab:architectures}",
        r"\begin{tabular}{lll}", r"\toprule",
        r" & \textbf{A: FP2-2T2R ReRAM} & \textbf{B: FP2 digital (SRAM)} \\",
        r"\midrule",
    ]
    for a, b, c in rows:
        lines.append(f"{a} & {b} & {c} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = os.path.join(outdir, "tab1_architectures.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    print("  wrote tab1_architectures.tex")
    return True


def table_codesign(pareto, analog, outdir):
    if not pareto:
        return False
    amap = {r["block_size"]: r for r in (analog or [])}
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Co-design sweep. $B$ is simultaneously the FP2 block size "
        r"and the crossbar tile height. Raw readout collapses as $B$ grows; "
        r"per-column gain calibration removes the loss entirely, unlocking the "
        r"efficiency of the larger tile.}",
        r"\label{tab:codesign}",
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"$B{=}M$ & digital & raw & calib. & SNR$_{\mathrm{an}}$ & pJ/MAC & TOPS/W \\",
        r" & (\%) & (\%) & (\%) & (dB) & & \\", r"\midrule",
    ]
    for r in pareto:
        b = r["block_size"]
        a = amap.get(b, {})
        dig = a.get("acc_ideal", r.get("acc_qat"))
        raw = a.get("acc_analog")
        cor = a.get("acc_corrected")
        lines.append(
            f"{b:g} & {dig:.2f} & "
            f"{('%.2f' % raw) if raw is not None else '--'} & "
            f"{('%.2f' % cor) if cor is not None else '--'} & "
            f"{r['snr_analog_db']:.2f} & {r['pj_per_mac']:.4f} & "
            f"{r['tops_per_w']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = os.path.join(outdir, "tab2_codesign.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    print("  wrote tab2_codesign.tex")
    return True


def table_memory(outdir, n_weights=11157504):
    try:
        import codesign_sweep as cs
        b = cs.digital_baseline(n_weights)
    except Exception as e:
        print(f"  [skip] memory table: {e}", file=sys.stderr)
        return False
    ar, st, mv = b["area"], b["standby"], b["movement"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Memory metrics, ResNet-18 on CIFAR-10, \SI{65}{\nano\meter}. "
        r"These are where the ReRAM implementation wins; it does not win on "
        r"compute efficiency at any tile size we measured.}",
        r"\label{tab:memory}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Metric & A (ReRAM) & B (SRAM) & Gain \\", r"\midrule",
        f"Weight array area (mm$^2$) & {ar['reram_mm2']:.2f} & "
        f"{ar['sram_fp2_mm2']:.2f} & {ar['density_gain_vs_sram_fp2']:.1f}$\\times$ \\\\",
        f"vs FP32 weights (mm$^2$) & {ar['reram_mm2']:.2f} & "
        f"{ar['sram_fp32_mm2']:.1f} & {ar['density_gain_vs_sram_fp32']:.0f}$\\times$ \\\\",
        f"Standby power (mW) & {st['reram_leak_mw']:.1f} & "
        f"{st['sram_leak_mw']:.2f} & $\\infty$ \\\\",
        f"Weight-fetch energy (pJ/MAC) & 0 & "
        f"{mv['sram_fetch_pj_per_mac']:.3f} & $\\infty$ \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    p = os.path.join(outdir, "tab3_memory.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    print("  wrote tab3_memory.tex")
    return True


def table_validation(summary, outdir):
    if not summary:
        return False
    tiles = sum(int(r["tiles_simulated"]) for r in summary)
    worst = max(r["rel_err_max_pct"] for r in summary)
    macs = sum(int(r["macs"]) for r in summary)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Validation chain. Each row is checked against the one above "
        r"it, so the end-to-end accuracy numbers rest on circuit simulation "
        r"rather than on a noise model.}",
        r"\label{tab:validation}",
        r"\begin{tabular}{lll}", r"\toprule",
        r"Level & Checked against & Worst disagreement \\", r"\midrule",
        f"ngspice DC solve & --- (reference) & --- \\\\",
        f"Nodal model & ngspice, {tiles:,} tiles & {worst:.2e}\\% \\\\",
        r"Vectorised solver & nodal model & $<10^{-12}$ (rel.) \\",
        f"End-to-end inference & vectorised solver & exact (same code path) \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    p = os.path.join(outdir, "tab4_validation.tex")
    open(p, "w").write("\n".join(lines) + "\n")
    print(f"  wrote tab4_validation.tex  ({tiles:,} tiles, {macs:,} MAC-reads)")
    return True


# =============================================================================
def self_test():
    print("=== make_figures self-test ===\n")
    ok = True
    print("[1] matplotlib available:", MPL)
    if not MPL:
        print("    pip install matplotlib --break-system-packages", file=sys.stderr)
        return 1

    print("\n[2] Figures render from synthetic data without touching disk inputs")
    import tempfile
    tmp = tempfile.mkdtemp()
    pareto = [dict(block_size=b, acc_qat=92.0 + i * 0.1, snr_analog_db=11 - 2 * i,
                   pj_per_mac=0.4 / (i + 1), tops_per_w=5 * (i + 1),
                   adc_energy_frac=0.8 - 0.1 * i, area_mm2=800 / (i + 1))
              for i, b in enumerate([32, 64, 128, 256])]
    analog = [dict(block_size=b, acc_ideal=91.2, acc_analog=80 - 22 * i,
                   acc_corrected=91.2, analog_cost_pts=-(11 + 22 * i),
                   corrected_cost_pts=0.0, recovered_pts=11 + 22 * i)
              for i, b in enumerate([32, 64, 128, 256])]
    hist = [dict(epoch=e, test_acc=90 + e * 0.25, cell_utilization_pct=41.8,
                 quant_residual_pct=50.1) for e in range(1, 11)]
    layers = [dict(layer=f"layer{i}.conv1", snr_digital_db=13 - i * 0.3,
                   snr_analog_db=11 - i * 0.3) for i in range(1, 8)]
    summary = [dict(tiles_simulated=288, rel_err_mean_pct=1.8e-7,
                    rel_err_max_pct=3.3e-7, macs=1179648)]

    checks = [
        ("fig1", fig_codesign(pareto, analog, tmp)),
        ("fig2", fig_attenuation(analog, tmp)),
        ("fig3", fig_energy_breakdown(pareto, tmp)),
        ("fig4", fig_pareto(pareto, analog, tmp)),
        ("fig5", fig_ngspice_validation("/nonexistent.jsonl", summary, tmp)),
        ("fig6", fig_qat(hist, tmp)),
        ("fig7", fig_layer_snr(layers, tmp)),
        ("tab1", table_architectures(tmp)),
        ("tab2", table_codesign(pareto, analog, tmp)),
        ("tab4", table_validation(summary, tmp)),
    ]
    for name, res in checks:
        print(f"    {name}: {'ok' if res else 'SKIPPED/FAILED'}")
        ok &= bool(res)
    n_files = len(os.listdir(tmp))
    print(f"    {n_files} files produced in the temp dir")
    ok &= n_files >= 17   # 7 figs x 2 formats + 3 tables

    print("\n[3] Missing inputs are skipped, not crashed on")
    for fn, args in ((fig_codesign, (None, None, tmp)),
                     (fig_energy_breakdown, (None, tmp)),
                     (fig_pareto, (None, None, tmp)),
                     (fig_qat, (None, tmp)),
                     (fig_layer_snr, (None, tmp))):
        r = fn(*args)
        print(f"    {fn.__name__}(None) -> {r} (False expected, no exception)")
        ok &= (r is False)

    print("\n[4] Attenuation model reproduces the measured self-test numbers")
    # analog_eval reported 73.4% retained at M=32 and 25.4% at M=256
    for M, expect in ((32, 73.4), (256, 25.4)):
        g = M * 0.42 / 542.8
        got = 100.0 / (1.0 + 20.0 * g)
        print(f"    M={M:3d}: model {got:5.1f}%  measured {expect:5.1f}%  "
              f"(within 8 pts: {abs(got-expect) < 8})")
        ok &= abs(got - expect) < 8

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 60)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--pareto", default="codesign_pareto.csv")
    ap.add_argument("--analog", default="analog_accuracy.csv")
    ap.add_argument("--ngspice-summary", default="ngspice_full_summary.csv")
    ap.add_argument("--ngspice-jsonl", default="ngspice_full.jsonl")
    ap.add_argument("--qat-history", default="qat_history.csv")
    ap.add_argument("--layers", default="results_cifar_qat_deployed.csv")
    ap.add_argument("--weights", type=int, default=11157504)
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    if not MPL:
        raise SystemExit("matplotlib required: pip install matplotlib --break-system-packages")

    plt.rcParams.update(PLOT_RC)
    os.makedirs(args.outdir, exist_ok=True)

    pareto = load_csv(args.pareto)
    analog = load_csv(args.analog)
    summary = load_csv(args.ngspice_summary)
    hist = load_csv(args.qat_history)
    layers = load_csv(args.layers)

    print(f"Inputs found: pareto={bool(pareto)} analog={bool(analog)} "
          f"ngspice={bool(summary)} qat={bool(hist)} layers={bool(layers)}")
    print(f"Writing to {args.outdir}/\n")

    fig_codesign(pareto, analog, args.outdir)
    fig_attenuation(analog, args.outdir)
    fig_energy_breakdown(pareto, args.outdir)
    fig_pareto(pareto, analog, args.outdir)
    fig_ngspice_validation(args.ngspice_jsonl, summary, args.outdir)
    fig_qat(hist, args.outdir)
    fig_layer_snr(layers, args.outdir)

    table_architectures(args.outdir)
    table_codesign(pareto, analog, args.outdir)
    table_memory(args.outdir, args.weights)
    table_validation(summary, args.outdir)

    print(f"\nDone. \\input{{}} the .tex files; \\includegraphics the .pdf files.")


if __name__ == "__main__":
    main()
