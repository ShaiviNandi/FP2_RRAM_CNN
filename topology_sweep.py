#!/usr/bin/env python3
"""
Sweep array size (M and K) for both 1T1R and 2T2R topologies and plot the
resulting error, presenting the real tradeoff in one place instead of
picking topology and size separately.

ERROR METRIC: for each (M, K, topology, seed), generate a random FP2
weight matrix, compute the analog result (golden nodal-analysis model,
includes shared-Rsense loading -- and for 1T1R, the sign-loss limitation
since negative weights are physically unrepresentable), recover it into
the same units as the true digital dot product, and report:

    relative error % = 100 * sum(|analog - digital|) / sum(|digital|)

aggregated across all K columns of one array, then averaged across
--seeds independent random weight matrices (a single random matrix can
have an unlucky sign distribution, especially for 1T1R where sign-loss
dominates -- averaging smooths that out into a stable curve).

Produces three panels:
    1. Error vs M (fixed K)      -- the main accuracy-vs-size tradeoff
    2. Error vs K (fixed M)      -- sanity check that K doesn't matter
    3. Physical cell count vs M  -- the real area cost of each choice

Usage:
    python3 topology_sweep.py --k-fixed 8 --m-fixed 8 --seeds 5
"""
import argparse
import statistics

import crossbar_array_test as cb


def topology_error_and_cells(M, K, topology, r_sense, vread_base, seed):
    W = cb.random_fp2_matrix(M, K, seed=seed)
    activations = [1.0] * M
    digital_exact = [sum(activations[i] * cb.quantize_to_fp2(W[i][k]) for i in range(M)) for k in range(K)]
    g_lrs = 1.0 / cb.col.R_FOR_MAGNITUDE[1.0]

    if topology == "2t2r":
        golden = cb.golden_array_matmul(W, activations, r_sense, vread_base)
        cells = 2 * M * K
    else:
        golden = cb.golden_array_matmul_1t1r(W, activations, r_sense, vread_base)
        cells = M * K

    analog_recovered = [g / (vread_base * g_lrs) for g in golden]
    num = sum(abs(a - d) for a, d in zip(analog_recovered, digital_exact))
    den = sum(abs(d) for d in digital_exact)
    rel_err = 100.0 * num / den if den > 1e-9 else float("nan")
    return rel_err, cells


def averaged_sweep(m_values, k_values, topology, r_sense, vread_base, n_seeds):
    """m_values and k_values are paired lists of equal length (one config
    per point) -- caller builds these for 'sweep M, K fixed' or 'sweep K,
    M fixed'. Returns (mean_errs, cell_counts) lists, one per config."""
    mean_errs, cell_counts = [], []
    for M, K in zip(m_values, k_values):
        errs = []
        cells = None
        for seed in range(n_seeds):
            e, c = topology_error_and_cells(M, K, topology, r_sense, vread_base, seed)
            if e == e:  # skip NaN (den==0, vanishingly rare with sparsity<1)
                errs.append(e)
            cells = c
        mean_errs.append(statistics.mean(errs) if errs else float("nan"))
        cell_counts.append(cells)
    return mean_errs, cell_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m-values", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128, 256],
                     help="M values to sweep (K held at --k-fixed)")
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                     help="K values to sweep (M held at --m-fixed)")
    ap.add_argument("--k-fixed", type=int, default=8, help="K used while sweeping M")
    ap.add_argument("--m-fixed", type=int, default=8, help="M used while sweeping K")
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5, help="Number of random weight matrices averaged per point")
    ap.add_argument("--out", default="topology_sweep.png", help="Output plot filename")
    args = ap.parse_args()

    print(f"Sweeping M={args.m_values} (K={args.k_fixed} fixed), averaging {args.seeds} random matrices per point...")
    m_err_2t2r, m_cells_2t2r = averaged_sweep(args.m_values, [args.k_fixed] * len(args.m_values),
                                               "2t2r", args.r_sense, args.vread, args.seeds)
    m_err_1t1r, m_cells_1t1r = averaged_sweep(args.m_values, [args.k_fixed] * len(args.m_values),
                                               "1t1r", args.r_sense, args.vread, args.seeds)

    print(f"Sweeping K={args.k_values} (M={args.m_fixed} fixed)...")
    k_err_2t2r, k_cells_2t2r = averaged_sweep([args.m_fixed] * len(args.k_values), args.k_values,
                                               "2t2r", args.r_sense, args.vread, args.seeds)
    k_err_1t1r, k_cells_1t1r = averaged_sweep([args.m_fixed] * len(args.k_values), args.k_values,
                                               "1t1r", args.r_sense, args.vread, args.seeds)

    print("\n=== M sweep (K fixed) ===")
    print(f"{'M':>6} {'2T2R err%':>10} {'1T1R err%':>10} {'2T2R cells':>11} {'1T1R cells':>11}")
    for i, M in enumerate(args.m_values):
        print(f"{M:6d} {m_err_2t2r[i]:10.2f} {m_err_1t1r[i]:10.2f} {m_cells_2t2r[i]:11d} {m_cells_1t1r[i]:11d}")

    print("\n=== K sweep (M fixed) ===")
    print(f"{'K':>6} {'2T2R err%':>10} {'1T1R err%':>10}")
    for i, K in enumerate(args.k_values):
        print(f"{K:6d} {k_err_2t2r[i]:10.2f} {k_err_1t1r[i]:10.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- run 'pip3 install matplotlib' to get the plot.")
        print("(Numeric results above are already complete without it.)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(args.m_values, m_err_2t2r, "o-", label="2T2R (differential, signed)")
    ax.plot(args.m_values, m_err_1t1r, "s-", label="1T1R (single-ended, unsigned)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"M (array height, K={args.k_fixed} fixed)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Error vs M")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(args.k_values, k_err_2t2r, "o-", label="2T2R")
    ax.plot(args.k_values, k_err_1t1r, "s-", label="1T1R")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"K (array width, M={args.m_fixed} fixed)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Error vs K (sanity check: should be flat)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(args.m_values, m_cells_2t2r, "o-", label="2T2R")
    ax.plot(args.m_values, m_cells_1t1r, "s-", label="1T1R")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"M (array height, K={args.k_fixed} fixed)")
    ax.set_ylabel("Physical cell count")
    ax.set_title("Cell count vs M")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"1T1R vs 2T2R crossbar tradeoff (Rsense={args.r_sense}ohm, {args.seeds} seeds/point)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nPlot saved to {args.out}")


if __name__ == "__main__":
    main()
