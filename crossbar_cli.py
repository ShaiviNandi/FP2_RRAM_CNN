#!/usr/bin/env python3
"""
Single-run crossbar CLI: pick M, K, weights, activations, and topology
(1T1R or 2T2R), and see the DIGITAL (exact, intended) output next to the
ANALOG (circuit-realistic, includes shared-Rsense loading, and for 1T1R
the sign-loss limitation) output, per column.

This is the fast golden-model path (no ngspice needed) -- use it to
explore the design space quickly. For a real device-physics run, feed
the same W/activations into crossbar_array_test.py's netlist builders.

Usage examples:
    # 4x3 array, random FP2 weights, all-ones activations, 2T2R
    python3 crossbar_cli.py --m 4 --k 3 --topology 2t2r

    # Same, but 1T1R (unsigned) for comparison
    python3 crossbar_cli.py --m 4 --k 3 --topology 1t1r

    # Explicit weight matrix (rows separated by ';', values by ',')
    python3 crossbar_cli.py --weights "1,-1,0.5;-0.5,0.5,0;0.5,0,-1;0,1,0.5" \\
        --activations "1,1,1,1" --topology 2t2r

    # Random activations instead of all-ones
    python3 crossbar_cli.py --m 8 --k 4 --activations random --topology 2t2r
"""
import argparse
import random
import sys

import crossbar_array_test as cb


def parse_matrix(s):
    rows = [r for r in s.split(";") if r.strip()]
    return [[float(v) for v in r.split(",")] for r in rows]


def parse_vector(s):
    return [float(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, help="Array height (inputs) -- used if --weights not given")
    ap.add_argument("--k", type=int, help="Array width (outputs) -- used if --weights not given")
    ap.add_argument("--weights", type=str, default=None,
                     help="Explicit weight matrix: rows separated by ';', values by ',' "
                          "e.g. '1,-1,0.5;-0.5,0.5,0'. Overrides --m/--k.")
    ap.add_argument("--activations", type=str, default="ones",
                     help="'ones' (default), 'random', or explicit comma-separated values e.g. '1,0.5,-1'")
    ap.add_argument("--topology", choices=["1t1r", "2t2r"], default="2t2r",
                     help="1t1r: single-ended, unsigned only, half the cells. "
                          "2t2r: differential, signed, double the cells (default).")
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0, help="Seed for random weights/activations")
    ap.add_argument("--sparsity", type=float, default=0.15, help="Fraction of random weights forced to 0.0")
    ap.add_argument("--netlist-out", default=None, help="If set, also write the ngspice netlist here")
    args = ap.parse_args()

    if args.weights:
        W = parse_matrix(args.weights)
        M, K = len(W), len(W[0])
    else:
        if args.m is None or args.k is None:
            print("ERROR: provide --weights, or both --m and --k", file=sys.stderr)
            sys.exit(1)
        M, K = args.m, args.k
        W = cb.random_fp2_matrix(M, K, seed=args.seed, sparsity=args.sparsity)

    if args.activations == "ones":
        activations = [1.0] * M
    elif args.activations == "random":
        rng = random.Random(args.seed + 1)
        activations = [round(rng.uniform(-1.0, 1.0), 3) for _ in range(M)]
    else:
        activations = parse_vector(args.activations)
        if len(activations) != M:
            print(f"ERROR: --activations has {len(activations)} values, expected {M} (=M)", file=sys.stderr)
            sys.exit(1)

    print(f"Topology: {args.topology.upper()}   M={M}  K={K}  Rsense={args.r_sense}ohm  Vread={args.vread}V")
    print(f"Weights (quantized to FP2):")
    for row in W:
        print("  " + " ".join(f"{cb.quantize_to_fp2(w):+.1f}" for w in row))
    print(f"Activations: {activations}")
    print()

    if args.topology == "2t2r":
        golden = cb.golden_array_matmul(W, activations, args.r_sense, args.vread)
        ideal = cb.ideal_target_currents(W, activations, args.vread)
        digital_exact = [sum(activations[i] * cb.quantize_to_fp2(W[i][k]) for i in range(M)) for k in range(K)]
        cell_count = 2 * M * K
        bitlines = 2 * K
    else:
        golden = cb.golden_array_matmul_1t1r(W, activations, args.r_sense, args.vread)
        ideal = cb.ideal_target_currents_1t1r(W, activations, args.vread)
        # True (unclipped) digital reference -- shows what 1T1R MISSES, not just noise.
        digital_exact = [sum(activations[i] * cb.quantize_to_fp2(W[i][k]) for i in range(M)) for k in range(K)]
        cell_count = M * K
        bitlines = K

    g_lrs = 1.0 / cb.col.R_FOR_MAGNITUDE[1.0]
    # Recover the analog result back into the same units as the digital
    # dot product (divide out the current->weighted-sum scale factor) --
    # this is what makes the comparison directly readable.
    analog_recovered = [g / (args.vread * g_lrs) for g in golden]

    print(f"Physical cells: {cell_count}   Bitlines/TIA channels needed: {bitlines}")
    print()
    print(f"{'col':>4} {'digital (exact)':>16} {'analog (recovered)':>19} {'abs err':>10} {'err %':>8}")
    for k in range(K):
        d = digital_exact[k]
        a = analog_recovered[k]
        abs_err = abs(a - d)
        pct = 100.0 * abs_err / abs(d) if abs(d) > 1e-9 else float("nan")
        print(f"{k:4d} {d:16.4f} {a:19.4f} {abs_err:10.4f} {pct:8.2f}")

    if args.topology == "1t1r":
        n_negative = sum(1 for row in W for w in row if cb.quantize_to_fp2(w) < 0)
        total = M * K
        print(f"\n1T1R note: {n_negative}/{total} weights ({100*n_negative/total:.0f}%) are negative and "
              f"WERE NOT REPRESENTED at all (clipped to 0) -- this is a real limitation, not analog noise. "
              f"The 'digital (exact)' column above includes those negative contributions; 1T1R physically cannot.")

    print(f"\nRaw analog currents (A): {['%.4e' % g for g in golden]}")

    if args.netlist_out:
        if args.topology == "2t2r":
            netlist = cb.build_array_read_netlist(W, activations, args.r_sense, args.vread)
        else:
            netlist = cb.build_array_read_netlist_1t1r(W, activations, args.r_sense, args.vread)
        with open(args.netlist_out, "w") as f:
            f.write(netlist)
        print(f"\nNetlist written to {args.netlist_out}")


if __name__ == "__main__":
    main()
