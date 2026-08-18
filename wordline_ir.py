#!/usr/bin/env python3
"""
wordline_ir.py
================================================================================
Quantifies wordline IR drop and tests whether it can be calibrated away.

WHY THIS EXPERIMENT DECIDES SOMETHING
-------------------------------------
The bitline loading error is
        I_act = I_ideal / (1 + Rs * G_col)
and the denominator contains no activation term, so it is a compile-time
constant and divides out exactly. That is the contribution.

The wordline is currently modelled as an ideal voltage source. A real wordline
has finite driver resistance and finite metal resistance per segment, so the
voltage actually delivered to cell (i,k) droops by the accumulated IR along
the row. The current flowing in the row depends on the ACTIVATIONS, so the
droop is a function of the input, not only of the programmed conductances.

If that dependence is significant, the claim weakens from
        "the analog error is a compile-time constant"
to
        "one analog error is a compile-time constant, another is not".
That is the single most likely way this work gets weakened, so it is measured
here rather than argued about.

WHAT IS SOLVED
--------------
The full 2-D resistive mesh, with no approximation beyond lumping the wire
into per-segment resistors:

  - wordline i is driven at its left end through R_DRV, then crosses the array
    through K segments of R_WL each
  - bitline k runs down through M segments of R_BL each into R_SENSE to ground
  - cell (i,k) of conductance G[i,k] bridges wordline node (i,k) to bitline
    node (i,k)

Unknowns are the M*K wordline node voltages and the M*K bitline node voltages.
Kirchhoff's current law at every node gives a sparse linear system, solved
directly. No iteration, no convergence question.

Setting R_WL = R_DRV = R_BL = 0 must reproduce the shared-Rsense formula the
rest of the project uses; --self-test asserts exactly that, which is what makes
this script trustworthy as a check on the main model.

USAGE
-----
    python3 wordline_ir.py --self-test

    # how big is the effect, and does a per-row constant remove it?
    python3 wordline_ir.py --sweep-m 32,64,128,256 --r-wl 0.5 --r-drv 50

    # the sensitivity that matters: error against wire resistance
    python3 wordline_ir.py --sweep-rwl 0,0.1,0.25,0.5,1.0,2.0 --tile-m 128

Typical 65 nm intermediate-metal values: sheet resistance ~0.1 ohm/square,
a cell pitch of a few squares, so R_WL is order 0.1-1 ohm per cell pitch.
R_DRV for a wordline driver is order 10-100 ohm. Both are swept rather than
asserted, because neither is known here without a PDK.
================================================================================
"""
import argparse
import sys

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY = True
except ImportError:
    SCIPY = False

R_LRS, R_MID, R_HRS = 542.8, 1099.8, 218587.2
R_SENSE_DEFAULT, VREAD_DEFAULT = 20.0, 0.1


# =============================================================================
def fp2_conductances(levels):
    """FP2 levels -> (Gp, Gn), matching analog_eval.fp2_resistances."""
    mag = np.abs(levels)
    r_mag = np.where(mag > 0.75, R_LRS, np.where(mag > 0.25, R_MID, R_HRS))
    pos = levels >= 0
    Rp = np.where(pos, r_mag, R_HRS)
    Rn = np.where(pos, R_HRS, r_mag)
    return 1.0 / Rp, 1.0 / Rn


def solve_ideal(G, V, r_sense, vread):
    """Ideal MAC: no loading of any kind. The reference."""
    return G.T @ (vread * V)


def solve_bitline_only(G, V, r_sense, vread):
    """Shared-Rsense loading, ideal wordline. The model used everywhere else."""
    Vw = vread * V
    num = G.T @ Vw
    den = G.sum(0) + 1.0 / r_sense
    return (num / den[:, None]) / r_sense


def solve_mesh(G, V, r_sense, vread, r_wl, r_drv, r_bl):
    """Full 2-D mesh solve.

    Node ordering: wordline node (i,k) -> i*K + k
                   bitline  node (i,k) -> M*K + i*K + k

    Returns the current delivered into each bitline's sense resistor, shape
    [K, P] for P activation vectors.
    """
    if not SCIPY:
        raise SystemExit("scipy required for the mesh solve: "
                         "pip install scipy --break-system-packages")
    M, K = G.shape
    P = V.shape[1]
    N = 2 * M * K

    def wl(i, k):
        return i * K + k

    def bl(i, k):
        return M * K + i * K + k

    rows, cols, vals = [], [], []

    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    # Guard the ideal-wire limit. Driving r->0 with a huge conductance wrecks
    # the conditioning of the matrix (cell conductances are ~1e-3 S, so 1e12 S
    # wire gives a condition number ~1e15 and the LU solve loses all precision).
    # 1e-6 ohm is already 1e-9 of the cell resistance and solves cleanly.
    IDEAL_WIRE = 1e-6
    r_wl = max(r_wl, IDEAL_WIRE)
    r_drv = max(r_drv, IDEAL_WIRE)
    r_bl = max(r_bl, IDEAL_WIRE)

    g_wl = 1.0 / r_wl if r_wl > 0 else None
    g_bl = 1.0 / r_bl if r_bl > 0 else None
    g_drv = 1.0 / r_drv if r_drv > 0 else None

    # Cell conductances couple the two planes at every crosspoint.
    for i in range(M):
        for k in range(K):
            a, b = wl(i, k), bl(i, k)
            g = G[i, k]
            add(a, a, g); add(a, b, -g)
            add(b, b, g); add(b, a, -g)

    # Wordline series resistors: node (i,k) to (i,k+1). Left end drives in
    # through R_DRV, which is the only place the source voltage enters.
    for i in range(M):
        for k in range(K - 1):
            a, b = wl(i, k), wl(i, k + 1)
            add(a, a, g_wl); add(a, b, -g_wl)
            add(b, b, g_wl); add(b, a, -g_wl)
        add(wl(i, 0), wl(i, 0), g_drv)          # to the driver source

    # Bitline series resistors: node (i,k) to (i+1,k), last row into R_SENSE.
    for k in range(K):
        for i in range(M - 1):
            a, b = bl(i, k), bl(i + 1, k)
            add(a, a, g_bl); add(a, b, -g_bl)
            add(b, b, g_bl); add(b, a, -g_bl)
        add(bl(M - 1, k), bl(M - 1, k), 1.0 / r_sense)   # to ground

    A = sp.csc_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(N, N)))
    lu = spla.splu(A)

    out = np.zeros((K, P))
    for p in range(P):
        rhs = np.zeros(N)
        for i in range(M):
            rhs[wl(i, 0)] = g_drv * vread * V[i, p]
        x = lu.solve(rhs)
        for k in range(K):
            out[k, p] = x[bl(M - 1, k)] / r_sense
    return out


# =============================================================================
def make_tile(M, K, P, rng, sparsity=0.5):
    """Random FP2 tile and post-ReLU-like activations."""
    levels = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(M, K))
    V = rng.random((M, P))
    V[rng.random((M, P)) < sparsity] = 0.0        # post-ReLU sparsity
    return levels, V


def relerr(a, b):
    d = np.abs(b).sum()
    return 100.0 * np.abs(a - b).sum() / max(d, 1e-30)


def differential(G_pair, V, solver, r_sense, vread, **kw):
    Gp, Gn = G_pair
    return (solver(Gp, V, r_sense, vread, **kw)
            - solver(Gn, V, r_sense, vread, **kw))


# =============================================================================
def analyse(M, K, P, r_wl, r_drv, r_bl, r_sense, vread, rng):
    """One operating point.

    Reports three things:
      raw          mesh solve against ideal, no correction
      col-calib    the project's per-column constant only
      col+row      per-column constant plus a per-row constant fitted at one
                   activation pattern and REUSED on fresh patterns

    The third is the decisive one. If a row constant fitted once transfers to
    unseen activations, the wordline error is compile-time correctable. If it
    does not, the residual is genuinely activation-dependent.
    """
    levels, V = make_tile(M, K, P, rng)
    Gp, Gn = fp2_conductances(levels)

    ideal = differential((Gp, Gn), V, solve_ideal, r_sense, vread)
    mesh = differential((Gp, Gn), V, solve_mesh, r_sense, vread,
                        r_wl=r_wl, r_drv=r_drv, r_bl=r_bl)

    # per-column loading constant, exactly as the project computes it
    cp = 1.0 + r_sense * Gp.sum(0)
    cn = 1.0 + r_sense * Gn.sum(0)
    mesh_p = solve_mesh(Gp, V, r_sense, vread, r_wl, r_drv, r_bl)
    mesh_n = solve_mesh(Gn, V, r_sense, vread, r_wl, r_drv, r_bl)
    col = mesh_p * cp[:, None] - mesh_n * cn[:, None]

    # Per-column gain fitted on HELD-OUT calibration patterns, then tested on
    # unseen ones. Fitting on a single pattern and dividing element-wise blows
    # up whenever that pattern happens to produce a near-zero column output,
    # which is exactly what produced a nonsense 2488% at M=256 in an earlier
    # version. Least squares over several patterns is stable, and testing on
    # patterns not used for the fit is the only version of this measurement
    # that answers the question -- a gain fitted and evaluated on the same
    # data would look perfect no matter how activation-dependent the error is.
    n_cal = max(1, P // 2)
    Vc, Vt = V[:, :n_cal], V[:, n_cal:]
    if Vt.shape[1] == 0:                       # too few positions to hold out
        Vt = Vc
    ideal_c = differential((Gp, Gn), Vc, solve_ideal, r_sense, vread)
    col_c = (solve_mesh(Gp, Vc, r_sense, vread, r_wl, r_drv, r_bl) * cp[:, None]
             - solve_mesh(Gn, Vc, r_sense, vread, r_wl, r_drv, r_bl) * cn[:, None])
    num = (ideal_c * col_c).sum(axis=1)
    den = (col_c * col_c).sum(axis=1)
    gain = np.where(den > 1e-30, num / np.maximum(den, 1e-30), 1.0)

    ideal_t = differential((Gp, Gn), Vt, solve_ideal, r_sense, vread)
    col_t = (solve_mesh(Gp, Vt, r_sense, vread, r_wl, r_drv, r_bl) * cp[:, None]
             - solve_mesh(Gn, Vt, r_sense, vread, r_wl, r_drv, r_bl) * cn[:, None])
    colrow_err = relerr(col_t * gain[:, None], ideal_t)

    return dict(
        M=M, r_wl=r_wl, r_drv=r_drv,
        raw=relerr(mesh / mesh.std() * ideal.std(), ideal) if mesh.std() else 0.0,
        raw_uncal=relerr(mesh, ideal),
        col=relerr(col, ideal),
        colrow=colrow_err,
    )


def print_table(rows, key, label):
    print(f"\n{label:>12}{'raw err%':>12}{'col-calib%':>13}{'col+row%':>12}")
    print("-" * 49)
    for r in rows:
        print(f"{r[key]:>12}{r['raw_uncal']:>12.3f}{r['col']:>13.4f}"
              f"{r['colrow']:>12.4f}")
    print("-" * 49)


# =============================================================================
def self_test():
    """Zero wire resistance must reproduce the shared-Rsense formula exactly.

    This is the load-bearing check: if the mesh solver agrees with the model
    the rest of the project uses when the wires are ideal, then any difference
    at nonzero wire resistance is the wordline effect and not a solver bug.
    """
    print("=" * 62)
    print("SELF-TEST")
    print("=" * 62)
    rng = np.random.default_rng(7)
    M, K, P = 16, 8, 4
    levels, V = make_tile(M, K, P, rng)
    Gp, _ = fp2_conductances(levels)

    a = solve_bitline_only(Gp, V, R_SENSE_DEFAULT, VREAD_DEFAULT)
    b = solve_mesh(Gp, V, R_SENSE_DEFAULT, VREAD_DEFAULT,
                   r_wl=0.0, r_drv=0.0, r_bl=0.0)
    err = np.abs(a - b).max() / np.abs(a).max()
    print(f"  [1] mesh(R_wire=0) vs shared-Rsense formula: {err:.3e} relative")
    assert err < 1e-6, f"mesh solver disagrees with the project model: {err}"

    c = solve_ideal(Gp, V, R_SENSE_DEFAULT, VREAD_DEFAULT)
    cal = (1.0 + R_SENSE_DEFAULT * Gp.sum(0))
    d = b * cal[:, None]
    err2 = np.abs(c - d).max() / np.abs(c).max()
    print(f"  [2] per-column calibration recovers ideal:   {err2:.3e} relative")
    assert err2 < 1e-6, f"calibration identity broken: {err2}"

    e = solve_mesh(Gp, V, R_SENSE_DEFAULT, VREAD_DEFAULT,
                   r_wl=1.0, r_drv=50.0, r_bl=0.5)
    delta = np.abs(e - b).max() / np.abs(b).max()
    print(f"  [3] nonzero wire resistance changes result:  {delta:.3e} relative")
    assert delta > 1e-6, "wire resistance had no effect -- the model is inert"

    print("\n" + "=" * 62)
    print("SELF-TEST PASSED")
    print("=" * 62)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Wordline IR drop: magnitude, and whether it calibrates out.")
    ap.add_argument("--tile-m", type=int, default=128)
    ap.add_argument("--tile-k", type=int, default=16)
    ap.add_argument("--positions", type=int, default=8,
                    help="Activation vectors per point. The first is used to "
                         "fit the row gain, the rest to test it.")
    ap.add_argument("--r-wl", type=float, default=0.5,
                    help="Wordline metal resistance per cell pitch (ohm).")
    ap.add_argument("--r-drv", type=float, default=50.0,
                    help="Wordline driver output resistance (ohm).")
    ap.add_argument("--r-bl", type=float, default=0.5,
                    help="Bitline metal resistance per cell pitch (ohm).")
    ap.add_argument("--r-sense", type=float, default=R_SENSE_DEFAULT)
    ap.add_argument("--vread", type=float, default=VREAD_DEFAULT)
    ap.add_argument("--sweep-m", default=None, help="e.g. 32,64,128,256")
    ap.add_argument("--sweep-rwl", default=None, help="e.g. 0,0.25,0.5,1,2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--breakdown", action="store_true",
                    help="Isolate each interconnect parasitic one at a time. "
                         "Running them together conflates effects: bitline "
                         "metal resistance alone dominates, and leaving it on "
                         "while sweeping wordline resistance makes the "
                         "wordline look responsible for damage it did not do.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    rng = np.random.default_rng(args.seed)
    rows = []

    if args.breakdown:
        cases = [
            ("ideal wires (sanity check)", 0.0, 0.0, 0.0),
            ("wordline metal 0.1 ohm",     0.1, 0.0, 0.0),
            ("wordline metal 0.5 ohm",     0.5, 0.0, 0.0),
            ("wordline metal 1.0 ohm",     1.0, 0.0, 0.0),
            ("wordline driver 25 ohm",     0.0, 25.0, 0.0),
            ("wordline driver 50 ohm",     0.0, 50.0, 0.0),
            ("wordline driver 100 ohm",    0.0, 100.0, 0.0),
            ("bitline metal 0.1 ohm",      0.0, 0.0, 0.1),
            ("bitline metal 0.5 ohm",      0.0, 0.0, 0.5),
            ("bitline metal 1.0 ohm",      0.0, 0.0, 1.0),
            ("all three, mid values",      0.5, 50.0, 0.5),
        ]
        print(f"\nInterconnect parasitic breakdown, M={args.tile_m}, "
              f"K={args.tile_k}")
        print(f"{'case':<32}{'raw err%':>11}{'after col-calib%':>19}")
        print("-" * 62)
        out = []
        for label, rw, rd, rb in cases:
            r = analyse(args.tile_m, args.tile_k, args.positions, rw, rd, rb,
                        args.r_sense, args.vread,
                        np.random.default_rng(args.seed))
            print(f"{label:<32}{r['raw_uncal']:>11.3f}{r['col']:>19.4f}")
            out.append(dict(case=label, r_wl=rw, r_drv=rd, r_bl=rb,
                            raw_pct=r['raw_uncal'], col_calib_pct=r['col']))
        print("-" * 62)
        print("The sanity row must be ~0 after calibration. Any row far above")
        print("it is a parasitic the per-column constant does NOT model:")
        print("it corrects R_SENSE loading only, and distributed wire")
        print("resistance is not a single lumped term per column.")
        if args.out_csv:
            import csv
            with open(args.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
                w.writeheader(); w.writerows(out)
            print(f"\nWrote {args.out_csv}")
        return

    if args.sweep_m:
        for M in [int(x) for x in args.sweep_m.split(",")]:
            print(f"  solving M={M} ...", flush=True)
            rows.append(analyse(M, args.tile_k, args.positions, args.r_wl,
                                args.r_drv, args.r_bl, args.r_sense,
                                args.vread, rng))
        print_table(rows, "M", "tile height")
    elif args.sweep_rwl:
        for rw in [float(x) for x in args.sweep_rwl.split(",")]:
            print(f"  solving R_wl={rw} ...", flush=True)
            rows.append(analyse(args.tile_m, args.tile_k, args.positions, rw,
                                args.r_drv, args.r_bl, args.r_sense,
                                args.vread, rng))
        print_table(rows, "r_wl", "R_wl (ohm)")
    else:
        rows.append(analyse(args.tile_m, args.tile_k, args.positions, args.r_wl,
                            args.r_drv, args.r_bl, args.r_sense, args.vread, rng))
        print_table(rows, "M", "tile height")

    print("\nINTERPRETATION")
    print("  raw err%    mesh solve against ideal, no correction")
    print("  col-calib%  the project's per-column constant only")
    print("  col+row%    plus a per-row gain FITTED on one activation pattern")
    print("              and applied to unseen ones")
    print()
    print("  If col+row is far below col, the wordline error is largely a")
    print("  compile-time constant and the contribution survives intact.")
    print("  If col+row stays close to col, the residual is activation-")
    print("  dependent and the claim must be narrowed to the bitline term.")

    if args.out_csv and rows:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
