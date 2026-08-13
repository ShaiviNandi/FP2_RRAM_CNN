#!/usr/bin/env python3
"""
ngspice_full_sweep.py
================================================================================
Runs EVERY crossbar tile of EVERY layer through real ngspice, in parallel,
with checkpoint/resume -- the exhaustive version of what
benchmark_resnet18.py --validate-ngspice does on 10 random samples.

Also extracts POWER, which the fast nodal model in benchmark_resnet18.py
never reported.

WHY THIS IS DIFFERENT FROM --validate-ngspice
---------------------------------------------
`--validate-ngspice 10` spot-checks ten randomly chosen (layer, tile,
position) triples and reports agreement of ~1e-7%. That is strong evidence
but it is a sample, and a sample cannot rule out a systematic failure
confined to a tile shape that happens not to be sampled (e.g. the trailing
partial M-tile of a layer whose M is not a multiple of 32, or a tile whose
weights are all zero so both bitlines float at the same potential). This
script visits all of them.

BATCHED NETLISTS -- AND WHY THAT IS PHYSICALLY HONEST
-----------------------------------------------------
One ngspice invocation per (tile, position) means ~600k process spawns, and
spawn+parse dominates the runtime of a circuit this small. So this script
packs `--positions-per-netlist` positions into a SINGLE netlist as that many
electrically independent copies of the same tile (node names suffixed _p).

That is not a modelling shortcut. The crossbar is weight-stationary: the
same programmed resistances are read repeatedly with different wordline
voltages, and in this read-only DC topology the columns share no node with
each other, so P sequential reads and P parallel isolated copies have
identical solutions. `--self-test` proves this rather than assuming it: it
diffs the batched netlist against (a) the unbatched
crossbar_array_test.build_array_read_netlist through the same ngspice and
(b) the nodal golden model, and fails loudly on any disagreement.

POWER
-----
Read power is computed from the ngspice node voltages by Ohm's law, which
is exact given those voltages -- no second simulation and no estimate:

    P_cells = sum_ijk (V_wl_i - V_bl_jk)^2 / R_ijk        (both polarities)
    P_sense = sum_jk V_bl_jk^2 / R_sense
    P_total = P_cells + P_sense

This is READ power for one MAC operation of one tile. It excludes: wordline
driver and DAC power, the ADC (typically the dominant term in a real analog
macro -- see hw_model.py, which models it), digital partial-sum accumulation,
and all WRITE/programming energy. Those exclusions are why hw_model.py
exists; this script supplies the array-level term it needs.

USAGE
-----
    # 0) Prove the batched netlist == unbatched netlist == nodal model
    python3 ngspice_full_sweep.py --self-test

    # 1) Time-estimate before committing to the run
    python3 ngspice_full_sweep.py --checkpoint resnet18_cifar10_fp2qat.pth \\
        --cifar-arch --num-classes 10 --calib-dataset cifar10 --calib-dir ./data \\
        --max-positions 32 --dry-run

    # 2) The real thing (resumable -- rerun the identical command after a
    #    crash or Ctrl-C and it picks up where it stopped)
    python3 ngspice_full_sweep.py --checkpoint resnet18_cifar10_fp2qat.pth \\
        --cifar-arch --num-classes 10 --calib-dataset cifar10 --calib-dir ./data \\
        --max-positions 32 --skip-first-last --workers 8 \\
        --out-jsonl ngspice_full.jsonl --out-csv ngspice_full_summary.csv

    # 3) Summarize an interrupted/finished run without re-simulating
    python3 ngspice_full_sweep.py --summarize-only --out-jsonl ngspice_full.jsonl \\
        --out-csv ngspice_full_summary.csv
================================================================================
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import crossbar_array_test as cb
import column_mac_test as col

R_LRS = col.R_FOR_MAGNITUDE[1.0]
G_LRS = 1.0 / R_LRS


# =============================================================================
# Batched netlist
# =============================================================================
def build_batched_read_netlist(W, act_cols, r_sense, vread, results_file):
    """P independent copies of one M x K differential read, one per column of
    `act_cols` ([M, P]). Mirrors crossbar_array_test.build_array_read_netlist
    element for element -- same device names, same resistance lookup via
    cb.decompose_differential, same Rsense-to-ground topology -- with a _p
    suffix on every node so the copies cannot interact.

    Emitting the resistors once per position is unavoidable in a linear DC
    netlist (there is no way to reuse a resistor at two different terminal
    voltages), so netlist size grows as P*M*K*2. Keep --positions-per-netlist
    moderate; the sweet spot is where spawn overhead and solve time balance,
    which --dry-run measures on the host machine rather than guessing."""
    M, K = len(W), len(W[0])
    P = len(act_cols[0])
    lines = [f"* batched 2T2R read: M={M} K={K} P={P}", ""]

    # Resistances depend only on the weights, so resolve them once instead of
    # calling decompose_differential P*M*K times.
    rp = [[0.0] * K for _ in range(M)]
    rn = [[0.0] * K for _ in range(M)]
    for i in range(M):
        for k in range(K):
            _, _, a, b = cb.decompose_differential(W[i][k])
            rp[i][k], rn[i][k] = a, b

    for p in range(P):
        for i in range(M):
            v = vread * act_cols[i][p]
            lines.append(f"Vrow{i}_{p} wl{i}_{p} 0 DC {v}")
            for k in range(K):
                lines.append(f"Rcell{i}_{k}p_{p} wl{i}_{p} blp{k}_{p} {rp[i][k]}")
                lines.append(f"Rcell{i}_{k}n_{p} wl{i}_{p} bln{k}_{p} {rn[i][k]}")
        for k in range(K):
            lines.append(f"Rsensep{k}_{p} blp{k}_{p} 0 {r_sense}")
            lines.append(f"Rsensen{k}_{p} bln{k}_{p} 0 {r_sense}")
        lines.append("")

    lines += [".op", ".control", "run"]
    vecs = " ".join(f"v(blp{k}_{p}) v(bln{k}_{p})" for p in range(P) for k in range(K))
    lines.append(f"wrdata {results_file} {vecs}")
    lines += [".endc", ".end"]
    return "\n".join(lines)


def run_batched_netlist(W, act_cols, r_sense, vread, ngspice_bin="ngspice", timeout=600):
    """Returns (v_blp[P][K], v_bln[P][K]) from a real ngspice DC solve."""
    M, K = len(W), len(W[0])
    P = len(act_cols[0])
    tmpdir = tempfile.mkdtemp(prefix="ngsweep_")
    try:
        res_path = os.path.join(tmpdir, "res.txt")
        net = build_batched_read_netlist(W, act_cols, r_sense, vread, res_path)
        net_path = os.path.join(tmpdir, "tile.sp")
        with open(net_path, "w") as f:
            f.write(net)
        proc = subprocess.run([ngspice_bin, "-b", net_path], cwd=tmpdir,
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not os.path.exists(res_path):
            raise RuntimeError(f"ngspice failed (exit {proc.returncode}): "
                               f"{proc.stderr[-1500:]}")
        with open(res_path) as f:
            vals = [float(x) for x in f.read().split()]
        expect = 2 * (2 * K * P)  # (x,y) pairs for 2*K*P vectors
        if len(vals) != expect:
            raise RuntimeError(f"expected {expect} values in wrdata output, got {len(vals)}")
        y = vals[1::2]
        v_blp = [[0.0] * K for _ in range(P)]
        v_bln = [[0.0] * K for _ in range(P)]
        idx = 0
        for p in range(P):
            for k in range(K):
                v_blp[p][k] = y[idx]; idx += 1
                v_bln[p][k] = y[idx]; idx += 1
        return v_blp, v_bln
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def tile_read_power(W, act_cols, v_blp, v_bln, r_sense, vread):
    """Exact read power per position, from the solved node voltages.
    Returns a list of P floats in watts."""
    M, K = len(W), len(W[0])
    P = len(act_cols[0])
    rp = [[0.0] * K for _ in range(M)]
    rn = [[0.0] * K for _ in range(M)]
    for i in range(M):
        for k in range(K):
            _, _, a, b = cb.decompose_differential(W[i][k])
            rp[i][k], rn[i][k] = a, b
    out = []
    for p in range(P):
        pw = 0.0
        for k in range(K):
            pw += v_blp[p][k] ** 2 / r_sense + v_bln[p][k] ** 2 / r_sense
        for i in range(M):
            v_wl = vread * act_cols[i][p]
            for k in range(K):
                pw += (v_wl - v_blp[p][k]) ** 2 / rp[i][k]
                pw += (v_wl - v_bln[p][k]) ** 2 / rn[i][k]
        out.append(pw)
    return out


# =============================================================================
# Worker
# =============================================================================
def _worker(job):
    """One (layer, m-tile, k-tile, position-chunk) unit of work."""
    (layer, m0, m1, k0, k1, p0, W_list, act_list, r_sense, vread,
     scale_tile, ngspice_bin) = job
    t0 = time.time()
    try:
        v_blp, v_bln = run_batched_netlist(W_list, act_list, r_sense, vread, ngspice_bin)
    except Exception as e:
        return dict(layer=layer, m0=m0, m1=m1, k0=k0, k1=k1, p0=p0,
                    error=str(e)[:500], seconds=time.time() - t0)

    K = len(W_list[0])
    P = len(act_list[0])
    powers = tile_read_power(W_list, act_list, v_blp, v_bln, r_sense, vread)

    spice_out, nodal_out = [], []
    for p in range(P):
        acts_p = [act_list[i][p] for i in range(len(act_list))]
        nodal = cb.golden_array_matmul(W_list, acts_p, r_sense, vread)
        for k in range(K):
            i_spice = (v_blp[p][k] - v_bln[p][k]) / r_sense
            # same recovery scaling benchmark_resnet18.analog_2t2r_forward uses
            spice_out.append(i_spice / (vread * G_LRS) * scale_tile[k])
            nodal_out.append(nodal[k] / (vread * G_LRS) * scale_tile[k])

    s = np.asarray(spice_out); n = np.asarray(nodal_out)
    den = np.abs(n).sum()
    return dict(
        layer=layer, m0=m0, m1=m1, k0=k0, k1=k1, p0=p0, P=P, K=K,
        rel_err_pct=float(100.0 * np.abs(s - n).sum() / den) if den > 1e-18 else 0.0,
        max_abs_diff=float(np.abs(s - n).max()),
        power_mean_w=float(np.mean(powers)), power_max_w=float(np.max(powers)),
        n_macs=int(P * K * (m1 - m0)),
        seconds=time.time() - t0,
    )


# =============================================================================
# Job construction
# =============================================================================
def build_jobs(layers, args):
    import benchmark_resnet18 as bm
    jobs = []
    for layer in layers:
        W_MK = bm.weight_to_MK(layer.weight)
        acts = layer.unfolded_input
        M, K = W_MK.shape
        scale = bm.weight_scale_factor(W_MK, block_size=args.tile_m,
                                       scale_mode=args.scale_mode)
        m_ranges = bm.tile_ranges(M, args.tile_m)
        k_ranges = bm.tile_ranges(K, args.tile_k)

        Wq = np.zeros_like(W_MK)
        for i, (m0, m1) in enumerate(m_ranges):
            Wq[m0:m1, :] = np.vectorize(cb.quantize_to_fp2)(W_MK[m0:m1, :] / scale[i][None, :])

        n_pos = acts.shape[1]
        for (k0, k1) in k_ranges:
            for i, (m0, m1) in enumerate(m_ranges):
                W_list = Wq[m0:m1, k0:k1].tolist()
                for p0 in range(0, n_pos, args.positions_per_netlist):
                    p1 = min(p0 + args.positions_per_netlist, n_pos)
                    act_list = acts[m0:m1, p0:p1].tolist()
                    jobs.append((layer.name, m0, m1, k0, k1, p0, W_list, act_list,
                                 args.r_sense, args.vread,
                                 scale[i, k0:k1].tolist(), args.ngspice_bin))
    return jobs


def job_key(j):
    return f"{j[0]}|{j[1]}|{j[3]}|{j[5]}"


def result_key(r):
    return f"{r['layer']}|{r['m0']}|{r['k0']}|{r['p0']}"


# =============================================================================
# Summary
# =============================================================================
def summarize(jsonl_path, csv_path):
    if not os.path.exists(jsonl_path):
        raise SystemExit(f"No results file at {jsonl_path}")
    per = defaultdict(list)
    errors = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                errors.append(r)
            else:
                per[r["layer"]].append(r)

    rows = []
    for layer, rs in per.items():
        rel = np.array([r["rel_err_pct"] for r in rs])
        pw = np.array([r["power_mean_w"] for r in rs])
        macs = sum(r["n_macs"] for r in rs)
        rows.append(dict(
            layer=layer, tiles_simulated=len(rs),
            rel_err_mean_pct=float(rel.mean()), rel_err_max_pct=float(rel.max()),
            power_mean_uw=float(pw.mean() * 1e6),
            power_max_uw=float(np.array([r["power_max_w"] for r in rs]).max() * 1e6),
            total_tile_power_mw=float(pw.sum() * 1e3),
            macs=macs,
            pJ_per_mac=float(pw.sum() * 1e12 / macs) if macs else float("nan"),
            sim_seconds=float(sum(r["seconds"] for r in rs)),
        ))
    rows.sort(key=lambda r: r["layer"])

    print(f"\n{'Layer':<24}{'tiles':>8}{'RelErr mean%':>14}{'RelErr max%':>13}"
          f"{'P mean(uW)':>12}{'P max(uW)':>11}")
    print("-" * 82)
    for r in rows:
        print(f"{r['layer']:<24}{r['tiles_simulated']:>8}{r['rel_err_mean_pct']:>14.3e}"
              f"{r['rel_err_max_pct']:>13.3e}{r['power_mean_uw']:>12.2f}{r['power_max_uw']:>11.2f}")
    print("-" * 82)
    if rows:
        allrel = np.array([r["rel_err_max_pct"] for r in rows])
        print(f"Worst per-layer max relative error (ngspice vs nodal): {allrel.max():.3e} %")
        print(f"Total tiles simulated: {sum(r['tiles_simulated'] for r in rows)}")
        tot_p = sum(r["total_tile_power_mw"] for r in rows)
        tot_m = sum(r["macs"] for r in rows)
        print(f"Array read power, summed over all simulated tile-reads: {tot_p:.3f} mW")
        print(f"Array-only read energy per MAC: {tot_p*1e9/tot_m:.4f} pJ/MAC "
              f"(EXCLUDES ADC/DAC/driver/digital -- see hw_model.py)")
        if allrel.max() < 1e-3:
            print("\n[PASS] Every simulated tile agrees with the nodal model to <1e-3%. "
                  "The fast model in benchmark_resnet18.py is exact for this topology, "
                  "not approximately right on a lucky sample.")
        else:
            print("\n[FAIL] At least one tile disagrees by >1e-3%. Inspect the worst "
                  "offenders before trusting any nodal-model result.", file=sys.stderr)
    if errors:
        print(f"\n{len(errors)} tiles ERRORED. First 3:", file=sys.stderr)
        for e in errors[:3]:
            print(f"  {e['layer']} m0={e['m0']} k0={e['k0']}: {e['error'][:200]}", file=sys.stderr)

    if csv_path and rows:
        import csv as _csv
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nSummary written to {csv_path}")
    return rows


# =============================================================================
# Self-test
# =============================================================================
def self_test(ngspice_bin="ngspice"):
    print("=== ngspice_full_sweep self-test ===\n")
    if shutil.which(ngspice_bin) is None:
        print(f"ngspice not found ({ngspice_bin}). Install: sudo apt-get install -y ngspice",
              file=sys.stderr)
        return 1
    ok = True
    rng = np.random.default_rng(0)

    print("[1] Batched netlist == unbatched netlist == nodal model")
    M, K, P = 12, 5, 4
    W = [[float(rng.choice(cb.FP2_LEVELS)) for _ in range(K)] for _ in range(M)]
    acts = rng.uniform(-1, 1, size=(M, P)).tolist()
    r_sense, vread = 20.0, 0.1

    v_blp, v_bln = run_batched_netlist(W, acts, r_sense, vread, ngspice_bin)
    worst_vs_unbatched = 0.0
    worst_vs_nodal = 0.0
    import ngspice_bridge as nb
    for p in range(P):
        acts_p = [acts[i][p] for i in range(M)]
        batched = [(v_blp[p][k] - v_bln[p][k]) / r_sense for k in range(K)]
        unbatched = nb.spice_array_matmul(W, acts_p, r_sense, vread)
        nodal = cb.golden_array_matmul(W, acts_p, r_sense, vread)
        for k in range(K):
            den = abs(unbatched[k]) if abs(unbatched[k]) > 1e-15 else 1.0
            worst_vs_unbatched = max(worst_vs_unbatched, abs(batched[k] - unbatched[k]) / den * 100)
            den2 = abs(nodal[k]) if abs(nodal[k]) > 1e-15 else 1.0
            worst_vs_nodal = max(worst_vs_nodal, abs(batched[k] - nodal[k]) / den2 * 100)
    print(f"    batched vs UNBATCHED ngspice : worst {worst_vs_unbatched:.3e} %")
    print(f"    batched vs nodal golden model: worst {worst_vs_nodal:.3e} %")
    ok &= worst_vs_unbatched < 1e-6 and worst_vs_nodal < 1e-3

    print("\n[2] Degenerate tiles (all-zero weights, zero activations)")
    Wz = [[0.0] * K for _ in range(M)]
    vp, vn = run_batched_netlist(Wz, acts, r_sense, vread, ngspice_bin)
    nod = cb.golden_array_matmul(Wz, [acts[i][0] for i in range(M)], r_sense, vread)
    d = max(abs((vp[0][k] - vn[0][k]) / r_sense - nod[k]) for k in range(K))
    print(f"    all-zero weight tile: max abs current diff {d:.3e} A "
          f"(both bitlines sit at the same HRS-divided potential)")
    ok &= d < 1e-12
    az = [[0.0] * P for _ in range(M)]
    vp2, _ = run_batched_netlist(W, az, r_sense, vread, ngspice_bin)
    print(f"    zero-activation tile: max |v_blp| {max(abs(x) for r in vp2 for x in r):.3e} V "
          f"(no drive => no current)")
    ok &= max(abs(x) for r in vp2 for x in r) < 1e-12

    print("\n[3] Power sanity")
    pw = tile_read_power(W, acts, v_blp, v_bln, r_sense, vread)
    # Every wordline is driven from |v|<=vread through >= R_LRS to a bitline
    # that is near ground, so per-cell power cannot exceed vread^2/R_LRS.
    bound = M * K * 2 * (vread ** 2) / R_LRS
    print(f"    mean {np.mean(pw)*1e6:.3f} uW   max {max(pw)*1e6:.3f} uW   "
          f"loose upper bound {bound*1e6:.3f} uW")
    ok &= all(0 <= p <= bound * 1.001 for p in pw)

    print("\n[4] Timing model for the full run")
    t0 = time.time(); run_batched_netlist(W, acts, r_sense, vread, ngspice_bin)
    dt = time.time() - t0
    print(f"    one {M}x{K} tile x {P} positions: {dt*1000:.1f} ms")

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 60)
    return 0 if ok else 1


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Re-summarize an existing --out-jsonl without simulating")
    ap.add_argument("--dry-run", action="store_true",
                    help="Enumerate the work and estimate wall time, then exit")

    # model / calibration -- mirrors benchmark_resnet18.py so the two sweeps
    # describe the identical weights and activations
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cifar-arch", action="store_true")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--calib-dataset", default="random",
                    choices=["random", "cifar10", "cifar100", "imagefolder"])
    ap.add_argument("--calib-dir", default="./data")
    ap.add_argument("--calib-images", type=int, default=16)
    ap.add_argument("--skip-first-last", action="store_true")
    ap.add_argument("--max-positions", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)

    # crossbar
    ap.add_argument("--tile-m", type=int, default=32)
    ap.add_argument("--tile-k", type=int, default=16)
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--scale-mode", default="e8m0", choices=["e8m0", "fp8", "none"])

    # execution
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--positions-per-netlist", type=int, default=8,
                    help="Positions packed into one ngspice invocation. Higher = fewer "
                         "process spawns but a larger solve; --dry-run measures the "
                         "trade-off on the host machine.")
    ap.add_argument("--ngspice-bin", default="ngspice")
    ap.add_argument("--out-jsonl", default="ngspice_full.jsonl")
    ap.add_argument("--out-csv", default="ngspice_full_summary.csv")
    ap.add_argument("--limit-jobs", type=int, default=0, help="Stop after N tiles (debug)")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test(args.ngspice_bin))
    if args.summarize_only:
        summarize(args.out_jsonl, args.out_csv)
        return

    if shutil.which(args.ngspice_bin) is None:
        raise SystemExit(f"ngspice not found ({args.ngspice_bin}). "
                         f"sudo apt-get install -y ngspice")

    import benchmark_resnet18 as bm
    np.random.seed(args.seed)

    print("Building model and capturing calibration activations...")
    model, is_synth = bm.build_model(args)
    layers = bm.extract_conv_layers(model, args)
    if args.skip_first_last and layers:
        print(f"--skip-first-last: excluding '{layers[0].name}'")
        layers = layers[1:]

    print("Enumerating tiles...")
    jobs = build_jobs(layers, args)
    if args.limit_jobs:
        jobs = jobs[: args.limit_jobs]

    done = set()
    if os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        if not r.get("error"):
                            done.add(result_key(r))
                    except json.JSONDecodeError:
                        pass  # torn last line from a hard kill; it just gets redone
        print(f"Resuming: {len(done)} tiles already complete in {args.out_jsonl}")

    todo = [j for j in jobs if job_key(j) not in done]
    total_macs = sum(len(j[7][0]) * len(j[6][0]) * len(j[6]) for j in jobs)
    print(f"\nTiles total {len(jobs)}   already done {len(jobs)-len(todo)}   to run {len(todo)}")
    print(f"ngspice invocations to run: {len(todo)}  "
          f"(positions packed {args.positions_per_netlist}/netlist)")
    print(f"MAC-reads represented: {total_macs:,}")

    if args.dry_run or todo:
        print("\nCalibrating wall-time on 3 representative tiles...")
        sample = [todo[0], todo[len(todo)//2], todo[-1]] if len(todo) >= 3 else todo[:1]
        t0 = time.time()
        for j in sample:
            _worker(j)
        per = (time.time() - t0) / max(len(sample), 1)
        est = per * len(todo) / max(args.workers, 1)
        print(f"  {per*1000:.0f} ms per tile-batch, {args.workers} workers")
        print(f"  estimated wall time: {est/60:.1f} min ({est/3600:.2f} h)")
    if args.dry_run:
        print("\n--dry-run: stopping before the real sweep.")
        return

    if not todo:
        print("Nothing to do -- all tiles already simulated. Summarizing.")
        summarize(args.out_jsonl, args.out_csv)
        return

    t_start = time.time()
    n_done, n_err = 0, 0
    # Append-and-flush per result: a kill at any moment loses at most the
    # in-flight tiles, and --resume picks up from the file.
    with open(args.out_jsonl, "a", buffering=1) as out, \
            ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, j): j for j in todo}
        for fut in as_completed(futs):
            r = fut.result()
            out.write(json.dumps(r) + "\n")
            n_done += 1
            if r.get("error"):
                n_err += 1
            if n_done % 200 == 0 or n_done == len(todo):
                el = time.time() - t_start
                rate = n_done / el
                eta = (len(todo) - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{len(todo)} tiles  {rate:.1f}/s  "
                      f"elapsed {el/60:.1f}m  ETA {eta/60:.1f}m  errors {n_err}")

    print(f"\nSweep finished in {(time.time()-t_start)/60:.1f} min, {n_err} errors.")
    summarize(args.out_jsonl, args.out_csv)


if __name__ == "__main__":
    main()
