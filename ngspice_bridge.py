#!/usr/bin/env python3
"""
ngspice_bridge.py
Actually invokes ngspice as a subprocess on netlists built by
crossbar_array_test.py, and parses its output back into the same units as
golden_array_matmul() -- so the two can be diffed directly.

This is the missing piece that closes the loop between the fast nodal-
analysis "golden model" (crossbar_array_test.golden_array_matmul, pure
Python, instant, used for the bulk per-layer sweep in
benchmark_resnet18.py) and real circuit simulation (ngspice solving the
same resistor network numerically). Since ngspice subprocess calls are
~10-50ms each (netlist write + process spawn + parse), this is meant for
spot-check validation of a handful of samples, not for replacing the fast
model in the full per-layer sweep -- see validate_against_ngspice() in
benchmark_resnet18.py for how it's actually used.

WHAT this does not DO: it still only exercises the fixed-resistor
READ-only netlist (build_array_read_netlist), i.e. DC .op steady-state
with each cell already at its programmed HRS/LRS resistance -- same as
golden_array_matmul(). It does not invoke the Stanford Verilog-A compact
model (rram_v_1_0_0_openvaf.va), so it still doesn't capture transient
switching dynamics, gap evolution, or thermal effects. It does confirm
the fast nodal model is solving the identical resistor network correctly,
which is a different (and cheaper to check) claim than "matches real
device physics."

Usage:
    python3 ngspice_bridge.py --self-test
        Runs a small built-in sanity check (2x2 crossbar, hand-verifiable)
        and reports nodal-vs-ngspice relative error.

    from ngspice_bridge import spice_array_matmul, check_ngspice_available
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import crossbar_array_test as cb


class NgspiceNotFoundError(RuntimeError):
    pass


class NgspiceRunError(RuntimeError):
    pass


def check_ngspice_available():
    """Returns the resolved ngspice binary path, or raises
    NgspiceNotFoundError with an actionable message if it's not installed."""
    path = shutil.which("ngspice")
    if path is None:
        raise NgspiceNotFoundError(
            "ngspice not found on PATH. Install it with:\n"
            "  sudo apt-get install -y ngspice\n"
            "(archive.ubuntu.com is reachable from this environment; this "
            "is the same package used to test this bridge.)"
        )
    return path


def _parse_op_wrdata(results_path, num_vectors):
    """Parses ngspice's `wrdata` output for a `.op` analysis. Format is a
    single line of (x, y) pairs, one pair per requested vector, e.g. for
    `wrdata out.txt v(blp0) v(bln0)`:
        <x0> <y0> <x1> <y1>
    where x is a placeholder sweep index (not meaningful for .op) and y is
    the actual DC node voltage. Returns the list of y-values in the order
    the vectors were requested."""
    with open(results_path) as f:
        vals = [float(x) for x in f.read().split()]
    if len(vals) != 2 * num_vectors:
        raise NgspiceRunError(
            f"Expected {2*num_vectors} values (x,y pairs for {num_vectors} "
            f"vectors) in {results_path}, got {len(vals)}. ngspice output "
            f"format may have changed, or the run errored silently -- "
            f"check the raw file."
        )
    return vals[1::2]  # every y (odd index)


def spice_array_matmul(W, activations, r_sense=20.0, vread_base=0.1, workdir=None, keep_files=False):
    """Real ngspice equivalent of crossbar_array_test.golden_array_matmul():
    builds the same fixed-resistor READ-ONLY netlist, actually runs
    `ngspice -b`, parses the resulting bitline voltages, and returns the
    same list of K differential currents (I+ - I-) in the same units.
    Raises NgspiceNotFoundError / NgspiceRunError on failure -- callers
    should decide whether to skip or hard-fail."""
    check_ngspice_available()

    M = len(activations)
    K = len(W[0])

    tmpdir = workdir or tempfile.mkdtemp(prefix="ngspice_run_")
    os.makedirs(tmpdir, exist_ok=True)
    netlist_path = os.path.join(tmpdir, "array_read.sp")
    results_path = os.path.join(tmpdir, "array_results.txt")

    netlist = cb.build_array_read_netlist(W, activations, r_sense=r_sense,
                                           vread_base=vread_base,
                                           results_file=results_path)
    with open(netlist_path, "w") as f:
        f.write(netlist)

    proc = subprocess.run(
        ["ngspice", "-b", netlist_path],
        cwd=tmpdir, capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0 or not os.path.exists(results_path):
        raise NgspiceRunError(
            f"ngspice failed (exit {proc.returncode}) on {netlist_path}\n"
            f"--- stdout ---\n{proc.stdout[-2000:]}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}"
        )

    y_vals = _parse_op_wrdata(results_path, num_vectors=2 * K)  # v(blp_k), v(bln_k) per column
    currents = []
    for k in range(K):
        v_blp = y_vals[2 * k]
        v_bln = y_vals[2 * k + 1]
        currents.append((v_blp - v_bln) / r_sense)

    if not keep_files and workdir is None:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return currents


def relative_error_pct(reference, test):
    num = sum(abs(t - r) for r, t in zip(reference, test))
    den = sum(abs(r) for r in reference)
    return 100.0 * num / den if den > 1e-15 else float("nan")


def self_test():
    """Small hand-checkable 2x2 case, run through both models, diffed."""
    print("Checking ngspice availability...")
    ngspice_path = check_ngspice_available()
    print(f"  found: {ngspice_path}\n")

    W = [[1.0, -0.5], [0.5, 1.0]]
    acts = [1.0, 0.7]
    r_sense, vread = 20.0, 0.1

    print(f"W={W}  activations={acts}  r_sense={r_sense}  vread={vread}\n")

    nodal = cb.golden_array_matmul(W, acts, r_sense, vread)
    print("Running actual ngspice...")
    spice = spice_array_matmul(W, acts, r_sense, vread)

    print(f"\n{'col':>4} {'nodal (A)':>16} {'ngspice (A)':>16} {'rel diff %':>12}")
    for k in range(len(nodal)):
        diff_pct = abs(spice[k] - nodal[k]) / abs(nodal[k]) * 100.0 if nodal[k] else float("nan")
        print(f"{k:4d} {nodal[k]:16.6e} {spice[k]:16.6e} {diff_pct:12.2e}")

    overall = relative_error_pct(nodal, spice)
    print(f"\nOverall relative error (nodal vs real ngspice): {overall:.2e} %")
    if overall < 1e-3:
        print("✓ Nodal-analysis golden model matches real ngspice DC solve "
              "(expected -- both solve the identical linear resistor network).")
    else:
        print("⚠ Larger-than-expected mismatch -- investigate before trusting "
              "the fast nodal model for further benchmarking.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="Run the built-in 2x2 sanity check")
    args = ap.parse_args()

    if args.self_test:
        try:
            self_test()
        except (NgspiceNotFoundError, NgspiceRunError) as e:
            print(f"FAILED: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Nothing to do -- pass --self-test, or import spice_array_matmul() "
              "from another script.", file=sys.stderr)
