#!/usr/bin/env python3
"""
Convert a $fdisplay digital-stimulus dump from the FP2 unpacker testbench
into LTspice-compatible PWL (piecewise-linear) voltage source files.

Input file format (one line per sample, whitespace separated):
    # comment lines starting with '#' are ignored
    <time> <Dir1> <Mag1_1> <Mag1_0> <Dir2> <Mag2_1> <Mag2_0>

<time> is assumed to be in the same units $time reports for your
`timescale (this testbench uses `timescale 1ns/1ps, and because Icarus
prints $time in units of the finest precision in the design, the raw
numbers in dac_stimulus.txt are picoseconds -- adjust --time-unit if your
own testbench's timescale differs).

For each signal column, writes a separate two-column PWL file
(<signal>.pwl) with logic 0 -> 0V and logic 1 -> --vhigh, with a finite
edge rate (--tr) instead of an instantaneous step, which is both more
physically realistic for a DAC output and avoids putting a literal
zero-time discontinuity in front of LTspice's transient solver.

Usage:
    python3 dac_to_pwl.py dac_stimulus.txt --vhigh 1.2 --tr 100e-12
"""
import argparse
import sys


def parse_stimulus(path, time_unit_seconds):
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = line.lstrip("#").split()
                continue
            parts = line.split()
            t = int(parts[0]) * time_unit_seconds
            bits = [int(b) for b in parts[1:]]
            rows.append((t, bits))
    if header is None:
        raise ValueError("No header line found (expected '# time_ns Dir1 Mag1_1 ...')")
    signal_names = header[1:]
    return signal_names, rows


def build_pwl(rows, col_index, vhigh, tr):
    """Return list of (time, voltage) points for one signal column,
    inserting a linear ramp of length tr at every transition instead of
    an instantaneous step."""
    points = []
    prev_val = None
    for t, bits in rows:
        val = bits[col_index]
        v = vhigh if val else 0.0
        if prev_val is None:
            points.append((t, v))
        elif val != prev_val:
            # hold previous level right up to the edge, then ramp
            prev_v = vhigh if prev_val else 0.0
            edge_start = max(t - tr, points[-1][0])
            points.append((edge_start, prev_v))
            points.append((t, v))
        else:
            # level unchanged: no new point needed, PWL holds last value,
            # but we add one anyway so the file is easy to eyeball/debug
            points.append((t, v))
        prev_val = val
    return points


def write_pwl_file(path, points):
    with open(path, "w") as f:
        f.write("* PWL data generated from Verilog $fdisplay stimulus dump\n")
        for t, v in points:
            f.write(f"{t:.6e} {v:.4f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stimulus_file", help="Path to dac_stimulus.txt from the Verilog testbench")
    ap.add_argument("--time-unit", type=float, default=1e-12,
                     help="Seconds per unit in the time column (default 1e-12 = ps, matching Icarus's default $time precision reporting)")
    ap.add_argument("--vhigh", type=float, default=1.2, help="Voltage representing logic 1 (default 1.2V)")
    ap.add_argument("--tr", type=float, default=100e-12, help="Edge transition time in seconds (default 100ps)")
    ap.add_argument("--outdir", default=".", help="Directory to write .pwl files into")
    args = ap.parse_args()

    signal_names, rows = parse_stimulus(args.stimulus_file, args.time_unit)
    print(f"Parsed {len(rows)} samples, signals: {signal_names}")

    for idx, name in enumerate(signal_names):
        points = build_pwl(rows, idx, args.vhigh, args.tr)
        out_path = f"{args.outdir}/{name}.pwl"
        write_pwl_file(out_path, points)
        print(f"  wrote {out_path} ({len(points)} points)")


if __name__ == "__main__":
    main()
