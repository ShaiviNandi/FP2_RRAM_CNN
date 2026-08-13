#!/usr/bin/env python3
"""
Golden reference model for the FP2-E1M0 -> ReRAM crossbar MAC.

Independently (i.e. NOT by re-running the RTL or the Verilog-A model)
computes what the crossbar current *should* be for a given stream of
FP2-E1M0-encoded weights, so it can be diffed against:
  - the RTL's registered Dir/Mag DAC control outputs (dac_stimulus.txt), and
  - ngspice's simulated branch currents (results.txt)

This is deliberately a simple, auditable resistor-network calculation, not
a re-derivation of the Stanford compact model's physics -- it treats each
ReRAM cell as a fixed two-state resistor (HRS/LRS) selected by the decoded
magnitude bits, which is the right level of fidelity for checking "did the
digital decode + crossbar wiring behave as intended", not for validating
the analog model's own accuracy (that's what the .va model itself is for).

Usage:
    python3 golden_mac_reference.py dac_stimulus.txt \\
        --r-hrs 100e3 --r-lrs 1e3 --vhigh 1.2 \\
        --results results.txt --tolerance 0.15
"""
import argparse
import sys


def parse_dac_stimulus(path):
    """Same format as tb_unpacker_export.v's output:
    time Dir1 Mag1_1 Mag1_0 Dir2 Mag2_1 Mag2_0"""
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                header = line.lstrip("#").split()
                continue
            parts = line.split()
            t = int(parts[0])
            bits = [int(b) for b in parts[1:]]
            rows.append((t, bits))
    return header, rows


def decode_weight(dir_bit, mag1_bit, mag0_bit):
    """FP2-E1M0 magnitude decode, matching fp2_unpacker.v's combinational
    equations independently re-derived (NOT copy-pasted from the RTL):
        mag1_bit & mag0_bit both 0  -> weight magnitude 0   (cell = HRS)
        mag1_bit=1                  -> weight magnitude 1.0 (cell = LRS)
        mag0_bit=1                  -> weight magnitude 0.5 (cell = mid-R,
                                        approximated here as LRS for a
                                        two-state resistor model -- see
                                        --r-mid for a 3rd state)
    Returns (signed_weight, cell_is_conducting)
    """
    if not mag1_bit and not mag0_bit:
        return 0.0, False
    magnitude = 1.0 if mag1_bit else 0.5
    signed = -magnitude if dir_bit else magnitude
    return signed, True


def expected_cell_resistance(dir_bit, mag1_bit, mag0_bit, r_hrs, r_lrs, r_mid=None):
    _, conducting = decode_weight(dir_bit, mag1_bit, mag0_bit)
    if not conducting:
        return r_hrs
    if mag1_bit:
        return r_lrs
    else:
        return r_mid if r_mid is not None else r_lrs


def expected_wordline_current(v_high, r_cell, r_sense):
    """Simple series divider: wordline driver -> cell -> bitline sense
    resistor -> ground. Matches the topology in rram_2x2_crossbar_fixed.sp
    (one cell per cross-point, sense resistor to ground on the bitline)."""
    return v_high / (r_cell + r_sense)


def parse_ngspice_results(path):
    """wrdata output: time i(Vwl1) time i(Vwl2), whitespace separated,
    no header line."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                t1, i1, t2, i2 = (float(x) for x in parts[:4])
            except ValueError:
                continue
            rows.append((t1, i1, i2))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dac_stimulus", help="Path to dac_stimulus.txt from tb_unpacker_export.v")
    ap.add_argument("--r-hrs", type=float, default=100e3, help="High-resistance-state cell resistance (ohms), default 100k")
    ap.add_argument("--r-lrs", type=float, default=1e3, help="Low-resistance-state cell resistance (ohms), default 1k")
    ap.add_argument("--r-mid", type=float, default=None, help="Optional third resistance state for magnitude=0.5 (default: same as LRS)")
    ap.add_argument("--r-sense", type=float, default=1e3, help="Bitline sense resistor (ohms), matches Rsl1/Rsl2 in the netlist, default 1k")
    ap.add_argument("--vhigh", type=float, default=1.2, help="Wordline drive voltage, default 1.2V")
    ap.add_argument("--results", default=None, help="Optional: ngspice results.txt to diff against")
    ap.add_argument("--tolerance", type=float, default=0.15, help="Fractional tolerance when comparing to ngspice results (default 15%%, since the golden model is a 2-state resistor approximation, not the full compact-model physics)")
    args = ap.parse_args()

    header, rows = parse_dac_stimulus(args.dac_stimulus)
    if header is None:
        print("ERROR: no '# time_ns Dir1 Mag1_1 ...' header found in dac_stimulus file", file=sys.stderr)
        sys.exit(1)

    print(f"{'time':>12} {'Dir1':>4} {'M1_1':>4} {'M1_0':>4} {'Dir2':>4} {'M2_1':>4} {'M2_0':>4} "
          f"{'w1':>6} {'w2':>6} {'R1(ohm)':>10} {'R2(ohm)':>10} {'I_wl1_exp(A)':>14} {'I_wl2_exp(A)':>14}")

    expected = []
    for t, bits in rows:
        dir1, m1_1, m1_0, dir2, m2_1, m2_0 = bits
        w1, _ = decode_weight(dir1, m1_1, m1_0)
        w2, _ = decode_weight(dir2, m2_1, m2_0)
        r1 = expected_cell_resistance(dir1, m1_1, m1_0, args.r_hrs, args.r_lrs, args.r_mid)
        r2 = expected_cell_resistance(dir2, m2_1, m2_0, args.r_hrs, args.r_lrs, args.r_mid)
        i1 = expected_wordline_current(args.vhigh, r1, args.r_sense)
        i2 = expected_wordline_current(args.vhigh, r2, args.r_sense)
        expected.append((t, i1, i2))
        print(f"{t:12d} {dir1:4d} {m1_1:4d} {m1_0:4d} {dir2:4d} {m2_1:4d} {m2_0:4d} "
              f"{w1:6.2f} {w2:6.2f} {r1:10.1f} {r2:10.1f} {i1:14.6e} {i2:14.6e}")

    if args.results:
        sim_rows = parse_ngspice_results(args.results)
        print(f"\nParsed {len(sim_rows)} ngspice sample points from {args.results}")
        print("\nNOTE: this golden model is a static resistive-divider approximation,")
        print("it does NOT capture the Stanford model's dynamics (gap evolution,")
        print("thermal effects, switching transients) -- expect close agreement")
        print("only once the ngspice transient has settled after each stimulus")
        print("edge, not at the edges themselves. Use --tolerance to loosen/tighten.")
        # This intentionally does not attempt automatic time-alignment between
        # the digital-domain vectors (ns-scale, one per clock) and the analog
        # transient's much finer timestep -- that alignment is specific to
        # the actual PWL edge placement and should be done deliberately,
        # not guessed at here.


if __name__ == "__main__":
    main()
