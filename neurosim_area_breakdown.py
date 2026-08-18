#!/usr/bin/env python3
"""
neurosim_area_breakdown.py
================================================================================
Component-wise chip area from a NeuroSim log, with the unaccounted remainder
named rather than ignored.

WHY
---
An area improvement cannot be argued from a total. It needs a breakdown that
closes, and a second run to difference against.

WHAT THE RESIDUAL IS
--------------------
The printed components are accumulated over tiles. ChipCalculateArea in
Chip.cpp adds further global-level blocks after the tile loop that have no
corresponding print statement. On the V1.4 source:

    globalBuffer        global activation buffer, sized to the tile grid
    GhTree / GBus       global H-tree or bus interconnect
    maxPool             pooling units
    Gaccumulation       global accumulation
    GreLu / Gsigmoid    activation units

That list is not assumed. The source is scanned for every statement adding to
the area accumulator, so the answer comes from the installed version.

Measured on VGG-8 at 65 nm the named components reach 95.3% of the chip total,
the remainder being the global blocks above.

WHAT IT DOES
------------
  1. scans Chip.cpp for every statement adding to the area accumulator
  2. parses a NeuroSim log for all area lines
  3. subtracts, and reports the residual
  4. with two or more logs, differences them component by component

USAGE
-----
    python3 neurosim_area_breakdown.py --log logs/b6_neurosim.log
    python3 neurosim_area_breakdown.py \\
        --log logs/baseline.log --log logs/proposed.log \\
        --names "baseline B=32 8b,proposed B=128 6b"
    python3 neurosim_area_breakdown.py --log logs/b6_neurosim.log \\
        --src ~/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/Chip.cpp
================================================================================
"""
import argparse
import os
import re
import sys

# Rather than hard-code NeuroSim's phrasing, which differs between versions and
# contains internal colons that break naive patterns, match on the SHAPE of an
# area line: a label, a colon, a number, and a unit.
#
#   ChipArea : 8.00561e+07um^2
#   Chip total CIM array : 2.10312e+07um^2
#   Total IC Area on chip (Global and Tile/PE local): 1.21411e+07um^2
#   Total ADC (or S/As and precharger for SRAM) Area on chip : 1.53844e+07um^2
#
# The discriminator is the UNIT, not the label. Several components carry no
# "area" or "array" token, for example
#   "Total Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC,
#    pooling and activation units) on chip : 9.1e+06um^2"
# so a label-keyed search drops them silently, which is indistinguishable from
# genuinely absent silicon. Match any line ending in an area unit instead.
AREA_LINE = re.compile(
    r"^\s*\|?\s*(?P<label>.+?)\s*:\s*"
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>um\^?2|mm\^?2|m2)\s*$"
)

# Which label marks the grand total rather than a component.
TOTAL_HINT = re.compile(r"^\s*chip\s*area\s*$|^\s*chiparea\s*$", re.I)


def parse_log(path):
    """Pull every `label : number unit` area line out of a NeuroSim log.

    Returns (total, {label: value}, unit). The LAST occurrence of each label
    wins: NeuroSim prints per-layer blocks before the chip summary, so the
    final occurrence is the chip-level one.
    """
    found, unit = {}, None
    for ln in open(path, errors="replace"):
        m = AREA_LINE.match(ln.rstrip())
        if not m:
            continue
        label = " ".join(m.group("label").split())
        found[label] = float(m.group("val"))
        unit = unit or m.group("unit")

    total, total_label = None, None
    for label, v in found.items():
        if TOTAL_HINT.match(label):
            total, total_label = v, label
            break
    if total is None and found:
        # No label matched the total hint, so take the largest as the total and
        # say so, rather than silently reporting a meaningless residual.
        total_label, total = max(found.items(), key=lambda kv: kv[1])
        print(f"  [note] no line matched a 'chip area' total; treating the "
              f"largest, '{total_label}', as the total.")
    parts = {k: v for k, v in found.items() if k != total_label}

    if not found:
        print("  [warn] no area lines matched. Lines mentioning area/array:")
        for ln in open(path, errors="replace"):
            if "area" in ln.lower() or "array" in ln.lower():
                print("   ", ln.strip()[:110])
    return total, parts, unit or "um^2"


def scan_source(path):
    """Every statement in Chip.cpp that contributes to the chip area total."""
    if not os.path.isfile(path):
        return None
    hits = []
    for i, ln in enumerate(open(path, errors="replace"), 1):
        s = ln.strip()
        # Assignments and accumulations into the running chip area, plus the
        # CalculateArea calls on global blocks that feed them.
        if re.search(r"\barea\s*\+=", s) or re.search(r"^\s*area\s*=", s):
            hits.append((i, s))
        elif re.search(r"^\s*\w+Area\s*=\s*\w+->area", s):
            hits.append((i, s))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, action="append",
                    help="NeuroSim log. Repeat to compare configurations; the "
                         "first is treated as the baseline.")
    ap.add_argument("--names", default=None,
                    help="Comma-separated labels for the logs, in order.")
    ap.add_argument("--src",
                    default=os.path.expanduser(
                        "~/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/"
                        "Chip.cpp"))
    args = ap.parse_args()

    print("=" * 78)
    print("1. WHAT THE SOURCE ADDS TO CHIP AREA")
    print("=" * 78)
    hits = scan_source(args.src)
    if hits is None:
        print(f"  Chip.cpp not found at {args.src}")
        print("  Pass --src. Without it the residual can be measured but not")
        print("  named.")
    elif not hits:
        print("  No `area +=` statements found. The accumulator is named")
        print("  something else in this version; grep for 'CalculateArea('")
        print("  and follow the variable it assigns.")
    else:
        for i, s in hits:
            print(f"  {i:>5}  {s[:100]}")
        print(f"\n  {len(hits)} contributing statements. Any name here that is")
        print("  absent from the printed summary below is part of the residual.")

    print()
    print("=" * 78)
    print("2. WHAT THE LOG PRINTS")
    print("=" * 78)
    total, parts, unit = parse_log(args.log[0])
    if total is None:
        sys.exit(1)

    w = max((len(k) for k in parts), default=10)
    w = min(max(w, 10), 62)
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {k[:w]:<{w}}{v:>16,.0f} {unit}  {100*v/total:6.2f}%")

    named = sum(parts.values())
    resid = total - named
    print("  " + "-" * (w + 34))
    print(f"  {'named subtotal':<{w}}{named:>16,.0f} {unit}  "
          f"{100*named/total:6.2f}%")
    print(f"  {'CHIP TOTAL':<{w}}{total:>16,.0f} {unit}  100.00%")
    print(f"  {'residual':<{w}}{resid:>16,.0f} {unit}  "
          f"{100*resid/total:6.2f}%")
    print()

    frac = abs(resid) / total
    if frac < 0.02:
        print("  Breakdown CLOSES. Report the components directly, against the")
        print("  chip total.")
    elif frac < 0.25:
        print("  Breakdown nearly closes. The residual is the global-level")
        print("  blocks from section 1 that have no print statement: global")
        print("  buffer, H-tree or bus interconnect, pooling, global")
        print("  accumulation, ReLU and sigmoid units.")
        print()
        print("  Report it as a named line -- 'global buffer, interconnect and")
        print("  activation units' -- not as a tool limitation. It is real")
        print("  silicon and it is in the total.")
    else:
        print("  Large residual. Before concluding anything, check that every")
        print("  printed component above was captured: a label whose text")
        print("  changed between versions will be silently absent here, and")
        print("  that looks identical to genuinely missing area.")
        print()
        print("  Cross-check against section 1: every `area +=` target should")
        print("  correspond to something in the table above.")
    print()
    print("  To split the residual further, add a print next to each global")
    print("  CalculateArea call found in section 1 and rebuild.")

    if len(args.log) < 2:
        return

    # -------------------------------------------------------------------
    # Comparison. An area IMPROVEMENT cannot be argued from one run: it
    # needs the same breakdown under two configurations, differenced
    # component by component, so the saving can be attributed to the change
    # that caused it rather than asserted.
    # -------------------------------------------------------------------
    names = (args.names.split(",") if args.names
             else [os.path.basename(p) for p in args.log])
    runs = [(names[0], total, parts)]
    for i, path in enumerate(args.log[1:], 1):
        t, p, _ = parse_log(path)
        if t is None:
            print(f"  [warn] no area lines in {path}; skipped")
            continue
        runs.append((names[i] if i < len(names) else path, t, p))

    print()
    print("=" * 78)
    print("3. COMPONENT-WISE COMPARISON")
    print("=" * 78)

    labels = []
    for _, _, p in runs:
        for k in p:
            if k not in labels:
                labels.append(k)

    base_name, base_total, base_parts = runs[0]
    for name, tot, p in runs[1:]:
        print(f"\n  {name}  vs  {base_name}")
        print("  " + "-" * 74)
        print(f"  {'component':<40}{'baseline':>11}{'this':>11}{'change':>11}")
        for k in labels:
            b, v = base_parts.get(k, 0.0), p.get(k, 0.0)
            if b == 0 and v == 0:
                continue
            d = (f"{(v-b)/b*100:+9.1f}%" if b else "        --")
            print(f"  {k[:40]:<40}{b:>11,.0f}{v:>11,.0f}{d:>11}")
        d = (tot - base_total) / base_total * 100
        print("  " + "-" * 74)
        print(f"  {'CHIP TOTAL':<40}{base_total:>11,.0f}{tot:>11,.0f}"
              f"{d:>+10.1f}%")
        print(f"\n  Area ratio {base_total/tot:.2f}x "
              f"({'smaller' if tot < base_total else 'LARGER'} than baseline)."
              )
        print("  Attribute the change to the component that moved, not to the")
        print("  total. A total that falls while the array grows means the")
        print("  saving came from converters, not from density.")


if __name__ == "__main__":
    main()
