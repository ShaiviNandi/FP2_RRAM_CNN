#!/usr/bin/env python3
"""
recover_drift_csv.py
Rebuild drift.csv from logs/drift.log.

The drift sweep completed but the summary printer crashed before the CSV was
written (a guard tested for acc_ideal, which the drift rows also carry, so they
fell into the accuracy printer and hit a KeyError on acc_analog). The numbers
are all in the log; this parses them back out rather than spending another two
hours of GPU time.

    python3 recover_drift_csv.py logs/drift.log drift.csv
"""
import csv
import re
import sys

HEAD = re.compile(r"drift sweep:\s*(\S+),\s*B=(\d+),\s*(\d+)\s*images")
CEIL = re.compile(r"FP2 digital-exact ceiling:\s*([\d.]+)%")
ROW = re.compile(
    r"t=\s*(\S+)\s+raw\s+([\d.]+)%\s+stale-calib\s+([\d.]+)%\s+"
    r"recalibrated\s+([\d.]+)%")

# fmt_time is lossy (1mo, 1y), so map the labels back to the seconds the sweep
# was actually given. Keep in step with run_all.sh's stage_drift.
SECONDS = {"1s": 1.0, "1h": 3600.0, "1d": 86400.0,
           "1mo": 2592000.0, "1y": 31536000.0}


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "logs/drift.log"
    out = sys.argv[2] if len(sys.argv) > 2 else "drift.csv"

    rows, ckpt, block, ceiling, n_img = [], None, None, None, None
    with open(log) as f:
        for line in f:
            line = line.lstrip("| ").rstrip()
            m = HEAD.search(line)
            if m:
                ckpt, block, n_img = m.group(1), int(m.group(2)), int(m.group(3))
                continue
            m = CEIL.search(line)
            if m:
                ceiling = float(m.group(1))
                continue
            m = ROW.search(line)
            if m and block is not None:
                lab = m.group(1)
                rows.append(dict(
                    checkpoint=ckpt, block_size=block, n_images=n_img,
                    t_seconds=SECONDS.get(lab, ""), t_label=lab,
                    acc_ideal=ceiling, raw=float(m.group(2)),
                    stale=float(m.group(3)), recalib=float(m.group(4)),
                    stale_cost=round(float(m.group(3)) - ceiling, 4),
                    recalib_cost=round(float(m.group(4)) - ceiling, 4),
                    refresh_value=round(float(m.group(4)) - float(m.group(3)), 4),
                ))
    if not rows:
        raise SystemExit(f"no drift rows found in {log}")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"recovered {len(rows)} rows -> {out}")

    blocks = sorted({r["block_size"] for r in rows})
    print(f"\n{'B':>5}{'t':>7}{'raw':>9}{'stale':>9}{'recalib':>9}"
          f"{'ceiling':>9}{'stale cost':>12}{'refresh worth':>15}")
    print("-" * 76)
    for b in blocks:
        for r in [x for x in rows if x["block_size"] == b]:
            print(f"{b:>5}{r['t_label']:>7}{r['raw']:>9.2f}{r['stale']:>9.2f}"
                  f"{r['recalib']:>9.2f}{r['acc_ideal']:>9.2f}"
                  f"{r['stale_cost']:>+12.2f}{r['refresh_value']:>+15.2f}")


if __name__ == "__main__":
    main()
