#!/usr/bin/env python3
"""
bn_baseline.py
================================================================================
BatchNorm recalibration as a competing baseline (TODO item N2).

WHY THIS EXPERIMENT IS REQUIRED
-------------------------------
BN recalibration is the closest prior art to the per-column loading-gain
correction: it is retraining-free, it absorbs the systematic column-current
attenuation caused by crossbar non-idealities, and the literature reports it
landing within 0.16% of floating point. A reviewer who knows that work will ask
why a new correction is needed.

The argued difference is that BN parameters are per OUTPUT CHANNEL and can
therefore only absorb a layer-wide average attenuation, whereas the loading
gain differs column to column because G_col differs column to column. Until
now that has been an argument rather than a measurement, and arguments lose to
measurements.

The prediction the argument makes, which this script tests:

    BN recalibration should partially recover accuracy, and should recover
    LESS as tile height grows, because column-to-column spread in G_col grows
    with M. Per-column calibration should be flat in M, since it is exact.

If BN recalibration matches per-column calibration at every M, the
contribution is much weaker and the paper must say so.

HOW BN RECALIBRATION IS IMPLEMENTED
-----------------------------------
Standard retraining-free procedure, no gradients anywhere:

  1. Convert the network to the raw, UNCORRECTED crossbar path.
  2. Reset every BatchNorm layer's running_mean and running_var.
  3. Put only the BN layers into training mode, so forward passes update
     those running statistics while every other layer stays in eval mode.
  4. Stream calibration images through the network. The BN layers now measure
     the attenuated activation statistics the crossbar actually produces.
  5. Return to eval mode and measure accuracy on the test set.

The affine weight/bias of each BN layer are left untouched: only the running
statistics are re-estimated, which is what makes this retraining-free.

Calibration images come from the TRAIN split, never the test split. Using test
images to fit the statistics and then reporting test accuracy would inflate
the baseline and make the comparison meaningless.

USAGE
-----
    python3 bn_baseline.py --self-test

    python3 bn_baseline.py --sweep 32,64,128,256 --data-dir ./data \\
        --max-images 0 --adc-bits 6 --out-csv bn_baseline.csv
================================================================================
"""
import argparse
import csv
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import analog_eval as ae


# =============================================================================
def train_loader(data_dir, n_images, batch, dataset="cifar10"):
    """Calibration images, drawn from the TRAIN split.

    Test images are never used to fit the BN statistics. Doing so would let
    the baseline see the evaluation set and would overstate it.
    """
    cls_name, mean, std, _ = ae.DATASETS[dataset]
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds = getattr(torchvision.datasets, cls_name)(
        root=data_dir, train=True, download=True, transform=tfm)
    if n_images and n_images < len(ds):
        # Stride rather than truncate. cifar_loader does the same for the test
        # split, for the reason given there: these sets are class-ordered in
        # places, so the first N images can cover only a few classes. BN
        # statistics fitted on a class-skewed sample would be measuring the
        # wrong distribution and would understate the baseline.
        idx = list(range(0, len(ds), max(len(ds) // n_images, 1)))[:n_images]
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                       num_workers=2)


def recalibrate_bn(model, loader, device, max_batches=None):
    """Re-estimate BN running statistics on the crossbar's own activations.

    Only the BN layers are switched to training mode. Leaving the whole model
    in train mode would also enable dropout and any other stochastic path,
    which would corrupt the statistics being measured.
    """
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    if not bns:
        raise SystemExit("no BatchNorm layers found -- BN recalibration does "
                         "not apply to this model")
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative average over all batches seen
        m.train()

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            model(x.to(device, non_blocking=True))

    for m in bns:
        m.eval()
    return len(bns)


# =============================================================================
def compare(ckpt, block, args, device):
    """Four numbers at one tile height.

    ideal      digital-exact FP2, the ceiling
    raw        uncorrected crossbar
    bn         uncorrected crossbar plus BN recalibration
    corrected  per-column loading-gain calibration (this work)
    """
    test = ae.cifar_loader(args.data_dir, args.max_images, args.batch_size,
                           args.dataset)[0]
    calib = train_loader(args.data_dir, args.calib_images, args.batch_size,
                         args.dataset)

    out = {}
    for label, mode in (("ideal", "ideal"), ("raw", "analog"),
                        ("corrected", "corrected")):
        m = ae.build_model(ckpt, args.num_classes, device)
        ae.convert_model(m, block, args.tile_k, args.r_sense, args.vread,
                         args.scale_mode, device,
                         skip_first_last=not args.no_skip_first_last,
                         mode=mode, verbose=False, adc_bits=args.adc_bits,
                         dtype=ae.DTYPE[args.dtype])
        out[label] = ae.top1(m, test, device)
        del m

    # BN recalibration applied to the RAW path -- the whole point is that it
    # is an alternative to the per-column correction, not an addition to it.
    m = ae.build_model(ckpt, args.num_classes, device)
    ae.convert_model(m, block, args.tile_k, args.r_sense, args.vread,
                     args.scale_mode, device,
                     skip_first_last=not args.no_skip_first_last,
                     mode="analog", verbose=False, adc_bits=args.adc_bits,
                     dtype=ae.DTYPE[args.dtype])
    n_bn = recalibrate_bn(m, calib, device, args.calib_batches)
    out["bn"] = ae.top1(m, test, device)
    del m

    out.update(block_size=block, n_bn_layers=n_bn,
               bn_gap=out["ideal"] - out["bn"],
               corrected_gap=out["ideal"] - out["corrected"],
               bn_recovers=out["bn"] - out["raw"])
    return out


# =============================================================================
def self_test():
    """Checks the recalibration mechanics without needing a checkpoint."""
    print("=" * 62)
    print("SELF-TEST")
    print("=" * 62)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4),
                        nn.ReLU()).eval()
    bn = net[1]
    bn.running_mean.fill_(99.0)          # deliberately wrong statistics
    bn.running_var.fill_(99.0)

    x = torch.randn(8, 3, 8, 8) * 3.0 + 5.0
    loader = [(x, None), (x * 1.1, None)]
    n = recalibrate_bn(net, loader, torch.device("cpu"))
    print(f"  [1] found and recalibrated {n} BN layer(s)")
    assert n == 1

    print(f"  [2] running_mean moved off the poisoned value: "
          f"{bn.running_mean.abs().max():.4f} (was 99.0)")
    assert bn.running_mean.abs().max() < 50.0, "statistics were not re-estimated"

    print(f"  [3] BN layers returned to eval mode: {not bn.training}")
    assert not bn.training

    # Affine parameters must be untouched -- this is what makes it
    # retraining-free rather than a one-layer fine-tune.
    assert torch.allclose(bn.weight, torch.ones_like(bn.weight))
    assert torch.allclose(bn.bias, torch.zeros_like(bn.bias))
    print("  [4] affine weight/bias untouched (retraining-free)")

    print("\n" + "=" * 62)
    print("SELF-TEST PASSED")
    print("=" * 62)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="BatchNorm recalibration baseline against per-column "
                    "loading-gain calibration.")
    ap.add_argument("--sweep", default="32,64,128,256")
    ap.add_argument("--ckpt-pattern", default="qat_b{B}.pth")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=0,
                    help="0 = full test set")
    ap.add_argument("--calib-images", type=int, default=2000,
                    help="Train-split images used to re-estimate BN statistics.")
    ap.add_argument("--calib-batches", type=int, default=0,
                    help="Cap on calibration batches; 0 = all.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--tile-k", type=int, default=16)
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--scale-mode", default="e8m0")
    ap.add_argument("--adc-bits", type=int, default=6)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--no-skip-first-last", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}   dataset: {args.dataset}")

    rows = []
    for b in [int(x) for x in args.sweep.split(",")]:
        ckpt = args.ckpt_pattern.format(B=b)
        print(f"\n=== B={b}, checkpoint {ckpt} ===", flush=True)
        r = compare(ckpt, b, args, device)
        rows.append(r)
        print(f"  ideal {r['ideal']:.2f}%   raw {r['raw']:.2f}%   "
              f"BN-recal {r['bn']:.2f}%   per-column {r['corrected']:.2f}%")

    print("\n" + "=" * 84)
    print("BN RECALIBRATION vs PER-COLUMN LOADING-GAIN CALIBRATION")
    print("=" * 84)
    print(f"{'B=M':>5}{'ideal':>9}{'raw':>9}{'BN-recal':>11}{'per-col':>10}"
          f"{'BN gap':>10}{'per-col gap':>14}")
    print("-" * 84)
    for r in rows:
        print(f"{r['block_size']:>5}{r['ideal']:>9.2f}{r['raw']:>9.2f}"
              f"{r['bn']:>11.2f}{r['corrected']:>10.2f}"
              f"{-r['bn_gap']:>+10.2f}{-r['corrected_gap']:>+14.2f}")
    print("-" * 84)

    # Derived from the data rather than asserted: a hardcoded conclusion
    # survives a change of sign in the numbers beneath it.
    worst_bn = max(r["bn_gap"] for r in rows)
    worst_col = max(r["corrected_gap"] for r in rows)
    print(f"Worst gap to ceiling:  BN recalibration {worst_bn:.2f} pts, "
          f"per-column {worst_col:.2f} pts.")

    tall = [r for r in rows if r["block_size"] >= 128]
    short = [r for r in rows if r["block_size"] < 128]
    if tall and short:
        gs = sum(r["bn_gap"] for r in short) / len(short)
        gt = sum(r["bn_gap"] for r in tall) / len(tall)
        print(f"BN gap averages {gs:.2f} pts at B<128 and {gt:.2f} pts at "
              f"B>=128.")
        if gt > gs + 0.5:
            print("BN recalibration degrades with tile height, as predicted: a")
            print("per-channel scalar cannot represent per-column variation in")
            print("G_col, and that variation grows with M. The per-column")
            print("correction is exact and flat in M. The distinction holds.")
        else:
            print("BN recalibration does NOT degrade appreciably with tile")
            print("height. The 'per-layer cannot capture per-column' argument")
            print("is not supported by this data and must be dropped. What")
            print("survives is exactness, freedom from calibration data, and")
            print("applicability to networks without BatchNorm.")

    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
