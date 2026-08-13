#!/usr/bin/env python3
"""
benchmark_resnet18.py
================================================================================
Layer-by-layer benchmarking harness for the FP2-E1M0 / 2T2R ReRAM crossbar
accelerator running ResNet-18 inference. Compares four fidelity levels:

    1. FP32 baseline        -- exact float32 conv, as PyTorch computes it
    2. FP2 digital           -- weights quantized to FP2-E1M0 {-1,-.5,0,.5,1},
                                 dot product computed exactly in software
                                 (isolates pure quantization error, zero
                                 circuit non-idealities)
    3. 2T2R analog crossbar  -- FP2 weights mapped onto TILE_M x TILE_K tiles
                                 and run through the resistive-divider nodal
                                 model in crossbar_array_test.py (this repo),
                                 including shared-Rsense bitline loading
                                 (isolates crossbar-induced error on top of
                                 quantization error)
    4. RTL simulation        -- optional: diffed against a psum dump produced
                                 by an actual RTL testbench run, if provided

For each Conv2d layer in the model, a calibration forward pass captures the
input-activation statistics via forward hooks + `F.unfold` (im2col), so the
crossbar error numbers reflect what that layer actually sees in inference.
Pass --calib-dataset cifar10 (or imagefolder) to drive that pass with REAL
IMAGES. The default, --calib-dataset random, drives it with Gaussian noise
and is retained only to reproduce earlier runs; it warns, and its numbers
should not be reported -- see build_calibration_batch() for why.

Because exhaustively running every spatial output position of every layer
through the resistive-divider model is unnecessary for a fidelity benchmark
(and expensive), a random subset of `--max-positions` output pixels per
layer is sampled -- controlled and reproducible via `--seed`.

USAGE
-----
    # Pipeline smoke test with random weights/activations (no torch needed
    # beyond quantization math -- use --synthetic if torch/torchvision are
    # not installed in this environment):
    python3 benchmark_resnet18.py --synthetic --max-positions 32

    # Real ResNet-18 (random-init, since no internet access to fetch
    # pretrained weights in this environment -- point --checkpoint at a
    # local .pth file for a trained model):
    python3 benchmark_resnet18.py --checkpoint my_resnet18.pth \\
        --max-positions 64 --r-sense 20.0 --vread 0.1 --seeds 5

    # Include RTL simulation diff (expects a JSON dump keyed by layer name,
    # see `parse_rtl_log` docstring for the expected format):
    python3 benchmark_resnet18.py --checkpoint my_resnet18.pth \\
        --rtl-log rtl_psum_dump.json

    # Include end-to-end Top-1 accuracy drop on a local image folder
    # (torchvision ImageFolder layout):
    python3 benchmark_resnet18.py --checkpoint my_resnet18.pth \\
        --eval-dir /path/to/val_subset --max-eval-images 200
================================================================================
"""
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchvision
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# Reuse the existing golden resistive-divider crossbar model from this repo
# instead of re-deriving the nodal analysis here -- keeps the RTL-side
# validation (crossbar_cli.py, golden_mac_reference.py) and this ResNet-level
# benchmark numerically consistent with each other.
import crossbar_array_test as cb

try:
    import ngspice_bridge
    NGSPICE_BRIDGE_AVAILABLE = True
except ImportError:
    NGSPICE_BRIDGE_AVAILABLE = False

TILE_M_DEFAULT = 32
TILE_K_DEFAULT = 16


# =============================================================================
# Layer extraction
# =============================================================================
@dataclass
class ConvLayerInfo:
    name: str
    weight: "np.ndarray"      # [Cout, Cin, Kh, Kw], float32
    stride: int
    padding: int
    in_shape: Optional[tuple] = None   # captured [N,Cin,H,W] at calibration time
    out_shape: Optional[tuple] = None
    unfolded_input: Optional["np.ndarray"] = None  # [Cin*Kh*Kw, num_positions] sample


def build_model(args):
    """Returns (model, is_synthetic). Falls back to a synthetic layer list
    if torch/torchvision aren't available, since pretrained-weight download
    (download.pytorch.org) isn't reachable from a network-restricted
    environment -- pass --checkpoint for a real trained model."""
    if args.synthetic or not (TORCH_AVAILABLE and TORCHVISION_AVAILABLE):
        if not args.synthetic:
            print("WARNING: torch/torchvision unavailable -- falling back to "
                  "--synthetic mode (a small hand-built conv stack standing "
                  "in for ResNet-18's layer shapes).", file=sys.stderr)
        return None, True

    num_classes = getattr(args, "num_classes", 1000)
    model = tv_models.resnet18(weights=None, num_classes=num_classes)  # no internet fetch
    if getattr(args, "cifar_arch", False):
        # CIFAR ResNet-18 stem: 3x3 stride-1 conv, no maxpool. Must match the
        # arch a qat_finetune_fp2.py checkpoint was trained with, or conv1/fc
        # silently fail to load (strict=False) and you benchmark random weights.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        # strict=False forgives MISSING and UNEXPECTED keys but still raises on
        # a SHAPE mismatch, which is the common case here: a CIFAR checkpoint's
        # 10-way head against this model's default 1000-way one. Rather than
        # let torch abort with a stack trace, drop the offending tensors and
        # name the flag that fixes it.
        own = model.state_dict()
        bad = {k: (tuple(v.shape), tuple(own[k].shape)) for k, v in state.items()
               if k in own and hasattr(v, "shape") and v.shape != own[k].shape}
        if bad:
            print(f"WARNING: {len(bad)} tensor(s) in {args.checkpoint} have a "
                  f"different shape than this model and were DROPPED, leaving "
                  f"them randomly initialised:", file=sys.stderr)
            for k, (a, b) in list(bad.items())[:6]:
                print(f"         {k}: checkpoint {a} vs model {b}", file=sys.stderr)
            if any(k.startswith("fc.") for k in bad):
                n = state["fc.weight"].shape[0] if "fc.weight" in state else "N"
                print(f"         The classifier head differs -- pass "
                      f"--num-classes {n} to match the checkpoint. (Harmless if "
                      f"you are only benchmarking conv layers, since fc is never "
                      f"read, but every accuracy number would be meaningless.)",
                      file=sys.stderr)
            state = {k: v for k, v in state.items() if k not in bad}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"WARNING: checkpoint only partially matched the model "
                  f"({len(missing)} missing, {len(unexpected)} unexpected keys). "
                  f"Un-matched layers keep their RANDOM init, which silently "
                  f"corrupts every number below. First few: "
                  f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}. "
                  f"If conv1/fc are listed, pass --cifar-arch and/or --num-classes "
                  f"to match how the checkpoint was trained.", file=sys.stderr)
    else:
        print("NOTE: no --checkpoint given; using randomly-initialized "
              "ResNet-18 weights. Layer-fidelity numbers (SNR, rel-error) "
              "are still meaningful; Top-1 accuracy numbers are NOT "
              "(random weights => random predictions).", file=sys.stderr)
    model.eval()
    return model, False


SYNTHETIC_LAYER_SHAPES = [
    # (name, Cout, Cin, Kh, Kw, stride, pad, in_H, in_W)  -- mimics
    # ResNet-18's stem + a couple of representative stage layers so the
    # pipeline can be exercised without torch/torchvision installed.
    ("stem_conv1",      64,   3, 7, 7, 2, 3, 224, 224),
    ("layer1.0.conv1",  64,  64, 3, 3, 1, 1,  56,  56),
    ("layer2.0.conv1", 128,  64, 3, 3, 2, 1,  56,  56),
    ("layer3.0.conv1", 256, 128, 3, 3, 2, 1,  28,  28),
    ("layer4.0.conv1", 512, 256, 3, 3, 2, 1,  14,  14),
    ("layer4.0.downsample.0", 512, 256, 1, 1, 2, 0, 14, 14),
]


CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_calibration_batch(args, res):
    """Returns the [N,3,res,res] tensor used to drive the calibration forward
    pass whose per-layer input activations get captured by the hooks.

    THIS MATTERS MORE THAN IT LOOKS. Every SNR / relative-error number this
    script reports is E[(W x - Wq x)^2] over the x captured here. Feed the
    network Gaussian noise and you are measuring quantization error against
    activations no real inference ever produces: after BN and ReLU, a noise
    image gives feature maps with roughly symmetric, dense, low-kurtosis
    channel statistics, whereas real images give sparse, heavy-tailed,
    spatially-correlated ones with a large fraction of exact zeros post-ReLU.
    Those are different distributions and they do not yield the same MAC
    error. Earlier revisions of this file ran `torch.randn(1,3,res,res)`
    while the module docstring claimed "REAL input-activation statistics" --
    that was wrong, and --calib-dataset is the fix. `random` is retained
    only so the old numbers stay reproducible, and it warns."""
    if args.calib_dataset == "random":
        print("WARNING: --calib-dataset random feeds the network Gaussian NOISE, not "
              "images. Every SNR/RelErr number below is then measured against "
              "activations that no real inference produces. Use --calib-dataset "
              "cifar10 (with --calib-dir) for numbers you can put in a report.",
              file=sys.stderr)
        return torch.randn(args.calib_images, 3, res, res)

    if not TORCHVISION_AVAILABLE:
        raise SystemExit("--calib-dataset needs torchvision installed.")

    if args.calib_dataset in ("cifar10", "cifar100"):
        mean, std = ((CIFAR10_MEAN, CIFAR10_STD) if args.calib_dataset == "cifar10"
                     else (CIFAR100_MEAN, CIFAR100_STD))
        tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        ds_cls = (torchvision.datasets.CIFAR10 if args.calib_dataset == "cifar10"
                  else torchvision.datasets.CIFAR100)
        try:
            ds = ds_cls(args.calib_dir, train=False, transform=tfm, download=False)
        except Exception as e:
            raise SystemExit(
                f"Could not load {args.calib_dataset} from --calib-dir {args.calib_dir!r}: {e}\n"
                f"Point --calib-dir at the directory qat_finetune_fp2.py downloaded into "
                f"(the one containing cifar-10-batches-py/).")
        if res != 32:
            raise SystemExit(f"--calib-dataset {args.calib_dataset} is 32x32 but this arch "
                             f"expects {res}x{res}. Did you forget --cifar-arch?")
    else:  # imagefolder
        tfm = T.Compose([T.Resize(int(res * 1.14)), T.CenterCrop(res),
                         T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        if not args.calib_dir:
            raise SystemExit("--calib-dataset imagefolder requires --calib-dir")
        ds = torchvision.datasets.ImageFolder(args.calib_dir, transform=tfm)

    n = min(len(ds), args.calib_images)
    # Stride through the set rather than taking the first N, which in a
    # class-sorted ImageFolder would be N images of a single class.
    step = max(len(ds) // n, 1)
    idx = list(range(0, len(ds), step))[:n]
    batch = torch.stack([ds[i][0] for i in idx])
    print(f"Calibration: {n} real images from {args.calib_dataset} "
          f"({tuple(batch.shape)}), activation stats captured from these.")
    return batch


def extract_conv_layers(model, args) -> list:
    """Returns a list of ConvLayerInfo with real captured activations
    (via forward hooks + unfold) when torch is available, or synthetic
    random data when in --synthetic mode."""
    layers = []

    if model is None:
        rng = np.random.default_rng(args.seed)
        for (name, cout, cin, kh, kw, stride, pad, ih, iw) in SYNTHETIC_LAYER_SHAPES:
            w = rng.standard_normal((cout, cin, kh, kw)).astype(np.float32) * 0.05
            oh = (ih + 2 * pad - kh) // stride + 1
            ow = (iw + 2 * pad - kw) // stride + 1
            n_pos = min(args.max_positions, oh * ow)
            patch_dim = cin * kh * kw
            unfolded = rng.standard_normal((patch_dim, n_pos)).astype(np.float32) * 0.3
            layers.append(ConvLayerInfo(
                name=name, weight=w, stride=stride, padding=pad,
                in_shape=(1, cin, ih, iw), out_shape=(1, cout, oh, ow),
                unfolded_input=unfolded,
            ))
        return layers

    # ---- real torch model path -------------------------------------------
    captured = {}

    def make_hook(name, mod):
        def hook(module, inputs, output):
            captured[name] = inputs[0].detach()
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(make_hook(name, mod)))

    # Input resolution must match the arch: a CIFAR-stem ResNet-18 fed 224x224
    # would produce 56x56 feature maps into layer4 and the captured activation
    # statistics would not be the ones the network actually sees at inference.
    res = 32 if getattr(args, "cifar_arch", False) else 224
    calib = build_calibration_batch(args, res)
    with torch.no_grad():
        model(calib)
    for h in handles:
        h.remove()

    rng = np.random.default_rng(args.seed)
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        if name not in captured:
            continue
        x = captured[name]
        kh, kw = mod.kernel_size
        stride = mod.stride[0]
        pad = mod.padding[0]

        # [N, Cin*Kh*Kw, L] -> [Cin*Kh*Kw, N*L]: pool the sliding-window
        # positions of EVERY calibration image into one column pool, so the
        # sampled positions span the whole batch instead of coming from a
        # single image (which would make the error estimate hostage to one
        # picture's content).
        unfolded_full = F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=pad)
        unfolded_full = unfolded_full.permute(1, 0, 2).reshape(unfolded_full.shape[1], -1).numpy()

        n_pos = unfolded_full.shape[1]
        n_sample = min(args.max_positions, n_pos)
        idx = rng.choice(n_pos, size=n_sample, replace=False)
        unfolded_sample = unfolded_full[:, idx]

        oh = (x.shape[2] + 2 * pad - kh) // stride + 1
        ow = (x.shape[3] + 2 * pad - kw) // stride + 1

        layers.append(ConvLayerInfo(
            name=name, weight=mod.weight.detach().numpy(),
            stride=stride, padding=pad,
            in_shape=tuple(x.shape), out_shape=(1, mod.out_channels, oh, ow),
            unfolded_input=unfolded_sample,
        ))

    return layers


# =============================================================================
# Tiling + per-fidelity forward passes
# =============================================================================
def weight_to_MK(weight: "np.ndarray") -> "np.ndarray":
    """[Cout, Cin, Kh, Kw] -> [M=Cin*Kh*Kw, K=Cout], matching the row=input /
    col=output convention used throughout crossbar_array_test.py /
    crossbar_cli.py (`digital_exact[k] = sum_i activations[i] * W[i][k]`)."""
    cout = weight.shape[0]
    flat = weight.reshape(cout, -1).T  # [Cin*Kh*Kw, Cout]
    return np.ascontiguousarray(flat)


def tile_ranges(total, tile):
    out = []
    start = 0
    while start < total:
        end = min(start + tile, total)
        out.append((start, end))
        start = end
    return out


def fp32_forward(W_MK, activations_MK):
    """Exact float32 dot product, per output position. activations_MK:
    [M, num_positions]. Returns [K, num_positions]."""
    return W_MK.T @ activations_MK


def quantize_shared_scale(scale, mode="e8m0", eps=1e-8):
    """Quantize the per-block shared scale to 8 bits, matching
    qat_finetune_fp2.quantize_scale exactly.

    The paper's scale is an "8-bit shared scale", not an FP32 one. Leaving it
    in FP32 (mode="none") makes every metric here optimistic: an FP32 scale
    sits exactly at the block max, whereas E8M0 rounds the exponent UP, so
    w/scale is systematically SMALLER and more weights fall below the 0.25
    threshold that separates the 0 level from the +-0.5 level. Concretely
    that moves ResNet-18 cell utilization by ~14 points (56% -> 42%), which
    is the difference between the number this benchmark used to report and
    the number qat_finetune_fp2.py reports for the same weights. Both were
    "right" for their own assumption; only e8m0 is right for the paper's
    hardware. Default here is "e8m0" for that reason -- pass
    --scale-mode none to reproduce the older, FP32-scale numbers."""
    if mode == "none":
        return scale
    exp = np.clip(np.ceil(np.log2(np.maximum(scale, eps))), -128.0, 127.0)
    pow2 = np.power(2.0, exp)
    if mode == "e8m0":
        return pow2
    if mode == "fp8":
        mant = np.ceil((scale / pow2) * 128.0) / 128.0
        return np.maximum(pow2 * mant, eps)
    raise ValueError(f"unknown scale mode {mode!r}")


def weight_scale_factor(W_MK, block_size=32, eps=1e-8, scale_mode="e8m0"):
    """Per-BLOCK max-abs scale, matching the FP2 paper's actual block
    granularity (Dang et al., "FP2: A 2-bit Floating-Point Format for
    Edge-AI Inference and Fine-Tuning", TCAS-I 2026): "32 original
    floating-point numbers are grouped into a block, with each block
    sharing an 8-bit shared scale X." Blocks are 32 CONSECUTIVE elements
    along the reduction (M / input-channel) axis, for a FIXED output
    channel -- i.e. exactly the (m0:m1, k) slices this benchmark's crossbar
    tiling already visits, since block_size defaults to the same 32 as
    TILE_M. This is coarser-than-per-element but much finer than the
    previous per-whole-channel scale (which could span thousands of
    elements and get dominated by a single outlier weight). Returns an
    array of shape [n_blocks, K] using the SAME block boundaries as
    tile_ranges(M, block_size), so callers can index it directly against
    the M-tiling loop in analog_2t2r_forward.

    NOTE ON PHYSICAL VALIDITY: block_size must equal (or be an exact
    divisor consistent with) the M-tiling step used when actually running
    the crossbar model. This benchmark rescales the crossbar's OUTPUT
    current by `scale` after each single-tile golden_array_matmul() call
    -- if a physical crossbar tile mixed rows from two different
    quantization blocks with different scales, output-side rescaling
    could no longer separate their contributions correctly. Keeping
    block_size == tile_m (the default) sidesteps this entirely."""
    M, K = W_MK.shape
    m_ranges = tile_ranges(M, block_size)
    scale = np.zeros((len(m_ranges), K))
    for i, (m0, m1) in enumerate(m_ranges):
        scale[i] = np.maximum(np.max(np.abs(W_MK[m0:m1, :]), axis=0), eps)
    return quantize_shared_scale(scale, scale_mode, eps)


def fp2_digital_forward(W_MK, activations_MK, scale, block_size=32):
    """FP2-E1M0 quantized weights, exact (noiseless) dot product -- isolates
    pure quantization error from crossbar circuit error. `scale` is the
    per-block array [n_blocks, K] from weight_scale_factor(); each block's
    contribution is quantized and rescaled independently, then summed --
    NOT a single global rescale, since different blocks can have very
    different scales (that's the whole point of block floating point)."""
    M, K = W_MK.shape
    m_ranges = tile_ranges(M, block_size)
    Wq = np.zeros_like(W_MK)
    out = np.zeros((K, activations_MK.shape[1]))
    for i, (m0, m1) in enumerate(m_ranges):
        Wq[m0:m1, :] = np.vectorize(cb.quantize_to_fp2)(W_MK[m0:m1, :] / scale[i][None, :])
        out += (Wq[m0:m1, :].T @ activations_MK[m0:m1, :]) * scale[i][:, None]
    return out, Wq


def analog_2t2r_forward(W_MK, activations_MK, tile_m, tile_k, r_sense, vread, scale, n_seeds_noise=1):
    """Runs the FP2-quantized weight matrix through the actual resistive-
    divider 2T2R golden model (cb.golden_array_matmul), tiled to (tile_m,
    tile_k), with results recovered into the same units as the digital dot
    product (matching crossbar_cli.py's recovery scaling) and summed across
    M-tiles (spatial reduction) / concatenated across K-tiles (output
    channel tiling). `scale` is the per-block array [n_blocks, K] from
    weight_scale_factor() -- REQUIRES tile_m == the block_size used to
    compute `scale`, since each M-tile here is treated as exactly one
    quantization block (see weight_scale_factor's docstring). Returns
    (result[K,num_positions], cell_utilization)."""
    M, K = W_MK.shape
    num_pos = activations_MK.shape[1]
    g_lrs = 1.0 / cb.col.R_FOR_MAGNITUDE[1.0]
    scale = np.asarray(scale)

    m_ranges = tile_ranges(M, tile_m)
    k_ranges = tile_ranges(K, tile_k)
    assert scale.shape[0] == len(m_ranges), (
        f"scale has {scale.shape[0]} blocks but tiling produces {len(m_ranges)} "
        f"M-tiles -- weight_scale_factor's block_size must match tile_m."
    )

    Wq = np.zeros_like(W_MK)
    for i, (m0, m1) in enumerate(m_ranges):
        Wq[m0:m1, :] = np.vectorize(cb.quantize_to_fp2)(W_MK[m0:m1, :] / scale[i][None, :])

    result = np.zeros((K, num_pos), dtype=np.float64)
    total_cells = 0
    active_cells = 0

    for (k0, k1) in k_ranges:
        for i, (m0, m1) in enumerate(m_ranges):
            scale_tile = scale[i, k0:k1]
            W_tile = Wq[m0:m1, k0:k1]
            total_cells += 2 * W_tile.shape[0] * W_tile.shape[1]  # 2T2R = 2 cells/synapse
            active_cells += int(np.count_nonzero(W_tile)) * 2

            W_tile_list = W_tile.tolist()
            for p in range(num_pos):
                act_tile = activations_MK[m0:m1, p].tolist()
                golden = cb.golden_array_matmul(W_tile_list, act_tile, r_sense, vread)
                recovered = [g / (vread * g_lrs) for g in golden]
                result[k0:k1, p] += np.asarray(recovered) * scale_tile

    cell_utilization = active_cells / total_cells if total_cells else float("nan")
    return result, cell_utilization


# =============================================================================
# Metrics
# =============================================================================
def snr_db(reference: "np.ndarray", test: "np.ndarray") -> float:
    err = test - reference
    sig_power = float(np.mean(reference ** 2))
    err_power = float(np.mean(err ** 2))
    if err_power <= 1e-20:
        return float("inf")
    if sig_power <= 1e-20:
        return float("nan")
    return 10.0 * math.log10(sig_power / err_power)


def relative_error_pct(reference: "np.ndarray", test: "np.ndarray") -> float:
    num = float(np.sum(np.abs(test - reference)))
    den = float(np.sum(np.abs(reference)))
    return 100.0 * num / den if den > 1e-12 else float("nan")


# =============================================================================
# Optional RTL log diff
# =============================================================================
def parse_rtl_log(path):
    """Expected JSON format, produced by an RTL testbench $fwrite/$fdisplay
    post-processing step (not included here -- RTL emits raw psum ints,
    a small converter script should scale them to float and dump this):

        {
          "layer1.0.conv1": [[v_k0_p0, v_k0_p1, ...], [v_k1_p0, ...], ...],
          ...
        }

    Shape per layer must match [K_sampled, num_positions] from that layer's
    analog benchmark run (same --max-positions and --seed must have been
    used when the RTL testbench was driven, so positions line up)."""
    with open(path) as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}


# =============================================================================
# Real ngspice spot-check validation
# =============================================================================
def validate_against_ngspice(layers, args, rng):
    """Actually runs a handful of (layer, tile, position) samples through
    real ngspice (via ngspice_bridge.spice_array_matmul) and diffs against
    the fast nodal-analysis golden model (cb.golden_array_matmul) used for
    the main per-layer sweep. This is a SPOT CHECK, not a replacement for
    the main sweep -- each ngspice subprocess call costs ~10-50ms, so
    running it for every tile/position across all layers would be far too
    slow. What this validates: that golden_array_matmul's closed-form
    nodal solve is actually solving the same resistor network ngspice
    would solve for the same weights/activations -- i.e. that the fast
    model used everywhere else in this benchmark is trustworthy. It does
    NOT validate against the Stanford Verilog-A compact model's device
    physics (transient switching, gap evolution, thermal effects) -- both
    this and the nodal model use fixed two-state (HRS/LRS) resistances.

    Returns a list of per-sample dicts, and prints a summary table."""
    if not NGSPICE_BRIDGE_AVAILABLE:
        print("\n[ngspice validation] ngspice_bridge.py not found next to this "
              "script -- skipping. Copy it alongside benchmark_resnet18.py to "
              "enable --validate-ngspice.", file=sys.stderr)
        return []

    try:
        ngspice_bridge.check_ngspice_available()
    except ngspice_bridge.NgspiceNotFoundError as e:
        print(f"\n[ngspice validation] {e}", file=sys.stderr)
        return []

    n_samples = args.validate_ngspice
    print(f"\n=== Real ngspice spot-check ({n_samples} samples) ===")
    print(f"{'layer':<22} {'tile(m0:m1,k0:k1)':<20} {'pos':>4} "
          f"{'nodal(A)':>14} {'ngspice(A)':>14} {'rel_err%':>10}")
    print("-" * 90)

    results = []
    attempts = 0
    while len(results) < n_samples and attempts < n_samples * 5:
        attempts += 1
        layer = layers[rng.integers(0, len(layers))]
        W_MK = weight_to_MK(layer.weight)
        acts = layer.unfolded_input
        M, K = W_MK.shape
        if M == 0 or K == 0 or acts.shape[1] == 0:
            continue

        scale = weight_scale_factor(W_MK, block_size=args.tile_m, scale_mode=args.scale_mode)
        m_ranges = tile_ranges(M, args.tile_m)
        k_ranges = tile_ranges(K, args.tile_k)

        block_idx = int(rng.integers(0, len(m_ranges)))
        k_idx = int(rng.integers(0, len(k_ranges)))
        m0, m1 = m_ranges[block_idx]
        k0, k1 = k_ranges[k_idx]
        p = int(rng.integers(0, acts.shape[1]))

        Wq_tile = np.vectorize(cb.quantize_to_fp2)(
            W_MK[m0:m1, k0:k1] / scale[block_idx, k0:k1][None, :])
        W_tile = Wq_tile.tolist()
        act_tile = acts[m0:m1, p].tolist()

        nodal = cb.golden_array_matmul(W_tile, act_tile, args.r_sense, args.vread)
        try:
            spice = ngspice_bridge.spice_array_matmul(W_tile, act_tile, args.r_sense, args.vread)
        except ngspice_bridge.NgspiceRunError as e:
            print(f"  [skip] ngspice run failed for {layer.name}: {e}", file=sys.stderr)
            continue

        relerr = ngspice_bridge.relative_error_pct(nodal, spice)
        results.append(dict(layer=layer.name, m0=m0, m1=m1, k0=k0, k1=k1, position=p,
                             nodal_mean_abs=float(np.mean(np.abs(nodal))),
                             spice_mean_abs=float(np.mean(np.abs(spice))),
                             relerr_pct=relerr))

        print(f"{layer.name:<22} ({m0}:{m1},{k0}:{k1}){'':<4} {p:4d} "
              f"{np.mean(np.abs(nodal)):14.6e} {np.mean(np.abs(spice)):14.6e} {relerr:10.2e}")

    if results:
        mean_relerr = float(np.mean([r['relerr_pct'] for r in results]))
        max_relerr = float(np.max([r['relerr_pct'] for r in results]))
        print("-" * 90)
        print(f"Mean relative error (nodal vs real ngspice): {mean_relerr:.2e} %")
        print(f"Max  relative error (nodal vs real ngspice): {max_relerr:.2e} %")
        if max_relerr < 1e-2:
            print("✓ Fast nodal model matches real ngspice closely across sampled "
                  "tiles -- the analog numbers in the main sweep above are trustworthy "
                  "as circuit-exact DC solves (still not full device-physics transient sims).")
        else:
            print("⚠ Mismatch larger than expected (>0.01%) on at least one sample -- "
                  "investigate before trusting the main sweep's analog numbers.")
    else:
        print("No successful ngspice samples collected.")

    return results


# =============================================================================
# Optional Top-1 accuracy-drop evaluation
# =============================================================================
def _load_imagenet_wnid_to_idx():
    """WNID -> standard ImageNet-1000 class index. Hardcodes the 10
    Imagenette WNIDs (sufficient for --eval-dir with the Imagenette
    download) plus an optional full-1000 override via IMAGENET_WNID_JSON
    env var (path to a {wnid: idx} JSON) if you have the full mapping and
    want to eval against a bigger local ImageNet-format validation set."""
    import os
    full_map_path = os.environ.get("IMAGENET_WNID_JSON")
    if full_map_path and os.path.exists(full_map_path):
        with open(full_map_path) as f:
            return json.load(f)

    # Standard WNID -> ImageNet-1000 index for Imagenette's 10 classes
    # (fixed, well-known mapping; see github.com/fastai/imagenette).
    return {
        "n01440764": 0,    # tench
        "n02102040": 217,  # English springer
        "n02979186": 482,  # cassette player
        "n03000684": 491,  # chain saw
        "n03028079": 497,  # church
        "n03394916": 566,  # French horn
        "n03417042": 569,  # garbage truck
        "n03425413": 571,  # gas pump
        "n03445777": 574,  # golf ball
        "n03888257": 701,  # parachute
    }


def evaluate_top1(model, args, quantize_weights_fn):
    """Runs FP32 vs FP2-quantized-weight Top-1 accuracy on a local image
    folder (torchvision ImageFolder layout). This does NOT run activations
    through the analog crossbar model (that would be prohibitively slow
    end-to-end); it measures the accuracy impact of weight quantization
    alone, which is the dominant term validated per-layer above. Returns
    (top1_fp32, top1_quant, drop_pct) or None if no --eval-dir given.

    IMPORTANT: ImageFolder assigns class indices 0..N-1 alphabetically by
    subfolder name, which is almost never the same ordering the model's
    1000-way ImageNet output head uses. Comparing argmax(logits) directly
    against ImageFolder's labels silently gives meaningless accuracy unless
    the folder names are the standard ImageNet WNIDs (n01440764, etc.) --
    in which case we remap them to the correct 0-999 index below. If your
    eval-dir's class folders are NOT real ImageNet WNIDs (e.g. a custom
    dataset), this remap will fail loudly rather than silently give wrong
    numbers -- you'd need a full fine-tuned/retrained head for that case,
    which is outside the scope of this quantization-effect benchmark."""
    if not args.eval_dir:
        return None
    if not (TORCH_AVAILABLE and TORCHVISION_AVAILABLE):
        print("Top-1 eval requested but torch/torchvision unavailable -- skipping.",
              file=sys.stderr)
        return None

    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(args.eval_dir, transform=tfm)

    # Remap ImageFolder's alphabetical 0..N-1 labels to real ImageNet-1000
    # indices via WNID lookup, so argmax(logits) is comparable to y.
    wnid_to_idx = _load_imagenet_wnid_to_idx()
    folder_idx_to_imagenet_idx = {}
    for wnid, folder_idx in dataset.class_to_idx.items():
        if wnid not in wnid_to_idx:
            raise ValueError(
                f"Eval folder '{wnid}' is not a recognized ImageNet WNID. "
                f"--eval-dir must use standard ImageNet WNID subfolder names "
                f"(e.g. Imagenette/ImageNet val layout) for Top-1 accuracy "
                f"against a stock ImageNet-pretrained model to be meaningful."
            )
        folder_idx_to_imagenet_idx[folder_idx] = wnid_to_idx[wnid]
    remap = torch.tensor([folder_idx_to_imagenet_idx[i] for i in range(len(dataset.classes))])

    n = min(len(dataset), args.max_eval_images)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(n)), batch_size=16)

    def run(m):
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                y_true = remap[y]  # folder-alphabetical -> real ImageNet-1000 index
                pred = m(x).argmax(dim=1)
                correct += int((pred == y_true).sum())
                total += y.shape[0]
        return 100.0 * correct / total if total else float("nan")

    top1_fp32 = run(model)

    import copy
    qmodel = copy.deepcopy(model)
    with torch.no_grad():
        for mod in qmodel.modules():
            if isinstance(mod, nn.Conv2d):
                w = mod.weight.data.numpy()
                orig_shape = w.shape
                # [Cout, Cin*Kh*Kw] -- block along axis=1 (the M/reduction
                # dimension) in chunks of 32, matching the paper's block
                # spec and the main per-layer benchmark's block_size.
                w_flat = w.reshape(orig_shape[0], -1)
                wq_flat = np.zeros_like(w_flat)
                block = 32
                for m0 in range(0, w_flat.shape[1], block):
                    m1 = min(m0 + block, w_flat.shape[1])
                    blk_scale = np.maximum(np.abs(w_flat[:, m0:m1]).max(axis=1, keepdims=True), 1e-8)
                    blk_scale = quantize_shared_scale(blk_scale, args.scale_mode)
                    wq_flat[:, m0:m1] = np.vectorize(cb.quantize_to_fp2)(
                        w_flat[:, m0:m1] / blk_scale) * blk_scale
                wq = wq_flat.reshape(orig_shape)
                mod.weight.data = torch.from_numpy(wq.astype(np.float32))
    top1_quant = run(qmodel)

    return top1_fp32, top1_quant, (top1_fp32 - top1_quant)


# =============================================================================
# Main benchmark loop
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="Local .pth ResNet-18 state_dict (no internet fetch of pretrained weights)")
    ap.add_argument("--synthetic", action="store_true", help="Skip torch entirely; use a small synthetic layer stack")
    ap.add_argument("--cifar-arch", action="store_true",
                    help="Build the CIFAR ResNet-18 stem (3x3 s1 conv, no maxpool) and use 32x32 "
                         "calibration inputs. Required for checkpoints from qat_finetune_fp2.py "
                         "trained on CIFAR-10/100.")
    ap.add_argument("--calib-dataset", default="random",
                    choices=["random", "cifar10", "cifar100", "imagefolder"],
                    help="Source of the calibration images whose activations the SNR/RelErr "
                         "numbers are measured against. 'random' is Gaussian NOISE and is only "
                         "kept for reproducing older runs -- it warns. Use cifar10 with "
                         "--calib-dir for anything you intend to report.")
    ap.add_argument("--calib-dir", default="./data",
                    help="Directory holding the calibration dataset (for cifar10/cifar100, the "
                         "dir containing cifar-10-batches-py/; for imagefolder, the image root).")
    ap.add_argument("--calib-images", type=int, default=16,
                    help="Number of calibration images. Sliding-window positions are pooled "
                         "across all of them before --max-positions sampling.")
    ap.add_argument("--skip-first-last", action="store_true",
                    help="Exclude the first conv from the table. The paper (and "
                         "qat_finetune_fp2.py by default) keeps the first and last layers at "
                         "full precision, so in the deployed model conv1 is NEVER mapped onto "
                         "the crossbar -- benchmarking it reports an error contribution the "
                         "hardware does not actually incur, and it is a persistent outlier "
                         "(M=27 gives it 1 partial block and the worst SNR in the table). The "
                         "last layer is an nn.Linear and is already absent here.")
    ap.add_argument("--scale-mode", default="e8m0", choices=["e8m0", "fp8", "none"],
                    help="Precision of the per-block shared scale. e8m0 (default) is the "
                         "paper's 8-bit hardware scale and matches qat_finetune_fp2.py; "
                         "none keeps it in FP32, which is optimistic (see quantize_shared_scale).")
    ap.add_argument("--num-classes", type=int, default=1000,
                    help="Classifier width; must match the checkpoint (10 for CIFAR-10).")
    ap.add_argument("--tile-m", type=int, default=TILE_M_DEFAULT)
    ap.add_argument("--tile-k", type=int, default=TILE_K_DEFAULT)
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--max-positions", type=int, default=32, help="Spatial output positions sampled per layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtl-log", default=None, help="Optional JSON psum dump from RTL sim, see parse_rtl_log()")
    ap.add_argument("--eval-dir", default=None, help="Optional ImageFolder dir for Top-1 accuracy-drop eval")
    ap.add_argument("--max-eval-images", type=int, default=200)
    ap.add_argument("--out-csv", default=None, help="Optional path to write the summary table as CSV")
    ap.add_argument("--validate-ngspice", type=int, default=0,
                     help="Run N real ngspice spot-checks (random layer/tile/position samples) "
                          "against the fast nodal model. Requires ngspice installed "
                          "(sudo apt-get install -y ngspice) and ngspice_bridge.py present "
                          "alongside this script. 0 = disabled (default).")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model, is_synthetic = build_model(args)
    layers = extract_conv_layers(model, args)

    if args.skip_first_last and layers:
        dropped = layers[0].name
        layers = layers[1:]
        print(f"--skip-first-last: excluding '{dropped}' (kept full-precision in the "
              f"deployed model, so it is not on the crossbar).", file=sys.stderr)

    rtl_data = parse_rtl_log(args.rtl_log) if args.rtl_log else {}

    print(f"{'Layer':<28} {'SNR_dig(dB)':>11} {'SNR_analog(dB)':>15} "
          f"{'RelErr_dig%':>12} {'RelErr_analog%':>15} {'CellUtil%':>10} {'RTL_RelErr%':>12}")
    print("-" * 108)

    rows = []
    for layer in layers:
        W_MK = weight_to_MK(layer.weight)
        acts = layer.unfolded_input  # [M, num_positions]

        ref = fp32_forward(W_MK, acts)

        w_scale = weight_scale_factor(W_MK, block_size=args.tile_m, scale_mode=args.scale_mode)
        dig, Wq = fp2_digital_forward(W_MK, acts, scale=w_scale, block_size=args.tile_m)
        analog, cell_util = analog_2t2r_forward(W_MK, acts, args.tile_m, args.tile_k,
                                                  args.r_sense, args.vread, scale=w_scale)

        snr_dig = snr_db(ref, dig)
        snr_analog = snr_db(ref, analog)
        relerr_dig = relative_error_pct(ref, dig)
        relerr_analog = relative_error_pct(ref, analog)

        rtl_relerr = float("nan")
        if layer.name in rtl_data:
            rtl_vals = rtl_data[layer.name]
            if rtl_vals.shape == analog.shape:
                rtl_relerr = relative_error_pct(analog, rtl_vals)
            else:
                print(f"  [warn] RTL log shape {rtl_vals.shape} != analog shape "
                      f"{analog.shape} for layer {layer.name}, skipping RTL diff",
                      file=sys.stderr)

        row = dict(
            layer=layer.name, snr_digital_db=snr_dig, snr_analog_db=snr_analog,
            relerr_digital_pct=relerr_dig, relerr_analog_pct=relerr_analog,
            cell_utilization_pct=100.0 * cell_util, rtl_relerr_pct=rtl_relerr,
            weight_scale_min=float(np.min(w_scale)), weight_scale_max=float(np.max(w_scale)),
            weight_scale_ratio=float(np.max(w_scale) / max(np.min(w_scale), 1e-12)),
            M=W_MK.shape[0], K=W_MK.shape[1],
        )
        rows.append(row)

        rtl_str = f"{rtl_relerr:12.3f}" if rtl_relerr == rtl_relerr else f"{'N/A':>12}"
        print(f"{layer.name:<28} {snr_dig:11.2f} {snr_analog:15.2f} "
              f"{relerr_dig:12.3f} {relerr_analog:15.3f} {100*cell_util:10.2f} {rtl_str}")

    if args.validate_ngspice > 0:
        rng = np.random.default_rng(args.seed + 1000)
        validate_against_ngspice(layers, args, rng)

    top1 = evaluate_top1(model, args, quantize_weights_fn=None)
    if top1 is not None:
        top1_fp32, top1_quant, drop = top1
        print(f"\nTop-1 accuracy: FP32={top1_fp32:.2f}%  FP2-weight-quantized={top1_quant:.2f}%  "
              f"drop={drop:.2f} pts  (weight-quantization effect only; does not include "
              f"crossbar analog noise, which per-layer SNR above bounds separately)")
    elif args.eval_dir is None:
        print("\n(No --eval-dir given: skipping Top-1 accuracy-drop evaluation. "
              "Pass a local torchvision-ImageFolder directory to enable it.)")

    if args.out_csv:
        try:
            import csv
            with open(args.out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nSummary written to {args.out_csv}")
        except IndexError:
            print("No layers benchmarked -- nothing to write.", file=sys.stderr)

    if is_synthetic:
        print("\nNOTE: ran in --synthetic mode (torch/torchvision unavailable or "
              "explicitly requested) -- layer shapes approximate ResNet-18's stem "
              "and one layer per stage, not the full 20-conv-layer network.")


if __name__ == "__main__":
    main()
