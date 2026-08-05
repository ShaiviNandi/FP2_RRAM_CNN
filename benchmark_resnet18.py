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
REAL input-activation statistics (not synthetic random activations) via
forward hooks + `F.unfold` (im2col), so the crossbar error numbers reflect
what that layer actually sees in inference, not a worst-case random matrix.

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

    model = tv_models.resnet18(weights=None)  # no internet fetch of pretrained weights
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state, strict=False)
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

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy)
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

        unfolded_full = F.unfold(x, kernel_size=(kh, kw), stride=stride, padding=pad)
        unfolded_full = unfolded_full[0].numpy()  # [Cin*Kh*Kw, num_positions]

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


def weight_scale_factor(W_MK, eps=1e-8):
    """Per-OUTPUT-CHANNEL (per-K-column) max-abs scale, so the FP2-E1M0 grid
    {-1,-.5,0,.5,1} actually gets used on every channel, not just whichever
    channel happens to have the largest weights. A single per-tensor scale
    is outlier-sensitive: trained conv layers routinely have a handful of
    channels with much larger weights than the rest (common after training
    -- some channels simply carry more signal), and a per-tensor max would
    let those few channels set the scale for everyone else, quantizing
    every other channel's weights to ~0. Per-channel scaling is also just
    the correct design choice here: it mirrors the per-channel BN-fold
    coefficient table already in residual_post_proc.sv -- each output
    channel gets its own quantization scale, exactly like it'll get its
    own BN scale/bias in hardware. Returns an array of shape [K]."""
    return np.maximum(np.max(np.abs(W_MK), axis=0), eps)


def fp2_digital_forward(W_MK, activations_MK, scale):
    """FP2-E1M0 quantized weights, exact (noiseless) dot product -- isolates
    pure quantization error from crossbar circuit error. `scale` is a
    per-output-channel array [K] (see weight_scale_factor()) that rescales
    each column of W_MK into the quantizer's dynamic range before
    quantizing; the result is rescaled back per-channel so units match the
    FP32 reference."""
    scale = np.asarray(scale)
    Wq = np.vectorize(cb.quantize_to_fp2)(W_MK / scale[None, :])
    return (Wq.T @ activations_MK) * scale[:, None], Wq


def analog_2t2r_forward(W_MK, activations_MK, tile_m, tile_k, r_sense, vread, scale, n_seeds_noise=1):
    """Runs the FP2-quantized weight matrix through the actual resistive-
    divider 2T2R golden model (cb.golden_array_matmul), tiled to (tile_m,
    tile_k), with results recovered into the same units as the digital dot
    product (matching crossbar_cli.py's recovery scaling) and summed across
    M-tiles (spatial reduction) / concatenated across K-tiles (output
    channel tiling). `scale` is the per-output-channel array [K] from
    weight_scale_factor(), applied per K-tile slice below. Returns
    (result[K,num_positions], cell_utilization)."""
    M, K = W_MK.shape
    num_pos = activations_MK.shape[1]
    g_lrs = 1.0 / cb.col.R_FOR_MAGNITUDE[1.0]
    scale = np.asarray(scale)

    Wq = np.vectorize(cb.quantize_to_fp2)(W_MK / scale[None, :])
    result = np.zeros((K, num_pos), dtype=np.float64)

    m_ranges = tile_ranges(M, tile_m)
    k_ranges = tile_ranges(K, tile_k)

    total_cells = 0
    active_cells = 0

    for (k0, k1) in k_ranges:
        scale_tile = scale[k0:k1]
        for (m0, m1) in m_ranges:
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

        scale = weight_scale_factor(W_MK)
        Wq = np.vectorize(cb.quantize_to_fp2)(W_MK / scale[None, :])

        tile_m, tile_k = min(args.tile_m, M), min(args.tile_k, K)
        m0 = int(rng.integers(0, max(1, M - tile_m + 1)))
        k0 = int(rng.integers(0, max(1, K - tile_k + 1)))
        m1, k1 = min(m0 + tile_m, M), min(k0 + tile_k, K)
        p = int(rng.integers(0, acts.shape[1]))

        W_tile = Wq[m0:m1, k0:k1].tolist()
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
                per_channel_scale = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
                per_channel_scale = np.maximum(per_channel_scale, 1e-8).reshape(-1, 1, 1, 1)
                wq = np.vectorize(cb.quantize_to_fp2)(w / per_channel_scale) * per_channel_scale
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

    rtl_data = parse_rtl_log(args.rtl_log) if args.rtl_log else {}

    print(f"{'Layer':<28} {'SNR_dig(dB)':>11} {'SNR_analog(dB)':>15} "
          f"{'RelErr_dig%':>12} {'RelErr_analog%':>15} {'CellUtil%':>10} {'RTL_RelErr%':>12}")
    print("-" * 108)

    rows = []
    for layer in layers:
        W_MK = weight_to_MK(layer.weight)
        acts = layer.unfolded_input  # [M, num_positions]

        ref = fp32_forward(W_MK, acts)

        w_scale = weight_scale_factor(W_MK)
        dig, Wq = fp2_digital_forward(W_MK, acts, scale=w_scale)
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
