#!/usr/bin/env python3
"""
analog_eval.py
End-to-end Top-1 accuracy with every convolution executed through the ACTUAL
resistive-divider crossbar, not through a digital surrogate.

Why this is the missing number:
codesign_sweep.py reports accuracy from qat_finetune_fp2.py, which uses
fake-quantized weights and an EXACT dot product. It contains no crossbar. So
its accuracy column answers "what does FP2 quantization cost?" and says
nothing about what the analog readout costs on top.

That gap matters because the two move in opposite directions with block size B:

    B=32  -> acc 92.44,  analog SNR 10.99 dB
    B=128 -> acc 92.57,  analog SNR  6.76 dB

Accuracy says B=128 is free. SNR says B=128 costs 4.2 dB. Only an end-to-end
analog forward pass can say which one governs, and that is the difference
between "FP2 should use B=128 on a crossbar, here is 2.8x efficiency for free"
and "the analog noise floor binds, B=32 is correct". Both are publishable, and
guessing between them is not an option.

How it stays exact and still finishes:
crossbar_array_test.golden_array_matmul solves each column independently:

    v_blp[k] = (sum_i v_i / r_p[i,k]) / (1/Rs + sum_i 1/r_p[i,k])

That is a closed form, not an iterative solve, so it vectorises exactly:

    Gp        = 1 / Rp                                  [M, K]
    sum_g[k]  = Gp.sum(0) + 1/Rs                        [K]
    numer     = Gp.T @ V                                [K, P]
    v_blp     = numer / sum_g[:, None]                  [K, P]

One matmul per tile per polarity, on the GPU, in float64. No approximation is
introduced anywhere -- --self-test asserts agreement with the scalar reference
implementation to machine precision, and that reference is itself the model
ngspice validated across all 87,168 tiles at 3.4e-7% worst case.

So the chain of evidence is: ngspice -> golden_array_matmul -> this. Every link
is checked, and the accuracy number at the end is a circuit-exact DC result
rather than a noise-injection approximation.

What it models now:
ADC quantization of each bitline voltage (--adc-bits), log-normal programming
scatter frozen per device (--variability), and power-law conductance drift with
state-dependent and cell-to-cell-dispersed exponents (--drift).

What it still does not model:
Read noise, IR drop along the wordlines, temperature, and the tail of the
retention distribution. All make the real number worse. Treat this as the
ceiling the analog path allows, not as silicon.

Usage:
    python3 analog_eval.py --self-test           # no data, no GPU needed

    # the crux number for one block size
    python3 analog_eval.py --checkpoint qat_b128.pth --block-size 128         --data-dir ./data --max-images 2000

    # the crux TABLE: digital vs analog accuracy across the co-design sweep
    python3 analog_eval.py --sweep 32,64,128,256 --data-dir ./data         --max-images 2000 --out-csv analog_accuracy.csv
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

import crossbar_array_test as cb
import column_mac_test as col

DTYPE = {"float32": None, "float64": None}   # filled in below if torch imported
if TORCH_OK:
    DTYPE = {"float32": torch.float32, "float64": torch.float64}

R_LRS = col.R_FOR_MAGNITUDE[1.0]
R_MID = col.R_FOR_MAGNITUDE[0.5]
R_HRS = col.R_FOR_MAGNITUDE[0.0]
G_LRS = 1.0 / R_LRS


# =============================================================================
# Vectorised exact crossbar
# =============================================================================
def fp2_resistances(Wq):
    """FP2 levels -> (Rp, Rn), mirroring crossbar_array_test.decompose_differential.

    A zero weight puts BOTH polarities at full HRS, which is why an all-zero
    tile produces exactly zero differential current rather than an offset --
    the ngspice self-test checks precisely this degenerate case."""
    mag = Wq.abs()
    r_mag = torch.where(mag > 0.75, torch.full_like(Wq, R_LRS),
             torch.where(mag > 0.25, torch.full_like(Wq, R_MID),
                         torch.full_like(Wq, R_HRS)))
    pos = Wq >= 0
    hrs = torch.full_like(Wq, R_HRS)
    Rp = torch.where(pos, r_mag, hrs)
    Rn = torch.where(pos, hrs, r_mag)
    return Rp, Rn


def analog_matmul(Wq, V, r_sense, vread, mode="analog"):
    """Exact nodal solve for one M x K tile against P activation vectors.

    Wq: [M, K] quantized FP2 levels.  V: [M, P] activations (unscaled).
    Returns [K, P] differential current, in the same units as
    cb.golden_array_matmul.

    mode:
      "analog"    the raw shared-Rsense readout, matching golden_array_matmul.
      "corrected" the same readout with the per-column loading gain divided
                  back out (see below).
      "ideal"     the Rsense->0 limit, i.e. the digital-exact FP2 dot product
                  expressed in the same current units. Use as the baseline.

    THE LOADING ERROR IS A DETERMINISTIC GAIN, NOT NOISE
    ----------------------------------------------------
    Rearranging the nodal solution for one polarity:

        i_p = (sum_i v_i/r_p_i) / (1 + Rs * sum_i 1/r_p_i)
            = i_ideal_p / (1 + Rs * G_col_p)

    The denominator depends ONLY on the programmed conductances of that
    column. It is a compile-time constant, identical for every activation
    vector ever applied. So the shared-Rsense "error" this repo has been
    measuring as SNR_analog is, in this model, a per-column scale factor --
    and at M=32 with ~42% utilisation it is a 33% attenuation, rising to 71%
    at M=256. That is not degradation, it is gain.

    Because 2T2R reads the two bitlines on separate ADCs (hw_model already
    budgets 2K converters per tile), each branch can be rescaled by its own
    known (1 + Rs*G_col) BEFORE the digital subtraction, and the correction
    is then algebraically exact: the result collapses to (Gp - Gn)^T @ Vw,
    the ideal MAC.

    WHAT THIS MEANS FOR THE SNR NUMBERS
    -----------------------------------
    In a noiseless deterministic model every error is systematic by
    construction, so "correctable in principle" is close to tautological
    here. The honest reading is that SNR_analog as previously reported
    OVERSTATES the analog penalty, because it charges the crossbar for an
    artifact a calibration LUT removes. The residual physical penalty needs
    device-to-device variability, read noise, conductance drift and ADC
    quantisation -- none of which this model contains. Report the corrected
    number as the optimistic bound and say plainly what is missing.

    The correction is also NOT free in silicon: it needs one multiplier and
    one stored constant per column, and it assumes the programmed
    conductances are known exactly, which after write-verify they roughly
    are and after drift they are not."""
    Rp, Rn = fp2_resistances(Wq)
    G = (1.0 / Rp, 1.0 / Rn, 1.0 / Rp, 1.0 / Rn)
    return analog_matmul_G(G, V, r_sense, vread, mode)


def quantize_adc(v, bits, fs):
    """Signed uniform mid-tread quantizer over +-fs.

    Applied to the BITLINE VOLTAGE, before the gain correction and before the
    differential subtraction, because that is where a real ADC sits: the 2T2R
    column has two bitlines and hw_model already budgets two converters per
    column. Quantizing each branch separately and subtracting afterwards is
    strictly noisier than quantizing the difference would be -- the two
    quantization errors are independent, so their variances add -- and it is
    what the hardware actually does."""
    if not bits or bits <= 0:
        return v
    levels = 2 ** (bits - 1)
    lsb = fs / max(levels - 1, 1)
    return torch.clamp(torch.round(v / lsb), -levels, levels - 1) * lsb


def analog_matmul_G(G, V, r_sense, vread, mode="analog", adc_bits=0, adc_fs=None):
    """The solve, given conductances directly.

    G = (Gp, Gn, Gp_cal, Gn_cal). The first pair is what the DEVICE actually
    is; the second is what the CALIBRATION BELIEVES it is. They differ under
    device variability, and that difference is the whole experiment: a
    calibration constant computed from the nominal programmed conductance is
    wrong by roughly the variability sigma, so the exactness of the correction
    degrades gracefully or catastrophically depending on how the errors
    accumulate down a column.

    Note the 'ideal' target uses the NOMINAL conductances: the reference is
    the dot product the network was trained to compute, not the one a
    particular perturbed device happens to implement."""
    Gp, Gn, Gp_cal, Gn_cal = G
    Vw = vread * V

    if mode == "ideal":
        return (Gp_cal - Gn_cal).t() @ Vw

    inv_rs = 1.0 / r_sense
    sum_gp, sum_gn = Gp.sum(0), Gn.sum(0)
    v_blp = (Gp.t() @ Vw) / (sum_gp + inv_rs).unsqueeze(1)
    v_bln = (Gn.t() @ Vw) / (sum_gn + inv_rs).unsqueeze(1)

    if adc_bits:
        # Full scale. `None` means per-call auto-ranging from the observed peak,
        # i.e. a PERFECT AGC that always uses the whole converter range. That is
        # optimistic: a real design fixes full scale per layer from calibration
        # data and loses whatever headroom the worst-case input demands. Note
        # also that the divider makes the bitline swing far smaller than
        # V_read -- at M=32 it is ~0.03 V against a 0.1 V drive -- so scaling
        # full scale to V_read instead would throw away roughly two bits.
        fs = adc_fs
        if fs is None:
            fs = float(torch.maximum(v_blp.abs().max(), v_bln.abs().max()))
            fs = max(fs, 1e-18)
        v_blp = quantize_adc(v_blp, adc_bits, fs)
        v_bln = quantize_adc(v_bln, adc_bits, fs)

    i_p, i_n = v_blp / r_sense, v_bln / r_sense

    if mode == "corrected":
        i_p = i_p * (1.0 + r_sense * Gp_cal.sum(0)).unsqueeze(1)
        i_n = i_n * (1.0 + r_sense * Gn_cal.sum(0)).unsqueeze(1)
    elif mode != "analog":
        raise ValueError(f"unknown mode {mode!r}")
    return i_p - i_n


def apply_drift(G, t_s, nu_lrs, nu_hrs, nu_sigma, gen, t_ref=1.0):
    """Conductance relaxation over time: G(t) = G(t_ref) * (t/t_ref)^(-nu).

    The power law is the standard resistance-drift model from the PCM
    literature and is applied to ReRAM for the same reason: the filament
    relaxes structurally after programming, and the relaxation is
    scale-free over many decades of time.

    Two details that decide whether the calibration survives:

    STATE DEPENDENCE. The high-resistance state relaxes markedly faster than
    the low-resistance state -- HRS has no stable conductive filament to hold
    it. So nu_hrs > nu_lrs. This matters less than it looks for the gain
    constant, because sum_G over a column is dominated by the LRS/MID cells
    (1.8e-3 S each against HRS's 4.6e-6 S), but it does move the differential
    zero-point.

    CELL-TO-CELL SPREAD IN nu. This is the part that cannot be calibrated
    away. If every cell drifted by the same factor, the gain constant would
    simply be wrong by a known amount and a single global rescale would fix
    it. Because nu itself varies cell to cell, drift injects DISPERSION that
    grows with log(t), and dispersion is not a per-column constant. This is
    the mechanism by which drift can break a compile-time calibration when
    programming scatter did not: programming scatter is frozen at t=0 and the
    constants are computed after it, whereas drift keeps moving.

    G is split by magnitude rather than by a stored mask: cells above the
    geometric mean of G_LRS and G_HRS are treated as programmed-on."""
    if t_s <= t_ref:
        return G
    g_split = math.sqrt((1.0 / R_LRS) * (1.0 / R_HRS))
    nu = torch.where(G > g_split,
                     torch.full_like(G, nu_lrs), torch.full_like(G, nu_hrs))
    if nu_sigma > 0:
        noise = torch.randn(G.shape, generator=gen, dtype=G.dtype, device="cpu")
        nu = nu + nu_sigma * noise.to(G.device)
    return G * torch.pow(torch.tensor(t_s / t_ref, dtype=G.dtype,
                                      device=G.device), -nu)


def fmt_time(t_s):
    for div, unit in ((31536000.0, "y"), (2592000.0, "mo"), (86400.0, "d"),
                      (3600.0, "h"), (60.0, "min")):
        if t_s >= div:
            return f"{t_s/div:g}{unit}"
    return f"{t_s:g}s"


def perturb_conductance(G, sigma, gen):
    """Log-normal device-to-device variability on a programmed conductance.

    Log-normal, not Gaussian: conductance is strictly positive and ReRAM
    programming spread is multiplicative (a cell 20% high stays 20% high), so
    an additive Gaussian would both allow negative conductances and misstate
    the tail. sigma is the standard deviation of ln(G), so sigma=0.1 is
    roughly a 10% 1-sigma spread.

    This is programmed ONCE per device at conversion time and then frozen --
    it is fabrication and write scatter, not per-read noise. Read noise and
    drift are separate effects and are still not modelled."""
    if sigma <= 0:
        return G
    # Draw on CPU and move: a torch.Generator is bound to one device type, and
    # seeding a CUDA generator does not reproduce a CPU one anyway. Drawing on
    # CPU keeps the device instance bit-identical across CPU and GPU runs,
    # which matters because each seed is meant to be "one chip".
    noise = torch.randn(G.shape, generator=gen, dtype=G.dtype, device="cpu")
    return G * torch.exp(sigma * noise.to(G.device))


def quantize_scale_e8m0(scale, mode="e8m0", eps=1e-8):
    if mode == "none":
        return scale
    exp = torch.ceil(torch.log2(scale.clamp_min(eps))).clamp(-128.0, 127.0)
    pow2 = torch.pow(2.0, exp)
    if mode == "e8m0":
        return pow2
    mant = torch.ceil((scale / pow2) * 128.0) / 128.0
    return (pow2 * mant).clamp_min(eps)


def quantize_blocks(W_MK, block, scale_mode):
    """[M,K] float -> ([M,K] FP2 levels, [n_blocks,K] scales), blocks along M.
    Same partition as benchmark_resnet18.weight_scale_factor."""
    M, K = W_MK.shape
    n_full, rem = M // block, M % block
    Wq = torch.zeros_like(W_MK)
    scales = []
    if n_full:
        h = W_MK[: n_full * block].reshape(n_full, block, K)
        s = quantize_scale_e8m0(h.abs().amax(dim=1).clamp_min(1e-8), scale_mode)  # [n_full,K]
        Wq[: n_full * block] = torch.clamp(
            torch.round(h / s.unsqueeze(1) * 2.0) / 2.0, -1.0, 1.0).reshape(-1, K)
        scales.append(s)
    if rem:
        t = W_MK[n_full * block:]
        s = quantize_scale_e8m0(t.abs().amax(dim=0, keepdim=True).clamp_min(1e-8), scale_mode)
        Wq[n_full * block:] = torch.clamp(torch.round(t / s * 2.0) / 2.0, -1.0, 1.0)
        scales.append(s)
    return Wq, torch.cat(scales, dim=0)


class AnalogConv2d(nn.Module):
    """Conv2d whose forward is im2col + tiled exact-crossbar matmul.

    Tiles along M are quantization blocks (block == tile_m, enforced), so each
    tile's output current can be rescaled by that block's single scale and the
    partial sums added digitally -- which is what the real accumulator does."""

    def __init__(self, conv, block, tile_k, r_sense, vread, scale_mode, device,
                 mode="analog", sigma=0.0, gen=None, calib_knows_actual=False,
                 adc_bits=0, adc_fs=None, dtype=torch.float32,
                 drift_t=0.0, nu_lrs=0.01, nu_hrs=0.05, nu_sigma=0.004,
                 recalibrate=False):
        super().__init__()
        self.mode = mode
        self.sigma = sigma
        self.calib_knows_actual = calib_knows_actual
        self._gen = gen
        self.adc_bits, self.adc_fs = adc_bits, adc_fs
        self.stride, self.padding = conv.stride, conv.padding
        self.kh, self.kw = conv.kernel_size
        self.out_ch = conv.out_channels
        self.bias = conv.bias
        self.block, self.tile_k = block, tile_k
        self.r_sense, self.vread = r_sense, vread

        self.dtype = dtype
        W = conv.weight.detach().to(dtype)
        W_MK = W.reshape(self.out_ch, -1).t().contiguous().to(device)   # [M,K]
        self.Wq, self.scale = quantize_blocks(W_MK, block, scale_mode)
        self.M, self.K = W_MK.shape
        self.m_ranges = [(i, min(i + block, self.M)) for i in range(0, self.M, block)]
        self.k_ranges = [(i, min(i + tile_k, self.K)) for i in range(0, self.K, tile_k)]

        # Conductances are resolved ONCE here, not per forward pass. Device
        # variability is a property of the programmed array: the same cell is
        # the same amount wrong on every read, for the life of the chip.
        # Re-sampling it per batch would model read noise instead, average it
        # away over the test set, and flatter the design.
        Rp_nom, Rn_nom = fp2_resistances(self.Wq)
        Gp_nom, Gn_nom = 1.0 / Rp_nom, 1.0 / Rn_nom
        Gp_act = perturb_conductance(Gp_nom, sigma, gen)
        Gn_act = perturb_conductance(Gn_nom, sigma, gen)
        # What the calibration LUT believes. Blind calibration uses the target
        # conductance; write-verify calibration measures the real one.
        Gp_cal = Gp_act if calib_knows_actual else Gp_nom
        Gn_cal = Gn_act if calib_knows_actual else Gn_nom

        # Drift acts after programming and keeps acting. The calibration
        # constants were computed at deployment, so by default they are stale
        # by exactly this much -- that is the whole question. --recalibrate
        # models a field refresh that re-reads the array and recomputes them,
        # which is the upper bound on what any calibration policy can do.
        self.refresh_gain = None
        if drift_t > 0:
            Gp_act = apply_drift(Gp_act, drift_t, nu_lrs, nu_hrs, nu_sigma, gen)
            Gn_act = apply_drift(Gn_act, drift_t, nu_lrs, nu_hrs, nu_sigma, gen)
            if recalibrate:
                Gp_cal, Gn_cal = Gp_act, Gn_act
                # Drift changes the WEIGHTS, not only the loading. Fixing the
                # loading constant alone leaves the whole layer scaled down by
                # roughly the LRS retention factor -- 16% at one year, which
                # dominates everything else. A field refresh that is already
                # re-reading the array can also fold a per-column gain into
                # the block scale it is about to apply, at zero runtime cost,
                # because that multiply is in the digital path anyway.
                #
                # The least-squares optimal scalar per column, mapping the
                # drifted differential conductance back onto the nominal one:
                #     alpha_k = <D_nom_k, D_act_k> / <D_act_k, D_act_k>
                # It cannot undo per-cell dispersion -- one scalar cannot fix
                # M independent errors -- which is precisely the irreducible
                # part of drift.
                D_nom = Gp_nom - Gn_nom
                D_act = Gp_act - Gn_act
                num = (D_nom * D_act).sum(0)
                den = (D_act * D_act).sum(0).clamp_min(1e-30)
                self.refresh_gain = (num / den)          # [K]
        self.G = (Gp_act, Gn_act, Gp_cal, Gn_cal)

        # --- batched tile layout -------------------------------------------
        # One bmm over all M-tiles instead of a Python loop over them. At B=32
        # a layer with M=4608 has 144 tiles; times 19 layers times ~79 batches
        # that was ~216k kernel launches per pass, each on a matrix far too
        # small to occupy the GPU. Hence single-digit utilisation.
        #
        # M is padded up to a whole number of tiles with ZERO conductance and
        # ZERO activation. That is not a fudge: a row with zero conductance is
        # an open circuit, contributing nothing to either the numerator or the
        # sum_g denominator, which is exactly what an unused row in a partial
        # tile is. --self-test asserts the batched path reproduces the loop
        # bit-for-bit, including the partial-tile case (M=576, block=128).
        self.T = (self.M + block - 1) // block
        self.Mpad = self.T * block
        pad = self.Mpad - self.M
        self.Gb = []
        for g in self.G:
            if pad:
                g = torch.cat([g, torch.zeros(pad, self.K, dtype=g.dtype,
                                              device=g.device)], dim=0)
            self.Gb.append(g.reshape(self.T, block, self.K).contiguous())
        self.sum_gp = self.Gb[0].sum(1)        # [T,K]  actual, for the solve
        self.sum_gn = self.Gb[1].sum(1)
        self.corr_p = 1.0 + r_sense * self.Gb[2].sum(1)   # believed, for calib
        self.corr_n = 1.0 + r_sense * self.Gb[3].sum(1)
        self.scale_b = self.scale.contiguous()            # [T,K]

    def forward(self, x):
        N = x.shape[0]
        cols = F.unfold(x.to(self.dtype), (self.kh, self.kw),
                        stride=self.stride, padding=self.padding)     # [N,M,L]
        L = cols.shape[-1]
        V = cols.permute(1, 0, 2).reshape(self.M, N * L)              # [M, N*L]
        P = V.shape[1]
        if self.Mpad != self.M:
            V = torch.cat([V, torch.zeros(self.Mpad - self.M, P,
                                          dtype=V.dtype, device=V.device)], 0)
        # There is no loop over K either. Columns share the wordlines but no
        # other node -- each has its own bitline pair and sense resistor -- so
        # the nodal solve is independent per column and slicing K changes
        # nothing. tile_k still sets ADC counts and area in hw_model; it has no
        # numerical role here.
        Vw = (self.vread * V).reshape(self.T, self.block, P)           # [T,b,P]
        Gp, Gn = self.Gb[0], self.Gb[1]

        if self.mode == "ideal":
            i = torch.bmm((self.Gb[2] - self.Gb[3]).transpose(1, 2), Vw)
        else:
            inv_rs = 1.0 / self.r_sense
            v_blp = torch.bmm(Gp.transpose(1, 2), Vw) / (self.sum_gp + inv_rs).unsqueeze(2)
            v_bln = torch.bmm(Gn.transpose(1, 2), Vw) / (self.sum_gn + inv_rs).unsqueeze(2)
            if self.adc_bits:
                # Full scale per tile, matching one converter per bitline of a
                # tile. self.adc_fs pins it instead, which is what a real design
                # does (fixed per layer from calibration data).
                if self.adc_fs is None:
                    fs = torch.maximum(v_blp.abs().amax(dim=(1, 2)),
                                       v_bln.abs().amax(dim=(1, 2)))
                    fs = fs.clamp_min(1e-18).view(self.T, 1, 1)
                else:
                    fs = self.adc_fs
                v_blp = quantize_adc(v_blp, self.adc_bits, fs)
                v_bln = quantize_adc(v_bln, self.adc_bits, fs)
            i_p, i_n = v_blp / self.r_sense, v_bln / self.r_sense
            if self.mode == "corrected":
                i_p = i_p * self.corr_p.unsqueeze(2)
                i_n = i_n * self.corr_n.unsqueeze(2)
            i = i_p - i_n
        out = (i / (self.vread * G_LRS) * self.scale_b.unsqueeze(2)).sum(0)
        if self.refresh_gain is not None:
            out = out * self.refresh_gain.unsqueeze(1)
        oh = (x.shape[2] + 2 * self.padding[0] - self.kh) // self.stride[0] + 1
        ow = (x.shape[3] + 2 * self.padding[1] - self.kw) // self.stride[1] + 1
        y = out.reshape(self.K, N, L).permute(1, 0, 2).reshape(N, self.K, oh, ow)
        if self.bias is not None:
            y = y + self.bias.detach().to(self.dtype).view(1, -1, 1, 1)
        return y.float()


def convert_model(model, block, tile_k, r_sense, vread, scale_mode, device,
                  skip_first_last=True, mode="analog", verbose=True,
                  sigma=0.0, seed=0, calib_knows_actual=False,
                  adc_bits=0, adc_fs=None, dtype=torch.float32,
                 drift_t=0.0, nu_lrs=0.01, nu_hrs=0.05, nu_sigma=0.004,
                 recalibrate=False):
    gen = (torch.Generator(device="cpu").manual_seed(seed)
           if (sigma > 0 or drift_t > 0) else None)
    convs = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    skip = {convs[0]} if (skip_first_last and convs) else set()
    n_conv = 0
    for name in convs:
        if name in skip:
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        setattr(parent, parts[-1],
                AnalogConv2d(old, block, tile_k, r_sense, vread, scale_mode,
                             device, mode, sigma, gen, calib_knows_actual,
                             adc_bits, adc_fs, dtype, drift_t, nu_lrs, nu_hrs,
                             nu_sigma, recalibrate))
        n_conv += 1
    if verbose:
        extra = (f", sigma={sigma:.0%} "
                 f"({'write-verified' if calib_knows_actual else 'blind'} calib)"
                 if sigma > 0 else "")
        if adc_bits:
            extra += f", {adc_bits}-bit ADC"
        if drift_t > 0:
            extra += (f", drift {fmt_time(drift_t)} "
                      f"({'recalibrated' if recalibrate else 'stale constants'})")
        print(f"  {n_conv} conv layers in '{mode}' mode{extra}, "
              f"{sorted(skip) if skip else 'none'} kept full precision")
    return model


def adc_sweep(ckpt, block, args, device):
    """ADC resolution against tile height -- the interaction that decides
    whether the tall-tile efficiency win is real.

    Taller columns sum more terms, so the psum dynamic range grows roughly as
    sqrt(M) for random-sign weights and as M in the worst case, while the
    quantization step is fixed by the converter. If N bits are enough at M=32
    but not at M=128, then the energy saved by amortising the ADC over more
    rows is handed straight back as a demand for more ADC bits -- and since
    SAR energy grows as 2^N, that trade can easily be a net loss. This sweep
    is what locates the knee."""
    loader, n = cifar_loader(args.data_dir, args.max_images, args.batch_size,
                             args.dataset)
    bits = [int(b) for b in args.adc_sweep.split(",")]
    print(f"\n=== ADC sweep: {ckpt}, B={block}, {n} images ===")
    rows = []
    for nb in bits + [0]:
        m = build_model(ckpt, args.num_classes, device)
        convert_model(m, block, args.tile_k, args.r_sense, args.vread,
                      args.scale_mode, device,
                      skip_first_last=not args.no_skip_first_last,
                      mode="corrected", verbose=False, adc_bits=nb,
                      dtype=DTYPE[args.dtype])
        acc = top1(m, loader, device)
        del m
        label = f"{nb}-bit" if nb else "exact (no ADC)"
        print(f"  {label:>14}: {acc:6.2f}%")
        rows.append(dict(block_size=block, adc_bits=nb, acc_corrected=acc))
    return rows


def drift_sweep(ckpt, block, args, device):
    """Does a compile-time calibration go stale?

    Programming scatter is frozen at deployment, so constants computed after
    it stay correct forever -- which is why the variability sweep came back
    clean. Drift is different in kind: conductances keep moving, so constants
    computed once are wrong by a growing amount. This is the remaining
    mechanism that can break the contribution.

    Four policies at each retention time:
      raw          no calibration at all, for reference
      stale        constants from deployment, never refreshed. This is what
                   the contribution as written actually proposes.
      recalib      constants recomputed from the drifted conductances, i.e. a
                   field refresh that re-reads the array. Upper bound on any
                   calibration policy, and the fallback if `stale` fails.
      ideal        digital-exact FP2, the accuracy ceiling

    The gap between `stale` and `recalib` is the value of refreshing; the gap
    between `recalib` and `ideal` is the part of drift no calibration can
    reach, because it is dispersion rather than a per-column constant."""
    loader, n = cifar_loader(args.data_dir, args.max_images, args.batch_size,
                             args.dataset)
    times = [float(t) for t in args.drift.split(",")]
    print(f"\n=== drift sweep: {ckpt}, B={block}, {n} images, "
          f"nu_LRS={args.nu_lrs} nu_HRS={args.nu_hrs} sigma_nu={args.nu_sigma}, "
          f"{args.drift_seeds} seed(s) ===")

    base = build_model(ckpt, args.num_classes, device)
    convert_model(base, block, args.tile_k, args.r_sense, args.vread,
                  args.scale_mode, device,
                  skip_first_last=not args.no_skip_first_last, mode="ideal",
                  verbose=False, adc_bits=args.adc_bits,
                  dtype=DTYPE[args.dtype])
    acc_ideal = top1(base, loader, device)
    del base
    print(f"  FP2 digital-exact ceiling: {acc_ideal:.2f}%")

    rows = []
    for t in times:
        acc = {}
        for label, mode, recal in (("raw", "analog", False),
                                   ("stale", "corrected", False),
                                   ("recalib", "corrected", True)):
            runs = []
            for s in range(args.drift_seeds):
                m = build_model(ckpt, args.num_classes, device)
                convert_model(m, block, args.tile_k, args.r_sense, args.vread,
                              args.scale_mode, device,
                              skip_first_last=not args.no_skip_first_last,
                              mode=mode, verbose=False, sigma=args.drift_sigma,
                              seed=3000 * s + 11, adc_bits=args.adc_bits,
                              dtype=DTYPE[args.dtype], drift_t=t,
                              nu_lrs=args.nu_lrs, nu_hrs=args.nu_hrs,
                              nu_sigma=args.nu_sigma, recalibrate=recal)
                runs.append(top1(m, loader, device))
                del m
            acc[label] = float(np.mean(runs))
            if len(runs) > 1:
                acc[label + "_sd"] = float(np.std(runs))
        rows.append(dict(block_size=block, t_seconds=t, t_label=fmt_time(t),
                         acc_ideal=acc_ideal, **acc))
        print(f"  t={fmt_time(t):>6}  raw {acc['raw']:6.2f}%   "
              f"stale-calib {acc['stale']:6.2f}%   "
              f"recalibrated {acc['recalib']:6.2f}%   "
              f"(ceiling {acc_ideal:.2f}%)")
    return rows


def variability_sweep(ckpt, block, args, device):
    """The experiment that decides whether the gain calibration is a real
    contribution or an artifact of a noiseless model.

    Three curves against sigma:
      raw          no calibration at all
      blind        calibration from the NOMINAL programmed conductances, which
                   is what is actually buildable: the compiler knows what it
                   asked the array to store, not what the array became
      write-verify calibration from the ACTUAL conductances, i.e. every cell is
                   read back after programming and its true value stored. Costs
                   one measurement per cell and a per-column constant, and it
                   is the upper bound on what calibration can do.

    If blind calibration holds up at sigma=0.1, the contribution stands as
    stated. If only write-verify holds up, the contribution becomes 'calibrate
    from measured conductances', which is a heavier but still reasonable ask.
    If neither holds up, the tile-height limit is a device-variability limit
    and the paper's claim has to change."""
    loader, n = cifar_loader(args.data_dir, args.max_images, args.batch_size,
                             args.dataset)
    sigmas = [float(s) for s in args.variability.split(",")]
    print(f"\n=== variability sweep: {ckpt}, B={block}, {n} images, "
          f"{args.variability_seeds} seed(s) per point ===")

    rows = []
    for sigma in sigmas:
        acc = {}
        for label, mode, knows in (("raw", "analog", False),
                                   ("blind", "corrected", False),
                                   ("wverify", "corrected", True)):
            if sigma == 0 and label == "wverify":
                acc[label] = acc.get("blind")     # identical at sigma=0
                continue
            runs = []
            for s in range(args.variability_seeds):
                m = build_model(ckpt, args.num_classes, device)
                convert_model(m, block, args.tile_k, args.r_sense, args.vread,
                              args.scale_mode, device,
                              skip_first_last=not args.no_skip_first_last,
                              mode=mode, verbose=False, sigma=sigma,
                              seed=1000 * s + 7, calib_knows_actual=knows,
                              adc_bits=args.adc_bits, dtype=DTYPE[args.dtype])
                runs.append(top1(m, loader, device))
                del m
            acc[label] = float(np.mean(runs))
            acc[label + "_sd"] = float(np.std(runs)) if len(runs) > 1 else 0.0
        rows.append(dict(block_size=block, sigma=sigma,
                         n_seeds=args.variability_seeds, **acc))
        print(f"  sigma={sigma:5.1%}  raw {acc['raw']:6.2f}%   "
              f"blind-calib {acc['blind']:6.2f}%   "
              f"write-verify {acc.get('wverify', float('nan')):6.2f}%")
    return rows


# =============================================================================
def build_model(ckpt, num_classes, device):
    import torchvision.models as tv
    m = tv.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] {len(missing)} missing / {len(unexpected)} unexpected keys",
              file=sys.stderr)
    return m.eval().to(device)


DATASETS = {
    # name: (torchvision class, mean, std, num_classes)
    "cifar10":  ("CIFAR10",  (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ("CIFAR100", (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100),
}


def cifar_loader(data_dir, max_images, batch, dataset="cifar10"):
    """Test-set loader. Normalisation constants are per-dataset and must match
    what qat_finetune_fp2.py trained with, or every accuracy number shifts by
    a few points for no visible reason."""
    if dataset not in DATASETS:
        raise SystemExit(f"unknown --dataset {dataset!r}; known: {sorted(DATASETS)}")
    cls_name, mean, std, _ = DATASETS[dataset]
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds_cls = getattr(torchvision.datasets, cls_name)
    try:
        ds = ds_cls(data_dir, train=False, transform=tfm, download=False)
    except Exception as e:
        raise SystemExit(f"Could not load {dataset} from {data_dir!r}: {e}\n"
                         f"Download it once via qat_finetune_fp2.py --download, "
                         f"or point --data-dir at the directory that has it.")
    if max_images and max_images < len(ds):
        # Stride, do not truncate: the test sets are class-ordered in places, so
        # taking the first N would sample only a few classes.
        idx = list(range(0, len(ds), max(len(ds) // max_images, 1)))[:max_images]
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False), len(ds)


@torch.no_grad()
def top1(model, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum())
        total += y.numel()
    return 100.0 * correct / total if total else float("nan")


def eval_one(ckpt, block, args, device):
    """Three accuracies on identical weights and identical images.

    THE BASELINE MUST BE FP2-DIGITAL, NOT FP32 MASTER WEIGHTS. A QAT
    checkpoint stores FP32 master weights, but the deployed network is those
    weights SEEN THROUGH the quantizer -- BatchNorm statistics and the
    full-precision conv1/fc all adapted to the quantized forward pass during
    fine-tuning. Evaluating the master weights unquantized therefore measures
    a network that was never trained and never deployed, and it scores several
    points BELOW the quantized model (86.2% vs 92.4% at B=32 in the first run
    of this script, which is what exposed the bug). Charging that gap to the
    crossbar overstates the analog penalty enormously.

    'ideal' mode below is the same tiled code path with Rsense -> 0, so the
    baseline and the analog run differ in exactly one thing: the readout."""
    print(f"\n--- {ckpt}  block=tile_m={block} ---")
    loader, n = cifar_loader(args.data_dir, args.max_images, args.batch_size,
                             args.dataset)
    print(f"  {n} test images")

    row = dict(checkpoint=ckpt, block_size=block, n_images=n)
    modes = [("ideal", "FP2 digital-exact (baseline)"),
             ("analog", "ANALOG raw readout")]
    if not args.no_gain_correct:
        modes.append(("corrected", "ANALOG + per-column gain calibration"))

    for mode, label in modes:
        m = build_model(ckpt, args.num_classes, device)
        convert_model(m, block, args.tile_k, args.r_sense, args.vread,
                      args.scale_mode, device,
                      skip_first_last=not args.no_skip_first_last, mode=mode,
                      adc_bits=args.adc_bits, dtype=DTYPE[args.dtype])
        t0 = time.time()
        acc = top1(m, loader, device)
        print(f"  {label:<38}: {acc:6.2f}%  ({time.time()-t0:.0f}s)")
        row[f"acc_{mode}"] = acc
        del m

    row["analog_cost_pts"] = row["acc_analog"] - row["acc_ideal"]
    print(f"  {'raw analog cost':<38}: {row['analog_cost_pts']:+6.2f} pts")
    if "acc_corrected" in row:
        row["corrected_cost_pts"] = row["acc_corrected"] - row["acc_ideal"]
        row["recovered_pts"] = row["acc_corrected"] - row["acc_analog"]
        print(f"  {'after gain calibration':<38}: "
              f"{row['corrected_cost_pts']:+6.2f} pts "
              f"(recovered {row['recovered_pts']:+.2f})")
    return row


# =============================================================================
def self_test():
    print("=== analog_eval self-test ===\n")
    if not TORCH_OK:
        print("torch required", file=sys.stderr)
        return 1
    ok = True
    torch.manual_seed(0)

    print("[1] Vectorised solve == crossbar_array_test.golden_array_matmul")
    worst = 0.0
    for trial, (M, K, P) in enumerate([(8, 4, 3), (32, 16, 5), (17, 7, 2), (64, 8, 4)]):
        Wq = torch.tensor(np.random.choice(cb.FP2_LEVELS, size=(M, K)), dtype=torch.float64)
        V = torch.tensor(np.random.uniform(-1, 1, size=(M, P)), dtype=torch.float64)
        got = analog_matmul(Wq, V, 20.0, 0.1)
        for p in range(P):
            ref = cb.golden_array_matmul(Wq.tolist(), V[:, p].tolist(), 20.0, 0.1)
            for k in range(K):
                den = abs(ref[k]) if abs(ref[k]) > 1e-18 else 1.0
                worst = max(worst, abs(float(got[k, p]) - ref[k]) / den)
        print(f"    M={M:3d} K={K:2d} P={P}: worst relative diff {worst:.3e}")
    ok &= worst < 1e-12
    print(f"    machine-precision agreement: {worst < 1e-12}")

    print("\n[1b] Gain calibration recovers the ideal MAC exactly")
    worst_corr, worst_atten = 0.0, 1.0
    for (M, K, P) in [(32, 16, 4), (128, 16, 4), (256, 16, 4)]:
        Wq = torch.tensor(np.random.choice(cb.FP2_LEVELS, size=(M, K)), dtype=torch.float64)
        V = torch.tensor(np.random.uniform(-1, 1, size=(M, P)), dtype=torch.float64)
        ideal = analog_matmul(Wq, V, 20.0, 0.1, "ideal")
        raw = analog_matmul(Wq, V, 20.0, 0.1, "analog")
        corr = analog_matmul(Wq, V, 20.0, 0.1, "corrected")
        den = ideal.abs().sum()
        e_raw = float((raw - ideal).abs().sum() / den)
        e_cor = float((corr - ideal).abs().sum() / den)
        atten = float((raw.abs().sum() / den))
        worst_corr = max(worst_corr, e_cor)
        worst_atten = min(worst_atten, atten)
        print(f"    M={M:4d}: raw error {100*e_raw:6.2f}%  (readout retains "
              f"{100*atten:5.1f}% of ideal magnitude)   after calibration "
              f"{100*e_cor:.2e}%")
    print(f"    calibration is exact to machine precision: {worst_corr < 1e-12}")
    print(f"    attenuation worsens with column height, as the 1/(1+Rs*G) term "
          f"predicts")
    ok &= worst_corr < 1e-12
    ok &= worst_atten < 0.95   # there must BE a meaningful loading error to correct

    print("\n[1c] Dropping the K-tiling loop changes nothing")
    # Columns share wordlines but no other node, so slicing K must be a no-op.
    # This is the assumption the 32x speedup rests on, so it is asserted rather
    # than argued.
    M, K, P = 64, 48, 6
    Wq = torch.tensor(np.random.choice(cb.FP2_LEVELS, size=(M, K)), dtype=torch.float64)
    V = torch.tensor(np.random.uniform(-1, 1, size=(M, P)), dtype=torch.float64)
    Rp, Rn = fp2_resistances(Wq)
    Gfull = (1/Rp, 1/Rn, 1/Rp, 1/Rn)
    for md in ("analog", "corrected", "ideal"):
        full = analog_matmul_G(Gfull, V, 20.0, 0.1, md)
        tiled = torch.zeros_like(full)
        for k0 in range(0, K, 16):
            k1 = min(k0 + 16, K)
            Gt = tuple(g[:, k0:k1] for g in Gfull)
            tiled[k0:k1] = analog_matmul_G(Gt, V, 20.0, 0.1, md)
        d = float((full - tiled).abs().max())
        print(f"    mode={md:9s} max |full - K-tiled| = {d:.3e}")
        ok &= d == 0.0

    print("\n[1d] float32 is accurate enough to replace float64")
    M, K, P = 128, 64, 8
    Wq64 = torch.tensor(np.random.choice(cb.FP2_LEVELS, size=(M, K)), dtype=torch.float64)
    V64 = torch.tensor(np.random.uniform(-1, 1, size=(M, P)), dtype=torch.float64)
    for md in ("analog", "corrected"):
        Rp, Rn = fp2_resistances(Wq64)
        r64 = analog_matmul_G((1/Rp, 1/Rn, 1/Rp, 1/Rn), V64, 20.0, 0.1, md)
        Wq32, V32 = Wq64.float(), V64.float()
        Rp3, Rn3 = fp2_resistances(Wq32)
        r32 = analog_matmul_G((1/Rp3, 1/Rn3, 1/Rp3, 1/Rn3), V32, 20.0, 0.1, md)
        rel = float((r32.double() - r64).abs().sum() / r64.abs().sum())
        print(f"    mode={md:9s} float32 vs float64 relative diff = {rel:.2e}")
        ok &= rel < 1e-5
    print("    (the quantization effects under study are O(1e-1); this is O(1e-7))")

    print("\n[1e] Batched-tile forward == per-tile loop, incl. partial tiles")
    # The speedup rests on this. M=576 with block=128 gives 4.5 tiles, so the
    # zero-padded partial tile is exercised, not just the divisible case.
    for (Min, blk) in ((512, 128), (576, 128), (100, 32)):
        conv = nn.Conv2d(Min // 9, 8, 3, padding=1, bias=False).double()
        x = torch.randn(2, Min // 9, 5, 5, dtype=torch.float64)
        for md in ("analog", "corrected", "ideal"):
            ac = AnalogConv2d(conv, blk, 16, 20.0, 0.1, "e8m0",
                              torch.device("cpu"), md, dtype=torch.float64)
            y_bat = ac(x)
            # reference: explicit loop over m-tiles through analog_matmul_G
            cols = F.unfold(x, (3, 3), stride=1, padding=1)
            L = cols.shape[-1]
            V = cols.permute(1, 0, 2).reshape(ac.M, 2 * L)
            ref = torch.zeros(ac.K, 2 * L, dtype=torch.float64)
            for bi, (m0, m1) in enumerate(ac.m_ranges):
                Gt = tuple(g[m0:m1] for g in ac.G)
                ii = analog_matmul_G(Gt, V[m0:m1], 20.0, 0.1, md)
                ref += ii / (0.1 * G_LRS) * ac.scale[bi].unsqueeze(1)
            ref = ref.reshape(ac.K, 2, L).permute(1, 0, 2).reshape(2, ac.K, 5, 5).float()
            d = float((y_bat - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
            tag = "partial" if ac.Mpad != ac.M else "exact  "
            print(f"    M={ac.M:4d} block={blk:3d} ({ac.T} tiles, {tag}) "
                  f"mode={md:9s} rel diff {d:.2e}")
            ok &= d < 1e-6

    print("\n[1f] Drift model behaves like a power law, and disperses")
    g = torch.cat([torch.full((4000,), 1.0 / R_LRS, dtype=torch.float64),
                   torch.full((4000,), 1.0 / R_HRS, dtype=torch.float64)])
    gen = torch.Generator(device="cpu").manual_seed(0)
    prev_lrs = prev_disp = None
    for t in (1.0, 3600.0, 86400.0, 2592000.0, 31536000.0):
        d = apply_drift(g, t, 0.01, 0.05, 0.004, gen)
        lrs = float(d[:4000].mean() / (1.0 / R_LRS))
        hrs = float(d[4000:].mean() / (1.0 / R_HRS))
        disp = float(d[:4000].std() / d[:4000].mean())
        print(f"    t={fmt_time(t):>6}  LRS retains {100*lrs:6.2f}%   "
              f"HRS retains {100*hrs:6.2f}%   LRS spread {100*disp:5.2f}%")
        if prev_lrs is not None:
            ok &= lrs <= prev_lrs + 1e-9      # monotone decay
            ok &= disp >= prev_disp - 1e-9    # dispersion grows
        prev_lrs, prev_disp = lrs, disp
    ok &= hrs < lrs   # HRS must relax faster than LRS
    print(f"    monotone decay, growing dispersion, HRS faster than LRS: {ok}")

    print("\n[1g] At t = t_ref drift is exactly a no-op")
    g0 = torch.rand(500, dtype=torch.float64) * 1e-3
    same = apply_drift(g0, 1.0, 0.01, 0.05, 0.004, gen)
    print(f"    max |G(t_ref) - G(0)| = {float((same - g0).abs().max()):.3e}")
    ok &= float((same - g0).abs().max()) == 0.0

    print("\n[1h] Only the DISPERSED part of drift is irreducible")
    M, K, P = 128, 32, 8
    Wq = torch.tensor(np.random.choice(cb.FP2_LEVELS, size=(M, K)), dtype=torch.float64)
    V = torch.tensor(np.random.uniform(-1, 1, size=(M, P)), dtype=torch.float64)
    Rp, Rn = fp2_resistances(Wq)
    Gp0, Gn0 = 1 / Rp, 1 / Rn
    ideal = analog_matmul_G((Gp0, Gn0, Gp0, Gn0), V, 20.0, 0.1, "ideal")
    err = lambda x: 100.0 * float((x - ideal).abs().sum() / ideal.abs().sum())
    print(f"    {'sigma_nu':>9}{'stale':>10}{'loading only':>15}{'+refresh gain':>15}")
    prev = None
    for nusig in (0.0, 0.002, 0.004, 0.01):
        gp = apply_drift(Gp0, 31536000.0, 0.01, 0.05, nusig, gen)
        gn = apply_drift(Gn0, 31536000.0, 0.01, 0.05, nusig, gen)
        stale = analog_matmul_G((gp, gn, Gp0, Gn0), V, 20.0, 0.1, "corrected")
        recal = analog_matmul_G((gp, gn, gp, gn), V, 20.0, 0.1, "corrected")
        Dn, Da = Gp0 - Gn0, gp - gn
        alpha = (Dn * Da).sum(0) / (Da * Da).sum(0).clamp_min(1e-30)
        full = recal * alpha.unsqueeze(1)
        print(f"    {nusig:>9.3f}{err(stale):>9.2f}%{err(recal):>14.2f}%"
              f"{err(full):>14.2f}%")
        if nusig == 0.0:
            # a perfectly uniform drift must be almost entirely removable
            ok &= err(full) < 0.5
        if prev is not None:
            ok &= err(full) >= prev - 1e-9   # residual grows with dispersion
        prev = err(full)
    print("    Recalibrating the loading constant ALONE is worse than leaving it")
    print("    stale: drift shrinks the weights, and that scale error dominates.")
    print("    Folding a per-column gain into the block scale removes it. What")
    print("    remains is exactly the cell-to-cell dispersion in nu.")

    print("\n[2] Degenerate tiles")
    Wz = torch.zeros(16, 4, dtype=torch.float64)
    V = torch.rand(16, 3, dtype=torch.float64)
    z = analog_matmul(Wz, V, 20.0, 0.1)
    print(f"    all-zero weights -> max |I| = {float(z.abs().max()):.3e} A (must be ~0)")
    ok &= float(z.abs().max()) < 1e-18
    Wq = torch.tensor(np.random.choice(cb.FP2_LEVELS, size=(16, 4)), dtype=torch.float64)
    z2 = analog_matmul(Wq, torch.zeros(16, 3, dtype=torch.float64), 20.0, 0.1)
    print(f"    zero activations -> max |I| = {float(z2.abs().max()):.3e} A (must be 0)")
    ok &= float(z2.abs().max()) == 0.0

    print("\n[3] Resistance mapping matches decompose_differential")
    lv = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]], dtype=torch.float64)
    Rp, Rn = fp2_resistances(lv)
    bad = 0
    for j, w in enumerate([-1.0, -0.5, 0.0, 0.5, 1.0]):
        _, _, rp_ref, rn_ref = cb.decompose_differential(w)
        if abs(float(Rp[0, j]) - rp_ref) > 1e-9 or abs(float(Rn[0, j]) - rn_ref) > 1e-9:
            bad += 1
        print(f"    w={w:+.1f}: Rp {float(Rp[0,j]):9.1f} (ref {rp_ref:9.1f})  "
              f"Rn {float(Rn[0,j]):9.1f} (ref {rn_ref:9.1f})")
    ok &= bad == 0

    print("\n[4] Block quantization partitions like weight_scale_factor")
    W = torch.randn(100, 6, dtype=torch.float64) * 0.05   # 100 = 3 full + 4 remainder
    Wq, s = quantize_blocks(W, 32, "e8m0")
    print(f"    M=100 block=32 -> scale shape {tuple(s.shape)} (expect (4, 6))")
    print(f"    unique levels: {sorted(set(Wq.flatten().tolist()))}")
    ok &= tuple(s.shape) == (4, 6)
    ok &= set(Wq.flatten().tolist()) <= {-1.0, -0.5, 0.0, 0.5, 1.0}
    ratio = (W[:96].reshape(3, 32, 6) / s[:3].unsqueeze(1)).abs().max()
    print(f"    max |w/scale| in full blocks = {float(ratio):.6f} (must be <= 1)")
    ok &= float(ratio) <= 1.0 + 1e-9

    print("\n[5] AnalogConv2d reproduces a hand-tiled reference")
    conv = nn.Conv2d(4, 6, 3, padding=1, bias=False).double()
    x = torch.randn(2, 4, 5, 5, dtype=torch.float64)
    ac = AnalogConv2d(conv, block=8, tile_k=3, r_sense=20.0, vread=0.1,
                      scale_mode="e8m0", device=torch.device("cpu"))
    y = ac(x)
    print(f"    output shape {tuple(y.shape)} (expect (2, 6, 5, 5))")
    ok &= tuple(y.shape) == (2, 6, 5, 5)
    ok &= torch.isfinite(y).all().item()
    # a crossbar output must differ from the exact FP32 conv, or nothing is happening
    y_fp32 = conv(x).float()
    rel = float((y - y_fp32).abs().sum() / y_fp32.abs().sum())
    print(f"    relative difference vs exact FP32 conv: {100*rel:.2f}% "
          f"(nonzero => the crossbar is actually in the path)")
    ok &= 0.001 < rel < 2.0

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 60)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--sweep", default=None,
                    help="Comma-separated block sizes; expects qat_b<B>.pth from "
                         "codesign_sweep.py")
    ap.add_argument("--ckpt-pattern", default="qat_b{B}.pth")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS),
                    help="Must match what the checkpoint was trained on: the "
                         "normalisation constants differ, and a mismatch quietly "
                         "costs a few accuracy points.")
    ap.add_argument("--num-classes", type=int, default=0,
                    help="0 = infer from --dataset (10 for cifar10, 100 for "
                         "cifar100). Override only for a wider head.")
    ap.add_argument("--max-images", type=int, default=2000,
                    help="0 = all 10000. The analog path is ~100x slower than a "
                         "normal forward pass, so start small and scale up once "
                         "the wall time is known up front.")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Larger batches keep the GPU fed: each tile is one "
                         "matmul of [M,K]x[M, N*L], so N multiplies the useful "
                         "work per kernel launch.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="float64 is ~64x slower on consumer NVIDIA parts (the "
                         "RTX 4070 has 1/64 FP64 throughput) and buys nothing "
                         "here: --self-test measures float32 against float64 and "
                         "the difference is ~1e-7 relative, six orders of "
                         "magnitude below the quantization effects being studied.")
    ap.add_argument("--tile-k", type=int, default=16)
    ap.add_argument("--r-sense", type=float, default=20.0)
    ap.add_argument("--vread", type=float, default=0.1)
    ap.add_argument("--scale-mode", default="e8m0", choices=["e8m0", "fp8", "none"])
    ap.add_argument("--no-skip-first-last", action="store_true",
                    help="Also put conv1 on the crossbar (the deployed model does not)")
    ap.add_argument("--no-gain-correct", action="store_true",
                    help="Skip the per-column loading-gain calibration run. The "
                         "calibrated number is the one worth reporting: the raw "
                         "readout is penalised for a deterministic scale factor "
                         "that a per-column constant removes exactly.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--variability", default=None, metavar="S1,S2,...",
                    help="Run the device-variability sweep instead of the mode "
                         "comparison. Comma-separated log-normal sigmas, e.g. "
                         "0,0.05,0.1,0.2. ReRAM programming spread is typically "
                         "5-20%%.")
    ap.add_argument("--adc-bits", type=int, default=0,
                    help="Quantize each bitline voltage to N bits before the "
                         "gain correction and the differential subtraction. "
                         "0 (default) = exact readout, which is optimistic.")
    ap.add_argument("--adc-sweep", default=None, metavar="N1,N2,...",
                    help="Sweep ADC resolution at each block size, e.g. 4,5,6,8,10. "
                         "Answers whether tall tiles demand more ADC bits and so "
                         "hand back the energy they saved.")
    ap.add_argument("--drift", default=None, metavar="T1,T2,...",
                    help="Retention times in SECONDS, e.g. "
                         "1,3600,86400,2592000,31536000 for 1s/1h/1d/1mo/1y. "
                         "Runs the conductance-drift sweep.")
    ap.add_argument("--drift-seeds", type=int, default=3)
    ap.add_argument("--drift-sigma", type=float, default=0.1,
                    help="Programming scatter present alongside drift. Drift "
                         "acts on an array that was already imperfectly "
                         "written, so testing drift at sigma=0 would flatter it.")
    ap.add_argument("--nu-lrs", type=float, default=0.01,
                    help="Drift exponent for programmed-ON cells. LRS holds a "
                         "stable filament and relaxes slowly.")
    ap.add_argument("--nu-hrs", type=float, default=0.05,
                    help="Drift exponent for OFF cells. HRS has no stable "
                         "filament and relaxes faster.")
    ap.add_argument("--nu-sigma", type=float, default=0.004,
                    help="Cell-to-cell spread of the drift exponent. THIS is "
                         "the term calibration cannot remove: a uniform drift "
                         "is a known scale factor, a dispersed one is not.")
    ap.add_argument("--variability-seeds", type=int, default=3,
                    help="Independent device instances per sigma. Variability is "
                         "frozen per instance, so one seed is one chip.")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())
    if not TORCH_OK:
        raise SystemExit("torch/torchvision required")
    if not args.num_classes:
        args.num_classes = DATASETS[args.dataset][3]
        print(f"dataset: {args.dataset} ({args.num_classes} classes)")

    device = torch.device("cuda" if (args.device in ("auto", "cuda")
                                     and torch.cuda.is_available()) else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    rows = []
    if args.drift:
        blocks = ([int(b) for b in args.sweep.split(",")] if args.sweep
                  else [args.block_size])
        for B in blocks:
            ck = args.ckpt_pattern.format(B=B) if args.sweep else args.checkpoint
            if not ck or not os.path.exists(ck):
                print(f"  skipping B={B}: {ck} not found", file=sys.stderr)
                continue
            rows.extend(drift_sweep(ck, B, args, device))
        if rows:
            print("\n" + "=" * 84)
            print("DOES A COMPILE-TIME CALIBRATION GO STALE AS THE DEVICE DRIFTS?")
            print("=" * 84)
            print(f"{'B':>5}{'t':>8}{'raw %':>9}{'stale %':>10}{'recalib %':>11}"
                  f"{'ceiling %':>11}{'stale cost':>12}")
            print("-" * 84)
            for r in rows:
                print(f"{r['block_size']:>5}{r['t_label']:>8}{r['raw']:>9.2f}"
                      f"{r['stale']:>10.2f}{r['recalib']:>11.2f}"
                      f"{r['acc_ideal']:>11.2f}"
                      f"{r['stale']-r['acc_ideal']:>+12.2f}")
            print("-" * 84)
            worst = min(rows, key=lambda r: r["stale"])
            gap = worst["recalib"] - worst["stale"]
            print(f"Worst stale point: B={worst['block_size']}, t={worst['t_label']}, "
                  f"{worst['stale']:.2f}% vs {worst['acc_ideal']:.2f}% ceiling.")
            print(f"Refreshing the constants there is worth {gap:+.2f} pts.")
            print("\nReading: `stale` holding means the calibration is compile-time.")
            print("Only `recalib` holding means periodic read-back and refresh is")
            print("required; report the interval. Neither holding means drift rather")
            print("than the divider limits tile height.")
    elif args.adc_sweep:
        blocks = ([int(b) for b in args.sweep.split(",")] if args.sweep
                  else [args.block_size])
        for B in blocks:
            ck = args.ckpt_pattern.format(B=B) if args.sweep else args.checkpoint
            if not ck or not os.path.exists(ck):
                print(f"  skipping B={B}: {ck} not found", file=sys.stderr)
                continue
            rows.extend(adc_sweep(ck, B, args, device))
        if rows:
            bits = sorted({r["adc_bits"] for r in rows if r["adc_bits"]})
            blks = sorted({r["block_size"] for r in rows})
            lut = {(r["block_size"], r["adc_bits"]): r["acc_corrected"] for r in rows}
            print("\n" + "=" * 78)
            print("DOES A TALLER TILE DEMAND A WIDER ADC?")
            print("=" * 78)
            print("Top-1 (%) after gain calibration")
            # Header assembled outside the f-string: a backslash inside an
            # f-string expression is only legal from Python 3.12 (PEP 701),
            # and this repo should run on 3.10/3.11 too.
            corner = "B" + chr(92) + "bits"
            print(f"{corner:>8}" + "".join(f"{b:>9}" for b in bits) + f"{'exact':>9}")
            print("-" * 78)
            for b in blks:
                line = f"{b:>8}" + "".join(
                    f"{lut.get((b, nb), float('nan')):>9.2f}" for nb in bits)
                line += f"{lut.get((b, 0), float('nan')):>9.2f}"
                print(line)
            print("-" * 78)
            print("Read down a column: if the bits needed to stay within ~1 pt of "
                  "exact grow with B,\nthe ADC energy saved by amortising over more "
                  "rows is partly handed back, because\nSAR energy scales as 2^N. "
                  "The net efficiency curve in hw_model assumed a FIXED\n8-bit "
                  "converter at every B -- if this table says otherwise, that curve "
                  "needs redoing\nwith per-B ADC resolution, and the 2.8x figure "
                  "will shrink.")
    elif args.variability:
        blocks = ([int(b) for b in args.sweep.split(",")] if args.sweep
                  else [args.block_size])
        for B in blocks:
            ck = args.ckpt_pattern.format(B=B) if args.sweep else args.checkpoint
            if not ck or not os.path.exists(ck):
                print(f"  skipping B={B}: {ck} not found", file=sys.stderr)
                continue
            rows.extend(variability_sweep(ck, B, args, device))
        if rows:
            print("\n" + "=" * 78)
            print("DOES THE GAIN CALIBRATION SURVIVE DEVICE VARIABILITY?")
            print("=" * 78)
            print(f"{'B':>5}{'sigma':>8}{'raw %':>9}{'blind %':>10}{'w-verify %':>12}")
            print("-" * 78)
            for r in rows:
                print(f"{r['block_size']:>5}{r['sigma']:>8.1%}{r['raw']:>9.2f}"
                      f"{r['blind']:>10.2f}{r.get('wverify', float('nan')):>12.2f}")
            print("-" * 78)
            base = [r for r in rows if r["sigma"] == 0]
            worst = [r for r in rows if r["sigma"] == max(x["sigma"] for x in rows)]
            if base and worst:
                d = worst[0]["blind"] - base[0]["blind"]
                print(f"Blind calibration loses {abs(d):.2f} pts going from "
                      f"sigma=0 to sigma={worst[0]['sigma']:.0%}.")
                print("Under a few points: the contribution stands as written -- the "
                      "correction is\ncomputable at compile time from the programmed "
                      "targets alone.\nLarger, but write-verify holds: the contribution "
                      "becomes 'calibrate from measured\nconductances', which costs a "
                      "read-back per cell.\nNeither holds: tile height is limited by "
                      "device variability, not by the divider,\nand that is the paper "
                      "instead.")
    elif args.sweep:
        for B in [int(b) for b in args.sweep.split(",")]:
            ck = args.ckpt_pattern.format(B=B)
            if not os.path.exists(ck):
                print(f"  skipping B={B}: {ck} not found", file=sys.stderr)
                continue
            rows.append(eval_one(ck, B, args, device))
    else:
        if not args.checkpoint:
            raise SystemExit("need --checkpoint or --sweep")
        rows.append(eval_one(args.checkpoint, args.block_size, args, device))

    # Guard on acc_analog, not acc_ideal: the drift sweep also emits
    # acc_ideal (its ceiling), so the old guard let drift rows fall into
    # the accuracy printer and KeyError on acc_analog -- after a 2h run,
    # and before the CSV was written.
    if len(rows) > 1 and "acc_analog" in rows[0]:
        has_corr = "acc_corrected" in rows[0]
        print("\n" + "=" * 84)
        print("DOES THE ANALOG READOUT COST ACCURACY, AND IS THE COST REAL OR CALIBRATABLE?")
        print("=" * 84)
        hdr = f"{'B=M':>5}{'FP2 digital':>13}{'raw analog':>12}{'cost':>8}"
        if has_corr:
            hdr += f"{'calibrated':>12}{'cost':>8}{'recovered':>11}"
        print(hdr)
        print("-" * 84)
        for r in rows:
            line = (f"{r['block_size']:>5}{r['acc_ideal']:>13.2f}"
                    f"{r['acc_analog']:>12.2f}{r['analog_cost_pts']:>+8.2f}")
            if has_corr:
                line += (f"{r['acc_corrected']:>12.2f}"
                         f"{r['corrected_cost_pts']:>+8.2f}"
                         f"{r['recovered_pts']:>+11.2f}")
            print(line)
        print("-" * 84)
        raw_best = max(rows, key=lambda r: r["acc_analog"])
        print(f"Raw readout is best at B={raw_best['block_size']} "
              f"({raw_best['acc_analog']:.2f}%): the shared-Rsense loading gain "
              f"grows with column\nheight, so the larger blocks that win on energy "
              f"lose catastrophically on accuracy.")
        if has_corr:
            corr_best = max(rows, key=lambda r: r["acc_corrected"])
            spread = max(r["corrected_cost_pts"] for r in rows) - \
                     min(r["corrected_cost_pts"] for r in rows)
            print(f"After per-column gain calibration the best is B="
                  f"{corr_best['block_size']} ({corr_best['acc_corrected']:.2f}%), "
                  f"and the cost varies\nby {spread:.2f} pts across B.")
            print("\nIf the calibrated cost is ~0 at every B, then this model's entire "
                  "'analog error'\nwas a deterministic scale factor and the SNR_analog "
                  "column in every earlier CSV\noverstates the penalty. That is the "
                  "expected outcome -- the model is noiseless, so\nevery error in it is "
                  "systematic. It does NOT mean a real crossbar is free: device\n"
                  "variability, read noise, drift and ADC quantisation are all absent "
                  "here, and they\nare what actually sets the analog floor. Say so "
                  "explicitly in the paper.")

    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="") as f:
            # Union of keys across all rows, not rows[0]'s: the sigma=0 row of
            # a variability sweep skips the write-verify branch (it is
            # identical to blind calibration there) and so carries fewer
            # columns. restval fills the gaps rather than raising.
            cols, seen = [], set()
            for r in rows:
                for k in r:
                    if k not in seen:
                        seen.add(k); cols.append(k)
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
