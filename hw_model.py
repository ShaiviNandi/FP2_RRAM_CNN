#!/usr/bin/env python3
"""
hw_model.py
================================================================================
Analytical AREA / POWER / DELAY model for the FP2-E1M0 2T2R ReRAM crossbar
accelerator, with every assumption exposed as a named, overridable constant.

READ THIS BEFORE QUOTING ANY NUMBER THIS PRODUCES
-------------------------------------------------
The numbers below come from three different epistemic tiers, and the report
labels each one so they never get conflated:

  [SIM]   Measured by real ngspice on the actual weight/activation data.
          Currently: array read power and the array's DC operating point.
          Ingested from ngspice_full_sweep.py's CSV via --power-csv.

  [MODEL] Computed from a closed-form expression whose inputs are the
          assumption constants below. Bitline settling delay, energy/MAC,
          throughput. The math is exact; the INPUTS are assumptions.

  [ASSUM] A literature-typical value for a technology this script has no
          access to. Cell area in F^2, ADC area and figure-of-merit, wire
          capacitance. These are placeholders of the right order of
          magnitude, NOT measurements, and NOT PDK-specific.
          Every one is a module-level constant, overridable from the
          CLI (--set NAME=VALUE) or by editing this file.

An [ASSUM]-derived area number is a sanity check on feasibility, not a tapeout
estimate. Given a real PDK, replacing TECH_F_NM, CELL_AREA_F2 and the ADC
constants with measured numbers makes the whole report defensible.

WHY THE ADC DOMINATES (and why that is the real finding here)
-------------------------------------------------------------
In essentially every published analog in-memory-compute macro, the ADC -- not
the resistive array -- sets area and energy. This model reproduces that, and
the reason is structural: the array does M*K multiply-accumulates per read for
one bitline discharge, so its cost amortizes over M*K operations, while the
ADC pays a fixed per-column conversion cost that amortizes over only M. So the
ADC's share grows as K/... no: as the array gets taller (larger M) the array
term amortizes better and the ADC's relative share RISES. That is exactly the
tension the SNR analysis already found from the other side -- larger M gives
better MAC efficiency but worse bitline loading and worse SNR. The two
constraints meet at a tile size, and --sweep-m finds it.

USAGE
-----
    python3 hw_model.py --self-test

    # Model the deployed CIFAR ResNet-18 mapping, array power from real ngspice
    python3 hw_model.py --layers-csv results_cifar_qat_deployed.csv \\
        --power-csv ngspice_full_summary.csv --adc-bits 8 --report hw_report.json

    # Where does the tile-size optimum sit?
    python3 hw_model.py --layers-csv results_cifar_qat_deployed.csv --sweep-m

    # Override any assumption
    python3 hw_model.py --layers-csv results_cifar_qat_deployed.csv \\
        --set TECH_F_NM=28 --set CELL_AREA_F2=25 --set ADC_FOM_FJ_PER_CONV_STEP=10
================================================================================
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import column_mac_test as col
    R_LRS = col.R_FOR_MAGNITUDE[1.0]
    R_MID = col.R_FOR_MAGNITUDE[0.5]
    R_HRS = col.R_FOR_MAGNITUDE[0.0]
except Exception:  # allow standalone use
    R_LRS, R_MID, R_HRS = 542.8, 1099.8, 218587.2


# =============================================================================
# ASSUMPTIONS -- every one of these is a placeholder, to be replaced with
# PDK data before publishing. Units are in the name.
# =============================================================================
ASSUMPTIONS = {
    # ---- technology -------------------------------------------------------
    "TECH_F_NM": 65.0,
    # 1T1R footprint in F^2. A ReRAM select transistor must pass the SET/RESET
    # current, so it is much wider than a minimum-size device and the cell is
    # far above the 6F^2 of an ideal crosspoint. Published 1T1R macros land
    # roughly in the 20-60 F^2 band depending on programming-current target;
    # 40 is a mid-range placeholder.
    "CELL_AREA_F2": 40.0,
    # Peripheral overhead multiplier on raw array area: row decoders, write
    # drivers, precharge, and routing that scale with the array but are not
    # cells. 1.0 = no overhead (unphysical); 1.3-1.6 is typical.
    "ARRAY_PERIPHERY_OVERHEAD": 1.4,

    # ---- ADC (per column, time-multiplexed across COLS_PER_ADC columns) ----
    "ADC_BITS": 8.0,
    # Walden-style figure of merit: energy per conversion = FOM * 2^bits.
    # 5-50 fJ/conv-step spans most published SAR ADCs; 20 is mid-range.
    "ADC_FOM_FJ_PER_CONV_STEP": 20.0,
    "ADC_AREA_MM2": 0.005,        # ~8-bit SAR at 65nm, order of magnitude
    "ADC_CLK_MHZ": 500.0,         # SAR internal clock; conversion takes ADC_BITS cycles
    "COLS_PER_ADC": 8.0,          # column mux ratio -> ADC area amortization

    # ---- how many converters per column ------------------------------------
    # 2 = one ADC per bitline, which the per-column gain calibration REQUIRES:
    # each branch is rescaled by its own (1 + Rs*G_col) before the digital
    # subtraction, and a single differential sense amp has already subtracted
    # them. 1 = conventional differential readout, cheaper, but the correction
    # is then impossible. Set it to 1 for a baseline a reviewer would call
    # fair, and report both.
    "ADCS_PER_COLUMN": 2.0,

    # ---- input drive (per row) --------------------------------------------
    # INPUT_SCHEME selects how a multi-bit activation reaches the wordline:
    #
    #   "dac"       one multi-level DAC per row, all activation bits applied
    #               in a single read. Area is dominated by the 2^bits unit
    #               elements of a binary-weighted or capacitive converter.
    #   "bitserial" one 1-bit driver per row, activation applied over
    #               DAC_BITS cycles with digital shift-add. This is what most
    #               published CIM macros actually do, including NeuroSim's
    #               default, and it replaces the DAC with a level shifter.
    #               Costs DAC_BITS times the read cycles.
    #
    # The previous DAC_AREA_MM2 = 0.0004 (400 um^2/row) was a placeholder and
    # a bad one: at M=128 it made the DAC 68% of tile area, larger than the
    # ADC. Both schemes below are built from unit-element counts instead.
    # DEFAULT IS THE PARALLEL DAC, deliberately: analog_eval's ADC sweep
    # applied the converter to a full multi-bit activation read, so the
    # measured "6 bits required" figure describes THAT scheme. Bit-serial
    # fires the ADC once per activation bit, so it would need its own sweep
    # (each read has a smaller psum range and likely needs fewer bits, so the
    # 4x is an upper bound, not a settled cost). Selecting bit-serial without
    # re-measuring would mix a resolution requirement from one scheme with an
    # energy model from another.
    "INPUT_SCHEME_BITSERIAL": 0.0,     # 1 = bit-serial, 0 = parallel DAC
    "DAC_BITS": 4.0,
    # Unit element of a capacitive/binary-weighted DAC at 65nm: one unit cap
    # plus its switch. A 4-bit converter needs 2^4 of them.
    "DAC_UNIT_AREA_UM2": 3.0,
    # A 1-bit wordline driver is a level shifter plus an inverter chain,
    # order 20 transistors, which at 65nm with routing is a few um^2.
    "WL_DRIVER_AREA_UM2": 8.0,
    "DAC_ENERGY_PJ_PER_ACTIVATION": 0.05,
    # Shift-add accumulation of the bit-serial partial products, per column
    # per activation bit.
    "SHIFTADD_ENERGY_PJ": 0.02,
    "SHIFTADD_AREA_UM2": 120.0,

    # ---- interconnect ----------------------------------------------------
    # Bitline capacitance contributed per cell (junction + wire over one cell
    # pitch). 0.2 fF/cell is a common order of magnitude at 65nm.
    "BITLINE_CAP_FF_PER_CELL": 0.2,
    "BITLINE_RES_OHM_PER_CELL": 0.5,   # metal resistance over one cell pitch

    # ---- operating point (must match the simulated configuration) ----------
    "R_SENSE_OHM": 20.0,
    "V_READ_V": 0.1,

    # ---- ReRAM write cost --------------------------------------------------
    # SET/RESET needs a current pulse orders of magnitude larger than a read,
    # and real macros loop on write-verify. 100 pJ / 100 ns per cell is at the
    # optimistic end of published 1T1R figures; the true cost is worse.
    # Mirrors codesign_sweep.BASE so the two scripts price writes identically.
    "RERAM_WRITE_PJ_PER_CELL": 100.0,
    "RERAM_WRITE_NS_PER_CELL": 100.0,

    # ---- array read power fallback ---------------------------------------
    # Anchor point from this repo's own exhaustive ngspice sweep
    # (ngspice_full_summary.csv): mean 85.8 uW for a 32x16 tile (512 weight
    # pairs) at 41% cell utilisation, over a 16 ns read. Used ONLY when
    # --power-csv is absent. Replace with a measured mean after re-running
    # the sweep at a different operating point.
    "ARRAY_REF_UW_PER_TILE": 85.8,
    "ARRAY_REF_CELLS": 512.0,
    "ARRAY_REF_UTIL": 0.41,

    # ---- digital accumulation --------------------------------------------
    # Energy to add one partial sum into the accumulator, per column, per
    # M-tile beyond the first. Order of a 16-bit add at 65nm.
    "PSUM_ADD_ENERGY_PJ": 0.05,
    "PSUM_ADDER_AREA_MM2": 0.0002,
}


def F_um():
    return ASSUMPTIONS["TECH_F_NM"] / 1000.0


# =============================================================================
# Area
# =============================================================================
def area_model(tile_m, tile_k, n_tiles_physical):
    """Area of ONE physical tile plus its periphery, and the total for
    n_tiles_physical instantiated tiles. 2T2R => 2 cells per synapse."""
    f = F_um()
    cell_um2 = ASSUMPTIONS["CELL_AREA_F2"] * f * f
    cells_per_tile = 2 * tile_m * tile_k
    array_um2 = cells_per_tile * cell_um2 * ASSUMPTIONS["ARRAY_PERIPHERY_OVERHEAD"]

    n_adc = ASSUMPTIONS["ADCS_PER_COLUMN"] * tile_k / ASSUMPTIONS["COLS_PER_ADC"]
    adc_um2 = n_adc * ASSUMPTIONS["ADC_AREA_MM2"] * 1e6

    bitserial = ASSUMPTIONS["INPUT_SCHEME_BITSERIAL"] >= 0.5
    if bitserial:
        dac_um2 = tile_m * ASSUMPTIONS["WL_DRIVER_AREA_UM2"]
        shiftadd_um2 = tile_k * ASSUMPTIONS["SHIFTADD_AREA_UM2"]
    else:
        dac_um2 = (tile_m * ASSUMPTIONS["DAC_UNIT_AREA_UM2"]
                   * (2 ** ASSUMPTIONS["DAC_BITS"]))
        shiftadd_um2 = 0.0
    psum_um2 = tile_k * ASSUMPTIONS["PSUM_ADDER_AREA_MM2"] * 1e6

    tile_um2 = array_um2 + adc_um2 + dac_um2 + psum_um2 + shiftadd_um2
    return dict(
        cells_per_tile=cells_per_tile,
        array_um2=array_um2, adc_um2=adc_um2, dac_um2=dac_um2,
        psum_um2=psum_um2, shiftadd_um2=shiftadd_um2,
        tile_um2=tile_um2,
        array_frac=array_um2 / tile_um2, adc_frac=adc_um2 / tile_um2,
        dac_frac=dac_um2 / tile_um2,
        n_tiles=n_tiles_physical,
        total_mm2=tile_um2 * n_tiles_physical / 1e6,
        adc_count_per_tile=n_adc,
        input_scheme="bit-serial" if bitserial else f"{ASSUMPTIONS['DAC_BITS']:.0f}-bit DAC",
    )


# =============================================================================
# Delay
# =============================================================================
def delay_model(tile_m, tile_k, cell_util):
    """Bitline settling + ADC conversion for one tile read.

    Settling: the bitline node sees R_eff = R_sense in parallel with the
    parallel combination of all M cell resistances on that column, and
    C_bl = M * BITLINE_CAP_FF_PER_CELL. With R_sense at 20 ohm the sense
    resistor dominates R_eff hard, which is exactly why this topology settles
    fast and also exactly why it loads the column and costs SNR -- the same
    trade-off from the other end.

    Resolving to ADC_BITS requires the transient to decay below 1 LSB, i.e.
    ln(2^bits) = bits*ln2 time constants."""
    # A utilization-weighted average cell conductance on one column
    g_avg = (cell_util * (1.0 / R_LRS) + (1.0 - cell_util) * (1.0 / R_HRS))
    g_col = tile_m * g_avg + 1.0 / ASSUMPTIONS["R_SENSE_OHM"]
    r_eff = 1.0 / g_col
    r_wire = ASSUMPTIONS["BITLINE_RES_OHM_PER_CELL"] * tile_m
    c_bl = tile_m * ASSUMPTIONS["BITLINE_CAP_FF_PER_CELL"] * 1e-15

    # Distributed RC of the bitline itself contributes ~R*C/2 (Elmore)
    tau = (r_eff + r_wire / 2.0) * c_bl
    n_tau = ASSUMPTIONS["ADC_BITS"] * math.log(2.0)
    t_settle_s = n_tau * tau

    t_adc_s = ASSUMPTIONS["ADC_BITS"] / (ASSUMPTIONS["ADC_CLK_MHZ"] * 1e6)
    t_tile_s = t_settle_s + t_adc_s
    if ASSUMPTIONS["INPUT_SCHEME_BITSERIAL"] >= 0.5:
        t_tile_s *= ASSUMPTIONS["DAC_BITS"]
    return dict(
        r_eff_ohm=r_eff, c_bl_ff=c_bl * 1e15, tau_ps=tau * 1e12,
        t_settle_ns=t_settle_s * 1e9, t_adc_ns=t_adc_s * 1e9,
        t_tile_ns=t_tile_s * 1e9,
        settle_frac=t_settle_s / t_tile_s,
        macs_per_tile_read=tile_m * tile_k,
        tops_per_tile=2.0 * tile_m * tile_k / t_tile_s / 1e12,  # 2 ops per MAC
    )


# =============================================================================
# Power / energy
# =============================================================================
def reprogram_model(total_logical_tiles, physical_tiles, tile_m, tile_k,
                    batch_size=1):
    """Energy and time to REPROGRAM the array between passes.

    Excluded from every earlier version of this model, and it turns out to
    dominate by four orders of magnitude, so the exclusion was not a rounding
    error -- it was hiding the single largest term in a time-multiplexed
    design.

    If the network does not fit resident, each inference must load every
    logical tile's weights into some physical tile. That is
    total_logical_tiles * 2*tile_m*tile_k cell writes per inference, at
    ~100 pJ and ~100 ns each. Both numbers are optimistic for ReRAM: SET/RESET
    needs a current pulse far larger than a read, and write-verify loops make
    the real figure worse.

    The one escape is BATCHING: hold the weights and push many images through
    before reloading, which divides the term by the batch size. The break-even
    batch -- where reprogramming equals compute -- is reported below, and for
    ResNet-18 at 65nm it is in the thousands of images. For latency-sensitive,
    batch-1 inference (the edge case FP2 targets, and the LLM decode case)
    time-multiplexing a ReRAM array is simply not viable, and the design must
    be fully weight-stationary. That in turn is what forces the move to a
    denser node, since 5464 resident tiles is 412 mm^2 at 65nm.

    Returns per-INFERENCE figures, already divided by batch_size."""
    if physical_tiles >= total_logical_tiles:
        return dict(reloads_per_inference=0.0, cells_written=0,
                    energy_j=0.0, time_s=0.0, resident=True,
                    breakeven_batch=float("nan"))
    cells = total_logical_tiles * 2 * tile_m * tile_k
    e = cells * ASSUMPTIONS["RERAM_WRITE_PJ_PER_CELL"] * 1e-12 / max(batch_size, 1)
    t = cells * ASSUMPTIONS["RERAM_WRITE_NS_PER_CELL"] * 1e-9 / max(batch_size, 1)
    return dict(reloads_per_inference=total_logical_tiles / physical_tiles,
                cells_written=cells, energy_j=e, time_s=t, resident=False,
                breakeven_batch=float("nan"))


def energy_model(tile_m, tile_k, t_tile_s, array_power_w):
    """Energy for one tile read, decomposed. array_power_w should come from
    ngspice ([SIM]); everything else is [MODEL] on [ASSUM] inputs."""
    e_array = array_power_w * t_tile_s
    n_conv = 2 * tile_k  # one conversion per bitline polarity
    e_adc = n_conv * (ASSUMPTIONS["ADC_FOM_FJ_PER_CONV_STEP"] * 1e-15
                      * (2 ** ASSUMPTIONS["ADC_BITS"]))
    e_dac = tile_m * ASSUMPTIONS["DAC_ENERGY_PJ_PER_ACTIVATION"] * 1e-12
    e_psum = tile_k * ASSUMPTIONS["PSUM_ADD_ENERGY_PJ"] * 1e-12
    if ASSUMPTIONS["INPUT_SCHEME_BITSERIAL"] >= 0.5:
        # One read per activation bit, so the ADC fires DAC_BITS times, and
        # each partial product is shift-added. This is the price of deleting
        # the per-row DAC.
        nb = ASSUMPTIONS["DAC_BITS"]
        e_adc *= nb
        e_psum += tile_k * nb * ASSUMPTIONS["SHIFTADD_ENERGY_PJ"] * 1e-12

    e_total = e_array + e_adc + e_dac + e_psum
    macs = tile_m * tile_k
    return dict(
        e_array_pj=e_array * 1e12, e_adc_pj=e_adc * 1e12,
        e_dac_pj=e_dac * 1e12, e_psum_pj=e_psum * 1e12,
        e_total_pj=e_total * 1e12,
        adc_frac=e_adc / e_total, array_frac=e_array / e_total,
        fj_per_mac=e_total * 1e15 / macs,
        power_w=e_total / t_tile_s,
        tops_per_w=2.0 * macs / e_total / 1e12,
    )


# =============================================================================
# Network-level rollup
# =============================================================================
def load_layers_csv(path):
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                layer=r["layer"], M=int(r["M"]), K=int(r["K"]),
                util=float(r["cell_utilization_pct"]) / 100.0,
                snr_analog_db=float(r["snr_analog_db"]),
                snr_digital_db=float(r["snr_digital_db"]),
            ))
    return rows


def load_power_csv(path):
    """Per-layer measured array read power from ngspice_full_sweep.py."""
    import csv
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["layer"]] = float(r["power_mean_uw"]) * 1e-6
    return out


def rollup(layers, power_by_layer, tile_m, tile_k, verbose=True,
           physical_tiles=0, batch_size=1):
    """physical_tiles = 0 means FULLY WEIGHT-STATIONARY: one physical tile is
    instantiated for every logical tile in the network, all weights resident
    at once, nothing ever reloaded. That is the architecture most CIM papers
    implicitly assume for their efficiency numbers, and for ResNet-18 at a
    32x16 tile it needs ~22k tiles and lands at hundreds of mm^2 -- which is
    the model reporting that the assumption does not survive contact with a real
    die, not a bug.

    physical_tiles = N instantiates N tiles and time-multiplexes the network
    over them in ceil(total_tiles / N) passes. Area drops by the reuse factor
    and latency rises by it.

    EXCLUDED, and this matters: the energy and time to REPROGRAM a tile
    between passes. ReRAM writes are orders of magnitude more expensive than
    reads (microseconds and nanojoules per cell against nanoseconds and
    femtojoules), so a design that reloads tiles often is dominated by a term
    this model does not contain. Treat time-multiplexed numbers as an
    optimistic bound until that term is added."""
    total_tiles = 0
    total_macs = 0
    total_energy_j = 0.0
    total_time_s = 0.0
    per_layer = []
    measured_layers = 0

    for L in layers:
        n_m = math.ceil(L["M"] / tile_m)
        n_k = math.ceil(L["K"] / tile_k)
        n_tiles = n_m * n_k
        d = delay_model(tile_m, tile_k, L["util"])

        p_arr = power_by_layer.get(L["layer"])
        if p_arr is None:
            # Fall back to a modelled array power from the same average-
            # conductance approximation the delay model uses. Flagged, so the
            # report can say how much of the total rests on measurement.
            # Empirical fallback, calibrated against this project's own ngspice
            # sweep. The previous closed form assumed every cell sat at its
            # nominal conductance with a full V_read across it, and came out
            # 30x above the measured 85.8 uW for a 32x16 tile at 41%
            # utilisation. Three reasons it was wrong: in 2T2R only the
            # magnitude side of a nonzero weight leaves HRS, half of those sit
            # at R_MID rather than R_LRS, and post-ReLU activations are sparse
            # and well below full scale, so the mean squared wordline voltage
            # is a small fraction of V_read^2.
            #
            # Scaling the measured point by cell count and utilisation is
            # crude, but it is anchored to simulation instead of to an
            # assumption, and it is only ever a fallback: pass --power-csv and
            # this branch is not taken.
            ref_uw_per_cellpair = (ASSUMPTIONS["ARRAY_REF_UW_PER_TILE"]
                                   / ASSUMPTIONS["ARRAY_REF_CELLS"])
            p_arr = (ref_uw_per_cellpair * 1e-6 * tile_m * tile_k
                     * (L["util"] / ASSUMPTIONS["ARRAY_REF_UTIL"]))
            measured = False
        else:
            measured = True
            measured_layers += 1

        e = energy_model(tile_m, tile_k, d["t_tile_ns"] * 1e-9, p_arr)
        macs = L["M"] * L["K"]
        # every M-tile must be read and its partial sums accumulated
        layer_energy = e["e_total_pj"] * 1e-12 * n_tiles
        layer_time = d["t_tile_ns"] * 1e-9 * n_m   # K-tiles run in parallel
        total_tiles += n_tiles
        total_macs += macs
        total_energy_j += layer_energy
        total_time_s += layer_time
        per_layer.append(dict(
            layer=L["layer"], M=L["M"], K=L["K"], tiles=n_tiles,
            macs=macs, util_pct=100 * L["util"],
            t_tile_ns=d["t_tile_ns"], layer_time_us=layer_time * 1e6,
            energy_nj=layer_energy * 1e9,
            fj_per_mac=e["fj_per_mac"], adc_energy_frac=e["adc_frac"],
            array_power_uw=p_arr * 1e6, power_measured=measured,
            snr_analog_db=L["snr_analog_db"],
        ))

    n_phys = total_tiles if physical_tiles <= 0 else min(physical_tiles, total_tiles)
    reuse = total_tiles / n_phys if n_phys else 1.0
    if physical_tiles > 0:
        # Serialising over fewer tiles multiplies latency by the reuse factor.
        # Energy is unchanged: the same tile-reads happen either way.
        total_time_s *= reuse
    rp = reprogram_model(total_tiles, n_phys, tile_m, tile_k, batch_size)
    if rp["energy_j"] > 0:
        rp["breakeven_batch"] = rp["energy_j"] * max(batch_size, 1) / total_energy_j
    total_energy_j += rp["energy_j"]
    total_time_s += rp["time_s"]

    area = area_model(tile_m, tile_k, n_phys)
    return dict(per_layer=per_layer, area=area, reprogram=rp,
                total_tiles=total_tiles, physical_tiles=n_phys, reuse_factor=reuse,
                total_macs=total_macs,
                total_energy_nj=total_energy_j * 1e9,
                total_time_us=total_time_s * 1e6,
                pj_per_mac=total_energy_j * 1e12 / total_macs if total_macs else float("nan"),
                tops=2.0 * total_macs / total_time_s / 1e12 if total_time_s else float("nan"),
                tops_per_w=2.0 * total_macs / total_energy_j / 1e12 if total_energy_j else float("nan"),
                tops_per_mm2=(2.0 * total_macs / total_time_s / 1e12) / area["total_mm2"]
                if total_time_s and area["total_mm2"] else float("nan"),
                measured_layers=measured_layers, total_layers=len(layers))


def print_report(R, tile_m, tile_k, layers):
    A, pl = R["area"], R["per_layer"]
    print("\n" + "=" * 92)
    print(f"FP2-E1M0 2T2R ReRAM ACCELERATOR -- AREA / POWER / DELAY")
    print(f"tile {tile_m}x{tile_k}   node {ASSUMPTIONS['TECH_F_NM']:.0f} nm   "
          f"ADC {ASSUMPTIONS['ADC_BITS']:.0f} bit   Rsense {ASSUMPTIONS['R_SENSE_OHM']:.0f} ohm   "
          f"Vread {ASSUMPTIONS['V_READ_V']:.2f} V")
    print("=" * 92)

    print(f"\n{'Layer':<24}{'tiles':>7}{'util%':>8}{'t_tile(ns)':>12}"
          f"{'E(nJ)':>10}{'fJ/MAC':>10}{'ADC%E':>8}{'P_arr(uW)':>11}{'src':>5}")
    print("-" * 92)
    for r in pl:
        print(f"{r['layer']:<24}{r['tiles']:>7}{r['util_pct']:>8.1f}{r['t_tile_ns']:>12.2f}"
              f"{r['energy_nj']:>10.2f}{r['fj_per_mac']:>10.1f}"
              f"{100*r['adc_energy_frac']:>8.1f}{r['array_power_uw']:>11.2f}"
              f"{'SIM' if r['power_measured'] else 'MOD':>5}")
    print("-" * 92)

    d = delay_model(tile_m, tile_k, np.mean([L["util"] for L in layers]))
    print(f"\nDELAY   [MODEL]")
    print(f"  bitline R_eff {d['r_eff_ohm']:.2f} ohm   C_bl {d['c_bl_ff']:.2f} fF   "
          f"tau {d['tau_ps']:.2f} ps")
    print(f"  settling {d['t_settle_ns']:.3f} ns ({100*d['settle_frac']:.1f}% of tile time)   "
          f"ADC {d['t_adc_ns']:.3f} ns")
    print(f"  tile read {d['t_tile_ns']:.3f} ns  ->  {d['tops_per_tile']:.3f} TOPS per tile")
    print(f"  whole-network latency (K-tiles parallel, M-tiles serial): "
          f"{R['total_time_us']:.2f} us")

    print(f"\nAREA    [MODEL on ASSUM]   input drive: {A['input_scheme']}")
    print(f"  per tile: array {A['array_um2']:.0f} um2 ({100*A['array_frac']:.1f}%)   "
          f"ADC {A['adc_um2']:.0f} um2 ({100*A['adc_frac']:.1f}%)   "
          f"drivers {A['dac_um2']:.0f} um2 ({100*A['dac_frac']:.1f}%)   "
          f"psum {A['psum_um2']:.0f} um2   shift-add {A['shiftadd_um2']:.0f} um2")
    print(f"  {ASSUMPTIONS['ADCS_PER_COLUMN']:.0f} ADC per column "
          f"({'gain calibration possible' if ASSUMPTIONS['ADCS_PER_COLUMN'] >= 2 else 'single differential sense -- calibration NOT possible'})")
    print(f"  {A['cells_per_tile']} ReRAM cells/tile, {A['adc_count_per_tile']:.1f} ADC/tile "
          f"(mux {ASSUMPTIONS['COLS_PER_ADC']:.0f}:1)")
    if R.get("reuse_factor", 1.0) > 1.0:
        print(f"  {R['total_tiles']} logical tiles time-multiplexed over "
              f"{R['physical_tiles']} physical tiles (reuse {R['reuse_factor']:.1f}x) "
              f"-> {A['total_mm2']:.3f} mm2")
        print(f"  Reprogramming between passes IS now costed -- see below.")
    else:
        print(f"  {R['total_tiles']} tiles, all weights resident (fully weight-stationary) "
              f"-> {A['total_mm2']:.3f} mm2")
        if A["total_mm2"] > 100:
            print(f"  A {A['total_mm2']:.0f} mm2 die is not manufacturable at sane cost. "
                  f"Re-run with --physical-tiles to time-multiplex, or raise --tile-m.")

    print(f"\nENERGY  [SIM array + MODEL periphery]")
    src = f"{R['measured_layers']}/{R['total_layers']} layers with ngspice-measured array power"
    print(f"  source: {src}")
    print(f"  total {R['total_energy_nj']:.1f} nJ for {R['total_macs']:,} MACs "
          f"= {R['pj_per_mac']:.4f} pJ/MAC")
    adc_share = np.mean([r["adc_energy_frac"] for r in pl])
    print(f"  ADC is {100*adc_share:.1f}% of energy on average")

    rp = R.get("reprogram", {})
    if rp and not rp.get("resident", True):
        print(f"\nREPROGRAMMING  [MODEL on ASSUM]  -- excluded from every "
              f"earlier version of this model")
        print(f"  {rp['cells_written']:,} cell writes per inference "
              f"({rp['reloads_per_inference']:.1f} reloads of the physical array)")
        print(f"  {rp['energy_j']*1e6:.1f} uJ and {rp['time_s']*1e3:.2f} ms per "
              f"inference at batch {R.get('batch_size',1)}")
        if rp.get("breakeven_batch") == rp.get("breakeven_batch"):
            print(f"  break-even batch (reprogram == compute): "
                  f"{rp['breakeven_batch']:,.0f} images")
            print(f"  Below that batch the array is a WRITE engine, not a "
                  f"compute engine. Batch-1 edge\n  inference cannot "
                  f"time-multiplex ReRAM; it must be fully weight-stationary, "
                  f"which is\n  what forces a denser node.")
    elif rp.get("resident"):
        print(f"\nREPROGRAMMING: none -- all weights resident, written once at "
              f"deployment.")

    print(f"\nEFFICIENCY")
    print(f"  {R['tops']:.2f} TOPS   {R['tops_per_w']:.1f} TOPS/W   "
          f"{R['tops_per_mm2']:.2f} TOPS/mm2")
    print("=" * 92)
    print("[SIM] measured by ngspice. [MODEL] closed form. [ASSUM] literature-typical "
          "placeholder -- replace with PDK data before publishing. Area and delay rest "
          "on [ASSUM]; the array power term does not.")


def sweep_m(layers, power_by_layer, tile_k, m_values=(8, 16, 32, 64, 128, 256),
            physical_tiles=0):
    print(f"\n{'tile_M':>7}{'tiles':>8}{'t_tile(ns)':>12}{'pJ/MAC':>10}{'ADC%E':>8}"
          f"{'area(mm2)':>11}{'TOPS/W':>9}{'TOPS/mm2':>10}")
    print("-" * 76)
    out = []
    for m in m_values:
        R = rollup(layers, power_by_layer, m, tile_k, verbose=False)
        d = delay_model(m, tile_k, np.mean([L["util"] for L in layers]))
        adc = np.mean([r["adc_energy_frac"] for r in R["per_layer"]])
        print(f"{m:>7}{R['total_tiles']:>8}{d['t_tile_ns']:>12.3f}{R['pj_per_mac']:>10.4f}"
              f"{100*adc:>8.1f}{R['area']['total_mm2']:>11.3f}{R['tops_per_w']:>9.1f}"
              f"{R['tops_per_mm2']:>10.2f}")
        out.append(dict(tile_m=m, tiles=R["total_tiles"], t_tile_ns=d["t_tile_ns"],
                        pj_per_mac=R["pj_per_mac"], adc_energy_frac=adc,
                        area_mm2=R["area"]["total_mm2"], tops_per_w=R["tops_per_w"],
                        tops_per_mm2=R["tops_per_mm2"]))
    print("-" * 76)
    print("Energy/MAC falls with tile_M because the ADC's fixed per-column cost amortizes "
          "over more rows. The SNR data pushes the other way (taller columns load the "
          "bitline harder). The optimal tile size is where those curves cross, which is "
          "an argument this table alone cannot settle -- pair it with a --tile-m sweep of "
          "benchmark_resnet18.py to see SNR at each M.")
    return out


# =============================================================================
# Self-test
# =============================================================================
def self_test():
    print("=== hw_model self-test ===\n")
    ok = True

    print("[1] Dimensional sanity of the area model")
    a = area_model(32, 16, 1)
    f = F_um()
    expect_cell = ASSUMPTIONS["CELL_AREA_F2"] * f * f
    print(f"    F={ASSUMPTIONS['TECH_F_NM']} nm -> 1 cell = {expect_cell:.4f} um2 "
          f"({ASSUMPTIONS['CELL_AREA_F2']:.0f} F^2)")
    print(f"    32x16 tile: {a['cells_per_tile']} cells, array {a['array_um2']:.1f} um2, "
          f"total {a['tile_um2']:.1f} um2")
    manual = 2 * 32 * 16 * expect_cell * ASSUMPTIONS["ARRAY_PERIPHERY_OVERHEAD"]
    print(f"    array area matches hand calculation: {abs(a['array_um2']-manual) < 1e-9}")
    ok &= abs(a["array_um2"] - manual) < 1e-9
    ok &= a["cells_per_tile"] == 2 * 32 * 16

    print("\n[2] Delay: settling must scale with M, ADC term must not")
    for m in (16, 32, 64, 128):
        d = delay_model(m, 16, 0.42)
        print(f"    M={m:4d}  R_eff {d['r_eff_ohm']:6.2f} ohm  C_bl {d['c_bl_ff']:5.1f} fF  "
              f"settle {d['t_settle_ns']:.4f} ns  ADC {d['t_adc_ns']:.3f} ns")
    d1, d2 = delay_model(16, 16, 0.42), delay_model(128, 16, 0.42)
    print(f"    settling grew with M: {d2['t_settle_ns'] > d1['t_settle_ns']}")
    print(f"    ADC time independent of M: {abs(d2['t_adc_ns']-d1['t_adc_ns']) < 1e-12}")
    ok &= d2["t_settle_ns"] > d1["t_settle_ns"]
    ok &= abs(d2["t_adc_ns"] - d1["t_adc_ns"]) < 1e-12

    print("\n[3] Energy: ADC share must fall as M grows (fixed cost, more MACs)")
    prev = None
    for m in (16, 32, 64, 128, 256):
        d = delay_model(m, 16, 0.42)
        e = energy_model(m, 16, d["t_tile_ns"] * 1e-9, 50e-6)
        print(f"    M={m:4d}  {e['fj_per_mac']:8.1f} fJ/MAC   ADC {100*e['adc_frac']:5.1f}% "
              f"of energy   {e['tops_per_w']:7.1f} TOPS/W")
        if prev is not None:
            ok &= e["adc_frac"] < prev
        prev = e["adc_frac"]
    print(f"    monotonic decrease: {ok}")

    print("\n[4] Order-of-magnitude check against published analog CIM macros")
    d = delay_model(32, 16, 0.42)
    e = energy_model(32, 16, d["t_tile_ns"] * 1e-9, 50e-6)
    print(f"    this model: {e['tops_per_w']:.1f} TOPS/W at {ASSUMPTIONS['ADC_BITS']:.0f}-bit ADC")
    print(f"    published analog CIM macros mostly report 10-100 TOPS/W for low-bit")
    print(f"    weights. In range: {1 <= e['tops_per_w'] <= 1000}")
    print(f"    (In-range is a NECESSARY not sufficient check -- it catches unit errors,")
    print(f"     it does not validate the ASSUM constants.)")
    ok &= 1 <= e["tops_per_w"] <= 1000

    print("\n[5] Energy conservation: components must sum to the total")
    parts = e["e_array_pj"] + e["e_adc_pj"] + e["e_dac_pj"] + e["e_psum_pj"]
    print(f"    sum of parts {parts:.4f} pJ vs total {e['e_total_pj']:.4f} pJ")
    ok &= abs(parts - e["e_total_pj"]) < 1e-9

    print("\n" + "=" * 60)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 60)
    return 0 if ok else 1


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--layers-csv", default=None,
                    help="Per-layer CSV from benchmark_resnet18.py (needs M, K, "
                         "cell_utilization_pct)")
    ap.add_argument("--power-csv", default=None,
                    help="Per-layer measured array power from ngspice_full_sweep.py. "
                         "Without it, array power is MODELLED and flagged MOD in the table.")
    ap.add_argument("--tile-m", type=int, default=32)
    ap.add_argument("--tile-k", type=int, default=16)
    ap.add_argument("--adc-bits", type=float, default=None)
    ap.add_argument("--physical-tiles", type=int, default=0,
                    help="Instantiate only N physical tiles and time-multiplex the network "
                         "over them. 0 (default) = fully weight-stationary, one physical tile "
                         "per logical tile. Reprogramming cost between passes is NOT modelled.")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Images per weight load. Only matters with "
                         "--physical-tiles: reprogramming energy is divided by "
                         "it. Batch 1 is the edge/decode case FP2 targets.")
    ap.add_argument("--adcs-per-column", type=float, default=None,
                    help="2 (default) = one ADC per bitline, required by the "
                         "gain calibration. 1 = conventional differential "
                         "sense, cheaper, calibration impossible. Use 1 for the "
                         "baseline a reviewer will ask for.")
    ap.add_argument("--parallel-dac", action="store_true",
                    help="Use a per-row multi-level DAC instead of the default "
                         "bit-serial 1-bit driver.")
    ap.add_argument("--sweep-m", action="store_true")
    ap.add_argument("--report", default=None, help="Write the full result as JSON")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Override any assumption constant, e.g. --set TECH_F_NM=28")
    args = ap.parse_args()

    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects NAME=VALUE, got {kv!r}")
        name, val = kv.split("=", 1)
        if name not in ASSUMPTIONS:
            raise SystemExit(f"Unknown assumption {name!r}. Known: {sorted(ASSUMPTIONS)}")
        ASSUMPTIONS[name] = float(val)
    if args.adc_bits is not None:
        ASSUMPTIONS["ADC_BITS"] = args.adc_bits
    if args.adcs_per_column is not None:
        ASSUMPTIONS["ADCS_PER_COLUMN"] = args.adcs_per_column
    if args.parallel_dac:
        ASSUMPTIONS["INPUT_SCHEME_BITSERIAL"] = 0.0

    if args.self_test:
        raise SystemExit(self_test())
    if not args.layers_csv:
        raise SystemExit("Need --layers-csv (or --self-test). See --help.")

    layers = load_layers_csv(args.layers_csv)
    power_by_layer = load_power_csv(args.power_csv) if args.power_csv else {}
    if args.power_csv:
        print(f"Loaded ngspice-measured array power for {len(power_by_layer)} layers.")
        # The sweep records power for whatever tile height it ran at. Applying
        # a 32-row measurement to a 128-row tile understates the array term by
        # the cell-count ratio. It is a ~1% error once the ADC is right-sized,
        # but say so rather than let it pass silently.
        if args.tile_m != 32:
            print(f"  NOTE: ngspice_full_sweep defaults to --tile-m 32. If that "
                  f"CSV was produced at a different tile height than the "
                  f"--tile-m {args.tile_m} requested here, the array term is off "
                  f"by the cell-count ratio ({args.tile_m/32:.0f}x). Re-run the "
                  f"sweep with --tile-m {args.tile_m} to remove the caveat.",
                  file=sys.stderr)
    else:
        print("WARNING: no --power-csv. The array term falls back to an "
              "empirical scaling of this project's measured 32x16 tile, which "
              "is anchored to simulation but not to this configuration. Pass "
              "--power-csv ngspice_full_summary.csv for a measured array term.",
              file=sys.stderr)

    if args.sweep_m:
        sweep = sweep_m(layers, power_by_layer, args.tile_k,
                        physical_tiles=args.physical_tiles)
        if args.report:
            with open(args.report, "w") as f:
                json.dump(dict(assumptions=ASSUMPTIONS, sweep_m=sweep), f, indent=2)
            print(f"\nWrote {args.report}")
        return

    R = rollup(layers, power_by_layer, args.tile_m, args.tile_k,
               physical_tiles=args.physical_tiles, batch_size=args.batch_size)
    R["batch_size"] = args.batch_size
    print_report(R, args.tile_m, args.tile_k, layers)

    if args.report:
        with open(args.report, "w") as f:
            json.dump(dict(assumptions=ASSUMPTIONS, tile_m=args.tile_m,
                           tile_k=args.tile_k, result=R), f, indent=2, default=float)
        print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
