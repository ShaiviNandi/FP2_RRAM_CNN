# Multi-Tile FP2/2T2R Accelerator — Integration Notes

Extends the existing FP2-E1M0 crossbar MAC (`fp2_unpacker.v`, `fp2_packer.v` /
`temporal_accumulator_std`) into a 4x4-tile ResNet-18 inference engine.

## Files

| File | What it is |
|---|---|
| `rtl/pe_tile.sv` | Digital shell around one 32x16 2T2R crossbar macro: weight-program path (reuses `fp2_e1m0_to_reram_unpacker`), activation read path, ADC collection, local accumulator |
| `rtl/im2col_stream_engine.sv` | Line-buffer patch streamer for 3x3 / 1x1 convs |
| `rtl/residual_post_proc.sv` | BN-fold + residual add + ReLU + 2x2 pool + requantize |
| `rtl/resnet18_reram_top.sv` | 4x4 tile grid, mesh interconnect, AXI4-Lite/Stream, controller FSM. Also contains `axil_lite_slave`, a small helper module bundled in the same file. |
| `python/benchmark_resnet18.py` | Layer-wise FP32 vs FP2-digital vs 2T2R-analog vs RTL benchmark, built on top of your existing `crossbar_array_test.py` golden model |

## Verified so far

All four `.sv` files elaborate cleanly (`iverilog -g2012 -t null`) against your
real `fp2_unpacker.v`, individually and as the full hierarchy. This confirms
port-list/width consistency and catches gross syntax errors — it does **not**
confirm timing closure, CDC correctness, or full Vivado synthesis (no Vivado
license in this environment). Two Icarus-only notices were left as harmless:
a "constant select" optimization warning (Icarus limitation, not a bug) and
the general caveat that Icarus's `automatic`-variable support inside
non-automatic procedural blocks is limited, which is why loop-scratch
variables in `pe_tile.sv` / `im2col_stream_engine.sv` are declared as static
scope-reused signals instead of `automatic` — functionally identical here
since each iteration fully consumes its value before the next, but worth
knowing if you extend those loops.

The Python harness only had its syntax checked (`py_compile`) — running it
end-to-end needs `torch`/`torchvision` installed locally (not available in
this sandbox) plus a local checkpoint, since pretrained-weight download
isn't reachable from here. `--synthetic` mode runs the full pipeline logic
without either.

## Architecture decisions worth knowing about

- **Weights are programmed, not streamed, into the crossbar.** The unpacker
  decodes FP2 blocks into SET/RESET pulses during a load phase; activations
  are read in parallel every cycle during compute. This is what makes the
  crossbar read a single-cycle O(1) MAC — re-deriving weights from a DAC
  every cycle would defeat the point of analog compute-in-memory.
- **Tile grid mapping:** row index = input-channel/spatial tile, column
  index = output-channel/temporal tile. Default 4x4 grid of 32x16 tiles
  covers 128 input channels x 64 output channels per pass. Layers exceeding
  that (e.g. `layer4`'s 512-channel convs) need multiple LOAD/COMPUTE passes
  — the controller FSM's structure supports this via repeated iteration, but
  the pass-counting logic itself is a stub (`shift_amt_g` is hard-wired to 0;
  see the `TODO`-style comments in `resnet18_reram_top.sv`).
- **`residual_post_proc` is a single time-multiplexed instance**, not
  replicated per column. Fine for a first integration; will bottleneck
  throughput once the tile array is fully utilized — worth revisiting once
  you have real cycle-count targets.

## What's explicitly NOT done here (by design, to keep this reviewable)

- No automatic weight-DMA sequencer or on-chip layer-descriptor table —
  firmware drives both per-layer config and weight loads over AXI4-Lite.
- No residual shortcut staging buffer — assumed external (BRAM/DDR),
  this module only exposes the streaming add port.
- Multi-pass tiling for layers wider than 128/64 channels is architecturally
  supported but not sequenced by the FSM yet.
- `benchmark_resnet18.py`'s Top-1 accuracy path measures weight-quantization
  effects only (not full crossbar-analog effects end-to-end) — full-network
  analog-in-the-loop accuracy would need integrating the per-position
  crossbar model into every layer's actual forward pass, which is a much
  larger runtime cost than the per-layer sampling done here.

## Suggested next step

Given where the FP2/K-means work already stands, the highest-value next
piece is probably wiring an actual DPI-C or behavioral stub for the analog
crossbar boundary (`adc_ipos_i`/`adc_ineg_i` in `pe_tile.sv`) so
`resnet18_reram_top.sv` can run in a real RTL simulation end-to-end, rather
than only elaborating. That would also let `benchmark_resnet18.py`'s
`--rtl-log` path actually get exercised against real simulation output
instead of a hand-built JSON file.
