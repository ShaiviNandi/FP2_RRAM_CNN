* 2x2 ReRAM crossbar test deck, Stanford rram_v_1_0_0.va via ngspice OSDI
*
* Corrected after a real test run against a compiled rram_v_1_0_0.osdi:
* - Loading must use `pre_osdi <file>.osdi` inside a .control block (not
*   plain `osdi`), so the model is loaded before the netlist is parsed.
* - Device instances need the `N` prefix, not `a` -- `a` is ngspice's
*   XSPICE code-model prefix, a different mechanism from OSDI devices.
* - A .model card maps a model name to the Verilog-A module name:
*     .model <model_name> <verilog_a_module_name> <params...>
*   Confirmed against: https://ngspice.sourceforge.io/osdi.html and
*   https://openvaf.semimod.de/docs/getting-started/usage/

.control
pre_osdi rram_v_1_0_0.osdi
.endc

* Wordline drivers (swap these for the real PWL files from Phase 3 once
* this sanity-check version runs clean)
Vwl1 wl1 0 PULSE(0 1.2 0 100p 100p 20n 40n)
Vwl2 wl2 0 PULSE(0 1.2 5n 100p 100p 20n 40n)

* Bitline loads (sense resistors to ground, standard 1T1R/crossbar sensing)
Rsl1 bl1 0 1k
Rsl2 bl2 0 1k

* 2x2 crossbar: one rram_v_1_0_0 instance per cross-point
* (node order per the .va port list: TE, BE)
N11 wl1 bl1 rram_model
N12 wl1 bl2 rram_model
N21 wl2 bl1 rram_model
N22 wl2 bl2 rram_model

.model rram_model rram_v_1_0_0

.tran 1n 100n
.control
run
wrdata results.txt i(Vwl1) i(Vwl2)
.endc

.end
