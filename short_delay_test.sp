* Isolated test: single cell, NO switch, SHORT pre-pulse delay (5ns
* instead of 240ns) -- tests whether it's the LENGTH of the idle period
* before the pulse that matters (relaxation/retention dynamics) rather
* than simply being delayed at all.
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n 0 5.05n 0 5.1n -1.5 166.05n -1.5 166.1n 0 245n 0 245.05n 0.1 247n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 247n
.control
run
wrdata short_delay_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
