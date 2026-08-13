* Same as short_delay_test.sp (5ns pre-pulse delay), but instead of any
* nonzero dither, insert dummy PWL points at the SAME 0V value every 0.5ns
* during the dwell -- tests whether it's breakpoint FREQUENCY (forcing the
* solver to refresh/reset its integration step-history) that matters, not
* any actual nonzero perturbation.
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n 0 0.5n 0 1n 0 1.5n 0 2n 0 2.5n 0 3n 0 3.5n 0 4n 0 4.5n 0 5.05n 0 5.1n -1.5 166.05n -1.5 166.1n 0 245n 0 245.05n 0.1 247n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 247n
.control
run
wrdata breakpoint_only_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
