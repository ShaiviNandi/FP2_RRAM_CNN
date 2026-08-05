* Same as short_delay_test.sp (5ns pre-pulse delay), but the dwell is held
* at a constant -1uV instead of exactly 0V -- tests whether avoiding the
* literal 0.0 value (not dithering, not breakpoints) is what matters.
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n -0.000001 5.05n -0.000001 5.1n -1.5 166.05n -1.5 166.1n 0.000000001 245n 0.000000001 245.05n 0.1 247n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 247n
.control
run
wrdata epsilon_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
