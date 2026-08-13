* Same as short_delay_test.sp (5ns pre-pulse delay), but the dwell is NOT
* a perfectly flat 0V hold -- it dithers with a tiny (1mV) triangular
* wiggle instead, to test whether avoiding a truly static DC-like segment
* avoids whatever pathology causes the saturation artifact.
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n 0 1n -0.001 2n 0 3n -0.001 4n 0 5.05n 0 5.1n -1.5 166.05n -1.5 166.1n 0 245n 0 245.05n 0.1 247n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 247n
.control
run
wrdata dither_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
