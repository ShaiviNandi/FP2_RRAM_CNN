* Isolated test: single cell, NO switch, 0.1ns pre-pulse delay.
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n 0 0.15n 0 0.2n -1.5 161.15n -1.5 161.2n 0 240.1n 0 240.15n 0.1 242.1n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 242.1n
.control
run
wrdata delay_0p1_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
