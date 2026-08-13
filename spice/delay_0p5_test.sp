* Isolated test: single cell, NO switch, 0.5ns pre-pulse delay --
* binary-search step between 0.05ns (known working) and 5ns (known failing).
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n 0 0.55n 0 0.6n -1.5 161.55n -1.5 161.6n 0 240.5n 0 240.55n 0.1 242.5n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 242.5n
.control
run
wrdata delay_0p5_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
