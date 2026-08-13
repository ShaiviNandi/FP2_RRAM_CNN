* Isolated test: single cell, NO switch at all, identical RESET pulse
* shape/width as the working single-cell case, but delayed to start at
* t=240.05n instead of t=0.05n -- tests whether absolute simulation time
* alone (e.g. ngspice adaptive timestep growth during a long quiet period)
* causes the saturation artifact, fully independent of any switch.
.control
pre_osdi /home/shaiv/fp2_reram/rram_v_1_0_0.osdi
.endc

Vprog wl 0 PWL(0n 0 240.05n 0 240.1n -1.5 401.05n -1.5 401.1n 0 480n 0 480.05n 0.1 482n 0.1)
Rprog bl 0 20.0
N1 wl bl rram_model
.model rram_model rram_v_1_0_0

.tran 50p 482n 0 25p
.control
run
wrdata delayed_pulse_results.txt i(Vprog) v(wl) v(bl)
.endc

.end
