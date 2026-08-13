* 2x2 ReRAM crossbar: two-stage transient (PROGRAM the weight, then COMPUTE)
*
* ============================================================================
* IMPORTANT DESIGN ASSUMPTIONS -- read before trusting any numbers out of this
* ============================================================================
* 1. WEIGHT STORAGE: each cell stores |weight| as a conductance level via
*    SET/RESET pulse programming. Sign (Dir bit) is NOT stored in the cell
*    itself -- this model uses a single device per weight, so sign has to be
*    realized some other way. I chose: flip the polarity of the COMPUTE-phase
*    read voltage by the Dir bit. This is a real design choice with real
*    consequences (it means the "activation" going into a negative-weight
*    cell is literally inverted, not just algebraically signed after the
*    fact) -- if your architecture instead uses a differential pair of
*    columns for sign (common in real crossbar accelerators), this whole
*    netlist needs restructuring, not just parameter tweaks. Flag this to
*    whoever owns the crossbar addressing scheme before trusting this.
*
* 2. MULTI-LEVEL PROGRAMMING (Mag=0 / 0.5 / 1.0): the model initializes at
*    gap_ini = gap_min = LRS (fully conductive) by default. To get 3 distinct
*    levels I'm using 3 different RESET pulse widths from that fully-SET
*    starting point (longer RESET -> larger gap -> higher resistance ->
*    smaller weight magnitude). THE EXACT WIDTHS BELOW ARE NOT CALIBRATED --
*    a real flow would do write-verify (program, measure resulting
*    resistance, adjust pulse width/amplitude, repeat) to hit specific
*    target resistances. Treat R1/R2 in the compute-phase results as
*    illustrative of the two-stage MECHANISM working, not as validated
*    absolute weight values, until you've actually calibrated against your
*    target R_HRS/R_LRS/R_mid specs.
*
* 3. Pulse amplitude uses model parameter V0=0.25 (the sinh drive term's
*    voltage scale, pulled directly from rram_v_1_0_0.va, not guessed) with
*    a margin multiplier -- again a starting point, not a calibrated value.
*
* 4. Programming pulse width (20n) matches the model's own declared
*    `pulse_width` parameter (line 116 of the .va file) -- that's a real
*    hint from the model author, not an arbitrary choice.
* ============================================================================

.control
pre_osdi rram_v_1_0_0.osdi
.endc

* ---- Cell 1 (weight 1): target magnitude = 1.0, Dir1 = 0 (positive) ----
* Program phase (0-20n): hold at 0V, i.e. leave the cell at its default
*   gap_ini = gap_min (fully LRS) init state == "already at magnitude 1.0".
*   (A real weight that's genuinely 1.0 needs no RESET at all from this
*   model's default init state -- this is a real consequence of gap_ini
*   being set equal to gap_min in the .va file, not a shortcut I'm taking.)
* Compute phase (100n-120n): +0.1V read (positive weight -> positive read).
Vprog1 wl1 0 PWL(0 0 99.95n 0 100.05n 0.1 120n 0.1)

* ---- Cell 2 (weight 2): target magnitude = 0.5, Dir2 = 1 (negative) ----
* Program phase (0-100n): STRONGER -1.5V RESET pulse (6x model's V0=0.25
*   parameter), held for 100n instead of 20n -- testing whether the
*   original -0.6V/20n was simply too weak/short to move the gap.
* Compute phase (100n-120n): -0.1V read (negative weight -> read polarity
*   flipped, per the sign-realization assumption documented above).
Vprog2 wl2 0 PWL(0 -8 99.9n -8 99.95n 0 100.05n -0.1 120n -0.1)

Rsl1 bl1 0 1k
Rsl2 bl2 0 1k

N11 wl1 bl1 rram_model
N22 wl2 bl2 rram_model

.model rram_model rram_v_1_0_0

.tran 100p 120n
.control
run
wrdata results_2stage_divider.txt i(Vprog1) i(Vprog2) v(wl1) v(wl2) v(bl1) v(bl2)
.endc

.end
