* 2x2 ReRAM crossbar test deck, Stanford rram_v_1_0_0.va via ngspice OSDI
*
* ---- BUILD RECIPE (run this on a machine with LLVM 21 available) ----
* 1. Download precompiled OpenVAF-Reloaded (produces OSDI 0.4 modules,
*    matches ngspice-42's OSDI loader, confirmed by `strings $(which ngspice)
*    | grep -i osdi` -> OSDI_VERSION_MAJOR/MINOR symbols present):
*      curl -LO https://github.com/OpenVAF/OpenVAF-Reloaded/releases/download/v24.0.1mob/openvaf-r-v24.0.1mob-linux-x86_64.tar.gz
*      tar xzf openvaf-r-v24.0.1mob-linux-x86_64.tar.gz
*    This binary dynamically links against system libLLVM.so.21.1 (LLVM 21).
*    NOTE: as of this build, Ubuntu 24.04's apt repos only ship LLVM up to
*    llvm-20, so this exact binary will NOT run out of the box on a stock
*    24.04 sandbox/container -- confirmed here by the loader error:
*      "version `LLVM_21.1' not found"
*    On your own machine, either:
*      a) use a distro/container that already has LLVM 21 (e.g. Debian 13,
*         or Arch via the `openvaf-reloaded` AUR package, which pulls its
*         own LLVM), or
*      b) install LLVM 21 manually (apt.llvm.org's install script supports
*         requesting version 21 on top of noble even though it's not in the
*         default repos), or
*      c) fall back to the classic admsXml route instead of OSDI (`apt
*         install adms` works fine on 24.04 -- confirmed here -- but that
*         path means rebuilding ngspice from source with the generated
*         model wired into src/spicelib/devices, which is a bigger lift
*         than a drop-in .osdi file).
*
* 2. Compile the model:
*      export LD_LIBRARY_PATH=$PWD/openvaf-r-v24.0.1mob-linux-x86_64/lib
*      ./openvaf-r-v24.0.1mob-linux-x86_64/bin/openvaf-r rram_v_1_0_0.va -o rram_v_1_0_0.osdi
*
* 3. Point this netlist at the resulting rram_v_1_0_0.osdi (same directory,
*    or give a full path in the .osdi line below).
*
* IMPORTANT: I have not been able to execute steps 2-3 myself in this
* sandbox (blocked at step 1's LLVM version mismatch), so the `.osdi`
* loader line and device instantiation syntax below are written from the
* ngspice OSDI documentation, not verified against a real run. Treat the
* device-card syntax (parameter names / node order) as a starting point --
* run `ngspice -o /dev/null rram_2x2_crossbar.sp` once you have the .osdi
* file and check for card-parsing errors before trusting the results.
* ------------------------------------------------------------------------

.control
osdi rram_v_1_0_0.osdi
.endc

* Wordline drivers (from Phase 2's PWL files -- reuse Dir/Mag PWLs here,
* or simple DC/pulse sources for a first sanity check)
Vwl1 wl1 0 PULSE(0 1.2 0 100p 100p 20n 40n)
Vwl2 wl2 0 PULSE(0 1.2 5n 100p 100p 20n 40n)

* Bitline loads (sense resistors to ground, standard 1T1R/crossbar sensing)
Rsl1 bl1 0 1k
Rsl2 bl2 0 1k

* 2x2 crossbar: one rram_v_1_0_0 instance per cross-point
* (node order per the .va port list: TE, BE)
a11 wl1 bl1 rram_v_1_0_0
a12 wl1 bl2 rram_v_1_0_0
a21 wl2 bl1 rram_v_1_0_0
a22 wl2 bl2 rram_v_1_0_0

.model rram_v_1_0_0 rram_v_1_0_0

.tran 1n 100n
.control
run
wrdata results.txt i(Vwl1) i(Vwl2)
.endc

.end
