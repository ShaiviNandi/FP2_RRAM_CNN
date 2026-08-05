// =============================================================================
// pe_tile.sv
// -----------------------------------------------------------------------------
// Processing-Engine tile: the fully synthesizable DIGITAL shell around one
// physical 2T2R ReRAM crossbar macro (TILE_M rows x TILE_K columns).
//
// The crossbar macro itself (analog: cell conductance, wordline drivers,
// differential bitline sense amps / ADCs) is NOT part of this module and is
// NOT synthesizable RTL -- it is a mixed-signal hard macro instantiated
// alongside this shell in the physical design, or replaced by a bit-accurate
// behavioral stub / DPI model in RTL simulation. This module owns everything
// on the digital side of that boundary:
//
//   1) WEIGHT PROGRAM PATH  (persistent, low-rate)
//      FP2-E1M0 packed blocks (S,E,C1,C2) -> fp2_e1m0_to_reram_unpacker ->
//      Dir/Mag1/Mag0 per weight -> local weight_mem shadow copy used to
//      drive SET/RESET program pulses into the crossbar (prog_* bus below).
//      Weights are programmed ONCE per weight-stationary tile load, not
//      re-driven every compute cycle -- this is what makes the crossbar
//      read (below) a single-cycle O(1) MAC across all TILE_M rows.
//
//   2) ACTIVATION READ PATH (streaming, high-rate)
//      Every cycle, TILE_M quantized activation drive-codes are applied in
//      parallel to the crossbar wordlines (act_code_i). After bitline
//      settling, the analog macro returns TILE_K differential ADC codes
//      (adc_ipos_i - adc_ineg_i), sampled synchronously by this shell.
//
//   3) LOCAL ACCUMULATOR
//      Digital shift-and-add reduction across (a) SPATIAL tiles -- multiple
//      physical TILE_M-row tiles covering one logical input-channel depth
//      wider than TILE_M -- and (b) TEMPORAL tiles -- multiple activation
//      passes needed when the logical output-channel count exceeds TILE_K,
//      or when a layer's contraction dimension is streamed over several
//      cycles. `shift_amt_i` aligns partial-sum scale between passes taken
//      at different quantization/exponent tiers (0 in the common case).
//
// Tile is bounded to TILE_M=32 x TILE_K=16 by default to keep shared-Rsense
// bitline loading error under ~2.5% (see topology_sweep.py / crossbar_cli.py
// golden-model results in this repo).
// =============================================================================
`default_nettype none

module pe_tile #(
    parameter int TILE_M      = 32,   // crossbar rows    (bounded for Rsense error)
    parameter int TILE_K      = 16,   // crossbar columns (bounded for Rsense error)
    parameter int ACT_WIDTH   = 4,    // activation drive-code width (per row)
    parameter int ADC_WIDTH   = 8,    // single-ended ADC code width (per column)
    parameter int ACC_WIDTH   = 24,   // local accumulator width (per column)
    parameter int ROWPAIR_ADDR_W = $clog2((TILE_M+1)/2),
    parameter int COL_ADDR_W     = $clog2(TILE_K)
) (
    input  wire                          clk,
    input  wire                          rst_n,

    // ---------------------------------------------------------------
    // Weight-program port: FP2-E1M0 packed block, one row-pair/column
    // per write. Matches fp2_e1m0_to_reram_unpacker's block format
    // (shared sign S / shared exponent E across a 2-weight group,
    // per-weight active flags C1, C2).
    // ---------------------------------------------------------------
    input  wire                          wr_en_i,
    input  wire                          wr_S_i,
    input  wire                          wr_E_i,
    input  wire                          wr_C1_i,
    input  wire                          wr_C2_i,
    input  wire [ROWPAIR_ADDR_W-1:0]     wr_rowpair_i,   // selects rows {2i, 2i+1}
    input  wire [COL_ADDR_W-1:0]         wr_col_i,
    output wire                          wr_busy_o,      // 1-cycle unpacker latency
    output wire                          weights_loaded_o,

    // ---------------------------------------------------------------
    // >>> ANALOG CROSSBAR BOUNDARY -- everything below this line is
    // >>> the digital-side interface to the physical 2T2R macro. <<<
    // ---------------------------------------------------------------
    // Program (write) side: pushed out whenever a weight_mem entry is
    // updated; the analog macro applies the corresponding SET/RESET
    // pulse pair (see rram_2stage_strong_pulse.sp) to R+/R- of that cell.
    output wire                          prog_pulse_en_o,
    output wire [COL_ADDR_W-1:0]         prog_col_o,
    output wire [TILE_M-1:0]             prog_dir_o,
    output wire [TILE_M-1:0]             prog_mag1_o,
    output wire [TILE_M-1:0]             prog_mag0_o,

    // Compute (read) side: activation drive codes out, differential ADC
    // codes in. Applied/sampled every cycle during COMPUTE.
    output wire [TILE_M*ACT_WIDTH-1:0]   act_code_o,
    input  wire [TILE_K*ADC_WIDTH-1:0]   adc_ipos_i,
    input  wire [TILE_K*ADC_WIDTH-1:0]   adc_ineg_i,
    // ---------------------------------------------------------------
    // <<< END ANALOG CROSSBAR BOUNDARY >>>
    // ---------------------------------------------------------------

    // Compute control
    input  wire                          compute_en_i,     // apply act_code_i this cycle
    input  wire [TILE_M*ACT_WIDTH-1:0]   act_code_i,       // activation vector in
    input  wire                          acc_clear_i,       // start of new logical MAC
    input  wire                          acc_flush_i,       // spatial/temporal tile done, emit psum
    input  wire signed [7:0]             shift_amt_i,       // scale-align between passes

    output logic [TILE_K*ACC_WIDTH-1:0]  psum_o,
    output logic                         psum_valid_o
);

    // ---------------------------------------------------------------
    // Weight shadow memory: {dir, mag1, mag0} per (row, col)
    // ---------------------------------------------------------------
    typedef struct packed {
        logic dir;
        logic mag1;
        logic mag0;
    } weight_t;

    weight_t weight_mem [0:TILE_M-1][0:TILE_K-1];
    logic weights_loaded_q;
    assign weights_loaded_o = weights_loaded_q;

    // Unpacker instance -- decodes one (S,E,C1,C2) block into two weights
    // (W1 -> row 2i, W2 -> row 2i+1) per fp2_unpacker.v's semantics.
    wire u_Dir1, u_Mag1_1, u_Mag1_0, u_Dir2, u_Mag2_1, u_Mag2_0;
    wire [7:0] u_y; // unused terminal-count output of the existing unpacker

    fp2_e1m0_to_reram_unpacker u_unpacker (
        .clk    (clk),
        .rst    (~rst_n),
        .S      (wr_S_i),
        .E      (wr_E_i),
        .C1     (wr_C1_i),
        .C2     (wr_C2_i),
        .Dir1   (u_Dir1),
        .Mag1_1 (u_Mag1_1),
        .Mag1_0 (u_Mag1_0),
        .Dir2   (u_Dir2),
        .Mag2_1 (u_Mag2_1),
        .Mag2_0 (u_Mag2_0),
        .y      (u_y)
    );

    // Unpacker registers its outputs one cycle after S/E/C1/C2 are valid,
    // so the write address must be delayed by one cycle to stay aligned.
    logic                     wr_en_d1;
    logic [ROWPAIR_ADDR_W-1:0] wr_rowpair_d1;
    logic [COL_ADDR_W-1:0]     wr_col_d1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_en_d1      <= 1'b0;
            wr_rowpair_d1 <= '0;
            wr_col_d1     <= '0;
        end else begin
            wr_en_d1      <= wr_en_i;
            wr_rowpair_d1 <= wr_rowpair_i;
            wr_col_d1     <= wr_col_i;
        end
    end

    assign wr_busy_o = wr_en_i; // one outstanding write in flight at a time

    // Program-pulse bus (combinational passthrough of the row-pair just
    // decoded, gated by wr_en_d1). A real physical design would latch/hold
    // this for the pulse duration; kept purely combinational here since the
    // analog macro's own pulse generator owns pulse-width timing.
    logic [TILE_M-1:0] prog_dir_q, prog_mag1_q, prog_mag0_q;
    logic               prog_pulse_en_q;
    logic [COL_ADDR_W-1:0] prog_col_q;
    int                 row0, row1; // scratch for weight write-address decode (reused each write)

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weights_loaded_q <= 1'b0;
            prog_pulse_en_q  <= 1'b0;
            prog_dir_q       <= '0;
            prog_mag1_q      <= '0;
            prog_mag0_q      <= '0;
            prog_col_q       <= '0;
            for (int r = 0; r < TILE_M; r++)
                for (int c = 0; c < TILE_K; c++)
                    weight_mem[r][c] <= 3'b000;
        end else begin
            prog_pulse_en_q <= 1'b0; // default: no pulse this cycle

            if (wr_en_d1) begin
                row0 = 2 * int'(wr_rowpair_d1);
                row1 = row0 + 1;

                weight_mem[row0][wr_col_d1] <= {u_Dir1, u_Mag1_1, u_Mag1_0};
                if (row1 < TILE_M)
                    weight_mem[row1][wr_col_d1] <= {u_Dir2, u_Mag2_1, u_Mag2_0};

                prog_pulse_en_q <= 1'b1;
                prog_col_q      <= wr_col_d1;
                for (int r = 0; r < TILE_M; r++) begin
                    if (r == row0) begin
                        prog_dir_q[r]  <= u_Dir1;
                        prog_mag1_q[r] <= u_Mag1_1;
                        prog_mag0_q[r] <= u_Mag1_0;
                    end else if (r == row1) begin
                        prog_dir_q[r]  <= u_Dir2;
                        prog_mag1_q[r] <= u_Mag2_1;
                        prog_mag0_q[r] <= u_Mag2_0;
                    end
                end
                weights_loaded_q <= 1'b1;
            end
        end
    end

    assign prog_pulse_en_o = prog_pulse_en_q;
    assign prog_col_o      = prog_col_q;
    assign prog_dir_o      = prog_dir_q;
    assign prog_mag1_o     = prog_mag1_q;
    assign prog_mag0_o     = prog_mag0_q;

    // ---------------------------------------------------------------
    // Activation read path: pass the drive codes straight to the
    // crossbar boundary whenever compute is enabled.
    // ---------------------------------------------------------------
    assign act_code_o = compute_en_i ? act_code_i : '0;

    // ---------------------------------------------------------------
    // ADC collection + local accumulator (digital shift-and-add).
    // Differential subtract per column, then accumulate with an
    // optional scale-alignment shift for cross-tile-pass combining.
    // ---------------------------------------------------------------
    logic signed [ACC_WIDTH-1:0] acc_q [0:TILE_K-1];
    logic                        compute_en_d1;
    logic signed [ADC_WIDTH:0]   diff;          // scratch, reused each column iteration
    logic signed [ACC_WIDTH-1:0] diff_ext, prev_shifted;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            compute_en_d1 <= 1'b0;
        end else begin
            compute_en_d1 <= compute_en_i; // ADC latency: 1 cycle to sample after drive
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum_valid_o <= 1'b0;
            psum_o       <= '0;
            for (int c = 0; c < TILE_K; c++) acc_q[c] <= '0;
        end else begin
            psum_valid_o <= 1'b0;

            if (acc_clear_i) begin
                for (int c = 0; c < TILE_K; c++) acc_q[c] <= '0;
            end else if (compute_en_d1) begin
                for (int c = 0; c < TILE_K; c++) begin
                    diff = $signed({1'b0, adc_ipos_i[c*ADC_WIDTH +: ADC_WIDTH]}) -
                           $signed({1'b0, adc_ineg_i[c*ADC_WIDTH +: ADC_WIDTH]});
                    diff_ext = ACC_WIDTH'(diff);

                    // shift_amt_i >= 0 : incoming term needs left-shift to align
                    // shift_amt_i <  0 : previously accumulated term needs
                    //                    right-shift (rare; scale-down pass)
                    if (shift_amt_i >= 0)
                        prev_shifted = acc_q[c] + (diff_ext <<< shift_amt_i);
                    else
                        prev_shifted = (acc_q[c] >>> (-shift_amt_i)) + diff_ext;

                    acc_q[c] <= prev_shifted;
                end
            end

            if (acc_flush_i) begin
                for (int c = 0; c < TILE_K; c++)
                    psum_o[c*ACC_WIDTH +: ACC_WIDTH] <= acc_q[c];
                psum_valid_o <= 1'b1;
            end
        end
    end

endmodule

`default_nettype wire
