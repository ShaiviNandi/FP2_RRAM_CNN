// =============================================================================
// im2col_stream_engine.sv
// -----------------------------------------------------------------------------
// Line-buffer based im2col patch streaming engine. Converts an incoming
// row-major feature-map pixel stream (CH channels/pixel, one pixel per valid
// beat) into KERNEL_SIZE x KERNEL_SIZE x CH patch vectors suitable for
// driving the PE array's activation port, for both 3x3 (stride 1 or 2, with
// SAME-style zero padding) and 1x1 (pointwise, patch == pixel) convolutions.
//
// Design: KERNEL_SIZE line buffers (BRAM-inferred), each holding one row of
// IMG_WIDTH*CH activations. A KERNEL_SIZE x KERNEL_SIZE sliding window reads
// out of the buffers to form each patch. For KERNEL_SIZE==1 the line buffers
// are bypassed entirely (patch = current pixel, zero extra latency).
//
// Patch elements are emitted CH-major, PATCH_M := KERNEL_SIZE*KERNEL_SIZE*CH
// wide, split into ceil(PATCH_M/TILE_M) beats of TILE_M elements each on
// patch_beat_o, so the PE array can consume oversized (channel-deep) patches
// across multiple spatial tile passes without stalling the line buffers.
// =============================================================================
`default_nettype none

module im2col_stream_engine #(
    parameter int IMG_WIDTH   = 56,
    parameter int IMG_HEIGHT  = 56,
    parameter int CH          = 64,
    parameter int DATA_WIDTH  = 8,
    parameter int KERNEL_SIZE = 3,     // 1 or 3
    parameter int STRIDE      = 1,     // 1 or 2
    parameter int PAD         = (KERNEL_SIZE == 3) ? 1 : 0,
    parameter int TILE_M      = 32,    // PE tile row count -- patch beat width
    parameter int OUT_WIDTH   = (IMG_WIDTH + 2*PAD - KERNEL_SIZE)/STRIDE + 1,
    parameter int OUT_HEIGHT  = (IMG_HEIGHT + 2*PAD - KERNEL_SIZE)/STRIDE + 1,
    parameter int PATCH_M     = KERNEL_SIZE*KERNEL_SIZE*CH,
    parameter int NUM_BEATS   = (PATCH_M + TILE_M - 1) / TILE_M,
    parameter int COL_ADDR_W  = $clog2(IMG_WIDTH)
) (
    input  wire clk,
    input  wire rst_n,

    // ---------------- Pixel input stream (row-major, CH lanes/pixel) -------
    input  wire                             px_valid_i,
    input  wire [CH*DATA_WIDTH-1:0]         px_data_i,   // one pixel, all channels
    output wire                             px_ready_o,
    input  wire                             frame_start_i, // pulse: reset row/col ptrs

    // ---------------- Patch output stream (TILE_M-wide beats) --------------
    output logic                            patch_valid_o,
    output logic [TILE_M*DATA_WIDTH-1:0]    patch_beat_o,
    output logic [$clog2(NUM_BEATS)-1:0]    patch_beat_idx_o,
    output logic                            patch_last_beat_o, // last beat of this patch
    output logic                            patch_last_of_row_o,
    output logic                            patch_last_of_frame_o,
    input  wire                             patch_ready_i
);

    localparam int ROW_BYTES = IMG_WIDTH * CH * DATA_WIDTH;

    // ---------------------------------------------------------------
    // Line buffers: KERNEL_SIZE deep (BRAM-inferred), each IMG_WIDTH
    // pixels x CH channels wide. Bypassed for KERNEL_SIZE==1.
    // ---------------------------------------------------------------
    generate
        if (KERNEL_SIZE > 1) begin : g_linebuf

            logic [CH*DATA_WIDTH-1:0] line_mem [0:KERNEL_SIZE-1][0:IMG_WIDTH-1];
            logic [COL_ADDR_W-1:0]    wr_col_q;
            logic [$clog2(KERNEL_SIZE)-1:0] wr_row_sel_q; // which buffer row is "newest"
            logic [$clog2(IMG_HEIGHT):0]    in_row_q;
            logic [$clog2(IMG_HEIGHT)+1:0]  valid_rows_q; // rows written so far this frame

            assign px_ready_o = 1'b1; // line buffer write always keeps pace (single-port write)

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    wr_col_q     <= '0;
                    wr_row_sel_q <= '0;
                    in_row_q     <= '0;
                    valid_rows_q <= '0;
                end else if (frame_start_i) begin
                    wr_col_q     <= '0;
                    wr_row_sel_q <= '0;
                    in_row_q     <= '0;
                    valid_rows_q <= '0;
                end else if (px_valid_i) begin
                    line_mem[wr_row_sel_q][wr_col_q] <= px_data_i;
                    if (wr_col_q == IMG_WIDTH-1) begin
                        wr_col_q     <= '0;
                        wr_row_sel_q <= (wr_row_sel_q == KERNEL_SIZE-1) ? '0 : wr_row_sel_q + 1'b1;
                        in_row_q     <= in_row_q + 1'b1;
                        if (valid_rows_q < KERNEL_SIZE)
                            valid_rows_q <= valid_rows_q + 1'b1;
                    end else begin
                        wr_col_q <= wr_col_q + 1'b1;
                    end
                end
            end

            // ---------------------------------------------------------------
            // Output-window walk: for each output (row, col), gather the
            // KERNEL_SIZE x KERNEL_SIZE x CH patch from the line buffers,
            // zero-padding at the frame border. Only fires once enough rows
            // are buffered (>= KERNEL_SIZE, or end-of-frame flush).
            // ---------------------------------------------------------------
            logic [$clog2(OUT_WIDTH)-1:0]  out_col_q;
            logic [$clog2(OUT_HEIGHT)-1:0] out_row_q;
            logic [$clog2(NUM_BEATS)-1:0]  beat_q;
            logic                          window_ready;
            // Scratch vars for per-lane patch-index decode below (reused
            // each loop iteration; fully consumed before next iteration).
            int  flat, ch, kx, ky, src_col, src_row_off, buf_row;
            bit  in_bounds;

            assign window_ready = (valid_rows_q >= KERNEL_SIZE) ||
                                   (in_row_q >= out_row_q*STRIDE + KERNEL_SIZE - PAD);

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    out_col_q <= '0;
                    out_row_q <= '0;
                    beat_q    <= '0;
                    patch_valid_o <= 1'b0;
                end else if (frame_start_i) begin
                    out_col_q <= '0;
                    out_row_q <= '0;
                    beat_q    <= '0;
                    patch_valid_o <= 1'b0;
                end else begin
                    patch_valid_o <= window_ready;

                    if (window_ready && patch_ready_i) begin
                        // Assemble one TILE_M-wide beat of the patch: walk the
                        // flat (ky, kx, ch) index space starting at beat_q*TILE_M.
                        for (int e = 0; e < TILE_M; e++) begin
                            flat = int'(beat_q)*TILE_M + e;
                            ch   = flat % CH;
                            kx   = (flat / CH) % KERNEL_SIZE;
                            ky   = (flat / CH) / KERNEL_SIZE;
                            src_col = int'(out_col_q)*STRIDE + kx - PAD;
                            src_row_off = ky; // relative row within buffered window
                            in_bounds = (flat < PATCH_M) &&
                                        (src_col >= 0) && (src_col < IMG_WIDTH);
                            if (in_bounds) begin
                                buf_row = (int'(wr_row_sel_q) + KERNEL_SIZE - 1 - src_row_off) % KERNEL_SIZE;
                                patch_beat_o[e*DATA_WIDTH +: DATA_WIDTH] <=
                                    line_mem[buf_row][src_col][ch*DATA_WIDTH +: DATA_WIDTH];
                            end else begin
                                patch_beat_o[e*DATA_WIDTH +: DATA_WIDTH] <= '0; // zero-pad
                            end
                        end

                        patch_beat_idx_o     <= beat_q;
                        patch_last_beat_o    <= (beat_q == NUM_BEATS-1);
                        patch_last_of_row_o  <= (beat_q == NUM_BEATS-1) && (out_col_q == OUT_WIDTH-1);
                        patch_last_of_frame_o<= (beat_q == NUM_BEATS-1) && (out_col_q == OUT_WIDTH-1) &&
                                                (out_row_q == OUT_HEIGHT-1);

                        if (beat_q == NUM_BEATS-1) begin
                            beat_q <= '0;
                            if (out_col_q == OUT_WIDTH-1) begin
                                out_col_q <= '0;
                                out_row_q <= out_row_q + 1'b1;
                            end else begin
                                out_col_q <= out_col_q + 1'b1;
                            end
                        end else begin
                            beat_q <= beat_q + 1'b1;
                        end
                    end
                end
            end

        end else begin : g_bypass
            // -------------------------------------------------------
            // 1x1 (pointwise) path: patch == current pixel's channels,
            // no line buffering, no padding, stride applied by simply
            // dropping (STRIDE-1) out of every STRIDE input pixels.
            // -------------------------------------------------------
            logic [$clog2(IMG_WIDTH)-1:0]  col_q;
            logic [$clog2(IMG_HEIGHT)-1:0] row_q;
            logic [$clog2(NUM_BEATS)-1:0]  beat_q;
            logic keep_pixel;
            int   flat; // scratch, reused each loop iteration

            assign keep_pixel = ((col_q % STRIDE) == 0) && ((row_q % STRIDE) == 0);
            assign px_ready_o = patch_ready_i || !keep_pixel;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    col_q <= '0; row_q <= '0; beat_q <= '0; patch_valid_o <= 1'b0;
                end else if (frame_start_i) begin
                    col_q <= '0; row_q <= '0; beat_q <= '0; patch_valid_o <= 1'b0;
                end else begin
                    patch_valid_o <= 1'b0;
                    if (px_valid_i && keep_pixel) begin
                        for (int e = 0; e < TILE_M; e++) begin
                            flat = int'(beat_q)*TILE_M + e;
                            patch_beat_o[e*DATA_WIDTH +: DATA_WIDTH] <=
                                (flat < CH) ? px_data_i[flat*DATA_WIDTH +: DATA_WIDTH] : '0;
                        end
                        patch_valid_o      <= 1'b1;
                        patch_beat_idx_o   <= beat_q;
                        patch_last_beat_o  <= (beat_q == NUM_BEATS-1);
                        // beat sequencing handled per-pixel: for NUM_BEATS>1,
                        // upstream must hold the pixel steady across beats
                        // (px_ready_o gates that via patch_ready_i above).
                    end
                    if (px_valid_i) begin
                        if (col_q == IMG_WIDTH-1) begin
                            col_q <= '0;
                            row_q <= row_q + 1'b1;
                        end else begin
                            col_q <= col_q + 1'b1;
                        end
                    end
                    patch_last_of_row_o   <= (col_q == IMG_WIDTH-1);
                    patch_last_of_frame_o <= (col_q == IMG_WIDTH-1) && (row_q == IMG_HEIGHT-1);
                end
            end
        end
    endgenerate

endmodule

`default_nettype wire
