// =============================================================================
// residual_post_proc.sv
// -----------------------------------------------------------------------------
// Post-processing pipeline sitting between the PE-array partial-sum output
// and the feature-map write-back / next-layer stream:
//
//   psum_i --> [BN fold: *scale +bias] --> [+ residual shortcut] --> [ReLU]
//          --> [optional 2x2 max/avg pool] --> [requantize to DATA_WIDTH]
//
// BatchNorm is folded per ResNet-18 standard practice: for inference,
// y = gamma*(x-mean)/sqrt(var+eps) + beta  ==  x*bn_scale + bn_bias, where
// bn_scale/bn_bias are precomputed offline (in the Python flow) per output
// channel and loaded into this module's per-channel coefficient table.
//
// Residual shortcut: for ResNet-18 identity blocks the shortcut is simply
// the block's input feature map (possibly through a 1x1/stride-2 projection
// conv computed earlier through the SAME pe_tile array and staged in
// `shortcut_buf` off-chip/on-chip by the controller); this module only does
// the elementwise add once both operands are aligned and valid.
//
// Pooling: fixed 2x2 window, stride 2, max or average, matching ResNet-18's
// stem pool and any stage-transition pooling. Bypass-able per layer.
// =============================================================================
`default_nettype none

module residual_post_proc #(
    parameter int ACC_WIDTH   = 24,   // width of incoming psum (post PE accumulate)
    parameter int DATA_WIDTH  = 8,    // output activation width (re-quantized)
    parameter int SCALE_WIDTH = 16,   // BN scale fixed-point width (Qm.n, n below)
    parameter int SCALE_FRAC  = 12,   // fractional bits of bn_scale
    parameter int BIAS_WIDTH  = 24,
    parameter int CH          = 64,   // channels in this layer's output (coeff table depth)
    parameter int CH_ADDR_W   = $clog2(CH)
) (
    input  wire clk,
    input  wire rst_n,

    // ---------------- Per-channel BN coefficient load (weight-stationary) --
    input  wire                          coeff_wr_en_i,
    input  wire [CH_ADDR_W-1:0]          coeff_wr_ch_i,
    input  wire signed [SCALE_WIDTH-1:0] coeff_wr_scale_i,
    input  wire signed [BIAS_WIDTH-1:0]  coeff_wr_bias_i,

    // ---------------- Conv partial-sum input --------------------------------
    input  wire                          psum_valid_i,
    input  wire signed [ACC_WIDTH-1:0]   psum_i,
    input  wire [CH_ADDR_W-1:0]          psum_ch_i,

    // ---------------- Residual shortcut input (same channel/pixel) ---------
    input  wire                          residual_valid_i,
    input  wire signed [DATA_WIDTH-1:0]  residual_i,
    input  wire                          residual_present_i, // 0 for non-skip layers

    // ---------------- Layer-level control -----------------------------------
    input  wire                          relu_en_i,
    input  wire                          pool_en_i,     // 2x2 stride-2 max/avg
    input  wire                          pool_is_avg_i, // 0=max, 1=avg
    input  wire                          pool_first_of_window_i, // clears pool accum
    input  wire                          pool_last_of_window_i,  // emits pooled result

    // ---------------- Output -------------------------------------------------
    output logic                         out_valid_o,
    output logic signed [DATA_WIDTH-1:0] out_data_o,
    output logic [CH_ADDR_W-1:0]         out_ch_o
);

    // ---------------------------------------------------------------
    // Per-channel BN coefficient table
    // ---------------------------------------------------------------
    logic signed [SCALE_WIDTH-1:0] bn_scale [0:CH-1];
    logic signed [BIAS_WIDTH-1:0]  bn_bias  [0:CH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int c = 0; c < CH; c++) begin
                bn_scale[c] <= '0;
                bn_bias[c]  <= '0;
            end
        end else if (coeff_wr_en_i) begin
            bn_scale[coeff_wr_ch_i] <= coeff_wr_scale_i;
            bn_bias[coeff_wr_ch_i]  <= coeff_wr_bias_i;
        end
    end

    // ---------------------------------------------------------------
    // Stage 1: BN fold  (x * scale) >> SCALE_FRAC + bias
    // ---------------------------------------------------------------
    localparam int MUL_WIDTH = ACC_WIDTH + SCALE_WIDTH;

    logic                          s1_valid_q;
    logic signed [MUL_WIDTH-1:0]   s1_mul_q;
    logic signed [BIAS_WIDTH-1:0]  s1_bias_q;
    logic [CH_ADDR_W-1:0]          s1_ch_q;
    logic                          s1_res_present_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_valid_q <= 1'b0;
        end else begin
            s1_valid_q <= psum_valid_i;
            if (psum_valid_i) begin
                s1_mul_q         <= psum_i * bn_scale[psum_ch_i];
                s1_bias_q        <= bn_bias[psum_ch_i];
                s1_ch_q          <= psum_ch_i;
                s1_res_present_q <= residual_present_i;
            end
        end
    end

    // ---------------------------------------------------------------
    // Stage 2: shift-normalize, add residual, ReLU
    // ---------------------------------------------------------------
    logic signed [BIAS_WIDTH-1:0] s2_bn_out;
    logic signed [BIAS_WIDTH-1:0] s2_sum;
    logic signed [BIAS_WIDTH-1:0] s2_relu;

    assign s2_bn_out = (s1_mul_q >>> SCALE_FRAC) + s1_bias_q;
    assign s2_sum    = s1_res_present_q ? (s2_bn_out + BIAS_WIDTH'(residual_i)) : s2_bn_out;
    assign s2_relu   = relu_en_i ? ((s2_sum[BIAS_WIDTH-1]) ? '0 : s2_sum) : s2_sum;

    logic                  s2_valid_q;
    logic signed [DATA_WIDTH-1:0] s2_data_q;
    logic [CH_ADDR_W-1:0]  s2_ch_q;

    // Requantize to DATA_WIDTH with saturation.
    function automatic logic signed [DATA_WIDTH-1:0] saturate(input logic signed [BIAS_WIDTH-1:0] v);
        localparam logic signed [DATA_WIDTH-1:0] MAXV = {1'b0, {(DATA_WIDTH-1){1'b1}}};
        localparam logic signed [DATA_WIDTH-1:0] MINV = {1'b1, {(DATA_WIDTH-1){1'b0}}};
        if (v > $signed({{(BIAS_WIDTH-DATA_WIDTH){1'b0}}, MAXV}))
            return MAXV;
        else if (v < $signed({{(BIAS_WIDTH-DATA_WIDTH){1'b1}}, MINV}))
            return MINV;
        else
            return v[DATA_WIDTH-1:0];
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_valid_q <= 1'b0;
        end else begin
            s2_valid_q <= s1_valid_q;
            if (s1_valid_q) begin
                s2_data_q <= saturate(s2_relu);
                s2_ch_q   <= s1_ch_q;
            end
        end
    end

    // ---------------------------------------------------------------
    // Stage 3: optional 2x2 stride-2 pool (max or average) across the
    // 4 pixels of one window; controller drives
    // pool_first_of_window_i / pool_last_of_window_i to bracket a
    // window's 4 arrivals for a given channel.
    // ---------------------------------------------------------------
    logic signed [DATA_WIDTH+1:0] pool_acc_q; // 2 extra bits for 4-way avg sum
    logic [1:0]                   pool_cnt_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_o <= 1'b0;
            pool_acc_q  <= '0;
            pool_cnt_q  <= '0;
        end else begin
            out_valid_o <= 1'b0;

            if (!pool_en_i) begin
                // straight passthrough
                out_valid_o <= s2_valid_q;
                out_data_o  <= s2_data_q;
                out_ch_o    <= s2_ch_q;
            end else if (s2_valid_q) begin
                if (pool_first_of_window_i) begin
                    pool_acc_q <= pool_is_avg_i ? {{2{s2_data_q[DATA_WIDTH-1]}}, s2_data_q}
                                                 : {{2{s2_data_q[DATA_WIDTH-1]}}, s2_data_q};
                    pool_cnt_q <= 2'd1;
                end else begin
                    if (pool_is_avg_i)
                        pool_acc_q <= pool_acc_q + {{2{s2_data_q[DATA_WIDTH-1]}}, s2_data_q};
                    else
                        pool_acc_q <= (s2_data_q > pool_acc_q[DATA_WIDTH-1:0]) ?
                                      {{2{s2_data_q[DATA_WIDTH-1]}}, s2_data_q} : pool_acc_q;
                    pool_cnt_q <= pool_cnt_q + 1'b1;
                end

                if (pool_last_of_window_i) begin
                    out_valid_o <= 1'b1;
                    out_ch_o    <= s2_ch_q;
                    if (pool_is_avg_i)
                        out_data_o <= saturate({{(BIAS_WIDTH-DATA_WIDTH-2){pool_acc_q[DATA_WIDTH+1]}},
                                                 (pool_acc_q + {{2{s2_data_q[DATA_WIDTH-1]}}, s2_data_q}) >>> 2});
                    else
                        out_data_o <= (s2_data_q > pool_acc_q[DATA_WIDTH-1:0]) ? s2_data_q : pool_acc_q[DATA_WIDTH-1:0];
                end
            end
        end
    end

endmodule

`default_nettype wire
