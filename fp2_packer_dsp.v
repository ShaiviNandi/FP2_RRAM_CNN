module temporal_accumulator_dsp_fixed (
    input  wire         clk,
    input  wire         rst,

    input  wire [15:0]  psum_m,
    input  wire [7:0]   psum_x,
    input  wire         valid,

    output reg  [47:0]  acc_m,  // DSP48 has a 48-bit accumulator
    output reg  [7:0]   acc_x
);

    // Combinational alignment (this is the DSP48 pre-adder/shifter stage).
    // NOTE: this still costs one extra pipeline cycle of latency vs the
    // std module because acc_m is registered from these combinational
    // values rather than computed in the same cycle the inputs arrive --
    // that is inherent to mapping onto the DSP48's registered ALU input,
    // so downstream logic must account for a 1-cycle-later result vs
    // temporal_accumulator_std, not just a bit-exact value.
    wire [7:0]  exp_diff;
    wire        incoming_is_larger;
    wire [47:0] aligned_acc_c;
    wire [47:0] aligned_psum_c;

    assign incoming_is_larger = (psum_x > acc_x);
    assign exp_diff = incoming_is_larger ? (psum_x - acc_x) : (acc_x - psum_x);

    // First valid sample after reset (acc_m == 0) is a straight load, same
    // as temporal_accumulator_std, so it isn't corrupted by a huge shift
    // against an exponent of 0.
    assign aligned_acc_c  = (acc_m == 48'b0) ? 48'b0 :
                             incoming_is_larger ? ((exp_diff <= 17) ? (acc_m >> exp_diff) : 48'b0) :
                             acc_m;

    assign aligned_psum_c = (acc_m == 48'b0) ? {32'b0, psum_m} :
                             incoming_is_larger ? {32'b0, psum_m} :
                             ((exp_diff <= 17) ? ({32'b0, psum_m} >> exp_diff) : 48'b0);

    (* use_dsp = "yes" *) reg [47:0] aligned_acc;
    (* use_dsp = "yes" *) reg [47:0] aligned_psum;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_m        <= 48'b0;
            acc_x        <= 8'b0;
            aligned_acc  <= 48'b0;
            aligned_psum <= 48'b0;
        end else if (valid) begin
            aligned_acc  <= aligned_acc_c;
            aligned_psum <= aligned_psum_c;
            acc_x        <= (acc_m == 48'b0) ? psum_x : (incoming_is_larger ? psum_x : acc_x);
            acc_m        <= aligned_acc + aligned_psum;
        end
    end

endmodule
