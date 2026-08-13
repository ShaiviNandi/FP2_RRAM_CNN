`timescale 1ns/1ps

module tb_accum;

    reg clk, rst;
    reg [15:0] psum_m;
    reg [7:0]  psum_x;
    reg        valid;

    wire [31:0] acc_m_std;
    wire [7:0]  acc_x_std;
    wire [47:0] acc_m_dsp;
    wire [7:0]  acc_x_dsp;
    wire [47:0] acc_m_dspfix;
    wire [7:0]  acc_x_dspfix;

    temporal_accumulator_std u_std (
        .clk(clk), .rst(rst),
        .psum_m(psum_m), .psum_x(psum_x), .valid(valid),
        .acc_m(acc_m_std), .acc_x(acc_x_std)
    );

    temporal_accumulator_dsp u_dsp (
        .clk(clk), .rst(rst),
        .psum_m(psum_m), .psum_x(psum_x), .valid(valid),
        .acc_m(acc_m_dsp), .acc_x(acc_x_dsp)
    );

    temporal_accumulator_dsp_fixed u_dspfix (
        .clk(clk), .rst(rst),
        .psum_m(psum_m), .psum_x(psum_x), .valid(valid),
        .acc_m(acc_m_dspfix), .acc_x(acc_x_dspfix)
    );

    always #5 clk = ~clk;

    task apply(input [15:0] m, input [7:0] x);
        begin
            @(negedge clk);
            psum_m = m; psum_x = x; valid = 1;
        end
    endtask

    initial begin
        $dumpfile("accum.vcd");
        $dumpvars(0, tb_accum);

        clk = 0; rst = 1; valid = 0; psum_m = 0; psum_x = 0;
        @(negedge clk); @(negedge clk);
        rst = 0;

        // First-ever valid sample after reset: this is the case where the
        // std module's explicit "acc_m==0 => load" branch kicks in, and the
        // dsp module has no equivalent branch.
        apply(16'd1000, 8'd10);
        @(negedge clk); valid = 0;
        @(posedge clk); #1;
        $display("t=%0t after 1st sample: std.acc_m=%0d (x=%0d)  dsp.acc_m=%0h (x=%0d)  dspfix.acc_m=%0h (x=%0d)",
                  $time, acc_m_std, acc_x_std, acc_m_dsp, acc_x_dsp, acc_m_dspfix, acc_x_dspfix);

        // Second sample, larger exponent -> forces a right-shift of the
        // buffered accumulator.
        apply(16'd2000, 8'd12);
        @(negedge clk); valid = 0;
        @(posedge clk); #1;
        $display("t=%0t after 2nd sample: std.acc_m=%0d (x=%0d)  dsp.acc_m=%0h (x=%0d)  dspfix.acc_m=%0h (x=%0d)",
                  $time, acc_m_std, acc_x_std, acc_m_dsp, acc_x_dsp, acc_m_dspfix, acc_x_dspfix);

        // Third sample, smaller exponent -> incoming gets shifted instead.
        apply(16'd500, 8'd9);
        @(negedge clk); valid = 0;
        @(posedge clk); #1;
        $display("t=%0t after 3rd sample: std.acc_m=%0d (x=%0d)  dsp.acc_m=%0h (x=%0d)  dspfix.acc_m=%0h (x=%0d)",
                  $time, acc_m_std, acc_x_std, acc_m_dsp, acc_x_dsp, acc_m_dspfix, acc_x_dspfix);

        // Let dsp's extra pipeline stage settle and compare again.
        @(negedge clk);
        @(posedge clk); #1;
        $display("t=%0t settle          : std.acc_m=%0d (x=%0d)  dsp.acc_m=%0h (x=%0d)  dspfix.acc_m=%0h (x=%0d)",
                  $time, acc_m_std, acc_x_std, acc_m_dsp, acc_x_dsp, acc_m_dspfix, acc_x_dspfix);

        if (^acc_m_dsp === 1'bx)
            $display("BUG CONFIRMED: dsp.acc_m contains X (uninitialized aligned_acc/aligned_psum consumed before first write)");

        $finish;
    end

endmodule
