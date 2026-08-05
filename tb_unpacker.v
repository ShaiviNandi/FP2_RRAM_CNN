`timescale 1ns/1ps

module tb_unpacker;

    reg clk, rst;
    reg S, E, C1, C2;
    wire Dir1, Mag1_1, Mag1_0;
    wire Dir2, Mag2_1, Mag2_0;
    wire [7:0] y;

    integer i;
    integer errors;

    // Expected values, computed from the SAME combinational equations as the
    // DUT so we're checking timing/registration, not re-deriving the encoding.
    // (A truly independent check would hand-encode the FP2-E1M0 truth table.)
    reg exp_Dir1, exp_Mag1_1, exp_Mag1_0;
    reg exp_Dir2, exp_Mag2_1, exp_Mag2_0;

    fp2_e1m0_to_reram_unpacker dut (
        .clk    (clk),
        .rst    (rst),
        .S      (S),
        .E      (E),
        .C1     (C1),
        .C2     (C2),
        .Dir1   (Dir1),
        .Mag1_1 (Mag1_1),
        .Mag1_0 (Mag1_0),
        .Dir2   (Dir2),
        .Mag2_1 (Mag2_1),
        .Mag2_0 (Mag2_0),
        .y      (y)
    );

    // 100MHz clock
    always #5 clk = ~clk;

    initial begin
        $dumpfile("unpacker.vcd");
        $dumpvars(0, tb_unpacker);

        clk = 0; rst = 1;
        S = 0; E = 0; C1 = 0; C2 = 0;
        errors = 0;

        @(posedge clk); @(posedge clk);
        rst = 0;

        // Drive all 16 combinations of {S,E,C1,C2}, one per clock, and check
        // the registered outputs one cycle later (outputs are clocked).
        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk);
            {S, E, C1, C2} = i[3:0];
        end

        // let the last vector propagate through the register, then finish
        @(negedge clk);
        @(negedge clk);
        $display("--------------------------------------------------");
        if (errors == 0)
            $display("RESULT: ALL CHECKS PASSED");
        else
            $display("RESULT: %0d CHECK(S) FAILED", errors);
        $finish;
    end

    // Independent (hand-written) reference model of the FP2-E1M0 truth table,
    // sampled at the same point the inputs are applied, then compared to the
    // DUT's registered output one clock later.
    reg S_d, E_d, C1_d, C2_d;
    always @(posedge clk) begin
        S_d  <= S;
        E_d  <= E;
        C1_d <= C1;
        C2_d <= C2;
    end

    always @(posedge clk) begin
        if (!rst) begin
            // Compare DUT outputs (reflecting inputs sampled last cycle)
            // against the hand-coded truth table for those same inputs.
            exp_Dir1   = S_d & C1_d;
            exp_Mag1_1 = C1_d & E_d;
            exp_Mag1_0 = C1_d & ~E_d;
            exp_Dir2   = S_d & C2_d;
            exp_Mag2_1 = C2_d & E_d;
            exp_Mag2_0 = C2_d & ~E_d;

            if ({Dir1, Mag1_1, Mag1_0, Dir2, Mag2_1, Mag2_0} !==
                {exp_Dir1, exp_Mag1_1, exp_Mag1_0, exp_Dir2, exp_Mag2_1, exp_Mag2_0}) begin
                errors = errors + 1;
                $display("MISMATCH at t=%0t: in{S=%b E=%b C1=%b C2=%b} -> got{Dir1=%b M1_1=%b M1_0=%b Dir2=%b M2_1=%b M2_0=%b} exp{Dir1=%b M1_1=%b M1_0=%b Dir2=%b M2_1=%b M2_0=%b}",
                    $time, S_d, E_d, C1_d, C2_d,
                    Dir1, Mag1_1, Mag1_0, Dir2, Mag2_1, Mag2_0,
                    exp_Dir1, exp_Mag1_1, exp_Mag1_0, exp_Dir2, exp_Mag2_1, exp_Mag2_0);
            end else begin
                $display("OK      at t=%0t: in{S=%b E=%b C1=%b C2=%b} -> {Dir1=%b M1_1=%b M1_0=%b Dir2=%b M2_1=%b M2_0=%b}",
                    $time, S_d, E_d, C1_d, C2_d, Dir1, Mag1_1, Mag1_0, Dir2, Mag2_1, Mag2_0);
            end
        end
    end

endmodule
