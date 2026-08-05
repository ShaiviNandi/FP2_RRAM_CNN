`timescale 1ns/1ps

module tb_unpacker_export;

    reg clk, rst;
    reg S, E, C1, C2;
    wire Dir1, Mag1_1, Mag1_0;
    wire Dir2, Mag2_1, Mag2_0;
    wire [7:0] y;

    integer i;
    integer fh;

    fp2_e1m0_to_reram_unpacker dut (
        .clk(clk), .rst(rst),
        .S(S), .E(E), .C1(C1), .C2(C2),
        .Dir1(Dir1), .Mag1_1(Mag1_1), .Mag1_0(Mag1_0),
        .Dir2(Dir2), .Mag2_1(Mag2_1), .Mag2_0(Mag2_0),
        .y(y)
    );

    always #5 clk = ~clk;

    initial begin
        fh = $fopen("dac_stimulus.txt", "w");
        // header: time(ns) Dir1 Mag1_1 Mag1_0 Dir2 Mag2_1 Mag2_0
        $fdisplay(fh, "# time_ns Dir1 Mag1_1 Mag1_0 Dir2 Mag2_1 Mag2_0");

        clk = 0; rst = 1;
        S = 0; E = 0; C1 = 0; C2 = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk);
            {S, E, C1, C2} = i[3:0];
        end
        @(negedge clk);
        @(negedge clk);

        $fclose(fh);
        $finish;
    end

    // Log the registered DAC control outputs on every clock edge.
    always @(posedge clk) begin
        if (!rst)
            $fdisplay(fh, "%0t %b %b %b %b %b %b", $time, Dir1, Mag1_1, Mag1_0, Dir2, Mag2_1, Mag2_0);
    end

endmodule
