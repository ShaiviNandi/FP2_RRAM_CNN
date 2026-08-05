module fp2_e1m0_to_reram_unpacker (
    input wire clk,
    input wire rst,
    
    // 4-bit FP2 Block Inputs
    input wire S,   // Shared Sign
    input wire E,   // Shared Exponent
    input wire C1,  // Flag for W1
    input wire C2,  // Flag for W2
    
    // Unpacked Outputs for W1 DAC
    output reg Dir1,
    output reg Mag1_1, // Represents 1.0
    output reg Mag1_0, // Represents 0.5
    
    // Unpacked Outputs for W2 DAC
    output reg Dir2,
    output reg Mag2_1, // Represents 1.0
    output reg Mag2_0, // Represents 0.5
    
    // Terminal count register
    output reg [7:0] y 
);

    // Combinational logic for W1 unpacking
    wire w_Dir1   = S & C1;
    wire w_Mag1_1 = C1 & E;
    wire w_Mag1_0 = C1 & ~E;

    // Combinational logic for W2 unpacking
    wire w_Dir2   = S & C2;
    wire w_Mag2_1 = C2 & E;
    wire w_Mag2_0 = C2 & ~E;

    // Synchronous clocked output block
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            y       <= 8'b0;
            Dir1    <= 1'b0;
            Mag1_1  <= 1'b0;
            Mag1_0  <= 1'b0;
            Dir2    <= 1'b0;
            Mag2_1  <= 1'b0;
            Mag2_0  <= 1'b0;
        end else begin
            // Increment terminal count register per clock cycle
            y       <= y + 1; 
            
            // Register DAC inputs
            Dir1    <= w_Dir1;
            Mag1_1  <= w_Mag1_1;
            Mag1_0  <= w_Mag1_0;
            Dir2    <= w_Dir2;
            Mag2_1  <= w_Mag2_1;
            Mag2_0  <= w_Mag2_0;
        end
    end

endmodule