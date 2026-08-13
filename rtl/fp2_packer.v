module temporal_accumulator_std (
    input  wire         clk,
    input  wire         rst,
    
    // Incoming Psum from the spatial MAC / ADC
    input  wire [15:0]  psum_m, // Raw fixed-point mantissa sum
    input  wire [7:0]   psum_x, // Shared 8-bit scaling factor (exponent)
    input  wire         valid,  // Data valid flag
    
    // Accumulated Output
    output reg  [31:0]  acc_m,  // Accumulated mantissa
    output reg  [7:0]   acc_x   // Accumulated exponent
);

    // Internal wires for exponent difference calculation
    wire [7:0] exp_diff;
    wire       incoming_is_larger;
    
    // Calculate the difference between the buffered exponent and incoming exponent
    assign incoming_is_larger = (psum_x > acc_x);
    assign exp_diff = incoming_is_larger ? (psum_x - acc_x) : (acc_x - psum_x);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc_m <= 32'b0;
            acc_x <= 8'b0;
        end else if (valid) begin
            if (acc_m == 32'b0) begin
                // Initialization: If buffer is empty, just load the incoming data
                acc_m <= {16'b0, psum_m};
                acc_x <= psum_x;
            end else if (incoming_is_larger) begin
                // Incoming exponent is larger: Shift the buffered accumulated mantissa right
                acc_m <= ({16'b0, psum_m}) + (acc_m >> exp_diff);
                acc_x <= psum_x; // Update to the new larger exponent
            end else begin
                // Buffered exponent is larger: Shift the incoming mantissa right
                acc_m <= acc_m + ({16'b0, psum_m} >> exp_diff);
                // acc_x remains unchanged
            end
        end
    end

endmodule