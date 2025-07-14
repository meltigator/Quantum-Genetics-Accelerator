// quant_interface.v
// Quantum Genetics Accelerator - Interface Module
// Bridge between FPGA quantum processing and external analysis layers

module quant_interface #(
    parameter DNA_WIDTH = 32,
    parameter BATCH_SIZE = 16,
    parameter RESULT_BUFFER_DEPTH = 64,
    parameter AGING_FACTOR_WIDTH = 8
) (
    input wire clk,
    input wire rst_n,
    
    // Host interface (UART/SPI/AXI-like)
    input wire host_req,
    input wire [7:0] host_cmd,
    input wire [DNA_WIDTH-1:0] host_dna_data,
    input wire [AGING_FACTOR_WIDTH-1:0] host_aging_factor,
    output reg host_ack,
    output reg [DNA_WIDTH-1:0] host_result_data,
    output reg [15:0] host_entropy_data,
    output reg [7:0] host_mutation_count,
    output reg host_data_valid,
    
    // Processor interface
    output reg proc_start,
    output reg [DNA_WIDTH-1:0] proc_dna_sequence,
    output reg [AGING_FACTOR_WIDTH-1:0] proc_aging_factor,
    output reg [1:0] proc_gate_select,
    input wire [DNA_WIDTH-1:0] proc_processed_dna,
    input wire [15:0] proc_entropy_measure,
    input wire proc_processing_done,
    input wire [7:0] proc_mutation_count,
    
    // Status and control
    output reg [7:0] batch_progress,
    output reg batch_complete,
    output reg error_flag,
    output reg [15:0] total_sequences_processed
);

    // Command definitions
    localparam CMD_SINGLE_PROCESS = 8'h01;
    localparam CMD_BATCH_PROCESS = 8'h02;
    localparam CMD_SET_GATE = 8'h03;
    localparam CMD_GET_STATUS = 8'h04;
    localparam CMD_RESET_COUNTERS = 8'h05;
    localparam CMD_CONFIGURE_AGING = 8'h06;
    localparam CMD_GET_BATCH_RESULTS = 8'h07;
    localparam CMD_CONTINUOUS_MODE = 8'h08;
    
    // Interface state machine
    reg [3:0] interface_state;
    localparam IF_IDLE = 4'h0;
    localparam IF_DECODE_CMD = 4'h1;
    localparam IF_SETUP_PROC = 4'h2;
    localparam IF_WAIT_PROC = 4'h3;
    localparam IF_COLLECT_RESULT = 4'h4;
    localparam IF_SEND_RESPONSE = 4'h5;
    localparam IF_BATCH_PROCESS = 4'h6;
    localparam IF_ERROR = 4'hF;
    
    // Internal registers
    reg [7:0] current_cmd;
    reg [1:0] current_gate_select;
    reg [AGING_FACTOR_WIDTH-1:0] configured_aging_factor;
    reg [DNA_WIDTH-1:0] batch_dna_buffer [0:BATCH_SIZE-1];
    reg [4:0] batch_counter;
    reg [4:0] batch_size_reg;
    reg continuous_mode;
    reg [15:0] sequence_counter;
    
    // Result buffer for batch processing
    reg [DNA_WIDTH-1:0] result_buffer [0:RESULT_BUFFER_DEPTH-1];
    reg [15:0] entropy_buffer [0:RESULT_BUFFER_DEPTH-1];
    reg [7:0] mutation_buffer [0:RESULT_BUFFER_DEPTH-1];
    reg [5:0] result_write_ptr;
    reg [5:0] result_read_ptr;
    reg result_buffer_full;
    
    // Timeout counter for error handling
    reg [15:0] timeout_counter;
    localparam TIMEOUT_LIMIT = 16'hFFFF;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            interface_state <= IF_IDLE;
            host_ack <= 0;
            host_data_valid <= 0;
            proc_start <= 0;
            batch_progress <= 0;
            batch_complete <= 0;
            error_flag <= 0;
            current_gate_select <= 2'b00;
            configured_aging_factor <= 8'h10;
            batch_counter <= 0;
            sequence_counter <= 0;
            total_sequences_processed <= 0;
            result_write_ptr <= 0;
            result_read_ptr <= 0;
            result_buffer_full <= 0;
            continuous_mode <= 0;
            timeout_counter <= 0;
        end else begin
            // Timeout handling
            if (interface_state != IF_IDLE && interface_state != IF_SEND_RESPONSE) begin
                timeout_counter <= timeout_counter + 1;
                if (timeout_counter >= TIMEOUT_LIMIT) begin
                    interface_state <= IF_ERROR;
                    error_flag <= 1;
                end
            end else begin
                timeout_counter <= 0;
            end
            
            case (interface_state)
                IF_IDLE: begin
                    host_ack <= 0;
                    host_data_valid <= 0;
                    proc_start <= 0;
                    error_flag <= 0;
                    
                    if (host_req) begin
                        current_cmd <= host_cmd;
                        interface_state <= IF_DECODE_CMD;
                    end
                end
                
                IF_DECODE_CMD: begin
                    case (current_cmd)
                        CMD_SINGLE_PROCESS: begin
                            proc_dna_sequence <= host_dna_data;
                            proc_aging_factor <= host_aging_factor;
                            proc_gate_select <= current_gate_select;
                            interface_state <= IF_SETUP_PROC;
                        end
                        
                        CMD_BATCH_PROCESS: begin
                            batch_counter <= 0;
                            batch_size_reg <= host_dna_data[4:0];  // Lower 5 bits as batch size
                            batch_complete <= 0;
                            interface_state <= IF_BATCH_PROCESS;
                        end
                        
                        CMD_SET_GATE: begin
                            current_gate_select <= host_dna_data[1:0];
                            interface_state <= IF_SEND_RESPONSE;
                        end
                        
                        CMD_CONFIGURE_AGING: begin
                            configured_aging_factor <= host_aging_factor;
                            interface_state <= IF_SEND_RESPONSE;
                        end
                        
                        CMD_GET_STATUS: begin
                            host_result_data <= {16'h0, batch_progress, 8'h0};
                            host_entropy_data <= total_sequences_processed;
                            host_mutation_count <= {7'h0, batch_complete};
                            interface_state <= IF_SEND_RESPONSE;
                        end
                        
                        CMD_RESET_COUNTERS: begin
                            total_sequences_processed <= 0;
                            sequence_counter <= 0;
                            result_write_ptr <= 0;
                            result_read_ptr <= 0;
                            result_buffer_full <= 0;
                            interface_state <= IF_SEND_RESPONSE;
                        end
                        
                        CMD_GET_BATCH_RESULTS: begin
                            if (result_read_ptr != result_write_ptr || result_buffer_full) begin
                                host_result_data <= result_buffer[result_read_ptr];
                                host_entropy_data <= entropy_buffer[result_read_ptr];
                                host_mutation_count <= mutation_buffer[result_read_ptr];
                                result_read_ptr <= (result_read_ptr + 1) % RESULT_BUFFER_DEPTH;
                                result_buffer_full <= 0;
                            end
                            interface_state <= IF_SEND_RESPONSE;
                        end
                        
                        CMD_CONTINUOUS_MODE: begin
                            continuous_mode <= host_dna_data[0];
                            interface_state <= IF_SEND_RESPONSE;
                        end
                        
                        default: begin
                            interface_state <= IF_ERROR;
                        end
                    endcase
                end
                
                IF_SETUP_PROC: begin
                    proc_start <= 1;
                    interface_state <= IF_WAIT_PROC;
                end
                
                IF_WAIT_PROC: begin
                    if (proc_processing_done) begin
                        proc_start <= 0;
                        interface_state <= IF_COLLECT_RESULT;
                    end
                end
                
                IF_COLLECT_RESULT: begin
                    host_result_data <= proc_processed_dna;
                    host_entropy_data <= proc_entropy_measure;
                    host_mutation_count <= proc_mutation_count;
                    
                    // Store in result buffer for batch processing
                    if (!result_buffer_full) begin
                        result_buffer[result_write_ptr] <= proc_processed_dna;
                        entropy_buffer[result_write_ptr] <= proc_entropy_measure;
                        mutation_buffer[result_write_ptr] <= proc_mutation_count;
                        result_write_ptr <= (result_write_ptr + 1) % RESULT_BUFFER_DEPTH;
                        if (result_write_ptr == result_read_ptr)
                            result_buffer_full <= 1;
                    end
                    
                    total_sequences_processed <= total_sequences_processed + 1;
                    interface_state <= IF_SEND_RESPONSE;
                end
                
                IF_SEND_RESPONSE: begin
                    host_ack <= 1;
                    host_data_valid <= 1;
                    
                    if (continuous_mode && current_cmd == CMD_SINGLE_PROCESS) begin
                        // In continuous mode, immediately process next sequence
                        interface_state <= IF_IDLE;
                    end else if (!host_req) begin
                        interface_state <= IF_IDLE;
                    end
                end
                
                IF_BATCH_PROCESS: begin
                    if (batch_counter < batch_size_reg) begin
                        // Generate synthetic DNA sequence for batch processing
                        proc_dna_sequence <= generate_synthetic_dna(batch_counter, configured_aging_factor);
                        proc_aging_factor <= configured_aging_factor;
                        proc_gate_select <= current_gate_select;
                        
                        proc_start <= 1;
                        batch_counter <= batch_counter + 1;
                        interface_state <= IF_WAIT_PROC;
                        
                        // Update progress
                        batch_progress <= (batch_counter * 100) / batch_size_reg;
                    end else begin
                        batch_complete <= 1;
                        batch_progress <= 100;
                        interface_state <= IF_SEND_RESPONSE;
                    end
                end
                
                IF_ERROR: begin
                    error_flag <= 1;
                    host_ack <= 1;
                    host_result_data <= 32'hDEADBEEF;  // Error signature
                    host_entropy_data <= 16'hFFFF;
                    host_mutation_count <= 8'hFF;
                    
                    if (!host_req) begin
                        interface_state <= IF_IDLE;
                    end
                end
            endcase
        end
    end
    
    // Synthetic DNA generation function for batch processing
    function [DNA_WIDTH-1:0] generate_synthetic_dna;
        input [4:0] sequence_id;
        input [AGING_FACTOR_WIDTH-1:0] aging;
        integer i;
        reg [DNA_WIDTH-1:0] synthetic_dna;
        reg [15:0] seed;
        begin
            seed = {sequence_id, aging, 3'b101};
            synthetic_dna = 0;
            
            for (i = 0; i < DNA_WIDTH; i = i + 2) begin
                // Generate pseudo-random DNA sequence with aging bias
                seed = seed * 1664525 + 1013904223;  // Linear congruential generator
                
                case (seed[1:0])
                    2'b00: synthetic_dna[i+1:i] = 2'b00;  // A
                    2'b01: synthetic_dna[i+1:i] = 2'b01;  // T
                    2'b10: synthetic_dna[i+1:i] = 2'b10;  // G
                    2'b11: synthetic_dna[i+1:i] = 2'b11;  // C
                endcase
                
                // Apply aging bias (more mutations in older sequences)
                if ((seed[7:0] < aging) && (aging > 8'h20)) begin
                    synthetic_dna[i+1:i] = synthetic_dna[i+1:i] ^ 2'b01;
                end
            end
            
            generate_synthetic_dna = synthetic_dna;
        end
    endfunction
    
    // Performance monitoring
    always @(posedge clk) begin
        if (rst_n) begin
            // Monitor processing throughput
            if (proc_processing_done) begin
                sequence_counter <= sequence_counter + 1;
            end
        end
    end

endmodule