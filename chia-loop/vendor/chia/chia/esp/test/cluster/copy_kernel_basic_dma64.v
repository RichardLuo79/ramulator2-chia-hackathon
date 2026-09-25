// Identity (copy) kernel for the ESP RTL-flow socket, basic_dma64 variant:
// DMA-reads `size` 32-bit tokens from offset 0, writes them back to the
// output region at token offset round_up(size, 2), then pulses acc_done.
// Replaces the accgen-generated stub (which moves no data) so the skeleton's
// self-test only passes when the accelerator really copies the buffer.
// Buffer capacity: 1024 tokens (512 beats).

module chiatest_rtl_basic_dma64( clk, rst, dma_read_chnl_valid, dma_read_chnl_data, dma_read_chnl_ready,
conf_info_size,
conf_done, acc_done, debug, dma_read_ctrl_valid, dma_read_ctrl_data_index, dma_read_ctrl_data_length, dma_read_ctrl_data_size, dma_read_ctrl_ready, dma_write_ctrl_valid, dma_write_ctrl_data_index, dma_write_ctrl_data_length, dma_write_ctrl_data_size, dma_write_ctrl_ready, dma_write_chnl_valid, dma_write_chnl_data, dma_write_chnl_ready);

   input clk;
   input rst;

   input [31:0]  conf_info_size;
   input         conf_done;

   input         dma_read_ctrl_ready;
   output reg    dma_read_ctrl_valid;
   output [31:0] dma_read_ctrl_data_index;
   output [31:0] dma_read_ctrl_data_length;
   output [2:0]  dma_read_ctrl_data_size;

   output        dma_read_chnl_ready;
   input         dma_read_chnl_valid;
   input [63:0]  dma_read_chnl_data;

   input         dma_write_ctrl_ready;
   output reg    dma_write_ctrl_valid;
   output [31:0] dma_write_ctrl_data_index;
   output [31:0] dma_write_ctrl_data_length;
   output [2:0]  dma_write_ctrl_data_size;

   input         dma_write_chnl_ready;
   output reg    dma_write_chnl_valid;
   output [63:0] dma_write_chnl_data;

   output reg    acc_done;
   output [31:0] debug;

   localparam [2:0] SIZE_WORD = 3'b010;  // 32-bit tokens

   localparam IDLE    = 3'd0,
              RD_CTRL = 3'd1,
              RD_DATA = 3'd2,
              WR_CTRL = 3'd3,
              WR_DATA = 3'd4,
              DONE    = 3'd5;

   reg [2:0]   state;
   reg [31:0]  size;        // tokens to copy (latched from conf_info_size)
   reg [9:0]   beat;        // current beat within the burst
   reg [63:0]  plm [0:511]; // up to 1024 tokens
   reg [63:0]  wdata;

   // DMA index/length are in BEATS of the 64-bit channel (the engine
   // counts one per transferred flit); two 32-bit tokens per beat. The
   // size field only selects token width for endianness handling.
   wire [31:0] nbeats = (size + 32'd1) >> 1;

   assign dma_read_ctrl_data_index   = 32'd0;
   assign dma_read_ctrl_data_length  = nbeats;
   assign dma_read_ctrl_data_size    = SIZE_WORD;
   // Output region starts right after the beat-aligned input region.
   assign dma_write_ctrl_data_index  = nbeats;
   assign dma_write_ctrl_data_length = nbeats;
   assign dma_write_ctrl_data_size   = SIZE_WORD;
   assign dma_read_chnl_ready  = (state == RD_DATA);
   assign dma_write_chnl_data  = wdata;
   assign debug = {29'd0, state};

   always @(posedge clk or negedge rst) begin
      if (!rst) begin
         state <= IDLE;
         dma_read_ctrl_valid <= 1'b0;
         dma_write_ctrl_valid <= 1'b0;
         dma_write_chnl_valid <= 1'b0;
         acc_done <= 1'b0;
         beat <= 10'd0;
         size <= 32'd0;
      end else begin
         acc_done <= 1'b0;
         case (state)
           IDLE: begin
              if (conf_done) begin
                 size <= (conf_info_size > 32'd1024) ? 32'd1024 : conf_info_size;
                 beat <= 10'd0;
                 dma_read_ctrl_valid <= 1'b1;
                 state <= RD_CTRL;
              end
           end
           RD_CTRL: begin
              if (dma_read_ctrl_ready) begin
                 dma_read_ctrl_valid <= 1'b0;
                 state <= RD_DATA;
              end
           end
           RD_DATA: begin
              if (dma_read_chnl_valid) begin
                 plm[beat] <= dma_read_chnl_data;
                 if (beat + 10'd1 == nbeats[9:0]) begin
                    beat <= 10'd0;
                    dma_write_ctrl_valid <= 1'b1;
                    state <= WR_CTRL;
                 end else begin
                    beat <= beat + 10'd1;
                 end
              end
           end
           WR_CTRL: begin
              if (dma_write_ctrl_ready) begin
                 dma_write_ctrl_valid <= 1'b0;
                 wdata <= plm[10'd0];
                 dma_write_chnl_valid <= 1'b1;
                 state <= WR_DATA;
              end
           end
           WR_DATA: begin
              if (dma_write_chnl_ready) begin
                 if (beat + 10'd1 == nbeats[9:0]) begin
                    dma_write_chnl_valid <= 1'b0;
                    state <= DONE;
                 end else begin
                    beat <= beat + 10'd1;
                    wdata <= plm[beat + 10'd1];
                 end
              end
           end
           DONE: begin
              acc_done <= 1'b1;
              state <= IDLE;
           end
         endcase
      end
   end

endmodule
