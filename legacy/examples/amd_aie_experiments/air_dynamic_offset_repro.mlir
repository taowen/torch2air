module {
  func.func @dynamic_offset_repro(%input: memref<64xi32>, %output: memref<64xi32>) {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index

    air.launch (%launch_i) in (%launch_size=%c4)
        args(%launch_input=%input, %launch_output=%output)
        : memref<64xi32>, memref<64xi32> {
      %launch_tile_width = arith.constant 8 : index
      %tile_offset = arith.muli %launch_i, %launch_tile_width : index

      air.segment @dynamic_offset_segment
          args(%seg_input=%launch_input, %seg_output=%launch_output, %seg_offset=%tile_offset)
          : memref<64xi32>, memref<64xi32>, index {
        %seg_one = arith.constant 1 : index

        air.herd @dynamic_offset_herd tile (%tile_i, %tile_j)
            in (%herd_rows=%seg_one, %herd_cols=%seg_one)
            args(%herd_input=%seg_input, %herd_output=%seg_output, %herd_offset=%seg_offset)
            : memref<64xi32>, memref<64xi32>, index {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c8_h = arith.constant 8 : index
          %c7 = arith.constant 7 : i32

          %l1_in = memref.alloc() : memref<8xi32, 2 : i32>
          %l1_out = memref.alloc() : memref<8xi32, 2 : i32>

          air.dma_memcpy_nd
              (%l1_in[] [] [],
               %herd_input[%herd_offset] [%c8_h] [%c1])
              : (memref<8xi32, 2 : i32>, memref<64xi32>)

          scf.for %i = %c0 to %c8_h step %c1 {
            %value = memref.load %l1_in[%i] : memref<8xi32, 2 : i32>
            %result = arith.addi %value, %c7 : i32
            memref.store %result, %l1_out[%i] : memref<8xi32, 2 : i32>
          }

          air.dma_memcpy_nd
              (%herd_output[%herd_offset] [%c8_h] [%c1],
               %l1_out[] [] [])
              : (memref<64xi32>, memref<8xi32, 2 : i32>)

          memref.dealloc %l1_in : memref<8xi32, 2 : i32>
          memref.dealloc %l1_out : memref<8xi32, 2 : i32>
        }
      }
    }

    return
  }
}
