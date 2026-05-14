module {
  func.func @dynamic_head_offset_repro(
      %input: memref<1x512xi32>,
      %output: memref<1x512xi32>) {
    air.launch () in ()
        args(%launch_input=%input, %launch_output=%output)
        : memref<1x512xi32>, memref<1x512xi32> {
      air.segment @dynamic_head_offset_segment
          args(%seg_input=%launch_input, %seg_output=%launch_output)
          : memref<1x512xi32>, memref<1x512xi32> {
        %seg_one = arith.constant 1 : index

        air.herd @dynamic_head_offset_herd tile (%tile_i, %tile_j)
            in (%herd_rows=%seg_one, %herd_cols=%seg_one)
            args(%herd_input=%seg_input, %herd_output=%seg_output)
            : memref<1x512xi32>, memref<1x512xi32> {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c7 = arith.constant 7 : i32
          %c128 = arith.constant 128 : index
          %c512 = arith.constant 512 : index

          %l1_in = memref.alloc() : memref<1x128xi32, 2 : i32>
          %l1_out = memref.alloc() : memref<1x128xi32, 2 : i32>

          scf.for %head_i = %c0 to %c4 step %c1 {
            %col_base = arith.muli %head_i, %c128 : index

            air.dma_memcpy_nd
                (%l1_in[] [] [],
                 %herd_input[%c0, %col_base] [%c1, %c128] [%c512, %c1])
                : (memref<1x128xi32, 2 : i32>, memref<1x512xi32>)

            scf.for %i = %c0 to %c128 step %c1 {
              %value = memref.load %l1_in[%c0, %i] : memref<1x128xi32, 2 : i32>
              %result = arith.addi %value, %c7 : i32
              memref.store %result, %l1_out[%c0, %i] : memref<1x128xi32, 2 : i32>
            }

            air.dma_memcpy_nd
                (%herd_output[%c0, %col_base] [%c1, %c128] [%c512, %c1],
                 %l1_out[] [] [])
                : (memref<1x512xi32>, memref<1x128xi32, 2 : i32>)
          }

          memref.dealloc %l1_in : memref<1x128xi32, 2 : i32>
          memref.dealloc %l1_out : memref<1x128xi32, 2 : i32>
        }
      }
    }

    return
  }
}
