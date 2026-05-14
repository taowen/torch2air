// Spike 3 fixture: three independent input copies before compute and one
// output copy after compute. Channel lowering should keep input DMA visible and
// force the output transfer after the compute body.

module {
  func.func @dma_ordering_channels(%A: memref<8x16xi32>, %B: memref<16x8xi32>, %S: memref<8xi32>, %C: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    scf.parallel (%launch_i, %launch_j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c8) step (%c8, %c8) {
        %a_tile = memref.subview %A[%i, %c0] [8, 16] [1, 1]
            : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
        %b_tile = memref.subview %B[%c0, %j] [16, 8] [1, 1]
            : memref<16x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
        %s_tile = memref.subview %S[%j] [8] [1]
            : memref<8xi32> to memref<8xi32, strided<[1], offset: ?>>
        %c_tile = memref.subview %C[%i, %j] [8, 8] [1, 1]
            : memref<8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>

        %a_l1 = memref.alloc() : memref<8x16xi32, 2>
        %b_l1 = memref.alloc() : memref<16x8xi32, 2>
        %s_l1 = memref.alloc() : memref<8xi32, 2>
        %c_l1 = memref.alloc() : memref<8x8xi32, 2>

        memref.copy %a_tile, %a_l1
            : memref<8x16xi32, strided<[16, 1], offset: ?>> to memref<8x16xi32, 2>
        memref.copy %b_tile, %b_l1
            : memref<16x8xi32, strided<[8, 1], offset: ?>> to memref<16x8xi32, 2>
        memref.copy %s_tile, %s_l1
            : memref<8xi32, strided<[1], offset: ?>> to memref<8xi32, 2>
        memref.copy %c_tile, %c_l1
            : memref<8x8xi32, strided<[8, 1], offset: ?>> to memref<8x8xi32, 2>

        linalg.matmul {cast = #linalg.type_fn<cast_signed>}
            ins(%a_l1, %b_l1 : memref<8x16xi32, 2>, memref<16x8xi32, 2>)
            outs(%c_l1 : memref<8x8xi32, 2>)

        %scale0 = memref.load %s_l1[%c0] : memref<8xi32, 2>
        memref.store %scale0, %c_l1[%c0, %c0] : memref<8x8xi32, 2>

        memref.copy %c_l1, %c_tile
            : memref<8x8xi32, 2> to memref<8x8xi32, strided<[8, 1], offset: ?>>

        memref.dealloc %a_l1 : memref<8x16xi32, 2>
        memref.dealloc %b_l1 : memref<16x8xi32, 2>
        memref.dealloc %s_l1 : memref<8xi32, 2>
        memref.dealloc %c_l1 : memref<8x8xi32, 2>

        scf.reduce
      }
      scf.reduce
    }
    return
  }
}
