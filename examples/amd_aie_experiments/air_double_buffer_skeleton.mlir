// Spike 4 fixture: two activation buffers and two weight buffers model the
// ping/pong slots a generated Q4_K linear schedule will use.

module {
  func.func @double_buffer_skeleton(%A: memref<8x32xi32>, %B: memref<32x8xi32>, %C: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    scf.parallel (%launch_i, %launch_j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c8) step (%c8, %c8) {
        %a0_tile = memref.subview %A[%i, %c0] [8, 16] [1, 1]
            : memref<8x32xi32> to memref<8x16xi32, strided<[32, 1], offset: ?>>
        %b0_tile = memref.subview %B[%c0, %j] [16, 8] [1, 1]
            : memref<32x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
        %a1_tile = memref.subview %A[%i, %c16] [8, 16] [1, 1]
            : memref<8x32xi32> to memref<8x16xi32, strided<[32, 1], offset: ?>>
        %b1_tile = memref.subview %B[%c16, %j] [16, 8] [1, 1]
            : memref<32x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
        %c_tile = memref.subview %C[%i, %j] [8, 8] [1, 1]
            : memref<8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>

        %a_ping = memref.alloc() : memref<8x16xi32, 2>
        %b_ping = memref.alloc() : memref<16x8xi32, 2>
        %a_pong = memref.alloc() : memref<8x16xi32, 2>
        %b_pong = memref.alloc() : memref<16x8xi32, 2>
        %c_l1 = memref.alloc() : memref<8x8xi32, 2>

        memref.copy %c_tile, %c_l1
            : memref<8x8xi32, strided<[8, 1], offset: ?>> to memref<8x8xi32, 2>

        // Prologue: fill ping.
        memref.copy %a0_tile, %a_ping
            : memref<8x16xi32, strided<[32, 1], offset: ?>> to memref<8x16xi32, 2>
        memref.copy %b0_tile, %b_ping
            : memref<16x8xi32, strided<[8, 1], offset: ?>> to memref<16x8xi32, 2>

        // Main body: prefetch pong before the next compute stage consumes it.
        memref.copy %a1_tile, %a_pong
            : memref<8x16xi32, strided<[32, 1], offset: ?>> to memref<8x16xi32, 2>
        memref.copy %b1_tile, %b_pong
            : memref<16x8xi32, strided<[8, 1], offset: ?>> to memref<16x8xi32, 2>

        linalg.matmul {cast = #linalg.type_fn<cast_signed>}
            ins(%a_ping, %b_ping : memref<8x16xi32, 2>, memref<16x8xi32, 2>)
            outs(%c_l1 : memref<8x8xi32, 2>)
        linalg.matmul {cast = #linalg.type_fn<cast_signed>}
            ins(%a_pong, %b_pong : memref<8x16xi32, 2>, memref<16x8xi32, 2>)
            outs(%c_l1 : memref<8x8xi32, 2>)

        memref.copy %c_l1, %c_tile
            : memref<8x8xi32, 2> to memref<8x8xi32, strided<[8, 1], offset: ?>>

        memref.dealloc %a_ping : memref<8x16xi32, 2>
        memref.dealloc %b_ping : memref<16x8xi32, 2>
        memref.dealloc %a_pong : memref<8x16xi32, 2>
        memref.dealloc %b_pong : memref<16x8xi32, 2>
        memref.dealloc %c_l1 : memref<8x8xi32, 2>

        scf.reduce
      }
      scf.reduce
    }
    return
  }
}
