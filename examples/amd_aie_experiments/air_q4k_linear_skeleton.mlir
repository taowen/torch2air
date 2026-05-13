// Spike 5 fixture: a Q4_K linear-shaped tile consumes packed uint32 words.
// Q4_K has 256 logical K values per block and 144 bytes = 36 uint32 words.

module {
  func.func @q4k_linear_skeleton(%X: memref<1x256xbf16>, %W: memref<8x36xi32>, %Y: memref<1x8xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.parallel (%launch_i, %launch_j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      scf.parallel (%i, %j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
        %x_tile = memref.subview %X[%c0, %c0] [1, 256] [1, 1]
            : memref<1x256xbf16> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
        %w_tile = memref.subview %W[%c0, %c0] [8, 36] [1, 1]
            : memref<8x36xi32> to memref<8x36xi32, strided<[36, 1], offset: ?>>
        %y_tile = memref.subview %Y[%c0, %c0] [1, 8] [1, 1]
            : memref<1x8xbf16> to memref<1x8xbf16, strided<[8, 1], offset: ?>>

        %x_l1 = memref.alloc() : memref<1x256xbf16, 2>
        %w_l1 = memref.alloc() : memref<8x36xi32, 2>
        %y_l1 = memref.alloc() : memref<1x8xbf16, 2>

        memref.copy %x_tile, %x_l1
            : memref<1x256xbf16, strided<[256, 1], offset: ?>> to memref<1x256xbf16, 2>
        memref.copy %w_tile, %w_l1
            : memref<8x36xi32, strided<[36, 1], offset: ?>> to memref<8x36xi32, 2>

        %w0 = memref.load %w_l1[%c0, %c0] : memref<8x36xi32, 2>
        %w16 = arith.trunci %w0 : i32 to i16
        %wbf = arith.sitofp %w16 : i16 to bf16
        memref.store %wbf, %y_l1[%c0, %c0] : memref<1x8xbf16, 2>

        memref.copy %y_l1, %y_tile
            : memref<1x8xbf16, 2> to memref<1x8xbf16, strided<[8, 1], offset: ?>>

        memref.dealloc %x_l1 : memref<1x256xbf16, 2>
        memref.dealloc %w_l1 : memref<8x36xi32, 2>
        memref.dealloc %y_l1 : memref<1x8xbf16, 2>

        scf.reduce
      }
      scf.reduce
    }
    return
  }
}
