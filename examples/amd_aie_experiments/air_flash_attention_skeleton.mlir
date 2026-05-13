// Spike 6 fixture: staged attention skeleton with Q resident in L1 and K/V
// copied per stage before updating the output tile.

module {
  func.func @flash_attention_skeleton(%Q: memref<1x64xbf16>, %K: memref<2x64xbf16>, %V: memref<2x64xbf16>, %O: memref<1x64xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    scf.parallel (%launch_i, %launch_j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      scf.parallel (%i, %j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
        %q_tile = memref.subview %Q[%c0, %c0] [1, 64] [1, 1]
            : memref<1x64xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
        %k0_tile = memref.subview %K[%c0, %c0] [1, 64] [1, 1]
            : memref<2x64xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
        %v0_tile = memref.subview %V[%c0, %c0] [1, 64] [1, 1]
            : memref<2x64xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
        %k1_tile = memref.subview %K[%c1, %c0] [1, 64] [1, 1]
            : memref<2x64xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
        %v1_tile = memref.subview %V[%c1, %c0] [1, 64] [1, 1]
            : memref<2x64xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>
        %o_tile = memref.subview %O[%c0, %c0] [1, 64] [1, 1]
            : memref<1x64xbf16> to memref<1x64xbf16, strided<[64, 1], offset: ?>>

        %q_l1 = memref.alloc() : memref<1x64xbf16, 2>
        %k_l1 = memref.alloc() : memref<1x64xbf16, 2>
        %v_l1 = memref.alloc() : memref<1x64xbf16, 2>
        %o_l1 = memref.alloc() : memref<1x64xbf16, 2>

        memref.copy %q_tile, %q_l1
            : memref<1x64xbf16, strided<[64, 1], offset: ?>> to memref<1x64xbf16, 2>
        memref.copy %k0_tile, %k_l1
            : memref<1x64xbf16, strided<[64, 1], offset: ?>> to memref<1x64xbf16, 2>
        memref.copy %v0_tile, %v_l1
            : memref<1x64xbf16, strided<[64, 1], offset: ?>> to memref<1x64xbf16, 2>

        %q0 = memref.load %q_l1[%c0, %c0] : memref<1x64xbf16, 2>
        %v0 = memref.load %v_l1[%c0, %c0] : memref<1x64xbf16, 2>
        %acc0 = arith.addf %q0, %v0 : bf16
        memref.store %acc0, %o_l1[%c0, %c0] : memref<1x64xbf16, 2>

        memref.copy %k1_tile, %k_l1
            : memref<1x64xbf16, strided<[64, 1], offset: ?>> to memref<1x64xbf16, 2>
        memref.copy %v1_tile, %v_l1
            : memref<1x64xbf16, strided<[64, 1], offset: ?>> to memref<1x64xbf16, 2>

        %v1 = memref.load %v_l1[%c0, %c0] : memref<1x64xbf16, 2>
        %acc1 = arith.addf %acc0, %v1 : bf16
        memref.store %acc1, %o_l1[%c0, %c0] : memref<1x64xbf16, 2>

        memref.copy %o_l1, %o_tile
            : memref<1x64xbf16, 2> to memref<1x64xbf16, strided<[64, 1], offset: ?>>

        memref.dealloc %q_l1 : memref<1x64xbf16, 2>
        memref.dealloc %k_l1 : memref<1x64xbf16, 2>
        memref.dealloc %v_l1 : memref<1x64xbf16, 2>
        memref.dealloc %o_l1 : memref<1x64xbf16, 2>

        scf.reduce
      }
      scf.reduce
    }
    return
  }
}
