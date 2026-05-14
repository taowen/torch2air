// Direct tiled matmul used to validate the first torch2air AIR spike.
//
// Memory-space convention for this fixture:
// - default memrefs are global/L3 buffers owned by the caller.
// - memref<..., 2> is the smallest accepted syntax for AIE tile-local L1
//   buffers in the current MLIR-AIR toolchain.

module {
  func.func @matmul_8x16x8(%A: memref<8x16xi32>, %B: memref<16x8xi32>, %C: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    scf.parallel (%launch_i, %launch_j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c8) step (%c8, %c8) {
        %a_tile = memref.subview %A[%i, %c0] [8, 16] [1, 1]
            : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
        %b_tile = memref.subview %B[%c0, %j] [16, 8] [1, 1]
            : memref<16x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
        %c_tile = memref.subview %C[%i, %j] [8, 8] [1, 1]
            : memref<8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>

        %a_l1 = memref.alloc() : memref<8x16xi32, 2>
        %b_l1 = memref.alloc() : memref<16x8xi32, 2>
        %c_l1 = memref.alloc() : memref<8x8xi32, 2>

        memref.copy %a_tile, %a_l1
            : memref<8x16xi32, strided<[16, 1], offset: ?>> to memref<8x16xi32, 2>
        memref.copy %b_tile, %b_l1
            : memref<16x8xi32, strided<[8, 1], offset: ?>> to memref<16x8xi32, 2>
        memref.copy %c_tile, %c_l1
            : memref<8x8xi32, strided<[8, 1], offset: ?>> to memref<8x8xi32, 2>

        linalg.matmul {cast = #linalg.type_fn<cast_signed>}
            ins(%a_l1, %b_l1 : memref<8x16xi32, 2>, memref<16x8xi32, 2>)
            outs(%c_l1 : memref<8x8xi32, 2>)

        memref.copy %c_l1, %c_tile
            : memref<8x8xi32, 2> to memref<8x8xi32, strided<[8, 1], offset: ?>>

        memref.dealloc %a_l1 : memref<8x16xi32, 2>
        memref.dealloc %b_l1 : memref<16x8xi32, 2>
        memref.dealloc %c_l1 : memref<8x8xi32, 2>

        scf.reduce
      }
      scf.reduce
    }

    return
  }
}
