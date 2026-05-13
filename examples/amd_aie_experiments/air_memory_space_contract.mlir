// Spike 2 fixture: explicit global -> L2 -> L1 -> L2 -> global movement.
//
// Memory-space contract used by torch2air templates:
// - default memrefs are host/global buffers.
// - memref<..., 1> is the memtile/L2 staging buffer.
// - memref<..., 2> is the AIE core-local L1 buffer.

module {
  func.func @memory_space_contract(%A: memref<8x16xi32>, %B: memref<16x8xi32>, %C: memref<8x8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    scf.parallel (%launch_i, %launch_j) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      %a_l2 = memref.alloc() : memref<8x16xi32, 1>
      %b_l2 = memref.alloc() : memref<16x8xi32, 1>
      %c_l2 = memref.alloc() : memref<8x8xi32, 1>

      memref.copy %A, %a_l2 : memref<8x16xi32> to memref<8x16xi32, 1>
      memref.copy %B, %b_l2 : memref<16x8xi32> to memref<16x8xi32, 1>
      memref.copy %C, %c_l2 : memref<8x8xi32> to memref<8x8xi32, 1>

      scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c8) step (%c8, %c8) {
        %a_l1 = memref.alloc() : memref<8x16xi32, 2>
        %b_l1 = memref.alloc() : memref<16x8xi32, 2>
        %c_l1 = memref.alloc() : memref<8x8xi32, 2>

        memref.copy %a_l2, %a_l1 : memref<8x16xi32, 1> to memref<8x16xi32, 2>
        memref.copy %b_l2, %b_l1 : memref<16x8xi32, 1> to memref<16x8xi32, 2>
        memref.copy %c_l2, %c_l1 : memref<8x8xi32, 1> to memref<8x8xi32, 2>

        linalg.matmul {cast = #linalg.type_fn<cast_signed>}
            ins(%a_l1, %b_l1 : memref<8x16xi32, 2>, memref<16x8xi32, 2>)
            outs(%c_l1 : memref<8x8xi32, 2>)

        memref.copy %c_l1, %c_l2 : memref<8x8xi32, 2> to memref<8x8xi32, 1>

        memref.dealloc %a_l1 : memref<8x16xi32, 2>
        memref.dealloc %b_l1 : memref<16x8xi32, 2>
        memref.dealloc %c_l1 : memref<8x8xi32, 2>

        scf.reduce
      }

      memref.copy %c_l2, %C : memref<8x8xi32, 1> to memref<8x8xi32>
      memref.dealloc %a_l2 : memref<8x16xi32, 1>
      memref.dealloc %b_l2 : memref<16x8xi32, 1>
      memref.dealloc %c_l2 : memref<8x8xi32, 1>
      scf.reduce
    }
    return
  }
}
