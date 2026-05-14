// Minimal external-kernel AIR fixture following the current MLIR-AIR style.
//
// The AIR program owns placement and movement:
// - air.herd maps work to one AIE tile.
// - memref<..., 2> is L1 tile-local memory.
// - air.dma_memcpy_nd copies the tile-local result back to L3.
//
// The C++ object is only the tile compute body.  The link object is attached to
// the external func.func declaration, which is the path AIRToAIE now recommends.

module {
  func.func @moo(%arg0 : memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    air.herd @cowfactory tile (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%out=%arg0) : memref<1024xi32> {
      %alloc = memref.alloc() {sym_name = "beef"} : memref<1024xi32, 2>
      func.call @beefmaker_kernel(%alloc) : (memref<1024xi32, 2>) -> ()
      air.dma_memcpy_nd (%out[] [] [], %alloc[] [] []) : (memref<1024xi32>, memref<1024xi32, 2>)
    }
    return
  }

  func.func private @beefmaker_kernel(memref<1024xi32, 2>) attributes {link_with = "beefmaker_kernel.o", llvm.emit_c_interface}
}
