// Q4_K linear Spike 1 tile ABI kernel.
//
// AIR owns L3/L1 movement and calls this function with tile-local L1 buffers.
// The body intentionally does not implement Q4_K dot yet. It only proves that
// hidden input, weight input, external call ABI, and output DMA are wired.

#include <cstdint>

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 1024
#endif

#ifndef OUTPUT_TILE_ROWS
#define OUTPUT_TILE_ROWS 16
#endif

extern "C" {

void q4k_linear_spike1_tile(
    float *__restrict hidden,
    int32_t *__restrict weight,
    float *__restrict output) {
  constexpr int stride = HIDDEN_SIZE / OUTPUT_TILE_ROWS;
  for (int row = 0; row < OUTPUT_TILE_ROWS; ++row) {
    output[row] = hidden[row * stride] + (float)weight[row];
  }
}

} // extern "C"
