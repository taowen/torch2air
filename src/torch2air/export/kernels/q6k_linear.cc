// Q6_K linear tile kernel for quantized_qwen3 attention projections.
//
// AIR supplies one token row, OUTPUT_TILE_ROWS packed Q6_K rows, and one output
// tile in L1. Each packed row stores GGUF Q6_K halfwords widened to i32 words,
// followed by f32 bit patterns for per-block d values:
//
//   words[0:BLOCKS_PER_ROW*105]             original Q6_K halfwords
//   words[ROW_HALFWORDS + block]            d as f32 bits

#include <cstdint>

#ifndef OUTPUT_TILE_ROWS
#define OUTPUT_TILE_ROWS 16
#endif

#ifndef BLOCKS_PER_ROW
#define BLOCKS_PER_ROW 4
#endif

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 1024
#endif

#define Q6K_HALFWORDS_PER_BLOCK 105
#define Q6K_ROW_HALFWORDS (BLOCKS_PER_ROW * Q6K_HALFWORDS_PER_BLOCK)
#define Q6K_WEIGHT_WORDS (Q6K_ROW_HALFWORDS + BLOCKS_PER_ROW)

static inline uint8_t q6k_byte(const uint32_t *__restrict block, int byte_offset) {
  const uint32_t half = block[byte_offset >> 1] & 0xffffu;
  return (uint8_t)(((byte_offset & 1) == 0) ? (half & 0xffu) : (half >> 8));
}

static inline int q6k_i8(const uint32_t *__restrict block, int byte_offset) {
  int value = (int)q6k_byte(block, byte_offset);
  return value >= 128 ? value - 256 : value;
}

static inline float f32_from_bits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } value;
  value.u = bits;
  return value.f;
}

static inline float q6k_value(const uint32_t *__restrict block, int local) {
  const int iqs = (local >> 1) & 127;
  const int lane = local & 1;
  const int section = iqs >> 6;
  const int b = ((iqs & 63) >> 5) * 4;
  const int is_b = (iqs & 15) >> 3;
  const int qhshift = ((iqs & 63) >> 4) * 2;
  const int scale_index = 8 * section + qhshift + is_b;
  const int qsi = section * 32 + (iqs & 31);
  const int qhi = section * 16 + (iqs & 15);
  const uint32_t ql = (block[qsi] >> b) & 0x0f0fu;
  const uint32_t qh = (block[64 + qhi] >> qhshift) & 0x0303u;
  const uint32_t packed = ql | (qh << 4);
  const uint32_t q = lane == 0 ? (packed & 0xffu) : ((packed >> 8) & 0xffu);
  const float scale = (float)q6k_i8(block, 192 + scale_index);
  return (float)((int)q - 32) * scale;
}

static inline float q6k_dot_block(
    const uint32_t *__restrict block,
    const float *__restrict x,
    float d) {
  float acc = 0.0f;
  for (int local = 0; local < 256; ++local) {
    acc += d * q6k_value(block, local) * x[local];
  }
  return acc;
}

extern "C" {

void q6k_linear_tile(
    float *__restrict hidden,
    uint32_t *__restrict packed_weights,
    float *__restrict output) {
  for (int row = 0; row < OUTPUT_TILE_ROWS; ++row) {
    const uint32_t *__restrict row_weight = packed_weights + row * Q6K_WEIGHT_WORDS;
    float acc = 0.0f;
    for (int block_index = 0; block_index < BLOCKS_PER_ROW; ++block_index) {
      const uint32_t *__restrict block = row_weight + block_index * Q6K_HALFWORDS_PER_BLOCK;
      const uint32_t d_bits = row_weight[Q6K_ROW_HALFWORDS + block_index];
      acc += q6k_dot_block(block, hidden + block_index * 256, f32_from_bits(d_bits));
    }
    output[row] = acc;
  }
}

} // extern "C"
